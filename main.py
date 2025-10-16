# =============================================================================
# Harara Heatwave API (Production Ready)
# - Loads trained artifacts (scaler, model, threshold)
# - Pulls features from Google Earth Engine
# - Produces 7-day heatwave risk per town
# - Stores results in SQLite + Firestore (Firebase)
# - Exposes HTTP endpoints (manual run, latest results, quick viz, mock)
# - Runs an automated daily prediction at 07:00
# =============================================================================

import os
import io
import json
import datetime as dt
from zoneinfo import ZoneInfo
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
from pydantic import BaseModel
from sqlmodel import SQLModel, Field, create_engine, Session, select

import tensorflow as tf
import joblib
import ee

# =============================================================================
# FIREBASE FIRESTORE SETUP
# =============================================================================
import firebase_admin
from firebase_admin import credentials, firestore

FIRESTORE_DB = None

def init_firestore():
    """Initialize Firebase Firestore using service account key."""
    global FIRESTORE_DB
    try:
        # Try environment variable first (for cloud deployment)
        firebase_key = os.getenv("FIREBASE_SERVICE_KEY")
        if firebase_key:
            firebase_info = json.loads(firebase_key)
            cred = credentials.Certificate(firebase_info)
            firebase_admin.initialize_app(cred)
            FIRESTORE_DB = firestore.client()
            print("✅ Firestore initialized with environment service account key")
        else:
            # Fallback to local file (for local development)
            cred_path = os.path.join(os.path.dirname(__file__), "firebase-key.json")
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
            FIRESTORE_DB = firestore.client()
            print("✅ Firestore initialized with local key file")
    except Exception as e:
        print(f"❌ Firestore init error: {e}")

# =============================================================================
# CONFIG / CONSTANTS
# =============================================================================

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "harara_artifacts")
DB_PATH = os.path.join(os.path.dirname(__file__), "harara.db")
TIMEZONE = "Africa/Kigali"
LOOKBACK_DAYS = 21
HORIZON_DAYS = 7
MAX_FFILL_GAP = 5
SCHEDULER_ENABLED = True

DATE_COL = "date"
TOWN_COL = "town"
FEATURE_COLS = [
    "LST_Day_1km", "LST_Night_1km", "air_temp_2m", "ndvi",
    "net_solar_radiation", "precipitation", "relative_humidity",
    "soil_moisture", "wind_speed", "longitude", "latitude"
]

EE_SERVICE_ACCOUNT = os.getenv("EE_SERVICE_ACCOUNT")
EE_PRIVATE_KEY_JSON_PATH = os.getenv("EE_PRIVATE_KEY_JSON_PATH")

LOCAL_EE_SERVICE_ACCOUNT = "harara-service@south-sudan-heatwave.iam.gserviceaccount.com"
LOCAL_EE_KEY_FILE = "south-sudan-heatwave-583da500ae5f.json"

# =============================================================================
# FASTAPI APP & CORS
# =============================================================================

app = FastAPI(
    title="Harara Heatwave API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# DATABASE (SQLite)
# =============================================================================

engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)

class Prediction(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    run_ts: dt.datetime
    start_date: dt.date
    end_date: dt.date
    town: str
    probability: float
    alert: int
    details_json: Optional[str] = None

# =============================================================================
# GLOBAL STATE
# =============================================================================

MODEL: Optional[tf.keras.Model] = None
SCALER = None
CALIBRATOR = None
THRESHOLD: float = 0.5
EE_READY = False
era5 = None
modis_lst = None
modis_ndvi = None
towns: Dict[str, ee.Geometry] = {}

# =============================================================================
# EARTH ENGINE INIT
# =============================================================================

def init_gee():
    """Initialize Google Earth Engine."""
    global EE_READY
    try:
        # Try environment variable first (for cloud deployment)
        key_json = os.getenv("EE_SERVICE_KEY")
        if key_json:
            service_account_info = json.loads(key_json)
            credentials = ee.ServiceAccountCredentials(
                service_account_info["client_email"],
                key_data=key_json
            )
            ee.Initialize(credentials)
            EE_READY = True
            print("✅ EE initialized with environment service account key")
        # Fallback to local file (for local development)
        elif os.path.exists(LOCAL_EE_KEY_FILE):
            credentials = ee.ServiceAccountCredentials(LOCAL_EE_SERVICE_ACCOUNT, LOCAL_EE_KEY_FILE)
            ee.Initialize(credentials)
            EE_READY = True
            print("✅ EE initialized with local key file")
        else:
            # Last resort: default authentication
            ee.Initialize()
            EE_READY = True
            print("✅ EE initialized with default credentials")
    except Exception as e:
        EE_READY = False
        print(f"❌ EE init error: {e}")

def build_ee_objects():
    """Create EE ImageCollections and town geometries."""
    global era5, modis_lst, modis_ndvi, towns
    if not EE_READY:
        raise RuntimeError("EE not initialized")

    era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
    modis_lst = ee.ImageCollection("MODIS/061/MOD11A1")
    modis_ndvi = ee.ImageCollection("MODIS/061/MOD13Q1")

    towns = {
        "Juba": ee.Geometry.Point([31.5804, 4.8594]).buffer(3000),
        "Wau": ee.Geometry.Point([28.0070, 7.7011]).buffer(3000),
        "Yambio": ee.Geometry.Point([28.4167, 4.5700]).buffer(3000),
        "Bor": ee.Geometry.Point([31.5594, 6.2065]).buffer(3000),
        "Malakal": ee.Geometry.Point([32.4730, 9.5330]).buffer(3000),
        "Bentiu": ee.Geometry.Point([29.7820, 9.2330]).buffer(3000),
    }
    print(" EE collections & towns ready")

# =============================================================================
# LOAD ML ARTIFACTS
# =============================================================================

def load_artifacts():
    global MODEL, SCALER, CALIBRATOR, THRESHOLD
    thr_path = os.path.join(ARTIFACT_DIR, "threshold.json")
    sc_path = os.path.join(ARTIFACT_DIR, "scaler.pkl")
    mdl_path = os.path.join(ARTIFACT_DIR, "model.keras")
    with open(thr_path) as f:
        THRESHOLD = float(json.load(f)["threshold"])
    SCALER = joblib.load(sc_path)
    MODEL = tf.keras.models.load_model(mdl_path)
    print(f" Artifacts loaded (threshold={THRESHOLD})")

# =============================================================================
# FEATURE FETCHING HELPERS (EE → pandas)
# =============================================================================

def _collection_to_df(imgcol, geom, scale=1000, band_rename=None, constant_cols=None):
    """Reduce an ImageCollection to a pandas DataFrame via mean over a geometry."""
    def extract_mean(img):
        d = img.date().format("YYYY-MM-dd")
        vals = img.reduceRegion(ee.Reducer.mean(), geom, scale=scale).set("date", d)
        if constant_cols:
            for col, value in constant_cols.items():
                vals = vals.set(col, value)
        return ee.Feature(None, vals)

    fc = imgcol.map(extract_mean)
    feats = fc.getInfo().get("features", [])
    rows = []
    for f in feats:
        props = f.get("properties", {})
        if "date" in props:
            rows.append(props)

    if not rows:
        cols = ["date"]
        if constant_cols:
            cols += list(constant_cols.keys())
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(rows)
    if band_rename:
        df = df.rename(columns=band_rename)
    return df

def _ensure_daily_index(df, start, end):
    """Ensure a continuous daily index and reindex onto it."""
    idx = pd.date_range(start, end, freq="D")
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).drop_duplicates(DATE_COL)
    df = df.set_index(DATE_COL).reindex(idx)
    df.index.name = DATE_COL
    return df.reset_index()

def fetch_features_for_town(town_name, geom, lon, lat, lookback_days=LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Pull last 'lookback_days' of daily features for a town, ending at yesterday,
    with light imputation and derived fields.
    """
    end = dt.datetime.now(ZoneInfo(TIMEZONE)).date() - dt.timedelta(days=1)
    start = end - dt.timedelta(days=lookback_days)

    const_cols = {TOWN_COL: town_name, "longitude": lon, "latitude": lat}

    # ERA5 (daily aggregates)
    era5_sel = (
        era5.filterDate(str(start), str(end))
            .select([
                "temperature_2m_max", "dewpoint_temperature_2m_max",
                "total_precipitation_sum", "surface_net_solar_radiation_sum",
                "u_component_of_wind_10m", "v_component_of_wind_10m",
                "volumetric_soil_water_layer_1"
            ])
    )
    df_era5 = _collection_to_df(
        era5_sel, geom, scale=1000,
        band_rename={
            "temperature_2m_max": "air_temp_2m",
            "total_precipitation_sum": "precipitation",
            "surface_net_solar_radiation_sum": "net_solar_radiation",
            "volumetric_soil_water_layer_1": "soil_moisture",
        },
        constant_cols=const_cols,
    )
    if "air_temp_2m" in df_era5:
        df_era5["air_temp_2m"] = df_era5["air_temp_2m"] - 273.15
    if "dewpoint_temperature_2m_max" in df_era5:
        df_era5["dewpoint_temperature_2m_max"] = df_era5["dewpoint_temperature_2m_max"] - 273.15

    # MODIS LST
    lst_sel = modis_lst.filterDate(str(start), str(end)).select(["LST_Day_1km", "LST_Night_1km"])
    df_lst = _collection_to_df(lst_sel, geom, scale=1000, constant_cols=const_cols)
    if "LST_Day_1km" in df_lst:
        df_lst["LST_Day_1km"] = df_lst["LST_Day_1km"] * 0.02
    if "LST_Night_1km" in df_lst:
        df_lst["LST_Night_1km"] = df_lst["LST_Night_1km"] * 0.02

    # MODIS NDVI (buffered window to improve availability)
    ndvi_sel = modis_ndvi.filterDate(str(start - dt.timedelta(days=90)), str(end)).select("NDVI")
    df_ndvi = _collection_to_df(ndvi_sel, geom, scale=250, constant_cols=const_cols)
    if "NDVI" in df_ndvi:
        df_ndvi["ndvi"] = df_ndvi["NDVI"] * 0.0001
        df_ndvi = df_ndvi.drop(columns=["NDVI"])

    # Merge sources
    dfs = [d for d in [df_era5, df_lst, df_ndvi] if d is not None and not d.empty]
    if not dfs:
        return pd.DataFrame()

    df = dfs[0]
    for other in dfs[1:]:
        df = pd.merge(df, other, on=[DATE_COL] + list(const_cols.keys()), how="outer")

    df = _ensure_daily_index(df, start, end)

    # Impute & derive
    for col in ["LST_Day_1km", "LST_Night_1km"]:
        if col in df:
            df[col] = df[col].interpolate(method="linear", limit_direction="both", limit=5)

    if {"air_temp_2m", "dewpoint_temperature_2m_max"}.issubset(df.columns):
        T, Td = df["air_temp_2m"], df["dewpoint_temperature_2m_max"]
        mask = T.notna() & Td.notna()
        es = 6.112 * np.exp((17.625 * T[mask]) / (T[mask] + 243.04))
        e  = 6.112 * np.exp((17.625 * Td[mask]) / (Td[mask] + 243.04))
        df.loc[mask, "relative_humidity"] = (e / es) * 100
        df["relative_humidity"] = df["relative_humidity"].clip(0, 100)

    if {"u_component_of_wind_10m", "v_component_of_wind_10m"}.issubset(df.columns):
        df["wind_speed"] = np.sqrt(df["u_component_of_wind_10m"]**2 + df["v_component_of_wind_10m"]**2)

    drop_cols = ["dewpoint_temperature_2m_max", "u_component_of_wind_10m", "v_component_of_wind_10m"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    if "ndvi" not in df.columns:
        df["ndvi"] = 0.5
    df["ndvi"] = df["ndvi"].ffill(limit=MAX_FFILL_GAP)
    if df["ndvi"].isna().all():
        df["ndvi"] = 0.5

    for c in df.columns:
        if c not in [DATE_COL, TOWN_COL] and pd.api.types.is_numeric_dtype(df[c]):
            if df[c].isna().any():
                df[c] = df[c].fillna(df[c].median())

    # Return just enough rows for the lookback window
    return df.sort_values(DATE_COL).tail(LOOKBACK_DAYS + 1)

# =============================================================================
# FIRESTORE UPLOAD
# =============================================================================

def upload_predictions_to_firestore(result):
    """Upload predictions to Firestore (alerts + predictions)."""
    if FIRESTORE_DB is None:
        print(" Firestore not initialized — skipping upload")
        return

    date_str = dt.datetime.now(ZoneInfo(TIMEZONE)).strftime("%Y-%m-%d")

    for p in result["predictions"]:
        town = p["town"]
        prob = float(p["probability"])
        alert_flag = bool(p["alert"])
        severity = "High" if prob >= 0.75 else "Moderate" if prob >= 0.45 else "None"
        message = (
            f" High heatwave risk expected in {town}." if alert_flag
            else f"No heatwave expected — conditions normal in {town}."
        )

        doc_data = {
            "town": town,
            "date": date_str,
            "probability": prob,
            "alert": alert_flag,
            "severity": severity,
            "message": message,
            "timestamp": dt.datetime.now(ZoneInfo(TIMEZONE)),
        }

        FIRESTORE_DB.collection("predictions").document(f"{date_str}_{town}").set(doc_data)
        if alert_flag:
            FIRESTORE_DB.collection("alerts").add(doc_data)

    print("📡 Uploaded predictions to Firestore successfully.")

# =============================================================================
# PREDICTION PIPELINE
# =============================================================================

def prepare_window(df_recent: pd.DataFrame) -> np.ndarray:
    arr = df_recent[FEATURE_COLS].tail(LOOKBACK_DAYS).values.astype(np.float32)
    if len(arr) < LOOKBACK_DAYS:
        pad_len = LOOKBACK_DAYS - len(arr)
        arr = np.vstack([np.zeros((pad_len, arr.shape[1])), arr])
    arr_scaled = SCALER.transform(arr)
    return arr_scaled.reshape(1, LOOKBACK_DAYS, len(FEATURE_COLS))

def predict_one_town(tname: str, df_recent: pd.DataFrame) -> Dict:
    X = prepare_window(df_recent)
    prob = float(MODEL.predict(X, verbose=0).ravel()[0])
    alert = int(prob >= THRESHOLD)
    return {"town": tname, "probability": prob, "alert": alert}

def run_predictions() -> Dict:
    """Run predictions, store locally + Firestore."""
    global era5, modis_lst, modis_ndvi, towns
    if not EE_READY:
        init_gee()
    if not towns:
        build_ee_objects()

    town_centroids = {t: geom.centroid().coordinates().getInfo() for t, geom in towns.items()}
    now_ts = dt.datetime.now(ZoneInfo(TIMEZONE))
    start_date = now_ts.date()
    end_date = start_date + dt.timedelta(days=HORIZON_DAYS)
    preds = []

    all_rows = []
    for tname, geom in towns.items():
        lon, lat = town_centroids[tname]
        df_town = fetch_features_for_town(tname, geom, lon, lat, LOOKBACK_DAYS)
        if not df_town.empty:
            df_town[TOWN_COL] = tname
            all_rows.append(df_town)

    if not all_rows:
        raise RuntimeError("No data for any town")

    df_all = pd.concat(all_rows, ignore_index=True)
    last_date = pd.to_datetime(df_all[DATE_COL]).max().date()
    start_date = last_date + dt.timedelta(days=1)
    end_date = last_date + dt.timedelta(days=HORIZON_DAYS)

    preds = []
    with Session(engine) as sess:
        for tname in sorted(df_all[TOWN_COL].unique()):
            df_t = df_all[df_all[TOWN_COL] == tname].sort_values(DATE_COL)
            out = predict_one_town(tname, df_t)
            preds.append(out)
            sess.add(Prediction(
                run_ts=now_ts,
                start_date=start_date,
                end_date=end_date,
                town=tname,
                probability=out["probability"],
                alert=out["alert"]
            ))
        sess.commit()

    result = {
        "run_ts": now_ts.isoformat(),
        "start_date": str(start_date),
        "end_date": str(end_date),
        "threshold": THRESHOLD,
        "predictions": preds,
    }
    upload_predictions_to_firestore(result)
    return result

# =============================================================================
# API ROUTES
# =============================================================================

class PredictResponse(BaseModel):
    run_ts: str
    start_date: str
    end_date: str
    threshold: float
    predictions: List[Dict]

@app.get("/health")
def health():
    return {"status": "ok", "ee_ready": EE_READY}

@app.post("/predict/run", response_model=PredictResponse)
def predict_run():
    try:
        result = run_predictions()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/viz/today.png")
def viz_today_png():
    """Return today's predictions as bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = predict_run()
    if "predictions" not in payload:
        return Response(content=b"", media_type="image/png")

    towns = [p["town"] for p in payload["predictions"]]
    probs = [p["probability"] for p in payload["predictions"]]

    plt.figure(figsize=(7, 4))
    plt.bar(towns, probs, color="orange")
    plt.axhline(THRESHOLD, linestyle="--", color="red")
    plt.title("Heatwave Probabilities")
    plt.ylabel("Probability")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return Response(content=buf.read(), media_type="image/png")

# =============================================================================
# SCHEDULER
# =============================================================================

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

scheduler: Optional[BackgroundScheduler] = None


# =============================================================================
# FIRESTORE READ ENDPOINTS (for Dashboard / Mobile App)
# =============================================================================

@app.get("/firestore/predictions/today")
def firestore_predictions_today():
    """Fetch today's predictions for all towns from Firestore."""
    try:
        if FIRESTORE_DB is None:
            raise HTTPException(status_code=500, detail="Firestore not initialized")

        today_str = dt.datetime.now(ZoneInfo(TIMEZONE)).strftime("%Y-%m-%d")
        docs = FIRESTORE_DB.collection("predictions").where("date", "==", today_str).stream()

        results = []
        for doc in docs:
            data = doc.to_dict()
            results.append({
                "town": data.get("town"),
                "probability": data.get("probability"),
                "alert": data.get("alert"),
                "severity": data.get("severity"),
                "message": data.get("message"),
                "date": data.get("date"),
            })

        if not results:
            return {"message": "It is a normal day, no heatwave."}

        return {
            "date": today_str,
            "count": len(results),
            "predictions": sorted(results, key=lambda x: x["town"]),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/firestore/alerts/latest")
def firestore_latest_alerts():
    """Fetch the latest alerts from Firestore."""
    try:
        if FIRESTORE_DB is None:
            raise HTTPException(status_code=500, detail="Firestore not initialized")

        alerts_ref = (
            FIRESTORE_DB.collection("alerts")
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .limit(10)
        )
        docs = alerts_ref.stream()
        results = [doc.to_dict() for doc in docs]

        if not results:
            return {"message": "No recent alerts found."}

        return {"latest_alerts": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/firestore/history/{days}")
def firestore_prediction_history(days: int = 7):
    """Fetch past N days of prediction data."""
    try:
        if FIRESTORE_DB is None:
            raise HTTPException(status_code=500, detail="Firestore not initialized")

        end_date = dt.datetime.now(ZoneInfo(TIMEZONE))
        start_date = end_date - dt.timedelta(days=days)

        all_docs = FIRESTORE_DB.collection("predictions").stream()
        results = []
        for doc in all_docs:
            data = doc.to_dict()
            date_str = data.get("date")
            if date_str:
                date_val = dt.datetime.strptime(date_str, "%Y-%m-%d")
                if start_date.date() <= date_val.date() <= end_date.date():
                    results.append(data)

        if not results:
            return {"message": f"No predictions found in the past {days} days."}

        grouped = {}
        for r in results:
            grouped.setdefault(r["date"], []).append(r)

        return {
            "start_date": str(start_date.date()),
            "end_date": str(end_date.date()),
            "records": grouped,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def scheduled_job():
    try:
        print("Running scheduled predictions...")
        result = run_predictions()
        print("Scheduled run done.")
    except Exception as e:
        print(f"Scheduled run failed: {e}")

@app.get("/scheduler/status")
def scheduler_status():
    jobs = []
    if scheduler:
        for j in scheduler.get_jobs():
            jobs.append({
                "id": j.id,
                "next_run_time": str(j.next_run_time),
                "trigger": str(j.trigger)
            })
    return {"enabled": SCHEDULER_ENABLED, "jobs": jobs}

@app.post("/scheduler/run-now")
def scheduler_run_now():
    scheduled_job()
    return {"status": "ok", "message": "Scheduled job executed immediately"}

# =============================================================================
# APP LIFECYCLE
# =============================================================================

TIMEZONE = "Africa/Kigali"  #  added timezone configuration

@app.on_event("startup")
def on_startup():
    SQLModel.metadata.create_all(engine)
    load_artifacts()
    init_gee()
    build_ee_objects()
    init_firestore()

    if SCHEDULER_ENABLED:
        global scheduler
        scheduler = BackgroundScheduler(timezone=ZoneInfo(TIMEZONE))
        #  ensure it runs daily at 7 AM Kigali time
        scheduler.add_job(
            scheduled_job,
            CronTrigger(hour=7, minute=0, timezone=ZoneInfo(TIMEZONE)),
            id="daily-07",
            replace_existing=True
        )
        scheduler.start()
        print("Scheduler started for 07:00 daily (Africa/Kigali)")

@app.on_event("shutdown")
def on_shutdown():
    global scheduler
    if scheduler:
        scheduler.shutdown(wait=False)
        print("Scheduler stopped")
