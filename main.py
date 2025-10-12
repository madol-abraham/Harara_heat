# main.py
# =============================================================================
# Harara Heatwave API
# - Loads trained artifacts (scaler, model, threshold)
# - Pulls features from Google Earth Engine
# - Produces 7-day heatwave risk per town
# - Stores results in SQLite
# - Exposes HTTP endpoints (manual run, latest results, quick viz, mock)
# - Runs an automated daily prediction at 07:00
# - Includes endpoints to view scheduler status and to run the scheduled
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

import tensorflow as tf  # use tf.keras only
import joblib
import ee

# =============================================================================
# CONFIG / CONSTANTS
# =============================================================================

# Portable defaults: artifacts & DB live beside this file
ARTIFACT_DIR = os.getenv(
    "ARTIFACT_DIR",
    os.path.join(os.path.dirname(__file__), "harara_artifacts")
)
DB_PATH = os.getenv("DB_PATH", os.path.join(os.path.dirname(__file__), "harara.db"))
TIMEZONE = os.getenv("TIMEZONE", "Africa/Kigali")

LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "21"))
HORIZON_DAYS = int(os.getenv("HORIZON_DAYS", "7"))
MAX_FFILL_GAP = 5

# Set to "0" to disable APScheduler (e.g., when running multiple workers)
SCHEDULER_ENABLED = os.getenv("SCHEDULER_ENABLED", "1") == "1"

DATE_COL = "date"
TOWN_COL = "town"
FEATURE_COLS = [
    "LST_Day_1km", "LST_Night_1km", "air_temp_2m", "ndvi",
    "net_solar_radiation", "precipitation", "relative_humidity",
    "soil_moisture", "wind_speed", "longitude", "latitude"
]

# Earth Engine service account (recommended in production)
# You may either:
#  1) Put the service account email & key path in env vars:
#       EE_SERVICE_ACCOUNT, EE_PRIVATE_KEY_JSON_PATH
#  2) Hardcode a pair below (only for quick local testing)
EE_SERVICE_ACCOUNT = os.getenv("EE_SERVICE_ACCOUNT")
EE_PRIVATE_KEY_JSON_PATH = os.getenv("EE_PRIVATE_KEY_JSON_PATH")

# (Optional local quickstart: uncomment & set your local JSON key file)
LOCAL_EE_SERVICE_ACCOUNT = "harara-service@south-sudan-heatwave.iam.gserviceaccount.com"
LOCAL_EE_KEY_FILE = "south-sudan-heatwave-583da500ae5f.json"  # ensure this file exists locally

# =============================================================================
# FASTAPI APP & CORS
# =============================================================================

app = FastAPI(
    title="Harara Heatwave API",
    version="1.0.0",
    docs_url="/docs",     # Swagger UI
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# DATABASE (SQLite via SQLModel)
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
    details_json: Optional[str] = None  # place for feature / debug info

# =============================================================================
# GLOBAL STATE (populated at startup)
# =============================================================================

MODEL: Optional[tf.keras.Model] = None
SCALER = None
CALIBRATOR = None
THRESHOLD: float = 0.5

# EE state placeholders – created AFTER ee.Initialize()
era5 = None
modis_lst = None
modis_ndvi = None
towns: Dict[str, ee.Geometry] = {}
EE_READY = False

# =============================================================================
# GOOGLE EARTH ENGINE INIT
# =============================================================================

EE_READY = False  # global flag for GEE initialization
def init_gee():
    """Initialize Google Earth Engine using a service account key from environment variables."""
    global EE_READY
    try:
        key_json = os.getenv("EE_SERVICE_KEY")
        if not key_json:
            raise ValueError("Missing EE_SERVICE_KEY environment variable")

        # Parse JSON string from environment
        service_account_info = json.loads(key_json)
        credentials = ee.ServiceAccountCredentials(
            service_account_info["client_email"],
            key_data=key_json
        )
        ee.Initialize(credentials)
        EE_READY = True
        print("✅ Earth Engine initialized using Render service account key")
    except Exception as e:
        EE_READY = False
        print(f"❌ EE init error: {e}")
def build_ee_objects():
    """Create EE ImageCollections and town geometries AFTER init."""
    global era5, modis_lst, modis_ndvi, towns
    if not EE_READY:
        raise RuntimeError("Earth Engine not initialized yet")

    era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
    modis_lst = ee.ImageCollection("MODIS/061/MOD11A1")
    modis_ndvi = ee.ImageCollection("MODIS/061/MOD13Q1")

    # Towns (3 km buffer)
    towns = {
        "Juba":    ee.Geometry.Point([31.5804, 4.8594]).buffer(3000),
        "Wau":     ee.Geometry.Point([28.0070, 7.7011]).buffer(3000),
        "Yambio":  ee.Geometry.Point([28.4167, 4.5700]).buffer(3000),
        "Bor":     ee.Geometry.Point([31.5594, 6.2065]).buffer(3000),
        "Malakal": ee.Geometry.Point([32.4730, 9.5330]).buffer(3000),
        "Bentiu":  ee.Geometry.Point([29.7820, 9.2330]).buffer(3000),
    }
    print(" EE collections & towns ready")

# =============================================================================
# ARTIFACT LOADING
# =============================================================================

def load_artifacts():
    """
    Loads:
      - threshold.json -> THRESHOLD
      - scaler.pkl     -> SCALER
      - model.keras    -> MODEL (tf.keras)
      - calibrator.pkl -> CALIBRATOR (optional, e.g., Platt/isotonic)
    """
    global MODEL, SCALER, CALIBRATOR, THRESHOLD

    thr_path = os.path.join(ARTIFACT_DIR, "threshold.json")
    sc_path  = os.path.join(ARTIFACT_DIR, "scaler.pkl")
    mdl_path = os.path.join(ARTIFACT_DIR, "model.keras")
    cal_path = os.path.join(ARTIFACT_DIR, "calibrator.pkl")  # optional

    if not os.path.exists(thr_path):
        raise FileNotFoundError(f"threshold.json not found at {thr_path}")
    if not os.path.exists(sc_path):
        raise FileNotFoundError(f"scaler.pkl not found at {sc_path}")
    if not os.path.exists(mdl_path):
        raise FileNotFoundError(f"model.keras not found at {mdl_path}")

    with open(thr_path) as f:
        THRESHOLD = float(json.load(f)["threshold"])

    SCALER = joblib.load(sc_path)
    MODEL = tf.keras.models.load_model(mdl_path)

    CALIBRATOR = None
    if os.path.exists(cal_path):
        try:
            CALIBRATOR = joblib.load(cal_path)
            print(" Calibrator loaded")
        except Exception as e:
            print(f" Failed to load calibrator: {e}")
            CALIBRATOR = None

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
# PREDICTION PIPE
# =============================================================================

def prepare_window(df_recent: pd.DataFrame) -> np.ndarray:
    """Make (1, LOOKBACK_DAYS, N_FEATURES) tensor with scaling & padding."""
    arr = df_recent[FEATURE_COLS].tail(LOOKBACK_DAYS).values.astype(np.float32)
    if len(arr) < LOOKBACK_DAYS:
        pad_len = LOOKBACK_DAYS - len(arr)
        arr = np.vstack([np.zeros((pad_len, arr.shape[1])), arr])
    if np.isnan(arr).any():
        arr = np.nan_to_num(arr, nan=0.0)
    arr_scaled = SCALER.transform(arr)  # scaler trained on 2D slices
    return arr_scaled.reshape(1, LOOKBACK_DAYS, len(FEATURE_COLS))

def predict_one_town(tname: str, df_recent: pd.DataFrame) -> Dict:
    """Return {town, probability, alert} using model + threshold (+ optional calibrator)."""
    X = prepare_window(df_recent)
    prob = float(MODEL.predict(X, verbose=0).ravel()[0])
    if CALIBRATOR is not None:
        prob = float(CALIBRATOR.transform([prob])[0])
    alert = int(prob >= THRESHOLD)
    return {"town": tname, "probability": prob, "alert": alert}

def _ensure_ee_objects_ready():
    """Guard to ensure EE init and objects exist before any run."""
    global era5, modis_lst, modis_ndvi, towns
    if not EE_READY:
        init_gee()
    if any(x is None for x in (era5, modis_lst, modis_ndvi)) or not towns:
        build_ee_objects()

def run_predictions() -> Dict:
    """
    Pull recent features for each town from EE, run the model, store results in DB,
    and return a structured payload including 7-day window dates.
    """
    _ensure_ee_objects_ready()

    # Town centroids for longitude / latitude features
    town_centroids = {t: geom.centroid().coordinates().getInfo() for t, geom in towns.items()}

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
    now_ts = dt.datetime.now(ZoneInfo(TIMEZONE))
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
                alert=out["alert"],
                details_json=None
            ))
        sess.commit()

    return {
        "run_ts": now_ts.isoformat(),
        "start_date": str(start_date),
        "end_date": str(end_date),
        "threshold": THRESHOLD,
        "predictions": preds,
    }

# =============================================================================
# NOTIFICATIONS (stub) — integrate FCM/SMS later
# =============================================================================

def send_alerts_if_needed(result: Dict):
    alerts = [p for p in result["predictions"] if p["alert"] == 1]
    if alerts:
        print("🚨 Send notifications for:", [a["town"] for a in alerts])
    # TODO: integrate FCM / SMS.

# =============================================================================
# API MODELS
# =============================================================================

class PredictResponse(BaseModel):
    run_ts: str
    start_date: str
    end_date: str
    threshold: float
    predictions: List[Dict]

class MockRequest(BaseModel):
    level: str = "alert"  # "alert" or "normal"

# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/health")
def health():
    """Quick status check."""
    return {
        "status": "ok",
        "tz": TIMEZONE,
        "ee_ready": EE_READY,
    }

@app.post("/predict/run", response_model=PredictResponse)
def predict_run():
    """
    Manually trigger a real prediction run (used by admin panel, cron, etc.).
    Pulls EE data, runs the model, stores results in DB, and returns the payload.
    """
    try:
        result = run_predictions()
        send_alerts_if_needed(result)
        return result
    except ee.EEException as e:
        raise HTTPException(status_code=500, detail=(
            f"Earth Engine error: {e}. "
            "If this is the first run on this machine, run `earthengine authenticate` "
            "or set EE_SERVICE_ACCOUNT/EE_PRIVATE_KEY_JSON_PATH."
        ))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predictions/today")
def predictions_today():
    """
    Return the latest run for today (if any) aggregated by run_ts.
    Useful for dashboards.
    """
    today = dt.datetime.now(ZoneInfo(TIMEZONE)).date()
    with Session(engine) as sess:
        stmt = (
            select(Prediction)
            .where(Prediction.run_ts >= dt.datetime.combine(today, dt.time(0, 0, tzinfo=ZoneInfo(TIMEZONE))))
            .order_by(Prediction.run_ts.desc())
        )
        rows = sess.exec(stmt).all()
        if not rows:
            return {"message": "No predictions today yet"}

        by_run: Dict[str, Dict] = {}
        for r in rows:
            key = r.run_ts.isoformat()
            by_run.setdefault(key, {
                "run_ts": key,
                "start_date": str(r.start_date),
                "end_date": str(r.end_date),
                "predictions": []
            })
            by_run[key]["predictions"].append({
                "town": r.town, "probability": r.probability, "alert": r.alert
            })
        latest_key = sorted(by_run.keys())[-1]
        return by_run[latest_key]

@app.get("/viz/today.png")
def viz_today_png():
    """Quick bar chart PNG of today's latest probabilities."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = predictions_today()
    if "predictions" not in payload:
        return Response(content=b"", media_type="image/png", headers={"X-Empty": "1"})

    towns_list = [p["town"] for p in payload["predictions"]]
    probs = [p["probability"] for p in payload["predictions"]]

    fig = plt.figure(figsize=(7, 4))
    plt.title(f"Heatwave Probabilities — {payload['start_date']} → {payload['end_date']}")
    plt.bar(towns_list, probs)
    plt.axhline(THRESHOLD, linestyle="--")
    plt.ylim(0, 1)
    plt.ylabel("Probability")
    plt.xlabel("Town")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Response(content=buf.read(), media_type="image/png")

@app.post("/predict/mock", response_model=PredictResponse)
def predict_mock(req: MockRequest):
    """
    MOCK endpoint for demos: does NOT hit EE or the model.
    - level="alert": forces high probabilities (~0.85)
    - level="normal": forces low probabilities (~0.22)
    """
    now = dt.datetime.now(ZoneInfo(TIMEZONE))
    start_date = now.date() + dt.timedelta(days=1)
    end_date = now.date() + dt.timedelta(days=HORIZON_DAYS)

    # Ensure towns exist (for consistency of response)
    _ensure_ee_objects_ready()

    preds = []
    for t in towns.keys():
        prob = 0.85 if req.level.lower() == "alert" else 0.22
        preds.append({"town": t, "probability": prob, "alert": int(prob >= THRESHOLD)})

    # Store mock into DB so UI still has "latest run"
    with Session(engine) as sess:
        for p in preds:
            sess.add(Prediction(
                run_ts=now, start_date=start_date, end_date=end_date,
                town=p["town"], probability=p["probability"], alert=p["alert"]
            ))
        sess.commit()

    return {
        "run_ts": now.isoformat(),
        "start_date": str(start_date),
        "end_date": str(end_date),
        "threshold": THRESHOLD,
        "predictions": preds,
    }

# =============================================================================
# SCHEDULING 
# =============================================================================

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

scheduler: Optional[BackgroundScheduler] = None

def scheduled_job():
    """What runs every day at 07:00 local TIMEZONE."""
    try:
        print(" Running scheduled predictions...")
        result = run_predictions()
        send_alerts_if_needed(result)
        print(" Scheduled predictions done")
    except Exception as e:
        print(" Scheduled run failed:", e)

# --- Scheduler control endpoints  ---

@app.get("/scheduler/status")
def scheduler_status():
    """
    See which jobs are registered and their next_run_time (helpful for debugging).
    """
    jobs = []
    if scheduler:
        for j in scheduler.get_jobs():
            jobs.append({
                "id": j.id,
                "next_run_time": (
                    j.next_run_time.astimezone(ZoneInfo(TIMEZONE)).isoformat()
                    if j.next_run_time else None
                ),
                "trigger": str(j.trigger)
            })
    return {"timezone": TIMEZONE, "enabled": SCHEDULER_ENABLED, "jobs": jobs}

@app.post("/scheduler/run-now")
def scheduler_run_now():
    """
    Manually execute the same function that the 07:00 job runs.
    Good for one-click verification.
    """
    scheduled_job()
    return {"status": "ok", "message": "Scheduled job executed immediately"}

# =============================================================================
# APP LIFECYCLE
# =============================================================================

@app.on_event("startup")
def on_startup():
    # Create DB tables
    SQLModel.metadata.create_all(engine)

    # Load ML artifacts
    load_artifacts()

    # Init Earth Engine & build ImageCollections/towns
    init_gee()
    build_ee_objects()

    # Start scheduler (07:00 daily)
    # Use a fixed id and replace_existing to avoid duplicates on reload.
    if SCHEDULER_ENABLED:
        global scheduler
        scheduler = BackgroundScheduler(timezone=ZoneInfo(TIMEZONE))
        scheduler.add_job(
            scheduled_job,
            CronTrigger(hour=7, minute=0),
            id="daily-07",
            replace_existing=True,
        )
        scheduler.start()
        print(" Scheduler started for 07:00 daily")
    else:
        print(" Scheduler disabled (SCHEDULER_ENABLED=0)")

@app.on_event("shutdown")
def on_shutdown():
    global scheduler
    if scheduler:
        scheduler.shutdown(wait=False)
        print(" Scheduler stopped")
