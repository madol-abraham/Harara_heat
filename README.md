#  Harara: Real-Time Heatwave Monitoring and Alert System for South Sudan

##  Description  

**Harara** is a machine learning–powered **heatwave forecasting and early-warning system** built to help communities in **South Sudan** anticipate and respond to extreme heat events.  
The project integrates **Google Earth Engine (GEE)** satellite data, a **Long Short-Term Memory (LSTM)** deep learning model, a **FastAPI backend**, and a **Flutter mobile app** to provide **7-day-ahead heatwave forecasts** and actionable alerts.  

The goal is to reduce the human and economic impacts of heatwaves by providing early, reliable, and accessible warnings for local communities, schools, farmers, and authorities.

---

##  Link to the Github repo: https://github.com/madol-abraham/Harara_heat.git  

**Demo:** https://youtu.be/CRM0xN97bFI

---

##  Environment and Project Setup  

###  Clone the Repository  
```bash
git clone https://github.com/madol-abraham/harara-heatwave.git
cd harara-heatwave

```

## Frontend UI interface


<img width="304" height="613" alt="iPhone-13-PRO-MAX-localhost (1)" src="https://github.com/user-attachments/assets/68fd5b62-f0bf-4b08-a488-f83e66d75ff0" />

<img width="304" height="613" alt="iPhone-13-PRO-MAX-localhost (2)" src="https://github.com/user-attachments/assets/b4633066-64da-4e5e-a5b7-704273e8cab5" />

<img width="304" height="613" alt="iPhone-13-PRO-MAX-localhost (3)" src="https://github.com/user-attachments/assets/830baca6-bc85-4067-a987-98d07da7d094" />

<img width="304" height="613" alt="iPhone-13-PRO-MAX-localhost (4)" src="https://github.com/user-attachments/assets/b90bf538-e946-4202-8031-7b6ea843ad4a" />

## Deployment Plan
### Phase 1 – Local Development ( Completed)

- LSTM model trained and evaluated with high accuracy
- FastAPI backend running locally on localhost:8000
-Flutter app (9 fully designed pages) completed and tested with mock data

### Phase 2 – API Hosting ( In Progress)

- Deploy FastAPI on Render, Railway, or Azure App Service
- Enable CORS for Flutter–API communication
- Automate daily model runs (07:00 UTC) for live predictions

### Phase 3 – App Integration ( Next)

- Connect Flutter app to API endpoints: /predict, /latest, /history
- Enable real-time forecast visualization
- Add Firebase Cloud Messaging (FCM) for push and SMS alerts

### Phase 4 – Public Deployment (Planned)

- Host model and API in the cloud for 24/7 access
- Deploy app on Google Play Store and App Gallery
- Integrate with government meteorological data system

## Project Summary
| Component            | Framework / Tool                               |
| -------------------- | ---------------------------------------------- |
| **Machine Learning** | TensorFlow / Keras (LSTM), scikit-learn        |
| **Data Source**      | Google Earth Engine (MODIS, ERA5)              |
| **Backend**          | FastAPI, Uvicorn                               |
| **Frontend**         | Flutter (Dart)                                 |
| **Database**         | SQLite (local) → Firebase (planned cloud sync) |
| **Visualization**    | Matplotlib, Seaborn                            |
| **Deployment**       | Render / Railway / Firebase Hosting            |

## Model Performance Summary

### The two-layer LSTM model achieved excellent generalization and predictive capability.

| Dataset        | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
| -------------- | -------- | --------- | ------ | -------- | ------- | ------ |
| **Validation** | 0.87     | 0.83      | 0.96   | 0.89     | 0.927   | 0.894  |
| **Test**       | 0.86     | 0.84      | 0.78   | 0.81     | 0.908   | 0.898  |

## Folder Structure
Harara_heat/
│
├── Api/                           
│   ├── .dockerignore
│   ├── Dockerfile                  
│   ├── main.py                     
│   ├── render.yaml                 
│   └── requirements.txt            
│
├── harara_app/                     
│   ├── .dart_tool/                 
│   ├── android/                   
│   ├── build/                      
│   ├── ios/                        
│   ├── lib/                        
│   │   ├── core/theme/             
│   │   ├── models/                 
│   │   ├── navigation/             
│   │   ├── routes/                 
│   │   ├── screens/               
│   │   ├── services/           
│   │   ├── widgets/                
│   │   ├── app.dart               
│   │   └── main.dart               
│   ├── pubspec.yaml                
│   └── test/                       
│
├── .gitignore                      
├── Heatwave_work.ipynb             
└── README.md              



## Interpretation

AUC values (0.91–0.93) and PR-AUC (0.89–0.90) indicate excellent class separation.

The balance between precision and recall confirms reliable heatwave detection with minimal false positives.

Nearly identical validation and test performance confirms low overfitting.

Early stopping at Epoch 23 ensured optimal convergence and model efficiency.

##  Insights

The model effectively learns from 21-day sequences of climatic data.

Features like LST_Day_1km, air_temp_2m, and relative_humidity drive predictive performance.

This architecture is now ready for operational deployment in Harara’s real-time alert system.
## Author

**Madol Abraham Kuol Madol**
Bachelor of Software Engineering (Hons), African Leadership University — Kigali, Rwanda
Machine Learning Engineer | AI for Climate Resilience Researcher
