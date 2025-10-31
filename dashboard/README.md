# Harara Heatwave Dashboard

A React-based web dashboard for monitoring and managing the Harara Heatwave Early Warning System in South Sudan.

## Features

- **Real-time Dashboard**: Live heatwave risk monitoring with interactive map
- **Alert Management**: Send manual alerts and view alert history
- **Data Export**: Export predictions, logs, and generate PDF reports
- **System Monitoring**: Track API health and system status
- **Responsive Design**: Works on desktop, tablet, and mobile devices

## Quick Start

### Development

1. Install dependencies:
```bash
cd dashboard
npm install
```

2. Create environment file:
```bash
cp .env.example .env
```

3. Update `.env` with your API URL:
```
REACT_APP_API_URL=http://localhost:8000
```

4. Start development server:
```bash
npm start
```

The dashboard will be available at `http://localhost:3000`

### Production Build

```bash
npm run build
```

## Netlify Deployment

1. **Connect Repository**: Link your GitHub repo to Netlify
2. **Build Settings**: 
   - Build command: `npm run build`
   - Publish directory: `build`
3. **Environment Variables**: Set `REACT_APP_API_URL` to your production API URL
4. **Deploy**: Netlify will automatically deploy on git push

## API Integration

The dashboard connects to your FastAPI backend using these endpoints:

- `GET /health` - System health check
- `GET /firestore/predictions/today` - Current predictions
- `GET /firestore/alerts/latest` - Recent alerts
- `POST /alerts/manual` - Send manual alerts
- `GET /api/export/*` - Data export endpoints

## Project Structure

```
dashboard/
├── public/           # Static files
├── src/
│   ├── components/   # Reusable components
│   ├── pages/        # Page components
│   ├── services/     # API services
│   ├── utils/        # Utilities and constants
│   └── App.jsx       # Main app component
├── package.json      # Dependencies
└── netlify.toml      # Netlify config
```

## Technologies Used

- **React 18** - Frontend framework
- **Tailwind CSS** - Styling
- **React Router** - Navigation
- **Leaflet** - Interactive maps
- **Chart.js** - Data visualization
- **Axios** - API requests
- **Lucide React** - Icons