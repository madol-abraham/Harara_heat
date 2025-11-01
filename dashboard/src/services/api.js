import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'https://harara-heat-dror.onrender.com';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Token management
let authToken = localStorage.getItem('harara_token');

// Add token to requests
api.interceptors.request.use((config) => {
  if (authToken) {
    config.headers.Authorization = `Bearer ${authToken}`;
  }
  return config;
});

// Handle token expiry
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('harara_token');
      localStorage.removeItem('harara_token_expiry');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export const apiService = {
  // Authentication
  login: (credentials) => api.post('/auth/login', credentials),
  verifyToken: () => api.post('/auth/verify'),
  getUserInfo: () => api.get('/auth/me'),
  
  // Health check
  getHealth: () => api.get('/health'),
  
  // Dashboard stats
  getDashboardStats: () => api.get('/dashboard/stats'),
  
  // Predictions
  runPredictions: () => api.post('/predict/run'),
  getTodayPredictions: () => api.get('/firestore/predictions/today'),
  getPredictionHistory: (days = 7) => api.get(`/firestore/history/${days}`),
  
  // Alerts
  getLatestAlerts: () => api.get('/firestore/alerts/latest'),
  sendManualAlert: (data) => api.post('/alerts/manual', data),
  triggerDemoAlert: () => api.post('/alerts/trigger-demo'),
  
  // Users
  getTownUsers: (town) => api.get(`/users/town/${town}`),
  registerUser: (data) => api.post('/users/register', data),
  
  // Scheduler
  getSchedulerStatus: () => api.get('/scheduler/status'),
  runSchedulerNow: () => api.post('/scheduler/run-now'),
  
  // Export
  exportPredictions: () => api.get('/api/export/predictions', { responseType: 'blob' }),
  exportLogs: () => api.get('/api/export/logs', { responseType: 'blob' }),
  exportReport: () => api.get('/api/export/report', { responseType: 'blob' }),
  
  // Visualization
  getTodayChart: () => api.get('/viz/today.png', { responseType: 'blob' }),
};

// Update token function
export const setAuthToken = (token) => {
  authToken = token;
  if (token) {
    api.defaults.headers.common['Authorization'] = `Bearer ${token}`;
  } else {
    delete api.defaults.headers.common['Authorization'];
  }
};

export default api;