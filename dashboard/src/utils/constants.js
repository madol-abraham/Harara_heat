export const TOWNS = [
  { name: 'Juba', lat: 4.8594, lng: 31.5804 },
  { name: 'Wau', lat: 7.7011, lng: 28.0070 },
  { name: 'Yambio', lat: 4.5700, lng: 28.4167 },
  { name: 'Bor', lat: 6.2065, lng: 31.5594 },
  { name: 'Malakal', lat: 9.5330, lng: 32.4730 },
  { name: 'Bentiu', lat: 9.2330, lng: 29.7820 },
];

export const SEVERITY_COLORS = {
  High: '#ef4444',
  Moderate: '#f59e0b',
  Low: '#10b981',
  None: '#6b7280',
};

export const RISK_LEVELS = {
  HIGH: { min: 0.75, color: '#ef4444', label: 'High Risk' },
  MODERATE: { min: 0.67, color: '#f59e0b', label: 'Moderate Risk' },
  LOW: { min: 0.0, color: '#10b981', label: 'Low Risk' },
};