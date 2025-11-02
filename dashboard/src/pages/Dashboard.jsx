import React, { useState, useEffect } from 'react';
import { AlertTriangle, Activity, Users, MapPin, RefreshCw } from 'lucide-react';
import StatsCard from '../components/StatsCard';
import { apiService } from '../services/api';

const Dashboard = () => {
  const [predictions, setPredictions] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [health, setHealth] = useState(null);
  const [users, setUsers] = useState([]);
  const [schedulerStatus, setSchedulerStatus] = useState(null);
  const [dashboardStats, setDashboardStats] = useState(null);
  const [loading, setLoading] = useState(true);

  const fetchData = async () => {
    try {
      setLoading(true);
      const [predictionsRes, alertsRes, healthRes, schedulerRes, statsRes] = await Promise.all([
        apiService.getTodayPredictions(),
        apiService.getLatestAlerts(),
        apiService.getHealth(),
        apiService.getSchedulerStatus(),
        apiService.getDashboardStats(),
      ]);

      setPredictions(predictionsRes.data.predictions || []);
      setAlerts(alertsRes.data.latest_alerts || []);
      setHealth(healthRes.data);
      setSchedulerStatus(schedulerRes.data);
      setDashboardStats(statsRes.data);
      
      const userPromises = ['Juba', 'Wau', 'Yambio', 'Bor', 'Malakal', 'Bentiu'].map(
        town => apiService.getTownUsers(town).catch(() => ({ data: { users: [] } }))
      );
      const userResults = await Promise.all(userPromises);
      const allUsers = userResults.flatMap(res => res.data.users || []);
      setUsers(allUsers);
    } catch (error) {
      console.error('Error fetching data:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const activeAlerts = dashboardStats?.active_alerts || 0;
  const highRiskTowns = dashboardStats?.high_risk_towns || 0;
  const totalUsers = dashboardStats?.total_users || 0;
  const systemStatus = dashboardStats?.system_status || health;
  
  const runPredictions = async () => {
    try {
      setLoading(true);
      await apiService.runPredictions();
      await fetchData();
      alert('Predictions updated successfully!');
    } catch (error) {
      console.error('Error running predictions:', error);
      alert('Failed to run predictions');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="h-8 w-8 animate-spin" style={{color: 'var(--secondary)'}} />
        <span className="ml-2" style={{color: 'var(--text)'}}>Loading dashboard...</span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold" style={{color: 'var(--text)'}}>Dashboard Overview</h1>
        <button
          onClick={fetchData}
          className="flex items-center px-4 py-2 text-white rounded-lg hover:opacity-90 transition-colors"
          style={{backgroundColor: 'var(--secondary)'}}
        >
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatsCard
          title="Active Alerts"
          value={activeAlerts}
          icon={AlertTriangle}
          color={activeAlerts > 0 ? "red" : "green"}
          subtitle={activeAlerts > 0 ? "Requiring attention" : "No active alerts"}
        />
        <StatsCard
          title="High Risk Towns"
          value={highRiskTowns}
          icon={MapPin}
          color={highRiskTowns > 0 ? "red" : "green"}
          subtitle={highRiskTowns > 0 ? "Above 75% probability" : "No high risk towns"}
        />
        <StatsCard
          title="System Status"
          value={systemStatus?.status === 'ok' ? "Online" : "Offline"}
          icon={Activity}
          color={systemStatus?.status === 'ok' ? "info" : "red"}
          subtitle={`EE: ${systemStatus?.ee_ready ? 'Ready' : 'Not Ready'} | Model: ${systemStatus?.model_loaded ? 'Loaded' : 'Not Loaded'}`}
        />
        <StatsCard
          title="Registered Users"
          value={totalUsers}
          icon={Users}
          color="default"
          subtitle="SMS alert recipients"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div style={{backgroundColor: 'var(--surface)', borderColor: 'var(--border)'}} className="rounded-lg p-4 border">
          <h3 className="text-lg font-semibold mb-4" style={{color: 'var(--text)'}}>Recent Alerts</h3>
          <div className="space-y-3">
            {alerts.slice(0, 5).map((alert, index) => {
              const alertTime = alert.timestamp?.seconds ? 
                new Date(alert.timestamp.seconds * 1000) : 
                new Date(alert.date || alert.timestamp);
              
              return (
                <div key={index} className="rounded-lg p-3" style={{backgroundColor: 'var(--background)', border: '1px solid var(--border)'}}>
                  <div className="flex justify-between items-start">
                    <div>
                      <h4 className="font-medium" style={{color: 'var(--text)'}}>{alert.town}</h4>
                      <p className="text-sm mt-1" style={{color: 'var(--text-secondary)'}}>{alert.message}</p>
                    </div>
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      alert.severity === 'High' ? 'bg-red-500 text-white' :
                      alert.severity === 'Moderate' ? 'bg-yellow-500 text-white' :
                      'bg-green-500 text-white'
                    }`}>
                      {alert.severity || 'Info'}
                    </span>
                  </div>
                  <div className="flex justify-between items-center mt-2">
                    <p className="text-xs" style={{color: 'var(--text-secondary)'}}>
                      {alertTime.toLocaleString()}
                    </p>
                    <span className={`px-2 py-1 rounded text-xs ${
                      alert.alert ? 'bg-red-600 text-white' : 'bg-gray-600 text-white'
                    }`}>
                      {alert.alert ? 'ACTIVE' : 'INFO'}
                    </span>
                  </div>
                </div>
              );
            })}
            {alerts.length === 0 && (
              <p className="text-center py-4" style={{color: 'var(--text-secondary)'}}>No recent alerts</p>
            )}
          </div>
        </div>
        
        <div style={{backgroundColor: 'var(--surface)', borderColor: 'var(--border)'}} className="rounded-lg p-4 border">
          <h3 className="text-lg font-semibold mb-4" style={{color: 'var(--text)'}}>Quick Actions</h3>
          <div className="space-y-3">
            <button
              onClick={() => window.location.href = '/alerts'}
              className="w-full text-white py-2 px-4 rounded-lg hover:opacity-90 transition-colors"
              style={{backgroundColor: 'var(--warning)'}}
            >
              Send Manual Alert
            </button>
            <button
              onClick={() => window.location.href = '/export'}
              className="w-full text-white py-2 px-4 rounded-lg hover:opacity-90 transition-colors"
              style={{backgroundColor: 'var(--secondary)'}}
            >
              Export Data
            </button>
            <button
              onClick={runPredictions}
              disabled={loading}
              className="w-full text-white py-2 px-4 rounded-lg hover:opacity-90 transition-colors disabled:opacity-50"
              style={{backgroundColor: 'var(--success)'}}
            >
              {loading ? 'Running...' : 'Run Predictions'}
            </button>
            <button
              onClick={fetchData}
              className="w-full text-white py-2 px-4 rounded-lg hover:opacity-90 transition-colors"
              style={{backgroundColor: 'var(--secondary)'}}
            >
              Refresh Data
            </button>
          </div>
        </div>
      </div>

      <div style={{backgroundColor: 'var(--surface)', borderColor: 'var(--border)'}} className="rounded-lg p-4 border">
        <h3 className="text-lg font-semibold mb-4" style={{color: 'var(--text)'}}>Today's Predictions</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b" style={{borderColor: 'var(--border)'}}>
                <th className="text-left py-3 px-4" style={{color: 'var(--text-secondary)'}}>Town</th>
                <th className="text-left py-3 px-4" style={{color: 'var(--text-secondary)'}}>Probability</th>
                <th className="text-left py-3 px-4" style={{color: 'var(--text-secondary)'}}>Risk Level</th>
                <th className="text-left py-3 px-4" style={{color: 'var(--text-secondary)'}}>Alert Status</th>
              </tr>
            </thead>
            <tbody>
              {predictions.map((prediction, index) => (
                <tr key={index} className="border-b" style={{borderColor: 'var(--border)'}}>
                  <td className="py-3 px-4 font-medium" style={{color: 'var(--text)'}}>{prediction.town}</td>
                  <td className="py-3 px-4" style={{color: 'var(--text)'}}>
                    {typeof prediction.probability === 'number' ? 
                      (prediction.probability * 100).toFixed(1) + '%' : 
                      'N/A'
                    }
                  </td>
                  <td className="py-3 px-4">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      prediction.probability >= 0.75 ? 'bg-red-500 text-white' :
                      prediction.probability >= 0.67 ? 'bg-yellow-500 text-white' :
                      'bg-green-500 text-white'
                    }`}>
                      {prediction.probability >= 0.75 ? 'High' :
                       prediction.probability >= 0.67 ? 'Moderate' : 'Low'}
                    </span>
                  </td>
                  <td className="py-3 px-4">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      prediction.alert ? 'bg-red-500 text-white' : 'bg-gray-500 text-white'
                    }`}>
                      {prediction.alert ? 'Active' : 'None'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;