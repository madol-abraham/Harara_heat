import React, { useState, useEffect } from 'react';
import { Send, AlertTriangle, CheckCircle, Clock } from 'lucide-react';
import { apiService } from '../services/api';
import { TOWNS } from '../utils/constants';

const Alerts = () => {
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({
    town: '',
    message: '',
    severity: 'High'
  });

  const fetchAlerts = async () => {
    try {
      const response = await apiService.getLatestAlerts();
      setAlerts(response.data.latest_alerts || []);
    } catch (error) {
      console.error('Error fetching alerts:', error);
    }
  };

  useEffect(() => {
    fetchAlerts();
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!formData.town || !formData.message) return;

    setLoading(true);
    try {
      await apiService.sendManualAlert(formData);
      setFormData({ town: '', message: '', severity: 'High' });
      fetchAlerts(); // Refresh alerts
      alert('Alert sent successfully!');
    } catch (error) {
      console.error('Error sending alert:', error);
      alert('Failed to send alert');
    } finally {
      setLoading(false);
    }
  };

  const triggerDemo = async () => {
    try {
      await apiService.triggerDemoAlert();
      alert('Demo alert triggered!');
      fetchAlerts();
    } catch (error) {
      console.error('Error triggering demo:', error);
      alert('Failed to trigger demo alert');
    }
  };

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold" style={{color: 'var(--text)'}}>Alert Management</h1>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Send Manual Alert */}
        <div style={{backgroundColor: 'var(--surface)', borderColor: 'var(--border)'}} className="rounded-lg p-6 border">
          <h2 className="text-xl font-semibold mb-4 flex items-center" style={{color: 'var(--text)'}}>
            <Send className="mr-2 h-5 w-5" />
            Send Manual Alert
          </h2>
          
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2" style={{color: 'var(--text)'}}>
                Town
              </label>
              <select
                value={formData.town}
                onChange={(e) => setFormData({ ...formData, town: e.target.value })}
                className="w-full px-3 py-2 rounded-lg focus:outline-none"
                style={{backgroundColor: 'var(--background)', border: '1px solid var(--border)', color: 'var(--text)'}}
                required
              >
                <option value="">Select a town</option>
                {TOWNS.map(town => (
                  <option key={town.name} value={town.name}>{town.name}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2" style={{color: 'var(--text)'}}>
                Severity
              </label>
              <select
                value={formData.severity}
                onChange={(e) => setFormData({ ...formData, severity: e.target.value })}
                className="w-full px-3 py-2 rounded-lg focus:outline-none"
                style={{backgroundColor: 'var(--background)', border: '1px solid var(--border)', color: 'var(--text)'}}
              >
                <option value="High">High</option>
                <option value="Moderate">Moderate</option>
                <option value="Low">Low</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2" style={{color: 'var(--text)'}}>
                Message
              </label>
              <textarea
                value={formData.message}
                onChange={(e) => setFormData({ ...formData, message: e.target.value })}
                rows={4}
                className="w-full px-3 py-2 rounded-lg focus:outline-none"
                style={{backgroundColor: 'var(--background)', border: '1px solid var(--border)', color: 'var(--text)'}}
                placeholder="Enter alert message..."
                required
              />
            </div>

            <div className="flex space-x-3">
              <button
                type="submit"
                disabled={loading}
                className="flex-1 text-white py-2 px-4 rounded-lg hover:opacity-90 transition-colors disabled:opacity-50"
                style={{backgroundColor: 'var(--warning)'}}
              >
                {loading ? 'Sending...' : 'Send Alert'}
              </button>
              
              <button
                type="button"
                onClick={triggerDemo}
                className="px-4 py-2 text-white rounded-lg hover:opacity-90 transition-colors"
                style={{backgroundColor: 'var(--secondary)'}}
              >
                Demo Alert
              </button>
            </div>
          </form>
        </div>

        {/* Alert Statistics */}
        <div style={{backgroundColor: 'var(--surface)', borderColor: 'var(--border)'}} className="rounded-lg p-6 border">
          <h2 className="text-xl font-semibold mb-4" style={{color: 'var(--text)'}}>Alert Statistics</h2>
          
          <div className="space-y-4">
            <div className="flex items-center justify-between p-3 rounded-lg" style={{backgroundColor: 'var(--background)', border: '1px solid var(--border)'}}>
              <div className="flex items-center">
                <AlertTriangle className="h-5 w-5 text-red-500 mr-2" />
                <span style={{color: 'var(--text)'}}>High Severity</span>
              </div>
              <span style={{color: 'var(--text)'}} className="font-bold">
                {alerts.filter(a => a.severity === 'High').length}
              </span>
            </div>
            
            <div className="flex items-center justify-between p-3 rounded-lg" style={{backgroundColor: 'var(--background)', border: '1px solid var(--border)'}}>
              <div className="flex items-center">
                <Clock className="h-5 w-5 text-yellow-500 mr-2" />
                <span style={{color: 'var(--text)'}}>Moderate Severity</span>
              </div>
              <span style={{color: 'var(--text)'}} className="font-bold">
                {alerts.filter(a => a.severity === 'Moderate').length}
              </span>
            </div>
            
            <div className="flex items-center justify-between p-3 rounded-lg" style={{backgroundColor: 'var(--background)', border: '1px solid var(--border)'}}>
              <div className="flex items-center">
                <CheckCircle className="h-5 w-5 text-green-500 mr-2" />
                <span style={{color: 'var(--text)'}}>Low Severity</span>
              </div>
              <span style={{color: 'var(--text)'}} className="font-bold">
                {alerts.filter(a => a.severity === 'Low').length}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Recent Alerts */}
      <div style={{backgroundColor: 'var(--surface)', borderColor: 'var(--border)'}} className="rounded-lg p-6 border">
        <h2 className="text-xl font-semibold mb-4" style={{color: 'var(--text)'}}>Recent Alerts</h2>
        
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b" style={{borderColor: 'var(--border)'}}>
                <th className="text-left py-3 px-4" style={{color: 'var(--text-secondary)'}}>Town</th>
                <th className="text-left py-3 px-4" style={{color: 'var(--text-secondary)'}}>Message</th>
                <th className="text-left py-3 px-4" style={{color: 'var(--text-secondary)'}}>Severity</th>
                <th className="text-left py-3 px-4" style={{color: 'var(--text-secondary)'}}>Time</th>
                <th className="text-left py-3 px-4" style={{color: 'var(--text-secondary)'}}>Status</th>
              </tr>
            </thead>
            <tbody>
              {alerts.map((alert, index) => (
                <tr key={index} className="border-b" style={{borderColor: 'var(--border)'}}>
                  <td className="py-3 px-4 font-medium" style={{color: 'var(--text)'}}>{alert.town}</td>
                  <td className="py-3 px-4 max-w-xs truncate" style={{color: 'var(--text-secondary)'}}>
                    {alert.message}
                  </td>
                  <td className="py-3 px-4">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      alert.severity === 'High' ? 'bg-red-500 text-white' :
                      alert.severity === 'Moderate' ? 'bg-yellow-500 text-white' :
                      'bg-green-500 text-white'
                    }`}>
                      {alert.severity}
                    </span>
                  </td>
                  <td className="py-3 px-4" style={{color: 'var(--text-secondary)'}}>
                    {new Date(alert.timestamp?.seconds * 1000).toLocaleString()}
                  </td>
                  <td className="py-3 px-4">
                    <span className="px-2 py-1 rounded text-xs font-medium bg-green-500 text-white">
                      Sent
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          
          {alerts.length === 0 && (
            <div className="text-center py-8" style={{color: 'var(--text-secondary)'}}>
              No alerts found
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Alerts;