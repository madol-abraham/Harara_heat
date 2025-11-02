import React, { useState, useEffect } from 'react';
import { BarChart3, TrendingUp, Calendar, RefreshCw } from 'lucide-react';
import { apiService } from '../services/api';
import StatsCard from '../components/StatsCard';

const Predictions = () => {
  const [predictions, setPredictions] = useState([]);
  const [history, setHistory] = useState({});
  const [loading, setLoading] = useState(false);
  const [days, setDays] = useState(7);

  const fetchData = async () => {
    try {
      setLoading(true);
      const [predictionsRes, historyRes] = await Promise.all([
        apiService.getTodayPredictions(),
        apiService.getPredictionHistory(days),
      ]);

      setPredictions(predictionsRes.data.predictions || []);
      setHistory(historyRes.data.records || {});
    } catch (error) {
      console.error('Error fetching predictions:', error);
    } finally {
      setLoading(false);
    }
  };

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

  useEffect(() => {
    fetchData();
  }, [days]);

  const totalPredictions = Object.values(history).flat().length;
  const avgProbability = predictions.length > 0 
    ? (predictions.reduce((sum, p) => sum + p.probability, 0) / predictions.length * 100).toFixed(1)
    : 0;

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold" style={{color: 'var(--text)'}}>Predictions Management</h1>
        <div className="flex space-x-3">
          <select
            value={days}
            onChange={(e) => setDays(Number(e.target.value))}
            className="px-3 py-2 rounded-lg"
            style={{backgroundColor: 'var(--background)', border: '1px solid var(--border)', color: 'var(--text)'}}
          >
            <option value={7}>Last 7 days</option>
            <option value={14}>Last 14 days</option>
            <option value={30}>Last 30 days</option>
          </select>
          <button
            onClick={runPredictions}
            disabled={loading}
            className="flex items-center px-4 py-2 text-white rounded-lg hover:opacity-90 transition-colors disabled:opacity-50"
            style={{backgroundColor: 'var(--secondary)'}}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
            {loading ? 'Running...' : 'Run Predictions'}
          </button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <StatsCard
          title="Total Predictions"
          value={totalPredictions}
          icon={BarChart3}
          color="default"
          subtitle={`Last ${days} days`}
        />
        <StatsCard
          title="Average Probability"
          value={`${avgProbability}%`}
          icon={TrendingUp}
          color="warning"
          subtitle="Current predictions"
        />
        <StatsCard
          title="Active Towns"
          value={predictions.length}
          icon={Calendar}
          color="info"
          subtitle="With predictions today"
        />
      </div>

      {/* Current Predictions */}
      <div style={{backgroundColor: 'var(--surface)', borderColor: 'var(--border)'}} className="rounded-lg p-6 border">
        <h2 className="text-xl font-semibold mb-4" style={{color: 'var(--text)'}}>Current Predictions</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b" style={{borderColor: 'var(--border)'}}>
                <th className="text-left py-3 px-4" style={{color: 'var(--text-secondary)'}}>Town</th>
                <th className="text-left py-3 px-4" style={{color: 'var(--text-secondary)'}}>Probability</th>
                <th className="text-left py-3 px-4" style={{color: 'var(--text-secondary)'}}>Risk Level</th>
                <th className="text-left py-3 px-4" style={{color: 'var(--text-secondary)'}}>Alert Status</th>
                <th className="text-left py-3 px-4" style={{color: 'var(--text-secondary)'}}>Severity</th>
              </tr>
            </thead>
            <tbody>
              {predictions.map((prediction, index) => (
                <tr key={index} className="border-b" style={{borderColor: 'var(--border)'}}>
                  <td className="py-3 px-4 font-medium" style={{color: 'var(--text)'}}>{prediction.town}</td>
                  <td className="py-3 px-4" style={{color: 'var(--text)'}}>{(prediction.probability * 100).toFixed(1)}%</td>
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
                  <td className="py-3 px-4" style={{color: 'var(--text-secondary)'}}>{prediction.severity || 'N/A'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Historical Data */}
      <div style={{backgroundColor: 'var(--surface)', borderColor: 'var(--border)'}} className="rounded-lg p-6 border">
        <h2 className="text-xl font-semibold mb-4" style={{color: 'var(--text)'}}>Historical Predictions ({days} days)</h2>
        <div className="space-y-4">
          {Object.entries(history).map(([date, dayPredictions]) => (
            <div key={date} className="border-l-4 pl-4" style={{borderColor: 'var(--secondary)'}}>
              <h3 className="font-semibold" style={{color: 'var(--text)'}}>{date}</h3>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-2 mt-2">
                {dayPredictions.map((pred, idx) => (
                  <div key={idx} className="rounded p-2 text-center" style={{backgroundColor: 'var(--background)', border: '1px solid var(--border)'}}>
                    <div className="font-medium text-sm" style={{color: 'var(--text)'}}>{pred.town}</div>
                    <div className="text-xs" style={{color: 'var(--text-secondary)'}}>{(pred.probability * 100).toFixed(1)}%</div>
                  </div>
                ))}
              </div>
            </div>
          ))}
          {Object.keys(history).length === 0 && (
            <p className="text-center py-4" style={{color: 'var(--text-secondary)'}}>No historical data available</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default Predictions;