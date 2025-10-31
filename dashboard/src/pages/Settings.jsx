import React, { useState, useEffect } from 'react';
import { Settings as SettingsIcon, Server, Clock, Database, Zap } from 'lucide-react';
import { apiService } from '../services/api';
import StatsCard from '../components/StatsCard';

const Settings = () => {
  const [health, setHealth] = useState(null);
  const [schedulerStatus, setSchedulerStatus] = useState(null);
  const [loading, setLoading] = useState(false);

  const fetchSystemStatus = async () => {
    try {
      setLoading(true);
      const [healthRes, schedulerRes] = await Promise.all([
        apiService.getHealth(),
        apiService.getSchedulerStatus(),
      ]);

      setHealth(healthRes.data);
      setSchedulerStatus(schedulerRes.data);
    } catch (error) {
      console.error('Error fetching system status:', error);
    } finally {
      setLoading(false);
    }
  };

  const runSchedulerNow = async () => {
    try {
      setLoading(true);
      await apiService.runSchedulerNow();
      alert('Scheduler job executed successfully!');
      await fetchSystemStatus();
    } catch (error) {
      console.error('Error running scheduler:', error);
      alert('Failed to run scheduler job');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSystemStatus();
    const interval = setInterval(fetchSystemStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const nextRunTime = schedulerStatus?.jobs?.[0]?.next_run_time || 'Not scheduled';
  const schedulerEnabled = schedulerStatus?.enabled || false;
  const activeJobs = schedulerStatus?.jobs?.length || 0;

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold" style={{color: 'var(--text)'}}>System Settings</h1>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatsCard
          title="API Status"
          value={health?.status === 'ok' ? 'Online' : 'Offline'}
          icon={Server}
          color={health?.status === 'ok' ? 'green' : 'red'}
          subtitle="FastAPI Backend"
        />
        <StatsCard
          title="Earth Engine"
          value={health?.ee_ready ? 'Ready' : 'Not Ready'}
          icon={Database}
          color={health?.ee_ready ? 'green' : 'red'}
          subtitle="Google Earth Engine"
        />
        <StatsCard
          title="Scheduler"
          value={schedulerEnabled ? 'Enabled' : 'Disabled'}
          icon={Clock}
          color={schedulerEnabled ? 'green' : 'red'}
          subtitle={`${activeJobs} active jobs`}
        />
        <StatsCard
          title="Next Run"
          value={nextRunTime !== 'Not scheduled' ? 'Scheduled' : 'None'}
          icon={Zap}
          color={nextRunTime !== 'Not scheduled' ? 'blue' : 'yellow'}
          subtitle="Daily predictions"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div style={{backgroundColor: 'var(--surface)', borderColor: 'var(--border)'}} className="rounded-lg p-6 border">
          <h2 className="text-xl font-semibold mb-4 flex items-center" style={{color: 'var(--text)'}}>
            <SettingsIcon className="mr-2 h-5 w-5" />
            System Information
          </h2>
          
          <div className="space-y-4">
            <div className="flex justify-between items-center p-3 rounded-lg" style={{backgroundColor: 'var(--background)', border: '1px solid var(--border)'}}>
              <span style={{color: 'var(--text-secondary)'}}>API Status</span>
              <span className={`font-medium ${health?.status === 'ok' ? 'text-green-400' : 'text-red-400'}`}>
                {health?.status === 'ok' ? 'Healthy' : 'Unhealthy'}
              </span>
            </div>
            
            <div className="flex justify-between items-center p-3 rounded-lg" style={{backgroundColor: 'var(--background)', border: '1px solid var(--border)'}}>
              <span style={{color: 'var(--text-secondary)'}}>Google Earth Engine</span>
              <span className={`font-medium ${health?.ee_ready ? 'text-green-400' : 'text-red-400'}`}>
                {health?.ee_ready ? 'Connected' : 'Disconnected'}
              </span>
            </div>
            
            <div className="flex justify-between items-center p-3 rounded-lg" style={{backgroundColor: 'var(--background)', border: '1px solid var(--border)'}}>
              <span style={{color: 'var(--text-secondary)'}}>Scheduler Status</span>
              <span className={`font-medium ${schedulerEnabled ? 'text-green-400' : 'text-red-400'}`}>
                {schedulerEnabled ? 'Running' : 'Stopped'}
              </span>
            </div>
            
            <div className="flex justify-between items-center p-3 rounded-lg" style={{backgroundColor: 'var(--background)', border: '1px solid var(--border)'}}>
              <span style={{color: 'var(--text-secondary)'}}>Active Jobs</span>
              <span className="font-medium" style={{color: 'var(--text)'}}>{activeJobs}</span>
            </div>
          </div>
        </div>

        <div style={{backgroundColor: 'var(--surface)', borderColor: 'var(--border)'}} className="rounded-lg p-6 border">
          <h2 className="text-xl font-semibold mb-4 flex items-center" style={{color: 'var(--text)'}}>
            <Clock className="mr-2 h-5 w-5" />
            Scheduler Control
          </h2>
          
          <div className="space-y-4">
            <div className="p-4 rounded-lg" style={{backgroundColor: 'var(--background)', border: '1px solid var(--border)'}}>
              <h3 className="font-medium mb-2" style={{color: 'var(--text)'}}>Next Scheduled Run</h3>
              <p className="text-sm" style={{color: 'var(--text-secondary)'}}>
                {nextRunTime !== 'Not scheduled' ? nextRunTime : 'No jobs scheduled'}
              </p>
            </div>
            
            <div className="p-4 rounded-lg" style={{backgroundColor: 'var(--background)', border: '1px solid var(--border)'}}>
              <h3 className="font-medium mb-2" style={{color: 'var(--text)'}}>Daily Predictions</h3>
              <p className="text-sm mb-3" style={{color: 'var(--text-secondary)'}}>
                Automatically runs predictions every day at 07:00 (Africa/Kigali time)
              </p>
              <button
                onClick={runSchedulerNow}
                disabled={loading}
                className="w-full text-white py-2 px-4 rounded-lg hover:opacity-90 transition-colors disabled:opacity-50"
                style={{backgroundColor: 'var(--secondary)'}}
              >
                {loading ? 'Running...' : 'Run Now'}
              </button>
            </div>
          </div>
        </div>
      </div>

      {schedulerStatus?.jobs && schedulerStatus.jobs.length > 0 && (
        <div style={{backgroundColor: 'var(--surface)', borderColor: 'var(--border)'}} className="rounded-lg p-6 border">
          <h2 className="text-xl font-semibold mb-4" style={{color: 'var(--text)'}}>Scheduled Jobs</h2>
          
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b" style={{borderColor: 'var(--border)'}}>
                  <th className="text-left py-3 px-4" style={{color: 'var(--text-secondary)'}}>Job ID</th>
                  <th className="text-left py-3 px-4" style={{color: 'var(--text-secondary)'}}>Next Run Time</th>
                  <th className="text-left py-3 px-4" style={{color: 'var(--text-secondary)'}}>Trigger</th>
                  <th className="text-left py-3 px-4" style={{color: 'var(--text-secondary)'}}>Status</th>
                </tr>
              </thead>
              <tbody>
                {schedulerStatus.jobs.map((job, index) => (
                  <tr key={index} className="border-b" style={{borderColor: 'var(--border)'}}>
                    <td className="py-3 px-4 font-medium" style={{color: 'var(--text)'}}>{job.id}</td>
                    <td className="py-3 px-4" style={{color: 'var(--text)'}}>{job.next_run_time}</td>
                    <td className="py-3 px-4" style={{color: 'var(--text-secondary)'}}>{job.trigger}</td>
                    <td className="py-3 px-4">
                      <span className="px-2 py-1 rounded text-xs font-medium bg-green-500 text-white">
                        Active
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div style={{backgroundColor: 'var(--surface)', borderColor: 'var(--border)'}} className="rounded-lg p-6 border">
        <h2 className="text-xl font-semibold mb-4" style={{color: 'var(--text)'}}>System Actions</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <button
            onClick={fetchSystemStatus}
            className="flex items-center justify-center px-4 py-3 text-white rounded-lg hover:opacity-90 transition-colors"
            style={{backgroundColor: 'var(--primary)'}}
          >
            <Server className="h-4 w-4 mr-2" />
            Refresh Status
          </button>
          
          <button
            onClick={runSchedulerNow}
            disabled={loading}
            className="flex items-center justify-center px-4 py-3 text-white rounded-lg hover:opacity-90 transition-colors disabled:opacity-50"
            style={{backgroundColor: 'var(--secondary)'}}
          >
            <Zap className="h-4 w-4 mr-2" />
            Run Predictions
          </button>
          
          <button
            onClick={() => window.location.href = '/export'}
            className="flex items-center justify-center px-4 py-3 text-white rounded-lg hover:opacity-90 transition-colors"
            style={{backgroundColor: 'var(--primary)'}}
          >
            <Database className="h-4 w-4 mr-2" />
            Export Data
          </button>
        </div>
      </div>
    </div>
  );
};

export default Settings;