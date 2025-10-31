import React, { useState } from 'react';
import { Download, FileText, Database, BarChart } from 'lucide-react';
import { apiService } from '../services/api';

const Export = () => {
  const [loading, setLoading] = useState({});

  const downloadFile = (blob, filename) => {
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  };

  const handleExport = async (type) => {
    setLoading({ ...loading, [type]: true });
    
    try {
      let response, filename;
      
      switch (type) {
        case 'predictions':
          response = await apiService.exportPredictions();
          filename = `predictions_${new Date().toISOString().split('T')[0]}.csv`;
          break;
        case 'logs':
          response = await apiService.exportLogs();
          filename = `logs_${new Date().toISOString().split('T')[0]}.csv`;
          break;
        case 'report':
          response = await apiService.exportReport();
          filename = `harara_report_${new Date().toISOString().slice(0, 7)}.pdf`;
          break;
        default:
          return;
      }
      
      downloadFile(response.data, filename);
    } catch (error) {
      console.error(`Error exporting ${type}:`, error);
      alert(`Failed to export ${type}`);
    } finally {
      setLoading({ ...loading, [type]: false });
    }
  };

  const exportOptions = [
    {
      id: 'predictions',
      title: 'Export Predictions',
      description: 'Download all prediction data as CSV file',
      icon: Database,
      color: 'bg-blue-500',
    },
    {
      id: 'logs',
      title: 'Export System Logs',
      description: 'Download system logs and events as CSV file',
      icon: FileText,
      color: 'bg-green-500',
    },
    {
      id: 'report',
      title: 'Generate Monthly Report',
      description: 'Generate comprehensive PDF report with charts',
      icon: BarChart,
      color: 'bg-purple-500',
    },
  ];

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold" style={{color: 'var(--text)'}}>Data Export</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {exportOptions.map((option) => {
          const Icon = option.icon;
          const isLoading = loading[option.id];
          
          return (
            <div key={option.id} style={{backgroundColor: 'var(--surface)', borderColor: 'var(--border)'}} className="rounded-lg p-6 border">
              <div className="flex items-center mb-4">
                <div className={`p-3 rounded-lg ${option.color}`}>
                  <Icon className="h-6 w-6 text-white" />
                </div>
                <h3 className="ml-3 text-lg font-semibold" style={{color: 'var(--text)'}}>{option.title}</h3>
              </div>
              
              <p className="mb-6" style={{color: 'var(--text-secondary)'}}>{option.description}</p>
              
              <button
                onClick={() => handleExport(option.id)}
                disabled={isLoading}
                className="w-full flex items-center justify-center px-4 py-2 text-white rounded-lg hover:opacity-90 transition-colors disabled:opacity-50"
                style={{backgroundColor: 'var(--primary)'}}
              >
                {isLoading ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    Exporting...
                  </>
                ) : (
                  <>
                    <Download className="h-4 w-4 mr-2" />
                    Export
                  </>
                )}
              </button>
            </div>
          );
        })}
      </div>

      {/* Export History */}
      <div style={{backgroundColor: 'var(--surface)', borderColor: 'var(--border)'}} className="rounded-lg p-6 border">
        <h2 className="text-xl font-semibold mb-4" style={{color: 'var(--text)'}}>Export Guidelines</h2>
        
        <div className="space-y-4" style={{color: 'var(--text-secondary)'}}>
          <div className="border-l-4 border-blue-500 pl-4">
            <h3 className="font-semibold" style={{color: 'var(--text)'}}>Predictions Export</h3>
            <p>Contains all prediction records with timestamps, towns, probabilities, and alert status. Useful for historical analysis and model performance evaluation.</p>
          </div>
          
          <div className="border-l-4 border-green-500 pl-4">
            <h3 className="font-semibold" style={{color: 'var(--text)'}}>System Logs Export</h3>
            <p>Includes system events, errors, and operational logs. Essential for troubleshooting and system monitoring.</p>
          </div>
          
          <div className="border-l-4 border-purple-500 pl-4">
            <h3 className="font-semibold" style={{color: 'var(--text)'}}>Monthly Report</h3>
            <p>Comprehensive PDF report with statistics, charts, and insights. Perfect for stakeholder presentations and monthly reviews.</p>
          </div>
        </div>
      </div>

      {/* Quick Stats */}
      <div style={{backgroundColor: 'var(--surface)', borderColor: 'var(--border)'}} className="rounded-lg p-6 border">
        <h2 className="text-xl font-semibold mb-4" style={{color: 'var(--text)'}}>Export Statistics</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="rounded-lg p-4 text-center" style={{backgroundColor: 'var(--background)', border: '1px solid var(--border)'}}>
            <h3 className="text-2xl font-bold" style={{color: 'var(--text)'}}>CSV</h3>
            <p style={{color: 'var(--text-secondary)'}}>Data Format</p>
          </div>
          
          <div className="rounded-lg p-4 text-center" style={{backgroundColor: 'var(--background)', border: '1px solid var(--border)'}}>
            <h3 className="text-2xl font-bold" style={{color: 'var(--text)'}}>PDF</h3>
            <p style={{color: 'var(--text-secondary)'}}>Report Format</p>
          </div>
          
          <div className="rounded-lg p-4 text-center" style={{backgroundColor: 'var(--background)', border: '1px solid var(--border)'}}>
            <h3 className="text-2xl font-bold" style={{color: 'var(--text)'}}>Real-time</h3>
            <p style={{color: 'var(--text-secondary)'}}>Data Freshness</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Export;