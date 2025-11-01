import React, { useState } from 'react';
import { Lock, User, AlertCircle } from 'lucide-react';
import { apiService } from '../services/api';

const Login = ({ onLogin }) => {
  const [credentials, setCredentials] = useState({ username: '', password: '' });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      const response = await apiService.login(credentials);
      const { access_token, expires_in } = response.data;
      
      // Store token and expiry
      localStorage.setItem('harara_token', access_token);
      localStorage.setItem('harara_token_expiry', Date.now() + (expires_in * 1000));
      
      onLogin(access_token);
    } catch (error) {
      setError(error.response?.data?.detail || 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center" style={{backgroundColor: 'white'}}>
      <div className="max-w-md w-full space-y-8 p-8 rounded-lg shadow-2xl" style={{backgroundColor: 'var(--background)', border: '1px solid var(--border)'}}>
        <div className="text-center">
          <div className="mb-4">
            <h1 className="text-4xl font-bold" style={{color: 'var(--primary)'}}>🌡️ Harara</h1>
            <h2 className="text-xl font-semibold" style={{color: 'var(--text)'}}>Admin Dashboard</h2>
          </div>
          <p className="text-sm" style={{color: 'var(--text-secondary)'}}>Heatwave Monitoring & Early Warning System</p>
          <p className="text-xs mt-1" style={{color: 'var(--text-secondary)'}}>Republic of South Sudan</p>
        </div>
        
        <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2" style={{color: 'var(--text)'}}>
                Username
              </label>
              <div className="relative">
                <User className="absolute left-3 top-3 h-5 w-5" style={{color: 'var(--text-secondary)'}} />
                <input
                  type="text"
                  required
                  value={credentials.username}
                  onChange={(e) => setCredentials({...credentials, username: e.target.value})}
                  className="w-full pl-10 pr-3 py-2 rounded-lg focus:outline-none focus:ring-2 focus:border-transparent"
                  style={{backgroundColor: 'var(--background)', border: '1px solid var(--border)', color: 'var(--text)', focusRingColor: 'var(--secondary)'}}
                  placeholder="Enter username"
                />
              </div>
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-2" style={{color: 'var(--text)'}}>
                Password
              </label>
              <div className="relative">
                <Lock className="absolute left-3 top-3 h-5 w-5" style={{color: 'var(--text-secondary)'}} />
                <input
                  type="password"
                  required
                  value={credentials.password}
                  onChange={(e) => setCredentials({...credentials, password: e.target.value})}
                  className="w-full pl-10 pr-3 py-2 rounded-lg focus:outline-none focus:ring-2 focus:border-transparent"
                  style={{backgroundColor: 'var(--background)', border: '1px solid var(--border)', color: 'var(--text)', focusRingColor: 'var(--secondary)'}}
                  placeholder="Enter password"
                />
              </div>
            </div>
          </div>

          {error && (
            <div className="flex items-center space-x-2 p-3 rounded-lg" style={{color: 'var(--error)', backgroundColor: 'rgba(214, 40, 40, 0.1)', border: '1px solid var(--error)'}}>
              <AlertCircle className="h-5 w-5" />
              <span className="text-sm">{error}</span>
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full flex justify-center py-2 px-4 border border-transparent rounded-lg shadow-sm text-sm font-medium text-white hover:opacity-90 focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50"
            style={{backgroundColor: 'var(--primary)', focusRingColor: 'var(--secondary)'}}
          >
            {loading ? 'Signing in...' : 'Sign in'}
          </button>
        </form>
        
        <div className="text-center text-sm" style={{color: 'var(--text-secondary)'}}>
          welcome the admin dashboard: manage users, monitor heatwaves, and ensure timely alerts for South Sudan.
        </div>
      </div>
    </div>
  );
};

export default Login;