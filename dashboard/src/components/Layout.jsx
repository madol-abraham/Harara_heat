import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Home, BarChart3, Bell, Users, Settings, Download, LogOut } from 'lucide-react';

const Layout = ({ children, onLogout }) => {
  const location = useLocation();

  const navigation = [
    { name: 'Dashboard', href: '/', icon: Home },
    { name: 'Predictions', href: '/predictions', icon: BarChart3 },
    { name: 'Alerts', href: '/alerts', icon: Bell },
    { name: 'Users', href: '/users', icon: Users },
    { name: 'Export', href: '/export', icon: Download },
    { name: 'Settings', href: '/settings', icon: Settings },
  ];

  return (
    <div className="min-h-screen" style={{backgroundColor: 'var(--background)'}}>
      <div className="flex">
        {/* Sidebar */}
        <div className="w-64 min-h-screen" style={{backgroundColor: 'var(--primary)'}}>
          <div className="p-6">
            <h1 className="text-2xl font-bold text-white">Harara Dashboard</h1>
            <p className="text-white opacity-75 text-sm mt-1">Heatwave Monitoring</p>
          </div>
          
          <nav className="mt-6">
            {navigation.map((item) => {
              const Icon = item.icon;
              const isActive = location.pathname === item.href;
              
              return (
                <Link
                  key={item.name}
                  to={item.href}
                  className={`flex items-center px-6 py-3 text-sm font-medium transition-colors ${
                    isActive
                      ? 'text-white border-r-2'
                      : 'text-white opacity-75 hover:opacity-100'
                  }`}
                  style={isActive ? {backgroundColor: 'var(--primary-light)', borderColor: 'var(--accent)'} : {}}
                >
                  <Icon className="mr-3 h-5 w-5" />
                  {item.name}
                </Link>
              );
            })}
            
            {/* Logout button */}
            <button
              onClick={onLogout}
              className="w-full flex items-center px-6 py-3 text-sm font-medium text-white opacity-75 hover:opacity-100 transition-colors mt-4"
            >
              <LogOut className="mr-3 h-5 w-5" />
              Logout
            </button>
          </nav>
        </div>

        {/* Main content */}
        <div className="flex-1">
          <header className="shadow-lg border-b" style={{background: 'linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%)', borderColor: 'var(--primary)'}}>
            <div className="px-6 py-6">
              <div className="flex items-center justify-between">
                <div>
                  <h1 className="text-2xl font-bold text-white mb-1">🌡️ Harara Dashboard</h1>
                  <p className="text-white opacity-90 text-sm font-medium">Real-time Heatwave Monitoring & Early Warning System</p>
                </div>
                <div className="text-right">
                  <p className="text-white opacity-75 text-xs">Harara</p>
                  <p className="text-white text-sm font-semibold">Climate Resilience Initiative</p>
                </div>
              </div>
            </div>
          </header>
          
          <main className="p-6">
            {children}
          </main>
        </div>
      </div>
    </div>
  );
};

export default Layout;