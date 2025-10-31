import React from 'react';

const StatsCard = ({ title, value, icon: Icon, color = 'blue', subtitle }) => {
  const getIconStyle = (color) => {
    const colors = {
      red: 'var(--error)',
      green: 'var(--success)', 
      yellow: 'var(--warning)',
      info: 'var(--info)',
      default: 'var(--secondary)'
    };
    return { backgroundColor: colors[color] || colors.default };
  };

  return (
    <div style={{backgroundColor: 'var(--surface)', borderColor: 'var(--border)'}} className="rounded-lg p-6 border shadow-sm">
      <div className="flex items-center">
        <div className="p-3 rounded-lg" style={getIconStyle(color)}>
          <Icon className="h-6 w-6 text-white" />
        </div>
        <div className="ml-4">
          <h3 className="text-sm font-medium" style={{color: 'var(--text-secondary)'}}>{title}</h3>
          <p className="text-2xl font-bold" style={{color: 'var(--text)'}}>{value}</p>
          {subtitle && (
            <p className="text-sm mt-1" style={{color: 'var(--text-secondary)'}}>{subtitle}</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default StatsCard;