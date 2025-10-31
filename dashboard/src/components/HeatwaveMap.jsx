import React from 'react';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import L from 'leaflet';
import { TOWNS, RISK_LEVELS } from '../utils/constants';

// Fix for default markers
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

const createCustomIcon = (color) => {
  return L.divIcon({
    className: 'custom-marker',
    html: `<div style="background-color: ${color}; width: 20px; height: 20px; border-radius: 50%; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.3);"></div>`,
    iconSize: [20, 20],
    iconAnchor: [10, 10],
  });
};

const HeatwaveMap = ({ predictions = [] }) => {
  const getRiskLevel = (probability) => {
    if (probability >= RISK_LEVELS.HIGH.min) return RISK_LEVELS.HIGH;
    if (probability >= RISK_LEVELS.MODERATE.min) return RISK_LEVELS.MODERATE;
    return RISK_LEVELS.LOW;
  };

  const getTownPrediction = (townName) => {
    return predictions.find(p => p.town === townName) || { probability: 0, alert: false };
  };

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <h3 className="text-lg font-semibold text-white mb-4">South Sudan Heatwave Risk Map</h3>
      <MapContainer
        center={[7.5, 30.0]}
        zoom={6}
        className="rounded-lg"
        style={{ height: '400px', width: '100%' }}
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        />
        
        {TOWNS.map((town) => {
          const prediction = getTownPrediction(town.name);
          const riskLevel = getRiskLevel(prediction.probability);
          
          return (
            <Marker
              key={town.name}
              position={[town.lat, town.lng]}
              icon={createCustomIcon(riskLevel.color)}
            >
              <Popup>
                <div className="text-gray-900">
                  <h4 className="font-bold">{town.name}</h4>
                  <p>Risk: {riskLevel.label}</p>
                  <p>Probability: {(prediction.probability * 100).toFixed(1)}%</p>
                  <p>Alert: {prediction.alert ? 'Active' : 'None'}</p>
                </div>
              </Popup>
            </Marker>
          );
        })}
      </MapContainer>
    </div>
  );
};

export default HeatwaveMap;