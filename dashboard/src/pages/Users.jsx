import React, { useState, useEffect } from 'react';
import { Users as UsersIcon, UserPlus, MapPin, Phone } from 'lucide-react';
import { apiService } from '../services/api';
import { TOWNS } from '../utils/constants';
import StatsCard from '../components/StatsCard';

const Users = () => {
  const [users, setUsers] = useState({});
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({
    phone: '',
    town: '',
    name: ''
  });

  const fetchUsers = async () => {
    try {
      setLoading(true);
      const userPromises = TOWNS.map(town => 
        apiService.getTownUsers(town.name).catch(() => ({ data: { users: [] } }))
      );
      const results = await Promise.all(userPromises);
      
      const usersData = {};
      TOWNS.forEach((town, index) => {
        usersData[town.name] = results[index].data.users || [];
      });
      
      setUsers(usersData);
    } catch (error) {
      console.error('Error fetching users:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!formData.phone || !formData.town) return;

    try {
      setLoading(true);
      await apiService.registerUser({
        phone: formData.phone,
        town: formData.town,
        name: formData.name
      });
      
      setFormData({ phone: '', town: '', name: '' });
      await fetchUsers();
      alert('User registered successfully!');
    } catch (error) {
      console.error('Error registering user:', error);
      alert('Failed to register user');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchUsers();
  }, []);

  const totalUsers = Object.values(users).flat().length;
  const activeTowns = Object.keys(users).filter(town => users[town].length > 0).length;
  const avgUsersPerTown = activeTowns > 0 ? Math.round(totalUsers / activeTowns) : 0;

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold" style={{color: 'var(--text)'}}>User Management</h1>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <StatsCard
          title="Total Users"
          value={totalUsers}
          icon={UsersIcon}
          color="default"
          subtitle="Registered for alerts"
        />
        <StatsCard
          title="Active Towns"
          value={activeTowns}
          icon={MapPin}
          color="info"
          subtitle="With registered users"
        />
        <StatsCard
          title="Avg Users/Town"
          value={avgUsersPerTown}
          icon={UserPlus}
          color="default"
          subtitle="Distribution"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div style={{backgroundColor: 'var(--surface)', borderColor: 'var(--border)'}} className="rounded-lg p-6 border">
          <h2 className="text-xl font-semibold mb-4 flex items-center" style={{color: 'var(--text)'}}>
            <UserPlus className="mr-2 h-5 w-5" />
            Register New User
          </h2>
          
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2" style={{color: 'var(--text)'}}>
                Phone Number
              </label>
              <input
                type="tel"
                value={formData.phone}
                onChange={(e) => setFormData({ ...formData, phone: e.target.value })}
                className="w-full px-3 py-2 rounded-lg focus:outline-none"
                style={{backgroundColor: 'var(--background)', border: '1px solid var(--border)', color: 'var(--text)'}}
                placeholder="+250792403010"
                required
              />
            </div>

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
                Name (Optional)
              </label>
              <input
                type="text"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                className="w-full px-3 py-2 rounded-lg focus:outline-none"
                style={{backgroundColor: 'var(--background)', border: '1px solid var(--border)', color: 'var(--text)'}}
                placeholder="User's name"
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full text-white py-2 px-4 rounded-lg hover:opacity-90 transition-colors disabled:opacity-50"
              style={{backgroundColor: 'var(--secondary)'}}
            >
              {loading ? 'Registering...' : 'Register User'}
            </button>
          </form>
        </div>

        <div style={{backgroundColor: 'var(--surface)', borderColor: 'var(--border)'}} className="rounded-lg p-6 border">
          <h2 className="text-xl font-semibold mb-4" style={{color: 'var(--text)'}}>Users by Town</h2>
          
          <div className="space-y-3">
            {TOWNS.map(town => {
              const townUsers = users[town.name] || [];
              return (
                <div key={town.name} className="flex items-center justify-between p-3 rounded-lg" style={{backgroundColor: 'var(--background)', border: '1px solid var(--border)'}}>
                  <div className="flex items-center">
                    <MapPin className="h-4 w-4 mr-2" style={{color: 'var(--secondary)'}} />
                    <span className="font-medium" style={{color: 'var(--text)'}}>{town.name}</span>
                  </div>
                  <span className="font-bold" style={{color: 'var(--text)'}}>{townUsers.length}</span>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      <div style={{backgroundColor: 'var(--surface)', borderColor: 'var(--border)'}} className="rounded-lg p-6 border">
        <h2 className="text-xl font-semibold mb-4" style={{color: 'var(--text)'}}>All Registered Users</h2>
        
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b" style={{borderColor: 'var(--border)'}}>
                <th className="text-left py-3 px-4" style={{color: 'var(--text-secondary)'}}>Name</th>
                <th className="text-left py-3 px-4" style={{color: 'var(--text-secondary)'}}>Phone</th>
                <th className="text-left py-3 px-4" style={{color: 'var(--text-secondary)'}}>Town</th>
                <th className="text-left py-3 px-4" style={{color: 'var(--text-secondary)'}}>Status</th>
                <th className="text-left py-3 px-4" style={{color: 'var(--text-secondary)'}}>Registered</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(users).flatMap(([town, townUsers]) =>
                townUsers.map((user, index) => (
                  <tr key={`${town}-${index}`} className="border-b" style={{borderColor: 'var(--border)'}}>
                    <td className="py-3 px-4" style={{color: 'var(--text)'}}>{user.name || 'N/A'}</td>
                    <td className="py-3 px-4 flex items-center" style={{color: 'var(--text)'}}>
                      <Phone className="h-4 w-4 mr-2" style={{color: 'var(--text-secondary)'}} />
                      {user.phone_number}
                    </td>
                    <td className="py-3 px-4" style={{color: 'var(--text)'}}>{town}</td>
                    <td className="py-3 px-4">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${
                        user.active ? 'bg-green-500 text-white' : 'bg-gray-500 text-white'
                      }`}>
                        {user.active ? 'Active' : 'Inactive'}
                      </span>
                    </td>
                    <td className="py-3 px-4" style={{color: 'var(--text-secondary)'}}>
                      {user.created_at ? new Date(user.created_at.seconds * 1000).toLocaleDateString() : 'N/A'}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
          
          {totalUsers === 0 && (
            <div className="text-center py-8" style={{color: 'var(--text-secondary)'}}>
              No users registered yet
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Users;