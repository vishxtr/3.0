// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Phish-Sim Project

import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { Shield, Activity, AlertTriangle, Settings, Play, Wifi, WifiOff } from 'lucide-react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import Dashboard from './pages/Dashboard';
import Analysis from './pages/Analysis';
import Simulator from './pages/Simulator';
import SettingsPage from './pages/Settings';
import { useWebSocketConnection } from './hooks/useWebSocket';
import { clsx } from 'clsx';

// Create a query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 3,
      retryDelay: 1000,
      staleTime: 5 * 60 * 1000, // 5 minutes
    },
  },
});

// Navigation component
const Navigation: React.FC = () => {
  const location = useLocation();
  const { isConnected: wsConnected } = useWebSocketConnection();

  const navItems = [
    { path: '/', label: 'Dashboard', icon: Activity },
    { path: '/analysis', label: 'Analysis', icon: AlertTriangle },
    { path: '/simulator', label: 'Simulator', icon: Play },
    { path: '/settings', label: 'Settings', icon: Settings },
  ];

  return (
    <nav className="bg-gray-800 border-b border-gray-700">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <Shield className="h-8 w-8 text-primary-500 mr-3" />
            <h1 className="text-xl font-bold text-white">Phish-Sim</h1>
            <span className="ml-3 text-sm text-gray-400">T006 - Frontend Integration</span>
          </div>
          
          <div className="flex items-center space-x-4">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = location.pathname === item.path;
              
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={clsx(
                    'px-3 py-2 rounded-md text-sm font-medium flex items-center transition-colors',
                    isActive
                      ? 'text-white bg-gray-700'
                      : 'text-gray-300 hover:text-white hover:bg-gray-700'
                  )}
                >
                  <Icon className="h-4 w-4 mr-2" />
                  {item.label}
                </Link>
              );
            })}
            
            {/* WebSocket Status Indicator */}
            <div className="flex items-center ml-4 pl-4 border-l border-gray-700">
              {wsConnected ? (
                <div className="flex items-center text-green-400 text-sm">
                  <Wifi className="h-4 w-4 mr-1" />
                  <span className="hidden sm:inline">Real-time</span>
                </div>
              ) : (
                <div className="flex items-center text-red-400 text-sm">
                  <WifiOff className="h-4 w-4 mr-1" />
                  <span className="hidden sm:inline">Offline</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
};

// Main App component
function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <div className="min-h-screen bg-gray-900">
          <Navigation />
          
          {/* Main Content */}
          <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/analysis" element={<Analysis />} />
              <Route path="/simulator" element={<Simulator />} />
              <Route path="/settings" element={<SettingsPage />} />
            </Routes>
          </main>
        </div>
      </Router>
    </QueryClientProvider>
  );
}

export default App;