// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Phish-Sim Project

import React, { useState, useEffect } from 'react';
import { Settings as SettingsIcon, Save, RefreshCw, Wifi, WifiOff, Bell, Monitor, Database, Zap } from 'lucide-react';
import { useConnectionStatus } from '../hooks/useApi';
import { useWebSocketConnection } from '../hooks/useWebSocket';
import Card from '../components/ui/Card';
import Button from '../components/ui/Button';
import Input from '../components/ui/Input';
import StatusBadge from '../components/ui/StatusBadge';
import { AppSettings } from '../types';

const SettingsPage: React.FC = () => {
  const [settings, setSettings] = useState<AppSettings>({
    api_endpoint: 'http://localhost:8001',
    websocket_endpoint: 'ws://localhost:8003',
    auto_refresh_interval: 30,
    theme: 'dark',
    notifications: {
      enabled: true,
      sound: true,
      desktop: false,
    },
    analysis: {
      auto_analyze: false,
      cache_enabled: true,
      confidence_threshold: 0.7,
    },
  });

  const [hasChanges, setHasChanges] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [saveStatus, setSaveStatus] = useState<'idle' | 'success' | 'error'>('idle');

  const { isConnected: apiConnected, checkConnection } = useConnectionStatus();
  const { isConnected: wsConnected, connect: connectWebSocket, disconnect: disconnectWebSocket } = useWebSocketConnection(false);

  // Load settings from localStorage on mount
  useEffect(() => {
    const savedSettings = localStorage.getItem('phish-sim-settings');
    if (savedSettings) {
      try {
        const parsed = JSON.parse(savedSettings);
        setSettings(prev => ({ ...prev, ...parsed }));
      } catch (error) {
        console.error('Failed to load settings:', error);
      }
    }
  }, []);

  // Check for changes
  useEffect(() => {
    const savedSettings = localStorage.getItem('phish-sim-settings');
    if (savedSettings) {
      try {
        const parsed = JSON.parse(savedSettings);
        setHasChanges(JSON.stringify(parsed) !== JSON.stringify(settings));
      } catch (error) {
        setHasChanges(true);
      }
    } else {
      setHasChanges(true);
    }
  }, [settings]);

  const handleSave = async () => {
    setIsSaving(true);
    setSaveStatus('idle');

    try {
      // Save to localStorage
      localStorage.setItem('phish-sim-settings', JSON.stringify(settings));
      
      // Update environment variables if needed
      if (typeof window !== 'undefined') {
        // Update API service endpoints
        const apiService = await import('../services/api');
        apiService.apiService.setBaseURL(settings.api_endpoint);
        
        // Update WebSocket service URL
        const wsService = await import('../services/websocket');
        wsService.websocketService.setUrl(settings.websocket_endpoint);
      }

      setSaveStatus('success');
      setHasChanges(false);
      
      // Clear success status after 3 seconds
      setTimeout(() => setSaveStatus('idle'), 3000);
    } catch (error) {
      console.error('Failed to save settings:', error);
      setSaveStatus('error');
    } finally {
      setIsSaving(false);
    }
  };

  const handleReset = () => {
    const defaultSettings: AppSettings = {
      api_endpoint: 'http://localhost:8001',
      websocket_endpoint: 'ws://localhost:8003',
      auto_refresh_interval: 30,
      theme: 'dark',
      notifications: {
        enabled: true,
        sound: true,
        desktop: false,
      },
      analysis: {
        auto_analyze: false,
        cache_enabled: true,
        confidence_threshold: 0.7,
      },
    };
    setSettings(defaultSettings);
  };

  const testConnection = async () => {
    await checkConnection();
  };

  const testWebSocket = () => {
    if (wsConnected) {
      disconnectWebSocket();
    } else {
      connectWebSocket();
    }
  };

  return (
    <div className="space-y-6">
      {/* API Configuration */}
      <Card title="API Configuration" subtitle="Configure backend API and WebSocket endpoints">
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Input
              label="API Endpoint"
              value={settings.api_endpoint}
              onChange={(value) => setSettings(prev => ({ ...prev, api_endpoint: value }))}
              placeholder="http://localhost:8001"
            />
            
            <Input
              label="WebSocket Endpoint"
              value={settings.websocket_endpoint}
              onChange={(value) => setSettings(prev => ({ ...prev, websocket_endpoint: value }))}
              placeholder="ws://localhost:8003"
            />
          </div>

          <div className="flex items-center justify-between pt-4 border-t border-gray-700">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <span className="text-gray-300">API Status:</span>
                <StatusBadge status={apiConnected ? 'connected' : 'disconnected'} />
              </div>
              <div className="flex items-center space-x-2">
                <span className="text-gray-300">WebSocket Status:</span>
                <StatusBadge status={wsConnected ? 'connected' : 'disconnected'} />
              </div>
            </div>

            <div className="flex space-x-2">
              <Button
                variant="secondary"
                size="sm"
                onClick={testConnection}
                className="flex items-center"
              >
                {apiConnected ? <Wifi className="h-4 w-4 mr-2" /> : <WifiOff className="h-4 w-4 mr-2" />}
                Test API
              </Button>
              <Button
                variant="secondary"
                size="sm"
                onClick={testWebSocket}
                className="flex items-center"
              >
                {wsConnected ? <Wifi className="h-4 w-4 mr-2" /> : <WifiOff className="h-4 w-4 mr-2" />}
                Test WS
              </Button>
            </div>
          </div>
        </div>
      </Card>

      {/* Application Settings */}
      <Card title="Application Settings" subtitle="Configure application behavior and preferences">
        <div className="space-y-6">
          <div>
            <h4 className="text-lg font-medium text-white mb-4">General</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Auto Refresh Interval (seconds)
                </label>
                <input
                  type="number"
                  min="5"
                  max="300"
                  value={settings.auto_refresh_interval}
                  onChange={(e) => setSettings(prev => ({ ...prev, auto_refresh_interval: parseInt(e.target.value) }))}
                  className="bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Theme
                </label>
                <select
                  value={settings.theme}
                  onChange={(e) => setSettings(prev => ({ ...prev, theme: e.target.value as 'dark' | 'light' }))}
                  className="bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white w-full"
                >
                  <option value="dark">Dark</option>
                  <option value="light">Light</option>
                </select>
              </div>
            </div>
          </div>

          <div>
            <h4 className="text-lg font-medium text-white mb-4">Notifications</h4>
            <div className="space-y-3">
              <label className="flex items-center space-x-3">
                <input
                  type="checkbox"
                  checked={settings.notifications.enabled}
                  onChange={(e) => setSettings(prev => ({
                    ...prev,
                    notifications: { ...prev.notifications, enabled: e.target.checked }
                  }))}
                  className="rounded border-gray-600 bg-gray-700 text-primary-500"
                />
                <span className="text-gray-300">Enable notifications</span>
              </label>

              <label className="flex items-center space-x-3">
                <input
                  type="checkbox"
                  checked={settings.notifications.sound}
                  onChange={(e) => setSettings(prev => ({
                    ...prev,
                    notifications: { ...prev.notifications, sound: e.target.checked }
                  }))}
                  className="rounded border-gray-600 bg-gray-700 text-primary-500"
                />
                <span className="text-gray-300">Sound notifications</span>
              </label>

              <label className="flex items-center space-x-3">
                <input
                  type="checkbox"
                  checked={settings.notifications.desktop}
                  onChange={(e) => setSettings(prev => ({
                    ...prev,
                    notifications: { ...prev.notifications, desktop: e.target.checked }
                  }))}
                  className="rounded border-gray-600 bg-gray-700 text-primary-500"
                />
                <span className="text-gray-300">Desktop notifications</span>
              </label>
            </div>
          </div>

          <div>
            <h4 className="text-lg font-medium text-white mb-4">Analysis</h4>
            <div className="space-y-4">
              <label className="flex items-center space-x-3">
                <input
                  type="checkbox"
                  checked={settings.analysis.auto_analyze}
                  onChange={(e) => setSettings(prev => ({
                    ...prev,
                    analysis: { ...prev.analysis, auto_analyze: e.target.checked }
                  }))}
                  className="rounded border-gray-600 bg-gray-700 text-primary-500"
                />
                <span className="text-gray-300">Auto-analyze content</span>
              </label>

              <label className="flex items-center space-x-3">
                <input
                  type="checkbox"
                  checked={settings.analysis.cache_enabled}
                  onChange={(e) => setSettings(prev => ({
                    ...prev,
                    analysis: { ...prev.analysis, cache_enabled: e.target.checked }
                  }))}
                  className="rounded border-gray-600 bg-gray-700 text-primary-500"
                />
                <span className="text-gray-300">Enable caching</span>
              </label>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Confidence Threshold
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.1"
                  value={settings.analysis.confidence_threshold}
                  onChange={(e) => setSettings(prev => ({
                    ...prev,
                    analysis: { ...prev.analysis, confidence_threshold: parseFloat(e.target.value) }
                  }))}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-gray-400 mt-1">
                  <span>0.1</span>
                  <span className="text-white font-medium">{settings.analysis.confidence_threshold}</span>
                  <span>1.0</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </Card>

      {/* System Information */}
      <Card title="System Information" subtitle="Current system status and configuration">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="text-lg font-medium text-white mb-4">Connection Status</h4>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-gray-300">API Connection</span>
                <StatusBadge status={apiConnected ? 'connected' : 'disconnected'} />
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-300">WebSocket Connection</span>
                <StatusBadge status={wsConnected ? 'connected' : 'disconnected'} />
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-300">Real-time Updates</span>
                <StatusBadge status={wsConnected ? 'connected' : 'disconnected'} />
              </div>
            </div>
          </div>

          <div>
            <h4 className="text-lg font-medium text-white mb-4">Configuration</h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-400">API Endpoint:</span>
                <span className="text-white font-mono">{settings.api_endpoint}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">WebSocket Endpoint:</span>
                <span className="text-white font-mono">{settings.websocket_endpoint}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Refresh Interval:</span>
                <span className="text-white">{settings.auto_refresh_interval}s</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Theme:</span>
                <span className="text-white capitalize">{settings.theme}</span>
              </div>
            </div>
          </div>
        </div>
      </Card>

      {/* Save Actions */}
      <Card title="Save Configuration">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            {saveStatus === 'success' && (
              <div className="flex items-center text-green-400">
                <Save className="h-4 w-4 mr-2" />
                Settings saved successfully
              </div>
            )}
            {saveStatus === 'error' && (
              <div className="flex items-center text-red-400">
                <SettingsIcon className="h-4 w-4 mr-2" />
                Failed to save settings
              </div>
            )}
            {hasChanges && saveStatus === 'idle' && (
              <div className="flex items-center text-yellow-400">
                <SettingsIcon className="h-4 w-4 mr-2" />
                You have unsaved changes
              </div>
            )}
          </div>

          <div className="flex space-x-2">
            <Button
              variant="secondary"
              onClick={handleReset}
              className="flex items-center"
            >
              <RefreshCw className="h-4 w-4 mr-2" />
              Reset
            </Button>
            <Button
              onClick={handleSave}
              disabled={!hasChanges || isSaving}
              loading={isSaving}
              className="flex items-center"
            >
              <Save className="h-4 w-4 mr-2" />
              Save Settings
            </Button>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default SettingsPage;