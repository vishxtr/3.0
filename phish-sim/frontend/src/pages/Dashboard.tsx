// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Phish-Sim Project

import React, { useEffect } from 'react';
import { Shield, Activity, AlertTriangle, Clock, Wifi, WifiOff, RefreshCw } from 'lucide-react';
import { useHealth, useModelInfo, useDashboardStats, useSystemStatus } from '../hooks/useApi';
import { useWebSocketConnection, useAnalysisUpdates } from '../hooks/useWebSocket';
import Card from '../components/ui/Card';
import StatusBadge from '../components/ui/StatusBadge';
import LoadingSpinner from '../components/ui/LoadingSpinner';
import Button from '../components/ui/Button';

const Dashboard: React.FC = () => {
  const { data: health, isLoading: healthLoading, error: healthError } = useHealth();
  const { data: modelInfo, isLoading: modelLoading } = useModelInfo();
  const { stats, incrementScans, incrementThreats, updateResponseTime } = useDashboardStats();
  const { status, updateWebSocketStatus, isHealthy } = useSystemStatus();
  
  const { 
    connectionState, 
    isConnected: wsConnected, 
    connect: connectWebSocket, 
    disconnect: disconnectWebSocket 
  } = useWebSocketConnection(true);
  
  const { updates, startListening, stopListening } = useAnalysisUpdates();

  // Update WebSocket status in system status
  useEffect(() => {
    updateWebSocketStatus(wsConnected);
  }, [wsConnected, updateWebSocketStatus]);

  // Start listening for analysis updates
  useEffect(() => {
    if (wsConnected) {
      startListening();
    } else {
      stopListening();
    }
  }, [wsConnected, startListening, stopListening]);

  // Update stats when new analysis updates arrive
  useEffect(() => {
    updates.forEach(update => {
      incrementScans();
      if (update.prediction === 'phish') {
        incrementThreats();
      }
      updateResponseTime(update.processing_time_ms);
    });
  }, [updates, incrementScans, incrementThreats, updateResponseTime]);

  const handleRefresh = () => {
    window.location.reload();
  };

  const handleWebSocketToggle = () => {
    if (wsConnected) {
      disconnectWebSocket();
    } else {
      connectWebSocket();
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-primary-600 to-primary-800 rounded-lg p-6 text-white">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold mb-2">Phish-Sim Dashboard</h1>
            <p className="text-primary-100">
              Real-Time AI/ML-Based Phishing Detection & Prevention — Web Simulation
            </p>
            <div className="mt-4 flex items-center space-x-4 text-sm">
              <div className="flex items-center">
                <Activity className="h-4 w-4 mr-2" />
                Status: {isHealthy ? 'Running' : 'Degraded'}
              </div>
              <div className="flex items-center">
                <Clock className="h-4 w-4 mr-2" />
                Task: T006 - Frontend Integration
              </div>
              <div className="flex items-center">
                {wsConnected ? (
                  <Wifi className="h-4 w-4 mr-2 text-green-400" />
                ) : (
                  <WifiOff className="h-4 w-4 mr-2 text-red-400" />
                )}
                WebSocket: {wsConnected ? 'Connected' : 'Disconnected'}
              </div>
            </div>
          </div>
          <div className="flex space-x-2">
            <Button
              variant="secondary"
              size="sm"
              onClick={handleWebSocketToggle}
              className="bg-white/10 hover:bg-white/20 text-white border-white/20"
            >
              {wsConnected ? 'Disconnect' : 'Connect'} WS
            </Button>
            <Button
              variant="secondary"
              size="sm"
              onClick={handleRefresh}
              className="bg-white/10 hover:bg-white/20 text-white border-white/20"
            >
              <RefreshCw className="h-4 w-4 mr-2" />
              Refresh
            </Button>
          </div>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card
          icon={<Shield className="h-8 w-8 text-green-500" />}
          title="Total Scans"
          subtitle={stats.total_scans.toString()}
        />

        <Card
          icon={<AlertTriangle className="h-8 w-8 text-red-500" />}
          title="Threats Detected"
          subtitle={stats.threats_detected.toString()}
        />

        <Card
          icon={<Activity className="h-8 w-8 text-blue-500" />}
          title="Avg Response Time"
          subtitle={`${stats.avg_response_time_ms.toFixed(1)}ms`}
        />

        <Card
          icon={<Shield className="h-8 w-8 text-purple-500" />}
          title="Cache Hit Rate"
          subtitle={`${stats.cache_hit_rate.toFixed(1)}%`}
        />
      </div>

      {/* Recent Activity */}
      <Card title="Recent Activity">
        {updates.length > 0 ? (
          <div className="space-y-3">
            {updates.slice(0, 5).map((update, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-700 rounded-lg">
                <div className="flex items-center space-x-3">
                  {update.prediction === 'phish' ? (
                    <AlertTriangle className="h-5 w-5 text-red-500" />
                  ) : (
                    <Shield className="h-5 w-5 text-green-500" />
                  )}
                  <div>
                    <p className="text-white text-sm font-medium">
                      {update.content.substring(0, 50)}...
                    </p>
                    <p className="text-gray-400 text-xs">
                      {update.content_type} • {update.prediction} • {update.confidence.toFixed(2)}
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-gray-400 text-xs">
                    {new Date(update.timestamp).toLocaleTimeString()}
                  </p>
                  <p className="text-gray-500 text-xs">
                    {update.processing_time_ms}ms
                  </p>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-gray-400 text-center py-8">
            <Shield className="h-12 w-12 mx-auto mb-4 text-gray-600" />
            <p>No activity yet. Start by analyzing a URL or text content.</p>
          </div>
        )}
      </Card>

      {/* System Status */}
      <Card title="System Status">
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-gray-300">Backend API</span>
            {healthLoading ? (
              <LoadingSpinner size="sm" />
            ) : (
              <StatusBadge status={status.backend_api} />
            )}
          </div>
          
          <div className="flex items-center justify-between">
            <span className="text-gray-300">ML Pipeline</span>
            {modelLoading ? (
              <LoadingSpinner size="sm" />
            ) : (
              <StatusBadge status={status.ml_pipeline} />
            )}
          </div>
          
          <div className="flex items-center justify-between">
            <span className="text-gray-300">Database</span>
            <StatusBadge status={status.database} />
          </div>
          
          <div className="flex items-center justify-between">
            <span className="text-gray-300">Redis Cache</span>
            <StatusBadge status={status.redis} />
          </div>
          
          <div className="flex items-center justify-between">
            <span className="text-gray-300">WebSocket</span>
            <StatusBadge status={status.websocket} />
          </div>
        </div>

        {/* Model Information */}
        {modelInfo && (
          <div className="mt-6 pt-4 border-t border-gray-700">
            <h4 className="text-sm font-medium text-gray-300 mb-3">Model Information</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div>
                <p className="text-gray-400">NLP Model</p>
                <p className="text-white">{modelInfo.nlp_model.name} v{modelInfo.nlp_model.version}</p>
              </div>
              <div>
                <p className="text-gray-400">Visual Model</p>
                <p className="text-white">{modelInfo.visual_model.name} v{modelInfo.visual_model.version}</p>
              </div>
              <div>
                <p className="text-gray-400">Phishing Threshold</p>
                <p className="text-white">{modelInfo.thresholds.phishing}</p>
              </div>
              <div>
                <p className="text-gray-400">Suspicious Threshold</p>
                <p className="text-white">{modelInfo.thresholds.suspicious}</p>
              </div>
            </div>
          </div>
        )}

        {/* Health Information */}
        {health && (
          <div className="mt-4 pt-4 border-t border-gray-700">
            <h4 className="text-sm font-medium text-gray-300 mb-3">Service Health</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div>
                <p className="text-gray-400">Service</p>
                <p className="text-white">{health.service}</p>
              </div>
              <div>
                <p className="text-gray-400">Version</p>
                <p className="text-white">{health.version}</p>
              </div>
              <div>
                <p className="text-gray-400">Uptime</p>
                <p className="text-white">{Math.floor(health.uptime_seconds / 3600)}h {Math.floor((health.uptime_seconds % 3600) / 60)}m</p>
              </div>
              <div>
                <p className="text-gray-400">Status</p>
                <StatusBadge status={health.status as any} />
              </div>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
};

export default Dashboard;