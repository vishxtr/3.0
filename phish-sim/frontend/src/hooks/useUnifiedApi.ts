// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Phish-Sim Project

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useState, useEffect, useCallback, useRef } from 'react';
import { apiService } from '../services/api';
import { webSocketService } from '../services/websocket';
import { 
  HealthResponse, 
  ModelInfoResponse, 
  AnalysisRequest, 
  AnalysisResponse, 
  SystemStatus, 
  DashboardStats,
  BatchAnalysisRequest
} from '../types';

// --- API Hooks ---

export const useHealth = () => {
  return useQuery<HealthResponse, Error>({
    queryKey: ['health'],
    queryFn: () => apiService.getHealth(),
    refetchInterval: 5000, // Refetch every 5 seconds
  });
};

export const useModelInfo = () => {
  return useQuery<ModelInfoResponse, Error>({
    queryKey: ['modelInfo'],
    queryFn: () => apiService.getModelInfo(),
    staleTime: Infinity, // Model info doesn't change often
  });
};

export const useSystemStatus = () => {
  return useQuery<SystemStatus, Error>({
    queryKey: ['systemStatus'],
    queryFn: () => apiService.getSystemStatus(),
    refetchInterval: 10000, // Refetch every 10 seconds
  });
};

export const useDashboardStats = () => {
  return useQuery<DashboardStats, Error>({
    queryKey: ['dashboardStats'],
    queryFn: () => apiService.getDashboardStats(),
    refetchInterval: 5000, // Refetch every 5 seconds
  });
};

export const useAnalysis = () => {
  return useMutation<AnalysisResponse, Error, AnalysisRequest>({
    mutationFn: (request) => apiService.analyzeContent(request),
  });
};

export const useBatchAnalysis = () => {
  return useMutation<AnalysisResponse[], Error, BatchAnalysisRequest>({
    mutationFn: (request) => apiService.analyzeBatch(request),
  });
};

// --- Dashboard & System Status Hooks ---

export const useDashboardStatsLocal = () => {
  const [stats, setStats] = useState<DashboardStats>({
    total_scans: 0,
    threats_detected: 0,
    avg_response_time_ms: 0,
    cache_hit_rate: 0,
    cache_hits: 0,
    cache_misses: 0,
  });

  const responseTimesRef = useRef<number[]>([]);

  const incrementScans = useCallback(() => {
    setStats(prev => ({ ...prev, total_scans: prev.total_scans + 1 }));
  }, []);

  const incrementThreats = useCallback(() => {
    setStats(prev => ({ ...prev, threats_detected: prev.threats_detected + 1 }));
  }, []);

  const updateResponseTime = useCallback((newTime: number) => {
    responseTimesRef.current.push(newTime);
    const total = responseTimesRef.current.reduce((sum, time) => sum + time, 0);
    setStats(prev => ({
      ...prev,
      avg_response_time_ms: total / responseTimesRef.current.length,
    }));
  }, []);

  const incrementCacheHits = useCallback(() => {
    setStats(prev => {
      const newHits = prev.cache_hits + 1;
      const newTotal = newHits + prev.cache_misses;
      const newHitRate = newTotal > 0 ? (newHits / newTotal) * 100 : 0;
      return { ...prev, cache_hits: newHits, cache_hit_rate: newHitRate };
    });
  }, []);

  const incrementCacheMisses = useCallback(() => {
    setStats(prev => {
      const newMisses = prev.cache_misses + 1;
      const newTotal = prev.cache_hits + newMisses;
      const newHitRate = newTotal > 0 ? (prev.cache_hits / newTotal) * 100 : 0;
      return { ...prev, cache_misses: newMisses, cache_hit_rate: newHitRate };
    });
  }, []);

  return { stats, incrementScans, incrementThreats, updateResponseTime, incrementCacheHits, incrementCacheMisses };
};

export const useSystemStatusLocal = () => {
  const [status, setStatus] = useState<SystemStatus>({
    backend_api: 'down',
    ml_pipeline: 'down',
    database: 'down',
    redis: 'down',
    websocket: 'down',
  });

  const { data: healthData } = useHealth();
  const { data: modelInfoData } = useModelInfo();

  useEffect(() => {
    if (healthData) {
      setStatus(prev => ({
        ...prev,
        backend_api: healthData.status === 'ok' ? 'healthy' : 'down',
        redis: healthData.components?.redis?.status === 'healthy' ? 'healthy' : 'down',
        database: healthData.components?.database?.status === 'healthy' ? 'healthy' : 'down',
      }));
    }
  }, [healthData]);

  useEffect(() => {
    if (modelInfoData) {
      const nlpStatus = modelInfoData.nlp_model.status === 'loaded' ? 'healthy' : 'down';
      const visualStatus = modelInfoData.visual_model.status === 'loaded' ? 'healthy' : 'down';
      setStatus(prev => ({
        ...prev,
        ml_pipeline: (nlpStatus === 'healthy' && visualStatus === 'healthy') ? 'healthy' : 'degraded',
      }));
    }
  }, [modelInfoData]);

  const updateWebSocketStatus = useCallback((isConnected: boolean) => {
    setStatus(prev => ({
      ...prev,
      websocket: isConnected ? 'healthy' : 'down',
    }));
  }, []);

  const isHealthy = Object.values(status).every(s => s === 'healthy');

  return { status, updateWebSocketStatus, isHealthy };
};

export const useAnalysisHistory = () => {
  const [history, setHistory] = useState<AnalysisResponse[]>(() => {
    try {
      const storedHistory = localStorage.getItem('analysisHistory');
      return storedHistory ? JSON.parse(storedHistory) : [];
    } catch (e) {
      console.error('Failed to load analysis history from localStorage', e);
      return [];
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem('analysisHistory', JSON.stringify(history));
    } catch (e) {
      console.error('Failed to save analysis history to localStorage', e);
    }
  }, [history]);

  const addToHistory = useCallback((result: AnalysisResponse) => {
    setHistory(prev => [result, ...prev.slice(0, 99)]); // Keep last 100 results
  }, []);

  const getHistoryByType = useCallback((type: string) => {
    return history.filter(item => item.content_type === type);
  }, [history]);

  const getHistoryByPrediction = useCallback((prediction: string) => {
    return history.filter(item => item.prediction === prediction);
  }, [history]);

  return { history, addToHistory, getHistoryByType, getHistoryByPrediction };
};

// --- WebSocket Hooks ---

export const useWebSocketConnection = (autoConnect: boolean = false, userId: string = 'anonymous') => {
  const [isConnected, setIsConnected] = useState(webSocketService.isConnected());
  const [connectionState, setConnectionState] = useState<'connecting' | 'connected' | 'disconnected'>('disconnected');

  const connect = useCallback(() => {
    setConnectionState('connecting');
    webSocketService.connect(userId);
  }, [userId]);

  const disconnect = useCallback(() => {
    webSocketService.disconnect();
  }, []);

  useEffect(() => {
    const handleConnectionChange = (connected: boolean) => {
      setIsConnected(connected);
      setConnectionState(connected ? 'connected' : 'disconnected');
    };

    webSocketService.addConnectionListener(handleConnectionChange);

    if (autoConnect) {
      connect();
    }

    return () => {
      webSocketService.removeConnectionListener(handleConnectionChange);
      if (!autoConnect) {
        webSocketService.disconnect();
      }
    };
  }, [autoConnect, connect]);

  return { isConnected, connectionState, connect, disconnect };
};

export const useAnalysisUpdates = () => {
  const [updates, setUpdates] = useState<AnalysisResponse[]>([]);
  const isListeningRef = useRef(false);

  const handleMessage = useCallback((message: any) => {
    if (message.type === 'analysis_complete' && message.payload) {
      setUpdates(prev => [message.payload as AnalysisResponse, ...prev]);
    }
  }, []);

  const startListening = useCallback(() => {
    if (!isListeningRef.current) {
      webSocketService.addMessageListener(handleMessage);
      isListeningRef.current = true;
      console.log('Started listening for WebSocket analysis updates.');
    }
  }, [handleMessage]);

  const stopListening = useCallback(() => {
    if (isListeningRef.current) {
      webSocketService.removeMessageListener(handleMessage);
      isListeningRef.current = false;
      console.log('Stopped listening for WebSocket analysis updates.');
    }
  }, [handleMessage]);

  return { updates, startListening, stopListening };
};

export const useAnalysisSubscription = (requestId: string | null) => {
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [loadingState, setLoadingState] = useState<'idle' | 'loading' | 'complete' | 'error'>('idle');
  const currentRequestIdRef = useRef<string | null>(null);

  useEffect(() => {
    if (requestId && requestId !== currentRequestIdRef.current) {
      setResult(null);
      setLoadingState('loading');
      currentRequestIdRef.current = requestId;
    } else if (!requestId) {
      setLoadingState('idle');
      currentRequestIdRef.current = null;
    }
  }, [requestId]);

  useEffect(() => {
    const handleMessage = (message: any) => {
      if (message.type === 'analysis_complete' && message.payload && message.payload.request_id === currentRequestIdRef.current) {
        setResult(message.payload as AnalysisResponse);
        setLoadingState('complete');
        currentRequestIdRef.current = null;
      } else if (message.type === 'analysis_error' && message.payload && message.payload.request_id === currentRequestIdRef.current) {
        console.error('WebSocket analysis error:', message.payload);
        setLoadingState('error');
        currentRequestIdRef.current = null;
      }
    };

    webSocketService.addMessageListener(handleMessage);

    return () => {
      webSocketService.removeMessageListener(handleMessage);
    };
  }, []);

  return { result, loadingState };
};