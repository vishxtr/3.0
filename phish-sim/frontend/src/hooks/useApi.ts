// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Phish-Sim Project

import { useState, useEffect, useCallback } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { 
  AnalysisRequest, 
  AnalysisResponse, 
  HealthResponse, 
  ModelInfo,
  LoadingState,
  ApiError 
} from '../types';
import { apiService } from '../services/api';

// Health check hook
export const useHealth = () => {
  return useQuery<HealthResponse, ApiError>({
    queryKey: ['health'],
    queryFn: () => apiService.getHealth(),
    refetchInterval: 30000, // Refetch every 30 seconds
    retry: 3,
    retryDelay: 1000,
  });
};

// Model info hook
export const useModelInfo = () => {
  return useQuery<ModelInfo, ApiError>({
    queryKey: ['modelInfo'],
    queryFn: () => apiService.getModelInfo(),
    refetchInterval: 60000, // Refetch every minute
    retry: 3,
    retryDelay: 1000,
  });
};

// Analysis hook
export const useAnalysis = () => {
  const queryClient = useQueryClient();
  
  const mutation = useMutation<AnalysisResponse, ApiError, AnalysisRequest>({
    mutationFn: (request) => apiService.analyzeContent(request),
    onSuccess: (data) => {
      // Invalidate and refetch analysis history
      queryClient.invalidateQueries({ queryKey: ['analysisHistory'] });
      
      // Update dashboard stats
      queryClient.invalidateQueries({ queryKey: ['dashboardStats'] });
    },
  });

  const analyze = useCallback((request: AnalysisRequest) => {
    return mutation.mutateAsync(request);
  }, [mutation]);

  return {
    analyze,
    isLoading: mutation.isPending,
    error: mutation.error,
    data: mutation.data,
    isSuccess: mutation.isSuccess,
    isError: mutation.isError,
    reset: mutation.reset,
  };
};

// Batch analysis hook
export const useBatchAnalysis = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const [results, setResults] = useState<AnalysisResponse[]>([]);

  const analyzeBatch = useCallback(async (requests: AnalysisRequest[]) => {
    setIsLoading(true);
    setError(null);
    setResults([]);

    try {
      const batchResults = await apiService.analyzeBatch(requests);
      setResults(batchResults);
      return batchResults;
    } catch (err) {
      const apiError = err as ApiError;
      setError(apiError);
      throw apiError;
    } finally {
      setIsLoading(false);
    }
  }, []);

  return {
    analyzeBatch,
    isLoading,
    error,
    results,
  };
};

// Connection status hook
export const useConnectionStatus = () => {
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  const checkConnection = useCallback(async () => {
    setIsConnecting(true);
    setError(null);

    try {
      const connected = await apiService.testConnection();
      setIsConnected(connected);
      if (!connected) {
        setError({
          message: 'Unable to connect to API',
          code: 'CONNECTION_FAILED',
          timestamp: new Date(),
        });
      }
    } catch (err) {
      const apiError = err as ApiError;
      setError(apiError);
      setIsConnected(false);
    } finally {
      setIsConnecting(false);
    }
  }, []);

  useEffect(() => {
    checkConnection();
    
    // Check connection every 30 seconds
    const interval = setInterval(checkConnection, 30000);
    
    return () => clearInterval(interval);
  }, [checkConnection]);

  return {
    isConnected,
    isConnecting,
    error,
    checkConnection,
  };
};

// Analysis history hook
export const useAnalysisHistory = () => {
  const [history, setHistory] = useState<AnalysisResponse[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const addToHistory = useCallback((result: AnalysisResponse) => {
    setHistory(prev => [result, ...prev.slice(0, 99)]); // Keep last 100 items
  }, []);

  const clearHistory = useCallback(() => {
    setHistory([]);
  }, []);

  const getHistoryByType = useCallback((type: string) => {
    return history.filter(item => item.content_type === type);
  }, [history]);

  const getHistoryByPrediction = useCallback((prediction: string) => {
    return history.filter(item => item.prediction === prediction);
  }, [history]);

  return {
    history,
    isLoading,
    addToHistory,
    clearHistory,
    getHistoryByType,
    getHistoryByPrediction,
  };
};

// Dashboard stats hook
export const useDashboardStats = () => {
  const [stats, setStats] = useState({
    total_scans: 0,
    threats_detected: 0,
    avg_response_time_ms: 0,
    accuracy_percent: 0,
    cache_hit_rate: 0,
    requests_per_second: 0,
  });

  const updateStats = useCallback((newStats: Partial<typeof stats>) => {
    setStats(prev => ({ ...prev, ...newStats }));
  }, []);

  const incrementScans = useCallback(() => {
    setStats(prev => ({ ...prev, total_scans: prev.total_scans + 1 }));
  }, []);

  const incrementThreats = useCallback(() => {
    setStats(prev => ({ ...prev, threats_detected: prev.threats_detected + 1 }));
  }, []);

  const updateResponseTime = useCallback((time: number) => {
    setStats(prev => ({
      ...prev,
      avg_response_time_ms: (prev.avg_response_time_ms + time) / 2,
    }));
  }, []);

  return {
    stats,
    updateStats,
    incrementScans,
    incrementThreats,
    updateResponseTime,
  };
};

// System status hook
export const useSystemStatus = () => {
  const [status, setStatus] = useState({
    backend_api: 'down' as const,
    ml_pipeline: 'initializing' as const,
    database: 'disconnected' as const,
    redis: 'disconnected' as const,
    websocket: 'disconnected' as const,
  });

  const healthQuery = useHealth();
  const modelQuery = useModelInfo();

  useEffect(() => {
    if (healthQuery.data) {
      setStatus(prev => ({
        ...prev,
        backend_api: 'healthy',
        database: 'connected',
        redis: 'connected',
      }));
    } else if (healthQuery.error) {
      setStatus(prev => ({
        ...prev,
        backend_api: 'down',
        database: 'disconnected',
        redis: 'disconnected',
      }));
    }
  }, [healthQuery.data, healthQuery.error]);

  useEffect(() => {
    if (modelQuery.data) {
      setStatus(prev => ({
        ...prev,
        ml_pipeline: 'healthy',
      }));
    } else if (modelQuery.error) {
      setStatus(prev => ({
        ...prev,
        ml_pipeline: 'down',
      }));
    }
  }, [modelQuery.data, modelQuery.error]);

  const updateWebSocketStatus = useCallback((connected: boolean) => {
    setStatus(prev => ({
      ...prev,
      websocket: connected ? 'connected' : 'disconnected',
    }));
  }, []);

  return {
    status,
    updateWebSocketStatus,
    isHealthy: status.backend_api === 'healthy' && status.ml_pipeline === 'healthy',
  };
};

// Custom hook for managing loading states
export const useLoadingState = (initialState: LoadingState = 'idle') => {
  const [state, setState] = useState<LoadingState>(initialState);

  const setLoading = useCallback(() => setState('loading'), []);
  const setSuccess = useCallback(() => setState('success'), []);
  const setError = useCallback(() => setState('error'), []);
  const setIdle = useCallback(() => setState('idle'), []);
  const reset = useCallback(() => setState(initialState), [initialState]);

  return {
    state,
    isLoading: state === 'loading',
    isSuccess: state === 'success',
    isError: state === 'error',
    isIdle: state === 'idle',
    setLoading,
    setSuccess,
    setError,
    setIdle,
    reset,
  };
};