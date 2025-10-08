// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Phish-Sim Project

import { useState, useEffect, useCallback, useRef } from 'react';
import { 
  WebSocketMessage, 
  AnalysisResponse,
  LoadingState 
} from '../types';
import { websocketService } from '../services/websocket';

// Main WebSocket hook
export const useWebSocket = () => {
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);

  const connect = useCallback(async () => {
    try {
      setIsConnecting(true);
      setError(null);
      await websocketService.connect();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Connection failed');
    } finally {
      setIsConnecting(false);
    }
  }, []);

  const disconnect = useCallback(() => {
    websocketService.disconnect();
  }, []);

  const send = useCallback((message: WebSocketMessage) => {
    websocketService.send(message);
  }, []);

  useEffect(() => {
    // Set up connection event handlers
    const handleConnectionChange = (connected: boolean) => {
      setIsConnected(connected);
      if (connected) {
        setError(null);
      }
    };

    const handleMessage = (message: WebSocketMessage) => {
      setLastMessage(message);
    };

    const handleError = (error: any) => {
      setError(error.message || 'WebSocket error');
    };

    // Register event handlers
    websocketService.onConnectionChange(handleConnectionChange);
    websocketService.on('message', handleMessage);
    websocketService.on('error', handleError);

    // Initial connection state
    setIsConnected(websocketService.isConnected());

    // Cleanup
    return () => {
      websocketService.offConnectionChange(handleConnectionChange);
      websocketService.off('message', handleMessage);
      websocketService.off('error', handleError);
    };
  }, []);

  return {
    isConnected,
    isConnecting,
    error,
    lastMessage,
    connect,
    disconnect,
    send,
    connectionState: websocketService.getConnectionState(),
    userId: websocketService.getUserId(),
  };
};

// Hook for real-time analysis updates
export const useAnalysisUpdates = () => {
  const [updates, setUpdates] = useState<AnalysisResponse[]>([]);
  const [isListening, setIsListening] = useState(false);

  const startListening = useCallback(() => {
    setIsListening(true);
    
    const handleAnalysisUpdate = (message: WebSocketMessage) => {
      if (message.type === 'analysis_update' && message.result) {
        setUpdates(prev => [message.result!, ...prev.slice(0, 49)]); // Keep last 50 updates
      }
    };

    websocketService.on('analysis_update', handleAnalysisUpdate);

    return () => {
      websocketService.off('analysis_update', handleAnalysisUpdate);
    };
  }, []);

  const stopListening = useCallback(() => {
    setIsListening(false);
  }, []);

  const clearUpdates = useCallback(() => {
    setUpdates([]);
  }, []);

  useEffect(() => {
    if (isListening) {
      return startListening();
    }
  }, [isListening, startListening]);

  return {
    updates,
    isListening,
    startListening,
    stopListening,
    clearUpdates,
  };
};

// Hook for system alerts
export const useSystemAlerts = () => {
  const [alerts, setAlerts] = useState<WebSocketMessage[]>([]);
  const [isListening, setIsListening] = useState(false);

  const startListening = useCallback(() => {
    setIsListening(true);
    
    const handleSystemAlert = (message: WebSocketMessage) => {
      if (message.type === 'system_alert') {
        setAlerts(prev => [message, ...prev.slice(0, 19)]); // Keep last 20 alerts
      }
    };

    websocketService.on('system_alert', handleSystemAlert);

    return () => {
      websocketService.off('system_alert', handleSystemAlert);
    };
  }, []);

  const stopListening = useCallback(() => {
    setIsListening(false);
  }, []);

  const clearAlerts = useCallback(() => {
    setAlerts([]);
  }, []);

  const dismissAlert = useCallback((alertId: string) => {
    setAlerts(prev => prev.filter(alert => alert.request_id !== alertId));
  }, []);

  useEffect(() => {
    if (isListening) {
      return startListening();
    }
  }, [isListening, startListening]);

  return {
    alerts,
    isListening,
    startListening,
    stopListening,
    clearAlerts,
    dismissAlert,
  };
};

// Hook for subscribing to specific analysis requests
export const useAnalysisSubscription = (requestId: string | null) => {
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [isSubscribed, setIsSubscribed] = useState(false);
  const [loadingState, setLoadingState] = useState<LoadingState>('idle');

  const subscribe = useCallback(() => {
    if (!requestId) return;

    setIsSubscribed(true);
    setLoadingState('loading');

    const handleUpdate = (analysisResult: AnalysisResponse) => {
      setResult(analysisResult);
      setLoadingState('success');
    };

    websocketService.subscribeToAnalysis(requestId, handleUpdate);

    return () => {
      websocketService.unsubscribeFromAnalysis(requestId, handleUpdate);
    };
  }, [requestId]);

  const unsubscribe = useCallback(() => {
    if (requestId) {
      websocketService.unsubscribeFromAnalysis(requestId, () => {});
    }
    setIsSubscribed(false);
    setLoadingState('idle');
    setResult(null);
  }, [requestId]);

  useEffect(() => {
    if (requestId && isSubscribed) {
      return subscribe();
    }
  }, [requestId, isSubscribed, subscribe]);

  return {
    result,
    isSubscribed,
    loadingState,
    subscribe,
    unsubscribe,
  };
};

// Hook for WebSocket connection management with auto-reconnect
export const useWebSocketConnection = (autoConnect: boolean = true) => {
  const [connectionState, setConnectionState] = useState<'disconnected' | 'connecting' | 'connected' | 'reconnecting'>('disconnected');
  const [reconnectAttempts, setReconnectAttempts] = useState(0);
  const [lastError, setLastError] = useState<string | null>(null);
  
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const connect = useCallback(async () => {
    if (connectionState === 'connected' || connectionState === 'connecting') {
      return;
    }

    setConnectionState('connecting');
    setLastError(null);

    try {
      await websocketService.connect();
      setConnectionState('connected');
      setReconnectAttempts(0);
    } catch (error) {
      setLastError(error instanceof Error ? error.message : 'Connection failed');
      setConnectionState('disconnected');
      
      // Schedule reconnection
      if (reconnectAttempts < 5) {
        setConnectionState('reconnecting');
        setReconnectAttempts(prev => prev + 1);
        
        const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000);
        reconnectTimeoutRef.current = setTimeout(() => {
          connect();
        }, delay);
      }
    }
  }, [connectionState, reconnectAttempts]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    websocketService.disconnect();
    setConnectionState('disconnected');
    setReconnectAttempts(0);
    setLastError(null);
  }, []);

  useEffect(() => {
    const handleConnectionChange = (connected: boolean) => {
      if (connected) {
        setConnectionState('connected');
        setReconnectAttempts(0);
        setLastError(null);
      } else {
        setConnectionState('disconnected');
      }
    };

    websocketService.onConnectionChange(handleConnectionChange);

    if (autoConnect) {
      connect();
    }

    return () => {
      websocketService.offConnectionChange(handleConnectionChange);
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [autoConnect, connect]);

  return {
    connectionState,
    reconnectAttempts,
    lastError,
    connect,
    disconnect,
    isConnected: connectionState === 'connected',
    isConnecting: connectionState === 'connecting',
    isReconnecting: connectionState === 'reconnecting',
  };
};