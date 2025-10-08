// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Phish-Sim Project

import { renderHook, act } from '@testing-library/react';
import { useWebSocket, useAnalysisUpdates, useSystemAlerts } from '../../hooks/useWebSocket';

// Mock the WebSocket service
const mockWebSocketService = {
  connect: jest.fn(),
  disconnect: jest.fn(),
  send: jest.fn(),
  isConnected: jest.fn(),
  getConnectionState: jest.fn(),
  getUserId: jest.fn(),
  setUrl: jest.fn(),
  on: jest.fn(),
  off: jest.fn(),
  onConnectionChange: jest.fn(),
  offConnectionChange: jest.fn(),
  subscribeToAnalysis: jest.fn(),
  unsubscribeFromAnalysis: jest.fn(),
};

jest.mock('../../services/websocket', () => ({
  websocketService: mockWebSocketService,
}));

describe('useWebSocket hooks', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockWebSocketService.isConnected.mockReturnValue(false);
    mockWebSocketService.getConnectionState.mockReturnValue('disconnected');
    mockWebSocketService.getUserId.mockReturnValue('test-user');
  });

  describe('useWebSocket', () => {
    test('initializes with disconnected state', () => {
      const { result } = renderHook(() => useWebSocket());

      expect(result.current.isConnected).toBe(false);
      expect(result.current.connectionState).toBe('disconnected');
      expect(result.current.userId).toBe('test-user');
    });

    test('connects to WebSocket', async () => {
      const { result } = renderHook(() => useWebSocket());

      await act(async () => {
        await result.current.connect();
      });

      expect(mockWebSocketService.connect).toHaveBeenCalled();
    });

    test('disconnects from WebSocket', () => {
      const { result } = renderHook(() => useWebSocket());

      act(() => {
        result.current.disconnect();
      });

      expect(mockWebSocketService.disconnect).toHaveBeenCalled();
    });

    test('sends messages', () => {
      const { result } = renderHook(() => useWebSocket());

      const message = { type: 'ping' as const };

      act(() => {
        result.current.send(message);
      });

      expect(mockWebSocketService.send).toHaveBeenCalledWith(message);
    });
  });

  describe('useAnalysisUpdates', () => {
    test('initializes with empty updates', () => {
      const { result } = renderHook(() => useAnalysisUpdates());

      expect(result.current.updates).toEqual([]);
      expect(result.current.isListening).toBe(false);
    });

    test('starts listening for updates', () => {
      const { result } = renderHook(() => useAnalysisUpdates());

      act(() => {
        result.current.startListening();
      });

      expect(result.current.isListening).toBe(true);
      expect(mockWebSocketService.on).toHaveBeenCalledWith('analysis_update', expect.any(Function));
    });

    test('stops listening for updates', () => {
      const { result } = renderHook(() => useAnalysisUpdates());

      act(() => {
        result.current.startListening();
        result.current.stopListening();
      });

      expect(result.current.isListening).toBe(false);
      expect(mockWebSocketService.off).toHaveBeenCalledWith('analysis_update', expect.any(Function));
    });

    test('clears updates', () => {
      const { result } = renderHook(() => useAnalysisUpdates());

      act(() => {
        result.current.clearUpdates();
      });

      expect(result.current.updates).toEqual([]);
    });
  });

  describe('useSystemAlerts', () => {
    test('initializes with empty alerts', () => {
      const { result } = renderHook(() => useSystemAlerts());

      expect(result.current.alerts).toEqual([]);
      expect(result.current.isListening).toBe(false);
    });

    test('starts listening for alerts', () => {
      const { result } = renderHook(() => useSystemAlerts());

      act(() => {
        result.current.startListening();
      });

      expect(result.current.isListening).toBe(true);
      expect(mockWebSocketService.on).toHaveBeenCalledWith('system_alert', expect.any(Function));
    });

    test('stops listening for alerts', () => {
      const { result } = renderHook(() => useSystemAlerts());

      act(() => {
        result.current.startListening();
        result.current.stopListening();
      });

      expect(result.current.isListening).toBe(false);
      expect(mockWebSocketService.off).toHaveBeenCalledWith('system_alert', expect.any(Function));
    });

    test('clears alerts', () => {
      const { result } = renderHook(() => useSystemAlerts());

      act(() => {
        result.current.clearAlerts();
      });

      expect(result.current.alerts).toEqual([]);
    });

    test('dismisses specific alert', () => {
      const { result } = renderHook(() => useSystemAlerts());

      // Mock some alerts
      act(() => {
        result.current.alerts.push(
          { type: 'system_alert', request_id: 'alert-1' },
          { type: 'system_alert', request_id: 'alert-2' }
        );
      });

      act(() => {
        result.current.dismissAlert('alert-1');
      });

      expect(result.current.alerts).toHaveLength(1);
      expect(result.current.alerts[0].request_id).toBe('alert-2');
    });
  });
});