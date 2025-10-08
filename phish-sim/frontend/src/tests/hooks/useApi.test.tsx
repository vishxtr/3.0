// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Phish-Sim Project

import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import React from 'react';
import { useHealth, useModelInfo, useAnalysis, useConnectionStatus } from '../../hooks/useApi';

// Mock the API service
const mockApiService = {
  getHealth: jest.fn(),
  getModelInfo: jest.fn(),
  analyzeContent: jest.fn(),
  testConnection: jest.fn(),
};

jest.mock('../../services/api', () => ({
  apiService: mockApiService,
}));

const createTestQueryClient = () => new QueryClient({
  defaultOptions: {
    queries: { retry: false },
    mutations: { retry: false },
  },
});

const wrapper = ({ children }: { children: React.ReactNode }) => {
  const queryClient = createTestQueryClient();
  return (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  );
};

describe('useApi hooks', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('useHealth', () => {
    test('fetches health data successfully', async () => {
      const mockHealthData = {
        status: 'ok',
        service: 'realtime_inference',
        version: '0.0.1',
        uptime_seconds: 100,
      };

      mockApiService.getHealth.mockResolvedValue(mockHealthData);

      const { result } = renderHook(() => useHealth(), { wrapper });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(result.current.data).toEqual(mockHealthData);
      expect(mockApiService.getHealth).toHaveBeenCalledTimes(1);
    });

    test('handles health fetch error', async () => {
      const mockError = new Error('Health check failed');
      mockApiService.getHealth.mockRejectedValue(mockError);

      const { result } = renderHook(() => useHealth(), { wrapper });

      await waitFor(() => {
        expect(result.current.isError).toBe(true);
      });

      expect(result.current.error).toEqual(mockError);
    });
  });

  describe('useModelInfo', () => {
    test('fetches model info successfully', async () => {
      const mockModelInfo = {
        nlp_model: { name: 'Test NLP', version: '1.0', status: 'loaded', path: '/test' },
        visual_model: { name: 'Test Visual', version: '1.0', status: 'loaded', path: '/test' },
        thresholds: { phishing: 0.7, suspicious: 0.4 },
      };

      mockApiService.getModelInfo.mockResolvedValue(mockModelInfo);

      const { result } = renderHook(() => useModelInfo(), { wrapper });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(result.current.data).toEqual(mockModelInfo);
      expect(mockApiService.getModelInfo).toHaveBeenCalledTimes(1);
    });
  });

  describe('useAnalysis', () => {
    test('analyzes content successfully', async () => {
      const mockAnalysisRequest = {
        content: 'test content',
        content_type: 'text' as const,
      };

      const mockAnalysisResponse = {
        request_id: 'test-123',
        content: 'test content',
        content_type: 'text',
        prediction: 'benign' as const,
        confidence: 0.9,
        explanation: { nlp: { reason: 'test' } },
        processing_time_ms: 50,
        cached: false,
        timestamp: '2024-01-01T00:00:00Z',
      };

      mockApiService.analyzeContent.mockResolvedValue(mockAnalysisResponse);

      const { result } = renderHook(() => useAnalysis(), { wrapper });

      await result.current.analyze(mockAnalysisRequest);

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(result.current.data).toEqual(mockAnalysisResponse);
      expect(mockApiService.analyzeContent).toHaveBeenCalledWith(mockAnalysisRequest);
    });

    test('handles analysis error', async () => {
      const mockError = new Error('Analysis failed');
      mockApiService.analyzeContent.mockRejectedValue(mockError);

      const { result } = renderHook(() => useAnalysis(), { wrapper });

      const mockRequest = {
        content: 'test content',
        content_type: 'text' as const,
      };

      await result.current.analyze(mockRequest);

      await waitFor(() => {
        expect(result.current.isError).toBe(true);
      });

      expect(result.current.error).toEqual(mockError);
    });
  });

  describe('useConnectionStatus', () => {
    test('checks connection successfully', async () => {
      mockApiService.testConnection.mockResolvedValue(true);

      const { result } = renderHook(() => useConnectionStatus(), { wrapper });

      await waitFor(() => {
        expect(result.current.isConnected).toBe(true);
      });

      expect(mockApiService.testConnection).toHaveBeenCalled();
    });

    test('handles connection failure', async () => {
      mockApiService.testConnection.mockResolvedValue(false);

      const { result } = renderHook(() => useConnectionStatus(), { wrapper });

      await waitFor(() => {
        expect(result.current.isConnected).toBe(false);
      });

      expect(result.current.error).toBeDefined();
    });
  });
});