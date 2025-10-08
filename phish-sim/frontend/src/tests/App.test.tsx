// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Phish-Sim Project

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import App from '../App';

// Mock the hooks
jest.mock('../hooks/useWebSocket', () => ({
  useWebSocketConnection: () => ({
    isConnected: true,
    connectionState: 'connected',
    connect: jest.fn(),
    disconnect: jest.fn(),
  }),
}));

// Mock the API service
jest.mock('../services/api', () => ({
  apiService: {
    getHealth: jest.fn().mockResolvedValue({
      status: 'ok',
      service: 'realtime_inference',
      version: '0.0.1',
      uptime_seconds: 100,
    }),
    getModelInfo: jest.fn().mockResolvedValue({
      nlp_model: { name: 'Test NLP', version: '1.0', status: 'loaded', path: '/test' },
      visual_model: { name: 'Test Visual', version: '1.0', status: 'loaded', path: '/test' },
      thresholds: { phishing: 0.7, suspicious: 0.4 },
    }),
    analyzeContent: jest.fn().mockResolvedValue({
      request_id: 'test-123',
      content: 'test content',
      content_type: 'text',
      prediction: 'benign',
      confidence: 0.9,
      explanation: { nlp: { reason: 'test' } },
      processing_time_ms: 50,
      cached: false,
      timestamp: '2024-01-01T00:00:00Z',
    }),
  },
}));

// Mock WebSocket service
jest.mock('../services/websocket', () => ({
  websocketService: {
    connect: jest.fn(),
    disconnect: jest.fn(),
    send: jest.fn(),
    isConnected: () => true,
    getConnectionState: () => 'connected',
    getUserId: () => 'test-user',
    setUrl: jest.fn(),
  },
}));

const createTestQueryClient = () => new QueryClient({
  defaultOptions: {
    queries: { retry: false },
    mutations: { retry: false },
  },
});

const TestWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const queryClient = createTestQueryClient();
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        {children}
      </BrowserRouter>
    </QueryClientProvider>
  );
};

describe('App', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders navigation with all menu items', () => {
    render(
      <TestWrapper>
        <App />
      </TestWrapper>
    );

    expect(screen.getByText('Phish-Sim')).toBeInTheDocument();
    expect(screen.getByText('Dashboard')).toBeInTheDocument();
    expect(screen.getByText('Analysis')).toBeInTheDocument();
    expect(screen.getByText('Simulator')).toBeInTheDocument();
    expect(screen.getByText('Settings')).toBeInTheDocument();
  });

  test('shows WebSocket connection status', () => {
    render(
      <TestWrapper>
        <App />
      </TestWrapper>
    );

    expect(screen.getByText('Real-time')).toBeInTheDocument();
  });

  test('navigates between pages', async () => {
    render(
      <TestWrapper>
        <App />
      </TestWrapper>
    );

    // Click on Analysis
    fireEvent.click(screen.getByText('Analysis'));
    await waitFor(() => {
      expect(screen.getByText('Phishing Analysis')).toBeInTheDocument();
    });

    // Click on Simulator
    fireEvent.click(screen.getByText('Simulator'));
    await waitFor(() => {
      expect(screen.getByText('Simulation Configuration')).toBeInTheDocument();
    });

    // Click on Settings
    fireEvent.click(screen.getByText('Settings'));
    await waitFor(() => {
      expect(screen.getByText('API Configuration')).toBeInTheDocument();
    });

    // Click on Dashboard
    fireEvent.click(screen.getByText('Dashboard'));
    await waitFor(() => {
      expect(screen.getByText('Phish-Sim Dashboard')).toBeInTheDocument();
    });
  });

  test('highlights active navigation item', () => {
    render(
      <TestWrapper>
        <App />
      </TestWrapper>
    );

    const dashboardLink = screen.getByText('Dashboard').closest('a');
    expect(dashboardLink).toHaveClass('text-white', 'bg-gray-700');
  });
});