// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Phish-Sim Project

// API Types
export interface AnalysisRequest {
  content: string;
  content_type: 'url' | 'email' | 'text';
  user_id?: string;
  session_id?: string;
  force_reanalyze?: boolean;
}

export interface AnalysisResponse {
  request_id: string;
  content: string;
  content_type: string;
  prediction: 'phish' | 'benign' | 'suspicious';
  confidence: number;
  explanation: {
    nlp?: {
      reason: string;
    };
    visual?: {
      reason: string;
    };
  };
  processing_time_ms: number;
  cached: boolean;
  timestamp: string;
}

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  uptime_seconds: number;
}

export interface ModelInfo {
  nlp_model: {
    name: string;
    version: string;
    status: string;
    path: string;
  };
  visual_model: {
    name: string;
    version: string;
    status: string;
    path: string;
  };
  thresholds: {
    phishing: number;
    suspicious: number;
  };
}

// WebSocket Types
export interface WebSocketMessage {
  type: 'ping' | 'analysis_update' | 'system_alert' | 'error';
  data?: any;
  request_id?: string;
  result?: AnalysisResponse;
  message?: string;
}

export interface WebSocketConnection {
  id: string;
  user_id: string;
  connected_at: Date;
  last_ping: Date;
  is_active: boolean;
}

// Dashboard Types
export interface DashboardStats {
  total_scans: number;
  threats_detected: number;
  avg_response_time_ms: number;
  accuracy_percent: number;
  cache_hit_rate: number;
  requests_per_second: number;
}

export interface SystemStatus {
  backend_api: 'healthy' | 'degraded' | 'down';
  ml_pipeline: 'healthy' | 'degraded' | 'down' | 'initializing';
  database: 'connected' | 'disconnected';
  redis: 'connected' | 'disconnected';
  websocket: 'connected' | 'disconnected';
}

export interface RecentActivity {
  id: string;
  timestamp: Date;
  content: string;
  content_type: string;
  prediction: 'phish' | 'benign' | 'suspicious';
  confidence: number;
  processing_time_ms: number;
}

// Analysis Types
export interface AnalysisHistory {
  id: string;
  timestamp: Date;
  content: string;
  content_type: string;
  prediction: 'phish' | 'benign' | 'suspicious';
  confidence: number;
  processing_time_ms: number;
  cached: boolean;
  explanation: AnalysisResponse['explanation'];
}

// Simulator Types
export interface SimulationConfig {
  name: string;
  description: string;
  attack_type: 'phishing_email' | 'malicious_url' | 'social_engineering';
  target_count: number;
  duration_minutes: number;
  intensity: 'low' | 'medium' | 'high';
}

export interface SimulationResult {
  id: string;
  config: SimulationConfig;
  start_time: Date;
  end_time?: Date;
  status: 'running' | 'completed' | 'failed' | 'paused';
  results: {
    total_attacks: number;
    detected: number;
    missed: number;
    false_positives: number;
    detection_rate: number;
  };
}

// Settings Types
export interface AppSettings {
  api_endpoint: string;
  websocket_endpoint: string;
  auto_refresh_interval: number;
  theme: 'dark' | 'light';
  notifications: {
    enabled: boolean;
    sound: boolean;
    desktop: boolean;
  };
  analysis: {
    auto_analyze: boolean;
    cache_enabled: boolean;
    confidence_threshold: number;
  };
}

// Error Types
export interface ApiError {
  message: string;
  code: string;
  details?: any;
  timestamp: Date;
}

// Loading States
export type LoadingState = 'idle' | 'loading' | 'success' | 'error';

// Component Props
export interface BaseComponentProps {
  className?: string;
  children?: React.ReactNode;
}

export interface CardProps extends BaseComponentProps {
  title?: string;
  subtitle?: string;
  icon?: React.ReactNode;
  actions?: React.ReactNode;
}

export interface ButtonProps extends BaseComponentProps {
  variant?: 'primary' | 'secondary' | 'danger' | 'success';
  size?: 'sm' | 'md' | 'lg';
  disabled?: boolean;
  loading?: boolean;
  onClick?: () => void;
  type?: 'button' | 'submit' | 'reset';
}

export interface InputProps extends BaseComponentProps {
  type?: 'text' | 'email' | 'password' | 'url' | 'textarea';
  placeholder?: string;
  value?: string;
  onChange?: (value: string) => void;
  disabled?: boolean;
  error?: string;
  label?: string;
  required?: boolean;
}

// Chart Types
export interface ChartData {
  name: string;
  value: number;
  timestamp?: Date;
  color?: string;
}

export interface TimeSeriesData {
  timestamp: Date;
  value: number;
  label?: string;
}

// Metrics Types
export interface PerformanceMetrics {
  requests_per_second: number;
  avg_response_time_ms: number;
  p95_response_time_ms: number;
  error_rate_percent: number;
  cache_hit_rate_percent: number;
  active_connections: number;
}

export interface SystemMetrics {
  cpu_usage_percent: number;
  memory_usage_percent: number;
  disk_usage_percent: number;
  network_io_mbps: number;
  uptime_seconds: number;
}