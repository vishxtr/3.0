// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Phish-Sim Project

import axios, { AxiosInstance, AxiosResponse } from 'axios';
import { 
  AnalysisRequest, 
  AnalysisResponse, 
  HealthResponse, 
  ModelInfo,
  ApiError 
} from '../types';

class ApiService {
  private api: AxiosInstance;
  private baseURL: string;

  constructor() {
    this.baseURL = import.meta.env.VITE_API_URL || 'http://localhost:8001';
    this.api = axios.create({
      baseURL: this.baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor
    this.api.interceptors.request.use(
      (config) => {
        console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
        return config;
      },
      (error) => {
        console.error('API Request Error:', error);
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.api.interceptors.response.use(
      (response: AxiosResponse) => {
        console.log(`API Response: ${response.status} ${response.config.url}`);
        return response;
      },
      (error) => {
        console.error('API Response Error:', error);
        const apiError: ApiError = {
          message: error.response?.data?.message || error.message || 'Unknown error',
          code: error.response?.status?.toString() || 'NETWORK_ERROR',
          details: error.response?.data,
          timestamp: new Date(),
        };
        return Promise.reject(apiError);
      }
    );
  }

  // Health Check
  async getHealth(): Promise<HealthResponse> {
    try {
      const response = await this.api.get<HealthResponse>('/health');
      return response.data;
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  }

  // Model Information
  async getModelInfo(): Promise<ModelInfo> {
    try {
      const response = await this.api.get<ModelInfo>('/model/info');
      return response.data;
    } catch (error) {
      console.error('Failed to get model info:', error);
      throw error;
    }
  }

  // Analysis
  async analyzeContent(request: AnalysisRequest): Promise<AnalysisResponse> {
    try {
      const response = await this.api.post<AnalysisResponse>('/analyze', request);
      return response.data;
    } catch (error) {
      console.error('Analysis failed:', error);
      throw error;
    }
  }

  // Batch Analysis
  async analyzeBatch(requests: AnalysisRequest[]): Promise<AnalysisResponse[]> {
    try {
      const promises = requests.map(request => this.analyzeContent(request));
      const responses = await Promise.allSettled(promises);
      
      return responses
        .filter((result): result is PromiseFulfilledResult<AnalysisResponse> => 
          result.status === 'fulfilled'
        )
        .map(result => result.value);
    } catch (error) {
      console.error('Batch analysis failed:', error);
      throw error;
    }
  }

  // Utility methods
  getBaseURL(): string {
    return this.baseURL;
  }

  setBaseURL(url: string): void {
    this.baseURL = url;
    this.api.defaults.baseURL = url;
  }

  // Test connection
  async testConnection(): Promise<boolean> {
    try {
      await this.getHealth();
      return true;
    } catch (error) {
      return false;
    }
  }
}

// Create singleton instance
export const apiService = new ApiService();
export default apiService;