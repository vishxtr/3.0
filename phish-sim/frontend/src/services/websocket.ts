// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Phish-Sim Project

import { 
  WebSocketMessage, 
  WebSocketConnection, 
  AnalysisResponse 
} from '../types';

type WebSocketEventHandler = (data: any) => void;
type ConnectionEventHandler = (connected: boolean) => void;

class WebSocketService {
  private ws: WebSocket | null = null;
  private url: string;
  private userId: string;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectInterval = 1000;
  private pingInterval: NodeJS.Timeout | null = null;
  private isConnecting = false;
  private isManualDisconnect = false;

  // Event handlers
  private eventHandlers: Map<string, WebSocketEventHandler[]> = new Map();
  private connectionHandlers: ConnectionEventHandler[] = [];

  constructor() {
    this.url = import.meta.env.VITE_WEBSOCKET_URL || 'ws://localhost:8003';
    this.userId = this.generateUserId();
  }

  private generateUserId(): string {
    return `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  // Connection management
  async connect(): Promise<void> {
    if (this.ws?.readyState === WebSocket.OPEN || this.isConnecting) {
      return;
    }

    this.isConnecting = true;
    this.isManualDisconnect = false;

    try {
      const wsUrl = `${this.url}/ws/${this.userId}`;
      console.log(`Connecting to WebSocket: ${wsUrl}`);
      
      this.ws = new WebSocket(wsUrl);
      
      this.ws.onopen = this.handleOpen.bind(this);
      this.ws.onmessage = this.handleMessage.bind(this);
      this.ws.onclose = this.handleClose.bind(this);
      this.ws.onerror = this.handleError.bind(this);

    } catch (error) {
      console.error('WebSocket connection error:', error);
      this.isConnecting = false;
      throw error;
    }
  }

  disconnect(): void {
    this.isManualDisconnect = true;
    this.isConnecting = false;
    
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  private handleOpen(): void {
    console.log('WebSocket connected');
    this.isConnecting = false;
    this.reconnectAttempts = 0;
    
    // Start ping interval
    this.startPingInterval();
    
    // Notify connection handlers
    this.connectionHandlers.forEach(handler => handler(true));
    
    // Emit connection event
    this.emit('connected', { userId: this.userId });
  }

  private handleMessage(event: MessageEvent): void {
    try {
      const message: WebSocketMessage = JSON.parse(event.data);
      console.log('WebSocket message received:', message);
      
      // Handle different message types
      switch (message.type) {
        case 'ping':
          this.handlePing();
          break;
        case 'analysis_update':
          this.emit('analysis_update', message);
          break;
        case 'system_alert':
          this.emit('system_alert', message);
          break;
        case 'error':
          this.emit('error', message);
          break;
        default:
          this.emit('message', message);
      }
    } catch (error) {
      console.error('Error parsing WebSocket message:', error);
    }
  }

  private handleClose(event: CloseEvent): void {
    console.log('WebSocket disconnected:', event.code, event.reason);
    this.isConnecting = false;
    
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }

    // Notify connection handlers
    this.connectionHandlers.forEach(handler => handler(false));
    
    // Emit disconnection event
    this.emit('disconnected', { code: event.code, reason: event.reason });

    // Attempt reconnection if not manual disconnect
    if (!this.isManualDisconnect && this.reconnectAttempts < this.maxReconnectAttempts) {
      this.scheduleReconnect();
    }
  }

  private handleError(error: Event): void {
    console.error('WebSocket error:', error);
    this.emit('error', { type: 'websocket_error', error });
  }

  private scheduleReconnect(): void {
    this.reconnectAttempts++;
    const delay = this.reconnectInterval * Math.pow(2, this.reconnectAttempts - 1);
    
    console.log(`Scheduling WebSocket reconnect attempt ${this.reconnectAttempts} in ${delay}ms`);
    
    setTimeout(() => {
      if (!this.isManualDisconnect) {
        this.connect();
      }
    }, delay);
  }

  private startPingInterval(): void {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
    }

    this.pingInterval = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.send({ type: 'ping' });
      }
    }, 30000); // Ping every 30 seconds
  }

  private handlePing(): void {
    // Respond to ping with pong
    this.send({ type: 'pong' });
  }

  // Message sending
  send(message: WebSocketMessage): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      try {
        this.ws.send(JSON.stringify(message));
        console.log('WebSocket message sent:', message);
      } catch (error) {
        console.error('Error sending WebSocket message:', error);
      }
    } else {
      console.warn('WebSocket not connected, cannot send message:', message);
    }
  }

  // Event handling
  on(event: string, handler: WebSocketEventHandler): void {
    if (!this.eventHandlers.has(event)) {
      this.eventHandlers.set(event, []);
    }
    this.eventHandlers.get(event)!.push(handler);
  }

  off(event: string, handler: WebSocketEventHandler): void {
    const handlers = this.eventHandlers.get(event);
    if (handlers) {
      const index = handlers.indexOf(handler);
      if (index > -1) {
        handlers.splice(index, 1);
      }
    }
  }

  private emit(event: string, data: any): void {
    const handlers = this.eventHandlers.get(event);
    if (handlers) {
      handlers.forEach(handler => {
        try {
          handler(data);
        } catch (error) {
          console.error(`Error in WebSocket event handler for ${event}:`, error);
        }
      });
    }
  }

  // Connection event handling
  onConnectionChange(handler: ConnectionEventHandler): void {
    this.connectionHandlers.push(handler);
  }

  offConnectionChange(handler: ConnectionEventHandler): void {
    const index = this.connectionHandlers.indexOf(handler);
    if (index > -1) {
      this.connectionHandlers.splice(index, 1);
    }
  }

  // Utility methods
  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  getConnectionState(): string {
    if (!this.ws) return 'disconnected';
    
    switch (this.ws.readyState) {
      case WebSocket.CONNECTING:
        return 'connecting';
      case WebSocket.OPEN:
        return 'connected';
      case WebSocket.CLOSING:
        return 'closing';
      case WebSocket.CLOSED:
        return 'closed';
      default:
        return 'unknown';
    }
  }

  getUserId(): string {
    return this.userId;
  }

  setUserId(userId: string): void {
    this.userId = userId;
  }

  setUrl(url: string): void {
    this.url = url;
  }

  // Analysis-specific methods
  subscribeToAnalysis(requestId: string, callback: (result: AnalysisResponse) => void): void {
    const handler = (message: WebSocketMessage) => {
      if (message.type === 'analysis_update' && message.request_id === requestId) {
        callback(message.result!);
      }
    };

    this.on('analysis_update', handler);
  }

  unsubscribeFromAnalysis(requestId: string, callback: (result: AnalysisResponse) => void): void {
    const handler = (message: WebSocketMessage) => {
      if (message.type === 'analysis_update' && message.request_id === requestId) {
        callback(message.result!);
      }
    };

    this.off('analysis_update', handler);
  }
}

// Create singleton instance
export const websocketService = new WebSocketService();
export default websocketService;