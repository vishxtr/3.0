# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
WebSocket real-time communication manager
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, Any, List, Optional, Set, Callable
from datetime import datetime
from collections import defaultdict

from fastapi import WebSocket, WebSocketDisconnect
import websockets
from websockets.exceptions import ConnectionClosed

from config import get_config, WebSocketConfig

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self, config: Optional[WebSocketConfig] = None):
        self.config = config or get_config("websocket")
        
        # Connection storage
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        self.room_connections: Dict[str, Set[str]] = defaultdict(set)
        self.connection_rooms: Dict[str, Set[str]] = defaultdict(set)
        
        # Rate limiting per connection
        self.connection_rates: Dict[str, List[float]] = defaultdict(list)
        self.max_messages_per_minute = 60
        
        # Message handlers
        self.message_handlers: Dict[str, Callable] = {}
        
        # Statistics
        self.stats = {
            "total_connections": 0,
            "active_connections": 0,
            "total_messages": 0,
            "messages_per_second": 0,
            "last_message_time": 0
        }
    
    async def connect(self, websocket: WebSocket, client_id: Optional[str] = None) -> str:
        """Accept a new WebSocket connection"""
        try:
            await websocket.accept()
            
            # Generate client ID if not provided
            if not client_id:
                client_id = f"client_{uuid.uuid4().hex[:8]}_{int(time.time())}"
            
            # Check connection limits
            if len(self.active_connections) >= self.config.max_connections:
                await websocket.close(code=1013, reason="Server overloaded")
                raise Exception("Maximum connections exceeded")
            
            # Store connection
            self.active_connections[client_id] = websocket
            self.connection_metadata[client_id] = {
                "connected_at": datetime.utcnow(),
                "last_activity": datetime.utcnow(),
                "message_count": 0,
                "ip_address": getattr(websocket.client, 'host', 'unknown'),
                "user_agent": websocket.headers.get('user-agent', 'unknown')
            }
            
            # Update statistics
            self.stats["total_connections"] += 1
            self.stats["active_connections"] += 1
            
            logger.info(f"WebSocket client {client_id} connected")
            return client_id
            
        except Exception as e:
            logger.error(f"Failed to connect WebSocket client: {e}")
            raise
    
    async def disconnect(self, client_id: str):
        """Disconnect a WebSocket client"""
        try:
            if client_id in self.active_connections:
                # Remove from rooms
                if client_id in self.connection_rooms:
                    for room in self.connection_rooms[client_id]:
                        self.room_connections[room].discard(client_id)
                    del self.connection_rooms[client_id]
                
                # Close connection
                websocket = self.active_connections[client_id]
                try:
                    await websocket.close()
                except:
                    pass
                
                # Remove from storage
                del self.active_connections[client_id]
                del self.connection_metadata[client_id]
                
                # Clean up rate limiting
                if client_id in self.connection_rates:
                    del self.connection_rates[client_id]
                
                # Update statistics
                self.stats["active_connections"] -= 1
                
                logger.info(f"WebSocket client {client_id} disconnected")
            
        except Exception as e:
            logger.error(f"Error disconnecting client {client_id}: {e}")
    
    async def send_message(self, client_id: str, message: Dict[str, Any]) -> bool:
        """Send message to specific client"""
        try:
            if client_id not in self.active_connections:
                return False
            
            websocket = self.active_connections[client_id]
            
            # Add timestamp and message ID
            message["timestamp"] = datetime.utcnow().isoformat()
            message["message_id"] = str(uuid.uuid4())
            
            # Send message
            await websocket.send_text(json.dumps(message))
            
            # Update metadata
            if client_id in self.connection_metadata:
                self.connection_metadata[client_id]["last_activity"] = datetime.utcnow()
                self.connection_metadata[client_id]["message_count"] += 1
            
            # Update statistics
            self.stats["total_messages"] += 1
            self.stats["last_message_time"] = time.time()
            
            return True
            
        except (WebSocketDisconnect, ConnectionClosed) as e:
            logger.info(f"Client {client_id} disconnected: {e}")
            await self.disconnect(client_id)
            return False
        except Exception as e:
            logger.error(f"Failed to send message to {client_id}: {e}")
            return False
    
    async def broadcast_message(self, message: Dict[str, Any], room: Optional[str] = None) -> int:
        """Broadcast message to all clients or specific room"""
        try:
            # Determine target connections
            if room and room in self.room_connections:
                target_clients = self.room_connections[room]
            else:
                target_clients = list(self.active_connections.keys())
            
            # Send to all target clients
            sent_count = 0
            for client_id in target_clients:
                if await self.send_message(client_id, message):
                    sent_count += 1
            
            logger.debug(f"Broadcasted message to {sent_count} clients")
            return sent_count
            
        except Exception as e:
            logger.error(f"Failed to broadcast message: {e}")
            return 0
    
    async def join_room(self, client_id: str, room: str) -> bool:
        """Add client to a room"""
        try:
            if client_id not in self.active_connections:
                return False
            
            # Add to room
            self.room_connections[room].add(client_id)
            self.connection_rooms[client_id].add(room)
            
            logger.debug(f"Client {client_id} joined room {room}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add client {client_id} to room {room}: {e}")
            return False
    
    async def leave_room(self, client_id: str, room: str) -> bool:
        """Remove client from a room"""
        try:
            if client_id not in self.active_connections:
                return False
            
            # Remove from room
            self.room_connections[room].discard(client_id)
            self.connection_rooms[client_id].discard(room)
            
            # Clean up empty rooms
            if not self.room_connections[room]:
                del self.room_connections[room]
            
            logger.debug(f"Client {client_id} left room {room}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove client {client_id} from room {room}: {e}")
            return False
    
    def is_rate_limited(self, client_id: str) -> bool:
        """Check if client is rate limited"""
        try:
            now = time.time()
            client_rates = self.connection_rates[client_id]
            
            # Remove old timestamps (older than 1 minute)
            client_rates[:] = [t for t in client_rates if now - t < 60]
            
            # Check if over limit
            if len(client_rates) >= self.max_messages_per_minute:
                return True
            
            # Add current timestamp
            client_rates.append(now)
            return False
            
        except Exception as e:
            logger.error(f"Error checking rate limit for {client_id}: {e}")
            return True
    
    async def handle_message(self, client_id: str, message: Dict[str, Any]):
        """Handle incoming message from client"""
        try:
            # Check rate limiting
            if self.is_rate_limited(client_id):
                await self.send_message(client_id, {
                    "type": "error",
                    "message": "Rate limit exceeded"
                })
                return
            
            # Update last activity
            if client_id in self.connection_metadata:
                self.connection_metadata[client_id]["last_activity"] = datetime.utcnow()
            
            # Get message type
            message_type = message.get("type", "unknown")
            
            # Route to appropriate handler
            if message_type in self.message_handlers:
                handler = self.message_handlers[message_type]
                await handler(client_id, message)
            else:
                # Default handler
                await self._handle_default_message(client_id, message)
            
        except Exception as e:
            logger.error(f"Error handling message from {client_id}: {e}")
            await self.send_message(client_id, {
                "type": "error",
                "message": "Internal server error"
            })
    
    async def _handle_default_message(self, client_id: str, message: Dict[str, Any]):
        """Default message handler"""
        message_type = message.get("type", "unknown")
        
        if message_type == "ping":
            await self.send_message(client_id, {"type": "pong"})
        
        elif message_type == "join_room":
            room = message.get("room")
            if room:
                await self.join_room(client_id, room)
                await self.send_message(client_id, {
                    "type": "room_joined",
                    "room": room
                })
        
        elif message_type == "leave_room":
            room = message.get("room")
            if room:
                await self.leave_room(client_id, room)
                await self.send_message(client_id, {
                    "type": "room_left",
                    "room": room
                })
        
        elif message_type == "get_stats":
            stats = self.get_connection_stats(client_id)
            await self.send_message(client_id, {
                "type": "stats",
                "stats": stats
            })
        
        else:
            await self.send_message(client_id, {
                "type": "error",
                "message": f"Unknown message type: {message_type}"
            })
    
    def register_message_handler(self, message_type: str, handler: Callable):
        """Register a custom message handler"""
        self.message_handlers[message_type] = handler
        logger.info(f"Registered handler for message type: {message_type}")
    
    def get_connection_stats(self, client_id: Optional[str] = None) -> Dict[str, Any]:
        """Get connection statistics"""
        try:
            if client_id:
                # Client-specific stats
                if client_id in self.connection_metadata:
                    metadata = self.connection_metadata[client_id]
                    return {
                        "client_id": client_id,
                        "connected_at": metadata["connected_at"].isoformat(),
                        "last_activity": metadata["last_activity"].isoformat(),
                        "message_count": metadata["message_count"],
                        "ip_address": metadata["ip_address"],
                        "rooms": list(self.connection_rooms.get(client_id, set()))
                    }
                else:
                    return {"error": "Client not found"}
            else:
                # Global stats
                return {
                    "total_connections": self.stats["total_connections"],
                    "active_connections": self.stats["active_connections"],
                    "total_messages": self.stats["total_messages"],
                    "rooms": len(self.room_connections),
                    "connections_per_room": {
                        room: len(clients) for room, clients in self.room_connections.items()
                    }
                }
        
        except Exception as e:
            logger.error(f"Error getting connection stats: {e}")
            return {"error": str(e)}
    
    async def cleanup_inactive_connections(self, timeout: int = 300):
        """Clean up inactive connections"""
        try:
            now = datetime.utcnow()
            inactive_clients = []
            
            for client_id, metadata in self.connection_metadata.items():
                last_activity = metadata["last_activity"]
                if (now - last_activity).total_seconds() > timeout:
                    inactive_clients.append(client_id)
            
            # Disconnect inactive clients
            for client_id in inactive_clients:
                logger.info(f"Disconnecting inactive client: {client_id}")
                await self.disconnect(client_id)
            
            return len(inactive_clients)
            
        except Exception as e:
            logger.error(f"Error cleaning up inactive connections: {e}")
            return 0

class WebSocketManager:
    """Main WebSocket manager"""
    
    def __init__(self, config: Optional[WebSocketConfig] = None):
        self.config = config or get_config("websocket")
        self.connection_manager = ConnectionManager(config)
        self.is_running = False
        
        # Register default handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default message handlers"""
        
        # Inference request handler
        async def handle_inference_request(client_id: str, message: Dict[str, Any]):
            try:
                # This would integrate with the pipeline orchestrator
                # For now, send a mock response
                await self.connection_manager.send_message(client_id, {
                    "type": "inference_result",
                    "request_id": message.get("request_id"),
                    "result": {
                        "prediction": "benign",
                        "confidence": 0.85,
                        "risk_score": 0.15,
                        "risk_level": "low"
                    }
                })
            except Exception as e:
                logger.error(f"Error handling inference request: {e}")
        
        self.connection_manager.register_message_handler("inference_request", handle_inference_request)
        
        # Status request handler
        async def handle_status_request(client_id: str, message: Dict[str, Any]):
            try:
                stats = self.connection_manager.get_connection_stats()
                await self.connection_manager.send_message(client_id, {
                    "type": "status_response",
                    "status": "healthy",
                    "stats": stats
                })
            except Exception as e:
                logger.error(f"Error handling status request: {e}")
        
        self.connection_manager.register_message_handler("status_request", handle_status_request)
    
    async def start(self):
        """Start the WebSocket manager"""
        try:
            self.is_running = True
            
            # Start cleanup task
            asyncio.create_task(self._cleanup_task())
            
            logger.info("WebSocket manager started")
            
        except Exception as e:
            logger.error(f"Failed to start WebSocket manager: {e}")
            raise
    
    async def stop(self):
        """Stop the WebSocket manager"""
        try:
            self.is_running = False
            
            # Disconnect all clients
            client_ids = list(self.connection_manager.active_connections.keys())
            for client_id in client_ids:
                await self.connection_manager.disconnect(client_id)
            
            logger.info("WebSocket manager stopped")
            
        except Exception as e:
            logger.error(f"Error stopping WebSocket manager: {e}")
    
    async def _cleanup_task(self):
        """Background cleanup task"""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Run every minute
                if self.is_running:
                    cleaned = await self.connection_manager.cleanup_inactive_connections()
                    if cleaned > 0:
                        logger.info(f"Cleaned up {cleaned} inactive connections")
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
    
    async def handle_connection(self, websocket: WebSocket, client_id: Optional[str] = None):
        """Handle a new WebSocket connection"""
        try:
            # Accept connection
            connected_client_id = await self.connection_manager.connect(websocket, client_id)
            
            # Send welcome message
            await self.connection_manager.send_message(connected_client_id, {
                "type": "welcome",
                "client_id": connected_client_id,
                "server_time": datetime.utcnow().isoformat(),
                "available_rooms": list(self.config.broadcast_rooms)
            })
            
            # Join default rooms
            for room in self.config.broadcast_rooms:
                await self.connection_manager.join_room(connected_client_id, room)
            
            # Handle messages
            while True:
                try:
                    # Receive message
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    # Handle message
                    await self.connection_manager.handle_message(connected_client_id, message)
                    
                except WebSocketDisconnect:
                    break
                except json.JSONDecodeError:
                    await self.connection_manager.send_message(connected_client_id, {
                        "type": "error",
                        "message": "Invalid JSON format"
                    })
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await self.connection_manager.send_message(connected_client_id, {
                        "type": "error",
                        "message": "Error processing message"
                    })
        
        except Exception as e:
            logger.error(f"Error handling WebSocket connection: {e}")
        
        finally:
            # Clean up connection
            if 'connected_client_id' in locals():
                await self.connection_manager.disconnect(connected_client_id)
    
    async def broadcast_to_room(self, room: str, message: Dict[str, Any]) -> int:
        """Broadcast message to specific room"""
        return await self.connection_manager.broadcast_message(message, room)
    
    async def broadcast_to_all(self, message: Dict[str, Any]) -> int:
        """Broadcast message to all connected clients"""
        return await self.connection_manager.broadcast_message(message)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get WebSocket manager statistics"""
        return self.connection_manager.get_connection_stats()
    
    def is_healthy(self) -> bool:
        """Check if WebSocket manager is healthy"""
        return self.is_running and len(self.connection_manager.active_connections) < self.config.max_connections

def create_websocket_manager(config: Optional[WebSocketConfig] = None) -> WebSocketManager:
    """Factory function to create WebSocket manager instance"""
    return WebSocketManager(config)