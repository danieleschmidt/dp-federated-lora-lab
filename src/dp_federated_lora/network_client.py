"""
Network client for DP-Federated LoRA training.

This module implements the client-side network communication for connecting
to the federated learning server via HTTP API.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
import json

import httpx
from pydantic import BaseModel

from .config import FederatedConfig, PrivacyConfig, LoRAConfig, SecurityConfig
from .exceptions import (
    NetworkError,
    AuthenticationError,
    CommunicationError,
    RegistrationError,
    TimeoutError,
    ErrorContext,
    ErrorSeverity
)
from .error_handler import (
    error_handler,
    with_error_handling,
    error_boundary,
    RetryConfig,
    CircuitBreakerConfig
)

logger = logging.getLogger(__name__)


# Backward compatibility
class NetworkClientError(NetworkError):
    """Network client error exception (deprecated - use NetworkError)."""
    pass


class FederatedNetworkClient:
    """
    Network client for federated learning communication.
    
    Handles HTTP communication with the federated learning server,
    including registration, parameter exchange, and update submission.
    """
    
    def __init__(
        self,
        server_url: str,
        client_id: str,
        timeout: int = 300,
        max_retries: int = 3,
        verify_ssl: bool = True,
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None
    ):
        """
        Initialize network client with enhanced error handling.
        
        Args:
            server_url: Base URL of the federated server
            client_id: Unique client identifier
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            verify_ssl: Whether to verify SSL certificates
            retry_config: Retry configuration for error handling
            circuit_breaker_config: Circuit breaker configuration
        """
        self.server_url = server_url.rstrip('/')
        self.client_id = client_id
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        
        # Error handling configuration
        self.retry_config = retry_config or RetryConfig(
            max_attempts=max_retries,
            base_delay=1.0,
            max_delay=30.0,
            exponential_base=2.0,
            jitter=True
        )
        
        self.circuit_breaker_name = f"network_client_{client_id}"
        if circuit_breaker_config:
            error_handler.get_circuit_breaker(self.circuit_breaker_name, circuit_breaker_config)
        
        # Authentication
        self.auth_token: Optional[str] = None
        self.server_config: Optional[Dict[str, Any]] = None
        
        # Connection state
        self.is_connected = False
        self.last_heartbeat = 0.0
        self.connection_attempts = 0
        
        # HTTP client with enhanced configuration
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            verify=verify_ssl,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            headers={"User-Agent": f"dp-federated-lora-client/{client_id}"}
        )
        
        logger.info(f"Initialized network client for server {server_url} with enhanced error handling")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers."""
        if not self.auth_token:
            raise AuthenticationError("Client not authenticated", {
                "client_id": self.client_id,
                "server_url": self.server_url
            })
        
        return {
            "Authorization": f"Bearer {self.client_id}:{self.auth_token}",
            "Content-Type": "application/json",
            "X-Client-ID": self.client_id,
            "X-Request-ID": f"{self.client_id}_{int(time.time())}"
        }
    
    @with_error_handling(
        component="network_client",
        operation="make_request",
        circuit_breaker_name="http_requests"
    )
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        require_auth: bool = True,
        timeout_override: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request with comprehensive error handling.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            data: Request data
            require_auth: Whether authentication is required
            timeout_override: Override default timeout
            
        Returns:
            Response data
            
        Raises:
            NetworkError: If request fails
            AuthenticationError: If authentication fails
            TimeoutError: If request times out
        """
        url = f"{self.server_url}{endpoint}"
        
        try:
            headers = self._get_auth_headers() if require_auth else {
                "Content-Type": "application/json",
                "X-Client-ID": self.client_id,
                "X-Request-ID": f"{self.client_id}_{int(time.time())}"
            }
        except AuthenticationError:
            if require_auth:
                raise
            headers = {"Content-Type": "application/json"}
        
        # Set timeout
        timeout = timeout_override or self.timeout
        
        async with error_boundary(
            component="network_client",
            operation=f"{method.upper()} {endpoint}",
            client_id=self.client_id
        ):
            try:
                # Track connection attempt
                self.connection_attempts += 1
                
                # Make request based on method
                if method.upper() == "GET":
                    response = await self.client.get(
                        url, 
                        headers=headers, 
                        timeout=timeout
                    )
                elif method.upper() == "POST":
                    response = await self.client.post(
                        url, 
                        headers=headers, 
                        json=data, 
                        timeout=timeout
                    )
                elif method.upper() == "PUT":
                    response = await self.client.put(
                        url, 
                        headers=headers, 
                        json=data, 
                        timeout=timeout
                    )
                elif method.upper() == "DELETE":
                    response = await self.client.delete(
                        url, 
                        headers=headers, 
                        timeout=timeout
                    )
                else:
                    raise CommunicationError(f"Unsupported HTTP method: {method}", {
                        "method": method,
                        "endpoint": endpoint,
                        "client_id": self.client_id
                    })
                
                # Handle HTTP status errors
                if response.status_code == 401:
                    self.auth_token = None  # Clear invalid token
                    raise AuthenticationError(
                        "Authentication failed - token invalid or expired",
                        {
                            "status_code": response.status_code,
                            "response": response.text,
                            "client_id": self.client_id,
                            "endpoint": endpoint
                        }
                    )
                elif response.status_code == 403:
                    raise AuthenticationError(
                        "Access forbidden - insufficient privileges",
                        {
                            "status_code": response.status_code,
                            "response": response.text,
                            "client_id": self.client_id,
                            "endpoint": endpoint
                        }
                    )
                elif response.status_code == 429:
                    # Rate limiting
                    retry_after = response.headers.get("Retry-After", "60")
                    raise CommunicationError(
                        f"Rate limited - retry after {retry_after} seconds",
                        {
                            "status_code": response.status_code,
                            "retry_after": retry_after,
                            "client_id": self.client_id,
                            "endpoint": endpoint
                        }
                    )
                elif response.status_code >= 500:
                    raise CommunicationError(
                        f"Server error: {response.status_code}",
                        {
                            "status_code": response.status_code,
                            "response": response.text,
                            "client_id": self.client_id,
                            "endpoint": endpoint
                        }
                    )
                elif response.status_code >= 400:
                    raise CommunicationError(
                        f"Client error: {response.status_code}",
                        {
                            "status_code": response.status_code,
                            "response": response.text,
                            "client_id": self.client_id,
                            "endpoint": endpoint
                        }
                    )
                
                # Parse response
                try:
                    result = response.json()
                    self.is_connected = True
                    self.last_heartbeat = time.time()
                    return result
                except (json.JSONDecodeError, ValueError) as e:
                    raise CommunicationError(
                        "Invalid JSON response from server",
                        {
                            "parse_error": str(e),
                            "response_text": response.text[:500],
                            "client_id": self.client_id,
                            "endpoint": endpoint
                        }
                    )
                
            except httpx.TimeoutException as e:
                self.is_connected = False
                raise TimeoutError(
                    f"Request timeout after {timeout}s",
                    {
                        "timeout": timeout,
                        "client_id": self.client_id,
                        "endpoint": endpoint,
                        "error_details": str(e)
                    }
                )
            except httpx.ConnectError as e:
                self.is_connected = False
                raise CommunicationError(
                    "Failed to connect to server",
                    {
                        "server_url": self.server_url,
                        "client_id": self.client_id,
                        "endpoint": endpoint,
                        "connection_attempts": self.connection_attempts,
                        "error_details": str(e)
                    }
                )
            except httpx.RequestError as e:
                self.is_connected = False
                raise NetworkError(
                    f"Request failed: {type(e).__name__}",
                    {
                        "client_id": self.client_id,
                        "endpoint": endpoint,
                        "error_details": str(e)
                    }
                )
    
    @with_error_handling(
        component="network_client",
        operation="register",
        retry_config=RetryConfig(max_attempts=5, base_delay=2.0)
    )
    async def register(self, capabilities: Dict[str, Any]) -> bool:
        """
        Register with the federated server.
        
        Args:
            capabilities: Client capabilities and metadata
            
        Returns:
            True if registration successful
            
        Raises:
            RegistrationError: If registration fails
            ValidationError: If capabilities are invalid
        """
        # Validate capabilities
        required_fields = ["num_examples", "compute_capability"]
        missing_fields = [field for field in required_fields if field not in capabilities]
        if missing_fields:
            from .exceptions import ValidationError
            raise ValidationError(
                f"Missing required capability fields: {missing_fields}",
                {"missing_fields": missing_fields, "client_id": self.client_id}
            )
        
        async with error_boundary(
            component="network_client",
            operation="client_registration",
            client_id=self.client_id
        ):
            data = {
                "client_id": self.client_id,
                "capabilities": capabilities
            }
            
            response = await self._make_request(
                "POST", "/register", data=data, require_auth=False
            )
            
            if response.get("success"):
                # Validate server response
                if "server_config" not in response or "token" not in response["server_config"]:
                    raise RegistrationError(
                        "Invalid server response - missing configuration or token",
                        {"response": response, "client_id": self.client_id}
                    )
                
                self.auth_token = response["server_config"]["token"]
                self.server_config = response["server_config"]
                self.is_connected = True
                self.last_heartbeat = time.time()
                
                logger.info(f"Successfully registered client {self.client_id}")
                return True
            else:
                raise RegistrationError(
                    f"Registration rejected by server: {response.get('message', 'Unknown error')}",
                    {"response": response, "client_id": self.client_id}
                )
    
    async def get_server_status(self) -> Dict[str, Any]:
        """
        Get current server status.
        
        Returns:
            Server status information
        """
        return await self._make_request("GET", "/status", require_auth=False)
    
    async def get_round_parameters(self, round_num: int) -> Dict[str, Any]:
        """
        Get global parameters for a training round.
        
        Args:
            round_num: Training round number
            
        Returns:
            Training round parameters and global model state
        """
        endpoint = f"/round/{round_num}/parameters"
        return await self._make_request("GET", endpoint)
    
    async def submit_client_update(
        self,
        round_num: int,
        parameters: Dict[str, Any],
        metrics: Dict[str, float],
        num_examples: int,
        privacy_spent: Dict[str, float]
    ) -> bool:
        """
        Submit client updates to the server.
        
        Args:
            round_num: Training round number
            parameters: Updated model parameters
            metrics: Training metrics
            num_examples: Number of training examples used
            privacy_spent: Privacy budget consumed
            
        Returns:
            True if submission successful
        """
        data = {
            "client_id": self.client_id,
            "round_num": round_num,
            "parameters": parameters,
            "metrics": metrics,
            "num_examples": num_examples,
            "privacy_spent": privacy_spent
        }
        
        endpoint = f"/round/{round_num}/submit"
        response = await self._make_request("POST", endpoint, data=data)
        
        return response.get("status") == "success"
    
    @with_error_handling(
        component="network_client",
        operation="health_check",
        retry_config=RetryConfig(max_attempts=2, base_delay=1.0),
        recoverable=True
    )
    async def health_check(self) -> bool:
        """
        Perform health check with the server.
        
        Returns:
            True if server is healthy
        """
        try:
            response = await self._make_request(
                "GET", "/health", 
                require_auth=False, 
                timeout_override=10  # Short timeout for health checks
            )
            is_healthy = response.get("status") == "healthy"
            
            if is_healthy:
                self.is_connected = True
                self.last_heartbeat = time.time()
            else:
                self.is_connected = False
                
            return is_healthy
        except Exception:
            self.is_connected = False
            return False
    
    async def start_heartbeat(self, interval: int = 30) -> None:
        """
        Start periodic heartbeat to maintain connection.
        
        Args:
            interval: Heartbeat interval in seconds
        """
        logger.info(f"Starting heartbeat with {interval}s interval")
        
        while True:
            try:
                await asyncio.sleep(interval)
                
                # Only do heartbeat if we think we're connected
                if self.is_connected:
                    healthy = await self.health_check()
                    if not healthy:
                        logger.warning(f"Heartbeat failed for client {self.client_id}")
                        # Could trigger reconnection logic here
                        
            except asyncio.CancelledError:
                logger.info(f"Heartbeat cancelled for client {self.client_id}")
                break
            except Exception as e:
                logger.error(f"Heartbeat error for client {self.client_id}: {e}")
                self.is_connected = False
    
    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get detailed connection status.
        
        Returns:
            Connection status information
        """
        return {
            "is_connected": self.is_connected,
            "last_heartbeat": self.last_heartbeat,
            "connection_attempts": self.connection_attempts,
            "has_auth_token": self.auth_token is not None,
            "server_url": self.server_url,
            "client_id": self.client_id,
            "time_since_heartbeat": time.time() - self.last_heartbeat if self.last_heartbeat > 0 else None
        }
    
    async def reconnect(self, capabilities: Optional[Dict[str, Any]] = None) -> bool:
        """
        Attempt to reconnect to the server.
        
        Args:
            capabilities: Client capabilities for re-registration
            
        Returns:
            True if reconnection successful
        """
        logger.info(f"Attempting to reconnect client {self.client_id}")
        
        try:
            # Clear previous state
            self.auth_token = None
            self.server_config = None
            self.is_connected = False
            
            # Health check first
            if not await self.health_check():
                return False
            
            # Re-register if capabilities provided
            if capabilities:
                return await self.register(capabilities)
            
            return True
            
        except Exception as e:
            logger.error(f"Reconnection failed for client {self.client_id}: {e}")
            return False
    
    def get_server_config(self) -> Dict[str, Any]:
        """
        Get server configuration received during registration.
        
        Returns:
            Server configuration
            
        Raises:
            NetworkClientError: If not registered
        """
        if not self.server_config:
            raise NetworkClientError("Not registered with server")
        return self.server_config
    
    def is_registered(self) -> bool:
        """Check if client is registered with the server."""
        return self.auth_token is not None and self.server_config is not None


class NetworkClientManager:
    """
    Manager for multiple network clients.
    
    Handles connection pooling and load balancing across multiple servers.
    """
    
    def __init__(self, server_urls: List[str], client_id: str):
        """
        Initialize network client manager.
        
        Args:
            server_urls: List of server URLs
            client_id: Unique client identifier
        """
        self.server_urls = server_urls
        self.client_id = client_id
        self.clients: List[FederatedNetworkClient] = []
        self.active_client: Optional[FederatedNetworkClient] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def initialize(self, capabilities: Dict[str, Any]) -> bool:
        """
        Initialize connections to all servers.
        
        Args:
            capabilities: Client capabilities
            
        Returns:
            True if at least one connection successful
        """
        self.clients = [
            FederatedNetworkClient(url, self.client_id)
            for url in self.server_urls
        ]
        
        # Try to register with each server
        successful_registrations = 0
        for client in self.clients:
            try:
                if await client.register(capabilities):
                    successful_registrations += 1
                    if not self.active_client:
                        self.active_client = client
            except NetworkClientError as e:
                logger.warning(f"Failed to register with {client.server_url}: {e}")
        
        logger.info(f"Successfully registered with {successful_registrations}/{len(self.clients)} servers")
        return successful_registrations > 0
    
    async def close(self):
        """Close all client connections."""
        for client in self.clients:
            await client.close()
        self.clients.clear()
        self.active_client = None
    
    def get_active_client(self) -> FederatedNetworkClient:
        """
        Get the active client connection.
        
        Returns:
            Active network client
            
        Raises:
            NetworkClientError: If no active client
        """
        if not self.active_client:
            raise NetworkClientError("No active client connection")
        return self.active_client
    
    async def failover_to_next_client(self) -> bool:
        """
        Failover to the next available client.
        
        Returns:
            True if failover successful
        """
        current_index = self.clients.index(self.active_client) if self.active_client else -1
        
        for i in range(len(self.clients)):
            next_index = (current_index + i + 1) % len(self.clients)
            client = self.clients[next_index]
            
            try:
                if await client.health_check():
                    self.active_client = client
                    logger.info(f"Failed over to server {client.server_url}")
                    return True
            except NetworkClientError:
                continue
        
        logger.error("No healthy servers available for failover")
        return False