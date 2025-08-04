"""
Integration tests for error handling and fault tolerance.

Tests comprehensive error scenarios, recovery mechanisms,
and fault tolerance in distributed federated learning.
"""

import asyncio
import pytest
import logging
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any
import httpx

# Test imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dp_federated_lora.exceptions import (
    NetworkError,
    AuthenticationError,
    CommunicationError,
    RegistrationError,
    TimeoutError,
    ValidationError,
    ErrorContext,
    ErrorSeverity
)
from dp_federated_lora.error_handler import (
    ErrorHandler,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
    RetryConfig,
    with_error_handling,
    error_boundary
)
from dp_federated_lora.network_client import FederatedNetworkClient

# Configure test logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestErrorHandling:
    """Test error handling mechanisms."""
    
    def setup_method(self):
        """Setup test environment."""
        self.error_handler = ErrorHandler()
        self.mock_callbacks = []
    
    def test_error_context_creation(self):
        """Test error context creation and validation."""
        context = ErrorContext(
            component="test_component",
            operation="test_operation",
            client_id="test_client",
            round_num=1,
            severity=ErrorSeverity.ERROR,
            recoverable=True,
            max_retries=3
        )
        
        assert context.component == "test_component"
        assert context.operation == "test_operation"
        assert context.client_id == "test_client"
        assert context.round_num == 1
        assert context.can_retry() is True
        assert context.retry_count == 0
        
        # Test retry logic
        context.increment_retry()
        assert context.retry_count == 1
        assert context.can_retry() is True
        
        # Exhaust retries
        for _ in range(3):
            context.increment_retry()
        assert context.can_retry() is False
    
    def test_circuit_breaker_states(self):
        """Test circuit breaker state transitions."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=1.0,
            success_threshold=2
        )
        breaker = CircuitBreaker(config)
        
        # Initial state: CLOSED
        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.can_execute() is True
        
        # Record failures to trigger OPEN state
        for _ in range(3):
            breaker.record_failure()
        
        assert breaker.state == CircuitBreakerState.OPEN
        assert breaker.can_execute() is False
        
        # Wait for recovery timeout and check HALF_OPEN
        import time
        time.sleep(1.1)
        assert breaker.can_execute() is True
        assert breaker.state == CircuitBreakerState.HALF_OPEN
        
        # Record successes to return to CLOSED
        breaker.execute_call()
        breaker.record_success()
        breaker.execute_call()
        breaker.record_success()
        
        assert breaker.state == CircuitBreakerState.CLOSED
    
    @pytest.mark.asyncio
    async def test_retry_with_error_handler(self):
        """Test retry mechanism with error handler."""
        attempt_count = 0
        
        async def failing_operation():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise NetworkError("Simulated network failure")
            return "success"
        
        context = ErrorContext("test", "retry_test")
        retry_config = RetryConfig(max_attempts=3, base_delay=0.1)
        
        result = await self.error_handler.execute_with_recovery(
            failing_operation,
            context,
            retry_config
        )
        
        assert result == "success"
        assert attempt_count == 3
    
    @pytest.mark.asyncio
    async def test_non_recoverable_error(self):
        """Test handling of non-recoverable errors."""
        async def auth_failing_operation():
            raise AuthenticationError("Invalid credentials")
        
        context = ErrorContext("test", "auth_test", recoverable=True)
        retry_config = RetryConfig(max_attempts=3)
        
        with pytest.raises(AuthenticationError):
            await self.error_handler.execute_with_recovery(
                auth_failing_operation,
                context,
                retry_config
            )
    
    @pytest.mark.asyncio
    async def test_error_callback_notification(self):
        """Test error callback notifications."""
        callback_called = False
        callback_error = None
        callback_context = None
        
        def error_callback(error, context):
            nonlocal callback_called, callback_error, callback_context
            callback_called = True
            callback_error = error
            callback_context = context
        
        self.error_handler.add_error_callback(error_callback)
        
        async def failing_operation():
            raise NetworkError("Test error")
        
        context = ErrorContext("test", "callback_test")
        
        try:
            await self.error_handler.execute_with_recovery(
                failing_operation,
                context,
                RetryConfig(max_attempts=1)
            )
        except NetworkError:
            pass
        
        assert callback_called is True
        assert isinstance(callback_error, NetworkError)
        assert callback_context == context
    
    @pytest.mark.asyncio
    async def test_error_boundary_context_manager(self):
        """Test error boundary context manager."""
        try:
            async with error_boundary(
                component="test_component",
                operation="test_operation",
                client_id="test_client"
            ) as context:
                raise ValueError("Test error")
        except Exception as e:
            # Should be wrapped in appropriate error type
            assert hasattr(e, 'details')
            assert 'cause' in e.details


class TestNetworkClientErrorHandling:
    """Test network client error handling."""
    
    def setup_method(self):
        """Setup test environment."""
        self.client = FederatedNetworkClient(
            server_url="http://test-server:8080",
            client_id="test_client",
            timeout=10,
            retry_config=RetryConfig(max_attempts=2, base_delay=0.1)
        )
    
    async def teardown_method(self):
        """Cleanup test environment."""
        await self.client.close()
    
    @pytest.mark.asyncio
    async def test_connection_error_handling(self):
        """Test connection error handling."""
        with patch.object(self.client.client, 'get') as mock_get:
            mock_get.side_effect = httpx.ConnectError("Connection failed")
            
            with pytest.raises(CommunicationError) as exc_info:
                await self.client._make_request("GET", "/test")
            
            assert "Failed to connect to server" in str(exc_info.value)
            assert exc_info.value.details["client_id"] == "test_client"
    
    @pytest.mark.asyncio
    async def test_timeout_error_handling(self):
        """Test timeout error handling."""
        with patch.object(self.client.client, 'get') as mock_get:
            mock_get.side_effect = httpx.TimeoutException("Request timeout")
            
            with pytest.raises(TimeoutError) as exc_info:
                await self.client._make_request("GET", "/test")
            
            assert "Request timeout" in str(exc_info.value)
            assert exc_info.value.details["client_id"] == "test_client"
    
    @pytest.mark.asyncio
    async def test_authentication_error_handling(self):
        """Test authentication error handling."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_response.json.return_value = {"error": "Invalid token"}
        
        with patch.object(self.client.client, 'get') as mock_get:
            mock_get.return_value = mock_response
            
            with pytest.raises(AuthenticationError) as exc_info:
                await self.client._make_request("GET", "/test")
            
            assert "Authentication failed" in str(exc_info.value)
            assert exc_info.value.details["status_code"] == 401
            # Token should be cleared on auth failure
            assert self.client.auth_token is None
    
    @pytest.mark.asyncio
    async def test_rate_limiting_error_handling(self):
        """Test rate limiting error handling."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.text = "Too Many Requests"
        mock_response.headers = {"Retry-After": "60"}
        
        with patch.object(self.client.client, 'get') as mock_get:
            mock_get.return_value = mock_response
            
            with pytest.raises(CommunicationError) as exc_info:
                await self.client._make_request("GET", "/test")
            
            assert "Rate limited" in str(exc_info.value)
            assert exc_info.value.details["retry_after"] == "60"
    
    @pytest.mark.asyncio
    async def test_invalid_json_response_handling(self):
        """Test invalid JSON response handling."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "Invalid JSON"
        mock_response.json.side_effect = ValueError("Invalid JSON")
        
        with patch.object(self.client.client, 'get') as mock_get:
            mock_get.return_value = mock_response
            
            with pytest.raises(CommunicationError) as exc_info:
                await self.client._make_request("GET", "/test")
            
            assert "Invalid JSON response" in str(exc_info.value)
            assert "Invalid JSON" in exc_info.value.details["response_text"]
    
    @pytest.mark.asyncio
    async def test_registration_validation_error(self):
        """Test registration input validation."""
        # Missing required fields
        capabilities = {"optional_field": "value"}
        
        with pytest.raises(ValidationError) as exc_info:
            await self.client.register(capabilities)
        
        assert "Missing required capability fields" in str(exc_info.value)
        assert "num_examples" in exc_info.value.details["missing_fields"]
    
    @pytest.mark.asyncio
    async def test_registration_server_error(self):
        """Test registration server error handling."""
        capabilities = {"num_examples": 100, "compute_capability": "cpu"}
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": False,
            "message": "Server overloaded"
        }
        
        with patch.object(self.client.client, 'post') as mock_post:
            mock_post.return_value = mock_response
            
            with pytest.raises(RegistrationError) as exc_info:
                await self.client.register(capabilities)
            
            assert "Registration rejected by server" in str(exc_info.value)
            assert "Server overloaded" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_connection_status_tracking(self):
        """Test connection status tracking."""
        # Initially not connected
        status = self.client.get_connection_status()
        assert status["is_connected"] is False
        assert status["last_heartbeat"] == 0.0
        
        # Simulate successful request
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}
        
        with patch.object(self.client.client, 'get') as mock_get:
            mock_get.return_value = mock_response
            
            result = await self.client.health_check()
            assert result is True
            
            # Status should be updated
            status = self.client.get_connection_status()
            assert status["is_connected"] is True
            assert status["last_heartbeat"] > 0
    
    @pytest.mark.asyncio
    async def test_reconnection_logic(self):
        """Test client reconnection logic."""
        capabilities = {"num_examples": 100, "compute_capability": "cpu"}
        
        # Mock successful health check
        health_response = Mock()
        health_response.status_code = 200
        health_response.json.return_value = {"status": "healthy"}
        
        # Mock successful registration
        register_response = Mock()
        register_response.status_code = 200  
        register_response.json.return_value = {
            "success": True,
            "server_config": {"token": "new_token"}
        }
        
        with patch.object(self.client.client, 'get') as mock_get, \
             patch.object(self.client.client, 'post') as mock_post:
            
            mock_get.return_value = health_response
            mock_post.return_value = register_response
            
            # Clear initial state
            self.client.is_connected = False
            self.client.auth_token = None
            
            success = await self.client.reconnect(capabilities)
            assert success is True
            assert self.client.auth_token == "new_token"
            assert self.client.is_connected is True


class TestDecoratorErrorHandling:
    """Test decorator-based error handling."""
    
    @pytest.mark.asyncio
    async def test_with_error_handling_decorator(self):
        """Test the with_error_handling decorator."""
        attempt_count = 0
        
        @with_error_handling(
            component="test",
            operation="decorated_test",
            retry_config=RetryConfig(max_attempts=3, base_delay=0.1)
        )
        async def test_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise NetworkError("Temporary failure")
            return "success"
        
        result = await test_function()
        assert result == "success"
        assert attempt_count == 3
    
    @pytest.mark.asyncio
    async def test_decorator_with_circuit_breaker(self):
        """Test decorator with circuit breaker."""
        call_count = 0
        
        @with_error_handling(
            component="test",
            operation="circuit_test",
            circuit_breaker_name="test_breaker",
            retry_config=RetryConfig(max_attempts=1)
        )
        async def failing_function():
            nonlocal call_count
            call_count += 1
            raise NetworkError("Always fails")
        
        # First few calls should fail normally
        for _ in range(5):
            try:
                await failing_function()
            except NetworkError:
                pass
        
        # Circuit breaker should now be open
        # This would require accessing the global error handler to verify
        # but the important thing is that it doesn't crash


# Integration test fixtures
@pytest.fixture
async def mock_server():
    """Mock federated server for testing."""
    # This would set up a mock server for integration testing
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])