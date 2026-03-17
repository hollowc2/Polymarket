"""Resilience patterns for production-grade API interactions.

Provides:
- CircuitBreaker: Prevents cascading failures by stopping requests to failing services
- RateLimiter: Prevents hitting API rate limits
- HealthCheck: Monitors system health state
"""

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeVar

from polymarket_algo.core.config import Config

# Sentinel used as default for log_fn to avoid mutable-default issues
_LOG_FN_DEFAULT: Callable[[str], None] = print


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreaker:
    """Circuit breaker pattern for API failure handling.

    States:
    - CLOSED: Normal operation, all requests pass through
    - OPEN: Too many failures, requests are blocked
    - HALF_OPEN: Testing recovery, allows limited requests

    Usage:
        breaker = CircuitBreaker(name="polymarket_api")

        if breaker.allow_request():
            try:
                result = api_call()
                breaker.record_success()
            except Exception as e:
                breaker.record_failure()
                raise
        else:
            # Circuit is open, skip the call
            raise CircuitOpenError("API circuit is open")
    """

    name: str
    failure_threshold: int = field(default_factory=lambda: Config.CIRCUIT_BREAKER_THRESHOLD)
    recovery_time: int = field(default_factory=lambda: Config.CIRCUIT_BREAKER_RECOVERY_TIME)
    half_open_max_calls: int = 3
    log_fn: Callable[[str], None] = field(default=_LOG_FN_DEFAULT)

    # Internal state
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failures: int = field(default=0, init=False)
    _successes: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _half_open_calls: int = field(default=0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    # Statistics
    total_calls: int = field(default=0, init=False)
    total_failures: int = field(default=0, init=False)
    total_blocked: int = field(default=0, init=False)
    state_changes: list = field(default_factory=list, init=False)

    @property
    def state(self) -> CircuitState:
        """Get current circuit state, checking for recovery."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                # Check if recovery time has passed
                if time.time() - self._last_failure_time >= self.recovery_time:
                    self._transition_to(CircuitState.HALF_OPEN)
            return self._state

    def _transition_to(self, new_state: CircuitState):
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        self.state_changes.append(
            {
                "from": old_state.value,
                "to": new_state.value,
                "timestamp": time.time(),
            }
        )

        if new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self.log_fn(
                f"[CB] Circuit '{self.name}' HALF_OPEN — testing recovery (total_failures={self.total_failures})"
            )
        elif new_state == CircuitState.CLOSED:
            self._failures = 0
            self._successes = 0
            self.log_fn(f"[CB] Circuit '{self.name}' CLOSED — recovered")
        elif new_state == CircuitState.OPEN:
            self.log_fn(
                f"[CB] Circuit '{self.name}' OPEN after {self._failures} consecutive failures "
                f"— blocking for {self.recovery_time}s"
            )

    def allow_request(self) -> bool:
        """Check if a request should be allowed."""
        current_state = self.state  # This may trigger state transition

        with self._lock:
            self.total_calls += 1

            if current_state == CircuitState.CLOSED:
                return True

            elif current_state == CircuitState.OPEN:
                self.total_blocked += 1
                return False

            elif current_state == CircuitState.HALF_OPEN:
                # Allow limited calls in half-open state
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

        return False

    def record_success(self):
        """Record a successful call."""
        with self._lock:
            self._successes += 1

            if self._state == CircuitState.HALF_OPEN:
                # After enough successes in half-open, close the circuit
                if self._successes >= self.half_open_max_calls:
                    self._transition_to(CircuitState.CLOSED)

            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failures = max(0, self._failures - 1)

    def record_failure(self):
        """Record a failed call."""
        with self._lock:
            self._failures += 1
            self.total_failures += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open opens the circuit again
                self._transition_to(CircuitState.OPEN)

            elif self._state == CircuitState.CLOSED:
                # Check if we've hit the failure threshold
                if self._failures >= self.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

    def reset(self):
        """Manually reset the circuit breaker."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
            self._failures = 0
            self._successes = 0

    @property
    def stats(self) -> dict:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failures": self._failures,
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "total_blocked": self.total_blocked,
            "failure_threshold": self.failure_threshold,
            "recovery_time": self.recovery_time,
            "last_failure_age": time.time() - self._last_failure_time if self._last_failure_time else None,
        }


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open and blocking requests."""

    pass


@dataclass
class RateLimiter:
    """Sliding window rate limiter.

    Prevents hitting API rate limits by tracking requests in a sliding window.

    Usage:
        limiter = RateLimiter(requests_per_minute=120)

        if limiter.allow_request():
            result = api_call()
        else:
            # Rate limited, wait or skip
            wait_time = limiter.time_until_allowed()
            time.sleep(wait_time)
    """

    requests_per_minute: int = field(default_factory=lambda: Config.RATE_LIMIT_REQUESTS_PER_MINUTE)
    window_size: float = 60.0  # seconds

    # Internal state
    _requests: deque = field(default_factory=deque, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    # Statistics
    total_requests: int = field(default=0, init=False)
    total_limited: int = field(default=0, init=False)

    def allow_request(self) -> bool:
        """Check if a request should be allowed."""
        with self._lock:
            now = time.time()

            # Remove old requests outside the window
            window_start = now - self.window_size
            while self._requests and self._requests[0] < window_start:
                self._requests.popleft()

            # Check if we're under the limit
            if len(self._requests) < self.requests_per_minute:
                self._requests.append(now)
                self.total_requests += 1
                return True

            self.total_limited += 1
            return False

    def time_until_allowed(self) -> float:
        """Get time in seconds until next request is allowed."""
        with self._lock:
            if len(self._requests) < self.requests_per_minute:
                return 0.0

            if not self._requests:
                return 0.0

            # Time until oldest request expires
            oldest = self._requests[0]
            expires_at = oldest + self.window_size
            wait_time = expires_at - time.time()
            return max(0.0, wait_time)

    def current_rate(self) -> float:
        """Get current request rate (requests per minute)."""
        with self._lock:
            now = time.time()
            window_start = now - self.window_size

            # Count requests in current window
            count = sum(1 for t in self._requests if t >= window_start)
            return count

    @property
    def stats(self) -> dict:
        """Get rate limiter statistics."""
        return {
            "limit": self.requests_per_minute,
            "current_rate": self.current_rate(),
            "total_requests": self.total_requests,
            "total_limited": self.total_limited,
            "utilization_pct": (self.current_rate() / self.requests_per_minute) * 100,
        }


class ErrorCategory(Enum):
    """Categories of errors for handling decisions."""

    RETRYABLE = "retryable"  # Transient error, retry
    FATAL = "fatal"  # Permanent error, don't retry
    RATE_LIMITED = "rate_limited"  # Hit rate limit, wait and retry
    CIRCUIT_OPEN = "circuit_open"  # Circuit breaker open


def categorize_error(error: Exception) -> ErrorCategory:
    """Categorize an error for handling decisions.

    Args:
        error: The exception to categorize

    Returns:
        ErrorCategory indicating how to handle the error
    """
    error_str = str(error).lower()
    type(error).__name__

    # Circuit breaker errors
    if isinstance(error, CircuitOpenError):
        return ErrorCategory.CIRCUIT_OPEN

    # Rate limiting
    if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
        return ErrorCategory.RATE_LIMITED

    # Retryable HTTP errors
    if any(code in error_str for code in ["500", "502", "503", "504"]):
        return ErrorCategory.RETRYABLE

    # Timeout errors are retryable
    if "timeout" in error_str or "timed out" in error_str:
        return ErrorCategory.RETRYABLE

    # Connection errors are retryable
    if "connection" in error_str and ("refused" in error_str or "reset" in error_str or "error" in error_str):
        return ErrorCategory.RETRYABLE

    # Client errors (4xx except 429) are usually fatal
    if any(code in error_str for code in ["400", "401", "403", "404", "422"]):
        return ErrorCategory.FATAL

    # Validation errors are fatal
    if "invalid" in error_str or "validation" in error_str:
        return ErrorCategory.FATAL

    # Insufficient funds/balance are fatal
    if "insufficient" in error_str or "balance" in error_str:
        return ErrorCategory.FATAL

    # Default to retryable for unknown errors
    return ErrorCategory.RETRYABLE


@dataclass
class HealthStatus:
    """Health status of a component."""

    healthy: bool
    component: str
    details: dict = field(default_factory=dict)
    last_check: float = field(default_factory=time.time)


class HealthCheck:
    """Health check system for monitoring component health.

    Usage:
        health = HealthCheck()

        # Register health check functions
        health.register("api", lambda: api_client.is_connected())
        health.register("websocket", lambda: ws.is_connected())

        # Check health
        status = health.check_all()
        if not health.is_healthy():
            alert_operator()
    """

    def __init__(self):
        self._checks: dict[str, Callable[[], bool | dict]] = {}
        self._status: dict[str, HealthStatus] = {}
        self._lock = threading.Lock()

    def register(self, name: str, check_fn: Callable[[], bool | dict]):
        """Register a health check function.

        Args:
            name: Component name
            check_fn: Function that returns True/False or dict with 'healthy' key
        """
        with self._lock:
            self._checks[name] = check_fn

    def check(self, name: str) -> HealthStatus:
        """Run a specific health check."""
        if name not in self._checks:
            return HealthStatus(healthy=False, component=name, details={"error": "unknown component"})

        try:
            result = self._checks[name]()

            if isinstance(result, bool):
                status = HealthStatus(healthy=result, component=name)
            elif isinstance(result, dict):
                healthy = result.get("healthy", True)
                status = HealthStatus(healthy=healthy, component=name, details=result)
            else:
                status = HealthStatus(healthy=bool(result), component=name)

        except Exception as e:
            status = HealthStatus(
                healthy=False,
                component=name,
                details={"error": str(e), "error_type": type(e).__name__},
            )

        with self._lock:
            self._status[name] = status

        return status

    def check_all(self) -> dict[str, HealthStatus]:
        """Run all registered health checks."""
        results = {}
        for name in list(self._checks.keys()):
            results[name] = self.check(name)
        return results

    def is_healthy(self) -> bool:
        """Check if all components are healthy."""
        with self._lock:
            if not self._status:
                return True  # No checks registered
            return all(s.healthy for s in self._status.values())

    def get_status(self) -> dict:
        """Get overall health status."""
        self.check_all()

        with self._lock:
            components = {
                name: {
                    "healthy": status.healthy,
                    "details": status.details,
                    "last_check": status.last_check,
                }
                for name, status in self._status.items()
            }

        return {
            "healthy": self.is_healthy(),
            "components": components,
            "timestamp": time.time(),
        }


# Type variable for generic retry function
T = TypeVar("T")


def with_retry[T](
    fn: Callable[[], T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    circuit_breaker: CircuitBreaker | None = None,
    rate_limiter: RateLimiter | None = None,
) -> T:
    """Execute a function with retry logic and resilience patterns.

    Args:
        fn: Function to execute
        max_retries: Maximum number of retries
        base_delay: Base delay between retries (exponential backoff)
        max_delay: Maximum delay between retries
        circuit_breaker: Optional circuit breaker to use
        rate_limiter: Optional rate limiter to use

    Returns:
        Result of fn()

    Raises:
        The last exception if all retries fail
    """
    last_error = None

    for attempt in range(max_retries + 1):
        # Check rate limiter
        if rate_limiter:
            while not rate_limiter.allow_request():
                wait_time = rate_limiter.time_until_allowed()
                if wait_time > 0:
                    time.sleep(min(wait_time, 1.0))

        # Check circuit breaker
        if circuit_breaker and not circuit_breaker.allow_request():
            raise CircuitOpenError(f"Circuit '{circuit_breaker.name}' is open")

        try:
            result = fn()

            # Record success
            if circuit_breaker:
                circuit_breaker.record_success()

            return result

        except Exception as e:
            last_error = e
            category = categorize_error(e)

            # Record failure in circuit breaker
            if circuit_breaker:
                circuit_breaker.record_failure()

            # Don't retry fatal errors
            if category == ErrorCategory.FATAL:
                raise

            # Don't retry if we've exhausted retries
            if attempt >= max_retries:
                raise

            # Calculate delay with exponential backoff
            delay = min(base_delay * (2**attempt), max_delay)

            # Add extra delay for rate limiting
            if category == ErrorCategory.RATE_LIMITED:
                delay = max(delay, 5.0)  # At least 5s for rate limits

            time.sleep(delay)

    # Should not reach here, but just in case
    if last_error:
        raise last_error
    raise RuntimeError("Retry loop completed without result or error")
