"""
Risk Management Module

Implements safety guardrails from HFT architecture blueprint:
- Pre-Trade Checks (atomic validation before any trade)
- Kill Switch / Circuit Breakers
- Rate Limiting
- Position Monitoring

All checks are synchronous and must PASS before trade execution.
"""

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum

from loguru import logger


class RiskCheckResult(Enum):
    """Result of a risk check."""
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"


@dataclass
class RiskLimits:
    """Configurable risk limits."""
    # Position limits
    max_position_size: float = 500.0  # Max $ per single position
    max_position_pct: float = 0.15  # Max % of capital per position
    max_total_exposure: float = 0.50  # Max % of capital at risk

    # Daily limits
    max_daily_loss: float = 0.05  # Max % daily loss before halt
    max_daily_trades: int = 100  # Max trades per day
    max_trades_per_hour: int = 20  # Rate limit

    # Circuit breakers
    drawdown_warning: float = 0.10  # Warn at 10% drawdown
    drawdown_halt: float = 0.20  # Halt at 20% drawdown
    consecutive_loss_halt: int = 5  # Halt after N consecutive losses

    # Fat finger protection
    min_trade_size: float = 10.0  # Minimum trade size
    max_single_trade_pct: float = 0.10  # Max single trade as % of capital


@dataclass
class RiskState:
    """Current risk state tracking."""
    initial_capital: float = 10000.0
    current_capital: float = 10000.0
    today_pnl: float = 0.0
    today_trades: int = 0
    hourly_trades: int = 0
    hour_start: datetime = field(default_factory=datetime.utcnow)
    consecutive_losses: int = 0
    open_positions: Dict[str, float] = field(default_factory=dict)
    total_exposure: float = 0.0
    is_halted: bool = False
    halt_reason: Optional[str] = None
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)


class RiskManager:
    """
    Comprehensive risk management system.

    Implements the "Risk Management (Guardrails)" layer from HFT blueprint:
    - Pre-Trade Checks: Validate every trade before execution
    - Kill Switch: Emergency halt on severe conditions
    - Circuit Breakers: Automatic pause on drawdown/losses
    - Rate Limiting: Prevent excessive trading

    Usage:
        rm = RiskManager(limits, state)

        # Before each trade
        result, reason = rm.pre_trade_check(trade_params)
        if result != RiskCheckResult.PASS:
            logger.warning(f"Trade blocked: {reason}")
            return

        # Execute trade...

        # After trade
        rm.post_trade_update(trade_result)

        # Periodic health check
        if not rm.heartbeat():
            rm.emergency_halt("Heartbeat failed")
    """

    def __init__(
        self,
        limits: Optional[RiskLimits] = None,
        state: Optional[RiskState] = None,
        on_halt: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize risk manager.

        Args:
            limits: Risk limit configuration
            state: Initial risk state
            on_halt: Callback when system is halted
        """
        self.limits = limits or RiskLimits()
        self.state = state or RiskState()
        self.on_halt = on_halt

        self._lock = threading.Lock()
        self._check_history: List[Tuple[datetime, str, RiskCheckResult]] = []

    def pre_trade_check(
        self,
        market_id: str,
        side: str,
        size: float,
        price: float
    ) -> Tuple[RiskCheckResult, str]:
        """
        Perform all pre-trade risk checks.

        Must PASS all checks before trade execution.

        Args:
            market_id: Unique market identifier
            side: "buy" or "sell"
            size: Trade size in dollars
            price: Current market price

        Returns:
            (RiskCheckResult, reason_string)
        """
        with self._lock:
            checks = [
                self._check_halted(),
                self._check_fat_finger(size),
                self._check_position_size(size),
                self._check_total_exposure(size),
                self._check_daily_loss(),
                self._check_rate_limit(),
                self._check_consecutive_losses(),
            ]

            for result, reason in checks:
                if result == RiskCheckResult.FAIL:
                    self._log_check("pre_trade", result, reason)
                    return result, reason
                elif result == RiskCheckResult.WARN:
                    logger.warning(f"Risk warning: {reason}")

            self._log_check("pre_trade", RiskCheckResult.PASS, "All checks passed")
            return RiskCheckResult.PASS, "All checks passed"

    def _check_halted(self) -> Tuple[RiskCheckResult, str]:
        """Check if system is halted."""
        if self.state.is_halted:
            return RiskCheckResult.FAIL, f"System halted: {self.state.halt_reason}"
        return RiskCheckResult.PASS, ""

    def _check_fat_finger(self, size: float) -> Tuple[RiskCheckResult, str]:
        """Fat finger protection - check for obviously wrong sizes."""
        if size < self.limits.min_trade_size:
            return RiskCheckResult.FAIL, f"Trade size ${size:.2f} below minimum ${self.limits.min_trade_size}"

        max_single = self.state.current_capital * self.limits.max_single_trade_pct
        if size > max_single:
            return RiskCheckResult.FAIL, f"Trade size ${size:.2f} exceeds single trade limit ${max_single:.2f}"

        return RiskCheckResult.PASS, ""

    def _check_position_size(self, size: float) -> Tuple[RiskCheckResult, str]:
        """Check position size limits."""
        if size > self.limits.max_position_size:
            return RiskCheckResult.FAIL, f"Position ${size:.2f} exceeds max ${self.limits.max_position_size}"

        max_pct = self.state.current_capital * self.limits.max_position_pct
        if size > max_pct:
            return RiskCheckResult.FAIL, f"Position ${size:.2f} exceeds {self.limits.max_position_pct:.0%} of capital"

        return RiskCheckResult.PASS, ""

    def _check_total_exposure(self, additional_size: float) -> Tuple[RiskCheckResult, str]:
        """Check total portfolio exposure."""
        new_exposure = self.state.total_exposure + additional_size
        max_exposure = self.state.current_capital * self.limits.max_total_exposure

        if new_exposure > max_exposure:
            return RiskCheckResult.FAIL, f"Total exposure ${new_exposure:.2f} would exceed limit ${max_exposure:.2f}"

        return RiskCheckResult.PASS, ""

    def _check_daily_loss(self) -> Tuple[RiskCheckResult, str]:
        """Check daily loss limit."""
        max_loss = self.state.initial_capital * self.limits.max_daily_loss
        if self.state.today_pnl < -max_loss:
            self._trigger_halt(f"Daily loss limit exceeded: ${self.state.today_pnl:.2f}")
            return RiskCheckResult.FAIL, f"Daily loss ${abs(self.state.today_pnl):.2f} exceeds limit ${max_loss:.2f}"

        return RiskCheckResult.PASS, ""

    def _check_rate_limit(self) -> Tuple[RiskCheckResult, str]:
        """Check trade rate limits."""
        # Reset hourly counter if needed
        now = datetime.utcnow()
        if (now - self.state.hour_start).total_seconds() > 3600:
            self.state.hourly_trades = 0
            self.state.hour_start = now

        if self.state.hourly_trades >= self.limits.max_trades_per_hour:
            return RiskCheckResult.FAIL, f"Hourly trade limit ({self.limits.max_trades_per_hour}) reached"

        if self.state.today_trades >= self.limits.max_daily_trades:
            return RiskCheckResult.FAIL, f"Daily trade limit ({self.limits.max_daily_trades}) reached"

        return RiskCheckResult.PASS, ""

    def _check_consecutive_losses(self) -> Tuple[RiskCheckResult, str]:
        """Check consecutive loss circuit breaker."""
        if self.state.consecutive_losses >= self.limits.consecutive_loss_halt:
            self._trigger_halt(f"Consecutive losses: {self.state.consecutive_losses}")
            return RiskCheckResult.FAIL, f"{self.state.consecutive_losses} consecutive losses - halted"

        return RiskCheckResult.PASS, ""

    def post_trade_update(
        self,
        market_id: str,
        size: float,
        pnl: float,
        is_win: bool
    ):
        """
        Update state after trade execution.

        Args:
            market_id: Market that was traded
            size: Position size
            pnl: Realized P&L from trade
            is_win: Whether trade was profitable
        """
        with self._lock:
            self.state.today_pnl += pnl
            self.state.today_trades += 1
            self.state.hourly_trades += 1
            self.state.current_capital += pnl

            if is_win:
                self.state.consecutive_losses = 0
            else:
                self.state.consecutive_losses += 1

            # Update exposure tracking
            if market_id in self.state.open_positions:
                self.state.total_exposure -= self.state.open_positions[market_id]
                del self.state.open_positions[market_id]

            # Check circuit breakers
            self._check_circuit_breakers()

    def add_position(self, market_id: str, size: float):
        """Track a new open position."""
        with self._lock:
            self.state.open_positions[market_id] = size
            self.state.total_exposure = sum(self.state.open_positions.values())

    def close_position(self, market_id: str):
        """Remove a closed position from tracking."""
        with self._lock:
            if market_id in self.state.open_positions:
                self.state.total_exposure -= self.state.open_positions[market_id]
                del self.state.open_positions[market_id]

    def _check_circuit_breakers(self):
        """Check and trigger circuit breakers if needed."""
        drawdown = (self.state.initial_capital - self.state.current_capital) / self.state.initial_capital

        if drawdown >= self.limits.drawdown_halt:
            self._trigger_halt(f"Drawdown circuit breaker: {drawdown:.1%}")
        elif drawdown >= self.limits.drawdown_warning:
            logger.warning(f"Drawdown warning: {drawdown:.1%}")

    def _trigger_halt(self, reason: str):
        """Trigger emergency halt."""
        self.state.is_halted = True
        self.state.halt_reason = reason
        logger.error(f"RISK HALT TRIGGERED: {reason}")

        if self.on_halt:
            try:
                self.on_halt(reason)
            except Exception as e:
                logger.error(f"Halt callback error: {e}")

    def resume(self, confirm: bool = False):
        """
        Resume trading after halt.

        Args:
            confirm: Must be True to actually resume
        """
        if not confirm:
            logger.warning("Resume requires confirm=True")
            return

        with self._lock:
            self.state.is_halted = False
            self.state.halt_reason = None
            self.state.consecutive_losses = 0
            logger.info("Trading resumed - risk state cleared")

    def heartbeat(self) -> bool:
        """
        Heartbeat check - call periodically to verify system health.

        Returns True if healthy, False if stale.
        """
        now = datetime.utcnow()
        self.state.last_heartbeat = now

        # Check for stale state (no activity for too long could indicate issues)
        # This is a simple implementation - in production would check more conditions
        return not self.state.is_halted

    def reset_daily(self):
        """Reset daily counters (call at start of trading day)."""
        with self._lock:
            self.state.today_pnl = 0.0
            self.state.today_trades = 0
            self.state.hourly_trades = 0
            self.state.hour_start = datetime.utcnow()
            logger.info("Daily risk counters reset")

    def get_status(self) -> Dict[str, Any]:
        """Get current risk status."""
        with self._lock:
            drawdown = (self.state.initial_capital - self.state.current_capital) / self.state.initial_capital

            return {
                "is_halted": self.state.is_halted,
                "halt_reason": self.state.halt_reason,
                "capital": self.state.current_capital,
                "today_pnl": self.state.today_pnl,
                "today_trades": self.state.today_trades,
                "hourly_trades": self.state.hourly_trades,
                "consecutive_losses": self.state.consecutive_losses,
                "total_exposure": self.state.total_exposure,
                "drawdown": drawdown,
                "open_positions": len(self.state.open_positions),
                "limits": {
                    "max_daily_loss": self.limits.max_daily_loss,
                    "max_position_size": self.limits.max_position_size,
                    "drawdown_halt": self.limits.drawdown_halt,
                }
            }

    def _log_check(self, check_type: str, result: RiskCheckResult, reason: str):
        """Log risk check for audit trail."""
        self._check_history.append((datetime.utcnow(), check_type, result))

        # Keep only recent history
        if len(self._check_history) > 1000:
            self._check_history = self._check_history[-500:]


def get_risk_manager(
    initial_capital: float = 10000.0,
    on_halt: Optional[Callable[[str], None]] = None
) -> RiskManager:
    """
    Get a configured risk manager.

    Args:
        initial_capital: Starting capital
        on_halt: Callback when trading is halted

    Returns:
        Configured RiskManager instance
    """
    limits = RiskLimits()
    state = RiskState(initial_capital=initial_capital, current_capital=initial_capital)
    return RiskManager(limits=limits, state=state, on_halt=on_halt)


if __name__ == "__main__":
    # Test the risk manager
    def on_halt(reason):
        print(f"HALTED: {reason}")

    rm = get_risk_manager(initial_capital=10000, on_halt=on_halt)

    print("Testing pre-trade checks...")

    # Normal trade - should pass
    result, reason = rm.pre_trade_check("market1", "buy", 100, 0.5)
    print(f"Normal trade: {result.value} - {reason}")

    # Fat finger - too small
    result, reason = rm.pre_trade_check("market2", "buy", 5, 0.5)
    print(f"Too small: {result.value} - {reason}")

    # Fat finger - too large
    result, reason = rm.pre_trade_check("market3", "buy", 2000, 0.5)
    print(f"Too large: {result.value} - {reason}")

    print("\nStatus:", rm.get_status())
