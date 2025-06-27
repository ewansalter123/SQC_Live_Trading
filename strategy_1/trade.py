from enum import Enum
from datetime import datetime, timedelta
import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")  # Change to "ERROR" or "CRITICAL" to silence more


class ExitReason(Enum):
    NONE = "none"
    TP = "take_profit"
    SL = "stop_loss"
    BE = "break_even"
    MANUAL = "manual"
    TIME = "time_exit"
    TRAIL = "trailing_stop"

class Trade:
    def __init__(self, trade_id, direction, entry_time, entry_price,
                 tp=None, sl=None, position_size=1.0, trailing_stop_pct=None, account_size=10000,
                 pip_value=0.0001, break_even_pips=15, risk_per_trade=None):
        self.trade_id = trade_id
        self.direction = direction  # 'long' or 'short'
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.tp = tp
        self.sl = sl
        self.position_size = position_size
        self.trailing_stop_pct = trailing_stop_pct
        self.account_size = account_size
        self.pip_value = pip_value
        self.break_even_pips = break_even_pips
        self.risk_per_trade = risk_per_trade

        self.tp_price = None
        self.sl_price = None
        self.break_even_price = entry_price
        self.break_even_triggered = False

        self.exit_price = None
        self.exit_time = None
        self.exit_reason = ExitReason.NONE
        self.active = True

        self._set_tp_sl_levels()

        logger.debug(
            f"[TRADE OPENED] ID: {self.trade_id} | Dir: {self.direction} | Entry: {self.entry_price:.5f} "
            f"| Time: {self.entry_time} | Size: {self.position_size:.2f} | TP: {self.tp_price:.5f} | SL: {self.sl_price:.5f} "
            f"| BE Trigger: {self.break_even_pips} pips | Pip Value: {self.pip_value:.5f} | Account Size: ${self.account_size:,.2f}"
        )

    def _set_tp_sl_levels(self):
        if self.tp is not None:
            self.tp_price = (self.entry_price + self.tp * self.pip_value
                             if self.direction == "long"
                             else self.entry_price - self.tp * self.pip_value)

        if self.sl is not None:
            self.sl_price = (self.entry_price - self.sl * self.pip_value
                             if self.direction == "long"
                             else self.entry_price + self.sl * self.pip_value)

        tp_str = f"{self.tp_price:.5f}" if self.tp_price is not None else "None"
        sl_str = f"{self.sl_price:.5f}" if self.sl_price is not None else "None"

        logger.debug(
            f"[SET TP/SL] Trade ID: {self.trade_id} | Direction: {self.direction} | "
            f"Entry: {self.entry_price:.5f} | TP: {tp_str} | SL: {sl_str}"
        )

    def update(self, time, high, low, high_time=None, low_time=None):
        if not self.active:
            return

        entry_str = f"{self.entry_price:.5f}" if self.entry_price is not None else "None"
        tp_str = f"{self.tp_price:.5f}" if self.tp_price is not None else "None"
        sl_str = f"{self.sl_price:.5f}" if self.sl_price is not None else "None"

        logger.debug(
            f"[UPDATE] Trade ID: {self.trade_id} | Dir: {self.direction} | Entry: {entry_str} | "
            f"TP: {tp_str} | SL: {sl_str} | Active: {self.active}"
        )

        if self.direction == 'long':
            if self.tp_price and self.sl_price and high >= self.tp_price and low <= self.sl_price:
                if high_time and low_time:
                    if high_time <= low_time:
                        logger.debug(f"[HIT BOTH] Trade ID: {self.trade_id} | TP hit first at {high_time}")
                        self.close(high_time, self.tp_price, ExitReason.TP)
                    else:
                        reason = ExitReason.BE if self.break_even_triggered else ExitReason.TRAIL if self.trailing_stop_pct else ExitReason.SL
                        logger.debug(
                            f"[HIT BOTH] Trade ID: {self.trade_id} | SL hit first at {low_time} | Reason: {reason}")
                        self.close(low_time, self.sl_price, reason)
            elif self.tp_price and high >= self.tp_price:
                logger.debug(f"[TP HIT] Trade ID: {self.trade_id} | TP: {self.tp_price:.5f} | Time: {time}")
                self.close(time, self.tp_price, ExitReason.TP)
            elif self.sl_price and low <= self.sl_price:
                reason = ExitReason.BE if self.break_even_triggered else ExitReason.TRAIL if self.trailing_stop_pct else ExitReason.SL
                logger.debug(
                    f"[SL HIT] Trade ID: {self.trade_id} | SL: {self.sl_price:.5f} | Time: {time} | Reason: {reason}")
                self.close(time, self.sl_price, reason)

        elif self.direction == 'short':
            if self.tp_price and self.sl_price and low <= self.tp_price and high >= self.sl_price:
                if high_time and low_time:
                    if low_time <= high_time:
                        logger.debug(f"[HIT BOTH] Trade ID: {self.trade_id} | TP hit first at {low_time}")
                        self.close(low_time, self.tp_price, ExitReason.TP)
                    else:
                        reason = ExitReason.BE if self.break_even_triggered else ExitReason.TRAIL if self.trailing_stop_pct else ExitReason.SL
                        logger.debug(
                            f"[HIT BOTH] Trade ID: {self.trade_id} | SL hit first at {high_time} | Reason: {reason}")
                        self.close(high_time, self.sl_price, reason)
            elif self.tp_price and low <= self.tp_price:
                logger.debug(f"[TP HIT] Trade ID: {self.trade_id} | TP: {self.tp_price:.5f} | Time: {time}")
                self.close(time, self.tp_price, ExitReason.TP)
            elif self.sl_price and high >= self.sl_price:
                reason = ExitReason.BE if self.break_even_triggered else ExitReason.TRAIL if self.trailing_stop_pct else ExitReason.SL
                logger.debug(
                    f"[SL HIT] Trade ID: {self.trade_id} | SL: {self.sl_price:.5f} | Time: {time} | Reason: {reason}")
                self.close(time, self.sl_price, reason)

    def close(self, time, price, reason):
        self.exit_time = time
        self.exit_price = price
        self.exit_reason = reason
        self.active = False

        entry_str = f"{self.entry_price:.5f}" if self.entry_price is not None else "None"
        exit_str = f"{self.exit_price:.5f}" if self.exit_price is not None else "None"
        size_str = f"{self.position_size:.2f}" if self.position_size is not None else "None"

        logger.debug(
            f"[TRADE CLOSED] ID: {self.trade_id} | Dir: {self.direction} | Entry: {entry_str} "
            f"| Exit: {exit_str} | Size: {size_str} | Reason: {self.exit_reason}"
        )

    def is_active(self):
        if not self.active:
            logger.debug(
                f"[TRADE CHECK] Trade ID: {self.trade_id} is no longer active. "
                f"Closed at: {self.exit_time} | Reason: {self.exit_reason.value}"
            )
        return self.active

    def pnl(self):
        if not self.exit_price:
            logger.warning(
                f"[PNL WARNING] Trade ID: {self.trade_id} | Attempted PnL calc before trade was closed."
            )
            return 0

        pip_diff = (self.exit_price - self.entry_price) / self.pip_value if self.direction == "long" else (
                                                                                                                      self.entry_price - self.exit_price) / self.pip_value
        pip_value_per_lot = (self.pip_value / self.entry_price) * 100_000  # You can make lot size configurable

        raw_pnl = pip_diff * self.position_size * pip_value_per_lot

        logger.debug(
            f"[PNL] Trade ID: {self.trade_id} | Dir: {self.direction} | Entry: {self.entry_price:.5f} | "
            f"Exit: {self.exit_price:.5f} | Size: {self.position_size:.2f} | Pips: {pip_diff:.2f} | PnL: ${raw_pnl:.2f}"
        )

        return raw_pnl

    def duration(self):
        duration = (self.exit_time - self.entry_time) if self.exit_time else None
        logger.debug(f"[DURATION] ID: {self.trade_id} | Duration: {duration}")
        return duration

    def to_dict(self):
        data = {
            "trade_id": self.trade_id,
            "direction": self.direction,
            "entry_time": self.entry_time,
            "entry_price": self.entry_price,
            "tp_price": self.tp_price,
            "sl_price": self.sl_price,
            "exit_time": self.exit_time,
            "exit_price": self.exit_price,
            "break_even_triggered": self.break_even_triggered,
            "exit_reason": self.exit_reason.value,
            "position_size": self.position_size,
            "pnl": self.pnl(),
            "duration": self.duration(),
            "implied_leverage": (self.position_size * self.entry_price) / self.account_size,
            "cash_pnl": self.pnl(),
            "pnl_in_pips": ((self.exit_price - self.entry_price) / self.pip_value) if self.exit_price else 0,
            "r_multiple": self.r_multiple(),

        }
        logger.debug(f"[TO_DICT] Trade Data: {data}")
        return data

    def check_break_even_trigger(self, high, low, pip_value):
        if self.break_even_triggered or self.break_even_pips is None:
            return False, 0

        if self.direction == "long":
            move = (high - self.entry_price) / pip_value
        elif self.direction == "short":
            move = (self.entry_price - low) / pip_value
        else:
            return False, 0

        if move >= self.break_even_pips:
            self.sl_price = self.entry_price
            self.break_even_triggered = True
            logger.debug(f"[BE TRIGGERED] ID: {self.trade_id} | Entry: {self.entry_price:.5f} | Move: {move:.2f} pips")
            return True, move

        logger.debug(f"[BE CHECK] ID: {self.trade_id} | Move: {move:.2f} pips | Threshold: {self.break_even_pips}")
        return False, move

    def r_multiple(self):
        if not self.exit_price or not self.active:
            return 0  # Trade hasn't closed yet
        gross_pnl = self.pnl()
        risk_cash = self.account_size * self.position_size * (self.sl * self.pip_value)
        if risk_cash == 0:
            return 0
        return gross_pnl / risk_cash
