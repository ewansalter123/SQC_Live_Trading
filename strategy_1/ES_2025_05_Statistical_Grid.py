######################################
#        EXTERNAL REQUIREMENTS       #
######################################
import os
import sys
import time as time_module
from datetime import datetime
import warnings
import pandas as pd
import numpy as np
from loguru import logger
warnings.filterwarnings("ignore")
import gc

######################################
#        INTERNAL REQUIREMENTS       #
######################################
from Quant_Backend.backtest_config import (
    strategy_params, IN_SAMPLE_START, IN_SAMPLE_END,
    get_pip_value, find_price_file, load_data,
    get_safe_n_jobs, DATA_DIR
)
from Quant_Backend.data_preprocessing import *
from Quant_Backend.trade import Trade

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Root directory of project

######################################
#        GLOBAL CONFIGURATION        #
######################################
params = strategy_params["Statistical_Grid"]  #  Replace with your actual strategy key
strat_name = next(name for name, p in strategy_params.items() if p is params)

######################################
#          STRATEGY CLASS            #
######################################
class StrategyTester:

    def __init__(self, data, parameters, account_size=10000):
        ######################################
        #         BASIC INITIALIZATION       #
        ######################################
        self.data = data  # ⬅️ OHLCV data
        self.parameters = parameters
        self.account_size = account_size
        self.current_balance = account_size
        self.position_size = 1.0
        self.output_dictionary = parameters.copy()

        ######################################
        #       TRADE MANAGEMENT FLAGS       #
        ######################################
        self.multiple_positions = self.parameters.get("multiple_positions", False)
        self.open_trades = [] if self.multiple_positions else None
        self.current_trade = None
        self.closed_trade = None
        self.trade_id_counter = 0

        ######################################
        #       TRADING WINDOW SETTINGS      #
        ######################################
        self.trading_start_hh = parameters["trading_start_hh"]
        self.trading_start_mm = parameters["trading_start_mm"]
        self.trading_end_hh = parameters["trading_end_hh"]
        self.trading_end_mm = parameters["trading_end_mm"]
        self.active_trading_days = parameters['active_trading_days']

        ######################################
        #         STRATEGY-SPECIFIC PARAMS   #
        ######################################
        # self.calculation_time_hh = parameters["calculation_time_hh"]
        # self.calculation_time_mm = parameters["calculation_time_mm"]
        # self.magic_percent = parameters["magic_percent"]
        self.grid_sma_value = parameters.get("grid_sma_value", 20)
        self.grid_std_multiplier = parameters.get("grid_std_multiplier", 1.5)
        self.move_to_be = self.parameters.get("move_to_be", False)
        self.break_even_pips = self.parameters.get("break_even_pips", 500) if self.move_to_be else None
        logger.debug(f"break_even_pips from config = {self.break_even_pips}")
        self.rsi_period = self.parameters.get("rsi_period", 14)
        self.rsi_upper = self.parameters.get("rsi_upper", 60)
        self.rsi_lower = self.parameters.get("rsi_lower", 40)
        self.sma_1 = self.parameters.get("sma_1", 200)

        ######################################
        #         COST MODEL SETTINGS        #
        ######################################
        self.cost_model = parameters["cost_model"]
        self.spread_pips = parameters["spread_pips"]
        self.commission_per_lot = parameters["commission_per_lot"]

        ######################################
        #         RISK & SL/TP SETTINGS      #
        ######################################
        symbol = self.parameters.get("symbol")
        self.pip_value = get_pip_value(symbol) or get_pip_value(asset)
        logger.debug(f"[DEBUG INIT] Asset: {symbol} | Pip Value: {self.pip_value}")
        self.tp_pips = parameters["tp"]
        self.sl_pips = parameters["sl"]
        self.risk_per_trade = self.parameters.get("risk_per_trade", 0.01)

        ######################################
        #     BACKTEST LOGGING + FEATURES    #
        ######################################
        logger.debug("Initializing StrategyTester with parameters: {}", parameters)
        logger.debug(
            "Backtest starting with account size: ${}, pip value: {}, SL (pips): {}, TP (pips): {}",
            self.account_size, self.pip_value, self.sl_pips, self.tp_pips
        )

        self.get_features()  # Compute indicators and signal column
        self.start_date_backtest = self.data.index[0]

        ######################################
        #      STATE TRACKING VARIABLES      #
        ######################################
        self.buy, self.sell = False, False
        self.entry_time, self.exit_time = None, None
        self.sl_price, self.tp_price = None, None

    def get_features(self):
        """Compute features and setup signals for this strategy."""
        logger.debug(" Starting feature engineering for strategy...")

        try:
            ######################################
            #      FILTER BY TRADING HOURS       #
            ######################################
            self.data = set_trading_hours(
                self.data,
                self.trading_start_hh,
                self.trading_start_mm,
                self.trading_end_hh,
                self.trading_end_mm
            )
            logger.debug(" Trading hours filtered from {:02d}:{:02d} to {:02d}:{:02d}",
                         self.trading_start_hh, self.trading_start_mm,
                         self.trading_end_hh, self.trading_end_mm)

            self.data = is_active_trading_day(self.data, self.active_trading_days)

            ######################################
            #    add_grid_statistical_features   #
            ######################################
            self.data = add_grid_statistical_features(
                self.data,
                self.grid_sma_value,
                self.grid_std_multiplier
            )
            # logger.debug(" Previous day close prices added at {:02d}:{:02d}",
            #              self.calculation_time_hh, self.calculation_time_mm)

            self.data = rsi(self.data, self.data['close'], n=self.rsi_period)

            self.data = sma(self.data, "close", self.sma_1)

            ######################################
            #     INIT STATE TRACKING COLUMNS    #
            ######################################
            self.data["break_even_triggered"] = False
            logger.debug(" break_even_triggered column forced to False for new run.")

            ######################################
            #         FINAL SIGNAL COLUMN        #
            ######################################
            self.data["signal"] = 0

            buy_condition = (
                    (self.data['ActiveTradingHour']) &
                    (self.data["ActiveTradingDay"]) &
                    (self.data['close'] < self.data['grid_lower_1']) &
                    (self.data['close'] > self.data['SMA_200'])
            )

            sell_condition = (
                    (self.data['ActiveTradingHour']) &
                    (self.data["ActiveTradingDay"]) &
                    (self.data['close'] > self.data['grid_upper_1']) &
                    (self.data['close'] < self.data['SMA_200'])
            )

            self.data.loc[buy_condition, "signal"] = 1
            self.data.loc[sell_condition, "signal"] = -1

            logger.debug(" Final signal column created: Buys={}, Sells={}",
                         (self.data['signal'] == 1).sum(),
                         (self.data['signal'] == -1).sum())

        except Exception as e:
            ######################################
            #         ERROR HANDLING LOGIC       #
            ######################################
            logger.exception(" Feature generation failed: {}", e)
            raise

    def get_entry_signal(self, time):
        ######################################
        #         ENTRY VALIDATION           #
        ######################################
        if len(self.data.loc[:time]) < 2:
            return 0, None  # Not enough history yet

        signal_value = self.data.loc[:time]["signal"][-2]  # Use previous bar's signal
        if signal_value == 0 or (not self.multiple_positions and self.current_trade):
            return 0, None  # No new trade if flat signal or already in trade (single trade mode)

        ######################################
        #       TRADE INITIALIZATION         #
        ######################################
        direction = "long" if signal_value == 1 else "short"
        entry_price = self.data.loc[time]["open"]

        self.trade_id_counter += 1

        #Calculate position size using pip value, SL distance, and risk %        # Calculate pip value per lot (standard = 100,000 units for FX, adjust for other assets if needed)
        unit_lot_size = 100000  #  You can set this via config or per asset for more flexibility
        # Accurate pip value calculation (quote currency per pip per lot)
        pip_value_per_lot = (self.pip_value / entry_price) * unit_lot_size
        # Total cash risk per trade
        sl_cash = self.sl_pips * pip_value_per_lot
        cash_risk = self.current_balance * self.risk_per_trade
        # Position size in lots (fractional lots supported)
        position_size = cash_risk / sl_cash if sl_cash > 0 else 0

        # logger.info(
        #     f"Trade {self.trade_id_counter}: entry_price={entry_price}, pip_value={self.pip_value}, sl_pips={self.sl_pips}, "
        #     f"pip_value_per_lot={pip_value_per_lot}, sl_cash={sl_cash}, cash_risk={cash_risk}, position_size={position_size}"
        # )

        new_trade = Trade(
            trade_id=self.trade_id_counter,
            direction=direction,
            entry_time=time,
            entry_price=entry_price,
            tp=self.tp_pips,
            sl=self.sl_pips,
            position_size=position_size,
            trailing_stop_pct=None,
            account_size=self.account_size,
            pip_value=self.pip_value,
            break_even_pips=self.break_even_pips,
            risk_per_trade=self.risk_per_trade,
        )

        ######################################
        #     TRADE DEBUG LOGGING FUNCTION   #
        ######################################
        def log_entry_trade():
            sl_price = (
                entry_price - self.sl_pips * self.pip_value if direction == "long"
                else entry_price + self.sl_pips * self.pip_value
            )
            tp_price = (
                entry_price + self.tp_pips * self.pip_value if direction == "long"
                else entry_price - self.tp_pips * self.pip_value
            )
            symbol_name = self.parameters.get("symbol", "UNKNOWN")

            logger.debug("\n🔵" + "=" * 60)
            logger.debug(f"[ENTRY DEBUG] Trade #{self.trade_id_counter}")
            logger.debug("-" * 60)
            logger.debug(f"Symbol              : {symbol_name}")
            logger.debug(f"Time                : {time}")
            logger.debug(f"Direction           : {direction}")
            logger.debug(f"Entry Price         : {entry_price:.5f}")
            logger.debug(
                f"SL (pips)           : {self.sl_pips} → Δ {self.sl_pips * self.pip_value:.5f} → SL Price: {sl_price:.5f}")
            logger.debug(
                f"TP (pips)           : {self.tp_pips} → Δ {self.tp_pips * self.pip_value:.5f} → TP Price: {tp_price:.5f}")
            logger.debug(f"Pip Value           : {self.pip_value}")
            logger.debug(f"Account Size        : {self.account_size}")
            logger.debug(f"Risk per Trade      : {self.risk_per_trade * 100:.1f}%")
            logger.debug(f"Cash Risk           : {cash_risk:.2f}")
            logger.debug(f"SL Cash             : {sl_cash:.2f}")
            logger.debug(f"Position Size       : {position_size:.2f}")
            logger.debug("=" * 60 + "\n")

        ######################################
        #         TRADE ASSIGNMENT           #
        ######################################
        if self.multiple_positions:
            self.open_trades.append(new_trade)
            if debug_trades:
                log_entry_trade()
        else:
            self.current_trade = new_trade
            if debug_trades:
                log_entry_trade()

        ######################################
        #         ENTRY CONFIRMATION LOG     #
        ######################################
        if self.current_trade:
            move_in_pips = (
                self.data.loc[time, "high"] - self.current_trade.entry_price
                if self.current_trade.direction == "long"
                else self.current_trade.entry_price - self.data.loc[time, "low"]
            ) / self.current_trade.pip_value

            logger.debug(
                "[DEBUG BE] Entry: {:.5f} | Move in pips: {:.2f} | Trigger: {} | Dir: {}",
                self.current_trade.entry_price,
                move_in_pips,
                self.current_trade.break_even_pips,
                self.current_trade.direction
            )

            logger.debug(
                " Trade #{:04d} opened | Dir: {} | Price: {:.5f} | Size: {:.2f} | SL: {} pips | TP: {} pips",
                self.current_trade.trade_id,
                direction,
                entry_price,
                position_size,
                self.sl_pips,
                self.tp_pips
            )

        return signal_value, time

    def get_exit_signal(self, time):
        ######################################
        #       FETCH CURRENT PRICE ROW      #
        ######################################
        row = self.data.loc[time]

        ######################################
        #          DEBUG EXIT LOGGER         #
        ######################################
        def log_exit_trade(trade, time, net_pnl):
            symbol_name = self.parameters.get("symbol", "UNKNOWN")
            entry_price = trade.entry_price
            exit_price = trade.exit_price
            exit_reason = trade.exit_reason.value
            gross_pnl = trade.pnl()
            r_multiple = trade.r_multiple()
            be_triggered = trade.break_even_triggered

            logger.debug("\n" + "=" * 60)
            logger.debug(f"[EXIT DEBUG] Trade #{trade.trade_id}")
            logger.debug("-" * 60)
            logger.debug(f"Symbol              : {symbol_name}")
            logger.debug(f"Time                : {time}")
            logger.debug(f"Direction           : {trade.direction}")
            logger.debug(f"Entry Price         : {entry_price:.5f}")
            logger.debug(f"Exit Price          : {exit_price:.5f}")
            logger.debug(f"Exit Reason         : {exit_reason}")
            logger.debug(f"Gross PnL           : ${gross_pnl:.2f}")
            logger.debug(f"Net PnL (after cost): ${net_pnl:.2f}")
            logger.debug(f"R-Multiple          : {r_multiple:.2f}")
            logger.debug(f"Break-even Triggered: {be_triggered}")
            logger.debug("=" * 60 + "\n")

        ######################################
        #        MULTIPLE TRADE MODE         #
        ######################################
        if self.multiple_positions:
            for trade in self.open_trades[:]:  # Safe iteration while modifying list
                # 🟨 Check break-even if enabled
                if self.break_even_pips and not trade.break_even_triggered:
                    trade.check_break_even_trigger(
                        high=row["high"],
                        low=row["low"],
                        pip_value=self.pip_value
                    )

                #  Update trade with current price action
                trade.update(
                    time,
                    row["high"],
                    row["low"],
                    row.get("high_time"),
                    row.get("low_time")
                )

                #  Check for exit condition
                if not trade.is_active():
                    gross_pnl = trade.pnl()
                    net_pnl = gross_pnl - self.compute_cost(trade.position_size)
                    r = trade.r_multiple()

                    # 🧾 Log exit
                    if debug_trades:
                        log_exit_trade(trade, time, net_pnl)
                    else:
                        logger.debug(
                            f"[ Trade Exit] Trade #{trade.trade_id} | Exit Time: {time} | "
                            f"Reason: {trade.exit_reason.value} | Gross PnL: ${gross_pnl:.2f} | "
                            f"Net PnL: ${net_pnl:.2f} | R: {r:.2f}"
                        )

                    #  Mark break-even if triggered
                    if trade.break_even_triggered and time in self.data.index:
                        self.data.at[time, "break_even_triggered"] = True

                    self.closed_trade = trade
                    self.open_trades.remove(trade)
                    return net_pnl, time

            return 0, None  # No trades exited this bar

        ######################################
        #         SINGLE TRADE MODE          #
        ######################################
        if not self.current_trade:
            return 0, None  # No open trade to manage

        # Check break-even trigger
        if self.break_even_pips and not self.current_trade.break_even_triggered:
            self.current_trade.check_break_even_trigger(
                high=row["high"],
                low=row["low"],
                pip_value=self.pip_value
            )

        #  Update trade with latest OHLC
        self.current_trade.update(
            time,
            row["high"],
            row["low"],
            row.get("high_time"),
            row.get("low_time")
        )

        #  Trade closed condition met
        if not self.current_trade.is_active():
            gross_pnl = self.current_trade.pnl()
            net_pnl = gross_pnl - self.compute_cost(self.current_trade.position_size)
            r = self.current_trade.r_multiple()

            if debug_trades:
                log_exit_trade(self.current_trade, time, net_pnl)
            else:
                logger.debug(
                    f"[ Trade Exit] Trade #{self.current_trade.trade_id} | Exit Time: {time} | "
                    f"Reason: {self.current_trade.exit_reason.value} | Gross PnL: ${gross_pnl:.2f} | "
                    f"Net PnL: ${net_pnl:.2f} | R: {r:.2f}"
                )

            if self.current_trade.break_even_triggered and time in self.data.index:
                self.data.at[time, "break_even_triggered"] = True

            self.closed_trade = self.current_trade
            self.current_trade = None
            return net_pnl, time

        return 0, None  # Trade still open

    def compute_cost(self, position_size: float) -> float:
        """
        Computes total cost per trade including spread and commission.

        Args:
            position_size (float): Trade size in units (e.g., 12,943 units).

        Returns:
            float: Total cost in USD for this trade.
        """

        ######################################
        #     IC MARKETS RAW COST MODEL      #
        ######################################
        if self.cost_model == "ic_markets_raw":
            spread_cost = self.spread_pips * self.pip_value * position_size
            commission_cost = (position_size / 100_000) * self.commission_per_lot
            total_cost = spread_cost + commission_cost
            return total_cost

        ######################################
        #           FLAT FEE MODEL           #
        ######################################
        elif self.cost_model == "flat":
            return self.cost_value  #  Flat cost must be defined in strategy config

        ######################################
        #       PER UNIT FEE MODEL           #
        ######################################
        elif self.cost_model == "per_unit":
            return position_size * self.cost_value  # Cost scales linearly with size

        ######################################
        #            FALLBACK CASE           #
        ######################################
        else:
            raise ValueError(f"Invalid cost model: {self.cost_model}")

######################################
# CREATE INSTANCE OF STRATEGY USING DEFINED PARAMS AND OHLC DATA
######################################
# params["symbol"] = assets_to_test[0]  # ensure it's injected early
# ST = StrategyTester(in_sample_df, params)

# WILL LOG ALL TRADES AND METRICS - CHECK ENTRY AND EXIT SIGNAL
debug_trades = False
