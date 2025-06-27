import MetaTrader5 as mt5
import pandas as pd
import time as std_time
from datetime import datetime
from loguru import logger
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mt5_interface import MT5Interface
from data_preprocessing import *
from trade import Trade
from live_trading_config import get_timeframe_enum
from ES_2025_05_Statistical_Grid import StrategyTester

logger.remove()
logger.add(sys.stderr, level="INFO")

# ---------------------------------------------
# CONFIGURATION
# ---------------------------------------------
symbol = "GBPNZD"
timeframe_str = "M1"
magic = 44667732

statistical_grid_GBPNZD_H1_PO_BEST_MAX_DD = {
    "active_trading_days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    "trading_start_hh": 4,
    "trading_start_mm": 0,
    "trading_end_hh": 22,
    "trading_end_mm": 59,
    "grid_sma_value": 22,
    "grid_std_multiplier": 1.7,
    "tp": 75,
    "sl": 50,
    "break_even_pips": 1000000,
    "cost_model": "ic_markets_raw",
    "spread_pips": 0.1,
    "commission_per_lot": 7.0,
    "risk_per_trade": 0.005,
    "multiple_positions": True,
    "lot_sizing_mode": "percent_equity",
    "timeframe": "H1",
    "strategy": "Statistical_Grid",
}

parameters = statistical_grid_GBPNZD_H1_PO_BEST_MAX_DD.copy()

# ---------------------------------------------
# INITIALIZATION
# ---------------------------------------------
mt5_int = MT5Interface(broker="icmarkets")

logger.info(f"{symbol} pip value: {mt5_int.get_pip_value(symbol)}")
parameters["pip_value"] = mt5_int.get_pip_value(symbol)
account_info = mt5.account_info()
symbol_info = mt5.symbol_info(symbol)
logger.info(symbol_info)
if account_info:
    logger.info("Connected to MT5 account: {} (Equity: ${:.2f})", account_info.login, account_info.equity)
else:
    logger.warning("!!!! Unable to fetch account info. Proceeding anyway.")

account_size = mt5_int.get_account_equity()
timeframe = get_timeframe_enum(timeframe_str)

logger.info("Event-driven live trading started for {} on timeframe {}", symbol, timeframe_str)

# ---------------------------------------------
# SUPPORTING FUNCTION
# ---------------------------------------------
def get_timeframe_str(timeframe):
    timeframe_mapping = {
        mt5.TIMEFRAME_M1: "M1",
        mt5.TIMEFRAME_M5: "M5",
        mt5.TIMEFRAME_M15: "M15",
        mt5.TIMEFRAME_M30: "M30",
        mt5.TIMEFRAME_H1: "H1",
        mt5.TIMEFRAME_H4: "H4",
        mt5.TIMEFRAME_D1: "D1",
        mt5.TIMEFRAME_W1: "W1",
        mt5.TIMEFRAME_MN1: "MN1"
    }
    return timeframe_mapping.get(timeframe, "UnknownTimeframe")

# ---------------------------------------------
# EVENT-DRIVEN MAIN LOOP
# ---------------------------------------------
previous_minute = None

def run_live_trading():
    global previous_minute
    while True:
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            std_time.sleep(0.2)
            continue

        server_time = datetime.utcfromtimestamp(tick.time)
        current_minute = server_time.replace(second=0, microsecond=0)

        if current_minute != previous_minute:
            previous_minute = current_minute

            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 1000)
            if rates is None or len(rates) == 0:
                logger.warning("No rate data fetched. Skipping this bar.")
                continue

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.set_index("time", inplace=True)

            latest_time = df.index[-2]
            latest_candle = df.iloc[-2]

            logger.info(
                "New Closed Candle - Time: {} | O: {:.5f} H: {:.5f} L: {:.5f} C: {:.5f} V: {}",
                latest_time.strftime("%Y-%m-%d %H:%M"),
                latest_candle["open"], latest_candle["high"],
                latest_candle["low"], latest_candle["close"], int(latest_candle["tick_volume"])
            )

            strategy = StrategyTester(df.copy(), parameters, account_size=account_size)
            open_positions = mt5_int.get_open_positions(symbol=symbol, magic=magic)

            strategy.get_exit_signal(latest_time)

            if not open_positions or parameters.get("multiple_positions", False):
                signal, entry_time = strategy.get_entry_signal(latest_time)
                if signal != 0:
                    direction = "buy" if signal == 1 else "sell"
                    live_lot = mt5_int.calculate_lot_size(symbol, parameters["sl"], account_size, parameters["risk_per_trade"])
                    logger.info("Live Entry Signal: {} at {} with calculated lot size: {}", direction.upper(), entry_time, live_lot)
                    mt5_int.submit_order(symbol, direction, live_lot, parameters["sl"], parameters["tp"], magic)
                else:
                    logger.info("No valid signal on this bar.")

        std_time.sleep(0.2)

if __name__ == "__main__":
    run_live_trading()