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

#using this strategy for testing!!
from ES_2025_05_Statistical_Grid import StrategyTester

logger.remove()
logger.add(sys.stderr, level="INFO")

# ---------------------------------------------
# CONFIGURATION (set these variables externally)
# ---------------------------------------------
symbol = "GBPNZD"
timeframe_str = "M1"
lot = 1
magic = 8989891
# parameters = strategy_params["Statistical_Grid"].copy()
#get parameter set from the json!!!!! But could be deleted so creating a back up commented below: 39755a57-4ca5-46df-940a-c492c990e760
statistical_grid_GBPNZD_H1_PO_BEST_MAX_DD = {
    "active_trading_days": [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ],
    "trading_start_hh": 0,
    "trading_start_mm": 0,
    "trading_end_hh": 23,
    "trading_end_mm": 59,
    "grid_sma_value": 22,
    "grid_std_multiplier": 1.7000000000000004,
    "tp": 75,
    "sl": 50,
    "break_even_pips": 1000000,
    "cost_model": "ic_markets_raw",
    "spread_pips": 0.1,
    "commission_per_lot": 7.0,
    "risk_per_trade": 0.005,
    "multiple_positions": True,
    "lot_sizing_mode": "percent_equity",
    "symbol": "GBPNZD",
    "timeframe": "H1",
    "strategy": "Statistical_Grid",
    "asset": "GBPNZD"
}

parameters = statistical_grid_GBPNZD_H1_PO_BEST_MAX_DD.copy()


# ---------------------------------------------
# INITIALIZATION
# ---------------------------------------------
mt5_int = MT5Interface(broker="icmarkets")

logger.info(f"{symbol} pip value: {mt5_int.get_pip_value(symbol)}")

account_info = mt5.account_info()
if account_info:
    logger.info("Connected to MT5 account: {} (Equity: ${:.2f})", account_info.login, account_info.equity)
else:
    logger.warning("!!!! Unable to fetch account info. Proceeding anyway.")

account_size = mt5_int.get_account_equity()

# ---------------------------------------------
# Supporting Functions
# ---------------------------------------------
def safe_live_lot(params, account_size, sl_pips, pip_value, symbol):
    info = mt5.symbol_info(symbol)
    min_lot = info.volume_min if info else 0.01
    max_lot = info.volume_max if info else 100
    step = info.volume_step if info else 0.01

    # Standard risk-based calculation
    mode = params.get("lot_sizing_mode", "fixed_lot")
    if mode == "fixed_lot":
        preferred_lot = params.get("fixed_lot", min_lot)
    elif mode == "percent_equity":
        risk_per_trade = account_size * params["risk_per_trade"]
        preferred_lot = risk_per_trade / (sl_pips * pip_value * 10)
    elif mode == "cash_risk":
        risk_per_trade = params["cash_risk"]
        preferred_lot = risk_per_trade / (sl_pips * pip_value * 10)
    else:
        raise ValueError("Invalid lot sizing mode")

    # Snap to step, cap at broker max/min
    lot = round(max(min_lot, min(max_lot, round(preferred_lot / step) * step)), 2)

    # Logging for transparency
    if preferred_lot > max_lot:
        logger.warning(f"Risk calculation lot ({preferred_lot:.2f}) exceeds broker max ({max_lot}). Using max allowed lot: {max_lot}")
    if preferred_lot < min_lot:
        logger.warning(f"Risk calculation lot ({preferred_lot:.2f}) below broker min ({min_lot}). Using min allowed lot: {min_lot}")

    return lot

def get_timeframe_str(timeframe):
    timeframe_mapping = {
        mt5.TIMEFRAME_M1: "M1",
        mt5.TIMEFRAME_M2: "M2",
        mt5.TIMEFRAME_M3: "M3",
        mt5.TIMEFRAME_M4: "M4",
        mt5.TIMEFRAME_M5: "M5",
        mt5.TIMEFRAME_M6: "M6",
        mt5.TIMEFRAME_M10: "M10",
        mt5.TIMEFRAME_M12: "M12",
        mt5.TIMEFRAME_M15: "M15",
        mt5.TIMEFRAME_M20: "M20",
        mt5.TIMEFRAME_M30: "M30",
        mt5.TIMEFRAME_H1: "H1",
        mt5.TIMEFRAME_H2: "H2",
        mt5.TIMEFRAME_H3: "H3",
        mt5.TIMEFRAME_H4: "H4",
        mt5.TIMEFRAME_H6: "H6",
        mt5.TIMEFRAME_H8: "H8",
        mt5.TIMEFRAME_H12: "H12",
        mt5.TIMEFRAME_D1: "D1",
        mt5.TIMEFRAME_W1: "W1",
        mt5.TIMEFRAME_MN1: "MN1"
    }

    return timeframe_mapping.get(timeframe, "UnknownTimeframe")

# ---------------------------------------------
# MAIN LOOP (BAR-CLOSING EXECUTION)
# ---------------------------------------------
timeframe = get_timeframe_enum(timeframe_str)
last_candle = mt5_int.get_last_closed_candle(symbol=symbol, timeframe=timeframe)
last_bar_time = None
logger.info("Live trading started for {} on timeframe {}", symbol, timeframe_str)

def main_loop():
    global last_bar_time
    try:
        # Existing logic to fetch rates and create dataframe
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 1000)
        if rates is None or len(rates) == 0:
            logger.warning("No rate data fetched. Retrying in 10s...")
            std_time.sleep(10)
            return

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)

        latest_time = df.index[-2]
        if latest_time == last_bar_time:
            std_time.sleep(5)
            return

        # Existing candle logging
        latest_candle = df.iloc[-2]
        logger.info(
            "New Closed Candle - Time: {} | O: {:.5f} H: {:.5f} L: {:.5f} C: {:.5f} V: {}",
            latest_time.strftime("%Y-%m-%d %H:%M"),
            latest_candle["open"], latest_candle["high"],
            latest_candle["low"], latest_candle["close"], int(latest_candle["tick_volume"])
        )

        # Existing strategy initialization
        strategy = StrategyTester(df.copy(), parameters, account_size=account_size)

        # Fetch open positions
        open_positions = mt5_int.get_open_positions(symbol=symbol, magic=magic)

        # Execute exits
        strategy.get_exit_signal(latest_time)

        # Handle entry signals
        if not open_positions or parameters.get("multiple_positions", False):
            signal, entry_time = strategy.get_entry_signal(latest_time)
            if signal != 0:
                direction = "buy" if signal == 1 else "sell"

                #Override position size for live trades:
                live_lot = safe_live_lot(parameters, account_size, parameters["sl"], get_pip_value(symbol), symbol)
                logger.info("Live Entry Signal: {} at {} with calculated lot size: {}", direction.upper(), entry_time, live_lot)

                # Submit order using live calculated lot size
                mt5_int.submit_order(
                    symbol=symbol,
                    direction=direction,
                    lot=live_lot,
                    sl_pips=parameters["sl"],
                    tp_pips=parameters["tp"],
                    magic=magic
                )
            else:
                logger.info("No valid signal on this bar.")

        last_bar_time = latest_time
        std_time.sleep(10)

    except Exception as e:
        logger.exception("Exception in live loop: {}", e)
        std_time.sleep(10)



if __name__ == "__main__":
    while True:
        main_loop()
