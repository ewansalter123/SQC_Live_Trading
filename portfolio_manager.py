import MetaTrader5 as mt5
import pandas as pd
import time as std_time
from datetime import datetime
from loguru import logger
import sys
import os

# ---------------------------------------------
# PROJECT PATH SETUP
# ---------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'strategy_1')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'strategy_2')))

from mt5_interface import MT5Interface
from my_strategy import StrategyTester as Strategy1
from my_other_strategy import StrategyTester as Strategy2
from live_trading_config import get_timeframe_enum

# ---------------------------------------------
# CONFIGURATION
# ---------------------------------------------
strategies = [
    {
        "name": "Statistical Grid",
        "symbol": "GBPNZD",
        "timeframe_str": "M1",
        "magic": 111111,
        "strategy_class": Strategy1,
        "parameters": {
            "sl": 50,
            "tp": 75,
            "risk_per_trade": 0.005,
            "multiple_positions": True,
            # ...other params
        }
    },
    {
        "name": "Mean Reversion",
        "symbol": "EURUSD",
        "timeframe_str": "M5",
        "magic": 222222,
        "strategy_class": Strategy2,
        "parameters": {
            "sl": 30,
            "tp": 50,
            "risk_per_trade": 0.003,
            "multiple_positions": False,
            # ...other params
        }
    }
]

# ---------------------------------------------
# INITIALIZATION
# ---------------------------------------------
logger.remove()
logger.add(sys.stderr, level="INFO")

mt5_int = MT5Interface(broker="icmarkets")

for strat in strategies:
    strat["parameters"]["pip_value"] = mt5_int.get_pip_value(strat["symbol"])
    strat["account_size"] = mt5_int.get_account_equity()
    strat["timeframe"] = get_timeframe_enum(strat["timeframe_str"])
    logger.info(f"Initialized {strat['name']} for {strat['symbol']} on {strat['timeframe_str']} timeframe")

# ---------------------------------------------
# MAIN EVENT-DRIVEN LOOP
# ---------------------------------------------
previous_minutes = {s["symbol"]: None for s in strategies}

while True:
    for strat in strategies:
        symbol = strat["symbol"]
        timeframe = strat["timeframe"]
        magic = strat["magic"]
        params = strat["parameters"]

        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            continue

        server_time = datetime.utcfromtimestamp(tick.time)
        current_minute = server_time.replace(second=0, microsecond=0)

        if current_minute == previous_minutes[symbol]:
            continue  # Already processed this bar

        previous_minutes[symbol] = current_minute

        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 1000)
        if rates is None or len(rates) == 0:
            logger.warning(f"No rate data for {symbol}. Skipping.")
            continue

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)

        latest_time = df.index[-2]
        latest_candle = df.iloc[-2]

        logger.info(
            "{} | New Closed Candle | O: {:.5f} H: {:.5f} L: {:.5f} C: {:.5f} V: {}",
            symbol, latest_candle["open"], latest_candle["high"],
            latest_candle["low"], latest_candle["close"], int(latest_candle["tick_volume"])
        )

        strategy_obj = strat["strategy_class"](df.copy(), params, account_size=strat["account_size"])
        open_positions = mt5_int.get_open_positions(symbol=symbol, magic=magic)

        strategy_obj.get_exit_signal(latest_time)

        if not open_positions or params.get("multiple_positions", False):
            signal, entry_time = strategy_obj.get_entry_signal(latest_time)
            if signal != 0:
                direction = "buy" if signal == 1 else "sell"
                lot_size = mt5_int.calculate_lot_size(symbol, params["sl"], strat["account_size"], params["risk_per_trade"])
                logger.info("{} | Live Signal: {} at {} | Lot: {}", symbol, direction.upper(), entry_time, lot_size)
                mt5_int.submit_order(symbol, direction, lot_size, params["sl"], params["tp"], magic)

    std_time.sleep(0.2)
