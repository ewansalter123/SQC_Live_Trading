import MetaTrader5 as mt5
import pandas as pd
import time
from datetime import datetime
from loguru import logger
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Quant_Backend.mt5_interface import MT5Interface
from Quant_Backend.backtest_config import get_pip_value

# --- CONFIGURATION ---
symbol = "BTCUSD"
timeframe = 1  # 1-minute bars, maps to mt5.TIMEFRAME_M1
lot = 1
magic = 888888
sl_pips = 5000
tp_pips = 5000
broker = "icmarkets"

# --- INIT ---
logger.remove()
logger.add(sys.stderr, level="INFO")
mt5_int = MT5Interface(broker=broker)

pip_value = get_pip_value(symbol)
logger.info("Pip value for {} = {}", symbol, pip_value)

account = mt5_int.get_open_positions()  # just to confirm connection
account_info = mt5.account_info()
if account_info:
    logger.info("Connected to MT5 account: {} (Equity: ${:.2f})", account_info.login, account_info.equity)

# --- STATE ---
last_bar_time = None

info = mt5.symbol_info(symbol)
logger.info(f"Min lot: {info.volume_min}, Max lot: {info.volume_max}, Step: {info.volume_step}")


# --- LOOP ---
while True:
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 3)
        if rates is None or len(rates) < 2:
            logger.warning("No candle data. Retrying in 10s...")
            time.sleep(10)
            continue

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)

        bar = df.iloc[-2]
        bar_time = bar.name

        if bar_time == last_bar_time:
            time.sleep(5)
            continue

        last_bar_time = bar_time
        o, c = bar["open"], bar["close"]
        logger.info("New candle closed: {} | O: {:.2f}, C: {:.2f}", bar_time.strftime("%Y-%m-%d %H:%M"), o, c)

        direction = "sell" if c > o else "buy"
        logger.info("\U0001F4E2 Opening {} order based on candle reversal", direction.upper())

        mt5_int.submit_order(symbol=symbol, direction=direction, lot=lot, sl_pips=sl_pips, tp_pips=tp_pips, magic=magic)

        time.sleep(10)

    except Exception as e:
        logger.exception("Loop error: {}", e)
        time.sleep(10)
