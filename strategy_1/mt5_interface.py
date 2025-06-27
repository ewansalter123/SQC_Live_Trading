import MetaTrader5 as mt5
from datetime import datetime
from loguru import logger
import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
logger.remove()
logger.add(sys.stderr, level="INFO")

# Define known terminal paths (customize as needed)
MT5_PATHS = {
    "metaquotes": r"C:\\Program Files\\MetaTrader 5\\terminal64.exe",
    "icmarkets": r"C:\\Program Files\\MetaTrader 5 IC Markets Global\\terminal64.exe",
}


class MT5Interface:
    def __init__(self, broker: str = "metaquotes"):
        self.broker = broker
        # self.application = application
        self.initialized = False
        self.initialize_mt5()

    def initialize_mt5(self):
        path = MT5_PATHS.get(self.broker.lower())
        if not path:
            raise ValueError(f"Unknown broker '{self.broker}'. Available: {list(MT5_PATHS.keys())}")

        if not mt5.initialize(path=path):
            logger.error("MT5 initialization failed for {}: {}", self.broker, mt5.last_error())
            raise ConnectionError(f"Failed to initialize MetaTrader5 for {self.broker}")

        logger.info("MT5 initialized successfully for {}", self.broker)
        self.initialized = True

    def get_open_positions(self, symbol=None, magic=None):

        positions = mt5.positions_get()
        if positions is None:
            return []

        filtered = []
        for pos in positions:
            pos_dict = pos._asdict()
            if symbol and pos_dict['symbol'] != symbol:
                continue
            if magic and pos_dict['magic'] != magic:
                continue
            filtered.append(pos_dict)
        return filtered

    def get_account_equity(self):
        account = mt5.account_info()
        return account.equity if account else None

    def calculate_sl_tp_prices(self, symbol, direction, entry_price, sl_pips, tp_pips):
        info = mt5.symbol_info(symbol)
        if not info:
            logger.error(f"Failed to retrieve symbol info for {symbol}")
            return entry_price, entry_price  # Fallback to prevent crashing

        pip_size = info.point * 10  # 0.0001 for most FX pairs, 0.01 for Yen pairs

        if direction == "buy":
            sl_price = entry_price - sl_pips * pip_size
            tp_price = entry_price + tp_pips * pip_size
        else:
            sl_price = entry_price + sl_pips * pip_size
            tp_price = entry_price - tp_pips * pip_size

        return sl_price, tp_price

    def get_pip_value(self, symbol: str) -> float:
        """
        Returns the pip value in quote currency per lot, based on MT5 symbol info.
        """
        info = mt5.symbol_info(symbol)
        if not info:
            logger.error(f"Failed to retrieve symbol info for {symbol}")
            return 0.0

        if info.trade_tick_value <= 0 or info.trade_tick_size <= 0:
            logger.error(f"Invalid tick value or size for {symbol}")
            return 0.0

        pip_value = info.trade_tick_value * (0.0001 / info.trade_tick_size)

        logger.info(
            f"Pip value for {symbol}: {pip_value:.5f} (Tick Value: {info.trade_tick_value}, Tick Size: {info.trade_tick_size})")
        return pip_value

    def get_last_closed_candle(self, symbol: str, timeframe):
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 2)
        if rates is None or len(rates) < 2:
            return None
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        return df.iloc[-2]

    def is_symbol_tradeable(self, symbol: str) -> bool:
        info = mt5.symbol_info(symbol)
        return info is not None and info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL

    def submit_order(self, symbol, direction, lot, sl_pips, tp_pips, magic, comment="LiveTrade"):
        mt5.symbol_select(symbol, True)
        tick = mt5.symbol_info_tick(symbol)
        info = mt5.symbol_info(symbol)
        account = mt5.account_info()

        if not tick or not info:
            logger.error(f"Failed to retrieve tick data for {symbol}")
            return None

        price = tick.ask if direction == "buy" else tick.bid

        if not info.trade_contract_size or info.trade_mode != mt5.SYMBOL_TRADE_MODE_FULL:
            logger.error(f"{symbol} is not tradeable (trade_mode={info.trade_mode})")
            return None

        min_stop_distance = info.trade_stops_level * info.point
        sl_price, tp_price = self.calculate_sl_tp_prices(symbol, direction, price, sl_pips, tp_pips)

        if abs(price - sl_price) < min_stop_distance or abs(price - tp_price) < min_stop_distance:
            logger.warning(f"SL/TP too close to price. Broker min distance: {min_stop_distance:.5f}. Adjusting...")
            if direction == "buy":
                sl_price = price - min_stop_distance
                tp_price = price + min_stop_distance
            else:
                sl_price = price + min_stop_distance
                tp_price = price - min_stop_distance

        logger.info(
            "Diagnostic — symbol={}, trade_mode={}, volume_step={}, min_volume={}, fill_mode={}, account_login={}",
            symbol, info.trade_mode, info.volume_step, info.volume_min, info.filling_mode,
            account.login if account else "None")

        logger.info(
            "Submitting {} order | Price: {:.5f}, SL: {:.5f}, TP: {:.5f}",
            direction.upper(), price, sl_price, tp_price
        )

        for fill_mode in [mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN]:
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(lot),
                "type": mt5.ORDER_TYPE_BUY if direction == "buy" else mt5.ORDER_TYPE_SELL,
                "price": float(price),
                "sl": float(sl_price),
                "tp": float(tp_price),
                "deviation": 10,
                "magic": int(magic),
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": fill_mode
            }

            check = mt5.order_check(request)
            if check is None:
                logger.warning("order_check returned None for fill_mode={}", fill_mode)
                continue

            logger.info("order_check for fill_mode={} → retcode={} | comment='{}'", fill_mode, check.retcode,
                        check.comment)

            if check.retcode in (0, mt5.TRADE_RETCODE_DONE):
                logger.success("Filling mode {} accepted — sending trade...", fill_mode)
                result = mt5.order_send(request)
                self.log_order_result(result, request)
                return result

        logger.error("All filling modes failed for {} — no order sent", symbol)
        return None

    def log_order_result(self, result, request):
        if result is None:
            logger.error("order_send() returned None — trade was not submitted. Request: {}", request)
            return

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error("Trade failed: {} | Req: {}", result.comment, request)
        else:
            logger.success(
                "✅ Trade executed: {} {} @ {:.5f} | SL: {:.5f}, TP: {:.5f}",
                request["symbol"],
                "BUY" if request["type"] == mt5.ORDER_TYPE_BUY else "SELL",
                result.price, request["sl"], request["tp"]
            )

    def modify_stop_loss(self, ticket, new_sl, symbol):
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": symbol,
            "sl": new_sl,
            "tp": 0.0
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error("SL modification failed: {}", result.comment)
        else:
            logger.info("SL modified for ticket {} to {:.5f}", ticket, new_sl)
        return result

    def close_position(self, ticket, symbol, volume, direction):
        tick = mt5.symbol_info_tick(symbol)
        price = tick.bid if direction == "buy" else tick.ask
        close_type = mt5.ORDER_TYPE_SELL if direction == "buy" else mt5.ORDER_TYPE_BUY

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "position": ticket,
            "volume": volume,
            "type": close_type,
            "price": price,
            "deviation": 10,
            "magic": 0,
            "comment": "ManualClose",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error("Close position failed: {}", result.comment)
        else:
            logger.success("Position closed: ticket {} at {:.5f}", ticket, price)
        return result

    def update_trailing_stop(self, ticket, symbol, new_sl):
        return self.modify_stop_loss(ticket, new_sl, symbol)

    def calculate_lot_size(self, symbol, sl_pips, account_size, risk_per_trade):
        """
        Calculates dynamic lot size based on SL distance, account size, and broker pip value.
        """
        info = mt5.symbol_info(symbol)
        if not info:
            logger.error(f"Failed to get symbol info for {symbol}")
            return 0.01  # Fallback to min lot

        min_lot = info.volume_min
        max_lot = info.volume_max
        step = info.volume_step

        pip_value = self.get_pip_value(symbol)
        if pip_value <= 0:
            logger.error(f"Invalid pip value for {symbol}")
            return min_lot

        # Total cash risk allowed per trade
        cash_risk = account_size * risk_per_trade

        # Lot size formula: Lot = Cash Risk / (SL pips * pip value per lot)
        raw_lot = cash_risk / (sl_pips * pip_value)

        # Snap to broker step size and respect lot limits
        snapped_lot = round(max(min_lot, min(max_lot, round(raw_lot / step) * step)), 2)

        logger.info(
            f"Calculated lot size for {symbol}: {snapped_lot} (Raw: {raw_lot:.4f}, Min: {min_lot}, Max: {max_lot})")
        return snapped_lot
