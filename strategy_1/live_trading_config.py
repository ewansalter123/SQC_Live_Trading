import numpy as np
from Quant_Backend.data_preprocessing import *
from Quant_Backend.mt5_interface import *
from joblib import load


def get_timeframe_enum(timeframe_str):
    mapping = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
        "W1": mt5.TIMEFRAME_W1,
        "MN1": mt5.TIMEFRAME_MN1
    }
    return mapping.get(timeframe_str, None)


def move_to_break_even(position, break_even_pips, point):
    """
    Adjust the stop-loss to the entry price (break-even) for a position that has reached a certain profit threshold.

    Args:
        position: A dictionary representing a position's details.
        break_even_pips: Profit threshold in pips to trigger the break-even adjustment.
        point: The point value for the current symbol.
    """
    print("move_to_break_even called")

    entry_price = position["price"]
    current_profit = position["profit"]
    symbol = position["symbol"]
    ticket = position["ticket"]
    current_sl = position["sl"]

    profit_threshold = break_even_pips * 10 * point  # Calculate profit threshold in terms of the symbol's point value

    print(f"Symbol: {symbol}, Entry Price: {entry_price}, Current Profit: {current_profit}, Profit Threshold: {profit_threshold}")

    if current_profit >= profit_threshold:
        if current_sl == entry_price:
            print(f"SL for ticket {ticket} is already at break-even ({current_sl}). No modification needed.")
        else:
            print(f"Moving SL to break-even for ticket {ticket}.")
            modify_stop_loss(ticket, entry_price, symbol)

def move_to_profit_lock(position, profit_lock_pips, point):
    """
    Move stop-loss to lock in profit (x pips above/below the entry price).

    Args:
        position: A dictionary representing a position's details.
        profit_lock_pips: The number of pips to lock in as profit.
        point: The point value for the current symbol.
    """
    if 'position' not in position:
        print(f"Error: 'position' key missing in position data: {position}")
        return

    position_type = position['position']  # 0 for Buy, 1 for Sell
    symbol = position['symbol']
    entry_price = position['price']
    current_sl = position['sl']
    ticket = int(position['ticket'])

    # Calculate the profit lock price
    if position_type == 0:  # Buy
        profit_lock_price = entry_price + (profit_lock_pips * 10 * point)
        if current_sl < profit_lock_price:
            print(f"Moving SL to lock in profit for ticket {ticket}.")
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": symbol,
                "sl": profit_lock_price,
                "tp": position['tp'],
                "position": ticket,
            }
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"Failed to modify SL for ticket {ticket}: {result.comment}")
            else:
                print(f"Successfully modified SL for ticket {ticket} to {profit_lock_price}.")
    elif position_type == 1:  # Sell
        profit_lock_price = entry_price - (profit_lock_pips * 10 * point)
        if current_sl > profit_lock_price:
            print(f"Moving SL to lock in profit for ticket {ticket}.")
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": symbol,
                "sl": profit_lock_price,
                "tp": position['tp'],
                "position": ticket,
            }
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"Failed to modify SL for ticket {ticket}: {result.comment}")
            else:
                print(f"Successfully modified SL for ticket {ticket} to {profit_lock_price}.")
    else:
        print(f"Unknown position type: {position_type}")

def trail_by_pips(position, trail_pips, point):
    """
    Dynamically adjust stop-loss to trail the price by a given number of pips.

    Args:
        position: A dictionary representing a position's details.
        trail_pips: The number of pips to trail behind the current price.
        point: The point value for the current symbol.
    """
    if 'position' not in position:
        print(f"Error: 'position' key missing in position data: {position}")
        return

    position_type = position['position']  # 0 for Buy, 1 for Sell
    symbol = position['symbol']
    current_price = mt5.symbol_info_tick(symbol).bid if position_type == 0 else mt5.symbol_info_tick(symbol).ask
    current_sl = position['sl']
    ticket = int(position['ticket'])

    if position_type == 0:  # Buy
        # Calculate trailing stop-loss price for Buy
        trail_price = current_price - (trail_pips * 10 * point)
        if current_sl >= trail_price:
            print(f"Buy Order: SL for ticket {ticket} is already at or above trailing level. Current SL: {current_sl}")
            return
        print(f"Updating SL for Buy ticket {ticket} to trail by {trail_pips} pips.")
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "sl": trail_price,
            "tp": position['tp'],
            "position": ticket,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Failed to modify SL for ticket {ticket}: {result.comment}")
        else:
            print(f"Successfully updated SL for Buy ticket {ticket} to {trail_price}.")
    elif position_type == 1:  # Sell
        # Calculate trailing stop-loss price for Sell
        trail_price = current_price + (trail_pips * 10 * point)
        if current_sl <= trail_price:
            print(f"Sell Order: SL for ticket {ticket} is already at or below trailing level. Current SL: {current_sl}")
            return
        print(f"Updating SL for Sell ticket {ticket} to trail by {trail_pips} pips.")
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "sl": trail_price,
            "tp": position['tp'],
            "position": ticket,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Failed to modify SL for ticket {ticket}: {result.comment}")
        else:
            print(f"Successfully updated SL for Sell ticket {ticket} to {trail_price}.")
    else:
        print(f"Unknown position type: {position_type}")

