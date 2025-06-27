import ta
from scipy.constants import micro
from ta.trend import SMAIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator
from ta.volatility import average_true_range, AverageTrueRange
from datetime import datetime, timedelta, time
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

import numpy as np


def detect_rolling_range_breakouts(df, lookback=20, buffer_pips=0.0):
    """
    Detects breakouts above or below a rolling high/low over a given lookback period.

    Args:
        df (pd.DataFrame): Must contain 'close', 'high', and 'low' columns.
        lookback (int): Number of candles to define the range window.
        buffer_pips (float): Optional buffer to avoid false breakouts (e.g., 0.5 pip = 0.00005).

    Returns:
        pd.DataFrame: Modified DataFrame with breakout columns.
    """
    df = df.copy()

    # Validate columns
    required = ['close', 'high', 'low']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Define the rolling high/low (excluding current candle)
    df['rolling_high'] = df['high'].shift(1).rolling(window=lookback).max()
    df['rolling_low'] = df['low'].shift(1).rolling(window=lookback).min()

    # Breakout logic
    df['Breakout_Up'] = df['close'] > (df['rolling_high'] + buffer_pips)
    df['Breakout_Down'] = df['close'] < (df['rolling_low'] - buffer_pips)

    return df

def detect_session_range_breakouts(df,
                                    session_start_hour, session_end_hour,
                                    breakout_start_hour, breakout_end_hour,
                                    buffer_pips=0.0):
    """
    Detects breakouts of a session's high/low range during a later breakout window.

    Args:
        df (pd.DataFrame): Must have 'high', 'low', 'close' columns and DatetimeIndex.
        session_start_hour (int): Hour when session window begins (e.g., 0 for 00:00)
        session_end_hour (int): Hour when session window ends (e.g., 8 for 08:00)
        breakout_start_hour (int): Hour when breakout detection starts (e.g., 8 for 08:00)
        breakout_end_hour (int): Hour when breakout detection ends (e.g., 12 for 12:00)
        buffer_pips (float): Optional buffer to avoid false breakouts (e.g., 0.0002)

    Returns:
        pd.DataFrame: Adds columns for session_high/low and breakout flags.
    """
    df = df.copy()

    df['SessionHigh'] = np.nan
    df['SessionLow'] = np.nan
    df['SessionBreakout_Up'] = False
    df['SessionBreakout_Down'] = False

    # Ensure datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Index must be a DatetimeIndex")

    # Convert int hours to time objects
    session_start = time(hour=session_start_hour)
    session_end = time(hour=session_end_hour)
    breakout_start = time(hour=breakout_start_hour)
    breakout_end = time(hour=breakout_end_hour)

    # Group by calendar day
    grouped = df.groupby(df.index.date)

    for date, group in grouped:
        session_window = group.between_time(session_start, session_end)
        breakout_window = group.between_time(breakout_start, breakout_end)

        if session_window.empty or breakout_window.empty:
            continue

        session_high = session_window['high'].max()
        session_low = session_window['low'].min()

        df.loc[breakout_window.index, 'SessionHigh'] = session_high
        df.loc[breakout_window.index, 'SessionLow'] = session_low

        breakout_up = breakout_window['close'] > (session_high + buffer_pips)
        breakout_down = breakout_window['close'] < (session_low - buffer_pips)

        df.loc[breakout_window.index, 'SessionBreakout_Up'] = breakout_up
        df.loc[breakout_window.index, 'SessionBreakout_Down'] = breakout_down

    return df




#Adds logic to detect momentum-driven breakouts above/below daily/weekly high/low with RSI conditions.
def weekly_daily_high_low_momentum_rsi_logic(df, rsi_high_threshold, rsi_low_threshold):
    """
    Adds logic to detect momentum-driven breakouts above/below daily/weekly high/low with RSI conditions.

    Args:
        df (pd.DataFrame): The input DataFrame with columns for price and RSI.
        rsi_column (str): The column name containing RSI values.
        rsi_high_threshold (int): RSI value to consider overbought conditions (default: 75).
        rsi_low_threshold (int): RSI value to consider oversold conditions (default: 25).

    Returns:
        pd.DataFrame: The DataFrame with additional Boolean columns for momentum logic.
    """
    # Check for required columns
    required_columns = ['close', 'DailyHigh', 'DailyLow', 'WeeklyHigh', 'WeeklyLow', 'RSI']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing from the DataFrame.")

    df = df.copy()

    # Momentum logic for Daily High
    df['MomentumAboveDailyHigh'] = (df['close'] > df['DailyHigh']) & (df["RSI"] >= rsi_high_threshold)
    # Momentum logic for Daily Low
    df['MomentumBelowDailyLow'] = (df['close'] < df['DailyLow']) & (df["RSI"] <= rsi_low_threshold)

    # Momentum logic for Weekly High
    df['MomentumAboveWeeklyHigh'] = (df['close'] > df['WeeklyHigh']) & (df["RSI"] >= rsi_high_threshold)
    # Momentum logic for Weekly Low
    df['MomentumBelowWeeklyLow'] = (df['close'] < df['WeeklyLow']) & (df["RSI"] <= rsi_low_threshold)

    # Display for debugging (optional)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print("Updated DataFrame with Momentum Logic:")
    momentum_true_df = df[
        (df['MomentumAboveDailyHigh']) |
        (df['MomentumBelowDailyLow']) |
        (df['MomentumAboveWeeklyHigh']) |
        (df['MomentumBelowWeeklyLow'])
        ]

    if not momentum_true_df.empty:
        print("Rows where momentum logic is True:")
        print(momentum_true_df)
    else:
        print("No momentum logic True rows found.")
    print(df.tail())

    return df

#Adds logic to detect momentum-driven breakouts above/below daily/weekly high/low using tick volume.
def weekly_daily_high_low_momentum_volume_logic(df, volume_threshold):
    """
    Adds logic to detect momentum-driven breakouts above/below daily/weekly high/low using tick volume.

    Args:
        df (pd.DataFrame): The input DataFrame with columns for price and tick volume.
        volume_threshold (float): The tick volume threshold to consider momentum-driven conditions.

    Returns:
        pd.DataFrame: The DataFrame with additional Boolean columns for momentum logic.
    """
    # Check for required columns
    required_columns = ['close', 'PrevDayHigh', 'PrevDayLow', 'PrevWeekHigh', 'PrevWeekLow', 'tick_volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing from the DataFrame.")

    df = df.copy()

    # Adjust volume threshold dynamically if needed
    if volume_threshold <= 1:
        volume_threshold = df['tick_volume'].quantile(0.50)  # Use median as threshold
        print(f"Adjusted volume threshold to {volume_threshold}")

    # Momentum logic for daily and weekly levels
    df['MomentumAboveDailyHigh'] = (df['close'] > df['PrevDayHigh']) & (df['tick_volume'] >= volume_threshold)
    df['MomentumBelowDailyLow'] = (df['close'] < df['PrevDayLow']) & (df['tick_volume'] >= volume_threshold)
    df['MomentumAboveWeeklyHigh'] = (df['close'] > df['PrevWeekHigh']) & (df['tick_volume'] >= volume_threshold)
    df['MomentumBelowWeeklyLow'] = (df['close'] < df['PrevWeekLow']) & (df['tick_volume'] >= volume_threshold)

    # # Print rows where conditions are met
    # print("Momentum Above Daily High:")
    # print(df[df['MomentumAboveDailyHigh']][['close', 'PrevDayHigh', 'tick_volume']])
    #
    # print("Momentum Below Daily Low:")
    # print(df[df['MomentumBelowDailyLow']][['close', 'PrevDayLow', 'tick_volume']])
    #
    # print("Momentum Above Weekly High:")
    # print(df[df['MomentumAboveWeeklyHigh']][['close', 'PrevWeekHigh', 'tick_volume']])
    #
    # print("Momentum Below Weekly Low:")
    # print(df[df['MomentumBelowWeeklyLow']][['close', 'PrevWeekLow', 'tick_volume']])

    return df

# Adds higher timeframe OHLC and wick-time columns to a lower-timeframe DataFrame
def add_htf_ohlc_with_wick_times(df, base_freq="1H", target_freq="4H"):
    """
    Adds higher timeframe (HTF) OHLC and wick-time columns to a lower-timeframe DataFrame.

    Args:
        df (pd.DataFrame): Lower timeframe OHLC dataframe with 'high_time' and 'low_time'.
        base_freq (str): String representing base timeframe (e.g., '1H', '5T', '15T').
        target_freq (str): String representing the higher timeframe to resample to (e.g., '4H', '1D').

    Returns:
        pd.DataFrame: DataFrame with additional HTF-prefixed OHLC and wick time columns.
    """
    df = df.copy()

    # Validate index and columns
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex")

    required_cols = ["open", "high", "low", "close", "high_time", "low_time"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Ensure time columns are datetime
    df["high_time"] = pd.to_datetime(df["high_time"])
    df["low_time"] = pd.to_datetime(df["low_time"])

    # Resample OHLC to target (higher) timeframe
    group = df.resample(target_freq, label="left", closed="left")
    df_htf = group.agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    })

    # Wick time logic: when did HTF high/low occur
    def get_extreme_time(subset, column, is_high=True):
        if subset.empty:
            return pd.NaT
        idx = subset[column].idxmax() if is_high else subset[column].idxmin()
        return subset.loc[idx, "high_time" if is_high else "low_time"]

    high_times, low_times = [], []
    for time_start in df_htf.index:
        time_end = time_start + pd.Timedelta(target_freq)
        sub_df = df.loc[time_start:time_end - pd.Timedelta(seconds=1)]

        high_times.append(get_extreme_time(sub_df, "high", is_high=True))
        low_times.append(get_extreme_time(sub_df, "low", is_high=False))

    df_htf["high_time"] = high_times
    df_htf["low_time"] = low_times

    # Generate proper prefix (e.g., 4H → H4, 1D → D1, 30T → M30)
    def tf_label(freq_str):
        f = freq_str.upper().replace("T", "M")  # 'T' = minutes
        return f if f[-1] in "HDWM" else f + "?"

    prefix = f"HTF_{tf_label(target_freq)}_"
    df_htf.columns = [prefix + col for col in df_htf.columns]

    # Reindex and forward-fill so we can merge into base_df
    reindex_index = pd.date_range(df.index.min(), df.index.max(), freq=base_freq)
    df_htf = df_htf.reindex(reindex_index).ffill()

    # Merge into original and ffill
    df = df.merge(df_htf, how="left", left_index=True, right_index=True)
    df = df.sort_index().ffill()

    return df

#Adds True/False to Bull/Bear column
def is_candle_bullish_or_bearish(df):
    df = df.copy()
    df['Bullish_Candle'] = False
    df['Bearish_Candle'] = False
    df.loc[(df["open"] - df["close"]) < 0, "Bullish_Candle"] = True
    df.loc[(df["open"] - df["close"]) > 0, "Bearish_Candle"] = True
    return df

def identify_flat_bottom_candles(df):
    """
    Adds a column 'Flat_Bottom' to the DataFrame to identify flat bottom candles
    with a specified tolerance.

    Args:
        dataframe (pd.DataFrame): The input DataFrame with columns 'open' and 'low'.
        tolerance (float): The acceptable difference between 'open' and 'low' prices.

    Returns:
        pd.DataFrame: The modified DataFrame with an additional 'Flat_Bottom' column.
    """
    df = df.copy()
    tolerance = 1e-5
    # Ensure 'open' and 'low' columns are present
    if 'open' not in df.columns or 'low' not in df.columns:
        raise ValueError("The dataframe must have 'open' and 'low' columns.")

    # Add the 'Flat_Bottom' column with tolerance
    df['Flat_Bottom'] = (abs(df['open'] - df['low']) <= tolerance)
    # flat_bottom_successes = df[df['Flat_Bottom']]
    # print(flat_bottom_successes)
    return df

def identify_flat_tops_candles(df):
    """
    Adds a column 'Flat_Bottom' to the DataFrame to identify flat bottom candles
    with a specified tolerance.

    Args:
        dataframe (pd.DataFrame): The input DataFrame with columns 'open' and 'low'.
        tolerance (float): The acceptable difference between 'open' and 'low' prices.

    Returns:
        pd.DataFrame: The modified DataFrame with an additional 'Flat_Bottom' column.
    """
    df = df.copy()
    tolerance = 1e-5
    # Ensure 'open' and 'low' columns are present
    if 'open' not in df.columns or 'low' not in df.columns:
        raise ValueError("The dataframe must have 'open' and 'low' columns.")

    # Add the 'Flat_Bottom' column with tolerance
    df['Flat_Top'] = (abs(df['open'] - df['high']) <= tolerance)
    # flat_top_successes = df[df['Flat_Top']]
    # print(flat_top_successes)

    return df

def identify_structure_and_bos(df):
    """
    Identifies structural points (HH, HL, LH, LL) and Break of Structure (BOS) in the market.

    Args:
        df (pd.DataFrame): Input DataFrame with 'high', 'low', and 'close' columns.

    Returns:
        pd.DataFrame: DataFrame with added columns for structural points and BOS markers.
    """
    df = df.copy()

    # Predefine columns
    df['Swing_High'] = False
    df['Swing_Low'] = False
    df['Last_High'] = np.nan
    df['Last_Low'] = np.nan
    df['BOS_Uptrend'] = False
    df['BOS_Downtrend'] = False

    # Pre-cache values for speed
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values

    last_high = np.nan
    last_low = np.nan

    for i in range(1, len(df) - 1):
        high = highs[i]
        low = lows[i]
        close = closes[i]

        # Check swing high
        if high > highs[i - 1] and high > highs[i + 1]:
            df.loc[df.index[i], 'Swing_High'] = True
            last_high = high

        # Check swing low
        if low < lows[i - 1] and low < lows[i + 1]:
            df.loc[df.index[i], 'Swing_Low'] = True
            last_low = low

        # Save last high/low (faster than using iloc)
        df.loc[df.index[i], 'Last_High'] = last_high
        df.loc[df.index[i], 'Last_Low'] = last_low

        # BOS logic
        if not np.isnan(last_high) and close > last_high:
            df.loc[df.index[i], 'BOS_Uptrend'] = True
        if not np.isnan(last_low) and close < last_low:
            df.loc[df.index[i], 'BOS_Downtrend'] = True

    return df

def identify_3_consecutive_candles(df):
    """
    Identifies streaks of 3+ bullish or bearish candles using vectorized logic.
    Compatible with pandas and cuDF.

    Returns:
        pd.DataFrame with two new columns:
            - 'Consecutive_Bullish'
            - 'Consecutive_Bearish'
    """
    df = df.copy()

    # 1 = bullish, -1 = bearish, 0 = neutral
    candle_direction = np.where(df['close'] > df['open'], 1,
                         np.where(df['close'] < df['open'], -1, 0))

    # Convert to Series for rolling operation
    direction_series = pd.Series(candle_direction, index=df.index)

    # Count consecutive values using shifting logic
    bullish_mask = (direction_series == 1)
    bearish_mask = (direction_series == -1)

    bullish_streak = bullish_mask & bullish_mask.shift(1) & bullish_mask.shift(2)
    bearish_streak = bearish_mask & bearish_mask.shift(1) & bearish_mask.shift(2)

    # Apply mask to current and previous two candles
    df['Consecutive_Bullish'] = bullish_streak | bullish_streak.shift(-1) | bullish_streak.shift(-2)
    df['Consecutive_Bearish'] = bearish_streak | bearish_streak.shift(-1) | bearish_streak.shift(-2)

    df['Consecutive_Bullish'] = df['Consecutive_Bullish'].fillna(False)
    df['Consecutive_Bearish'] = df['Consecutive_Bearish'].fillna(False)

    return df

def identify_trade_signals_with_flat_top_or_bottom_and_bos(df):
    """
    Vectorized identification of buy/sell signals based on flat candle structure,
    3-bar momentum, and BOS confirmation. Works with pandas and GPU-safe logic.
    """
    df = df.copy()
    n = len(df)

    df['Sell_Signal'] = False
    df['Buy_Signal'] = False
    df['Entry_Price'] = np.nan
    df['Stop_Loss'] = np.nan

    # Vectorized setup scan (2 candles after flat candle)
    sell_candidates = (
        df['Flat_Top'].shift(0) &
        df['Consecutive_Bearish'].shift(-2) &
        df['BOS_Downtrend'].shift(-2)
    )

    buy_candidates = (
        df['Flat_Bottom'].shift(0) &
        df['Consecutive_Bullish'].shift(-2) &
        df['BOS_Uptrend'].shift(-2)
    )

    # Entry/SL logic: use candle before setup (i-1)
    entry_sell = df['low'].shift(1)
    stop_sell = df['high'].shift(1)

    entry_buy = df['high'].shift(1)
    stop_buy = df['low'].shift(1)

    df['Sell_Setup'] = sell_candidates
    df['Buy_Setup'] = buy_candidates

    df.loc[sell_candidates, 'Entry_Price'] = entry_sell[sell_candidates]
    df.loc[sell_candidates, 'Stop_Loss'] = stop_sell[sell_candidates]

    df.loc[buy_candidates, 'Entry_Price'] = entry_buy[buy_candidates]
    df.loc[buy_candidates, 'Stop_Loss'] = stop_buy[buy_candidates]

    # Apply retracement condition to future rows using rolling windows (simplified logic)
    # Optional: Create a mask that delays activation by 3 bars (approximate retrace logic)
    df['Sell_Signal'] = (
        df['Sell_Setup'].shift(3).fillna(False) &
        (df['close'] >= df['Entry_Price']) &
        (df['close'] <= df['Stop_Loss'])
    )

    df['Buy_Signal'] = (
        df['Buy_Setup'].shift(3).fillna(False) &
        (df['close'] <= df['Entry_Price']) &
        (df['close'] >= df['Stop_Loss'])
    )

    # Clean up intermediate columns
    df.drop(columns=['Sell_Setup', 'Buy_Setup'], inplace=True)

    return df

def add_donchian_channel(df, period=20):
    """
    Adds Donchian Channel columns to the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame with 'high' and 'low' columns.
        period (int): The look-back period for the Donchian Channel. Default is 20.

    Returns:
        pd.DataFrame: The DataFrame with Donchian Channel columns added.
    """
    df = df.copy()

    # Validate required columns
    if 'high' not in df.columns or 'low' not in df.columns:
        raise ValueError("The DataFrame must contain 'high' and 'low' columns.")

    # Column names for Donchian Channel
    upper_col = f"donchian_upper_{period}"
    lower_col = f"donchian_lower_{period}"
    midline_col = f"donchian_midline_{period}"

    # Calculate Donchian Channel
    df[upper_col] = df['high'].rolling(window=period).max()
    df[lower_col] = df['low'].rolling(window=period).min()
    df[midline_col] = (df[upper_col] + df[lower_col]) / 2

    return df

def add_lwti(df, lwti_period=25, smoothing=False, smoothing_type="SMA", smoothing_period=20):
    """
    Adds Larry Williams Trade Index (LWTI) to the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame with 'close', 'high', and 'low' columns.
        period (int): Lookback period for LWTI calculation. Default is 8.
        smoothing (bool): Whether to smooth the LWTI output. Default is False.
        smoothing_type (str): Type of smoothing ('SMA', 'EMA', 'WMA', 'RMA'). Default is 'SMA'.
        smoothing_period (int): Period for smoothing if enabled. Default is 5.

    Returns:
        pd.DataFrame: DataFrame with added LWTI columns.
    """
    df = df.copy()
    # Ensure required columns exist
    required_columns = ['close', 'high', 'low']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Calculate close difference
    df['LWTI_CloseDiff'] = df['close'] - df['close'].shift(lwti_period)

    # Calculate ATR
    atr = AverageTrueRange(df['high'], df['low'], df['close'], window=lwti_period)
    df['LWTI_ATR'] = atr.average_true_range()

    # Calculate LWTI
    df['LWTI'] = (df['LWTI_CloseDiff'] / df['LWTI_ATR']) * 50 + 50

    # Smoothing if required
    if smoothing:
        if smoothing_type == "SMA":
            df['LWTI_Smoothed'] = df['LWTI'].rolling(window=smoothing_period).mean()
        elif smoothing_type == "EMA":
            df['LWTI_Smoothed'] = df['LWTI'].ewm(span=smoothing_period).mean()
        elif smoothing_type == "WMA":
            weights = np.arange(1, smoothing_period + 1)
            df['LWTI_Smoothed'] = df['LWTI'].rolling(window=smoothing_period).apply(
                lambda x: np.dot(x, weights) / weights.sum(), raw=True)
        elif smoothing_type == "RMA":
            df['LWTI_Smoothed'] = df['LWTI'].ewm(alpha=1/smoothing_period).mean()
        else:
            raise ValueError("Invalid smoothing type. Choose from 'SMA', 'EMA', 'WMA', 'RMA'.")
    else:
        df['LWTI_Smoothed'] = df['LWTI']

    # Add signal columns
    midpoint = 50
    df['LWTI_LongSignal'] = (df['LWTI_Smoothed'] > midpoint) & (df['LWTI_Smoothed'].shift(1) <= midpoint)
    df['LWTI_ShortSignal'] = (df['LWTI_Smoothed'] < midpoint) & (df['LWTI_Smoothed'].shift(1) >= midpoint)

    return df

def add_average_tick_volume(df, period=14):
    """
    Adds a column for the average tick volume over a specified period to the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame with a 'tick_volume' column.
        period (int): The lookback period for calculating the average tick volume.

    Returns:
        pd.DataFrame: The DataFrame with an additional column for average tick volume.
    """
    df = df.copy()
    # Ensure the required column exists
    if 'tick_volume' not in df.columns:
        raise ValueError("The DataFrame must have a 'tick_volume' column.")

    # Validate the period
    if not isinstance(period, int) or period <= 0:
        raise ValueError("The 'period' parameter must be a positive integer.")

    # Replace 0 or missing tick_volume values (if needed)
    df['tick_volume'] = df['tick_volume'].replace(0, np.nan).fillna(method='ffill')

    # Calculate the rolling average of tick volume
    df['average_tick_volume'] = df['tick_volume'].rolling(window=period).mean()

    return df

def calculate_vwap(df):
    """
    Adds a VWAP (Volume-Weighted Average Price) column to the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame with 'high', 'low', 'close', and 'tick_volume' columns.

    Returns:
        pd.DataFrame: The DataFrame with an additional 'VWAP' column.
    """
    df = df.copy()
    # Ensure required columns exist
    required_columns = ['high', 'low', 'close', 'tick_volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing from the DataFrame.")

    # Calculate the typical price
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3

    # Calculate the cumulative volume and cumulative price-volume
    df['cum_volume'] = df['tick_volume'].cumsum()
    df['cum_price_volume'] = (df['typical_price'] * df['tick_volume']).cumsum()

    # Calculate VWAP
    df['VWAP'] = df['cum_price_volume'] / df['cum_volume']

    # Drop intermediate columns if not needed
    df.drop(columns=['typical_price', 'cum_volume', 'cum_price_volume'], inplace=True)

    return df

#Adds what day of the week each cell is
def add_day_of_week(df):
    """
    Adds a column to the DataFrame indicating the day of the week.

    Args:
        df (pd.DataFrame): The input DataFrame with a datetime index.

    Returns:
        pd.DataFrame: The DataFrame with an added 'DayOfWeek' column.
    """
    # Ensure the index is datetime
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("DataFrame index must be a datetime type.")
    df = df.copy()

    # Map each row's index to the corresponding day of the week
    df['DayOfWeek'] = df.index.map(lambda x: x.strftime('%A'))

    return df

#Adds true/false if the cell is within the defined days to trade within parameters
def is_active_trading_day(df, active_trading_days):
    df = df.copy()
    # Ensure the index is datetime
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("DataFrame index must be a datetime type.")

    # Validate active_trading_days is a list of strings
    if not isinstance(active_trading_days, list) or not all(isinstance(day, str) for day in active_trading_days):
        raise ValueError("'active_trading_days' must be a list of strings representing day names (e.g., ['Monday', 'Tuesday']).")

    # Map each row's index to a boolean indicating active trading day
    df['ActiveTradingDay'] = df.index.map(lambda x: x.strftime('%A') in active_trading_days)

    return df

#Adds weekly and daily high/low columns and Boolean columns for price breakouts
def add_weekly_daily_high_low(df):
    """
    Adds weekly and daily high/low columns and Boolean columns for price breakouts to the DataFrame,
    using the previous day's high/low for signals.

    Args:
        df (pd.DataFrame): The input DataFrame with a datetime index.

    Returns:
        pd.DataFrame: The DataFrame with added high/low and breakout Boolean columns.
    """
    df = df.copy()
    # Ensure the index is datetime
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("DataFrame index must be a datetime type.")

    # Daily high/low from the previous day
    daily_high_low = df.groupby(df.index.date).agg({'high': 'max', 'low': 'min'}).shift(1)
    df['PrevDayHigh'] = df.index.map(lambda x: daily_high_low.loc[x.date(), 'high'])
    df['PrevDayLow'] = df.index.map(lambda x: daily_high_low.loc[x.date(), 'low'])

    # Weekly high/low from the previous week
    df['Week'] = df.index.to_period('W')  # Convert to weekly periods
    weekly_high_low = df.groupby('Week').agg({'high': 'max', 'low': 'min'})
    weekly_high_low['PrevWeekHigh'] = weekly_high_low['high'].shift(1)
    weekly_high_low['PrevWeekLow'] = weekly_high_low['low'].shift(1)

    # Map weekly high/low back to the original DataFrame
    df['PrevWeekHigh'] = df['Week'].map(weekly_high_low['PrevWeekHigh'])
    df['PrevWeekLow'] = df['Week'].map(weekly_high_low['PrevWeekLow'])

    # Create Boolean columns for price passing above/below previous day's highs/lows
    df['PassedAbovePrevDayHigh'] = df['close'] > df['PrevDayHigh']
    df['PassedBelowPrevDayLow'] = df['close'] < df['PrevDayLow']

    # Create Boolean columns for price passing above/below previous week's highs/lows
    df['PassedAbovePrevWeekHigh'] = df['close'] > df['PrevWeekHigh']
    df['PassedBelowPrevWeekLow'] = df['close'] < df['PrevWeekLow']

    # Drop the temporary 'Week' column
    df.drop(columns='Week', inplace=True)

    return df

#Adds boolean columns 'open_time' and 'close_time' to a DataFrame based on the index time.
def add_to_open_to_close_time_bool(df, open_time_hh, open_time_mm, close_time_hh, close_time_mm):
    """
    Adds boolean columns 'open_time' and 'close_time' to a DataFrame based on the index time.

    Args:
        df (pd.DataFrame): DataFrame with a DateTime index.
        open_time_hh (int): Hour for open time.
        open_time_mm (int): Minute for open time.
        close_time_hh (int): Hour for close time.
        close_time_mm (int): Minute for close time.

    Returns:
        pd.DataFrame: Copy of the DataFrame with new boolean columns.
    """
    df = df.copy()

    # Ensure index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex")

    try:
        open_time = time(hour=int(open_time_hh), minute=int(open_time_mm))
        close_time = time(hour=int(close_time_hh), minute=int(close_time_mm))
    except ValueError as e:
        raise ValueError(f"Invalid time value: {e}")

    df["to_open_time"] = df.index.time == open_time
    df["close_time"] = df.index.time == close_time

    return df

# Add column for true/false if within trading hours
def set_trading_hours(df, trading_start_hh, trading_start_mm, trading_end_hh, trading_end_mm):
    df = df.copy()

    # Validate inputs to ensure they are not NaN
    if pd.isna(trading_start_hh) or pd.isna(trading_start_mm) or pd.isna(trading_end_hh) or pd.isna(trading_end_mm):
        raise ValueError("Trading start or end times contain NaN values. Check your inputs.")

    # Convert inputs to integers
    try:
        start_time = time(hour=int(trading_start_hh), minute=int(trading_start_mm))
        end_time = time(hour=int(trading_end_hh), minute=int(trading_end_mm))
    except ValueError as e:
        raise ValueError(f"Invalid time value: {e}")

    # Create a new column with True/False depending on whether within the trading hours
    df['ActiveTradingHour'] = df.index.map(
        lambda x: start_time <= x.replace(second=0, microsecond=0).time() <= end_time
    )
    return df

#Add close price set by calculation time
def add_prev_day_close_price(df, calculation_time_hh, calculation_time_mm):
    df = df.copy()
    calc_time = time(hour=int(calculation_time_hh), minute=int(calculation_time_mm))
    df['PrevDayClosPrice'] = None
    df.loc[df.index.time == calc_time, 'PrevDayClosPrice'] = df['close']
    df['PrevDayClosPrice'] = df['PrevDayClosPrice'].ffill().astype(float)
    return df

#Calculate percentage diff and apply true/false the close is </>
def calculate_percentage_difference(df, magic_percent):
    df = df.copy()
    df['PercentDiff'] = df['PrevDayClosPrice'] * (magic_percent / 100)
    df['HigherThanPrevClose'] = df['close'] > (df['PrevDayClosPrice'] + df['PercentDiff'])
    df['LowerThanPrevClose'] = df['close'] < (df['PrevDayClosPrice'] - df['PercentDiff'])
    return df

# #FOR RANGE FINDER ONLY True/False for long/short column based on the first row for each day until new calculation time
def assign_daily_range_finder_signals(df):
    df = df.copy()
    # Initialize the signal columns to False
    df['LongSignal'] = False
    df['ShortSignal'] = False

    # Group by date to ensure the logic is applied per day
    df['date'] = df.index.date  # Extract date from the index
    grouped = df.groupby('date')

    for _, group in grouped:
        long_condition = group['ActiveTradingHour'] & group['LowerThanPrevClose']
        short_condition = group['ActiveTradingHour'] & group['HigherThanPrevClose']

        # Assign True to the first row in each group meeting the condition
        if long_condition.any():
            df.loc[long_condition.idxmax(), 'LongSignal'] = True
        if short_condition.any():
            df.loc[short_condition.idxmax(), 'ShortSignal'] = True

    # Drop the temporary 'date' column
    df.drop(columns='date', inplace=True)

    return df

#Applies a moving average crossover signal
def two_moving_average_crossover(df, ma1, ma2, ma_type):
    """
    Adds columns to the DataFrame indicating bullish and bearish moving average crossovers.

    Parameters:
    df (pd.DataFrame): DataFrame with at least a 'close' column.
    ma1 (int): Period for the first moving average (shorter period).
    ma2 (int): Period for the second moving average (longer period).
    ma_type (str): Type of moving average, either 'sma' or 'ema'.

    Returns:
    pd.DataFrame: The DataFrame with two new Boolean columns:
                  'bullish_crossover' and 'bearish_crossover'.
    """
    df = df.copy()

    if ma_type == "sma":
        df['ma1'] = df['close'].rolling(window=ma1).mean()
        df['ma2'] = df['close'].rolling(window=ma2).mean()
    elif ma_type == "ema":
        df['ma1'] = df['close'].ewm(span=ma1, adjust=False).mean()
        df['ma2'] = df['close'].ewm(span=ma2, adjust=False).mean()
    else:
        raise ValueError("Invalid ma_type. Use 'sma' or 'ema'.")

    # Detect bullish and bearish crossovers
    df['bullish_crossover'] = (df['ma1'] > df['ma2']) & (df['ma1'].shift(1) <= df['ma2'].shift(1))
    df['bearish_crossover'] = (df['ma1'] < df['ma2']) & (df['ma1'].shift(1) >= df['ma2'].shift(1))

    # Drop intermediate moving average columns if not needed
    df.drop(['ma1', 'ma2'], axis=1, inplace=True)

    return df

def add_grid_statistical_features(df: pd.DataFrame, window: int = 20, std_multiplier: float = 1.5) -> pd.DataFrame:
    """
    Adds grid-relevant statistical features to the DataFrame.
    """
    df = df.copy()
    window = int(window)

    df['mean_price'] = df['close'].rolling(window=window).mean()
    df['rolling_std'] = df['close'].rolling(window=window).std()
    df['z_score'] = (df['close'] - df['mean_price']) / df['rolling_std']
    df['grid_upper_1'] = df['mean_price'] + std_multiplier * df['rolling_std']
    df['grid_lower_1'] = df['mean_price'] - std_multiplier * df['rolling_std']
    df['hedge_trigger_zone'] = (df['z_score'].abs() > std_multiplier).astype(int)
    df['LongSignal'] = df['close'] < df['grid_lower_1']
    df['ShortSignal'] = df['close'] > df['grid_upper_1']

    return df


#SimpleMovingAverage
def sma(df, col, n):
    df = df.copy()
    df[f"SMA_{n}"] = ta.trend.SMAIndicator(df[col],int(n)).sma_indicator()
    return df

#SimpleMovingAverage Difference
def sma_diff(df, col, n, m):
    df = df.copy()
    df[f"SMA_d_{n}"] = ta.trend.SMAIndicator(df[col], int(n)).sma_indicator()
    df[f"SMA_d_{m}"] = ta.trend.SMAIndicator(df[col], int(m)).sma_indicator()

    df[f"SMA_diff"] = df[f"SMA_d_{n}"] - df[f"SMA_d_{m}"]
    return df

#ExponentialMovingAverage
def ema(df, col, n):
    df = df.copy()
    df[f"EMA_{n}"] = ta.trend.EMAIndicator(col, int(n)).ema_indicator()
    return df
#RSI
def rsi(df, col, n):
    df = df.copy()
    df[f"RSI"] = ta.momentum.RSIIndicator(col, int(n)).rsi()
    return df
#MACD
def macd(df, fast_period=12, slow_period=26, signal_period=9):
    df = df.copy()
    # Colum names
    fast_ema_col = f"MACD_EMA_{fast_period}"
    slow_ema_col = f"MACD_EMA_{slow_period}"
    macd_col = "MACD_Line"
    signal_col = "MACD_Signal_Line"
    histogram_col = f"MACD_Histogram_{fast_period}_{slow_period}_{signal_period}"
    uptrend_col = "MACD_Uptrend"
    downtrend_col = "MACD_Downtrend"
    strong_uptrend_col = "MACD_Strong_Uptrend"
    strong_downtrend_col = "MACD_Strong_Downtrend"


    # Create fast and slow EMAs
    df[fast_ema_col] = ta.trend.EMAIndicator(df['close'], int(fast_period)).ema_indicator()
    df[slow_ema_col] = ta.trend.EMAIndicator(df['close'], int(slow_period)).ema_indicator()

    # Calculate MACD line
    df[macd_col] = df[fast_ema_col] - df[slow_ema_col]

    # Calculate signal line
    df[signal_col] = ta.trend.EMAIndicator(df[macd_col], int(signal_period)).ema_indicator()

    # Calculate MACD histogram
    df[histogram_col] = df[macd_col] - df[signal_col]

    # Determine if uptrend or downtrend
    df[uptrend_col] = df[macd_col] > df[signal_col]
    df[downtrend_col] = df[macd_col] < df[signal_col]
    df[strong_uptrend_col] = ((df[macd_col] > df[signal_col]) & (df[macd_col] > 0) & (df[signal_col] > 0))
    df[strong_downtrend_col] = ((df[macd_col] < df[signal_col]) & (df[macd_col] < 0) & (df[signal_col] > 0))

    return df

def atr(df, n):
    df = df.copy()
    df[f"ATR"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], int(n)).average_true_range()
    return df

def sto_rsi(df, col, n):
    df = df.copy()

    StoRsi = ta.momentum.StochRSIIndicator(df[col], int(n))
    df[f"STO_RSI"] = StoRsi.stochrsi() * 100
    df[f"STO_RSI_D"] = StoRsi.stochrsi_d() * 100
    df[f"STO_RSI_K"] = StoRsi.stochrsi_k() * 100
    return df

def apply_ichimoku(df: pd.DataFrame, n1: int = 9, n2: int = 26, n3: int =52) -> pd.DataFrame:
    """
    Applies the Ichimoku Cloud indicator to an OHLC DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame with 'high' and 'low' columns.
    - n1 (int): Conversion line period (usually 9).
    - n2 (int): Base line period (usually 26).
    - n3 (int): Conversion line period (usually 52).

    Returns:
    - pd.DataFrame: Original df with Ichimoku columns added:
        ['CONVERSION', 'BASE', 'SPAN_A', 'SPAN_B']
    """
    df = df.copy()

    # Initialize IchimokuIndicator
    ichimoku = IchimokuIndicator(
        high=df["high"],
        low=df["low"],
        window1=n1,
        window2=n2,
        window3=n3  # Standard default for SPAN_B (Senkou Span B)
    )

    # Add Ichimoku components
    df["CONVERSION"] = ichimoku.ichimoku_conversion_line()
    df["BASE"] = ichimoku.ichimoku_base_line()
    df["SPAN_A"] = ichimoku.ichimoku_a()
    df["SPAN_B"] = ichimoku.ichimoku_b()

    return df

def previous_ret(df,col,n):
    df = df.copy()
    df["previous_ret"] = (df[col].shift(int(n)) - df[col]) / df[col]
    return df

def k_enveloppe(df, n=10):
    df = df.copy()
    df[f"EMA_HIGH_{n}"] = df["high"].ewm(span=n).mean()
    df[f"EMA_LOW_{n}"] = df["low"].ewm(span=n).mean()

    df["pivots_high"] = (df["close"] - df[f"EMA_HIGH_{n}"])/ df[f"EMA_HIGH_{n}"]
    df["pivots_low"] = (df["close"] - df[f"EMA_LOW_{n}"])/ df[f"EMA_LOW_{n}"]
    df["pivots"] = (df["pivots_high"]+df["pivots_low"])/2
    return df

def candle_information(df):
    df = df.copy()
    # Candle color
    df["candle_way"] = -1
    df.loc[(df["open"] - df["close"]) < 0, "candle_way"] = 1

    # Filling percentage
    df["filling"] = np.abs(df["close"] - df["open"]) / np.abs(df["high"] - df["low"])

    # Amplitude
    df["amplitude"] = np.abs(df["close"] - df["open"]) / (df["open"] / 2 + df["close"] / 2) * 100

    return df

def data_split(df_model, split, list_x, list_y):
    df = df.copy()
    # Train set creation
    X_train = df_model[list_x].iloc[0:split-1, :]
    y_train = df_model[list_y].iloc[1:split]

    # Test set creation
    X_test = df_model[list_x].iloc[split:-1, :]
    y_test = df_model[list_y].iloc[split+1:]

    return X_train, X_test, y_train, y_test

def quantile_signal(df, n, quantile_level=0.67,pct_split=0.8):
    df = df.copy()
    n = int(n)

    # Create the split between train and test set to do not create a Look-Ahead bais
    split = int(len(df) * pct_split)

    # Copy the dataframe to do not create any intereference
    df_copy = df.copy()

    # Create the fut_ret column to be able to create the signal
    df_copy["fut_ret"] = (df_copy["close"].shift(-n) - df_copy["open"]) / df_copy["open"]

    # Create a column by name, 'Signal' and initialize with 0
    df_copy['Signal'] = 0

    # Assign a value of 1 to 'Signal' column for the quantile with the highest returns
    df_copy.loc[df_copy['fut_ret'] > df_copy['fut_ret'][:split].quantile(q=quantile_level), 'Signal'] = 1

    # Assign a value of -1 to 'Signal' column for the quantile with the lowest returns
    df_copy.loc[df_copy['fut_ret'] < df_copy['fut_ret'][:split].quantile(q=1-quantile_level), 'Signal'] = -1

    return df_copy

def binary_signal(df, n):
    df = df.copy()
    n = int(n)

    # Copy the dataframe to do not create any intereference
    df_copy = df.copy()

    # Create the fut_ret column to be able to create the signal
    df_copy["fut_ret"] = (df_copy["close"].shift(-n) - df_copy["open"]) / df_copy["open"]

    # Create a column by name, 'Signal' and initialize with 0
    df_copy['Signal'] = -1

    # Assign a value of 1 to 'Signal' column for the quantile with the highest returns
    df_copy.loc[df_copy['fut_ret'] > 0, 'Signal'] = 1

    return df_copy

def apply_kama_indicator(df: pd.DataFrame, er_period: int = 10, fast_ema: int = 2, slow_ema: int = 30) -> pd.DataFrame:
    """
    Applies the Kaufman Adaptive Moving Average (KAMA) to a DataFrame.

    Parameters:
    - df (pd.DataFrame): OHLCV DataFrame with a 'close' column.
    - er_period (int): Period used to calculate the efficiency ratio.
    - fast_ema (int): Fastest EMA smoothing constant.
    - slow_ema (int): Slowest EMA smoothing constant.

    Returns:
    - pd.DataFrame: Original df with 'kama' column added.
    """
    close = df['close']
    er_period = int(er_period)
    # fast_ema = int(fast_ema)
    # slow_ema = int(slow_ema)

    # 1. Efficiency Ratio (ER)
    change = abs(close - close.shift(er_period))
    volatility = close.diff().abs().rolling(er_period).sum()
    er = change / volatility
    er.replace([np.inf, -np.inf], 0, inplace=True)
    er.fillna(0, inplace=True)

    # 2. Smoothing Constant (SC)
    fast_sc = 2 / (fast_ema + 1)
    slow_sc = 2 / (slow_ema + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    # 3. Initialize KAMA with first close
    kama = [close.iloc[0]]
    for i in range(1, len(close)):
        kama.append(kama[-1] + sc.iloc[i] * (close.iloc[i] - kama[-1]))

    df['kama'] = kama
    return df