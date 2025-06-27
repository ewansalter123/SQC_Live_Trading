import MetaTrader5 as mt5
from datetime import datetime
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.style import Style

# Initialize
console = Console()
from_date = datetime(2022, 1, 1)
to_date = datetime.now()

# Connect to MT5
if not mt5.initialize():
    console.print(f"[bold red]MT5 initialization failed[/], error code: {mt5.last_error()}")
    exit()

# Get all trading deals
deals = mt5.history_deals_get(from_date, to_date)
mt5.shutdown()

if not deals:
    console.print(f"[bold red]No trades found or error retrieving them[/], code: {mt5.last_error()}")
    exit()

# Convert to DataFrame
df = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())
df['time'] = pd.to_datetime(df['time'], unit='s')
df['profit'] = df['profit'].astype(float)
df['volume'] = df['volume'].astype(float)

# Filter only buy/sell market trades
df = df[df['type'].isin([0, 1])]

# Compute Metrics
total_trades = len(df)
wins = df[df['profit'] > 0]
losses = df[df['profit'] < 0]
win_rate = len(wins) / total_trades if total_trades else 0
avg_win = wins['profit'].mean() if not wins.empty else 0
avg_loss = losses['profit'].mean() if not losses.empty else 0
total_pnl = df['profit'].sum()
sharpe_estimate = df['profit'].mean() / df['profit'].std() * (len(df) ** 0.5) if df['profit'].std() else 0

# === METRICS DASHBOARD ===
table = Table(title="🚀 [bold green]Strategy Performance Summary[/]", box=box.SQUARE, expand=False)
table.add_column("Metric", style="cyan bold", justify="left")
table.add_column("Value", style="magenta", justify="right")
table.add_row("Total Trades", str(total_trades))
table.add_row("Win Rate", f"{win_rate*100:.2f}%")
table.add_row("Average Win", f"${avg_win:.2f}")
table.add_row("Average Loss", f"${avg_loss:.2f}")
table.add_row("Total PnL", f"${total_pnl:.2f}")
table.add_row("Sharpe (Est.)", f"{sharpe_estimate:.2f}")

console.print(table)

# === LATEST BTCUSD TRADES ===
btc_df = df[df['symbol'] == 'BTCUSD'].sort_values(by='time', ascending=False).head(10)

if btc_df.empty:
    console.print("[yellow]No BTCUSD trades found.[/]")
else:
    btc_table = Table(title="[bold blue]Recent BTCUSD Trades", box=box.ROUNDED)
    btc_table.add_column("Time", style="dim")
    btc_table.add_column("Type", style="bold")
    btc_table.add_column("Volume")
    btc_table.add_column("Price")
    btc_table.add_column("Profit", justify="right")

    for _, row in btc_df.iterrows():
        ttype = "Buy" if row['type'] == 0 else "Sell"
        btc_table.add_row(
            row['time'].strftime("%Y-%m-%d %H:%M"),
            ttype,
            f"{row['volume']:.2f}",
            f"{row['price']:.2f}",
            f"${row['profit']:.2f}"
        )

    console.print(btc_table)
