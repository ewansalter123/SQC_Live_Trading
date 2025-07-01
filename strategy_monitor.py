import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'strategy_1')))

from strategy_1.mt5_interface import MT5Interface
import pandas as pd
import time
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
import MetaTrader5 as mt5

console = Console()
mt5_int = MT5Interface(broker="icmarkets")

# === CONFIGURATION ===
symbol = "GBPNZD"               # Symbol being traded
magic = 776655432166             # Your strategy's unique magic number
refresh_interval = 30           # Seconds between updates

def display_status():
    account = mt5.account_info()
    all_positions = mt5_int.get_open_positions(symbol=symbol)

    tick = mt5.symbol_info_tick(symbol)
    if tick:
        broker_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(tick.time))
    else:
        broker_time = "Unavailable"

    console.clear()

    # Aesthetic Section Break
    console.rule(f"[bold green]Broker Time: {broker_time}")
    console.rule("[bold green]Live Account & Strategy Monitor")

    # === ACCOUNT SUMMARY & RISK ===
    table = Table(title="[bold green]Account Summary & Risk Metrics", box=box.SQUARE)
    table.add_column("Metric", style="cyan", justify="left")
    table.add_column("Value", style="magenta", justify="right")

    table.add_row("Balance", f"${account.balance:,.2f}")
    table.add_row("Equity", f"${account.equity:,.2f}")
    table.add_row("Free Margin", f"${account.margin_free:,.2f}")
    table.add_row("Margin Used", f"${account.margin:,.2f}")

    # Margin Level %
    if account.margin > 0:
        margin_level = (account.equity / account.margin) * 100
    else:
        margin_level = 0

    # Color-Coding Based on Risk
    if margin_level < 100:
        margin_str = f"[bold red]{margin_level:.2f}%[/]"
    elif margin_level < 300:
        margin_str = f"[bold yellow]{margin_level:.2f}%[/]"
    else:
        margin_str = f"[bold green]{margin_level:.2f}%[/]"

    table.add_row("Margin Level", margin_str)

    table.add_row("Total Open Positions", str(len(all_positions)))

    total_pnl = sum([pos['profit'] for pos in all_positions])
    table.add_row("Total PnL (All Open Trades)", f"${total_pnl:,.2f}")

    console.print(table)

    # Aesthetic Section Break
    console.rule("[bold yellow]Multi-Strategy Breakdown (By Magic Number)")

    # Find unique magic numbers in current open positions
    magic_numbers = sorted(set(pos['magic'] for pos in all_positions))

    if magic_numbers:
        magic_table = Table(title="[bold yellow]PnL & Position Count by Magic", box=box.ROUNDED)
        magic_table.add_column("Magic Number", justify="right", style="cyan")
        magic_table.add_column("Open Positions", justify="center")
        magic_table.add_column("Total PnL", justify="right", style="magenta")

        for m in magic_numbers:
            magic_positions = [pos for pos in all_positions if pos['magic'] == m]
            pnl = sum(pos['profit'] for pos in magic_positions)
            magic_table.add_row(str(m), str(len(magic_positions)), f"${pnl:.2f}")

        console.print(magic_table)
    else:
        console.print(Panel("[green]No open positions detected.[/]", expand=False))


    # Aesthetic Section Break
    console.rule("[bold blue]Strategy-Specific Breakdown")


    # === FILTERED BY MAGIC ===
    positions = [pos for pos in all_positions if pos['magic'] == magic]

    pnl_magic = sum([pos['profit'] for pos in positions])

    strategy_table = Table(title=f"[bold blue]PnL for Magic #{magic}", box=box.ROUNDED)
    strategy_table.add_column("Metric", style="cyan")
    strategy_table.add_column("Value", style="magenta")

    strategy_table.add_row("Open Positions", str(len(positions)))
    strategy_table.add_row("Total PnL (This Strategy)", f"${pnl_magic:.2f}")

    console.print(strategy_table)

    # === POSITION DETAILS IF OPEN ===
    if positions:
        pos_table = Table(title="Open Trades (Filtered by Magic)", box=box.ROUNDED)
        pos_table.add_column("Ticket")
        pos_table.add_column("Type")
        pos_table.add_column("Size")
        pos_table.add_column("Price")
        pos_table.add_column("Profit", justify="right")

        for pos in positions:
            price = pos.get('price', None)
            price_str = f"{price:.5f}" if price is not None else "N/A"

            pos_table.add_row(
                str(pos.get('ticket', 'N/A')),
                "Buy" if pos.get('type', -1) == 0 else "Sell" if pos.get('type', -1) == 1 else "N/A",
                f"{pos.get('volume', 0):.2f}",
                price_str,
                f"${pos.get('profit', 0):.2f}"
            )

        console.print(pos_table)
    else:
        console.print(Panel("[green]No open positions matching magic number.[/]", expand=False))

    # Final aesthetic footer break
    console.rule("[dim]End of Update — Awaiting next refresh")


if __name__ == "__main__":
    console.print(Panel("[bold yellow]Strategy Monitor Started. Press Ctrl+C to stop.[/]", expand=False))
    while True:
        try:
            display_status()
            time.sleep(refresh_interval)
        except KeyboardInterrupt:
            console.print("\n[red]Monitor stopped by user.[/]")
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/] {e}")
            time.sleep(refresh_interval)
