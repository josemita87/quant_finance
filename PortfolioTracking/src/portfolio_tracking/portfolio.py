"""Utilities for analysing and visualising personal portfolio transactions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent
IMAGES_DIR = DATA_DIR / "images"


@dataclass(frozen=True)
class PortfolioSnapshot:
    """Aggregated portfolio state after each transaction."""

    date: pd.Timestamp
    shares: float
    total_cost: float
    current_price: float
    current_value: float
    unrealised_pl: float
    realised_pl: float

    @property
    def total_pl(self) -> float:
        """Return combined unrealised and realised profit or loss."""

        return self.unrealised_pl + self.realised_pl

    @property
    def average_cost(self) -> float:
        """Return average cost basis for remaining shares."""

        return 0.0 if self.shares == 0 else self.total_cost / self.shares


def load_transactions(path: Path) -> pd.DataFrame:
    """Load transactions from a CSV file sorted by date."""

    frame = pd.read_csv(path)
    frame["Date"] = pd.to_datetime(frame["Date"], format="%d/%m/%Y")
    frame = frame.sort_values("Date").reset_index(drop=True)
    frame["Price"] = frame["Amount"] / frame["Shares"]
    return frame


def compute_snapshots(transactions: pd.DataFrame, override_price: Optional[float] = None) -> list[PortfolioSnapshot]:
    """Compute running snapshots after each transaction."""

    snapshots: list[PortfolioSnapshot] = []
    shares = 0.0
    total_cost = 0.0
    realised_pl = 0.0

    for index, row in transactions.iterrows():
        price = float(row["Price"]) if override_price is None or index != len(transactions) - 1 else override_price
        if row["Type"].lower().startswith("limit buy"):
            total_cost += float(row["Amount"]) + float(row["Fee"]) 
            shares += float(row["Shares"])
        elif row["Type"].lower().startswith("limit sell") and shares > 0:
            avg_cost = total_cost / shares if shares else 0.0
            sale_proceeds = float(row["Amount"]) - float(row["Fee"]) 
            realised_pl += sale_proceeds - float(row["Shares"]) * avg_cost
            shares -= float(row["Shares"])
            total_cost = avg_cost * shares
        current_value = shares * price
        unrealised_pl = current_value - total_cost
        snapshots.append(
            PortfolioSnapshot(
                date=row["Date"],
                shares=shares,
                total_cost=total_cost,
                current_price=price,
                current_value=current_value,
                unrealised_pl=unrealised_pl,
                realised_pl=realised_pl,
            )
        )
    return snapshots


def snapshots_to_frame(snapshots: list[PortfolioSnapshot]) -> pd.DataFrame:
    """Convert snapshot objects into a Pandas DataFrame for plotting."""

    return pd.DataFrame(
        {
            "Date": [snapshot.date for snapshot in snapshots],
            "Shares": [snapshot.shares for snapshot in snapshots],
            "Total Cost": [snapshot.total_cost for snapshot in snapshots],
            "Current Price": [snapshot.current_price for snapshot in snapshots],
            "Current Value": [snapshot.current_value for snapshot in snapshots],
            "Unrealised P/L": [snapshot.unrealised_pl for snapshot in snapshots],
            "Realised P/L": [snapshot.realised_pl for snapshot in snapshots],
            "Total P/L": [snapshot.total_pl for snapshot in snapshots],
            "Average Cost": [snapshot.average_cost for snapshot in snapshots],
        }
    )


def plot_performance(frame: pd.DataFrame) -> None:
    """Generate value and P/L charts from a transaction dataframe."""

    IMAGES_DIR.mkdir(exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(frame["Date"], frame["Current Value"], label="Current Value", marker="o")
    plt.plot(frame["Date"], frame["Total Cost"], label="Total Cost", linestyle="--")
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45)
    plt.title("Portfolio Value vs Cost")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "portfolio_value.png", dpi=100)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(frame["Date"], frame["Unrealised P/L"], label="Unrealised P/L", marker="o")
    plt.plot(frame["Date"], frame["Realised P/L"], label="Realised P/L", marker="s")
    plt.plot(frame["Date"], frame["Total P/L"], label="Total P/L", marker="^")
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45)
    plt.title("Profit and Loss Breakdown")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "portfolio_pl.png", dpi=100)
    plt.close()


__all__ = [
    "DATA_DIR",
    "IMAGES_DIR",
    "PortfolioSnapshot",
    "compute_snapshots",
    "load_transactions",
    "plot_performance",
    "snapshots_to_frame",
]
