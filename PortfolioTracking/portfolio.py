import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from typing import Tuple, Dict
import numpy as np
import argparse

class PortfolioAnalyzer:
    def __init__(self, transactions_file: str, current_price: float = None):
        """Initialize the portfolio analyzer with a transactions file and optional current price."""
        self.df = pd.read_csv(transactions_file)
        # Clean column names by stripping whitespace
        self.df.columns = self.df.columns.str.strip()
        self.df['Date'] = pd.to_datetime(self.df['Date'], format='%m/%d/%Y')
        self.df = self.df.sort_values('Date')
        
        # Calculate price per share for each transaction
        self.df['Price_Per_Share'] = abs(self.df['Amount'] / self.df['Shares'])
        self.current_price = current_price
        
    def calculate_basic_metrics(self) -> Dict:
        """Calculate basic portfolio metrics."""
        total_investment = abs(self.df[self.df['Transaction Type'] == 'Limit Buy']['Amount'].sum())
        total_sales = self.df[self.df['Transaction Type'] == 'Limit Sell']['Amount'].sum()
        total_fees = self.df['Fee'].sum()
        current_shares = self.df['Shares'].sum()
        
        return {
            'total_investment': total_investment,
            'total_sales': total_sales,
            'total_fees': total_fees,
            'current_shares': current_shares,
            'realized_pl': total_sales - total_fees
        }
    
    def calculate_portfolio_returns(self) -> Dict:
        """Calculate detailed portfolio returns and percentage changes."""
        # Initialize running calculations
        running_shares = 0
        running_cost_basis = 0
        total_realized_return = 0
        returns_data = []
        
        for _, row in self.df.iterrows():
            transaction_amount = abs(row['Amount'])
            shares = abs(row['Shares'])
            price = row['Price_Per_Share']
            
            if row['Transaction Type'] == 'Limit Buy':
                # Update cost basis and shares for buys
                running_cost_basis += transaction_amount
                running_shares += shares
                avg_cost = running_cost_basis / running_shares if running_shares > 0 else 0
                
                returns_data.append({
                    'date': row['Date'],
                    'action': 'Buy',
                    'shares': running_shares,
                    'price': price,
                    'avg_cost': avg_cost,
                    'unrealized_return_pct': ((price / avg_cost) - 1) * 100 if avg_cost > 0 else 0
                })
            else:
                # Calculate realized returns for sells
                if running_shares > 0:
                    avg_cost = running_cost_basis / running_shares
                    realized_return_pct = ((price / avg_cost) - 1) * 100
                    realized_return = (price - avg_cost) * shares
                    total_realized_return += realized_return
                    
                    # Adjust cost basis and shares
                    cost_basis_per_share = running_cost_basis / running_shares
                    running_cost_basis -= (shares * cost_basis_per_share)
                    running_shares -= shares
                    
                    returns_data.append({
                        'date': row['Date'],
                        'action': 'Sell',
                        'shares': running_shares,
                        'price': price,
                        'avg_cost': avg_cost,
                        'realized_return_pct': realized_return_pct,
                        'realized_return': realized_return
                    })
        
        # Calculate final position metrics using current price if available
        final_price = self.current_price if self.current_price is not None else self.df.iloc[-1]['Price_Per_Share']
        final_avg_cost = running_cost_basis / running_shares if running_shares > 0 else 0
        final_unrealized_return_pct = ((final_price / final_avg_cost) - 1) * 100 if final_avg_cost > 0 else 0
        unrealized_pl = (final_price - final_avg_cost) * running_shares if running_shares > 0 else 0
        
        # Calculate combined return metric
        total_investment = abs(self.df[self.df['Transaction Type'] == 'Limit Buy']['Amount'].sum())
        combined_return = (total_realized_return + unrealized_pl) / total_investment * 100 if total_investment > 0 else 0
        
        return {
            'returns_data': returns_data,
            'final_metrics': {
                'current_shares': running_shares,
                'avg_cost_basis': final_avg_cost,
                'last_price': final_price,
                'unrealized_return_pct': final_unrealized_return_pct,
                'unrealized_pl': unrealized_pl,
                'total_realized_return': total_realized_return,
                'combined_return_pct': combined_return
            }
        }
    
    def plot_portfolio_analysis(self):
        """Plot all portfolio analysis charts in a single figure."""
        plt.style.use('seaborn')
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Portfolio Returns (top left)
        ax1 = plt.subplot2grid((2, 2), (0, 0))
        returns_analysis = self.calculate_portfolio_returns()
        returns_data = returns_analysis['returns_data']
        df_plot = pd.DataFrame(returns_data)
        
        # Plot price vs average cost
        ax1.plot(df_plot['date'], df_plot['price'], label='Transaction Price', marker='o')
        ax1.plot(df_plot['date'], df_plot['avg_cost'], label='Average Cost Basis', linestyle='--')
        
        if self.current_price is not None:
            ax1.axhline(y=self.current_price, color='r', linestyle=':', 
                       label=f'Current Price (${self.current_price:.2f})')
        
        ax1.set_title('Portfolio Returns Analysis')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 2. Returns Percentage (top right)
        ax2 = plt.subplot2grid((2, 2), (0, 1))
        realized_returns = [r.get('realized_return_pct', None) for r in returns_data]
        unrealized_returns = [r.get('unrealized_return_pct', None) for r in returns_data]
        
        ax2.plot(df_plot['date'], unrealized_returns, label='Unrealized Returns %', marker='o')
        realized_points = [(date, ret) for date, ret in zip(df_plot['date'], realized_returns) if ret is not None]
        if realized_points:
            dates, returns = zip(*realized_points)
            ax2.scatter(dates, returns, label='Realized Returns %', marker='^', color='red', s=100)
        
        if self.current_price is not None:
            final_metrics = returns_analysis['final_metrics']
            ax2.axhline(y=final_metrics['unrealized_return_pct'], color='r', linestyle=':', 
                       label=f'Current Return ({final_metrics["unrealized_return_pct"]:.2f}%)')
            # Add combined return indicator
            ax2.axhline(y=final_metrics['combined_return_pct'], color='g', linestyle='--',
                       label=f'Combined Return ({final_metrics["combined_return_pct"]:.2f}%)')
        
        ax2.set_title('Returns Percentage Over Time')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Return (%)')
        ax2.legend()
        ax2.grid(True)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # 3. Portfolio Evolution (bottom left)
        ax3 = plt.subplot2grid((2, 2), (1, 0))
        self.df['Cumulative Shares'] = self.df['Shares'].cumsum()
        ax3.plot(self.df['Date'], self.df['Cumulative Shares'], marker='o')
        ax3.set_title('Portfolio Size Evolution')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Number of Shares')
        ax3.grid(True)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        # 4. Transaction Distribution (bottom right)
        ax4 = plt.subplot2grid((2, 2), (1, 1))
        buys = self.df[self.df['Transaction Type'] == 'Limit Buy']['Amount'].abs()
        sells = self.df[self.df['Transaction Type'] == 'Limit Sell']['Amount']
        ax4.hist([buys, sells], label=['Buys', 'Sells'], bins=10)
        ax4.set_title('Distribution of Transaction Amounts')
        ax4.set_xlabel('Amount ($)')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig('portfolio_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_trading_patterns(self) -> Dict:
        """Analyze trading patterns and frequencies."""
        buys = self.df[self.df['Transaction Type'] == 'Limit Buy']
        sells = self.df[self.df['Transaction Type'] == 'Limit Sell']
        
        avg_buy_size = abs(buys['Amount'].mean())
        avg_sell_size = sells['Amount'].mean() if not sells.empty else 0
        
        monthly_trades = self.df.groupby(self.df['Date'].dt.to_period('M')).size()
        
        return {
            'total_trades': len(self.df),
            'buy_trades': len(buys),
            'sell_trades': len(sells),
            'avg_buy_size': avg_buy_size,
            'avg_sell_size': avg_sell_size,
            'most_active_month': monthly_trades.idxmax(),
            'trades_in_most_active_month': monthly_trades.max()
        }
    
    def plot_portfolio_evolution(self):
        """Plot the evolution of the portfolio over time."""
        plt.figure(figsize=(12, 6))
        
        # Calculate cumulative shares
        self.df['Cumulative Shares'] = self.df['Shares'].cumsum()
        
        # Plot cumulative shares
        plt.plot(self.df['Date'], self.df['Cumulative Shares'], marker='o')
        plt.title('Portfolio Size Evolution Over Time')
        plt.xlabel('Date')
        plt.ylabel('Number of Shares')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('portfolio_evolution.png')
        plt.close()
    
    def plot_transaction_distribution(self):
        """Plot the distribution of transaction amounts."""
        plt.figure(figsize=(10, 6))
        
        # Create separate plots for buys and sells
        buys = self.df[self.df['Transaction Type'] == 'Limit Buy']['Amount'].abs()
        sells = self.df[self.df['Transaction Type'] == 'Limit Sell']['Amount']
        
        plt.hist([buys, sells], label=['Buys', 'Sells'], bins=10)
        plt.title('Distribution of Transaction Amounts')
        plt.xlabel('Amount ($)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.savefig('transaction_distribution.png')
        plt.close()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze portfolio transactions')
    parser.add_argument('--price', type=float, help='Current market price', default=None)
    args = parser.parse_args()
    
    # Initialize analyzer with optional current price
    analyzer = PortfolioAnalyzer('transactions.csv', args.price)
    
    # Calculate and display basic metrics
    metrics = analyzer.calculate_basic_metrics()
    print("\n=== Portfolio Metrics ===")
    print(f"Total Investment: ${metrics['total_investment']:.2f}")
    print(f"Total Sales: ${metrics['total_sales']:.2f}")
    print(f"Total Fees: ${metrics['total_fees']:.2f}")
    print(f"Current Shares: {metrics['current_shares']}")
    print(f"Realized P/L: ${metrics['realized_pl']:.2f}")
    
    # Calculate and display returns metrics
    returns = analyzer.calculate_portfolio_returns()
    final_metrics = returns['final_metrics']
    print("\n=== Portfolio Returns Analysis ===")
    print(f"Current Position: {final_metrics['current_shares']:.0f} shares")
    print(f"Average Cost Basis: ${final_metrics['avg_cost_basis']:.2f}")
    print(f"{'Current' if args.price is not None else 'Last Transaction'} Price: ${final_metrics['last_price']:.2f}")
    print(f"Unrealized Return: {final_metrics['unrealized_return_pct']:.2f}%")
    print(f"Unrealized P/L: ${final_metrics['unrealized_pl']:.2f}")
    print(f"Total Realized Return: ${final_metrics['total_realized_return']:.2f}")
    print(f"Combined Return: {final_metrics['combined_return_pct']:.2f}%")
    
    # Analyze trading patterns
    patterns = analyzer.analyze_trading_patterns()
    print("\n=== Trading Patterns ===")
    print(f"Total Trades: {patterns['total_trades']}")
    print(f"Buy Trades: {patterns['buy_trades']}")
    print(f"Sell Trades: {patterns['sell_trades']}")
    print(f"Average Buy Size: ${patterns['avg_buy_size']:.2f}")
    print(f"Average Sell Size: ${patterns['avg_sell_size']:.2f}")
    print(f"Most Active Month: {patterns['most_active_month']}")
    print(f"Trades in Most Active Month: {patterns['trades_in_most_active_month']}")
    
    # Generate combined visualization
    analyzer.plot_portfolio_analysis()
    print("\n=== Visualization Generated ===")
    print("- Portfolio analysis saved as 'portfolio_analysis.png'")

if __name__ == "__main__":
    main()
