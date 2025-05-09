import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
import argparse
import os

def ensure_images_dir():
    """Create images directory if it doesn't exist."""
    images_dir = 'images'
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    return images_dir

def load_transactions(file_path):
    """Load transactions from CSV file."""
    df = pd.read_csv(file_path)
    # Convert date string to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    # Sort by date
    df = df.sort_values('Date')
    # Calculate price per share for each transaction
    df['Price'] = df['Amount'] / df['Shares']
    return df

def calculate_portfolio_stats(transactions, current_price=None):
    """Calculate portfolio statistics over time with simplified approach."""
    results = []
    
    # Initialize tracking variables
    shares_owned = 0
    total_cost = 0
    realized_pl = 0
    total_invested = 0
    total_shares_bought = 0
    
    # Detailed transaction summary
    print("=== Detailed Transaction Log ===")
    print(f"{'Date':12} {'Type':10} {'Shares':8} {'Amount':10} {'Price/Share':12}")
    print(f"{'-'*50}")
    
    for idx, row in transactions.iterrows():
        date = row['Date']
        transaction_type = row['Type']
        shares = row['Shares']
        amount = row['Amount']
        fee = row['Fee']
        price = row['Price']
        
        # For the last entry, use provided current price if available
        is_last_entry = idx == len(transactions) - 1
        if is_last_entry and current_price is not None:
            current_price_to_use = current_price
        else:
            current_price_to_use = price
        
        # Print transaction details
        print(f"{date.strftime('%Y-%m-%d'):12} {transaction_type:10} {shares:8.0f} ${amount:8.2f} ${price:10.2f}")
        
        # Update portfolio based on transaction type
        if transaction_type == 'Limit Buy':
            # Buy shares
            total_cost += amount + fee
            shares_owned += shares
            total_invested += amount + fee
            total_shares_bought += shares
        elif transaction_type == 'Limit Sell':
            # Sell shares - calculate realized P/L based on average cost
            if shares_owned > 0:
                # Calculate average cost before sell
                avg_cost = total_cost / shares_owned
                # Calculate realized P/L for this transaction
                sale_proceeds = amount - fee
                cost_of_shares_sold = shares * avg_cost
                transaction_pl = sale_proceeds - cost_of_shares_sold
                realized_pl += transaction_pl
                
                # Update shares and adjust total cost proportionally
                proportion_sold = shares / shares_owned
                total_cost = total_cost * (1 - proportion_sold)
                shares_owned -= shares
        
        # Calculate current values (using the current price)
        current_value = shares_owned * current_price_to_use
        
        # Calculate average cost basis
        avg_cost_basis = total_cost / shares_owned if shares_owned > 0 else 0
        
        # Calculate unrealized P/L
        unrealized_pl = current_value - total_cost
        unrealized_pl_pct = (unrealized_pl / total_cost * 100) if total_cost > 0 else 0
        
        # Total P/L (unrealized + realized)
        total_pl = unrealized_pl + realized_pl
        
        # Calculate effective cost basis (accounting for realized P/L)
        # This shows what your real cost basis would be if you factored in realized gains/losses
        if shares_owned > 0:
            effective_cost = total_cost - realized_pl
            effective_cost_basis = effective_cost / shares_owned if effective_cost > 0 else 0
        else:
            effective_cost = 0
            effective_cost_basis = 0
            
        # Calculate lifetime average cost (total invested / total shares bought)
        lifetime_avg_cost = total_invested / total_shares_bought if total_shares_bought > 0 else 0
        
        results.append({
            'Date': date,
            'Shares': shares_owned,
            'Total Cost': total_cost,
            'Avg Cost Basis': avg_cost_basis,
            'Effective Cost Basis': effective_cost_basis,
            'Lifetime Avg Cost': lifetime_avg_cost,
            'Current Price': current_price_to_use,
            'Current Value': current_value,
            'Unrealized P/L': unrealized_pl,
            'Unrealized P/L %': unrealized_pl_pct,
            'Realized P/L': realized_pl,
            'Total P/L': total_pl
        })
    
    # Print final position
    print(f"\nFinal position: {shares_owned} shares with total cost ${total_cost:.2f}")
    print(f"Total realized P/L: ${realized_pl:.2f}")
    if current_price is not None:
        print(f"Using current price: ${current_price:.2f} per share (user provided)")
    
    return pd.DataFrame(results)

def plot_portfolio_charts(stats):
    """Create portfolio performance charts."""
    # Ensure images directory exists
    images_dir = ensure_images_dir()
    
    # Chart 1: Portfolio Value & Cost
    plt.figure(figsize=(12, 7))
    plt.plot(stats['Date'], stats['Current Value'], marker='o', linestyle='-', linewidth=2, color='#4CAF50', label='Current Value')
    plt.plot(stats['Date'], stats['Total Cost'], marker='x', linestyle='--', linewidth=1.5, color='#F44336', label='Total Cost')
    
    # Format x-axis date labels
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    
    plt.title('Portfolio Value vs Cost', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Amount ($)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'portfolio_value.png'), dpi=100)
    plt.close()
    
    # Chart 2: Profit/Loss (Unrealized, Realized, and Total)
    plt.figure(figsize=(12, 7))
    plt.plot(stats['Date'], stats['Unrealized P/L'], marker='o', linestyle='-', linewidth=2, color='#2196F3', label='Unrealized P/L')
    plt.plot(stats['Date'], stats['Realized P/L'], marker='s', linestyle='-', linewidth=2, color='#FF9800', label='Realized P/L')
    plt.plot(stats['Date'], stats['Total P/L'], marker='x', linestyle='--', linewidth=2.5, color='#4CAF50', label='Total P/L')
    
    # Add zero line
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Format x-axis date labels
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    
    plt.title('Portfolio Profit/Loss Over Time', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Profit/Loss ($)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'profit_loss.png'), dpi=100)
    plt.close()
    
    # Chart 3: Share Count Evolution
    plt.figure(figsize=(12, 7))
    plt.plot(stats['Date'], stats['Shares'], marker='o', linestyle='-', linewidth=2, color='#673AB7')
    plt.fill_between(stats['Date'], stats['Shares'], color='#673AB7', alpha=0.2)
    
    # Format x-axis date labels
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    
    plt.title('Total Shares Over Time', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Number of Shares', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'shares_count.png'), dpi=100)
    plt.close()
    
    # Chart 4: Cost Basis Evolution (accounting for both realized and unrealized P/L)
    plt.figure(figsize=(12, 7))
    plt.plot(stats['Date'], stats['Avg Cost Basis'], marker='o', linestyle='-', linewidth=2, color='#9C27B0', label='Standard Cost Basis')
    plt.plot(stats['Date'], stats['Effective Cost Basis'], marker='s', linestyle='--', linewidth=2, color='#E91E63', label='Effective Cost Basis (incl. Realized P/L)')
    plt.plot(stats['Date'], stats['Lifetime Avg Cost'], marker='x', linestyle='-.', linewidth=1.5, color='#795548', label='Lifetime Average Cost')
    plt.plot(stats['Date'], stats['Current Price'], marker='.', linestyle=':', linewidth=1.5, color='#607D8B', label='Current Price')
    
    # Add annotations for significant changes
    for i, row in stats.iterrows():
        if i > 0:
            prev_row = stats.iloc[i-1]
            # If there's a significant change in cost basis
            if abs(row['Avg Cost Basis'] - prev_row['Avg Cost Basis']) > 0.05 and row['Shares'] > 0:
                plt.annotate(f"${row['Avg Cost Basis']:.2f}", 
                             xy=(row['Date'], row['Avg Cost Basis']),
                             xytext=(5, 5),
                             textcoords='offset points',
                             fontsize=8)
    
    # Format x-axis date labels
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    
    plt.title('Cost Basis Evolution Over Time', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cost per Share ($)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.legend(fontsize=9, loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'cost_basis_evolution.png'), dpi=100)
    plt.close()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Portfolio Tracking Tool')
    parser.add_argument('-p', '--price', type=float, help='Current price per share (optional)')
    parser.add_argument('-f', '--file', type=str, default='transactions.csv', help='Transaction CSV file (default: transactions.csv)')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Get current price from arguments if provided
    current_price = args.price
    if current_price is not None:
        print(f"Using provided current price: ${current_price:.2f} per share\n")
    
    # Load transactions
    transactions = load_transactions(args.file)
    
    # Print transaction summary for verification
    print("=== Transaction Summary ===")
    buy_shares = transactions[transactions['Type'] == 'Limit Buy']['Shares'].sum()
    sell_shares = transactions[transactions['Type'] == 'Limit Sell']['Shares'].sum()
    net_shares = buy_shares - sell_shares
    
    print(f"Total Buy Shares: {buy_shares}")
    print(f"Total Sell Shares: {sell_shares}")
    print(f"Expected Net Shares: {net_shares}")
    print()
    
    # Calculate portfolio stats
    stats = calculate_portfolio_stats(transactions, current_price)
    
    # Display stats
    pd.set_option('display.float_format', '${:.2f}'.format)
    
    print("\n=== Portfolio Evolution ===")
    print(stats[['Date', 'Shares', 'Total Cost', 'Avg Cost Basis', 'Effective Cost Basis', 'Current Value', 'Unrealized P/L', 'Realized P/L', 'Total P/L']])
    
    # Summary of current position
    latest = stats.iloc[-1]
    
    print("\n=== Current Position ===")
    print(f"Shares Owned: {latest['Shares']:.0f}")
    print(f"Total Cost: ${latest['Total Cost']:.2f}")
    print(f"Avg Cost Basis: ${latest['Avg Cost Basis']:.2f} per share")
    print(f"Effective Cost Basis (incl. Realized P/L): ${latest['Effective Cost Basis']:.2f} per share")
    print(f"Lifetime Average Cost: ${latest['Lifetime Avg Cost']:.2f} per share")
    print(f"Current Price: ${latest['Current Price']:.2f} per share")
    print(f"Current Value: ${latest['Current Value']:.2f}")
    print(f"Unrealized P/L: ${latest['Unrealized P/L']:.2f} ({latest['Unrealized P/L %']:.2f}%)")
    print(f"Realized P/L: ${latest['Realized P/L']:.2f}")
    print(f"Total P/L: ${latest['Total P/L']:.2f}")
    
    # Plot charts
    plot_portfolio_charts(stats)
    print("\nCharts saved in 'images' folder:")
    print("- 'portfolio_value.png' (Portfolio value vs cost)")
    print("- 'profit_loss.png' (Profit/loss over time)")
    print("- 'shares_count.png' (Share count over time)")
    print("- 'cost_basis_evolution.png' (Cost basis evolution)")

if __name__ == "__main__":
    main()
