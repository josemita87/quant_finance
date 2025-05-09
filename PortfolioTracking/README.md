# Portfolio Tracking Tool

A simple Python tool to track investment portfolio statistics and visualize unrealized value over time.

## Features

- Calculate cumulative shares and average cost basis
- Track unrealized value over time
- Generate portfolio performance statistics
- Visualize portfolio value with a time-series chart

## Usage

1. Ensure your transaction data is in CSV format with the following columns:
   - Date (DD/MM/YYYY)
   - Type (Limit Buy/Limit Sell)
   - Amount (transaction value)
   - Shares (number of shares)
   - Fee (transaction fee)

2. Run the script:
   ```
   python portfolio.py
   ```

3. The script will:
   - Display portfolio statistics over time
   - Show current position summary
   - Generate a chart of portfolio value saved as 'portfolio_value.png'

## Requirements

- Python 3.6+
- pandas
- numpy
- matplotlib

## Sample Output

The script generates:
- A table showing portfolio statistics over time
- A summary of your current position
- A chart visualizing the unrealized value over time
