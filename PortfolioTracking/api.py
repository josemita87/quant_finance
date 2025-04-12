from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import portfolio as portfolio_analyzer
import os
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get basic portfolio metrics."""
    # Get optional current price from query parameters
    current_price = request.args.get('current_price', type=float)
    
    # Initialize analyzer with optional current price
    analyzer = portfolio_analyzer.PortfolioAnalyzer('transactions.csv', current_price)
    
    # Get metrics
    basic_metrics = analyzer.calculate_basic_metrics()
    returns = analyzer.calculate_portfolio_returns()
    patterns = analyzer.analyze_trading_patterns()
    
    # Format dates for JSON serialization
    returns_data = returns['returns_data']
    for entry in returns_data:
        entry['date'] = entry['date'].strftime('%Y-%m-%d')
    
    # Combine all metrics
    all_metrics = {
        'basic_metrics': basic_metrics,
        'returns': returns,
        'patterns': patterns
    }
    
    return jsonify(all_metrics)

@app.route('/api/transactions', methods=['GET'])
def get_transactions():
    """Get raw transaction data."""
    analyzer = portfolio_analyzer.PortfolioAnalyzer('transactions.csv')
    
    # Convert DataFrame to dict for JSON serialization
    transactions = analyzer.df.to_dict(orient='records')
    
    # Format dates for JSON serialization
    for transaction in transactions:
        transaction['Date'] = transaction['Date'].strftime('%Y-%m-%d')
    
    return jsonify(transactions)

@app.route('/api/charts/portfolio_analysis', methods=['GET'])
def get_portfolio_analysis():
    """Generate and return combined portfolio analysis chart."""
    # Get optional current price from query parameters
    current_price = request.args.get('current_price', type=float)
    refresh = request.args.get('refresh') == 'true'
    
    # Create the visualization if it doesn't exist or refresh is requested
    if not os.path.exists('portfolio_analysis.png') or refresh:
        analyzer = portfolio_analyzer.PortfolioAnalyzer('transactions.csv', current_price)
        analyzer.plot_portfolio_analysis()
    
    return send_file('portfolio_analysis.png', mimetype='image/png')

@app.route('/api/price/update', methods=['POST'])
def update_price():
    """Update the current price for analysis."""
    data = request.get_json()
    current_price = data.get('price')
    
    if current_price is None:
        return jsonify({'error': 'Price not provided'}), 400
    
    # Generate analysis with updated price
    analyzer = portfolio_analyzer.PortfolioAnalyzer('transactions.csv', current_price)
    returns = analyzer.calculate_portfolio_returns()
    
    return jsonify({
        'price': current_price,
        'returns': returns
    })

if __name__ == '__main__':
    app.run(debug=True) 