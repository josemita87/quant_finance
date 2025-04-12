# Quantitative Finance Portfolio Optimization

## Overview
This repository implements an advanced portfolio optimization system using metaheuristic algorithms. The implementation focuses on efficient, vectorized operations in Python for portfolio optimization using Simulated Annealing and Tabu Search algorithms, with multiple objective functions including Value at Risk (VaR), Sharpe Ratio, and Maximum Drawdown (MDD).

## Key Features

### Optimization Algorithms
- **Simulated Annealing**: Temperature-based metaheuristic with adaptive reheating
- **Tabu Search**: Memory-based optimization with aspiration criteria

### Objective Functions
- **Value at Risk (VaR)**: Measures potential losses at 5% confidence level
- **Sharpe Ratio**: Risk-adjusted return metric
- **Maximum Drawdown (MDD)**: Peak-to-trough decline measurement

### Data Processing
- Weekly log return aggregation from daily stock prices
- Monte Carlo simulation for portfolio performance evaluation
- Efficient vectorized operations using NumPy

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/quant_finance.git
cd quant_finance

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

The main script provides a comprehensive portfolio optimization workflow:

```python
# Load and process stock data
data = load_data("SP500_data.csv")
weekly_data = aggregate_data(data)
mean, std = calculate_mean_std(weekly_data)

# Generate initial portfolio allocation
initial_solution = initial_solution(num_companies, amount=100)

# Optimize using different algorithms and objectives
# Example with Simulated Annealing and VaR:
solution, eval_value = simulated_annealing_algorithm(
    objective_function_VaR,
    num_companies,
    mean,
    std,
    initial_solution,
    temperature=110,
    alpha=0.475,
    num_iter=200
)
```

## Implementation Details

### Optimization Parameters
- Initial Temperature: 110
- Cooling Rate (α): 0.475
- Iterations: 200
- Tabu Tenure: 10
- Investment Amount: 100 MEuros

### Performance Features
- Efficient memory management with pre-allocated arrays
- Vectorized operations for Monte Carlo simulations
- Hash-based tabu list for O(1) lookup
- Adaptive reheating mechanism in Simulated Annealing

## Dependencies
- NumPy: Core numerical operations and array manipulations
- Time: Performance measurement
- OS: File operations

## Project Structure
```
.
├── templateFINAL.py     # Main implementation file
└── README.md           # Documentation
```

## Contributing
Contributions are welcome! Key areas for improvement include:
- Additional optimization algorithms
- More objective functions
- Enhanced data preprocessing capabilities
- Performance optimizations

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer
This software is for educational and research purposes only. It should not be considered as financial advice. The implemented optimization strategies do not guarantee future performance.

---
**Note**: The implementation focuses on computational efficiency and modern portfolio optimization techniques. The code is thoroughly documented and follows best practices for numerical computations in Python.

## Contact
For questions and feedback, please open an issue in the repository. 