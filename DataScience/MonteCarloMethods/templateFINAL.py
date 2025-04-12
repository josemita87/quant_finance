import numpy as np
import os
import time

def load_data(filepath="SP500_data.csv")->np.ndarray:
    """
    Load the S&P 500 stock market data from a CSV file, excluding the first row (company names)
    and the first column (dates). The data is stored in a NumPy array.

    Args:
        filepath (str): Path to the CSV file containing stock market data.

    Returns:
        numpy.ndarray: Array containing stock prices with shape (num_days, num_companies).
    """
    try:
        return np.genfromtxt(fname=filepath, delimiter=',', skip_header=1)[:, 1:]
        
    except FileNotFoundError:
        print(f"Error: '{filepath}' not found in {os.getcwd()}. Please place it there or specify the full path.")
        raise

def aggregate_data(data:np.ndarray)->np.ndarray:
    """
    Aggregate the stock prices by computing the log ratio for each company over weekly blocks
    (approximately 7 calendar days, or 5 trading days since the data includes only trading days).

    Args:
        data (numpy.ndarray): Stock price data with shape (num_days, num_companies).

    Returns:
        numpy.ndarray: Aggregated log ratios with shape (num_weeks, num_companies).
    """
    # Calculate the number of complete weeks
    num_weeks = (data.shape[0] - 5) // 5
    
    # Indices of the start and end of each week (broadcasting)
    index = 5 * np.arange(1, num_weeks + 1)
    
    # Efficiently compute log ratios over weekly blocks 
    return np.log(data[index] / data[index - 5])

def calculate_mean_std(data:np.ndarray)->tuple[float, float]:
    """
    Compute the mean and standard deviation of the aggregated log ratios for each company.

    Args:
        data (numpy.ndarray): Aggregated data with shape (num_weeks, num_companies).

    Returns:
        Tuple: Mean and standard deviation vectors of shape (num_companies,).
    """
    return np.mean(data, axis=0), np.std(data, axis=0)

def initial_solution(size: int, amount: float = 100) -> np.ndarray:
    """
    Generate an initial asset allocation using a uniform distribution of capital.

    Args:
        size (int): Number of companies.
        amount (float): Total capital to invest.

    Returns:
        numpy.ndarray: Allocation vector of length 'size' summing to 'amount'.
    """
    if size <= 0:
        raise ValueError("Size must be a positive integer.")
    
    # Generate random weights from a uniform distribution
    raw_weights = np.random.uniform(low=0.0, high=1.0, size=size)
    
    # Normalize to ensure total equals 'amount'
    allocation = (raw_weights / raw_weights.sum()) * amount

    return allocation

def compute_portfolio_returns(mean, std, solution, num_simulations:int=100)->np.ndarray:
    """
    Generate random portfolio returns for a given asset allocation using Monte Carlo simulation.
    
    This helper function is used by all objective functions to simulate portfolio returns
    based on the mean and standard deviation of each asset's log returns. It uses vectorized
    operations for efficiency and focuses only on assets with non-zero allocations.
    
    Args:
        mean (np.ndarray): Mean vector of log returns for each company.
        std (np.ndarray): Standard deviation vector of log returns for each company.
        solution (np.ndarray): Asset allocation vector.
        num_simulations (int): Number of Monte Carlo simulations to run.
        
    Returns:
        np.ndarray: Array of portfolio returns for each simulation.
    """
    # Extract only assets with non-zero allocations to reduce computation
    non_zero_idx = solution > 0
    active_mean = mean[non_zero_idx]
    active_std = std[non_zero_idx]
    active_solution = solution[non_zero_idx]

    # Skip computation if no active investments
    if len(active_mean) == 0:
        return np.zeros(num_simulations)

    # Generate random returns using a multivariate normal distribution (simplified)
    # For each company: return = exp(mean + std * Z) - 1, where Z ~ N(0, 1)
    Z = np.random.randn(num_simulations, len(active_mean))  # Standard normal random variables
    log_ratios = active_mean + Z * active_std.reshape(1, -1)  # Log returns
    returns = np.exp(log_ratios) - 1  # Convert log returns to simple returns

    # Calculate portfolio returns as weighted sum of individual asset returns
    return np.sum(active_solution * returns, axis=1)

def objective_function_VaR(
    mean:np.ndarray, 
    std:np.ndarray, 
    solution:np.ndarray, 
    num_simulations:int=100
    )->float:
    """
    Calculate Value at Risk (VaR) at the 5% confidence level using Monte Carlo simulation.
    
    VaR represents the potential loss in portfolio value over a specified time period
    at a given confidence level. Here we use the 5th percentile of simulated returns,
    which represents the potential loss that is expected to be exceeded only 5% of the time.
    
    A higher (less negative) VaR value is better, as it indicates lower potential losses.
    In the optimization context, we maximize this value to reduce downside risk.

    Args:
        mean (np.ndarray): Mean vector of log ratios for each company.
        std (np.ndarray): Standard deviation vector of log ratios for each company.
        solution (np.ndarray): Current asset allocation vector.
        num_simulations (int): Number of Monte Carlo simulations (default: 100).

    Returns:
        float: 5% VaR value representing the 5th percentile of possible returns.
    """
    # Compute portfolio returns using Monte Carlo simulation
    portfolio_returns = compute_portfolio_returns(mean, std, solution, num_simulations)
    
    # The 5th percentile corresponds to the worst 5% of cases
    worst_return = np.percentile(portfolio_returns, 5)
    
    return worst_return

def objective_function_mdd(
        mean: np.ndarray,
        std: np.ndarray,
        solution: np.ndarray,
        num_simulations: int = 100
    ) -> float:
    """
    Calculate Maximum Drawdown (MDD) for the portfolio using Monte Carlo simulation.
    
    MDD measures the largest peak-to-trough decline in portfolio value during a specific 
    period. It's an important measure of downside risk that captures the worst possible
    loss an investor could have experienced.
    
    The implementation uses a simplified approach where MDD is approximated by:
    (min_returns - max_returns) / max_returns
    
    A higher value (less negative) is better, as it indicates smaller maximum drawdowns.

    Args:
        mean (np.ndarray): Mean vector of log returns for each company.
        std (np.ndarray): Standard deviation vector of log returns for each company.
        solution (np.ndarray): Asset allocation vector.
        num_simulations (int): Number of Monte Carlo simulations (default: 100).

    Returns:
        float: Estimated MDD value; higher values indicate lower risk.
    """
    # Generate random portfolio returns
    portfolio_returns = compute_portfolio_returns(mean, std, solution, num_simulations)
    
    # Find minimum and maximum returns across all simulations
    min_returns = np.min(portfolio_returns)
    max_returns = np.max(portfolio_returns)

    # Calculate the simplified drawdown approximation
    # A smaller value (negative) indicates larger drawdowns (worse)
    if max_returns <= 0:
        return -1.0  # Avoid division by zero or negative values
        
    drawdown = (min_returns - max_returns) / abs(max_returns)
    # Return the drawdown value
    return np.mean(drawdown)

def objective_function_sharpe(
        mean:np.ndarray, 
        std:np.ndarray, 
        solution:np.ndarray, 
        num_simulations:int=100,
    )->float:
    """
    Calculate Sharpe Ratio for a given portfolio allocation using Monte Carlo simulation.
    
    The Sharpe Ratio measures the excess return (return above risk-free rate) per unit of risk.
    It is a widely used metric for portfolio performance that allows comparison of different
    investment strategies while accounting for risk.
    
    Sharpe Ratio = (Portfolio Return - Risk-Free Rate) / Portfolio Standard Deviation
    
    In this implementation, we assume a zero risk-free rate for simplicity.
    A higher Sharpe Ratio indicates better risk-adjusted returns.

    Args:
        mean (np.ndarray): Mean vector of log returns for each company.
        std (np.ndarray): Standard deviation vector of log returns for each company.
        solution (np.ndarray): Asset allocation vector.
        num_simulations (int): Number of Monte Carlo simulations (default: 100).
        
    Returns:
        float: Sharpe Ratio of the portfolio; higher values are better.
    """
    # Generate portfolio returns through Monte Carlo simulation
    portfolio_returns = compute_portfolio_returns(mean, std, solution, num_simulations)

    # Calculate portfolio standard deviation (risk)
    portfolio_std = np.std(portfolio_returns, ddof=1)
    
    # Calculate Sharpe Ratio (assuming zero risk-free rate)
    sharpe_ratio = np.mean(portfolio_returns) / portfolio_std

    return sharpe_ratio


def simulated_annealing_algorithm(
        objective_function:callable, 
        num_companies:int, 
        mean:float, 
        std:float, 
        solution:np.ndarray, 
        temperature:float, 
        alpha:float, 
        num_iter:int
    )->tuple[np.ndarray, float]:
    """
    Optimize portfolio asset allocation using the Simulated Annealing metaheuristic.
    
    This implementation follows the classic SA approach with several efficiency improvements:
    - Memory-efficient updates using pre-allocated arrays and in-place operations
    - Caching of positive allocation indices to speed up neighbor generation
    - Reheating mechanism when stuck in local optima (adaptive temperature)
    - Early termination tracking for faster convergence
    
    The algorithm explores the solution space by making random moves, accepting improvements
    immediately and occasionally accepting worse solutions based on the current temperature.
    As temperature decreases, the algorithm becomes more selective, focusing on exploitation.
    
    Args:
        objective_function (callable): Function to maximize (e.g., VaR, Sharpe, MDD).
        num_companies (int): Number of companies in the portfolio.
        mean (numpy.ndarray): Mean vector of log ratios for each company.
        std (numpy.ndarray): Standard deviation vector of log ratios for each company.
        solution (numpy.ndarray): Initial solution vector (asset allocation).
        temperature (float): Initial temperature controlling acceptance probability.
        alpha (float): Cooling parameter (0 < alpha < 1) that determines cooling rate.
        num_iter (int): Markov chain length (iterations per temperature level).

    Returns:
        tuple: (best_solution, best_eval) - Best asset allocation found and its objective value.
    """    
    # Initialize variables (working solution and best-so-far)
    sol = solution.copy()
    eval = objective_function(mean, std, sol)
    best_sol = sol.copy()
    best_eval = eval

    # Cache indices of companies with positive allocations for efficient neighbor generation
    positive_indices = np.where(sol > 0)[0]
    if len(positive_indices) == 0:
        # Can't proceed with no positive allocations
        return sol, eval

    # Simulated annealing control parameters
    min_temperature = 1
    initial_temperature = temperature  # Store initial temperature for reheating
    max_iter_without_improvement = num_iter * 2  # Threshold for reheating
    reheating_factor = 1.5  # Factor to increase temperature when stuck
    iter_since_improvement = 0  # Counter for iterations without improvement
    max_reheats = 3  # Maximum number of times we can reheat
    reheat_count = 0  # Counter for number of reheats performed                     

    # Pre-allocate neighbor buffer to avoid repeated memory allocations
    neighbor = np.zeros_like(sol)

    # Main simulated annealing loop - continues until temperature falls below threshold
    while temperature > min_temperature and reheat_count < max_reheats:
        # Markov chain loop - fixed number of iterations at each temperature
        for _ in range(num_iter):
            # Generate neighbor by transferring funds between companies
            np.copyto(neighbor, sol)  # Reuse buffer (more efficient than creating new arrays)
            
            # Select a source with funds and a random destination
            source_idx = np.random.choice(positive_indices)
            dest_idx = np.random.randint(0, num_companies)
            
            # Determine transfer amount (random percentage of source allocation)
            amount = sol[source_idx] * np.random.random()
            
            # Update neighbor allocation
            neighbor[source_idx] -= amount
            neighbor[dest_idx] += amount

            # Evaluate neighbor solution
            neighbor_eval = objective_function(mean, std, neighbor)

            # Calculate improvement (positive delta means better solution)
            delta = neighbor_eval - eval
            
            # Accept new solution if better or with probability based on temperature
            # (Metropolis criterion allows occasional uphill moves to escape local optima)
            if delta > 0 or np.random.random() < np.exp(delta / temperature):
                # Update current solution
                np.copyto(sol, neighbor)
                eval = neighbor_eval

                # Update cached positive indices after allocation changed
                positive_indices = np.where(sol > 0)[0]

                # Update best solution if we found an improvement
                if eval > best_eval:
                    np.copyto(best_sol, sol)
                    best_eval = eval
                    iter_since_improvement = 0  # Reset counter on improvement
                else:
                    iter_since_improvement += 1  # Increment counter when no improvement
            else:
                iter_since_improvement += 1  # Increment counter when solution rejected

            # Check if we're stuck and should reheat
            if iter_since_improvement >= max_iter_without_improvement:
                # Reheat by increasing temperature
                temperature = initial_temperature * reheating_factor
                iter_since_improvement = 0  # Reset counter
                reheat_count += 1  # Increment reheat counter
                break  # Exit inner loop to start with new temperature

        # Standard cooling schedule if we haven't reheated
        if iter_since_improvement < max_iter_without_improvement:
            temperature *= alpha

    return best_sol, best_eval


def tabu_search_algorithm(
    objective_function: callable,
    num_companies: int,
    mean: np.ndarray,
    std: np.ndarray,
    solution: np.ndarray,
    tabu_tenure: int,
    num_iter: int
) -> tuple[np.ndarray, float]:
    """
    Tabu Search algorithm for portfolio optimization with efficient implementation.
    
    This algorithm uses a hash-based tabu list (via tobytes()) for fast lookup of previously 
    explored solutions to avoid cycling. It implements the aspiration criterion, which allows 
    accepting a tabu move if it leads to a better solution than the current best.
    
    The implementation uses pre-allocated arrays and in-place operations to minimize memory
    allocations during the search process.
    
    Args:
        objective_function (callable): Function to maximize (e.g., VaR, Sharpe, or MDD).
        num_companies (int): Number of companies in the portfolio.
        mean (np.ndarray): Mean vector of log ratios for each company.
        std (np.ndarray): Standard deviation vector of log ratios for each company.
        solution (np.ndarray): Initial solution vector (asset allocation).
        tabu_tenure (int): Maximum number of solutions to keep in the tabu list.
        num_iter (int): Number of iterations for the main search loop.

    Returns:
        tuple: (best_solution, best_eval) - Best asset allocation found and its objective value.
    """

    # Initialize working solution and keep track of best found
    sol = solution.copy()
    best_sol = sol.copy()
    best_eval = objective_function(mean, std, sol)

    # Use a set for O(1) lookup of tabu solutions and a queue to maintain tabu tenure
    tabu_set = set()  # For fast lookup (hash-based)
    tabu_queue = []   # For FIFO behavior when removing old tabu entries

    # Pre-allocate arrays to avoid memory allocations in the inner loop
    neighbor = np.empty_like(sol)                      # Buffer for current neighbor
    neighbors_array = np.empty((num_companies, sol.size))  # Store valid neighbors
    evals_array = np.full(num_companies, -np.inf)      # Store neighbor evaluations

    # Main search loop
    for _ in range(num_iter):
        # Get indices of companies with positive allocations (potential sources)
        positive_indices = np.where(sol > 0)[0]
        if len(positive_indices) == 0:
            break  # No valid moves possible when all allocations are zero

        valid_count = 0  # Counter for valid neighbors found in this iteration

        # Generate and evaluate neighbors
        for _ in range(num_companies):
            # Generate a neighbor by transferring funds between companies
            # Randomly select a source company from the positive allocations
            i = np.random.choice(positive_indices)
            # Randomly select a destination company
            j = np.random.randint(0, num_companies)

            # Ensure the destination company is different from the source company
            while j == i:
                j = np.random.randint(0, num_companies)

            # Copy the current solution to the neighbor buffer (avoids allocation)
            np.copyto(neighbor, sol)
            
            # Transfer a random amount from source to destination
            amount = np.random.uniform(0, sol[i])
            neighbor[i] -= amount
            neighbor[j] += amount

            # Use binary representation as a hash key for the tabu list
            neighbor_key = neighbor.tobytes()
            
            # Evaluate the neighbor
            neigh_eval = objective_function(mean, std, neighbor)

            # Accept if not tabu or if it passes aspiration criterion (improves best_eval)
            if (neighbor_key not in tabu_set) or (neigh_eval > best_eval):
                # Store valid neighbor and its evaluation
                neighbors_array[valid_count] = neighbor
                evals_array[valid_count] = neigh_eval
                valid_count += 1

        # Select the best non-tabu neighbor (or one that passes aspiration)
        idx_best = np.argmax(evals_array[:valid_count])
        best_neighbor = neighbors_array[idx_best]
        best_eval_candidate = evals_array[idx_best]
        best_key = best_neighbor.tobytes()

        # Update the current solution
        np.copyto(sol, best_neighbor)
        
        # Update best solution if the new solution is better
        if best_eval_candidate > best_eval:
            best_eval = best_eval_candidate
            np.copyto(best_sol, best_neighbor)

        # Add the selected move to the tabu list
        tabu_set.add(best_key)
        tabu_queue.append(best_key)

        # Maintain tabu list size by removing oldest entries when needed
        if len(tabu_queue) > tabu_tenure:
            old_key = tabu_queue.pop(0)  # Remove oldest entry (FIFO)
            tabu_set.discard(old_key)    # Remove from set as well

    
    return best_sol, best_eval

# Main
def main():
    """
    Main program execution flow for portfolio optimization.
    
    This function orchestrates the entire portfolio optimization process:
    1. Data loading and preprocessing: Loads historical stock data and calculates weekly returns
    2. Initial solution generation: Creates a random starting portfolio allocation
    3. Baseline assessment: Calculates initial VaR, Sharpe ratio, and MDD metrics
    4. Optimization: Runs multiple optimization algorithms with different objective functions
    5. Results reporting: Prints results for each optimization combination
    
    The implementation uses a structured approach with dictionaries to organize optimization
    configurations, allowing for easy extension with new algorithms or objective functions.
    """
    # ======================================================================
    # INITIALIZATION
    # ======================================================================
    import time
    
    # Hyperparameters for optimization algorithms
    amount = 100       # Investment amount in MEuros
    temperature = 110  # Initial temperature for simulated annealing
    alpha = 0.475      # Cooling rate (temperature reduction factor)
    num_iter = 200      # Number of iterations per temperature level / tabu iterations
    tabu_tenure = 10  # Tabu list size (number of recent solutions to avoid)
   
    
    print("\n" + "=" * 60)
    print("Portfolio Optimization using Simulated Annealing and Tabu Search")
    print("=" * 60)
    
    # ======================================================================
    # DATA LOADING AND PREPROCESSING
    # ======================================================================
    print("Loading and processing data...")
    # Load raw stock price data
    data = load_data()
    print(f"Data loaded: {data.shape[0]} days × {data.shape[1]} companies")

    # Transform daily prices to weekly log returns
    data_week = aggregate_data(data)
    print(f"Weekly data: {data_week.shape[0]} weeks × {data_week.shape[1]} companies")

    # Calculate statistics (mean and standard deviation) for each company
    mean, std = calculate_mean_std(data_week)

    # Get the number of companies in our dataset
    num_companies = mean.shape[0]
    print(f"Number of companies in portfolio: {num_companies}")

    # ======================================================================
    # INITIAL SOLUTION GENERATION
    # ======================================================================
    print("\nGenerating initial portfolio allocation...")
    # Create a starting solution (all capital in one random company)
    isolution = initial_solution(num_companies, amount)

    # ======================================================================
    # BASELINE ASSESSMENT
    # ======================================================================
    print("\nEvaluating initial portfolio allocation over multiple runs...")
    # Calculate baseline metrics for the initial solution with multiple runs
    num_initial_runs = 10
    num_simulations = 200  # Number of Monte Carlo simulations for all evaluations
    initial_VaR_values = []
    initial_sharpe_values = []
    initial_mdd_values = []
    
    for i in range(num_initial_runs):
        initial_VaR_values.append(objective_function_VaR(mean, std, isolution, num_simulations))
        initial_sharpe_values.append(objective_function_sharpe(mean, std, isolution, num_simulations))
        initial_mdd_values.append(objective_function_mdd(mean, std, isolution, num_simulations))
    
    # Calculate averages and standard deviations
    initial_VaR = np.mean(initial_VaR_values)
    initial_VaR_std = np.std(initial_VaR_values)
    initial_sharpe = np.mean(initial_sharpe_values)
    initial_sharpe_std = np.std(initial_sharpe_values)
    initial_mdd = np.mean(initial_mdd_values)
    initial_mdd_std = np.std(initial_mdd_values)
    
    print(f"Initial VaR (5% confidence): M€{initial_VaR:.2f}")
    print(f"Initial Sharpe Ratio: {initial_sharpe:.2f}")
    print(f"Initial MDD: {initial_mdd:.2f}")
    
    # ======================================================================
    # OPTIMIZATION CONFIGURATION
    # ======================================================================
    # Define objective functions with their associated data
    evaluation_functions = {
        "VaR": {
            "func": objective_function_VaR,
            "initial_value": initial_VaR,
            "initial_std": initial_VaR_std,
            "initial_label": "Initial VaR (5%) in MEuros",
            "optimized_label": "Optimized VaR (5%) in MEuros"
        },
        "Sharpe": {
            "func": objective_function_sharpe,
            "initial_value": initial_sharpe,
            "initial_std": initial_sharpe_std,
            "initial_label": "Initial Sharpe Ratio ",
            "optimized_label": "Optimized Sharpe Ratio "
        },
        "MDD": {
            "func": objective_function_mdd,
            "initial_value": initial_mdd,
            "initial_std": initial_mdd_std,
            "initial_label": "Initial MDD",
            "optimized_label": "Optimized MDD"
        }
    }

    # Define optimization algorithms with their parameter configurations
    algorithms = [
        {
            "name": "Simulated Annealing",
            "function": simulated_annealing_algorithm,
            "params": lambda obj_func: [obj_func, num_companies, mean, std, isolution, temperature, alpha, num_iter],
            "param_desc": f"T₀={temperature}, α={alpha}, iterations per temperature={num_iter}"
        },
        {
            "name": "Tabu Search",
            "function": tabu_search_algorithm,
            "params": lambda obj_func: [obj_func, num_companies, mean, std, isolution, tabu_tenure, num_iter],
            "param_desc": f"tabu tenure={tabu_tenure}, iterations={num_iter}"
        }
    ]

    # ======================================================================
    # OPTIMIZATION EXECUTION
    # ======================================================================
    # Run all combinations of algorithms and objective functions with multiple repetitions for averaging
    num_repetitions = 4  # Number of runs to average results
    
    for algo in algorithms:
        for eval_name, eval_data in evaluation_functions.items():
            print("\n" + "=" * 60)
            print(f"\nRunning {algo['name']} with {eval_name} optimization...")
            print(f"Parameters: {algo['param_desc']} x {num_repetitions} repetitions")
            print(f"MC simulations: {num_simulations} for all evaluations")
            
            # Create a wrapper function that uses fewer simulations during optimization
            def optimized_obj_func(mean, std, solution):
                return eval_data["func"](mean, std, solution, num_simulations)
            
            # Arrays to store results from all repetitions
            all_times = []
            all_values = []
            all_solutions = []
            
            for rep in range(num_repetitions):
                # Measure optimization time
                start_time = time.time()
                
                # Modify algorithm parameters to use the optimized objective function
                if algo["name"] == "Simulated Annealing":
                    params = [optimized_obj_func, num_companies, mean, std, isolution, temperature, alpha, num_iter]
                else:  # Tabu Search
                    params = [optimized_obj_func, num_companies, mean, std, isolution, tabu_tenure, num_iter]
                
                # Execute optimization
                sol, _ = algo["function"](*params)
                
                # Re-evaluate with more accurate simulation for final result
                optimized_value = eval_data["func"](mean, std, sol, num_simulations)
                
                # Calculate elapsed time
                elapsed_time = time.time() - start_time
                
                # Store results
                all_times.append(elapsed_time)
                all_values.append(optimized_value)
                all_solutions.append(sol)
                
            
            # Calculate statistics
            avg_time = np.mean(all_times)
            avg_value = np.mean(all_values)
           
            # ======================================================================
            # RESULTS REPORTING
            # ======================================================================
            print(f"\n{algo['name']} Results (Averaged over {num_repetitions} runs):")
            print("=" * 60)
            print(f"Average optimization time: {avg_time:.2f} seconds")
            print(f"{eval_data['initial_label']}: {eval_data['initial_value']:.4f}")
            print(f"{eval_data['optimized_label']} (avg): {avg_value:.4f}")

# Script entry point
if __name__ == "__main__":
    main()
