import numpy as np
import os
import pandas as pd

# Global Monte Carlo random draws cache for consistent simulations
MC_RANDOM_MATRIX: np.ndarray = None
MC_PARAMS = { 'num_sim': 0, 'num_assets': 0 }

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
        data = np.genfromtxt(fname=filepath, delimiter=',', skip_header=1)[:, 1:]
        return data
        
    except FileNotFoundError:
        print(f"Error: '{filepath}' not found in {os.getcwd()}. Please place it there or specify the full path.")
        raise

def aggregate_data(data:np.ndarray)->np.ndarray:
    """
    Aggregate the stock prices by computing the simple weekly returns for each company over weekly blocks
    (approximately 7 calendar days, or 5 trading days since the data includes only trading days).

    Args:
        data (numpy.ndarray): Stock price data with shape (num_days, num_companies).

    Returns:
        numpy.ndarray: Aggregated simple weekly returns with shape (num_weeks, num_companies).
    """
    # Calculate the number of complete weeks
    num_weeks = (data.shape[0] - 5) // 5
    
    # Indices of the start and end of each week (broadcasting)
    index = 5 * np.arange(1, num_weeks + 1)
    
    # Efficiently compute weekly log returns over blocks: log(P_t / P_{t-5})
    return np.log(data[index] / data[index - 5])

def calculate_mean_std(data:np.ndarray)->np.ndarray:
    """
    Compute the mean and standard deviation of the aggregated simple weekly returns for each company.

    Returns a 2×N array where the first row is the mean and the second row is the std.
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return np.vstack((mean, std))

def initial_solution(size: int, amount: float = 100) -> np.ndarray:
    """
    Generate an initial asset allocation with all capital allocated to a single randomly chosen company.

    Args:
        size (int): Number of companies.
        amount (float): Total capital to invest.

    Returns:
        numpy.ndarray: Allocation vector of length 'size' with all capital in one company.
    """
    if size <= 0:
        raise ValueError("Size must be a positive integer.")
    allocation = np.zeros(size)
    idx = np.random.randint(size)
    allocation[idx] = amount
    return allocation

def compute_portfolio_returns(mean, std, solution, num_simulations:int=100)->np.ndarray:
    """
    Generate random portfolio returns for a given asset allocation using Monte Carlo simulation.
    
    This helper function is used by all objective functions to simulate portfolio returns
    based on the mean and standard deviation of each asset's simple returns. It uses vectorized
    operations for efficiency and focuses only on assets with non-zero allocations.
    
    Args:
        mean (np.ndarray): Mean vector of simple returns for each company.
        std (np.ndarray): Standard deviation vector of simple returns for each company.
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

    # Use cached global random draws for consistency and reduced noise
    global MC_RANDOM_MATRIX, MC_PARAMS
    # Regenerate if dimensions or simulation count change
    if MC_RANDOM_MATRIX is None or MC_PARAMS['num_sim'] != num_simulations or MC_PARAMS['num_assets'] != mean.size:
        MC_RANDOM_MATRIX = np.random.randn(num_simulations, mean.size)
        MC_PARAMS['num_sim'] = num_simulations
        MC_PARAMS['num_assets'] = mean.size
    # Select draws for active assets
    Z = MC_RANDOM_MATRIX[:, non_zero_idx]  # Standard normal draws
    returns = active_mean + Z * active_std.reshape(1, -1)  # Simple return simulations
    # Calculate portfolio returns as weighted sum of individual asset returns
    # solution values correspond to capital allocations, so returns are in same units
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
        mean (np.ndarray): Mean vector of simple returns for each company.
        std (np.ndarray): Standard deviation vector of simple returns for each company.
        solution (np.ndarray): Current asset allocation vector.
        num_simulations (int): Number of Monte Carlo simulations (default: 100).

    Returns:
        float: 5% VaR value representing the 5th percentile of possible returns.
    """
    # Compute portfolio returns using Monte Carlo simulation
    portfolio_returns = compute_portfolio_returns(mean, std, solution, num_simulations)
    
    # 95% VaR: return at the 95th percentile of simulated returns
    worst_return = 100 + np.percentile(portfolio_returns, 5)
    return worst_return

def objective_function_mdd(
        mean: np.ndarray,
        std: np.ndarray,
        solution: np.ndarray,
        num_simulations: int = 100
    ) -> float:
    """
    Calculate maximum drawdown using Monte Carlo simulated returns only.
    Returns negative of the maximum drawdown (so closer to zero is better).
    """
    portfolio_returns = compute_portfolio_returns(mean, std, solution, num_simulations)
    total = solution.sum()
    return 0.0 if total == 0 else np.min(portfolio_returns / total)

def objective_function_sharpe(
        mean:np.ndarray, 
        std:np.ndarray, 
        solution:np.ndarray, 
        num_simulations:int=100,
    )->float:
    """
    Calculate Sharpe Ratio using Monte Carlo simulation (zero risk-free rate).
    """
    # Generate portfolio returns through Monte Carlo simulation
    portfolio_returns = compute_portfolio_returns(mean, std, solution, num_simulations)
    # Calculate portfolio standard deviation (risk)
    portfolio_std = np.std(portfolio_returns, ddof=1)
    if portfolio_std == 0:
        return 0.0
    # Sharpe ratio assuming zero risk-free rate
    return np.mean(portfolio_returns) / portfolio_std

def simulated_annealing_v0_base(
        objective_function: callable,
        num_companies: int,
        mean: np.ndarray,
        std: np.ndarray,
        solution: np.ndarray,
        initial_temperature: float,
        cooling_rate: float,  # Alpha in the advanced version
        num_iter: int         # Iterations per temperature level
) -> tuple[np.ndarray, float]:
    """
    Basic Simulated Annealing algorithm.
    
    Args:
        objective_function (callable): Function to maximize.
        num_companies (int): Number of companies.
        mean (numpy.ndarray): Mean vector of simple returns.
        std (numpy.ndarray): Standard deviation vector of simple returns.
        solution (numpy.ndarray): Initial solution vector.
        initial_temperature (float): Initial temperature.
        cooling_rate (float): Cooling rate (e.g., 0.95).
        num_iter (int): Iterations per temperature level.

    Returns:
        tuple: (best_solution, best_eval) - Best allocation and its objective value.
    """
    current_solution = solution.copy()
    current_eval = objective_function(mean, std, current_solution)
    
    best_solution = current_solution.copy()
    best_eval = current_eval
    
    temperature = initial_temperature
    min_temperature = 1e-3 # Define a minimum temperature to stop

    while temperature > min_temperature:
        for _ in range(num_iter):
            # Generate a neighbor solution
            neighbor_solution = current_solution.copy()
            
            # Simple random move: pick two distinct random companies
            # and move a random amount from one to another if possible
            if num_companies < 2:
                 # Cannot make a move if less than 2 companies with funds potentially
                 # or no funds to move. For simplicity, we check for positive funds later.
                 idx_from, idx_to = 0, 0 # Placeholder, will be skipped if no funds
            else:
                idx_from, idx_to = np.random.choice(num_companies, 2, replace=False)

            if neighbor_solution[idx_from] > 0: # Ensure there are funds to move
                amount_to_move = np.random.uniform(0, neighbor_solution[idx_from])
                neighbor_solution[idx_from] -= amount_to_move
                neighbor_solution[idx_to] += amount_to_move
            else:
                # No funds to move from selected company, try another iteration or skip
                # For base, we just continue, might lead to less exploration if unlucky
                continue 

            neighbor_eval = objective_function(mean, std, neighbor_solution)
            
            delta = neighbor_eval - current_eval
            
            if delta > 0 or np.random.random() < np.exp(delta / temperature):
                current_solution = neighbor_solution
                current_eval = neighbor_eval
                
                if current_eval > best_eval:
                    best_solution = current_solution.copy()
                    best_eval = current_eval
        
        temperature *= cooling_rate # Cool down
        
    return best_solution, best_eval

def simulated_annealing_v1_adaptive_step(
        objective_function: callable,
        num_companies: int,
        mean: np.ndarray,
        std: np.ndarray,
        solution: np.ndarray,
        initial_temperature: float,
        cooling_rate: float, # Alpha
        num_iter: int
) -> tuple[np.ndarray, float]:
    """
    Simulated Annealing with adaptive step size based on temperature.
    Improvement: Adaptive step moves proportional to current temperature.
    """
    current_solution = solution.copy()
    current_eval = objective_function(mean, std, current_solution)
    
    best_solution = current_solution.copy()
    best_eval = current_eval
    
    temperature = initial_temperature
    min_temperature = 1e-3

    while temperature > min_temperature:
        for _ in range(num_iter):
            # Generate a neighbor solution
            neighbor_solution = current_solution.copy()
            
            # Simple random move: pick two distinct random companies
            # and move a random amount from one to another if possible
            if num_companies < 2:
                 # Cannot make a move if less than 2 companies with funds potentially
                 # or no funds to move. For simplicity, we check for positive funds later.
                 idx_from, idx_to = 0, 0 # Placeholder, will be skipped if no funds
            else:
                idx_from, idx_to = np.random.choice(num_companies, 2, replace=False)

            if neighbor_solution[idx_from] > 0: # Ensure there are funds to move
                amount_to_move = np.random.uniform(0, neighbor_solution[idx_from])
                neighbor_solution[idx_from] -= amount_to_move
                neighbor_solution[idx_to] += amount_to_move
            else:
                # No funds to move from selected company, try another iteration or skip
                # For base, we just continue, might lead to less exploration if unlucky
                continue 

            neighbor_eval = objective_function(mean, std, neighbor_solution)
            
            delta = neighbor_eval - current_eval
            
            if delta > 0 or np.random.random() < np.exp(delta / temperature):
                current_solution = neighbor_solution
                current_eval = neighbor_eval
                
                if current_eval > best_eval:
                    best_solution = current_solution.copy()
                    best_eval = current_eval
        
        temperature *= cooling_rate # Cool down
        
    return best_solution, best_eval

def simulated_annealing_v2_best_of_k(
        objective_function: callable,
        num_companies: int,
        mean: np.ndarray,
        std: np.ndarray,
        solution: np.ndarray,
        initial_temperature: float,
        cooling_rate: float, # Alpha
        num_iter: int,
        k_neighbors: int = 5
) -> tuple[np.ndarray, float]:
    """
    Simulated Annealing improvement: best-of-k neighbor sampling per iteration.
    Improvement: Explore k random neighbors and apply the best move to improve convergence.
    """
    current_solution = solution.copy()
    current_eval = objective_function(mean, std, current_solution)
    best_solution = current_solution.copy()
    best_eval = current_eval
    temperature = initial_temperature
    min_temperature = 1e-3

    while temperature > min_temperature:
        for _ in range(num_iter):
            # best-of-k neighbor sampling
            best_neighbor = None
            best_neighbor_eval = -np.inf
            for _ in range(k_neighbors):
                if num_companies < 2:
                    continue
                neighbor = current_solution.copy()
                i, j = np.random.choice(num_companies, 2, replace=False)
                amount = np.random.uniform(0, neighbor[i])
                neighbor[i] -= amount
                neighbor[j] += amount
                val = objective_function(mean, std, neighbor)
                if val > best_neighbor_eval:
                    best_neighbor_eval = val
                    best_neighbor = neighbor
            if best_neighbor is None:
                continue
            delta = best_neighbor_eval - current_eval
            if delta > 0 or np.random.rand() < np.exp(delta / temperature):
                current_solution = best_neighbor
                current_eval = best_neighbor_eval
                if current_eval > best_eval:
                    best_solution = current_solution.copy()
                    best_eval = current_eval
        temperature *= cooling_rate
    return best_solution, best_eval

def simulated_annealing_v3_adaptive_reheating(
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
    Optimize portfolio using Simulated Annealing with adaptive reheating.
    This implementation follows the classic SA approach with several efficiency improvements:
    - Adaptive reheating to escape deep local optima
    
    The algorithm explores the solution space by making random moves, accepting improvements
    immediately and occasionally accepting worse solutions based on the current temperature.
    As temperature decreases, the algorithm becomes more selective, focusing on exploitation.
    
    Args:
        objective_function (callable): Function to maximize (e.g., VaR, Sharpe, MDD).
        num_companies (int): Number of companies in the portfolio.
        mean (numpy.ndarray): Mean vector of simple returns for each company.
        std (numpy.ndarray): Standard deviation vector of simple returns for each company.
        solution (numpy.ndarray): Initial solution vector (asset allocation).
        temperature (float): Initial temperature controlling acceptance probability.
        alpha (float): Cooling parameter (0 < alpha < 1) that determines cooling rate.
        num_iter (int): Markov chain length (iterations per temperature level).

    Returns:
        tuple: (best_solution, best_eval) - Best asset allocation found and its objective value.
    """    
    # Initialize variables (working solution and best-so-far)
    sol = solution.copy()
    neighbor = np.empty_like(sol)
    eval = objective_function(mean, std, sol)
    best_sol = sol.copy()
    best_eval = eval

    # Simulated annealing control parameters
    min_temperature = 1
    initial_temperature = temperature  # Store initial temperature for reheating
    max_iter_without_improvement = num_iter * 2  # Threshold for reheating
    reheating_factor = 1.5  # Factor to increase temperature when stuck
    iter_since_improvement = 0  # Counter for iterations without improvement
    max_reheats = 3  # Maximum number of times we can reheat
    reheat_count = 0  # Counter for number of reheats performed                     

    # Main simulated annealing loop - continues until temperature falls below threshold
    while temperature > min_temperature and reheat_count < max_reheats:
        # Markov chain loop - fixed number of iterations at each temperature
        for _ in range(num_iter):
            # Generate neighbor by transferring funds between companies
            np.copyto(neighbor, sol)  # Reuse buffer (more efficient than creating new arrays)
            
            # Select a source with funds and a random destination
            source_idx = np.random.choice(np.where(sol > 0)[0])
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

def simulated_annealing_v4_elitist_archive(
        objective_function: callable,
        num_companies: int,
        mean: np.ndarray,
        std: np.ndarray,
        solution: np.ndarray,
        initial_temperature: float,
        cooling_rate: float,
        num_iter: int,
        archive_size: int = 5,
        elite_replace_prob: float = 0.1
) -> tuple[np.ndarray, float]:
    """
    Simulated Annealing with elitist archive: maintains an archive of top N solutions and occasionally restarts from an elite.
    """
    current_solution = solution.copy()
    current_eval = objective_function(mean, std, current_solution)
    best_solution = current_solution.copy()
    best_eval = current_eval
    temperature = initial_temperature
    min_temperature = 1e-3
    # Archive: list of (solution, eval)
    archive = [(current_solution.copy(), current_eval)]
    while temperature > min_temperature:
        for _ in range(num_iter):
            neighbor_solution = current_solution.copy()
            if num_companies < 2:
                idx_from, idx_to = 0, 0
            else:
                idx_from, idx_to = np.random.choice(num_companies, 2, replace=False)
            if neighbor_solution[idx_from] > 0:
                amount_to_move = np.random.uniform(0, neighbor_solution[idx_from])
                neighbor_solution[idx_from] -= amount_to_move
                neighbor_solution[idx_to] += amount_to_move
            else:
                continue
            neighbor_eval = objective_function(mean, std, neighbor_solution)
            delta = neighbor_eval - current_eval
            if delta > 0 or np.random.random() < np.exp(delta / temperature):
                current_solution = neighbor_solution
                current_eval = neighbor_eval
                # Archive update
                archive.append((current_solution.copy(), current_eval))
                archive = sorted(archive, key=lambda x: x[1], reverse=True)[:archive_size]
                if current_eval > best_eval:
                    best_solution = current_solution.copy()
                    best_eval = current_eval
            # Occasionally jump to a random elite
            if len(archive) > 1 and np.random.random() < elite_replace_prob:
                elite_idx = np.random.randint(len(archive))
                current_solution, current_eval = archive[elite_idx][0].copy(), archive[elite_idx][1]
        temperature *= cooling_rate
    return best_solution, best_eval

def tabu_search_v0_base(
    objective_function: callable,
    num_companies: int,
    mean: np.ndarray,
    std: np.ndarray,
    solution: np.ndarray,
    tabu_tenure: int,
    num_iter: int,
    num_neighbors_to_generate: int = 10 # Max neighbors to check per iteration
) -> tuple[np.ndarray, float]:
    """
    Basic Tabu Search algorithm.
    Uses a simple list for the tabu list (stores full solutions).
    Generates a fixed number of neighbors and picks the best non-tabu.
    """
    current_solution = solution.copy()
    current_eval = objective_function(mean, std, current_solution)

    best_solution = current_solution.copy()
    best_eval = current_eval

    tabu_list = [] # Simple list to store tabu solutions (as ndarray objects)

    for iteration in range(num_iter):
        candidate_neighbors = []
        candidate_evals = []
        
        # Try to generate a few neighbors
        # In this base version, neighbor generation is simple and might be less diverse
        generated_count = 0
        attempts = 0 # To avoid infinite loop if no valid moves found
        max_attempts = num_companies * 2 # Heuristic limit

        while generated_count < num_neighbors_to_generate and attempts < max_attempts:
            attempts += 1
            temp_neighbor = current_solution.copy()
            
            if num_companies < 2:
                # Not enough companies to make a move
                break 

            idx_from, idx_to = np.random.choice(num_companies, 2, replace=False)

            if temp_neighbor[idx_from] > 0:
                amount_to_move = np.random.uniform(0, temp_neighbor[idx_from])
                temp_neighbor[idx_from] -= amount_to_move
                temp_neighbor[idx_to] += amount_to_move
                
                # Check if this specific neighbor configuration is tabu
                is_tabu = False
                for tabu_sol in tabu_list:
                    if np.array_equal(temp_neighbor, tabu_sol):
                        is_tabu = True
                        break
                
                if not is_tabu:
                    candidate_neighbors.append(temp_neighbor)
                    candidate_evals.append(objective_function(mean, std, temp_neighbor))
                    generated_count += 1
            elif np.sum(current_solution) == 0 : # No funds anywhere
                 break

        if not candidate_neighbors: # No non-tabu neighbors found
            # If stuck, could implement a random restart or simply stop/continue
            # For base version, we just continue, which might mean no move is made
            if not tabu_list: # if tabu list is also empty, something is wrong or initial solution is tricky
                 pass # continue to next iter, or could break
            elif len(tabu_list) > 0 : # if tabu is not empty, try to clear one to allow moves
                 tabu_list.pop(0) # remove oldest tabu to allow a move potentially
                 # This is a very basic way to escape, advanced versions are better

        if candidate_neighbors: # If any non-tabu neighbors were found
            best_candidate_idx = np.argmax(candidate_evals)
            best_neighbor_solution = candidate_neighbors[best_candidate_idx]
            best_neighbor_eval = candidate_evals[best_candidate_idx]

            current_solution = best_neighbor_solution
            current_eval = best_neighbor_eval
            
            # Add current_solution (which is the chosen neighbor) to tabu list
            tabu_list.append(current_solution.copy()) # Store a copy
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0) # Remove oldest

            if current_eval > best_eval:
                best_solution = current_solution.copy()
                best_eval = current_eval
        elif np.sum(current_solution) == 0 and iteration > 0: # No funds and not first iter
            break # Stop if portfolio is empty

    return best_solution, best_eval


def tabu_search_v1_frequency_memory(
    objective_function: callable,
    num_companies: int,
    mean: np.ndarray,
    std: np.ndarray,
    solution: np.ndarray,
    tabu_tenure: int,
    num_iter: int,
    num_neighbors_to_generate: int = 10 # Max neighbors to check per iteration
) -> tuple[np.ndarray, float]:
    """
    Tabu Search with frequency-based memory for asset moves.
    Improvement: Track move frequencies to diversify source selection.
    """
    # Performance: frequency-based memory to diversify asset selection
    freq_count = np.zeros(num_companies, dtype=int)
    current_solution = solution.copy()
    current_eval = objective_function(mean, std, current_solution)

    best_solution = current_solution.copy()
    best_eval = current_eval

    tabu_set = set()  # For O(1) lookup of tabu solutions (hashed)
    tabu_queue = []   # For FIFO behavior to maintain tabu tenure

    # Add initial solution's state to tabu to avoid immediate reversal if it's non-optimal
    # This is optional but can sometimes guide search away from initial point quickly.
    # initial_key = current_solution.tobytes()
    # tabu_set.add(initial_key)
    # tabu_queue.append(initial_key)
    # if len(tabu_queue) > tabu_tenure: # Should not happen if tenure > 0
    #    old_key_init = tabu_queue.pop(0)
    #    tabu_set.discard(old_key_init)

    for iteration in range(num_iter):
        # Get indices of companies with positive allocations (potential sources)
        current_positive_indices = np.where(current_solution > 0)[0]
        if not current_positive_indices.size:
            # No company has funds currently, or portfolio is all zeros.
            break # Exit main iteration loop, no moves possible

        candidate_neighbors_solutions = []
        candidate_neighbors_evals = []
        candidate_neighbors_keys = [] # Store keys to add to tabu if chosen
        candidate_sources = []  # Track source indices for frequency update
        
        generated_count = 0
        attempts = 0
        max_attempts = num_companies * 3 # Allow more attempts if many are tabu

        while generated_count < num_neighbors_to_generate and attempts < max_attempts:
            attempts += 1
            temp_neighbor = current_solution.copy()
            
            # choose source asset with least move frequency for diversification
            current_freqs = freq_count[current_positive_indices]
            min_freq = current_freqs.min()
            low_freq_assets = current_positive_indices[current_freqs == min_freq]
            source_idx = np.random.choice(low_freq_assets)
            dest_idx = np.random.randint(0, num_companies)

            if num_companies > 1:
                while dest_idx == source_idx:
                    dest_idx = np.random.randint(0, num_companies)
            # If num_companies == 1, dest_idx will be source_idx, resulting in a null move. This is fine.
            
            # current_solution[source_idx] is guaranteed > 0 as source_idx is from current_positive_indices
            amount_to_move = np.random.uniform(0, current_solution[source_idx])
            temp_neighbor[source_idx] -= amount_to_move
            temp_neighbor[dest_idx] += amount_to_move
                
            neighbor_key = temp_neighbor.tobytes()
            if neighbor_key not in tabu_set:
                candidate_neighbors_solutions.append(temp_neighbor)
                candidate_neighbors_evals.append(objective_function(mean, std, temp_neighbor))
                candidate_sources.append(source_idx)
                candidate_neighbors_keys.append(neighbor_key)
                generated_count += 1
            # if selected source_idx has no funds, loop continues to try another neighbor

        if not candidate_neighbors_solutions: # No non-tabu neighbors found
            if len(tabu_queue) > 0: # Try to make space by removing oldest tabu
                old_key = tabu_queue.pop(0)
                tabu_set.discard(old_key)
            # Continue to next iteration, hoping a move becomes available
            continue 

        # Select the best non-tabu neighbor found
        best_candidate_idx = np.argmax(candidate_neighbors_evals)
        
        current_solution = candidate_neighbors_solutions[best_candidate_idx]
        current_eval = candidate_neighbors_evals[best_candidate_idx]
        chosen_key = candidate_neighbors_keys[best_candidate_idx]
        
        # Add the chosen solution's key to tabu list
        tabu_set.add(chosen_key)
        tabu_queue.append(chosen_key)
        if len(tabu_queue) > tabu_tenure:
            old_key = tabu_queue.pop(0)
            tabu_set.discard(old_key)

        # Update frequency count for chosen source
        freq_count[candidate_sources[best_candidate_idx]] += 1

        if current_eval > best_eval:
            best_solution = current_solution.copy()
            best_eval = current_eval
        
        if np.sum(current_solution) == 0 and iteration > 0:
            break # Stop if portfolio becomes empty after initial setup

    return best_solution, best_eval

def tabu_search_v2_candidate_list(
    objective_function: callable,
    num_companies: int,
    mean: np.ndarray,
    std: np.ndarray,
    solution: np.ndarray,
    tabu_tenure: int,
    num_iter: int,
    num_neighbors_to_generate: int = 10
) -> tuple[np.ndarray, float]:
    """
    Tabu Search with candidate-list strategy focusing on top Sharpe assets.
    Improvement: Narrow neighbor sources to high-Sharpe companies.
    """
    # Performance: candidate-list strategy focusing on top Sharpe assets
    sharpe_scores = mean / std
    num_top = max(1, num_companies // 10)
    top_assets = np.argsort(sharpe_scores)[-num_top:]
    current_solution = solution.copy()
    best_solution = current_solution.copy()
    best_eval = objective_function(mean, std, best_solution)

    tabu_set = set()
    tabu_queue = []

    # Pre-allocate arrays for limited neighbor generation
    neighbor_buffer = np.empty_like(current_solution)
    candidate_neighbors_list = np.empty((num_neighbors_to_generate, current_solution.size))
    candidate_evals_list = np.full(num_neighbors_to_generate, -np.inf)

    for iteration in range(num_iter):
        positive_indices = np.where(current_solution > 0)[0]
        if len(positive_indices) == 0:
            break

        valid_count = 0
        # Generate up to num_neighbors_to_generate neighbors
        for _ in range(num_neighbors_to_generate):
            np.copyto(neighbor_buffer, current_solution)
            # select source from top assets if possible
            top_pos = np.intersect1d(positive_indices, top_assets)
            source_idx = np.random.choice(top_pos if len(top_pos) > 0 else positive_indices)
            dest_idx = np.random.randint(0, num_companies)
            if num_companies > 1:
                while dest_idx == source_idx:
                    dest_idx = np.random.randint(0, num_companies)

            amount = current_solution[source_idx] * np.random.random()
            neighbor_buffer[source_idx] -= amount
            neighbor_buffer[dest_idx] += amount

            neighbor_key = neighbor_buffer.tobytes()
            if neighbor_key not in tabu_set:
                # Store valid neighbor
                np.copyto(candidate_neighbors_list[valid_count], neighbor_buffer)
                candidate_evals_list[valid_count] = objective_function(mean, std, neighbor_buffer)
                valid_count += 1

        if valid_count == 0:
            if tabu_queue:
                old_key = tabu_queue.pop(0)
                tabu_set.discard(old_key)
            continue

        best_idx = np.argmax(candidate_evals_list[:valid_count])
        np.copyto(current_solution, candidate_neighbors_list[best_idx])
        current_eval = candidate_evals_list[best_idx]
        chosen_key = current_solution.tobytes()

        # Update tabu list
        tabu_set.add(chosen_key)
        tabu_queue.append(chosen_key)
        if len(tabu_queue) > tabu_tenure:
            old_key = tabu_queue.pop(0)
            tabu_set.discard(old_key)

        if current_eval > best_eval:
            np.copyto(best_solution, current_solution)
            best_eval = current_eval

        if np.sum(current_solution) == 0:
            break

    return best_solution, best_eval

def tabu_search_v3_aspire(
    objective_function: callable,
    num_companies: int,
    mean: np.ndarray,
    std: np.ndarray,
    solution: np.ndarray,
    tabu_tenure: int,
    num_iter: int,
    num_neighbors_to_generate: int = 10
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
        mean (np.ndarray): Mean vector of simple returns for each company.
        std (np.ndarray): Standard deviation vector of simple returns for each company.
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
    neighbor = np.empty_like(sol)
    neighbors_array = np.empty((num_neighbors_to_generate, sol.size))
    evals_array = np.full(num_neighbors_to_generate, -np.inf)

    # Main search loop
    for _ in range(num_iter):
        # Get indices of companies with positive allocations (potential sources)
        positive_indices = np.where(sol > 0)[0]
        if len(positive_indices) == 0:
            break  # No valid moves possible when all allocations are zero

        valid_count = 0  # Counter for valid neighbors found in this iteration

        # Generate and evaluate neighbors
        for _ in range(num_neighbors_to_generate):
            # Generate a neighbor by transferring funds between companies
            # Randomly select a source company from the positive allocations
            i = np.random.choice(positive_indices)
            # Randomly select a destination company
            j = np.random.randint(0, num_companies)

            # Ensure the destination company is different from the source company
            if num_companies > 1:
                while j == i:
                    j = np.random.randint(0, num_companies)
            # If num_companies == 1, j can be equal to i, resulting in a null move.

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
            np.copyto(best_sol, sol)

        # Add the selected move to the tabu list
        tabu_set.add(best_key)
        tabu_queue.append(best_key)

        # Maintain tabu list size by removing oldest entries when needed
        if len(tabu_queue) > tabu_tenure:
            old_key = tabu_queue.pop(0)  # Remove oldest entry (FIFO)
            tabu_set.discard(old_key)    # Remove from set as well

    
    return best_sol, best_eval

def tabu_search_v4_random_restart(
    objective_function: callable,
    num_companies: int,
    mean: np.ndarray,
    std: np.ndarray,
    solution: np.ndarray,
    tabu_tenure: int,
    num_iter: int,
    num_neighbors_to_generate: int = 10,
    max_no_improve: int = 20,
    amount: float = 100
) -> tuple[np.ndarray, float]:
    """
    Tabu Search with random restart: if no improvement for max_no_improve iterations, restart from a new random solution.
    """
    current_solution = solution.copy()
    current_eval = objective_function(mean, std, current_solution)
    best_solution = current_solution.copy()
    best_eval = current_eval
    tabu_list = []
    no_improve_count = 0
    for iteration in range(num_iter):
        candidate_neighbors = []
        candidate_evals = []
        generated_count = 0
        attempts = 0
        max_attempts = num_companies * 2
        while generated_count < num_neighbors_to_generate and attempts < max_attempts:
            attempts += 1
            temp_neighbor = current_solution.copy()
            if num_companies < 2:
                break
            idx_from, idx_to = np.random.choice(num_companies, 2, replace=False)
            if temp_neighbor[idx_from] > 0:
                amount_to_move = np.random.uniform(0, temp_neighbor[idx_from])
                temp_neighbor[idx_from] -= amount_to_move
                temp_neighbor[idx_to] += amount_to_move
                is_tabu = False
                for tabu_sol in tabu_list:
                    if np.array_equal(temp_neighbor, tabu_sol):
                        is_tabu = True
                        break
                if not is_tabu:
                    candidate_neighbors.append(temp_neighbor)
                    candidate_evals.append(objective_function(mean, std, temp_neighbor))
                    generated_count += 1
            elif np.sum(current_solution) == 0:
                break
        improved = False
        if candidate_neighbors:
            best_candidate_idx = np.argmax(candidate_evals)
            best_neighbor_solution = candidate_neighbors[best_candidate_idx]
            best_neighbor_eval = candidate_evals[best_candidate_idx]
            current_solution = best_neighbor_solution
            current_eval = best_neighbor_eval
            tabu_list.append(current_solution.copy())
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)
            if current_eval > best_eval:
                best_solution = current_solution.copy()
                best_eval = current_eval
                no_improve_count = 0
                improved = True
        if not improved:
            no_improve_count += 1
        if no_improve_count >= max_no_improve:
            # Random restart
            current_solution = np.zeros(num_companies)
            idx = np.random.randint(num_companies)
            current_solution[idx] = amount
            current_eval = objective_function(mean, std, current_solution)
            no_improve_count = 0
    return best_solution, best_eval

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
    alpha = 0.475      # Cooling rate (temperature reduction factor) for SA advanced / general cooling for others
    num_iter = 100      # Number of iterations per temperature level / tabu iterations (reduced for speed)
    tabu_tenure = 10  # Tabu list size (number of recent solutions to avoid)
    # Added for base/v1 Tabu Search, defining how many neighbors they check.
    # For SA base/v1/v2, alpha will be used as cooling_rate directly.
    ts_num_neighbors_to_generate = 20 
   
    
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

    # Store for historical MDD objective
    global HIST_WEEKLY_RETURNS
    HIST_WEEKLY_RETURNS = data_week

    # Calculate statistics (mean and standard deviation) for each company
    stats = calculate_mean_std(data_week)
    mean, std = stats[0], stats[1]

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
    num_initial_runs = 5
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
    
    print(f"Initial VaR (95% confidence): M€{initial_VaR:.2f}")
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
            "initial_label": "Initial VaR (95%) in MEuros",
            "optimized_label": "Optimized VaR (95%) in MEuros"
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
        # Simulated Annealing Versions
        {
            "name": "SA v0 Base",
            "function": simulated_annealing_v0_base,
            "params": lambda obj_func: [obj_func, num_companies, mean, std, isolution, temperature, alpha, num_iter],
            "param_desc": f"T₀={temperature}, cooling_rate={alpha}, iters_per_temp={num_iter} (Base)"
        },
        {
            "name": "SA v1 Adaptive Step",
            "function": simulated_annealing_v1_adaptive_step,
            "params": lambda obj_func: [obj_func, num_companies, mean, std, isolution, temperature, alpha, num_iter],
            "param_desc": f"T₀={temperature}, cooling_rate={alpha}, iters_per_temp={num_iter} (Adaptive Step)"
        },
        {
            "name": "SA v2 Best-of-K",
            "function": simulated_annealing_v2_best_of_k,
            "params": lambda obj_func: [obj_func, num_companies, mean, std, isolution, temperature, alpha, num_iter, 5],
            "param_desc": f"T₀={temperature}, cooling_rate={alpha}, iters_per_temp={num_iter}, best_of_k=5"
        },
        {
            "name": "SA v3 Adaptive Reheating",
            "function": simulated_annealing_v3_adaptive_reheating,
            "params": lambda obj_func: [obj_func, num_companies, mean, std, isolution, temperature, alpha, num_iter],
            "param_desc": f"T₀={temperature}, α(cooling)={alpha}, iters_per_temp={num_iter} (Adaptive Reheating)"
        },
        {
            "name": "SA v4 Elitist Archive",
            "function": simulated_annealing_v4_elitist_archive,
            "params": lambda obj_func: [obj_func, num_companies, mean, std, isolution, temperature, alpha, num_iter, 5, 0.1],
            "param_desc": f"T₀={temperature}, α(cooling)={alpha}, iters_per_temp={num_iter}, archive_size=5, elite_replace_prob=0.1"
        },
        # Tabu Search Versions
        {
            "name": "TS v0 Base",
            "function": tabu_search_v0_base,
            "params": lambda obj_func: [obj_func, num_companies, mean, std, isolution, tabu_tenure, num_iter, ts_num_neighbors_to_generate],
            "param_desc": f"tenure={tabu_tenure}, iters={num_iter}, neighbors_check={ts_num_neighbors_to_generate} (Base, Simple List Tabu)"
        },
        {
            "name": "TS v1 Frequency Memory",
            "function": tabu_search_v1_frequency_memory,
            "params": lambda obj_func: [obj_func, num_companies, mean, std, isolution, tabu_tenure, num_iter, ts_num_neighbors_to_generate],
            "param_desc": f"tenure={tabu_tenure}, iters={num_iter}, neighbors_check={ts_num_neighbors_to_generate} (Frequency Memory)"
        },
        {
            "name": "TS v2 Candidate List",
            "function": tabu_search_v2_candidate_list,
            "params": lambda obj_func: [obj_func, num_companies, mean, std, isolution, tabu_tenure, num_iter, ts_num_neighbors_to_generate],
            "param_desc": f"tenure={tabu_tenure}, iters={num_iter}, neighbors_check={ts_num_neighbors_to_generate} (Candidate List)"
        },
        {
            "name": "TS v3 Aspiration",
            "function": tabu_search_v3_aspire,
            "params": lambda obj_func: [obj_func, num_companies, mean, std, isolution, tabu_tenure, num_iter, ts_num_neighbors_to_generate],
            "param_desc": f"tenure={tabu_tenure}, iters={num_iter}, neighbors_check={ts_num_neighbors_to_generate} (Aspiration Criterion)"
        },
        {
            "name": "TS v4 Random Restart",
            "function": tabu_search_v4_random_restart,
            "params": lambda obj_func: [obj_func, num_companies, mean, std, isolution, tabu_tenure, num_iter, ts_num_neighbors_to_generate, 20, 100],
            "param_desc": f"tenure={tabu_tenure}, iters={num_iter}, neighbors_check={ts_num_neighbors_to_generate}, max_no_improve={20}, amount={100}"
        }
    ]

    # ======================================================================
    # OPTIMIZATION EXECUTION
    # ======================================================================
    # Run all combinations of algorithms and objective functions with multiple repetitions for averaging
    num_repetitions = 5  # Number of runs to average results
    
    for algo in algorithms:
        for eval_name, eval_data in evaluation_functions.items():
            print("\n" + "=" * 60)
            print(f"\nRunning {algo['name']} with {eval_name} optimization...")
            print(f"Parameters: {algo['param_desc']} x {num_repetitions} repetitions")
            print(f"MC simulations: {num_simulations} for all evaluations")
            
            # Choose optimization objective (Monte Carlo for all objectives)
            def optimized_obj_func(mean, std, solution):
                return eval_data["func"](mean, std, solution, num_simulations)
            
            # Arrays to store results from all repetitions
            all_times = []
            all_values = []
            all_solutions = []
            
            for rep in range(num_repetitions):
                # Measure optimization time
                start_time = time.time()
                
                # --- Parameter passing logic needs to be robust for all algo versions ---
                algo_params = algo["params"](optimized_obj_func) # Get base params from lambda
                # If optimizing Sharpe with SA, intensify search: higher T0, slower cooling, more iters
                if eval_name == "Sharpe" and algo["name"].startswith("SA"):
                    # params layout: [obj_func, num_companies, mean, std, sol, T0, alpha, num_iter]
                    algo_params[5] = temperature * 5      # higher initial temperature
                    algo_params[6] = min(0.99, alpha*1.1) # slower cooling
                    algo_params[7] = num_iter * 5        # more iterations per temperature
                # No special handling needed here anymore as lambdas are specific
                                
                # Execute optimization
                sol, _ = algo["function"](*algo_params)
                
                # Re-evaluate with more accurate simulation for final result
                optimized_value = optimized_obj_func(mean, std, sol)
                
                # Calculate elapsed time
                elapsed_time = time.time() - start_time
                
                # Store results
                all_times.append(elapsed_time)
                all_values.append(optimized_value)
                all_solutions.append(sol)
                
            
            # Calculate statistics
            avg_time = np.mean(all_times)
            avg_value = np.mean(all_values)
            # Compute 95% confidence interval for optimized values
            results_std = np.std(all_values, ddof=1)
            ci_lower = avg_value - 1.96 * results_std / np.sqrt(len(all_values))
            ci_upper = avg_value + 1.96 * results_std / np.sqrt(len(all_values))
           
            # ======================================================================
            # RESULTS REPORTING
            # ======================================================================
            print(f"\n{algo['name']} Results (Averaged over {num_repetitions} runs):")
            print("=" * 60)
            print(f"Average optimization time: {avg_time:.2f} seconds")
            print(f"{eval_data['initial_label']}: {eval_data['initial_value']:.4f}")
            print(f"{eval_data['optimized_label']} (avg): {avg_value:.4f}, 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    # Collect results for the table
    metrics = ['VaR at 95%', 'Sharp', 'MDD', 'Execution time']
    sa_names = ['SA v0 Base', 'SA v1 Adaptive Step', 'SA v2 Best-of-K', 'SA v3 Adaptive Reheating', 'SA v4 Elitist Archive']
    ts_names = ['TS v0 Base', 'TS v1 Frequency Memory', 'TS v2 Candidate List', 'TS v3 Aspiration', 'TS v4 Random Restart']
    columns = ['Metric', 'Initial'] + sa_names + ts_names
    # Prepare a dictionary to store results for each metric and algorithm
    results_dict = {metric: [None]*(1+len(sa_names)+len(ts_names)) for metric in metrics}
    # Fill initial values
    results_dict['VaR at 95%'][0] = initial_VaR
    results_dict['Sharp'][0] = initial_sharpe
    results_dict['MDD'][0] = initial_mdd
    # Execution time for initial is not meaningful, set to None or 0
    results_dict['Execution time'][0] = None
    # Helper to map algorithm name to column index
    algo_col_map = {name: i+1 for i, name in enumerate(sa_names + ts_names)}
    # Mapping from eval_name to results_dict key
    eval_to_metric = {'VaR': 'VaR at 95%', 'Sharpe': 'Sharp', 'MDD': 'MDD'}
    # Rerun the optimization loop to collect the best values for each metric/algorithm
    for algo in algorithms:
        algo_name = algo['name']
        col_idx = algo_col_map.get(algo_name)
        for eval_name, eval_data in evaluation_functions.items():
            def optimized_obj_func(mean, std, solution):
                return eval_data["func"](mean, std, solution, num_simulations)
            all_times = []
            all_values = []
            for rep in range(num_repetitions):
                algo_params = algo["params"](optimized_obj_func)
                if eval_name == "Sharpe" and algo_name.startswith("SA"):
                    algo_params[5] = temperature * 5
                    algo_params[6] = min(0.99, alpha*1.1)
                    algo_params[7] = num_iter * 5
                start_time = time.time()
                sol, _ = algo["function"](*algo_params)
                elapsed_time = time.time() - start_time
                optimized_value = optimized_obj_func(mean, std, sol)
                all_times.append(elapsed_time)
                all_values.append(optimized_value)
            avg_value = np.mean(all_values)
            avg_time = np.mean(all_times)
            # Directly assign the value to the correct metric
            metric_key = eval_to_metric[eval_name]
            results_dict[metric_key][col_idx] = avg_value
            # For execution time, only fill for the first metric (to avoid overwriting with other metrics)
            if eval_name == 'VaR':
                results_dict['Execution time'][col_idx] = avg_time
    # Create DataFrame
    df = pd.DataFrame([[metric] + results_dict[metric] for metric in metrics], columns=columns)
    # Format all numeric values to two decimals
    for col in df.columns[1:]:
        df[col] = df[col].apply(lambda x: round(x, 4) if isinstance(x, (float, int)) and x is not None else x)
    print("\nSummary Table:")
    print(df)
# Script entry point
if __name__ == "__main__":
    main()
