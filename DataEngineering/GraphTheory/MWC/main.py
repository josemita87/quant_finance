#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Assignment 2: Monte Carlo Simulation

@author: Josep, Anna, Marco, Luuk
"""
import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from colorama import Fore
from itertools import product
from typing import List, Tuple, Union, Optional

   
def print_matrix(G: np.ndarray) -> None:
    """
    Print a matrix with colored output based on cell values.
    
    Parameters
    ----------
    G : np.ndarray
        Matrix to print
    
    Returns
    -------
    None
        Prints the matrix to console
    """
    for i, j in product(range(G.shape[0]), range(G.shape[1])):
        ending = "\n" if j == G.shape[0]-1 else ""
        if G[i, j] == 0:
            print(Fore.WHITE, f'{round(G[i, j], 2)}', end=ending)
        else:
            print(Fore.GREEN, f'{round(G[i, j], 2)}', end=ending)

def generate_erdos_renyi_graph(n: int, e: int) -> np.ndarray:
    """
    Generate an Erdős-Rényi random graph with n nodes and e edges.
    
    Parameters
    ----------
    n : int
        Number of nodes
    e : int
        Number of edges
    
    Returns
    -------
    np.ndarray
        Adjacency matrix of the generated graph
    """
    # Generate e unique random edges (i, j) where i ≠ j
    i, j = np.triu_indices(n, k=1)  # Get upper triangular indices (avoid self-loops & duplicates)

    # Pick e random edges (at most the number of possible edges)
    e = min(e, len(i))
    edge_indices = np.random.choice(len(i), e, replace=False) 
    
    # Create adjacency matrix and fill selected edges
    m = np.zeros((n, n), dtype=float)
    # Fill the adjacency matrix with the selected edges (symmetric matrix)
    m[i[edge_indices], j[edge_indices]] = 1
    m[j[edge_indices], i[edge_indices]] = 1  

    return m

def draw_degree_distribution(graph: np.ndarray) -> None:
    """
    Plot the out-degree distribution of a graph.
    
    Parameters
    ----------
    graph : np.ndarray
        Adjacency matrix of the graph
    
    Returns
    -------
    None
        Displays a histogram of node out-degrees
    """
    # Sum of each row gives the out-degree of each node
    out_degrees = np.sum(graph, axis=1)
    
    # Count how many times each degree occurs
    unique_degrees, counts = np.unique(out_degrees, return_counts=True)

    plt.figure(figsize=(8, 5))
    plt.bar(unique_degrees, counts, color='blue', edgecolor='black')
    plt.xlabel('Out-degree')
    plt.ylabel('Number of nodes')
    plt.title('Out-degree Distribution')
    plt.grid(True, alpha=0.3)
    plt.show()

def normalize_out_degree(graph: np.ndarray) -> np.ndarray:
    """
    Normalize the weight of each edge according to the out-degree of its source node.
    
    Parameters
    ----------
    graph : np.ndarray
        Adjacency matrix of the graph
    
    Returns
    -------
    np.ndarray
        Normalized adjacency matrix
    """
    # Compute the out-degree (sum of each row)
    out_degree = np.sum(graph, axis=1, keepdims=True)  

    # Avoid division by zero (for nodes with no outgoing edges)
    out_degree[out_degree == 0] = 1  
    
    # Normalize each row by its out-degree
    return graph / out_degree

def initial_infection(n: int, istates: int = 1) -> np.ndarray:
    """
    Initialize a population with specified number of infected individuals.
    
    Parameters
    ----------
    n : int
        Total population size
    istates : int, optional
        Number of initially infected individuals, by default 1
    
    Returns
    -------
    np.ndarray
        Array of node states ('s' for susceptible, 'i' for infected)
    """
    # Create an array of susceptible nodes
    state_nodes = np.array(['s'] * n)
    
    # Ensure we don't try to infect more nodes than available
    istates = min(istates, n)
    
    # Infect istates random distinct nodes
    infected_indices = random.sample(range(n), istates)
    state_nodes[infected_indices] = 'i'
    
    return state_nodes

def _calculate_infection_probability(graph: np.ndarray, state_nodes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate infection probabilities and create infected mask.
    
    Parameters
    ----------
    graph : np.ndarray
        Adjacency matrix of the graph
    state_nodes : np.ndarray
        Array of node states
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        infected_mask: Boolean mask of infected nodes
        infected_connections: Sum of infected neighbors for each node
    """
    infected_mask = (state_nodes == "i").astype(float)
    infected_connections = graph @ infected_mask
    return infected_mask, infected_connections

def SI_model(graph: np.ndarray, state_nodes: np.ndarray, beta: float, mu: Optional[float] = None) -> np.ndarray:
    """
    Susceptible-Infected (SI) epidemic model.
    
    In this model, susceptible individuals become infected with probability β times 
    the number of infected neighbors. Once infected, individuals remain infected.
    
    Parameters
    ----------
    graph : np.ndarray
        Adjacency matrix of the graph
    state_nodes : np.ndarray
        Current state of each node ('s' or 'i')
    beta : float
        Infection rate parameter (0 to 1)
    mu : float, optional
        Recovery rate parameter (not used in SI model)
    
    Returns
    -------
    np.ndarray
        Updated state of each node
    """
    # Get infection probabilities
    _, infected_connections = _calculate_infection_probability(graph, state_nodes)

    # Generate random probabilities and determine new infections
    new_infections = (np.random.rand(len(graph)) < (beta * infected_connections)) & (state_nodes == "s")

    # Update state: keep infected as "i", and turn newly infected "s" into "i"
    return np.where(new_infections, "i", state_nodes)

def SIS_model(graph: np.ndarray, state_nodes: np.ndarray, beta: float, mu: float) -> np.ndarray:
    """
    Susceptible-Infected-Susceptible (SIS) epidemic model.
    
    In this model, susceptible individuals become infected with probability β times 
    the number of infected neighbors. Infected individuals recover with probability μ 
    and become susceptible again.
    
    Parameters
    ----------
    graph : np.ndarray
        Adjacency matrix of the graph
    state_nodes : np.ndarray
        Current state of each node ('s' or 'i')
    beta : float
        Infection rate parameter (0 to 1)
    mu : float
        Recovery rate parameter (0 to 1)
    
    Returns
    -------
    np.ndarray
        Updated state of each node
    """
    # Get infection probabilities
    infected_mask, infected_connections = _calculate_infection_probability(graph, state_nodes)

    # Generate random probabilities and determine new recoveries
    recoveries = (np.random.rand(len(graph)) < mu) & (infected_mask == 1)

    # Initialize new_state_nodes with the current state, but recovered nodes become susceptible
    new_state_nodes = np.where(recoveries, "s", state_nodes)

    # Generate random probabilities and determine new infections
    new_infections = (np.random.rand(len(graph)) < (beta * infected_connections)) & (new_state_nodes == "s")

    # Update state with new infections
    return np.where(new_infections, "i", new_state_nodes)

def SIR_model(graph: np.ndarray, state_nodes: np.ndarray, beta: float, mu: float) -> np.ndarray:
    """
    Susceptible-Infected-Recovered (SIR) epidemic model.
    
    In this model, susceptible individuals become infected with probability β times 
    the number of infected neighbors. Infected individuals recover with probability μ 
    and gain permanent immunity.
    
    Parameters
    ----------
    graph : np.ndarray
        Adjacency matrix of the graph
    state_nodes : np.ndarray
        Current state of each node ('s', 'i', or 'r')
    beta : float
        Infection rate parameter (0 to 1)
    mu : float
        Recovery rate parameter (0 to 1)
    
    Returns
    -------
    np.ndarray
        Updated state of each node
    """
    # Get infection probabilities
    infected_mask, infected_connections = _calculate_infection_probability(graph, state_nodes)

    # Generate random probabilities and determine new infections and recoveries
    new_infections = (np.random.rand(len(graph)) < (beta * infected_connections)) & (state_nodes == "s")
    recoveries = (np.random.rand(len(graph)) < mu) & (state_nodes == "i")
    
    # Create new state array and update
    new_state_nodes = state_nodes.copy()
    new_state_nodes[new_infections] = "i"
    new_state_nodes[recoveries] = "r"
    
    return new_state_nodes

def rmse(current_state: np.ndarray, previous_state: np.ndarray) -> float:
    """
    Calculate the Root Mean Square Error between two state vectors.
    
    This can be used to measure convergence in simulations.
    
    Parameters
    ----------
    current_state : np.ndarray
        Current state vector
    previous_state : np.ndarray
        Previous state vector
    
    Returns
    -------
    float
        RMSE between the states
    
    Raises
    ------
    ValueError
        If the state vectors have different lengths
    """
    if len(current_state) != len(previous_state):
        raise ValueError("State vectors must have the same length")
    
    # Map infected to 1 and all others to 0 for numerical difference
    cur = np.where(current_state == "i", 1, 0)
    prev = np.where(previous_state == "i", 1, 0)
    
    # Calculate RMSE
    n = len(current_state)
    
    return np.sqrt(np.sum((cur - prev)**2) / n)

class SimulationPlotter:
    """
    A class for simulating and plotting epidemic models on networks.
    Encapsulates common functionality for different plot types.
    """
    
    def __init__(self, models=[SI_model, SIS_model, SIR_model], steps=100, repetitions=100):
        """
        Initialize the SimulationPlotter with common parameters.
        
        Parameters
        ----------
        models : list
            List of model functions to simulate
        steps : int
            Number of simulation steps
        repetitions : int
            Number of simulation repetitions to average over
        """
        self.models = models
        self.steps = steps
        self.repetitions = repetitions
        
        # Define common styling elements
        self.line_styles = {'s': '-', 'i': '--', 'r': ':'}
        self.model_colors = {
            models[0].__name__: 'blue',
            models[1].__name__: 'green',
            models[2].__name__: 'red'
        } if len(models) >= 3 else {}
        
        # Define state colors for network visualization
        self.node_colors = {'s': '#A0CBE2', 'i': 'red', 'r': 'green'}
        self.node_labels = {'s': 'Susceptible', 'i': 'Infected', 'r': 'Recovered'}

    def _setup_plot(self, figsize=(10, 8)):
        """Helper method to set up a plot with common settings"""
        plt.figure(figsize=figsize)
        return plt
    
    def _finalize_plot(self, xlabel=None, ylabel=None, title=None, ylim=None, show_grid=True, show_legend=True, show_plot=True):
        """Helper method to finalize plot with common settings"""
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        if title:
            plt.title(title)
        if ylim:
            plt.ylim(ylim)
        if show_grid:
            plt.grid(True)
        if show_legend:
            plt.legend()
        plt.tight_layout()
        if show_plot:
            plt.show()
    
    def _plot_network_colored_states(self, G, state_nodes):
        """
        Visualize a network with nodes colored according to their states.
        
        Parameters
        ----------
        G : np.ndarray
            Adjacency matrix of the network
        state_nodes : np.ndarray
            Array of node states ('s', 'i', 'r')
        
        Returns
        -------
        None
            Displays a plot of the network
        """
        graph = nx.from_numpy_array(G, create_using=nx.DiGraph())
        susceptible_nodes = np.where(state_nodes == "s")[0]
        infected_nodes = np.where(state_nodes == "i")[0]
        recovered_nodes = np.where(state_nodes == "r")[0]
        
        # Position nodes using circular layout algorithm
        pos = nx.circular_layout(graph)
        
        plt.axis('off')
        nx.draw_networkx_nodes(graph, pos, nodelist=susceptible_nodes, 
                              node_color=self.node_colors['s'], 
                              label=self.node_labels['s'])
        nx.draw_networkx_nodes(graph, pos, nodelist=infected_nodes, 
                              node_color=self.node_colors['i'], 
                              label=self.node_labels['i'])
        
        # Only draw recovered nodes if they exist
        if len(recovered_nodes) > 0:
            nx.draw_networkx_nodes(graph, pos, nodelist=recovered_nodes, 
                                  node_color=self.node_colors['r'], 
                                  label=self.node_labels['r'])
        
        nx.draw_networkx_edges(graph, pos, alpha=0.4)
        nx.draw_networkx_labels(graph, pos, font_size=12, font_family='verdana')
        
        plt.legend()
        plt.title('Network State Visualization')

    def _run_simulation(self, model, G, state_nodes, beta, mu, steps):
        """
        Run simulation for a given model and parameters.
        
        Parameters
        ----------
        model : callable
            Epidemic model function
        G : np.ndarray
            Adjacency matrix of the graph
        state_nodes : np.ndarray
            Initial state of nodes
        beta : float
            Infection rate parameter
        mu : float
            Recovery rate parameter
        steps : int
            Number of steps to simulate
            
        Returns
        -------
        np.ndarray
            Updated state of nodes after simulation
        """
        for step in range(steps):
            if model.__name__ == 'SI_model':
                # SI model doesn't use mu parameter
                state_nodes = model(G, state_nodes, beta)
            else:
                state_nodes = model(G, state_nodes, beta, mu)
        return state_nodes
    
    def _count_states(self, state_nodes, n):
        """
        Count the fraction of nodes in each state.
        
        Parameters
        ----------
        state_nodes : np.ndarray
            State of nodes
        n : int
            Total number of nodes
            
        Returns
        -------
        dict
            Dictionary with fractions of nodes in each state
        """
        return {
            's': np.count_nonzero(state_nodes == "s") / n,
            'i': np.count_nonzero(state_nodes == "i") / n,
            'r': np.count_nonzero(state_nodes == "r") / n
        }
    
    def _plot_state_curves(self, x_values, data, model_name, include_recovered=True):
        """Helper method to plot state curves with consistent styling"""
        plt.plot(x_values, data['s'], label=f"{model_name} - Susceptible", 
                 linestyle=self.line_styles['s'], color=self.model_colors[model_name])
        plt.plot(x_values, data['i'], label=f"{model_name} - Infected", 
                 linestyle=self.line_styles['i'], color=self.model_colors[model_name])
        
        # Plot recovered state conditionally
        if include_recovered and (model_name == 'SIR_model' or np.any(data['r'] > 0)):
            plt.plot(x_values, data['r'], label=f"{model_name} - Recovered", 
                     linestyle=self.line_styles['r'], color=self.model_colors[model_name])
    
    def simulate_and_plot_network(self, n, e, initial_infections, beta, mu, model_index=2, steps=None):
        """
        Simulate an epidemic on a network and visualize the final network state.
        
        Parameters
        ----------
        n : int
            Number of nodes
        e : int
            Number of edges
        initial_infections : int
            Number of initially infected nodes
        beta : float
            Infection rate parameter
        mu : float
            Recovery rate parameter
        model_index : int, optional
            Index of the model to use from self.models, by default 2 (SIR_model)
        steps : int, optional
            Number of steps to simulate, by default None (uses self.steps)
        
        Returns
        -------
        None
            Displays a plot of the network with colored nodes
        """
        # Use provided steps or default to self.steps
        steps = steps if steps is not None else self.steps
        
        # Ensure model_index is valid
        if model_index < 0 or model_index >= len(self.models):
            raise ValueError(f"model_index must be between 0 and {len(self.models)-1}")
        
        # Generate network
        G = generate_erdos_renyi_graph(n, e)
        G = normalize_out_degree(G)
        
        # Initialize infection state
        state_nodes = initial_infection(n, istates=initial_infections)
        
        # Get the model
        model = self.models[model_index]
        model_name = model.__name__
        
        # Run simulation
        state_nodes = self._run_simulation(model, G, state_nodes, beta, mu, steps)
        
        # Count states for title information
        state_counts = self._count_states(state_nodes, n)
        
        # Plot the network with colored states
        self._setup_plot(figsize=(10, 8))
        self._plot_network_colored_states(G, state_nodes)
        plt.suptitle(f"{model_name} after {steps} steps - β={beta:.2f}, μ={mu:.2f}\n" + 
                   f"S: {state_counts['s']:.2%}, I: {state_counts['i']:.2%}, " + 
                   f"R: {state_counts['r']:.2%}")
        plt.tight_layout()
        
        return state_nodes
    
    def plot_custom_network(self, n, e, state_distribution, title="Custom Network Visualization"):
        """
        Create and visualize a network with custom state distribution.
        
        Parameters
        ----------
        n : int
            Number of nodes
        e : int
            Number of edges
        state_distribution : dict
            Dictionary with percentage of 's', 'i', and 'r' states
        title : str, optional
            Plot title, by default "Custom Network Visualization"
        """
        G = generate_erdos_renyi_graph(n, e)
        
        # Initialize all nodes as susceptible
        state_nodes = np.array(['s'] * n)
        
        # Calculate counts for each state
        i_count = int(state_distribution.get('i', 0) * n)
        r_count = int(state_distribution.get('r', 0) * n)
        
        # Create random indices for each state
        all_indices = list(range(n))
        random.shuffle(all_indices)
        
        infected_indices = all_indices[:i_count]
        recovered_indices = all_indices[i_count:i_count+r_count]
        
        # Set states
        state_nodes[infected_indices] = 'i'
        state_nodes[recovered_indices] = 'r'
        
        # Visualize
        self._setup_plot(figsize=(10, 8))
        self._plot_network_colored_states(G, state_nodes)
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def plot_evolution_over_time(self, n, e, initial_infections_list, beta, mu):
        """
        Simulate and plot the evolution over time of state fractions for different initial infection counts.
        Each figure corresponds to one initial infection count and shows all models together.
        
        Parameters
        ----------
        n : int
            Number of nodes
        e : int
            Number of edges
        initial_infections_list : list
            List of numbers of initially infected nodes
        beta : float
            Infection rate parameter
        mu : float
            Recovery rate parameter
        """
        # Organize results by initial infection first, then by model
        results = {init_inf: {model.__name__: {} for model in self.models} for init_inf in initial_infections_list}
        
        # Gather simulation results
        for init_inf in initial_infections_list:
            for model in self.models:
                s_total = np.zeros(self.steps + 1)
                i_total = np.zeros(self.steps + 1)
                r_total = np.zeros(self.steps + 1)
                
                for rep in range(self.repetitions):
                    G_local = generate_erdos_renyi_graph(n, e)
                    G_local = normalize_out_degree(G_local)
                    state_nodes = initial_infection(n, istates=init_inf)
                    s_series, i_series, r_series = [], [], []
                    
                    for step in range(self.steps + 1):
                        state_counts = self._count_states(state_nodes, n)
                        s_series.append(state_counts['s'])
                        i_series.append(state_counts['i'])
                        r_series.append(state_counts['r'])
                        
                        if step < self.steps:
                            state_nodes = self._run_simulation(model, G_local, state_nodes, beta, mu, 1)
                    
                    s_total += np.array(s_series)
                    i_total += np.array(i_series)
                    r_total += np.array(r_series)
                
                results[init_inf][model.__name__] = {
                    's': s_total / self.repetitions,
                    'i': i_total / self.repetitions,
                    'r': r_total / self.repetitions
                }
        
        # Create plots
        time_steps = np.arange(self.steps + 1)
        for init_inf, model_results in results.items():
            self._setup_plot()
            
            for model_name, data in model_results.items():
                self._plot_state_curves(time_steps, data, model_name, include_recovered=True)
            
            self._finalize_plot(
                xlabel="Time Steps",
                ylabel="Fraction of Population",
                title=f"Evolution over Time with {init_inf} Initial Infections",
                ylim=(0, 1),
                show_grid=True,
                show_legend=True
            )
    
    def plot_ratio_vs_beta(self, n, e, beta_values, mu_values, initial_infections):
        """
        For a fixed initial infection count and different recovery rates, simulate and plot 
        the final state ratios as a function of the infection rate β.
        
        Parameters
        ----------
        n : int
            Number of nodes
        e : int
            Number of edges
        beta_values : array_like
            Values of infection rate to simulate
        mu_values : array_like
            Values of recovery rate to simulate
        initial_infections : int
            Number of initially infected nodes
        """
        # Organize results by mu, then by model
        results = {mu: {model.__name__: {} for model in self.models} for mu in mu_values}
        
        # Gather simulation results
        for mu in mu_values:
            for model in self.models:
                s_final, i_final, r_final = [], [], []
                
                for beta in beta_values:
                    s_acc, i_acc, r_acc = 0, 0, 0
                    
                    for _ in range(self.repetitions):
                        G_local = generate_erdos_renyi_graph(n, e)
                        G_local = normalize_out_degree(G_local)
                        state_nodes = initial_infection(n, istates=initial_infections)
                        
                        state_nodes = self._run_simulation(model, G_local, state_nodes, beta, mu, self.steps)
                        state_counts = self._count_states(state_nodes, n)
                        
                        s_acc += state_counts['s']
                        i_acc += state_counts['i']
                        r_acc += state_counts['r']
                    
                    s_final.append(s_acc / self.repetitions)
                    i_final.append(i_acc / self.repetitions)
                    r_final.append(r_acc / self.repetitions)
                
                results[mu][model.__name__] = {'s': s_final, 'i': i_final, 'r': r_final}
        
        # Create plots
        for mu, model_results in results.items():
            self._setup_plot()
            
            for model_name, data in model_results.items():
                # For the beta plot, only show recovered for SIR model
                self._plot_state_curves(beta_values, data, model_name, include_recovered=(model_name == 'SIR_model'))
            
            self._finalize_plot(
                xlabel="Infection Rate (β)",
                ylabel="Fraction of Population",
                title=f"Final State Ratios vs Infection Rate (μ = {mu})",
                ylim=(0, 1),
                show_grid=True,
                show_legend=True
            )
    
    def plot_ratio_vs_avg_degree(self, n, edges, beta, mu, initial_infections):
        """
        For a single network size, simulate and plot the final state ratios
        as a function of average node degree by varying the number of edges.
        
        Parameters
        ----------
        n : int
            Number of nodes
        edges : list
            List of edge counts to simulate
        beta : float
            Infection rate parameter
        mu : float
            Recovery rate parameter
        initial_infections : int
            Number of initially infected nodes
        """
        # Initialize results structure for each model
        results = {model.__name__: {'avg_degrees': [], 's': [], 'i': [], 'r': []} for model in self.models}
        
        for e in edges:
            outdegree = e/n  # Average degree
            
            for model in self.models:
                s_acc, i_acc, r_acc = 0, 0, 0
                
                for _ in range(self.repetitions):
                    G_local = generate_erdos_renyi_graph(n, e)
                    G_local = normalize_out_degree(G_local)
                    state_nodes = initial_infection(n, istates=initial_infections)
                    
                    state_nodes = self._run_simulation(model, G_local, state_nodes, beta, mu, self.steps)
                    state_counts = self._count_states(state_nodes, n)
                    
                    s_acc += state_counts['s']
                    i_acc += state_counts['i']
                    r_acc += state_counts['r']
                
                # Store results for this model at this edge count
                results[model.__name__]['avg_degrees'].append(outdegree)
                results[model.__name__]['s'].append(s_acc / self.repetitions)
                results[model.__name__]['i'].append(i_acc / self.repetitions)
                results[model.__name__]['r'].append(r_acc / self.repetitions)
        
        # Create plot
        self._setup_plot(figsize=(12, 8))
        
        for model_name, data in results.items():
            # For the degree plot, only show recovered for SIR model
            self._plot_state_curves(data['avg_degrees'], data, model_name, include_recovered=(model_name == 'SIR_model'))
        
        self._finalize_plot(
            xlabel="Average Node Degree",
            ylabel="Fraction of Population",
            title=f"Final State Ratios vs Average Node Degree (N = {n}, β = {beta}, μ = {mu})",
            ylim=(0, 1),
            show_grid=True,
            show_legend=True
        )

    def plot_super_spreader_impact(self, n, e, infection_count, beta, mu, model_index=2):
        """
        Compare epidemic spread based on different initial infection placement strategies:
        - Random nodes (baseline)
        - Highest-degree nodes (super-spreaders)
        - Lowest-degree nodes
        
        Parameters
        ----------
        n : int
            Number of nodes
        e : int
            Number of edges
        infection_count : int
            Number of initially infected nodes
        beta : float
            Infection rate parameter
        mu : float
            Recovery rate parameter
        model_index : int, optional
            Index of the model to use from self.models, by default 2 (SIR_model)
        """
        # Ensure model_index is valid
        if model_index < 0 or model_index >= len(self.models):
            raise ValueError(f"model_index must be between 0 and {len(self.models)-1}")
        
        # Get the model
        model = self.models[model_index]
        model_name = model.__name__
        
        # Generate network
        G = generate_erdos_renyi_graph(n, e)
        G_norm = normalize_out_degree(G)
        
        # Calculate node degrees
        degrees = np.sum(G, axis=1)
        
        # Create strategies for selecting initial infections
        strategies = {
            'Random': np.random.choice(n, infection_count, replace=False),
            'Highest-degree': np.argsort(degrees)[-infection_count:],
            'Lowest-degree': np.argsort(degrees)[:infection_count]
        }
        
        # Prepare to store results
        time_results = {strategy: {'s': [], 'i': [], 'r': []} for strategy in strategies}
        final_states = {}
        
        # Run simulations for each strategy
        for strategy_name, infected_indices in strategies.items():
            # Initialize all nodes as susceptible
            state_nodes = np.array(['s'] * n)
            # Set initially infected nodes
            state_nodes[infected_indices] = 'i'
            
            # Record initial state
            time_series = {
                's': [np.count_nonzero(state_nodes == "s") / n],
                'i': [np.count_nonzero(state_nodes == "i") / n],
                'r': [np.count_nonzero(state_nodes == "r") / n]
            }
            
            # Run simulation and record state at each step
            for step in range(self.steps):
                if model.__name__ == 'SI_model':
                    state_nodes = model(G_norm, state_nodes, beta)
                else:
                    state_nodes = model(G_norm, state_nodes, beta, mu)
                    
                # Record state
                time_series['s'].append(np.count_nonzero(state_nodes == "s") / n)
                time_series['i'].append(np.count_nonzero(state_nodes == "i") / n)
                time_series['r'].append(np.count_nonzero(state_nodes == "r") / n)
            
            # Store results
            time_results[strategy_name] = time_series
            final_states[strategy_name] = state_nodes
        
        # Plot time evolution for each strategy
        self._setup_plot(figsize=(12, 8))
        
        time_steps = np.arange(self.steps + 1)
        for strategy, data in time_results.items():
            plt.plot(time_steps, data['i'], label=f"{strategy}", linewidth=2)
        
        self._finalize_plot(
            xlabel="Time Steps",
            ylabel="Fraction Infected",
            title=f"Impact of Initial Infection Placement on {model_name} Dynamics\n(N = {n}, E = {e}, β = {beta}, μ = {mu})",
            ylim=(0, 1),
            show_grid=True,
            show_legend=True
        )
        
        # Plot final network states for each strategy
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        plt.suptitle(f"Final Network States After {self.steps} Steps ({model_name})", fontsize=16)
        
        for i, (strategy, state_nodes) in enumerate(final_states.items()):
            plt.sca(axes[i])
            plt.title(f"Strategy: {strategy}")
            self._plot_network_colored_states(G, state_nodes)
            
            # Add statistics
            s_pct = np.count_nonzero(state_nodes == "s") / n * 100
            i_pct = np.count_nonzero(state_nodes == "i") / n * 100
            r_pct = np.count_nonzero(state_nodes == "r") / n * 100
            
            stats = f"S: {s_pct:.1f}%, I: {i_pct:.1f}%, R: {r_pct:.1f}%"
            plt.annotate(stats, xy=(0.5, 0.02), xycoords='figure fraction', 
                        ha='center', fontsize=12)
        
        plt.tight_layout()
        plt.show()
        
        return time_results, final_states


if __name__ == "__main__":
    # Configuration parameters
    config = {
        'network': {
            'small': {'n': 30, 'e': 45},
            'medium': {'n': 50, 'e': 150},
            'large': {'n': 100, 'e': 300}
        },
        'epidemic': {
            'low': {'beta': 0.2, 'mu': 0.1},
            'medium': {'beta': 0.5, 'mu': 0.1},
            'high': {'beta': 0.8, 'mu': 0.1}
        }
    }
    
    # Create simulation plotter instance
    plotter = SimulationPlotter(
        models=[SI_model, SIS_model, SIR_model],
        steps=100,
        repetitions=50  # Lower for faster execution in demo
    )
    
    print("Monte Carlo Epidemic Simulation Demo")
    print("=" * 40)
    
    # 1. Visualize networks with different epidemic models
    print("\n1. Network Visualization with Different Models")
    for i, model in enumerate(["SI", "SIS", "SIR"]):
        print(f"  - Simulating {model} model...")
        plotter.simulate_and_plot_network(
            n=config['network']['small']['n'],
            e=config['network']['small']['e'],
            initial_infections=2,
            beta=config['epidemic']['medium']['beta'],
            mu=config['epidemic']['medium']['mu'],
            model_index=i
        )
    
    # 2. Custom network with specific state distribution
    print("\n2. Custom Network Visualization")
    plotter.plot_custom_network(
        n=config['network']['small']['n'],
        e=config['network']['small']['e'],
        state_distribution={'s': 0.6, 'i': 0.3, 'r': 0.1},
        title="Custom Network - 60% Susceptible, 30% Infected, 10% Recovered"
    )
    
    # 3. Evolution over time with different initial infections
    print("\n3. Evolution Over Time with Different Initial Infections")
    plotter.plot_evolution_over_time(
        n=config['network']['medium']['n'],
        e=config['network']['medium']['e'],
        initial_infections_list=[1, 5, 10],
        beta=config['epidemic']['medium']['beta'],
        mu=config['epidemic']['medium']['mu']
    )
    
    # 4. Ratio vs Beta with different recovery rates
    print("\n4. State Ratios vs Infection Rate (Beta)")
    plotter.plot_ratio_vs_beta(
        n=config['network']['medium']['n'],
        e=config['network']['medium']['e'],
        beta_values=np.linspace(0.05, 0.95, 10),  # 10 points for faster execution
        mu_values=[0.1, 0.5],  # Two recovery rates
        initial_infections=5
    )
    
    # 5. Ratio vs Average Degree
    print("\n5. State Ratios vs Average Node Degree")
    n = config['network']['medium']['n']
    plotter.plot_ratio_vs_avg_degree(
        n=n,
        edges=[int(n*0.5), int(n), int(n*2), int(n*3)],  # Different densities
        beta=config['epidemic']['medium']['beta'],
        mu=config['epidemic']['medium']['mu'],
        initial_infections=5
    )
    
    # 6. Super-Spreader Impact Analysis
    print("\n6. Impact of Super-Spreaders vs. Random Infection Placement")
    plotter.plot_super_spreader_impact(
        n=50,
        e=100,
        infection_count=3,
        beta=0.5,
        mu=0.1,
        model_index=2  # SIR model
    )
    
    print("\nSimulation Complete!")