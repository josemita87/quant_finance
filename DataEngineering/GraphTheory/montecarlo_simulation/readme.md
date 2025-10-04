# Monte Carlo Epidemic Simulation

## Overview
This project implements Monte Carlo simulations of epidemic spreading on random networks. It models how diseases spread through populations using three classic epidemiological models (SI, SIS, SIR) and allows for visualization and analysis of transmission dynamics under various conditions.

## Features
### Multiple Epidemic Models:
- **SI (Susceptible-Infected):** Infected individuals remain permanently infected.
- **SIS (Susceptible-Infected-Susceptible):** Infected individuals can recover and become susceptible again.
- **SIR (Susceptible-Infected-Recovered):** Infected individuals recover with immunity.

### Network Generation:
- Erdős-Rényi random graph model.
- Customizable network sizes and densities.

### Visualization Tools:
- Network state visualization with color-coded nodes.
- Evolution of epidemic states over time.
- Impact of parameter variations (infection rate, recovery rate, network connectivity).
- Super-spreader analysis.

## Installation
### Dependencies
Ensure you have the following dependencies installed:
```bash
pip install numpy networkx matplotlib
```

### Clone Repository
```bash
git clone https://github.com/your-repo/monte-carlo-epidemic.git
cd monte-carlo-epidemic
```

## Usage
Run the simulation demo with:
```bash
python simulation.py
```

## Models Explanation
### SI Model (Susceptible-Infected)
- Susceptible individuals become infected with probability **β** times the number of infected neighbors.
- Once infected, individuals remain infected permanently.
- Simple model for diseases with no recovery.

### SIS Model (Susceptible-Infected-Susceptible)
- Susceptible individuals become infected with probability **β** times the number of infected neighbors.
- Infected individuals recover with probability **μ** and become susceptible again.
- Models diseases where recovery doesn't confer immunity (like the common cold).

### SIR Model (Susceptible-Infected-Recovered)
- Susceptible individuals become infected with probability **β** times the number of infected neighbors.
- Infected individuals recover with probability **μ** and gain permanent immunity.
- Models diseases like measles or chickenpox where recovery typically provides immunity.

## Available Analyses
### Network Visualization
Visualize network states with nodes colored by infection status:
- **Blue:** Susceptible.
- **Red:** Infected.
- **Green:** Recovered.

### Evolution Over Time
Track changes in population fractions (S/I/R) over time for different initial conditions.

### Parameter Analysis
- **Ratio vs β:** How infection rates affect final epidemic size.
- **Ratio vs Connectivity:** How network connectivity affects disease spread.
- **Super-Spreader Impact:** Compare random infection vs. targeting high-degree nodes.

## Key Parameters
- **β (beta):** Infection rate parameter (0 to 1).
- **μ (mu):** Recovery rate parameter (0 to 1).
- **n:** Number of nodes in the network.
- **e:** Number of edges in the network.
- **steps:** Simulation time steps.
- **repetitions:** Number of Monte Carlo repetitions to average results.

## Technical Details
The simulation uses:
- Normalized adjacency matrices to represent networks.
- Monte Carlo method to simulate probabilistic transitions between states.
- NetworkX for graph visualization.
- NumPy for efficient vector operations.

## Authors
- **Josep**
- **Anna**
- **Marco**
- **Luuk**