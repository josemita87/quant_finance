# Connect Four AI

This repository contains an implementation of the classic **Connect Four** game with an AI opponent. The AI uses the **Minimax algorithm** with **heuristics** and can optionally utilize **Alpha-Beta pruning** for performance optimization. The game supports both player vs player and player vs AI modes.

## Features
- **Player vs Player**: Two human players can play against each other.
- **Player vs AI**: One player competes against an AI that uses a decision-making algorithm.
- **AI Strategy**: The AI uses the Minimax algorithm to simulate and evaluate all possible moves to select the best one.
  - Can use **Alpha-Beta pruning** for faster decision-making.
  - **Heuristics** such as prioritizing center columns, blocking opponent’s winning moves, and attempting to win immediately.

## How It Works
- The game is implemented on a 6x7 grid where players alternate placing their markers (either 'X' or 'O').
- The AI uses a **Minimax algorithm** to evaluate potential moves, choosing the most favorable outcome. It can simulate moves several turns ahead to predict and block an opponent’s strategy.
- Heuristic evaluations improve the AI's performance by prioritizing winning strategies and blocking the opponent's potential victories.
- You can adjust the depth of the AI's decision-making and choose whether to use Alpha-Beta pruning for optimization.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/connect-four-ai.git
   ```

2. Navigate to the project directory:
   ```bash
   cd connect-four-ai
   ```

3. Make sure you have Python installed (Python 3.x recommended).

4. Run the game:
   ```bash
   python connect_four.py
   ```

## Game Modes
- **Human vs Human**: Two players take turns to place their markers on the board.
- **Human vs AI**: The human plays against the AI, which uses the Minimax algorithm to decide the best move.

## How to Play

1. When it's your turn, enter the number of the column where you'd like to drop your piece (1-7).
2. The AI will take its turn automatically if you're playing against it.
3. The game ends when a player (or AI) forms a horizontal, vertical, or diagonal line of four consecutive pieces.

## AI Configuration

The AI uses the following settings that can be configured:
- **Minimax Algorithm**: Recursively evaluates the best possible moves.
- **Alpha-Beta Pruning**: Speeds up decision-making by pruning the search tree.
- **Heuristics**: Prioritizes moves that lead to winning, blocks the opponent’s win, and favors center columns.

## Session Overview

### SESSION 1

#### `def play_player(board, player)`
**Purpose**: Updates the board to reflect the player's move in the specified column. The move is placed in the lowest available row of that column.

#### `def check_p(board, column)`
**Purpose**: Determines the row index in the specified column where the player's move can be placed.

#### `def play_random_computer(board, player)`
**Purpose**: Simulates the computer's turn by randomly selecting a column from those that are not full and places its marker in the appropriate position.

#### `def play_heuristic_computer(board, player)`
**Purpose**: Chooses the best possible move for the computer by checking for moves that allow the computer to win immediately.

#### `def play_center_columns(board, player)`
**Purpose**: Selects an available column near the center of the board to maximize the chances of winning.

#### `def make_move(board, player)`
**Purpose**: Places the player's marker in the lowest available row of the specified column.

#### `def is_moves_left(board)`
**Purpose**: Checks whether there are any available moves left on the board.

#### `def check_win(board, player)`
**Purpose**: Determines whether a specified player has won the game by forming a sequence of four consecutive markers on the board (horizontal, vertical, or diagonal).

### SESSION 2

#### `def heuristic_score(board, player)`
**Purpose**: Determines how favorable the board is for the player based on positions that increase their chances of winning or block the opponent.

#### `def evaluate_window(window, player)`
**Purpose**: Assigns a score to a "window" of 4 cells that reflects how favorable it is for a given player. It checks for winning sequences, potential blocks, and near-wins.

#### `def minmax_algorithm(board, depth, maximizing_player, player, heuristic)`
**Purpose**: The minmax_algorithm function determines the best move in a Connect Four game using the Minimax algorithm. It evaluates potential future game states recursively up to a specified depth and alternates between the maximizing and minimizing players to find the optimal move.

**Parameters**:
- `board`: Current game state (2D list).
- `depth`: Maximum number of moves ahead to simulate in the search tree.
- `maximizing_player`: Boolean value indicating whether the current player is trying to maximize the score (True) or minimize the score (False).
- `player`: The current player’s marker (‘X’ or ‘O’) to be used for evaluation.
- `heuristic`: Boolean flag indicating whether to use heuristic scoring for evaluation.

**Function Logic**:
- **Base Case**: The recursion halts if `depth == 0`, if there are no valid columns to drop a piece, or if a win condition is detected. It returns the evaluation score of the current board.
- **Maximizing Player**: Simulates all valid moves for the maximizing player, evaluates the opponent's response recursively, and tracks the best score for the maximizing player.
- **Minimizing Player**: Simulates all valid moves for the minimizing player (opponent), evaluates the maximizing player's response, and tracks the worst score for the opponent.

**Output**: Returns the best column to play and its associated evaluation score.

#### `def minmax_with_alphabeta_pruning(board, depth, alpha, beta, maximizing_player, player, heuristic)`
**Purpose**: Optimizes the Minimax algorithm by using Alpha-Beta pruning, reducing the number of unnecessary node evaluations during the decision-making process.

**Parameters**:
- `board`: Current game state (2D list).
- `depth`: Maximum depth for the search tree.
- `alpha`: Best value achievable for the maximizing player (initially -inf).
- `beta`: Best value achievable for the minimizing player (initially +inf).
- `maximizing_player`: Whether the current player is maximizing the score.
- `player`: Current player’s marker (‘X’ or ‘O’).
- `heuristic`: Use heuristic scoring for evaluation if True.

**Function Logic**:
- **Base Case**: Stops recursion if depth is 0, no valid moves exist, or a win condition is detected.
- **Maximizing Player**: Simulates all valid moves and recursively evaluates the opponent’s responses, updating alpha and performing a beta cut-off.
- **Minimizing Player**: Simulates all valid moves, evaluates the maximizing player’s responses, and updates beta while performing an alpha cut-off.

**Output**: Returns the best column to play and its associated score.

#### `def play_computer_minmax(board, player, use_pruning=False, heuristic=False)`
**Purpose**: Determines the best move for the computer player using either the basic Minimax algorithm or the Alpha-Beta pruning optimization, depending on the `use_pruning` flag.

**Parameters**:
- `board`: Current game state (2D list).
- `player`: The computer’s marker (‘X’ or ‘O’).
- `use_pruning`: Boolean flag indicating whether to use Alpha-Beta pruning for optimization.
- `heuristic`: Boolean flag indicating whether to use heuristic scoring for evaluation.

**Function Logic**:
- **Base Case**: Checks if a valid move is available. If not, prints an error message.
- **Alpha-Beta Pruning**: If `use_pruning` is True, uses `minmax_with_alphabeta_pruning()` to determine the best column.
- **Basic Minimax**: If `use_pruning` is False, falls back to using `minmax_algorithm()` for decision-making.

**Output**: Returns the best column to play, chosen based on the evaluation of the game state.

### Heuristic Evaluation

The AI uses several heurist
