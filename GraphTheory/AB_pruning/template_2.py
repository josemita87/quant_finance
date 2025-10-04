from colorama import Fore, Back, Style
from os import system, name
import random
import math
'''
@authors: Oscar Thiele, Julian Herman, Adriana Martinez, Josep Cubedo
'''

def clear_screen():
    """Clear the console screen based on the operating system."""
    if name == 'nt':
        _ = system('cls')
    else:
        _ = system('clear')

def draw_board(board):
    """Print the current state of the Connect Four board with colors."""
    clear_screen()
    print("  1   2   3   4   5   6   7")  # Adjusted column spacing
    print("-" * 29)  # Adjusted border length
    for row in board:
        print("| ", end="")
        for cell in row:
            if cell == "X":
                print(Fore.RED + cell + Style.RESET_ALL + " | ", end="")  # Red X
            elif cell == "O":
                print(Fore.BLUE + cell + Style.RESET_ALL + " | ", end="")  # Blue O
            else:
                print(cell + " | ", end="")
        print("\n" + "-" * 29)

def user_input(board):
    """Prompt the user for a column input, validate it, and return it."""
    while True:
        try:
            column = int(input('Enter a column number (1-7): ')) - 1
            if 0 <= column < 7 and board[0][column] == " ":
                return column
            else:
                print('Column is full or out of range, please try another one.')
        except ValueError:
            print('Invalid input, please enter an integer.')

def evaluation_score(board, player, heuristic=False):
    """Evaluate the board and return a score based on the current state."""
    if heuristic:
        # Calculate heuristic score if heuristic parameter is True.
        return heuristic_score(board, player)
    else:
        # Simple win-loss evaluation
        opponent = 'X' if player == 'O' else 'O'
        if check_win(board, player):
            return 1000  # Win for player
        if check_win(board, opponent):
            return -1000  # Loss for player
        return 0  # Neutral or no win/loss detected

def heuristic_score(board, player):
    """Calculate the heuristic score for the board based on the player."""
    score = 0
    opponent = 'X' if player == 'O' else 'O'  # Identify the opponent's marker.

    # Score the center column (column index 3)
    # Center column is often strategically advantageous as it connects more potential winning lines.
    center_array = [row[3] for row in board]
    center_count = center_array.count(player)
    score += center_count * 3  # Give extra weight to center positions for the player.

    # Evaluate horizontal windows of 4 cells
    for row in board:
        for c in range(7 - 3):  # Iterate through all possible horizontal groups of 4.
            window = row[c:c+4]
            score += evaluate_window(window, player)  # Evaluate the group for the player's advantage.

    # Evaluate vertical windows of 4 cells
    for c in range(7):  # Loop through each column.
        col_array = [board[r][c] for r in range(6)]  # Extract the entire column as a list.
        for r in range(6 - 3):  # Iterate through all possible vertical groups of 4.
            window = col_array[r:r+4]
            score += evaluate_window(window, player)  # Evaluate the group for the player's advantage.

    # Evaluate positive diagonal windows of 4 cells
    # A positive diagonal goes from top-left to bottom-right.
    for r in range(6 - 3):  # Rows where a diagonal of 4 can start.
        for c in range(7 - 3):  # Columns where a diagonal of 4 can start.
            window = [board[r+i][c+i] for i in range(4)]  # Extract the diagonal group of 4.
            score += evaluate_window(window, player)  # Evaluate the diagonal for the player's advantage.

    # Evaluate negative diagonal windows of 4 cells
    # A negative diagonal goes from bottom-left to top-right.
    for r in range(6 - 3):  # Rows where a diagonal of 4 can start.
        for c in range(7 - 3):  # Columns where a diagonal of 4 can start.
            window = [board[r+3-i][c+i] for i in range(4)]  # Extract the diagonal group of 4.
            score += evaluate_window(window, player)  # Evaluate the diagonal for the player's advantage.

    return score


def evaluate_window(window, player):
    """Evaluate a window of four cells and assign a score."""
    score = 0
    opponent = 'X' if player == 'O' else 'O'  # Identify the opponent's marker.

    # Check for winning conditions for the player
    if window.count(player) == 4:  # If all 4 cells are occupied by the player, it's a winning group.
        score += 100
    elif window.count(player) == 3 and window.count(' ') == 1:  # 3 player markers and 1 empty space.
        score += 5
    elif window.count(player) == 2 and window.count(' ') == 2:  # 2 player markers and 2 empty spaces.
        score += 2

    # Check for potential blocks for the opponent
    if window.count(opponent) == 3 and window.count(' ') == 1:  # 3 opponent markers and 1 empty space.
        score -= 4  # Subtract points to prioritize blocking the opponent.

    return score


def get_valid_locations(board):
    """Get a list of columns that are not full."""
    valid_locations = []
    for col in range(7):  # Iterate over all columns (0 to 6).
        if board[0][col] == ' ':  # Check if the top cell of the column is empty.
            valid_locations.append(col)  # Add the column to the list of valid locations.
    return valid_locations  # Return the list of valid columns.


def is_moves_left(board):
    """Check if there are any moves left on the board."""
    for row in board:  # Iterate through each row of the board.
        if ' ' in row:  # If there is at least one empty cell (' '), moves are still possible.
            return True  # Return True as moves are available.
    return False  # If no empty cells are found, return False.


def make_move(board, col, player):
    """Place the player's marker in the chosen column."""
    # Iterate from the bottom-most row upwards to find the first empty cell in the column.
    for row in reversed(range(6)):  # Start from row 5 and move to row 0.
        if board[row][col] == ' ':  # Check if the cell is empty.
            board[row][col] = player  # Place the player's marker in the cell.
            return  # Exit the function after making the move.

def check_win(board, player):
    """Check if the specified player has won the game."""
    rows = len(board)  # Number of rows in the board (6).
    cols = len(board[0])  # Number of columns in the board (7).

    # Check horizontal wins
    for row in range(rows):  # Iterate through all rows.
        for col in range(cols - 3):  # Iterate through possible starting points for groups of 4.
            # Check if 4 consecutive cells in the row are occupied by the player.
            if (board[row][col] == player and
                board[row][col + 1] == player and
                board[row][col + 2] == player and
                board[row][col + 3] == player):
                return True

    # Check vertical wins
    for col in range(cols):  # Iterate through all columns.
        for row in range(rows - 3):  # Iterate through possible starting points for groups of 4.
            # Check if 4 consecutive cells in the column are occupied by the player.
            if (board[row][col] == player and
                board[row + 1][col] == player and
                board[row + 2][col] == player and
                board[row + 3][col] == player):
                return True

    # Check positive diagonal wins
    for row in range(rows - 3):  # Rows where positive diagonals can start.
        for col in range(cols - 3):  # Columns where positive diagonals can start.
            # Check if 4 consecutive cells in a positive diagonal are occupied by the player.
            if (board[row][col] == player and
                board[row + 1][col + 1] == player and
                board[row + 2][col + 2] == player and
                board[row + 3][col + 3] == player):
                return True

    # Check negative diagonal wins
    for row in range(3, rows):  # Rows where negative diagonals can start.
        for col in range(cols - 3):  # Columns where negative diagonals can start.
            # Check if 4 consecutive cells in a negative diagonal are occupied by the player.
            if (board[row][col] == player and
                board[row - 1][col + 1] == player and
                board[row - 2][col + 2] == player and
                board[row - 3][col + 3] == player):
                return True

    return False  # Return False if no winning condition is met.


def minmax_algorithm(board, depth, maximizing_player, player, heuristic=False):
    """
    Basic Minimax algorithm to calculate the best move for the current player.
    """
    valid_locations = get_valid_locations(board)  # Get all columns where a move is possible.

    # Base case: If depth is 0, no valid moves remain, or a win condition is met, evaluate the board.
    if depth == 0 or not valid_locations or check_win(board, player) or check_win(board, 'X' if player == 'O' else 'O'):
        return None, evaluation_score(board, player, heuristic=heuristic)

    opponent = 'X' if player == 'O' else 'O'  # Determine the opponent's marker.

    if maximizing_player:
        value = -math.inf  # Initialize the best value as negative infinity.
        best_column = random.choice(valid_locations)  # Pick a random valid column to start.
        for col in valid_locations:
            temp_board = [row[:] for row in board]  # Create a copy of the board to simulate the move.
            make_move(temp_board, col, player)  # Simulate the player's move in the column.
            # Recursively call Minimax for the minimizing player (opponent).
            new_score = minmax_algorithm(temp_board, depth-1, False, player, heuristic)[1]
            if new_score > value:  # If the new score is better, update the best value and column.
                value = new_score
                best_column = col
        return best_column, value  # Return the best column and its score.
    else:
        value = math.inf  # Initialize the best value as positive infinity.
        best_column = random.choice(valid_locations)  # Pick a random valid column to start.
        for col in valid_locations:
            temp_board = [row[:] for row in board]  # Create a copy of the board to simulate the move.
            make_move(temp_board, col, opponent)  # Simulate the opponent's move in the column.
            # Recursively call Minimax for the maximizing player.
            new_score = minmax_algorithm(temp_board, depth-1, True, player, heuristic)[1]
            if new_score < value:  # If the new score is better (lower), update the best value and column.
                value = new_score
                best_column = col
        return best_column, value  # Return the best column and its score.


def minmax_with_alphabeta_pruning(board, depth, alpha, beta, maximizing_player, player, heuristic=False):
    """
    Minimax algorithm with alpha-beta pruning to optimize the move selection process.
    """
    valid_locations = get_valid_locations(board)  # Get all columns where a move is possible.

    # Base case: If depth is 0, no valid moves remain, or a win condition is met, evaluate the board.
    if depth == 0 or not valid_locations or check_win(board, player) or check_win(board, 'X' if player == 'O' else 'O'):
        return None, evaluation_score(board, player, heuristic=heuristic)

    opponent = 'X' if player == 'O' else 'O'  # Determine the opponent's marker.

    if maximizing_player:
        value = -math.inf  # Initialize the best value as negative infinity.
        best_column = random.choice(valid_locations)  # Pick a random valid column to start.
        for col in valid_locations:
            temp_board = [row[:] for row in board]  # Create a copy of the board to simulate the move.
            make_move(temp_board, col, player)  # Simulate the player's move in the column.
            # Recursively call Minimax with alpha-beta pruning for the minimizing player.
            new_score = minmax_with_alphabeta_pruning(temp_board, depth-1, alpha, beta, False, player, heuristic)[1]
            if new_score > value:  # Update the best value and column if the new score is better.
                value = new_score
                best_column = col
            alpha = max(alpha, value)  # Update alpha to the maximum value encountered.
            if alpha >= beta:  # Beta cut-off: Stop searching further as it won't improve.
                break
        return best_column, value  # Return the best column and its score.
    else:
        value = math.inf  # Initialize the best value as positive infinity.
        best_column = random.choice(valid_locations)  # Pick a random valid column to start.
        for col in valid_locations:
            temp_board = [row[:] for row in board]  # Create a copy of the board to simulate the move.
            make_move(temp_board, col, opponent)  # Simulate the opponent's move in the column.
            # Recursively call Minimax with alpha-beta pruning for the maximizing player.
            new_score = minmax_with_alphabeta_pruning(temp_board, depth-1, alpha, beta, True, player, heuristic)[1]
            if new_score < value:  # Update the best value and column if the new score is better.
                value = new_score
                best_column = col
            beta = min(beta, value)  # Update beta to the minimum value encountered.
            if alpha >= beta:  # Alpha cut-off: Stop searching further as it won't improve.
                break
        return best_column, value  # Return the best column and its score.


def play_player(board, player):
    """
    Allow the player to make a move on the board.
    """
    column = user_input(board)  # Get the player's chosen column.
    make_move(board, column, player)  # Place the player's marker in the chosen column.
    return board  # Return the updated board.


def play_computer_minmax(board, player, use_pruning=False, heuristic=False):
    """
    Use the Minimax or Alpha-Beta pruning algorithm to determine the computer's move.
    """
    depth = 6  # Set the search depth to limit computation time.
    if use_pruning:
        # Determine the best column using Alpha-Beta pruning.
        col, minimax_score = minmax_with_alphabeta_pruning(board, depth, -math.inf, math.inf, True, player, heuristic)
    else:
        # Determine the best column using basic Minimax.
        col, minimax_score = minmax_algorithm(board, depth, True, player, heuristic)
    if col is not None:
        make_move(board, col, player)  # Make the move in the chosen column.
    else:
        print("No valid moves available for the computer!")  # Handle cases where no valid moves exist.


def main():
    board = [[" "]*7 for _ in range(6)]
    draw_board(board)
    
    player = 'X'  # Human is 'X', Computer is 'O'
    while is_moves_left(board):
        if player == 'X':
            play_player(board, player)
        else:
            # Uncomment the following line to use Minimax without pruning
            # play_computer_minmax(board, player, use_pruning=False, heuristic=False)
            
            # Uncomment the following line to use Minimax with Alpha-Beta pruning
            play_computer_minmax(board, player, use_pruning=True, heuristic=False)
            
            # Uncomment one of the below for the computer's turn
            # play_random_computer(board, player)  # For random computer moves
            # play_heuristic_computer(board, player)  # For heuristic computer moves
        
        draw_board(board)
        if check_win(board, player):
            print(f"Player {player} wins!")
            break
        
        player = 'O' if player == 'X' else 'X'
    
    if not any(check_win(board, p) for p in ['X', 'O']):
        print("Tie game")

if __name__ == '__main__':
    main()
