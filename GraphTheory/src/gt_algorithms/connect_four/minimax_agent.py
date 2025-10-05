"""Connect Four game utilities with a documented minimax agent."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import numpy as np

ROWS = 6
COLUMNS = 7
CONNECT = 4
HUMAN_TOKEN = "X"
AI_TOKEN = "O"


@dataclass
class Move:
    """Representation of a move and its associated score."""

    column: int
    score: float


class ConnectFourBoard:
    """Mutable Connect Four board with helper methods for search."""

    def __init__(self) -> None:
        self.grid = np.full((ROWS, COLUMNS), " ", dtype=str)

    def copy(self) -> "ConnectFourBoard":
        """Return a deep copy of the board."""

        new_board = ConnectFourBoard()
        new_board.grid = self.grid.copy()
        return new_board

    def drop_disc(self, column: int, token: str) -> bool:
        """Insert a token in the selected column.

        Args:
            column: Column index where the token should be dropped.
            token: Player token (``"X"`` or ``"O"``).

        Returns:
            True if the move was applied, False if the column is full.
        """

        for row in range(ROWS - 1, -1, -1):
            if self.grid[row, column] == " ":
                self.grid[row, column] = token
                return True
        return False

    def available_columns(self) -> List[int]:
        """Return columns that can accept a new disc."""

        return [col for col in range(COLUMNS) if self.grid[0, col] == " "]

    def check_winner(self, token: str) -> bool:
        """Check whether ``token`` has achieved four consecutive discs."""

        # Horizontal
        for row in range(ROWS):
            for col in range(COLUMNS - CONNECT + 1):
                if np.all(self.grid[row, col : col + CONNECT] == token):
                    return True

        # Vertical
        for col in range(COLUMNS):
            column_vals = self.grid[:, col]
            for row in range(ROWS - CONNECT + 1):
                if np.all(column_vals[row : row + CONNECT] == token):
                    return True

        # Positive diagonal
        for row in range(ROWS - CONNECT + 1):
            for col in range(COLUMNS - CONNECT + 1):
                if all(self.grid[row + offset, col + offset] == token for offset in range(CONNECT)):
                    return True

        # Negative diagonal
        for row in range(CONNECT - 1, ROWS):
            for col in range(COLUMNS - CONNECT + 1):
                if all(self.grid[row - offset, col + offset] == token for offset in range(CONNECT)):
                    return True

        return False

    def is_full(self) -> bool:
        """Return True if the board does not accept more moves."""

        return not np.any(self.grid == " ")


def score_position(board: ConnectFourBoard, token: str) -> float:
    """Heuristic evaluation that rewards good alignment and centre control."""

    opponent = HUMAN_TOKEN if token == AI_TOKEN else AI_TOKEN
    if board.check_winner(token):
        return 1_000.0
    if board.check_winner(opponent):
        return -1_000.0

    centre_column = board.grid[:, COLUMNS // 2]
    score = float(np.count_nonzero(centre_column == token) * 3)

    def windows() -> Iterable[np.ndarray]:
        for row in range(ROWS):
            for col in range(COLUMNS - CONNECT + 1):
                yield board.grid[row, col : col + CONNECT]
        for col in range(COLUMNS):
            column_vals = board.grid[:, col]
            for row in range(ROWS - CONNECT + 1):
                yield column_vals[row : row + CONNECT]
        for row in range(ROWS - CONNECT + 1):
            for col in range(COLUMNS - CONNECT + 1):
                yield np.array([board.grid[row + offset, col + offset] for offset in range(CONNECT)])
        for row in range(CONNECT - 1, ROWS):
            for col in range(COLUMNS - CONNECT + 1):
                yield np.array([board.grid[row - offset, col + offset] for offset in range(CONNECT)])

    for window in windows():
        score += evaluate_window(window, token)
    return score


def evaluate_window(window: np.ndarray, token: str) -> float:
    """Assign a score to a four-cell window."""

    opponent = HUMAN_TOKEN if token == AI_TOKEN else AI_TOKEN
    tokens = np.count_nonzero(window == token)
    empties = np.count_nonzero(window == " ")
    opponent_tokens = np.count_nonzero(window == opponent)

    if tokens == 4:
        return 100.0
    if tokens == 3 and empties == 1:
        return 5.0
    if tokens == 2 and empties == 2:
        return 2.0
    if opponent_tokens == 3 and empties == 1:
        return -4.0
    return 0.0


def minimax(
    board: ConnectFourBoard,
    depth: int,
    maximizing_player: bool,
    token: str = AI_TOKEN,
    alpha: float = -math.inf,
    beta: float = math.inf,
) -> Move:
    """Run minimax with alpha-beta pruning and return the best move.

    Args:
        board: Current game position.
        depth: Remaining search depth.
        maximizing_player: True when it is the AI's turn to move.
        token: Token representing the AI.
        alpha: Alpha bound for pruning.
        beta: Beta bound for pruning.

    Returns:
        Move containing the chosen column and its heuristic score.
    """

    opponent = HUMAN_TOKEN if token == AI_TOKEN else AI_TOKEN
    if depth == 0 or board.check_winner(token) or board.check_winner(opponent) or board.is_full():
        return Move(column=-1, score=score_position(board, token))

    if maximizing_player:
        best = Move(column=-1, score=-math.inf)
        for column in board.available_columns():
            child = board.copy()
            child.drop_disc(column, token)
            evaluation = minimax(child, depth - 1, False, token, alpha, beta)
            if evaluation.score > best.score:
                best = Move(column=column, score=evaluation.score)
            alpha = max(alpha, best.score)
            if alpha >= beta:
                break
        return best

    best = Move(column=-1, score=math.inf)
    for column in board.available_columns():
        child = board.copy()
        child.drop_disc(column, opponent)
        evaluation = minimax(child, depth - 1, True, token, alpha, beta)
        if evaluation.score < best.score:
            best = Move(column=column, score=evaluation.score)
        beta = min(beta, best.score)
        if alpha >= beta:
            break
    return best


__all__ = ["ConnectFourBoard", "Move", "evaluate_window", "minimax", "score_position"]
