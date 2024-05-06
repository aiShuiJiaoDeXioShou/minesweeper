import numpy as np
import random

from minesweeper_game.game_interface import CellState

class TraditionalMinesweeperSolver:
    def __init__(self, game):
        self.game = game
        self.changed = False  # 跟踪是否有变化
        self.marked_mines = set()  # 存储被标记为雷的格子的索引

    def solve(self):
        self.changed = False
        for row in range(self.game.mode().height()):
            for col in range(self.game.mode().width()):
                current_cell = self.game.field()[row, col]
                if current_cell in range(1, 9):
                    num_flagged = sum(
                        self.game.field()[r, c] == CellState.MINE for r, c in self.get_neighbors(row, col))
                    num_closed = sum(
                        self.game.field()[r, c] == CellState.CLOSED for r, c in self.get_neighbors(row, col))
                    clue = current_cell

                    # print(f"Processing cell ({row}, {col}): clue={clue}, flagged={num_flagged}, closed={num_closed}")

                    if clue == num_flagged:
                        for r, c in self.get_neighbors(row, col):
                            if self.game.field()[r, c] == CellState.CLOSED:
                                idx = np.ravel_multi_index((r, c), self.game.field().shape)
                                self.game.open(idx)
                                # print(f"Opened cell ({r}, {c}) based on clue match.")
                                self.changed = True
                    elif clue == num_closed + num_flagged:
                        for r, c in self.get_neighbors(row, col):
                            if self.game.field()[r, c] == CellState.CLOSED:
                                mine_idx = np.ravel_multi_index((r, c), self.game.field().shape)
                                self.marked_mines.add(mine_idx)
                                # print(f"Marked cell ({r}, {c}) as mine.")

    def get_neighbors(self, row, col):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                r, c = row + dr, col + dc
                if 0 <= r < self.game.mode().height() and 0 <= c < self.game.mode().width():
                    yield (r, c)

    def random_open(self):
        """ 随机选择一个未打开且未标记为雷的格子进行打开 """
        closed_indices = [(i, j) for i in range(self.game.mode().height()) for j in range(self.game.mode().width())
                          if self.game.field()[i, j] == CellState.CLOSED and np.ravel_multi_index((i, j), self.game.field().shape) not in self.marked_mines]
        if closed_indices:
            row, col = random.choice(closed_indices)
            idx = np.ravel_multi_index((row, col), self.game.field().shape)
            self.game.open(idx)
            return True
        return False
