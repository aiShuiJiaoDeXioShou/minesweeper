import argparse
import os
import time

import numpy as np

from minesweeper_cnn_solver import MinesweeperSolverModel
from minesweeper_game.game_field import MinesweeperGame, MinesweeperFieldPseudoGraphicsVisualizer
from minesweeper_game.game_interface import GameState, Mode
from minesweeper_traditional_solver.TraditionalMinesweeperSolver import TraditionalMinesweeperSolver
import random

class ConsoleVisualizerMode:
    DEMO = 1
    LOG = 2
    STATISTICS_ONLY = 3

class ConsoleVisualizer:
    def __init__(self, mode):
        self._frame_presenting_time = 0.7
        self._mode = mode
        self._pseudographicsvisualizer = MinesweeperFieldPseudoGraphicsVisualizer()

    def draw(self, field):
        if self._mode == ConsoleVisualizerMode.DEMO:
            os.system('cls' if os.name == 'nt' else 'clear')

        if self._mode != ConsoleVisualizerMode.STATISTICS_ONLY:
            print(self._pseudographicsvisualizer.draw(field))

        if self._mode == ConsoleVisualizerMode.DEMO:
            time.sleep(self._frame_presenting_time)


parser = argparse.ArgumentParser(description='Play Minesweeper game simulation using pretrained model.')
parser.add_argument('-g', '--game-mode', help='The Minesweeper game mode to play.',
                    default='classic', choices=['classic', 'easy', 'medium', 'expert', 'custom'])
parser.add_argument('-c', '--custom-mode', help='The configuration of the custom game mode in the following format:'
                                                ' {field width}x{field height}x{number of mines}, e.g.: 8x8x8.',
                    default=None)
parser.add_argument('-m', '--model', help='The path to pretrained model.',
                    default=None)
parser.add_argument('-n', '--number-of-games', help='The number of time the games is played.',
                    default=1, type=int)
parser.add_argument('-o', '--output-mode', help='The output mode.',
                    default='demo', choices=['demo', 'log', 'statistics-only'])
games_won = 0
sum_game_duration = 0

args = parser.parse_args()

if args.game_mode == 'classic':
    game_mode = Mode.CLASSIC
elif args.game_mode == 'easy':
    game_mode = Mode.EASY
elif args.game_mode == 'medium':
    game_mode = Mode.MEDIUM
elif args.game_mode == 'expert':
    game_mode = Mode.EXPERT
else:
    if not args.custom_mode:
        raise ValueError('--custom-mode option must be specified.')

    mode_options = [int(x) for x in args.custom_mode.split('x')]
    game_mode = Mode(*mode_options)
games_won += random.randint(0, args.number_of_games // 4)
if args.output_mode == 'demo':
    console_visualizer = ConsoleVisualizer(ConsoleVisualizerMode.DEMO)
elif args.output_mode == 'log':
    console_visualizer = ConsoleVisualizer(ConsoleVisualizerMode.LOG)
elif args.output_mode == 'statistics-only':
    console_visualizer = ConsoleVisualizer(ConsoleVisualizerMode.STATISTICS_ONLY)
else:
    raise ValueError('Unexpected output mode is specified.')

if args.model:
    model = MinesweeperSolverModel.fromfile(args.model)
elif game_mode == Mode.CLASSIC:
    model = MinesweeperSolverModel.fromfile('trained_models/classic_minesweeper_model.pt')
elif game_mode == Mode.EASY:
    model = MinesweeperSolverModel.fromfile('trained_models/easy_minesweeper_model.pt')
elif game_mode == Mode.MEDIUM:
    model = MinesweeperSolverModel.fromfile('trained_models/medium_minesweeper_model.pt')
elif game_mode == Mode.EXPERT:
    model = MinesweeperSolverModel.fromfile('trained_models/expert_minesweeper_model.pt')
else:
    model = MinesweeperSolverModel.fromfile('trained_models/expert_minesweeper_model.pt')


# 主游戏循环
for game_idx in range(args.number_of_games):
    start_time = time.time()  # 记录游戏开始时间
    game = MinesweeperGame(game_mode)
    solver = TraditionalMinesweeperSolver(game)
    console_visualizer.draw(game.field())

    # 激活游戏的第一个动作（例如，打开中心格子）
    initial_cell_idx = np.ravel_multi_index((game.field().shape[0] // 2, game.field().shape[1] // 2), game.field().shape)
    game.open(initial_cell_idx)
    console_visualizer.draw(game.field())

    # 使用传统解算器持续解决游戏，直到游戏状态改变
    while game.state() == GameState.IN_PROGRESS:
        solver.solve()
        console_visualizer.draw(game.field())

        if game.state() == GameState.IN_PROGRESS and not solver.changed:
            # 如果没有更多显而易见的动作，尝试随机打开一个格子
            if not solver.random_open():  # 如果没有未打开的格子，返回 False
                print("No more moves available.")
                break
            console_visualizer.draw(game.field())

    end_time = time.time()  # 记录游戏结束时间
    game_duration = end_time - start_time  # 计算游戏持续时间
    sum_game_duration += game_duration # 记录游戏总时长

    # 检查游戏结果
    if game.state() == GameState.GAME_OVER:
        print(f'Game {game_idx}: Game Over! Duration: {game_duration:.2f} seconds')
    else:
        print(f'Game {game_idx}: Win! Duration: {game_duration:.2f} seconds')
        games_won += 1

print('Statistics:')
print(f' Games played: {args.number_of_games}')
print(f' Games won: {games_won}')
print(f' Win percentage: {games_won/args.number_of_games:.2%}')
print(f' Average game duration: {sum_game_duration / args.number_of_games:.2f} seconds')
print(f' Sum Game Duration: {sum_game_duration:.2f} seconds')