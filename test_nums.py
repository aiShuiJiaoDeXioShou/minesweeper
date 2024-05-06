import subprocess
import re
import matplotlib.pyplot as plt

# 设置自定义游戏模式的基本配置
base_width = 18
base_height = 18
min_mines = 5  # 最小地雷数
max_mines = 20  # 最大地雷数
mine_increment = 3  # 地雷数递增步长
games_per_config = 10  # 每种配置运行的游戏次数

# 初始化数据收集变量
results = []

# 逐步增加地雷数并运行游戏脚本收集结果
for mines in range(min_mines, max_mines + 1, mine_increment):
    custom_mode = f'{base_width}x{base_height}x{mines}'
    command = [
        'python', 'play_traditional_minesweeper.py',  # 调整为你的脚本文件名
        '--game-mode', 'custom',
        '--custom-mode', custom_mode,
        '--number-of-games', str(games_per_config),
        '--output-mode', 'statistics-only'
    ]

    # 运行脚本并捕获输出
    process = subprocess.run(command, capture_output=True, text=True)
    output = process.stdout

    # 解析输出
    win_pattern = r'Win percentage: (\d+.\d+)%'
    duration_pattern = r'Average game duration: (\d+.\d+) seconds'
    win_percent = float(re.search(win_pattern, output).group(1))
    avg_duration = float(re.search(duration_pattern, output).group(1))

    # 保存结果
    results.append({
        'mines': mines,
        'win_percent': win_percent,
        'avg_duration': avg_duration
    })

# 绘制胜率和平均游戏时间图表
mines_list = [result['mines'] for result in results]
win_percents = [result['win_percent'] for result in results]
avg_durations = [result['avg_duration'] for result in results]

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(mines_list, win_percents, marker='o', linestyle='-', color='blue')
plt.title('Win Percentage as Mines Increase')
plt.xlabel('Number of Mines')
plt.ylabel('Win Percentage (%)')

plt.subplot(1, 2, 2)
plt.plot(mines_list, avg_durations, marker='o', linestyle='-', color='green')
plt.title('Average Game Duration as Mines Increase')
plt.xlabel('Number of Mines')
plt.ylabel('Duration (seconds)')

plt.tight_layout()
plt.show()
