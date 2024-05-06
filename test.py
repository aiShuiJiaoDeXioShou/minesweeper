import subprocess
import re
import matplotlib.pyplot as plt

# 定义游戏配置
configs = [
    {'mode': 'easy', 'games': 10},
    {'mode': 'medium', 'games': 10},
    {'mode': 'expert', 'games': 10},
    {'mode': 'custom', 'custom_mode': '8x8x8', 'games': 10},
    {'mode': 'custom', 'custom_mode': '16x16x40', 'games': 10}
]

# 初始化数据收集变量
results = []

# 运行游戏脚本并收集结果
for config in configs:
    mode = config['mode']
    games = config['games']
    custom_mode = config.get('custom_mode', '')
    command = [
        'python', 'play_minesweeper.py',  # 调整为你的脚本文件名
        '--game-mode', mode,
        '--number-of-games', str(games),
        '--output-mode', 'statistics-only'
    ]
    if custom_mode:
        command.extend(['--custom-mode', custom_mode])

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
        'mode': mode,
        'win_percent': win_percent,
        'avg_duration': avg_duration
    })

# 绘制胜率和平均游戏时间图表
modes = [result['mode'] for result in results]
win_percents = [result['win_percent'] for result in results]
avg_durations = [result['avg_duration'] for result in results]

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(modes, win_percents, color='blue')
plt.title('Win Percentage by Game Mode')
plt.ylabel('Win Percentage (%)')

plt.subplot(1, 2, 2)
plt.bar(modes, avg_durations, color='green')
plt.title('Average Game Duration by Game Mode')
plt.ylabel('Duration (seconds)')

plt.tight_layout()
plt.show()
