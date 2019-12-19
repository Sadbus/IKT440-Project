from matplotlib.patches import Circle
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 7))
ax.set_facecolor('blue')

# Format: rows starting from the bottom, board[0] = row 1, board[1] = row 2, etc.
board = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0]
]

color = ["White", "Yellow", "Red"]

for row in range(6):
    y = row * 1/7 + 1/14

    for col in range(7):
        x = col * 1/7 + 1/14
        circle = Circle((x, y), 1 / 21, color=color[board[row][col]])
        ax.add_artist(circle)

ax.set_xticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
ax.set_xticks([1/14, 3/14, 5/14, 7/14, 9/14, 11/14, 13/14])

ax.set_yticklabels([1, 2, 3, 4, 5, 6])
ax.set_yticks([1/14, 3/14, 5/14, 7/14, 9/14, 11/14])

ax.axes.set_aspect('equal')
plt.ylim(0, 6/7)
plt.savefig('connect_four.png', bbox_inches='tight', pad_inches=0.1)
plt.show()
