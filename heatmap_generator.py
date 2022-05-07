import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

human = ["0", "3"]
objects = ['1', '2']
score = np.array([[0, 0.0681],[0, 0.1616]])

fig, ax = plt.subplots()
im = ax.imshow(score, cmap=cm.Blues)

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(objects)))
ax.set_xticklabels(objects)
ax.set_yticks(np.arange(len(human)))
ax.set_yticklabels(human)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(human)):
    for j in range(len(objects)):
        if score[i,j] < 0.01:
            text = ax.text(j, i, score[i, j],
                       ha="center", va="center", color="grey")
        else:
            text = ax.text(j, i, score[i, j],
                       ha="center", va="center", color="white")

ax.set_title("interaction scores (row - human, column - objects)")
fig.tight_layout()
plt.show()