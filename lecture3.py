import numpy as np
import matplotlib.pyplot as plt

la = np.linalg

words = ["I", "like", "enjoy", "deep","learning", "NLP", 'flying', "."]

X = np.array([[0, 2, 1, 0, 0, 0, 0, 0],
              [2, 0, 0, 1, 0, 1, 0, 0],
              [1, 0, 0, 0, 0, 0, 1, 0],
              [0, 1, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 1],
              [0, 1, 0, 0, 0, 0, 0, 1],
              [0, 0, 1, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, 1, 1, 1, 0]
             ])

U, S, Vh = la.svd(X, full_matrices=False)
print(U)

# we take the first two colums of U and plot them
for i in range(len(words)):
    plt.text(U[i, 0], U[i, 1], words[i])
    
# there is a bug here. fix it    
plt.show()