import numpy as np


# x, y
tennis_court_model_points = np.asarray([
    [0,0],         # 0 # top left corner
    [18,0],        # 1
    [126,0],       # 2
    [144,0],       # 3
    [18,72],       # 4
    [126,72],      # 5
    [18,240],      # 6
    [126,240],     # 7
    [0,312],       # 8 top right corner
    [18,312],      # 9
    [126,312],     # 10
    [144,312],     # 11
    [72,72],       # 12
    [72,240],      # 13
])

tennis_court_model_lines = np.asarray([
    [0, 3],
    [0, 8],
    [1, 9],
    [4, 5],
    [6, 7],
    [12, 13],
    [3, 11],
    [2, 10],
    [8, 11],
])