import numpy as np


# y, x
tennis_court_model_points = np.asarray([
    [0, 0], # top left corner
    [0, 4.5],
    [0, 31.5],
    [0, 36],
    [18, 4.5],
    [18, 31.5],
    [60, 4.5],
    [60, 31.5],
    [78, 0], # top right corner
    [78, 4.5],
    [78, 31.5],
    [78, 36],
])

tennis_court_model_lines = [
    [0, 0],
    []
]