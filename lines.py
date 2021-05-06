import numpy as np


# y, x
tennis_court_model_points = np.asarray([
    [0, 0], # top left corner
    [0, 18],
    [0, 126],
    [0, 144],
    [72, 18],
    [72, 126],
    [240, 18],
    [240, 126],
    [312, 0], # top right corner
    [312, 18],
    [312, 126],
    [312, 144],
])

tennis_court_model_lines = [
    [0, 0],
    []
]