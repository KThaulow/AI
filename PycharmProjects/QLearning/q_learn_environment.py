import numpy as np


class QLearnEnvironment:
    # Defining the states
    location_to_state = {'A': 0,
                         'B': 1,
                         'C': 2,
                         'D': 3,
                         'E': 4,
                         'F': 5,
                         'G': 6,
                         'H': 7,
                         'I': 8,
                         'J': 9,
                         'K': 10,
                         'L': 11}

    # Defining the actions
    actions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    # Defining the rewards
    # Rows are locations and columns are actions
    # E.g. row 0 is state A.
    # The columns define that it can only transition to index 1, which is B

    R = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]])
