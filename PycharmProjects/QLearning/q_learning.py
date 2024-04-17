# Artificial Intelligence A-Z

# A Q-Learning Implementation for Process Optimization

# Importing the libraries
import numpy as np

from q_learn_environment import QLearnEnvironment

# Setting the parameters gamma and alpha for the Q-Learning
gamma = 0.75  # discount factor
alpha = 0.9  # learning rate

# PART 1 - DEFINING THE ENVIRONMENT
state = QLearnEnvironment()

# PART 2 - BUILDING THE AI SOLUTION WITH Q-LEARNING

# Making a mapping from the states to the locations
state_to_location = {state: location for location, state in state.location_to_state.items()}


# Making a function that returns the shortest route from a starting to ending location
def route(starting_location, ending_location):
    R_new = np.copy(state.R)
    ending_state = state.location_to_state[ending_location]
    R_new[ending_state, ending_state] = 1000
    Q = np.array(np.zeros([12, 12]))
    for i in range(1000):
        current_state = np.random.randint(0, 12)
        playable_actions = []
        for j in range(12):
            if R_new[current_state, j] > 0:
                playable_actions.append(j)
        next_state = np.random.choice(playable_actions)
        TD = R_new[current_state, next_state] + gamma * Q[next_state, np.argmax(Q[next_state,])] - Q[current_state, next_state]

        if current_state == 0:
            print(Q[current_state, next_state], R_new[current_state, next_state] + gamma * Q[next_state, np.argmax(Q[next_state,])], TD)

        Q[current_state, next_state] = Q[current_state, next_state] + alpha * TD

    route = [starting_location]
    next_location = starting_location
    while next_location != ending_location:
        starting_state = state.location_to_state[starting_location]
        next_state = np.argmax(Q[starting_state,])
        next_location = state_to_location[next_state]
        route.append(next_location)
        starting_location = next_location

    #    print("Q-Values:")
    #    print(Q.astype(int))

    return route


# PART 3 - GOING INTO PRODUCTION

# Making the final function that returns the optimal route
def best_route(starting_location, intermediary_location, ending_location):
    return route(starting_location, intermediary_location) + route(intermediary_location, ending_location)[1:]


# Printing the final route
print('First Route:')
firstRoute = route('E', 'G')
print(firstRoute)

print('Route:')
route = best_route('E', 'K', 'G')

print(route)
