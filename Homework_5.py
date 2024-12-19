import numpy as np
import pandas as pd

from scipy.stats import poisson, binom
from tqdm import tqdm

np.random.seed(42)

def bellman_backup(state: tuple, policy: dict, value_func: np.array, reward: np.array, transition: np.array):
    action = policy[state]

    reward_value = reward[action].reshape(-1)[state_indices[state]]
    transition_vector = transition[action][state_indices[state]]

    return reward_value + DISCOUNT_FACTOR * np.dot(transition_vector, value_func)

# Changable Features
BAG_CAPACITY = 25
AVG_CUSTOMERS_PER_DAY = 400
CONVERGENCE_THRESHOLD = 0.01

MAX_INVENTORY = AVG_CUSTOMERS_PER_DAY // BAG_CAPACITY
MAX_ORDER = AVG_CUSTOMERS_PER_DAY // BAG_CAPACITY

# Constant Variables
DISCOUNT_FACTOR = 0.95
BAG_COST = 10
PROFIT_PER_CUP = 2
UNSATISFIED_PENALTY = 1
STALE_PROBABILITY = 0.25
NON_STANDARD_PROBABILITY = 0.05

states_space = [(inventory_size, num_customers) for inventory_size in range(MAX_INVENTORY + 1) for num_customers in range(AVG_CUSTOMERS_PER_DAY + 1)]
state_indices = {state: idx for idx, state in enumerate(states_space)}

state_matrix = pd.get_dummies(states_space, dtype=np.int64).values

actions = list(range(MAX_ORDER + 1))

action_reward_dict = {action: None for action in actions}

for action in tqdm(actions):
    reward_vector = []
    action_cost = action * BAG_COST

    num_bags_not_standard = np.random.binomial(action, NON_STANDARD_PROBABILITY)
    standard_coffee = action - num_bags_not_standard

    for inventory, customers in states_space:
        new_inventory = inventory + standard_coffee

        cups_sold = min(customers, new_inventory * BAG_CAPACITY)
        unsatisfied = max(0, customers - cups_sold)
        revenue_reward = (PROFIT_PER_CUP * cups_sold) - (UNSATISFIED_PENALTY * unsatisfied)

        reward_vector.append(revenue_reward - action_cost)

    action_reward_dict[action] = np.array(reward_vector)

customer_probability_matrix = np.zeros((AVG_CUSTOMERS_PER_DAY + 1, AVG_CUSTOMERS_PER_DAY + 1), dtype=np.float64)

for customers_today in range(AVG_CUSTOMERS_PER_DAY + 1):
    customer_probability_matrix[customers_today, :] = poisson.pmf(k=np.arange(AVG_CUSTOMERS_PER_DAY + 1), mu=(100 + (3 / 4) * customers_today))

action_transition_dict = {}

for action in actions:
    not_standard_probability = np.random.choice(binom.pmf(k=list(range(action + 1)), n=action, p=NON_STANDARD_PROBABILITY))
    action_transition_matrix = np.zeros(state_matrix.shape, dtype=np.float32)

    for inventory, customers in tqdm(states_space, desc=f"Building Transition Matrices for Action ({action}): "):
        current_state_index = state_indices[(inventory, customers)]

        new_inventory = inventory + action - np.random.binomial(action, NON_STANDARD_PROBABILITY)

        cups_sold = np.minimum(customers, new_inventory * BAG_CAPACITY)
        bags_used = np.minimum(new_inventory, cups_sold // BAG_CAPACITY)
        leftover = int(new_inventory - bags_used)

        stale_counts = np.arange(leftover + 1)
        stale_probs = binom.pmf(k=stale_counts, n=leftover, p=STALE_PROBABILITY)

        for next_day_customers in range(AVG_CUSTOMERS_PER_DAY + 1):
            customer_prob = customer_probability_matrix[customers, next_day_customers]

            for stale_bags, stale_prob in zip(stale_counts, stale_probs):
                usable_bags = int(leftover - stale_bags)
                next_state = (usable_bags, next_day_customers)

                if next_state in state_indices:
                    next_state_index = state_indices[next_state]
                    action_transition_matrix[current_state_index, next_state_index] += customer_prob * (1 - stale_prob) * (1 - not_standard_probability)

    row_sums = action_transition_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    action_transition_matrix /= row_sums

    action_transition_dict[action] = action_transition_matrix

policy = {state: 0 for state in states_space}
value_function = np.zeros(state_matrix.shape[0], dtype=np.float64)

delta = float('inf')
iteration = 0
pbar = tqdm(total=50, desc="Iterations (approx.)", leave=True)

while delta > CONVERGENCE_THRESHOLD:
    delta = 0
    Q_values = []
    new_value_function = np.zeros_like(value_function)

    for action in actions:
        reward_vector = action_reward_dict[action].reshape(-1)
        transition_matrix = action_transition_dict[action]

        action_value = reward_vector + DISCOUNT_FACTOR * np.dot(transition_matrix, value_function)
        Q_values.append(action_value)

    new_actions = np.argmax(Q_values, axis=0)

    for state_index, state in enumerate(states_space):
        policy[state] = new_actions[state_index]

    new_value_function = np.array([bellman_backup(
            state=state, policy=policy, value_func=value_function, reward=action_reward_dict, transition=action_transition_dict) for state in states_space])

    delta = np.max(np.abs(new_value_function - value_function))

    value_function = new_value_function.copy()
    iteration += 1

    pbar.set_description(f"Iteration {iteration}, Delta: {delta:.3f}")
    pbar.update(1)

policy = {state: 0 for state in states_space}
old_policy = None

new_value_function = np.zeros(state_matrix.shape[0], dtype=np.float64)
value_function = new_value_function - 0.1

value_iteration = 0
policy_iteration = 0

while policy != old_policy:
    delta = float('inf')

    while delta > CONVERGENCE_THRESHOLD:
        delta = 0
        new_value_function = np.array([bellman_backup(
            state=state, policy=policy, value_func=value_function, reward=action_reward_dict, transition=action_transition_dict) for state in states_space])

        delta = np.sum(new_value_function - value_function)

        value_function = new_value_function.copy()
        value_iteration += 1

    print(f"Number of Value Iterations: {value_iteration}")

    Q_values = []
    new_policy = {}

    for action in actions:
        reward_vector = action_reward_dict[action].reshape(-1)
        transition_matrix = action_transition_dict[action]

        action_value = reward_vector + DISCOUNT_FACTOR * np.dot(transition_matrix, value_function)
        Q_values.append(action_value)

    new_actions = np.argmax(Q_values, axis=0)
    new_value_function = np.max(Q_values, axis=0)

    for state_index, state in enumerate(states_space):
        new_policy[state] = new_actions[state_index]

    value_function = new_value_function.copy()

    old_policy = policy.copy()
    policy = new_policy

    policy_iteration += 1
    print(f"Number of Policy Iterations: {policy_iteration}")