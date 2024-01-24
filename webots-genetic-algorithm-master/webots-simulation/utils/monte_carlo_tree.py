import math
import random

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

def mcts(root_state, budget):
    root_node = Node(state=root_state)

    for _ in range(budget):
        node = root_node
        while not node.state.is_terminal():
            if not node.children or random.uniform(0, 1) < 0.2:  # Exploration vs Exploitation
                node = expand(node)
            else:
                node = best_child(node)

        reward = rollout(node.state)
        backpropagate(node, reward)

    best_child_node = best_child(root_node, exploration_weight=0.0)  # Pure exploitation
    return best_child_node.state

def expand(node):
    actions = node.state.get_legal_actions()
    action = random.choice(actions)
    new_state = node.state.perform_action(action)
    new_node = Node(state=new_state, parent=node)
    node.children.append(new_node)
    return new_node

def best_child(node, exploration_weight=1.0):
    children_with_scores = [
        (child, child.value / (child.visits + 1e-6) + exploration_weight * math.sqrt(math.log(node.visits + 1) / (child.visits + 1e-6)))
        for child in node.children
    ]
    return max(children_with_scores, key=lambda x: x[1])[0]

def rollout(state):
    while not state.is_terminal():
        action = random.choice(state.get_legal_actions())
        state = state.perform_action(action)
    return state.get_reward()

def backpropagate(node, reward):
    while node is not None:
        node.visits += 1
        node.value += reward
        node = node.parent

# Example usage:
# Define your own State class with methods: is_terminal(), get_legal_actions(), perform_action(action), get_reward()
# initial_state = YourStateClass()
# final_state = mcts(initial_state, budget=1000)
