import numpy as np

class Node:
    def __init__(self, parent=None, action=None):
        self.parent = parent
        self.children = {}
        self.action = action
        self.visits = 0
        self.total_reward = 0

def select(node, exploration_weight=1.0):
    """Select a child node based on UCB1 exploration-exploitation strategy."""
    if not node.children:
        return node

    ucb_values = {
        child_action: child.total_reward / (child.visits + 1e-6) +
        exploration_weight * np.sqrt(np.log(node.visits) / (child.visits + 1e-6))
        for child_action, child in node.children.items()
    }

    selected_action = max(ucb_values, key=ucb_values.get)
    return node.children[selected_action]

def expand(node, action):
    """Expand the tree by adding a new child node."""
    new_node = Node(parent=node, action=action)
    node.children[action] = new_node
    return new_node

def simulate(node):
    """Simulate a random trajectory (rollout) from the given node."""
    total_reward = 0
    current_node = node

    while current_node.children:
        selected_action = max(current_node.children, key=lambda a: current_node.children[a].total_reward)
        current_node = current_node.children[selected_action]
        total_reward += np.random.normal()  # Simulate a reward (in a real scenario, this would be the actual outcome)

    return total_reward

def backpropagate(node, reward):
    """Backpropagate the reward information up the tree."""
    current_node = node

    while current_node is not None:
        current_node.visits += 1
        current_node.total_reward += reward
        current_node = current_node.parent

def mcts(root_node, num_iterations=1000, exploration_weight=1.0):
    """Monte Carlo Tree Search."""
    for _ in range(num_iterations):
        selected_node = select(root_node, exploration_weight)
        action = np.random.choice(range(10))  # In this example, there are 10 possible actions (slot machines)
        
        if action not in selected_node.children:
            new_node = expand(selected_node, action)
            reward = simulate(new_node)
            backpropagate(new_node, reward)
        else:
            existing_node = selected_node.children[action]
            reward = simulate(existing_node)
            backpropagate(existing_node, reward)

    best_action = max(root_node.children, key=lambda a: root_node.children[a].total_reward / (root_node.children[a].visits + 1e-6))
    return best_action

# Example Usage:
root = Node()
best_action = mcts(root, num_iterations=1000, exploration_weight=1.0)
print("Best Action:", best_action)
