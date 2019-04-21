import numpy as np
import anytree
from anytree import AnyNode as Node
from anytree import RenderTree


def get_child_node(curr_node, action):
    return anytree.search.find(curr_node, lambda node: list(node.id.keys())[0] == action and node.parent == curr_node)


def calculate_score(exploration_rate, parent_visit_count, self_visit_count, mean_value, action_prior):
    q_score = mean_value
    exploration_score = action_prior * np.sqrt(parent_visit_count) / float(1 + self_visit_count)

    score = q_score + exploration_rate * exploration_score

    return score


class SelectChild:
    def __init__(self, transition_func, exploration_rate):
        self.transition_func = transition_func
        self.exploration_rate = exploration_rate

    def __call__(self, curr_node):
        # calculate score for next node selection
        action_scores = [calculate_score(self.exploration_rate, curr_node.num_visited, child.num_visited,
                                         child.sum_value / child.num_visited, child.action_prior) for child in curr_node.children]
        action = np.argmax(action_scores)
        child = get_child_node(curr_node, action)

        curr_state = list(curr_node.id.values())[0]
        next_state = self.transition_func(curr_state, action)

        return child, action, next_state


class Expand:
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def __call__(self, node_to_expand, action, state):
        if action is None or state is None:
            print("Action or state is None. Unable to expand.")
            exit()

        node_to_expand.id = {action: state}
        node_to_expand.is_expanded = True

        # Create empty children for each action
        for action_index in range(self.num_actions):
            Node(parent=node_to_expand, id={action_index: None}, num_visited=1, sum_value=0,
                 action_prior=1 / self.num_actions, is_expanded=False)

        return node_to_expand


class RollOut:
    def __init__(self, rollout_policy, max_rollout_step, transition_func, reward_func, is_terminal):
        self.transition_func = transition_func
        self.reward_func = reward_func
        self.max_rollout_step = max_rollout_step
        self.rollout_policy = rollout_policy
        self.is_terminal = is_terminal

    def __call__(self, curr_node):
        curr_state = list(curr_node.id.values())[0]
        sum_reward = 0
        for rollout_step in range(self.max_rollout_step):
            action = self.rollout_policy(curr_state)
            next_state = self.transition_func(curr_state, action)

            sum_reward += self.reward_func(curr_state, action)
            if self.is_terminal(curr_state):
                return sum_reward

            curr_state = next_state

        return sum_reward


def backup(value, node_list):
    for node in node_list:
        node.sum_value += value
        node.num_visited += 1


class MCTS:
    def __init__(self, exploration_batch_size, select_child, expand, rollout, backup):
        self.exploration_batch_size = exploration_batch_size
        self.select_child = select_child
        self.expand = expand
        self.rollout = rollout
        self.backup = backup

    def __call__(self, curr_root):
        curr_node = curr_root
        for explore_step in range(self.exploration_batch_size):
            node_list = list()

            # Placeholders for action and next_state
            action, next_state = None, None
            while curr_node.is_expanded:
                next_node, action, next_state = self.select_child(curr_node)

                # node list does not include current root
                node_list.append(next_node)

                curr_node = next_node

            curr_node = self.expand(curr_node, action, next_state)
            value = self.rollout(curr_node)
            self.backup(value, node_list)


def main():
    num_action_space = 2
    default_action_prior = 1 / num_action_space

    state = [[1, 1], [2, 2]]
    root = Node(id={1: state}, num_visited=1, sum_value=0, action_prior=default_action_prior, is_expanded=True)
    level1_1 = Node(parent=root, id={0: state}, num_visited=2, sum_value=5, action_prior=default_action_prior, is_expanded=False)
    level1_2 = Node(parent=root, id={1: state}, num_visited=3, sum_value=10, action_prior=default_action_prior, is_expanded=False)
    #
    level2_1 = Node(parent=level1_1, id={0: state}, num_visited=1, sum_value=3, action_prior=default_action_prior,
                    is_expanded=False)
    level2_2 = Node(parent=level1_1, id={1: state}, num_visited=1, sum_value=5, action_prior=default_action_prior,
                    is_expanded=False)
    # root.id = {2: [1, 2]}
    print(RenderTree(root))

    # # Test calculate score
    # parent_visit_number = 1
    # exploration_rate = 1.0
    # left_child_score = calculate_score(exploration_rate, parent_visit_count=level1_1.num_visited,
    #                                    self_visit_count=level2_1.num_visited,
    #                                    mean_value=level2_1.sum_value/level2_1.num_visited,
    #                                    action_prior=level2_1.action_prior)
    #
    # print("score for left child at level 2 is calculated to be", left_child_score, "it should be:", np.sqrt(2)/4 + 3)

    # # Test Expand
    # expand = Expand(num_action_space)
    # expanded_level1_1 = expand(level1_1, action=0, state=[[2,2],[3,3]])
    # print(RenderTree(root))

    # Test Rollout










if __name__ == "__main__":
    main()