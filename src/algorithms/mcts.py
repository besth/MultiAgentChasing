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
    def __init__(self, transition, exploration_rate):
        self.transition = transition
        self.exploration_rate = exploration_rate

    def __call__(self, curr_node, parent_visit_count):
        # calculate score for next node selection
        action_scores = [calculate_score(self.exploration_rate, parent_visit_count, child.num_visited,
                                         child.sum_value / child.num_visited, child.action_prior) for child in curr_node.children]
        action = np.argmax(action_scores)
        child = get_child_node(curr_node, action)

        curr_state = list(curr_node.id.values())[0]
        next_state = self.transition(curr_state, action)

        return child, action, next_state

class MCTS:
    def __init__(self):
        pass

    def __call__(self):
        pass


def main():
    num_action_space = 2
    exploration_rate = 1.0



    state = [[1, 1], [2, 2]]
    root = Node(id={1: state}, num_visited=1, sum_value=0, action_prior=1/num_action_space, is_expanded=True)
    root_child_1 = Node(parent=root, id={1: state}, num_visited=1, sum_value=5, action_prior=1/num_action_space, is_expanded=False)
    root_child_2 = Node(parent=root, id={0: state}, num_visited=1, sum_value=10, action_prior=1 / num_action_space, is_expanded=False)
    print(list(root.id.values())[0])






if __name__ == "__main__":
    main()