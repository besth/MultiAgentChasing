import numpy as np
import anytree
from anytree import AnyNode as Node
from anytree import RenderTree

class CalculateScore:
    def __init__(self, c_init, c_base):
        self.c_init = c_init
        self.c_base = c_base
    
    def __call__(self, curr_node, child):
        parent_visit_count = curr_node.num_visited
        self_visit_count = child.num_visited
        mean_value = child.sum_value / self_visit_count
        action_prior = child.action_prior

        exploration_rate = np.log((1 + parent_visit_count + self.c_base) / self.c_base) + self.c_init
        # exploration_rate = 1.0
        u_score = exploration_rate * action_prior * np.sqrt(parent_visit_count) / float(1 + self_visit_count) 
        q_score = mean_value

        score = q_score + u_score
        return score


class SelectChild:
    def __init__(self, calculate_score):
        self.calculate_score = calculate_score

    def __call__(self, curr_node):
        scores = [self.calculate_score(curr_node, child) for child in curr_node.children]
        selected_child_index = np.argmax(scores)
        child = curr_node.children[selected_child_index]
        return child

class GetActionPrior:
    def __init__(self, action_space):
        self.action_space = action_space
        
    def __call__(self, curr_state):
        action_prior = {action: 1/len(self.action_space) for action in self.action_space}
        # action_prior = {-1: 0.48, 1: 0.52}
        return action_prior 

class Expand:
    def __init__(self, action_prior_func, transition_func, is_terminal):
        self.action_prior_func = action_prior_func
        self.transition_func = transition_func
        self.is_terminal = is_terminal

    def __call__(self, leaf_node):
        curr_state = list(leaf_node.id.values())[0]
        if not self.is_terminal(curr_state):
            leaf_node.is_expanded = True
            action_prior_distribution = self.action_prior_func(curr_state)
            
            for action, prior in action_prior_distribution.items():
                next_state = self.transition_func(curr_state, action)
                Node(parent=leaf_node, id={action: next_state}, num_visited=1, sum_value=0,
                    action_prior=prior, is_expanded=False)

        return leaf_node


class RollOut:
    def __init__(self, rollout_policy, max_rollout_step, transition_func, reward_func, is_terminal):
        self.transition_func = transition_func
        self.reward_func = reward_func
        self.max_rollout_step = max_rollout_step
        self.rollout_policy = rollout_policy
        self.is_terminal = is_terminal

    def __call__(self, leaf_node):
        curr_state = list(leaf_node.id.values())[0]
        sum_reward = 0
        for rollout_step in range(self.max_rollout_step):
            action = self.rollout_policy(curr_state)
            sum_reward += self.reward_func(curr_state, action)
            if self.is_terminal(curr_state):
                break

            next_state = self.transition_func(curr_state, action)
            curr_state = next_state

        return sum_reward


def backup(value, node_list):
    for node in node_list:
        node.sum_value += value
        node.num_visited += 1

class SelectNextRoot:
    def __init__(self, resetRoot):
        self.resetRoot = resetRoot
    def __call__(self, curr_root):
        children_mean_value = [child.sum_value / child.num_visited for child in curr_root.children]
        if children_mean_value[0] != children_mean_value[1]:
            selected_child_index = np.argmax(children_mean_value)
        else:
            selected_child_index = np.random.choice([0, 1])
        
        selected_child = curr_root.children[selected_child_index]
        next_state = list(selected_child.id.values())[0]
        next_root = self.resetRoot(next_state)
        return next_root


class ResetRoot:
    def __init__(self, actionSpace, transition, getActionPrior):
        self.actionSpace = actionSpace
        self.transition = transition
        self.getActionPrior = getActionPrior

    def __call__(self, rootState):
        rootAction = np.random.choice(self.actionSpace)
        initActionPrior = self.getActionPrior(rootState)
        initRootNode = Node(id={rootAction: rootState}, num_visited=1, sum_value=0, action_prior=initActionPrior[rootAction], is_expanded=True)

        # Create non-expanded children of the root.
        for action in self.actionSpace:
            nextState = self.transition(rootState, action)
            Node(parent=initRootNode, id={action: nextState}, num_visited=1, sum_value=0, action_prior=initActionPrior[action], is_expanded=False)

        return initRootNode

class MCTS:
    def __init__(self, num_simulation, selectChild, expand, rollout, backup, select_next_root):
        self.num_simulation = num_simulation
        self.select_child = selectChild
        self.expand = expand
        self.rollout = rollout
        self.backup = backup
        self.select_next_root = select_next_root 

    def __call__(self, curr_root):
        # curr_root_store = copy.deepcopy(curr_root)
        for explore_step in range(self.num_simulation):
            curr_node = curr_root
            node_path = list()

            while curr_node.is_expanded:
                next_node = self.select_child(curr_node)

                node_path.append(next_node)

                curr_node = next_node

            leaf_node = self.expand(curr_node)
            value = self.rollout(leaf_node)
            self.backup(value, node_path)
        
        next_root = self.select_next_root(curr_root)
        # print(RenderTree(next_root))
        return next_root

def main():
    pass

if __name__ == "__main__":
    main()
