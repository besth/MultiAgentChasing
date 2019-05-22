import numpy as np
from anytree import AnyNode as Node
from anytree import RenderTree


class CalculateScore:
    def __init__(self, c_init, c_base):
        self.c_init = c_init
        self.c_base = c_base
    
    def __call__(self, curr_node, child):
        parent_visit_count = curr_node.num_visited
        self_visit_count = child.num_visited
        action_prior = child.action_prior

        if self_visit_count == 0:
            u_score = np.inf
            q_score = 0
        else:
            exploration_rate = np.log((1 + parent_visit_count + self.c_base) / self.c_base) + self.c_init
            u_score = exploration_rate * action_prior * np.sqrt(parent_visit_count) / float(1 + self_visit_count) 
            q_score = child.sum_value / self_visit_count

        score = q_score + u_score
        return score


class SelectChild:
    def __init__(self, calculate_score):
        self.calculate_score = calculate_score

    def __call__(self, curr_node):
        scores = [self.calculate_score(curr_node, child) for child in curr_node.children]
        maxIndex = np.argwhere(scores == np.max(scores)).flatten()
        selected_child_index = np.random.choice(maxIndex)
        selected_child = curr_node.children[selected_child_index]
        return selected_child


class GetActionPrior:
    def __init__(self, action_space):
        self.action_space = action_space
        
    def __call__(self, curr_state):
        action_prior = {action: 1/len(self.action_space) for action in self.action_space}
        return action_prior 


class Expand:
    def __init__(self, transition_func, is_terminal, initializeChildren):
        self.transition_func = transition_func
        self.is_terminal = is_terminal
        self.initializeChildren = initializeChildren

    def __call__(self, leaf_node):
        curr_state = list(leaf_node.id.values())[0]
        if not self.is_terminal(curr_state):
            leaf_node.is_expanded = True
            leaf_node = self.initializeChildren(leaf_node)

        return leaf_node


def rollout_heuristic(curr_pos, terminal_pos, weight=0.1):
    # curr_pos = curr_state[0][2:4]
    # terminal_pos = terminal_state[0][2:4]
    distance = np.sqrt(np.sum(np.square(curr_pos - terminal_pos)))

    reward = -weight * distance
    return reward


class RollOut:
    def __init__(self, rollout_policy, max_rollout_step, transition_func, reward_func, is_terminal, num_simulations, use_heuristic):
        self.transition_func = transition_func
        self.reward_func = reward_func
        self.max_rollout_step = max_rollout_step
        self.rollout_policy = rollout_policy
        self.is_terminal = is_terminal

        # Only used for testing. Delete later
        self.num_simulations = num_simulations
        self.use_heuristic = use_heuristic

    def __call__(self, leaf_node):
        reached_terminal = False
        # dis = 16
        # f = open("rollout_total_heuristic_{}_{}.txt".format(dis, self.num_simulations), "a+")
        # print(1, file=f)
        curr_state = list(leaf_node.id.values())[0]
        sum_reward = 0
        for rollout_step in range(self.max_rollout_step):
            action = self.rollout_policy(curr_state)
            sum_reward += self.reward_func(curr_state, action)
            if self.is_terminal(curr_state):
                # f = open("rollout_terminal_heuristic_{}_{}.txt".format(dis, self.num_simulations), "a+")
                # print(1, file=f)
                reached_terminal = True
                break

            next_state = self.transition_func(curr_state, action)
            curr_state = next_state

        if self.use_heuristic:
            # Heuristics based on distance between the last state of rollout and target state
            if not reached_terminal:
                terminal_pos = curr_state[1][2:4]
                curr_pos = curr_state[0][2:4]
                heuristic_reward = rollout_heuristic(curr_pos, terminal_pos)
                # print(curr_pos, terminal_pos, heuristic_reward)
                sum_reward += heuristic_reward

        return sum_reward


def backup(value, node_list):
    for node in node_list:
        node.sum_value += value
        node.num_visited += 1


class SelectNextRoot:
    def __init__(self, transition):
        self.transition = transition

    def __call__(self, curr_root):
        children_visit = [child.num_visited for child in curr_root.children]
        maxIndex = np.argwhere(children_visit == np.max(children_visit)).flatten()
        selected_child_index = np.random.choice(maxIndex)

        curr_state = list(curr_root.id.values())[0]
        action = list(curr_root.children[selected_child_index].id.keys())[0]
        # action = (-10, 0)
        next_state = self.transition(curr_state, action)

        next_root = Node(id={action: next_state}, num_visited=0, sum_value=0, is_expanded = False)
        return next_root


class InitializeChildren:
    def __init__(self, actionSpace, transition, getActionPrior):
        self.actionSpace = actionSpace
        self.transition = transition
        self.getActionPrior = getActionPrior

    def __call__(self, node):
        state = list(node.id.values())[0]
        initActionPrior = self.getActionPrior(state)

        for action in self.actionSpace:
            nextState = self.transition(state, action)
            # print("child initialized", state, action, nextState)
            Node(parent=node, id={action: nextState}, num_visited=0, sum_value=0, action_prior=initActionPrior[action], is_expanded=False)

        return node


class MCTS:
    def __init__(self, num_simulation, selectChild, expand, rollout, backup, select_next_root):
        self.num_simulation = num_simulation
        self.select_child = selectChild
        self.expand = expand
        self.rollout = rollout
        self.backup = backup
        self.select_next_root = select_next_root 

    def __call__(self, curr_root):
        curr_root = self.expand(curr_root)
        for explore_step in range(self.num_simulation):
            if explore_step % 100 == 0:
                print("simulation step:", explore_step, "out of", self.num_simulation)
            curr_node = curr_root
            node_path = [curr_node]

            while curr_node.is_expanded:
                next_node = self.select_child(curr_node)

                node_path.append(next_node)

                curr_node = next_node

            leaf_node = self.expand(curr_node)
            value = self.rollout(leaf_node)
            self.backup(value, node_path)

        # trajectory = sampleTrajectory(curr_root)
        # f_tree = open("tree.txt", "w+")
        # print(RenderTree(curr_root), file = f_tree)
        # f = open("traj.txt", "w+")
        # print(trajectory, file=f)

        next_root = self.select_next_root(curr_root)
        return next_root

def main():
    pass

if __name__ == "__main__":
    main()
