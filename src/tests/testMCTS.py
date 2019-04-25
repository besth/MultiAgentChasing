import sys
sys.path.append('..')
import unittest
import numpy as np
from ddt import ddt, data, unpack
from anytree import AnyNode as Node
# from MultiAgent-Chasing
from algorithms.mcts import CalculateScore, SelectChild, Expand, RollOut, backup
from simple1DEnv import TransitionFunction, RewardFunction, Terminal


@ddt
class TestMCTS(unittest.TestCase):
    def setUp(self):
        # Env param
        bound_low = 0
        bound_high = 7
        self.transition = TransitionFunction(bound_low, bound_high)

        self.action_space = [-1, 1]
        self.num_action_space = len(self.action_space)

        step_penalty = -1
        catch_reward = 1
        self.target_state = bound_high
        self.isTerminal = Terminal(self.target_state)

        # self.reward = RewardFunction(step_penalty, catch_reward, self.target_state)

        self.c_init = 0
        self.c_base = 1
        self.calculateScore = CalculateScore(self.c_init, self.c_base)

        init_state = 3
        level1_0_state = self.transition(init_state, action=0)
        # print("level 1 state", level1_0_state)
        level1_1_state = self.transition(init_state, action=1)
        self.default_action_prior = 0.5

        self.root = Node(id={1: init_state}, num_visited=1, sum_value=0, action_prior=self.default_action_prior, is_expanded=True)
        self.level1_0 = Node(parent=self.root, id={0: level1_0_state}, num_visited=2, sum_value=5, action_prior=self.default_action_prior, is_expanded=False)
        self.level1_1 = Node(parent=self.root, id={1: level1_1_state}, num_visited=3, sum_value=10, action_prior=self.default_action_prior, is_expanded=False)

        self.expand = Expand(self.num_action_space, self.transition, self.isTerminal)

    @data((0, 1, 0, 1, 0))
    @unpack
    def testCalculateScore(self, parent_visit_number, self_visit_number, sum_value, action_prior, groundtruth_score):
        curr_node = Node(num_visited = parent_visit_number)
        child = Node(num_visited = self_visit_number, sum_value = sum_value, action_prior = action_prior)
        score = self.calculateScore(curr_node, child)
        self.assertEqual(score, groundtruth_score)


    @unittest.skip  
    @data()
    @unpack
    def testSelectChild(self, firstChildVisited, firstChildSumValue, secondChildVisited, secondChildSumValue):
        child = self.selectChild(self.root)
        child_id_action = list(child.id.keys())[0]
        gt_score_0 = 5 / 2 + 1.0 * 0.5 * 1 / (1 + 2)
        gt_score_1 = 10 / 3 + 1.0 * 0.5 * 1 / (1 + 3)
        gt_action = np.argmax([gt_score_0, gt_score_1])

        self.assertEqual(gt_action, child_id_action)
       
    @unittest.skip  
    def testExpand(self):
        # test whether children have been created with the correct values.
        curr_node = self.level1_0

        new_curr_node = self.expand(self.level1_0)

        children = new_curr_node.children

        child_0 = children[0]
        child_1 = children[1]
        child_0_cal_state = list(child_0.id.values())[0]
        self.assertEqual(child_0_cal_state, 1)

        child_1_cal_state = list(child_1.id.values())[0]
        self.assertEqual(child_1_cal_state, 3)


    @unittest.skip 
    @data((4, 3, 0.125), (3, 4, 0.25))
    @unpack
    def testRollout(self, max_rollout_step, init_state, gt_sum_value):
        max_iteration = 100000

        target_state = 6
        isTerminal = Terminal(target_state)

        catch_reward = 1
        step_penalty = 0
        reward_func = RewardFunction(step_penalty, catch_reward, isTerminal)

        rollout_policy = lambda state: np.random.choice(self.action_space)
        leaf_node = Node(id={1: init_state}, num_visited=1, sum_value=0, action_prior=self.default_action_prior, is_expanded=True)
        rollout = RollOut(rollout_policy, max_rollout_step, self.transition, reward_func, isTerminal)
        stored_reward = []
        for curr_iter in range(max_iteration):
            # print(curr_iter, rollout(leaf_node))
            stored_reward.append(rollout(leaf_node))
        
        calc_sum_value = np.mean(stored_reward)
        # print(stored_reward)
        self.assertAlmostEqual(gt_sum_value, calc_sum_value, places=2)



        # init_state = self.rollout_init_state



if __name__ == "__main__":
    unittest.main()


