import unittest
import numpy as np
from ddt import ddt, data, unpack
from anytree import AnyNode as Node
from algorithms.mcts import calculate_score, SelectChild, Expand, RollOut, backup
from simple1DEnv import TransitionFunction, RewardFunction


@ddt
class TestMCTS(unittest.TestCase):
    def setUp(self):
        # Env param
        bound_low = 0
        bound_high = 7
        self.transition = TransitionFunction(bound_low, bound_high)

        step_penalty = -1
        catch_reward = 1
        target_state = bound_high

        self.reward = RewardFunction(step_penalty, catch_reward, target_state)

        self.num_action_space = 2
        self.exploration_rate = 1.0
        self.selectChild = SelectChild(self.exploration_rate)
        self.default_action_prior = 1 / self.num_action_space

        init_state = 3
        level1_0_state = self.transition(init_state, action=0)
        # print("level 1 state", level1_0_state)
        level1_1_state = self.transition(init_state, action=1)

        self.root = Node(id={1: init_state}, num_visited=1, sum_value=0, action_prior=self.default_action_prior, is_expanded=True)
        self.level1_0 = Node(parent=self.root, id={0: level1_0_state}, num_visited=2, sum_value=5, action_prior=self.default_action_prior, is_expanded=False)
        self.level1_1 = Node(parent=self.root, id={1: level1_1_state}, num_visited=3, sum_value=10, action_prior=self.default_action_prior, is_expanded=False)

        self.expand = Expand(self.num_action_space, self.transition)

    @data()
    @unpack
    def testCalculateScore(self, exploration_rate, parent_visit_number, self_visit_number, sum_value, action_prior, true_score):
        exploration_rate = 1.0
        parent_visit_number = 2
        self_visit_number = 1
        mean_value = 3 / self_visit_number
        action_prior = 0.5

        score = calculate_score(exploration_rate, parent_visit_number, self_visit_number, mean_value, action_prior)
        gt_score = np.sqrt(2) / 4 + 3
        self.assertEqual(score, gt_score)

    @data()
    @unpack
    def testSelectChild(self, firstChildVisited, firstChildSumValue, secondChildVisited, secondChildSumValue):
        child = self.selectChild(self.root)
        child_id_action = list(child.id.keys())[0]
        gt_score_0 = 5 / 2 + 1.0 * 0.5 * 1 / (1 + 2)
        gt_score_1 = 10 / 3 + 1.0 * 0.5 * 1 / (1 + 3)
        gt_action = np.argmax([gt_score_0, gt_score_1])

        self.assertEqual(gt_action, child_id_action)

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

    def testRollout(self):
        pass



if __name__ == "__main__":
    unittest.main()


