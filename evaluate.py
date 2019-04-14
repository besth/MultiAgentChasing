import numpy as np
class Evaluate:
    def __init__(self, num_trajectories, approximatePolicy, sampleTrajectory, accumulateRewards, rewardFunction):
        self.num_trajectories = num_trajectories
        self.approximatePolicy = approximatePolicy
        self.sampleTrajectory = sampleTrajectory
        self.accumulateRewards = accumulateRewards
        self.rewardFunction = rewardFunction

    def __call__(self, actor_model, critic_model):
        # actor_model, critic_model = model
        mean_episode_rewards = list()

        actor = lambda state: self.approximatePolicy(state, actor_model)
        trajectories = [self.sampleTrajectory(actor) for _ in range(self.num_trajectories)]
        episode_rewards = [self.accumulateRewards([self.rewardFunction(state, action) for state, action in trajectory]) for trajectory in trajectories]
        mean_episode_rewards.append(np.mean(episode_rewards))

        return mean_episode_rewards


    

