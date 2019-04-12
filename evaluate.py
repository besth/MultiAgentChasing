
def sampleTrajectory(actor):
    return []

def accumulateRewards(trajectory):
    return 0

def approximatePolicy(state, actor_model):
    return None



def evaluate(model, num_trajectories):
    # models: {param_1: (actor_1, critic_1), ..., param_n: (actor_n, critic_n)}
    actor_model, critic_model = model
    mean_episode_rewards = list()

    actor = lambda state: approximatePolicy(state, actor_model)
    trajectories = [sampleTrajectory(actor) for _ in range(num_trajectories)]
    episode_rewards = [accumulateRewards(trajectory) for trajectory in trajectories]
    mean_episode_rewards.append(np.mean(episode_rewards))

    return mean_episode_rewards

    

