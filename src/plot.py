import os
import matplotlib.pyplot as plt


num_sim = [100, 200, 500, 1000, 1500, 2000]
distances = [1, 2, 4, 8, 12, 16]

# rollout hit target rate
rollout_reach_target_prob_no_heuristic_distance = [0.954047619047619, 0.9104166666666667, 0.8246078431372549, 0.6477848101265823, 0.46820754716981133, 0.41257692307692306]
rollout_reach_target_prob_no_heuristic_simulation_distance16 = [0.3172, 0.4126, 0.5807, 0.6829]
rollout_reach_target_prob_with_heuristic_distance = [0.9712727272727273, 0.9503888888888888, 0.9300851063829787, 0.8400821917808219, 0.7460222222222223, 0.6762222222222222]

# rollout hit target rate at initial state only
rollout_in_init_state_reach_target_prob_no_heuristic_simulation_distance16 = [0.013, 0.0137, 0.01648, 0.0233, 0.021213333333333334, 0.03697]
rollout_in_init_state_reach_target_prob_with_heuristic_simulation_distance16 = [0.0294, 0.0356, 0.08168, 0.34642, 0.37722666666666665, 0.54976]
# first step optimality
mean_first_step_distance_to_optimal_no_heuristic = [0.1751776429621799, 0.15343741661206808, 0.10593741661206812, 0.1198498445057071, 0.07234984450570713, 0.01808746112642678]
mean_first_step_distance_to_optimal_with_heuristic_test50 = [0.1520751831034078, 0.08320232118156319, 0.08681981340684855, 0.08991983560283538, 0.05064489115399499, 0.05787987560456569]
mean_first_step_distance_to_optimal_no_heuristic_test50 = [0.20149913343136788, 0.20104943879628082, 0.140337871969986, 0.11110731892524901, 0.09631981340684856, 0.08108261325336098]

# mean episode length
mean_episode_length_no_heuristic_dis16_test10 = [13.5, 12.2, 10.2, 9.8, 10.1, 9.7]
mean_episode_length_with_heuristic_dis16_test10 = [12.8, 10.4, 10.3, 10.1, 9.8, 10.3]
mean_episode_length_no_heuristic_dis16_test50 = [15.02, 11.76, 10.3, 10.24, 9.78, 9.58]
mean_episode_length_with_heuristic_dis16_test50 = [13.74, 10.98, 10.12, 10.56, 10.24, 10.38]

plt.figure()
# plt.plot(distance, prob_heuristic)

plt.plot(num_sim, rollout_in_init_state_reach_target_prob_no_heuristic_simulation_distance16, label="no heuristic")
plt.plot(num_sim, rollout_in_init_state_reach_target_prob_with_heuristic_simulation_distance16, label="with heuristic")
plt.legend()
plt.title("Effect of rollout heuristic on initial state rollout hit rate")
# plt.xlabel("Distances")
plt.xlabel("Number of simulations")
plt.ylabel("Percentage of rollout reached target")
# plt.ylabel("Mean distance (Max possible: 0.43; Min possible: 0.00)")
# plt.ylabel("Mean episode duration. (Min: 7)")

if not os.path.exists("plots/"):
    os.mkdir("plots")
plt.savefig("plots/rollout_hit_rate_in_first_state_comparison.png")
