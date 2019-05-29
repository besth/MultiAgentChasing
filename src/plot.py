import os
import numpy as np
import matplotlib.pyplot as plt


num_simulations = [100, 250, 500, 1000, 1500, 2000]
distances = [2, 4, 6, 8, 10]


def extract_action_probability():
    action_dist = []
    # for num_simulation in num_simulations:
    #     lines = [line.rstrip() for line in open("data/action_distribution_sim{}".format(num_simulation))]
    for distance in distances:
        lines = [line.rstrip() for line in open("data/action_distribution_sim500_vdis{}".format(distance))]
        num_up, num_down, num_right, num_left = 0, 0, 0, 0
        for line in lines:
            if line == "(10, 0)":
                num_right += 1
            elif line == "(-10, 0)":
                num_left += 1
            elif line == "(0, 10)":
                num_up += 1
            elif line == "(0, -10)":
                num_down += 1

        action_dist.append([num_up, num_down, num_left, num_right])

    # print(action_dist)
    return action_dist


def plot_action_probability_with_simulation(action_dist):
    frequencies = [[dist[i] for dist in action_dist] for i in range(4)]
    plt.figure()

    # Simulations
    # plt.title("first step action probability and number of simulations")
    # plt.plot(num_simulations, frequencies[0], label="up")
    # plt.plot(num_simulations, frequencies[1], label="down")
    # plt.plot(num_simulations, frequencies[2], label="left")
    # plt.plot(num_simulations, frequencies[3], label="right")
    # plt.xlabel("number of simulations")

    # Distances
    plt.title("first step action probability and vertical distance (sim=500)")
    plt.plot(distances, frequencies[0], label="up")
    plt.plot(distances, frequencies[1], label="down")
    plt.plot(distances, frequencies[2], label="left")
    plt.plot(distances, frequencies[3], label="right")
    plt.xlabel("vertical distance to the wall (horizontal dis=2)")

    plt.legend()
    plt.ylabel("first step action frequency (out of 100)")
    # print(frequencies)
    if not os.path.exists("plots/"):
        os.mkdir("plots")
    # plt.savefig("plots/action_freq_with_simulation_dis10.png")
    plt.savefig("plots/action_freq_vertical_distance.png")


# def plot_action_probability_with_distance(action_dist):



#if __name__ == "__main__":
#    action_dist = extract_action_probability()
#    plot_action_probability_with_simulation(action_dist)


# num_sim = [100, 200, 500, 1000, 1500, 2000]
# distances = [1, 2, 4, 8, 12, 16]
#
# # rollout hit target rate
# rollout_reach_target_prob_no_heuristic_distance = [0.954047619047619, 0.9104166666666667, 0.8246078431372549, 0.6477848101265823, 0.46820754716981133, 0.41257692307692306]
# rollout_reach_target_prob_no_heuristic_simulation_distance16 = [0.3172, 0.4126, 0.5807, 0.6829]
# rollout_reach_target_prob_with_heuristic_distance = [0.9712727272727273, 0.9503888888888888, 0.9300851063829787, 0.8400821917808219, 0.7460222222222223, 0.6762222222222222]
#
# # rollout hit target rate at initial state only
# rollout_in_init_state_reach_target_prob_no_heuristic_simulation_distance16 = [0.013, 0.0137, 0.01648, 0.0233, 0.021213333333333334, 0.03697]
# rollout_in_init_state_reach_target_prob_with_heuristic_simulation_distance16 = [0.0294, 0.0356, 0.08168, 0.34642, 0.37722666666666665, 0.54976]
# # first step optimality
# mean_first_step_distance_to_optimal_no_heuristic = [0.1751776429621799, 0.15343741661206808, 0.10593741661206812, 0.1198498445057071, 0.07234984450570713, 0.01808746112642678]
# mean_first_step_distance_to_optimal_with_heuristic_test50 = [0.1520751831034078, 0.08320232118156319, 0.08681981340684855, 0.08991983560283538, 0.05064489115399499, 0.05787987560456569]
# mean_first_step_distance_to_optimal_no_heuristic_test50 = [0.20149913343136788, 0.20104943879628082, 0.140337871969986, 0.11110731892524901, 0.09631981340684856, 0.08108261325336098]
#
# # mean episode length
# mean_episode_length_no_heuristic_dis16_test10 = [13.5, 12.2, 10.2, 9.8, 10.1, 9.7]
# mean_episode_length_with_heuristic_dis16_test10 = [12.8, 10.4, 10.3, 10.1, 9.8, 10.3]
# mean_episode_length_no_heuristic_dis16_test50 = [15.02, 11.76, 10.3, 10.24, 9.78, 9.58]
# mean_episode_length_with_heuristic_dis16_test50 = [13.74, 10.98, 10.12, 10.56, 10.24, 10.38]
#
# # distance to target for different max steps
# max_steps = np.arange(1, 13)
# mean_distance_to_target_with_heuristic_sim500 = [15.8023620654299, 15.168412262015034, 14.0994769287341, 12.82320309149236, 9.072876300272878, 11.241406887751191, 7.350859511354235, 5.076094301679079, 2.3281031160852104, 1.5318510838271664, 0.5465276075034005, 0.42568716167768067]
# mean_distance_to_target_no_heuristic_sim500 = [15.877199130529553, 15.39005088146643, 14.641825993422605, 13.426876447882593, 11.384739251102731, 10.337768169146067, 9.108861308268747, 5.46807972738342, 2.3192946502288994, 2.062786824404347, 1.373861174287822, 0.717182282511181]
#
# mean_distance_to_target_with_heuristic_sim250 = [15.81523941371039, 15.220306487926594, 14.164854082490212, 12.89100968641116, 11.370796828713457, 9.58697206499547, 7.412923694907733, 5.737312493223849, 2.9733575476286638, 1.1179794499550506, 0.7187301602859624, 0.7128785027394149]
# mean_distance_to_target_no_heuristic_sim250 = [15.87294196248092, 15.463919576025674, 14.7756689004439, 14.147214597790418, 12.460833529493826, 11.457087372564882, 8.610470008188555, 9.068221770213118, 5.386417635782457, 3.515563401112227, 1.9225428041498076, 3.856280834466229]
#
# mean_distance_to_target_with_heuristic_sim100 = [15.834730450693742, 15.3061017528146, 14.576259363725642, 13.538340468990066, 12.06662504616054, 10.892617675949031, 8.98281009269319, 7.156571454523663, 5.087340698038521, 4.167520951507694, 3.0642430282319153, 2.329895023051942]
# mean_distance_to_target_no_heuristic_sim100 = [15.939617828013558, 15.71482396400861, 15.361495807883665, 14.844995567217955, 15.310726842719346, 12.761183656874934, 12.065966577295194, 10.52455915910288, 8.946875473211149, 7.25191195805128, 7.114647127427693, 5.322244553265623]
#
# mean_distance_to_target_optimal = [15.025, 13.787500000000001, 12.05, 9.8125, 7.074999999999999, 3.8375000000000004, 0.29875000000000007, 0, 0, 0, 0, 0]
#
# max_steps_small_action = np.arange(1, 21)
# mean_distance_to_target_heuristic_small_action_sim250 = [15.894153140296517, 15.59979927136164, 15.108228363191154, 14.340013824059934, 13.340503247828988, 12.23333840036797, 10.930984877377222, 9.93296342721213, 8.50364800877818, 6.464630606194635, 4.8988470384285225, 3.431049262406836, 1.399252337821949, 0.5307849863334508, 0.44163637120472915, 0.36074617097374884, 0.356260238750292, 0.35897913220596983, 0.3616077968394601, 0.3589081067628006]
# mean_distance_to_target_no_heuristic_small_action_sim250 = [15.946973720770547, 15.876708654930708, 15.55134601990904, 15.68, 14.79, 14.36, 13.63, 12.44, 12.23, 10.76, 11.03, 8.66, 7.88, 6.03, 5.41, 4.52, 2.33, 5.06, 3.10, 3.20]
# mean_distance_to_target_optimal_small_action = [15.5125, 14.89375, 14.025, 12.90625, 11.5375, 9.91875, 8.05, 5.93125, 3.5625, 0.94375, 0.39, 0,0,0,0,0,0,0,0,0]
#



# Test mean episode length with respect to distance
#dis = [2,4,6,8,9]
#dis2 = np.dot(dis, np.sqrt(2))
#mean_ep_len_wall_random = [76.98, 80.34, 82.64, 82.28, 81.98]
#mean_ep_len_corner_random = [82.56, 80.58, 87.8, 89.96, 90.26]
#plt.figure()
#plt.plot(dis, mean_ep_len_wall_random, label="to wall")
#plt.plot(dis2, mean_ep_len_corner_random, label="to corner")
# # print(len(mean_distance_to_target_no_heuristic_small_action_sim250))
# # plt.plot(distance, prob_heuristic)
#
# plt.plot(max_steps_small_action, mean_distance_to_target_no_heuristic_small_action_sim250, label="no heuristic")
# plt.plot(max_steps_small_action, mean_distance_to_target_heuristic_small_action_sim250, label="with heuristic")
# plt.plot(max_steps_small_action, mean_distance_to_target_optimal_small_action, label="optimal")
#plt.legend()
#plt.title("mean episode length of random policy")
#plt.xlabel("Distances")
#plt.ylabel("mean episode length (max=100)")
# plt.xlabel("Number steps")
# plt.ylabel("Distance to target ([0, 16])")
# # plt.ylabel("Mean distance (Max possible: 0.43; Min possible: 0.00)")
# # plt.ylabel("Mean episode duration. (Min: 7)")


# Test killzone size effect
sizes = [0.2, 0.4, 0.6, 0.7, 0.8, 1.0]
random_survival_time_max10 = [7.28, 5.8, 3.96, 2.88, 2.52, 1.64]
mcts_survival_time_max10 = [10.0, 10.0, 10.0, 10.0, 4.0, 3.8]
plt.figure()
plt.plot(sizes, random_survival_time_max10, label="random")
plt.plot(sizes, mcts_survival_time_max10, label="mcts")

plt.legend()
plt.title("mean episode length(sim=250)")
plt.xlabel("killzone sizes")
plt.ylabel("mean episode length (max=10)")




if not os.path.exists("plots/"):
    os.mkdir("plots")
plt.savefig("plots/killzone_size_comparison.png")
