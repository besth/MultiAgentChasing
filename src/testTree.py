import numpy as np
import anytree
from anytree import AnyNode as Node
from anytree import RenderTree
import mujoco_py as mujoco
# import as array
import skvideo.io
import time
# from anytree import DotExporter

def sampleTrajectory(root_node):
    curr_node = root_node
    trajectory = []
    while(curr_node.children != ()):
        children_visit = [child.num_visited for child in curr_node.children]
        maxIndex = np.argwhere(children_visit == np.max(children_visit)).flatten()
        selected_child_index = np.random.choice(maxIndex)
        curr_state = list(curr_node.id.values())[0]
        trajectory.append(curr_state)

        next_node = curr_node.children[selected_child_index]
        curr_node = next_node

    return trajectory

def visualizeTrajectory(trajectory):
    model_name = "twoAgents"
    model = mujoco.load_model_from_path('xmls/' + model_name + '.xml')
    simulation = mujoco.MjSim(model)
    viewer = mujoco.MjViewer(simulation)
    frames = []
    num_simulation_frames = 20
    for state in trajectory:

        simulation.data.qpos[:] = np.array([state[0][:2], state[1][:2]]).flatten()
        # simulation.data.body_xpos[:] = np.array([state[0][2:4], state[1][2:4]]).flatten()
        # time.sleep(5)
        for _ in range(num_simulation_frames):
            simulation.step()
            frame = simulation.render(1024, 1024, camera_name="center")#, mode="window")
            frames.append(frame)

    return frames

# def renderTree(tree):
#     DotExporter(tree).to_picture("tree.png")


def main():
    # f = open("traj.txt", "r")
    # trajectory = f.read()
    # print(trajectory, type(trajectory))
    trajectories = [[np.array([[ 1.51470683, -0.16627798,  2.9915595 ,  1.37550927,  2.31473365,
        -4.17872471],
       [-0.99781386,  0.20037078, -0.99781386,  0.20037078,  0.        ,
         0.        ]]), np.array([[ 1.79390356, -0.81827292,  3.28825623,  0.70601432,  0.56473365,
        -2.42872471],
       [-0.99781386,  0.20037078, -0.99781386,  0.20037078,  0.        ,
         0.        ]]), np.array([[ 1.6443503 , -1.30401786,  3.16370296,  0.22026938, -1.93526635,
        -2.42872471],
       [-0.99781386,  0.20037078, -0.99781386,  0.20037078,  0.        ,
         0.        ]]), np.array([[ 1.44104703, -1.60601281,  2.94289969, -0.09922556, -0.18526635,
        -0.67872471],
       [-0.99781386,  0.20037078, -0.99781386,  0.20037078,  0.        ,
         0.        ]]), np.array([[ 1.22024376, -1.55800775,  2.73959642, -0.0687205 , -1.93526635,
         1.07127529],
       [-0.99781386,  0.20037078, -0.99781386,  0.20037078,  0.        ,
         0.        ]]), np.array([[ 0.64944049, -1.52750269,  2.18629315, -0.02071544, -3.68526635,
        -0.67872471],
       [-0.99781386,  0.20037078, -0.99781386,  0.20037078,  0.        ,
         0.        ]]), np.array([[-0.35011278, -1.66324763,  1.21173988, -0.15646038, -6.18526635,
        -0.67872471],
       [-0.99781386,  0.20037078, -0.99781386,  0.20037078,  0.        ,
         0.        ]]), np.array([[-1.58716605, -1.53649257, -0.02531339, -0.05470532, -6.18526635,
         1.82127529],
       [-0.99781386,  0.20037078, -0.99781386,  0.20037078,  0.        ,
         0.        ]])],
[np.array([[ 0.84313895,  0.38504973,  2.36777628,  1.96409773, -2.46373316,
        -7.90479941],
       [ 0.64910007,  0.07663953,  0.64910007,  0.07663953,  0.        ,
         0.        ]]), np.array([[ 0.08789232, -1.19591015,  1.63752965,  0.38313784, -4.96373316,
        -7.90479941],
       [ 0.64910007,  0.07663953,  0.64910007,  0.07663953,  0.        ,
         0.        ]]), np.array([[-1.08860431, -2.59312003,  0.47853302, -1.03157204, -6.71373316,
        -6.15479941],
       [ 0.64910007,  0.07663953,  0.64910007,  0.07663953,  0.        ,
         0.        ]]), np.array([[-2.24760095, -3.64032992, -0.69796361, -2.09628192, -4.96373316,
        -4.40479941],
       [ 0.64910007,  0.07663953,  0.64910007,  0.07663953,  0.        ,
         0.        ]]), np.array([[-3.50284758, -4.5212898 , -1.92821025, -2.97724181, -7.46373316,
        -4.40479941],
       [ 0.64910007,  0.07663953,  0.64910007,  0.07663953,  0.        ,
         0.        ]]), np.array([[-4.81184421, -5.58599968, -3.25470688, -4.02445169, -5.71373316,
        -6.15479941],
       [ 0.64910007,  0.07663953,  0.64910007,  0.07663953,  0.        ,
         0.        ]])],
[np.array([[ 0.70017747, -0.75222703,  2.19629368,  0.80374472,  0.38837871,
        -5.59717481],
       [ 0.79153919,  0.5956463 ,  0.79153919,  0.5956463 ,  0.        ,
         0.        ]]), np.array([[ 1.04035321, -1.87166199,  2.51146942, -0.31569025,  2.88837871,
        -5.59717481],
       [ 0.79153919,  0.5956463 ,  0.79153919,  0.5956463 ,  0.        ,
         0.        ]]), np.array([[ 1.61802895, -3.25359696,  3.08914516, -1.67262521,  2.88837871,
        -8.09717481],
       [ 0.79153919,  0.5956463 ,  0.79153919,  0.5956463 ,  0.        ,
         0.        ]]), np.array([[ 2.37945469, -5.05678192,  3.83307091, -3.45831017,  4.63837871,
        -9.84717481],
       [ 0.79153919,  0.5956463 ,  0.79153919,  0.5956463 ,  0.        ,
         0.        ]]), np.array([[ 3.49088044, -6.84246688,  4.92699665, -5.26149513,  6.38837871,
        -8.09717481],
       [ 0.79153919,  0.5956463 ,  0.79153919,  0.5956463 ,  0.        ,
         0.        ]]), np.array([[ 4.95230618, -8.64565184,  6.37092239, -7.04718009,  8.13837871,
        -9.84717481],
       [ 0.79153919,  0.5956463 ,  0.79153919,  0.5956463 ,  0.        ,
         0.        ]]), np.array([[  6.84248192, -10.6150868 ,   8.23609814,  -9.01661505,
         10.63837871,  -9.84717481],
       [  0.79153919,   0.5956463 ,   0.79153919,   0.5956463 ,
          0.        ,   0.        ]])],
[np.array([[ 0.07346129,  1.39954574,  1.59730483,  2.93463583, -2.38435355,
        -3.50900837],
       [-0.03894206,  1.98707938, -0.03894206,  1.98707938,  0.        ,
         0.        ]]), np.array([[-0.58715942,  0.88149407,  0.95418412,  2.39908415, -4.13435355,
        -1.75900837],
       [-0.03894206,  1.98707938, -0.03894206,  1.98707938,  0.        ,
         0.        ]])],
[np.array([[-0.13267544, -0.63349502,  1.35389683,  0.95118029,  1.34277263,
        -8.4675312 ],
       [ 1.60055657,  0.17762824,  1.60055657,  0.17762824,  0.        ,
         0.        ]])]]





    frames = []
    for traj in trajectories:
        frame = visualizeTrajectory(traj)
        # np.concatenate(frames, frame)
        frames += frame


    # frames = np.array([visualizeTrajectory(trajectory)for trajectory in trajectories])#.flatten()
    # print(frames)
    if len(frames) != 0:
        print("Generating video")
        skvideo.io.vwrite("./video_test.mp4", frames)

if __name__ == "__main__":
    # tree =
    main()
