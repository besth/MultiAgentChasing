import numpy as np
import anytree
from anytree import AnyNode as Node
from anytree import RenderTree
import mujoco_py as mujoco
# import as array
import skvideo.io
import time

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
    for state in trajectory:

        simulation.data.qpos[:] = np.array([state[0][:2], state[1][:2]]).flatten()
        # simulation.data.body_xpos[:] = np.array([state[0][2:4], state[1][2:4]]).flatten()
        time.sleep(5)
        simulation.step()
        frame = simulation.render(1024, 1024, camera_name="center")#, mode="window")
        frames.append(frame)

    return frames


def main():
    f = open("traj.txt", "r")
    trajectory = f.read()
    print(trajectory, type(trajectory))
    trajectory = [np.array([[0.43001841, 0.37541288, 1.90851749, 1.85664224, 2.15009203,
        1.87706442],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ]]), np.array([[ 0.86003681,  0.48832577,  2.33853589,  1.99455512,  2.15009203,
        -0.62293558],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ]]), np.array([[ 1.02755522,  0.36373865,  2.5310543 ,  1.86996801, -0.34990797,
        -0.62293558],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ]]), np.array([[ 0.69507362,  0.23915154,  2.2235727 ,  1.74538089, -2.84990797,
        -0.62293558],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ]]), np.array([[ 0.12509203, -0.14793558,  1.65359111,  1.38329378, -2.84990797,
        -3.12293558],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ]]), np.array([[-0.62863957, -0.95627269,  0.91735951,  0.59245666, -4.59990797,
        -4.87293558],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ]])]

    frames = visualizeTrajectory(trajectory)
    if len(frames) != 0:
        print("Generating video")
        skvideo.io.vwrite("./video_test.mp4", frames*10)

if __name__ == "__main__":
    main()
