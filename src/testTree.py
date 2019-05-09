import numpy as np
import anytree
from anytree import AnyNode as Node
from anytree import RenderTree
import mujoco_py as mujoco
# import as array

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
    for state in trajectory:

        simulation.data.qpos[:] = np.array([state[0][:2], state[1][:2]]).flatten()
        # simulation.data.body_xpos[:] = np.array([state[0][2:4], state[1][2:4]]).flatten()
        simulation.step()
        simulation.render(1024, 1024, camera_name="center", mode="window")


def main():
    f = open("traj.txt", "r")
    trajectory = f.read()
    print(trajectory, type(trajectory))
    # trajectory 

    visualizeTrajectory(trajectory)

if __name__ == "__main__":
    main()
