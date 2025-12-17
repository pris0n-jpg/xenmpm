import numpy as np
from enum import Enum

from xengym.ezgym import *
from xengym.ezgym.randomization import UniformNoiseModel
from xengym import Xensim
from xengym import ASSET_DIR


class State(Enum):
    InitToAbove = 0
    AboveToGrasp = 1
    GraspToAbove = 2
    Default = 4

state = State.InitToAbove

def create_sim(gym, args):
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.contact_offset = 0.001
    sim_params.physx.friction_offset_threshold = 0.001
    sim_params.physx.friction_correlation_distance = 0.0005
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

    return gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)



if __name__ == '__main__':
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    # ==== Add Tactile simulator ====
    simulator = Xensim(
            fem_file=ASSET_DIR/"data/fem_data_vec4070.npz",
            urdf_file=ASSET_DIR/"panda/panda_with_vectouch.urdf",
            object_file=ASSET_DIR/"obj/cube_15mm.obj",
            visible=False,
            left_visible=True,
            right_visible=True
        )
    simulator.cameraLookAt([1.5, 0, 0.8], [0, 0, 0.4], [0, 0, 1])

    # ==== create gym and sim =====
    # Add arguments
    args = gymutil.parse_arguments(description="Xense gym")
    args.use_gpu_pipeline = False
    device = args.sim_device if args.use_gpu_pipeline else 'cpu'

    gym = gymapi.acquire_gym()
    sim = create_sim(gym, args)
    create_ground(gym, sim)
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())

    # ==== prepare assets =====
    # table
    table_asset = BoxAsset(gym, sim, name="table", size=(0.6, 1.0, 0.4), pos=(0.5, 0.0, 0.2), euler=(0, 0, 0))

    # box
    box_asset = BoxAsset(
        gym, sim,
        name="box",
        size=(0.015, 0.015, 0.015),
        pos=UniformNoiseModel((3,), low=(0.5-0.2, 0.0-0.3, 0.4225), high=(0.5+0.1, 0.0+0.3, 0.4225)),
        euler=UniformNoiseModel((3,), low=(0, 0, -180), high=(0, 0, 180)),
    )

    # franka
    franka = FrankaAsset(gym, sim, name="franka", asset_file="panda/panda_with_vectouch.urdf")

    # ==== create env =====
    envs = Envs(gym, sim, 1)
    envs.addAssets(table_asset, box_asset, franka)
    envs.acquireTensors()
    gym.viewer_camera_look_at(viewer, envs[0], gymapi.Vec3(4, 3, 2), gymapi.Vec3(-4, -3, 0))

    # simulation loop
    franka_init_pose = franka.getPose()

    while not gym.query_viewer_has_closed(viewer):
        gym.simulate(sim)
        envs.refreshTensors()
        box_tf = box_asset.getPose()
        hand_tf = franka.getPose()
        hand_vel = franka.getVel()
        dof_pos = franka.getDofPos()

        if state == State.InitToAbove:
            box_yaw = box_tf.euler[2] % 90
            box_yaw = box_yaw if box_yaw < 45 else box_yaw - 90
            box_xyz = box_tf.xyz
            target_pose: Matrix4x4 = franka_init_pose.copy().rotate(-box_yaw, 0, 0, 1)
            target_pose.moveto(box_xyz[0], box_xyz[1], box_xyz[2]+0.2)
            target_gripper = torch.tensor([0.04, 0.04], device=device)

            if np.allclose(hand_tf.toVector7d(), target_pose.toVector7d(), atol=0.05) and hand_vel.norm() < 0.15:
                state = State.AboveToGrasp

        elif state == State.AboveToGrasp:
            box_yaw = box_tf.euler[2] % 90
            box_yaw = box_yaw if box_yaw < 45 else box_yaw - 90
            box_xyz = box_tf.xyz
            target_pose: Matrix4x4 = franka_init_pose.copy().rotate(-box_yaw, 0, 0, 1)
            target_pose.moveto(box_xyz[0], box_xyz[1], box_xyz[2] + 0.115)

            if np.allclose(hand_tf.toVector7d(), target_pose.toVector7d(), atol=0.05) and hand_vel.norm() < 0.15:
                target_gripper = torch.tensor([0.006, 0.006], device=device)
            else:
                target_gripper = torch.tensor([0.04, 0.04], device=device)

            if dof_pos[-2] + dof_pos[-1] < 0.0155:
                state = State.GraspToAbove

        elif state == State.GraspToAbove:
            target_pose = franka_init_pose.copy()
            target_gripper = torch.tensor([0.006, 0.006], device=device)
            if hand_tf.xyz[2] > 0.7:
                target_gripper = torch.tensor([0.04, 0.04], device=device)

            if np.allclose(hand_tf.toVector7d(), target_pose.toVector7d(), atol=0.05) and hand_vel.norm() < 0.15:
                state = State.InitToAbove

        else:
            state = State.Default

        # control franka
        franka.control(matrix_to_gym7d(target_pose).unsqueeze(0), target_gripper)

        # update tactile sim
        simulator.set_data(obj_pose=box_tf.toVector7d(), panda_joints=dof_pos.tolist())
        simulator.left_sensor.step()
        simulator.left_sensor.update()
        simulator.right_sensor.step()
        simulator.right_sensor.update()

        # update viewer
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, False)
        gym.sync_frame_time(sim)

    # cleanup
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)