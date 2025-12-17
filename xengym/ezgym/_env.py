from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import math
import numpy as np
import torch
from typing import Union, Sequence, Tuple

from ezgl import Matrix4x4


def gymtf_to_matrix(tf: gymapi.Transform) -> Matrix4x4:
    return Matrix4x4.fromVector7d(tf.p.x, tf.p.y, tf.p.z, tf.r.w, tf.r.x, tf.r.y, tf.r.z)

def gym_pos_rot_to_matrix(pos, rot) -> Matrix4x4:
    return Matrix4x4.fromVector7d(*pos, rot[3], rot[0], rot[1], rot[2])

def matrix_to_gymtf(mat: Matrix4x4):
    rot = mat.quat
    return gymapi.Transform(
        gymapi.Vec3(*mat.xyz), gymapi.Quat(rot.x(), rot.y(), rot.z(), rot.scalar())
    )

def matrix_to_gym7d(mat: Matrix4x4):
    rot = mat.quat
    return torch.Tensor([*mat.xyz, rot.x(), rot.y(), rot.z(), rot.scalar()])

def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

def create_ground(gym, sim, normal=(0, 0, 1)):
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(*normal)
    return gym.add_ground(sim, plane_params)


class Envs:

    class TensorType:
        ActorRootState = 1  # (num_actors, 13), pos([0:3]), rot([3:7]), linear vel([7:10]), angular vel([10:13]).
        DofForce = 2  # (num_dofs, 1)
        DofState = 3  # (num_dofs, 2), pos([0]), vel([1])
        ForceSensor = 4  # (num_force_sensors, 6), force([0:3]), torque([3:6])
        NetContactForce = 7  # (num_rigid_bodies, 3)
        RigidBodyState = 8  # (num_rigid_bodies, 13), pos([0:3]), rot([3:7]), linear vel([7:10]), angular vel([10:13]).
        Jiacobian = 5  #
        MassMatrix = 6  #

    RefreshFuncs = set()
    TensorBuffer = dict()
    envs = list()

    def __init__(
        self,
        gym: gymapi.Gym,
        sim: gymapi.Sim,
        num_envs: int=1,
        env_lower: gymapi.Vec3=gymapi.Vec3(-1, -1, 0.0),
        env_upper: gymapi.Vec3=gymapi.Vec3(1, 1, 1),
    ):
        """
        封装环境
        """
        assert len(self.envs) == 0, "Envs should be a singleton class"
        self.__tensor_api = {
            Envs.TensorType.ActorRootState :    [gym.acquire_actor_root_state_tensor,   gym.refresh_actor_root_state_tensor],
            Envs.TensorType.DofForce :          [gym.acquire_dof_force_tensor,          gym.refresh_dof_force_tensor],
            Envs.TensorType.DofState :          [gym.acquire_dof_state_tensor,          gym.refresh_dof_state_tensor],
            Envs.TensorType.ForceSensor :       [gym.acquire_force_sensor_tensor,       gym.refresh_force_sensor_tensor],
            Envs.TensorType.NetContactForce :   [gym.acquire_net_contact_force_tensor,  gym.refresh_net_contact_force_tensor],
            Envs.TensorType.RigidBodyState :    [gym.acquire_rigid_body_state_tensor,   gym.refresh_rigid_body_state_tensor],
            Envs.TensorType.Jiacobian :         [gym.acquire_jacobian_tensor,           gym.refresh_jacobian_tensors],
            Envs.TensorType.MassMatrix :        [gym.acquire_mass_matrix_tensor,        gym.refresh_mass_matrix_tensors],
        }
        self._gym = gym
        self._sim = sim
        self._assets = []
        self._env_lower = env_lower
        self._env_upper = env_upper
        self._num_envs = num_envs

    def __len__(self):
        return len(self.envs)

    def __iter__(self):
        return iter(self.envs)

    def __getitem__(self, idx):
        return self.envs[idx]

    def setRequiredTensor(self, asset: "Asset"):
        self._assets.append(asset)
        # setup required_tensors
        for tensor_type in asset.getTensorKeys():
            self.TensorBuffer[tensor_type] = None

            if isinstance(tensor_type, tuple):
                tensor_type, name = tensor_type

            self.RefreshFuncs.add(self.__tensor_api[tensor_type][1])

    def acquireTensors(self):
        self._gym.prepare_sim(self._sim)

        for tensor_type in Envs.TensorBuffer.keys():
            if isinstance(tensor_type, tuple):
                tensor_t, name = tensor_type
                Envs.TensorBuffer[tensor_type] = gymtorch.wrap_tensor(self.__tensor_api[tensor_t][0](self._sim, name))
            else:
                Envs.TensorBuffer[tensor_type] = gymtorch.wrap_tensor(self.__tensor_api[tensor_type][0](self._sim))

    def refreshTensors(self):
        self._gym.fetch_results(self._sim, True)
        for refresh_func in self.RefreshFuncs:
            refresh_func(self._sim)

    def addAssets(self, *assets):
        num_per_row = int(math.sqrt(self._num_envs))

        # set assets' required tensors
        for asset in assets:
            self.setRequiredTensor(asset)

        # create envs and assets
        for i in range(self._num_envs):
            env = self._gym.create_env(self._sim, self._env_lower, self._env_upper, num_per_row)

            for asset in assets:
                asset.create(env, i)

            Envs.envs.append(env)

    @classmethod
    def get(cls, tensor_type):
        return cls.TensorBuffer[tensor_type]



class Asset:

    def __init__(
        self,
        gym: gymapi.Gym,
        sim: gymapi.Sim,
        name: str,
        required_tensors: Sequence[Envs.TensorType] = tuple(),
    ):
        self.gym = gym
        self.sim = sim
        self._name = name
        self._tensor_types = required_tensors

    @property
    def name(self):
        return self._name

    def _getTensorKey(self, tensor_type):
        if tensor_type in [Envs.TensorType.Jiacobian, Envs.TensorType.MassMatrix]:
            return (tensor_type, self._name)
        return tensor_type

    def getTensorKeys(self):
        return [self._getTensorKey(tensor_type) for tensor_type in self._tensor_types]

    def create(self, env, group):
        raise NotImplementedError

    def getTensor(self, tensor_type: Envs.TensorType):
        return Envs.TensorBuffer[self._getTensorKey(tensor_type)]
