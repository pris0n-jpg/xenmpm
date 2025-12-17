from ._env import *
from .randomization import GaussianNoiseModel, UniformNoiseModel, NoiseModel


class BoxAsset(Asset):
    def __init__(
        self,
        gym: gymapi.Gym,
        sim: gymapi.Sim,
        name: str,
        size = (0.6, 1.0, 0.4),
        pos: Union[NoiseModel, Sequence] = (0.5, 0.0, 0.2),
        euler: Union[NoiseModel, Sequence] = (0, 0, 0),
        fixed_base = False,
        required_tensors: Sequence[Envs.TensorType] = tuple(),
    ):
        super().__init__(gym, sim, name, required_tensors)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = fixed_base
        self._asset = gym.create_box(sim, *size, asset_options)

        self._pos_sampler = UniformNoiseModel((3,), pos, pos) if isinstance(pos, Sequence) else pos
        self._euler_sampler = UniformNoiseModel((3,), euler, euler) if isinstance(euler, Sequence) else euler
        self._actors = []
        self._idxs = []

    def create(self, env, group):
        handle = self.gym.create_actor(env, self._asset, self.randomPose(), self.name, group, 0)
        self._actors.append(handle)
        self._idxs.append(self.gym.find_actor_rigid_body_index(env, handle, self.name, gymapi.DOMAIN_SIM))

    def randomPose(self):
        pose = Matrix4x4.fromEulerAngles(*self._euler_sampler.sample()).moveto(*self._pos_sampler.sample())
        pose = matrix_to_gymtf(pose)
        return pose

    def getPose(self, idx=0) -> Matrix4x4:
        pos_rot = Envs.get(Envs.TensorType.RigidBodyState)[self._idxs[idx], :7]
        pos, rot = pos_rot[:3], pos_rot[3:]
        return Matrix4x4.fromVector7d(*pos, rot[3], rot[0], rot[1], rot[2])

