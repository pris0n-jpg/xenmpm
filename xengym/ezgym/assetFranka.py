from ._env import *
from .. import PROJ_DIR


class FrankaAsset(Asset):

    def __init__(
        self,
        gym: gymapi.Gym,
        sim: gymapi.Sim,
        name: str,
        asset_file: str,
        asset_root: str = str(PROJ_DIR / "assets"),
        controller: str = "ik",  # Controller to use for Franka. Options are {ik, osc}
        required_tensors: Sequence[Envs.TensorType] = (
            Envs.TensorType.Jiacobian,
            Envs.TensorType.MassMatrix,
            Envs.TensorType.RigidBodyState,
            Envs.TensorType.DofState,
        ),
    ):
        super().__init__(gym, sim, name, required_tensors)

        self.controller = controller

        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = True
        self._asset = self.gym.load_asset(sim, asset_root, asset_file, asset_options)
        self._actors = []
        self._hand_idx_in_asset = None
        self._hand_idxs = []
        self._dof_num = None
        self.initProps()

    def initProps(self):
        """
        初始化属性
        """
        self._dof_props = self.gym.get_asset_dof_properties(self._asset)
        if self.controller == "ik":
            self._dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
            self._dof_props["stiffness"][:7].fill(400.0)
            self._dof_props["damping"][:7].fill(40.0)
        else:       # osc
            self._dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_EFFORT)
            self._dof_props["stiffness"][:7].fill(0.0)
            self._dof_props["damping"][:7].fill(0.0)

        # grippers
        self._dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
        self._dof_props["stiffness"][7:].fill(800.0)
        self._dof_props["damping"][7:].fill(40.0)

        self._lower_limits = self._dof_props["lower"]
        self._upper_limits = self._dof_props["upper"]
        # self._dof_pos_default = 0.3 * (self._upper_limits + self._lower_limits)
        self._dof_pos_default = np.array([0, 0, 0, -3*np.pi/8, 0, 3*np.pi/8, np.pi/4, 0.4, 0.4], dtype=np.float32)
        self._dof_num = len(self._dof_pos_default)
        self._dof_state_default = np.zeros(self.gym.get_asset_dof_count(self._asset), gymapi.DofState.dtype)
        self._dof_state_default["pos"] = self._dof_pos_default

        franka_link_dict = self.gym.get_asset_rigid_body_dict(self._asset)
        self._hand_idx_in_asset = franka_link_dict["panda_hand"]

    def create(self, env, group):
        """
        创建机器人
        """
        pose = gymapi.Transform()
        filter = 2
        _handle = self.gym.create_actor(env, self._asset, pose, self.name, group, filter)
        self.gym.set_actor_dof_properties(env, _handle, self._dof_props)
        self.gym.set_actor_dof_states(env, _handle, self._dof_state_default, gymapi.STATE_POS)
        self.gym.set_actor_dof_position_targets(env, _handle, self._dof_pos_default)
        self._actors.append(_handle)
        self._hand_idxs.append(self.gym.find_actor_rigid_body_index(env, _handle, "panda_hand", gymapi.DOMAIN_SIM))

    @property
    def j_eef(self):
        return self.getTensor(Envs.TensorType.Jiacobian)[:, self._hand_idx_in_asset - 1, :, :7]

    @property
    def mm(self):
        return self.getTensor(Envs.TensorType.MassMatrix)[:, :7, :7]

    def control(self, target: torch.Tensor, gripper: torch.Tensor):
        """
        Parameters:
        - target : torch.Tensor, shape=(num_envs, 7), x y z qx qy qz qw
        - gripper : torch.Tensor, shape=(num_envs, 2), left and right gripper
        """
        curr_pos = Envs.get(Envs.TensorType.RigidBodyState)[self._hand_idxs, :3]
        curr_rot = Envs.get(Envs.TensorType.RigidBodyState)[self._hand_idxs, 3:7]

        pos_err = target[:, :3] - curr_pos
        orn_err = orientation_error(target[:, 3:], curr_rot)
        dpose = torch.cat([pos_err, orn_err], -1)

        # ik control
        damping = 0.05
        j_eef = self.j_eef
        j_eef_T = torch.transpose(j_eef, 1, 2)
        lmbda = torch.eye(6) * (damping ** 2)
        u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose.unsqueeze(-1)).view(len(self._actors), 7)
        u = torch.clamp(u, -0.1, 0.1)

        # current joint angles
        dof_pos = Envs.get(Envs.TensorType.DofState)[:, 0].view(len(self._actors), 9)

        # target joint angles
        dof_pos_target = torch.zeros_like(dof_pos)
        dof_pos_target[:, :7] = dof_pos[:, :7] + u
        dof_pos_target[:, 7:] = dof_pos[:, 7:] + torch.clamp(gripper - dof_pos[:, 7:], -0.005, 0.005)
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(dof_pos_target))

    def getPoseTensor(self, idx=0) -> Matrix4x4:
        """
        获取机器人末端位姿
        """
        pos_rot = Envs.get(Envs.TensorType.RigidBodyState)[self._hand_idxs[idx], :7]
        pos, rot = pos_rot[:3], pos_rot[3:]
        return Matrix4x4.fromVector7d(*pos, rot[3], rot[0], rot[1], rot[2])

    def getPose(self, idx=0) -> Matrix4x4:
        hand_handle = self.gym.find_actor_rigid_body_handle(Envs.envs[idx], self._actors[idx], "panda_hand")
        hand_pose = self.gym.get_rigid_transform(Envs.envs[idx], hand_handle)
        return gymtf_to_matrix(hand_pose)

    def getVel(self, idx=0):
        return Envs.get(Envs.TensorType.RigidBodyState)[self._hand_idxs[idx], 7:10]

    def getDofPos(self, idx=0):
        return Envs.get(Envs.TensorType.DofState)[:, 0].view(-1, 9)[idx]

    def getDofVel(self, idx=0):
        return Envs.get(Envs.TensorType.DofState)[:, 1].view(-1, 9)[idx]
