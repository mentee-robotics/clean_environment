# Copyright (c) 2021-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask
from typing import Dict, Any, Tuple
import pytorch_kinematics as pk

@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values will be mapped to 0

    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # type: (Tensor, float) -> Tensor
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
        torch.cos(angle[idx, :] / 2.0)
    ], dim=-1)

    # Reshape and return output
    quat = quat.reshape(list(input_shape) + [4, ])
    return quat


class FrankaCubeStack(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.franka_position_noise = self.cfg["env"]["frankaPositionNoise"]
        self.franka_rotation_noise = self.cfg["env"]["frankaRotationNoise"]
        self.franka_dof_noise = self.cfg["env"]["frankaDofNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        # Create dicts to pass to reward function
        self.reward_settings = {
            "r_dist_scale": self.cfg["env"]["distRewardScale"],
            "r_lift_scale": self.cfg["env"]["liftRewardScale"],
            "r_align_scale": self.cfg["env"]["alignRewardScale"],
            "r_stack_scale": self.cfg["env"]["stackRewardScale"],
        }

        # Controller type
        self.control_type = self.cfg["env"]["controlType"]
        assert self.control_type in {"osc", "joint_tor"},\
            "Invalid control type specified. Must be one of: {osc, joint_tor}"

        # dimensions
        # obs include: cubeA_pose (7) + cubeB_pos (3) + eef_pose (7) + q_gripper (2)
        self.cfg["env"]["numObservations"] = 12
        # actions include: delta EEF if OSC (6) or joint torques (7) + bool gripper (1)
        self.cfg["env"]["numActions"] = 6

        # Values to be filled in at runtime
        self.states = {}                        # will be dict filled with relevant states to use for reward calculation
        self.handles = {}                       # will be dict mapping names to relevant sim handles
        self.num_dofs = None                    # Total number of DOFs per env
        self.actions = None                     # Current actions to be deployed
        self._init_cubeA_state = None           # Initial state of cubeA for the current env
        self._init_cubeB_state = None           # Initial state of cubeB for the current env
        self._cubeA_state = None                # Current state of cubeA for the current env
        self._cubeB_state = None                # Current state of cubeB for the current env
        self._cubeA_id = None                   # Actor ID corresponding to cubeA for a given env
        self._cubeB_id = None                   # Actor ID corresponding to cubeB for a given env

        # Tensor placeholders
        self._root_state = None             # State of root body        (n_envs, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof)
        self._q = None  # Joint positions           (n_envs, n_dof)
        self._qd = None                     # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        self._contact_forces = None     # Contact forces in sim
        self._eef_state = None  # end effector state (at grasping point)
        self._eef_lf_state = None  # end effector state (at left fingertip)
        self._eef_rf_state = None  # end effector state (at left fingertip)
        self._j_eef = None  # Jacobian for end effector
        self._mm = None  # Mass matrix
        self._arm_control = None  # Tensor buffer for controlling arm
        self._gripper_control = None  # Tensor buffer for controlling gripper
        self._pos_control = None            # Position actions
        self._effort_control = None         # Torque actions
        self._franka_effort_limits = None        # Actuator effort limits for franka
        self._global_indices = None         # Unique indices corresponding to all envs in flattened array

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self.actions = torch.zeros((self.num_envs, self.num_actions)).to(self.device)
        self.current_cube_obs = torch.zeros((self.num_envs, 3)).to(self.device)
        self.current_targets = torch.zeros((self.num_envs, 3)).to(self.device)

        self.height_target = torch.ones((self.num_envs, 1)).to(self.device ) * 0.075
        # Franka defaults
        self.franka_default_dof_pos = to_torch(
            [-1, 1.28, 0, -1.6, 0, 0.87, 0.87, 0.87, 0.87], device=self.device
        )

        chain = pk.build_serial_chain_from_urdf(open(
            "/home/guy/VSprojects/clean_environment/assets/urdf/happybot_arms_v1_0/urdf/happy_arms_gym_no_encoding.urdf").read(),
                                                "right_gripper_ee")
        self.chain = chain.to(dtype= torch.float32, device=self.device)


        init_points = []
        import ast
        with open('/home/guy/VSprojects/clean_environment/isaacgymenvs/starting_points_new_01.txt', 'r') as f:
            init_points = [ast.literal_eval(lst.strip()) for lst in f.readlines()]
        self.init_points = torch.tensor(init_points, device = self.device)

        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # Refresh tensors
        #self._refresh()
        self.compute_observations()




    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def compute_torques(self, actions):
        kp = torch.tensor([100., 100., 65., 100., 10., 10., 10., 10., 10.]).to(self.device)
        kd = torch.tensor([50., 50., 65., 50., 0.15, 0.001, 0.001, 0.001, 0.001]).to(self.device)
        torques = kp * (actions - self.states['q']) - kd * self.states['q_vel']
        return torques

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics of the environment.

        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """

        # randomize actions
        if self.dr_randomizations.get('actions', None):
            actions = self.dr_randomizations['actions']['noise_lambda'](actions)

        # action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        action_tensor = actions
        # apply actions
        self.pre_physics_step(action_tensor)

        # step physics and render each frame
        # mean_torques = torch.zeros((self.first_step_torques.shape[0], self.first_step_torques.shape[1]), device = self.first_step_torques.device)
        # mean_gripper_torque = torch.zeros((self.num_envs, 1), device = mean_torques.device)

        # self.states['previous_env_dt_q'] = self.states['q'][:, -1].clone().unsqueeze(1)
        # self.states['prev_ee_pos'] = self.states['eef_pos'][:, :].clone() - self._base_link_state[:, :3]
        # self.states['prev_ee_quat'] = self.states['eef_quat'][:, :].clone()

        for i in range(20):
            torques = self.compute_torques(self._pos_control) #+ self.additional_torques
            if i == 0:
                self.first_step_torques = torques.clone()
            upper_torque_limit = torch.tensor([10, 10, 10, 5, 5, 5, 5, 5, 5]).to(self.device)
            torques = torch.clip(torques, min=-upper_torque_limit, max=upper_torque_limit)
            # mean_torques += torques.abs()
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques.detach()))
            self.states['previous_q'] = self.states['q'].clone()
            # mean_gripper_torque += torques[:, 6:]
            self.gym.simulate(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self._update_states()
            if self.force_render:
                self.render()
        if self.num_envs < 100 and False:
            print(self.actions[-1])
            print(self.first_step_torques[-1])

        # mean_torques /= self.decimation
        # print(mean_torques.mean(dim=0))

        # self.mean_gripper_torque = mean_gripper_torque / self.decimation

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()

        # fill time out buffer: set to 1 if we reached the max episode length AND the reset buffer is 1. Timeout == 1 makes sense only if the reset buffer is 1.
        self.timeout_buf = (self.progress_buf >= self.max_episode_length - 1) & (self.reset_buf != 0)

        # randomize observations
        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)

        self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)

        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return self.obs_dict, self.rew_buf.to(self.rl_device), self.reset_buf.to(self.rl_device), self.extras


    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        franka_asset_file = "urdf/final_new_gripper/urdf/happy_arms_gripper.urdf"

        # load franka asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.replace_cylinder_with_capsule = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        # asset_options.vhacd_enabled = True
        asset_options.disable_gravity = False
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = 3
        asset_options.use_mesh_materials = True

        asset_options.vhacd_params.resolution = 300000
        asset_options.vhacd_params.max_convex_hulls = 2
        asset_options.vhacd_params.max_num_vertices_per_ch = 64

        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)

        franka_dof_stiffness = to_torch([0] * 11, dtype=torch.float, device=self.device)
        franka_dof_damping = to_torch([1.617, 0.893, 1.553, 0.64,  0.07] + [0.001] * 4, dtype=torch.float, device=self.device)
        franka_dof_armature = to_torch([0.078, 0.862, 0.395, 0.303,  0.003] + [0.001] * 4, dtype=torch.float, device=self.device)
        franka_dof_friction = to_torch([0., 0.003, 0., 0.056, 0.017] + [0.001] * 4, dtype=torch.float, device=self.device)


        # Create table asset
        table_pos = [0.0, 0.0, 1.0]
        table_thickness = 0.05
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, *[1.2, 1.2, table_thickness], table_opts)

        # Create table stand asset
        table_stand_height = 0.1
        table_stand_pos = [-0.5, 0.0, 1.0 + table_thickness / 2 + table_stand_height / 2]
        table_stand_opts = gymapi.AssetOptions()
        table_stand_opts.fix_base_link = True
        table_stand_asset = self.gym.create_box(self.sim, *[0.2, 0.2, table_stand_height], table_opts)

        self.cubeA_size = 0.050
        self.cubeB_size = 0.030

        # Create cubeA asset
        cubeA_opts = gymapi.AssetOptions()
        cubeA_asset = self.gym.create_box(self.sim, *([self.cubeA_size] * 3), cubeA_opts)
        cubeA_color = gymapi.Vec3(0.6, 0.1, 0.0)

        # Create cubeB asset
        cubeB_opts = gymapi.AssetOptions()
        cubeB_opts.disable_gravity = True
        cubeB_asset = self.gym.create_box(self.sim, *([self.cubeB_size] * 3), cubeB_opts)
        cubeB_color = gymapi.Vec3(0.0, 0.4, 0.1)

        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)

        print("num franka bodies: ", self.num_franka_bodies)
        print("num franka dofs: ", self.num_franka_dofs)

        # set franka dof properties
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        self._franka_effort_limits = []
        for i in range(9):
            franka_dof_props['driveMode'][i] = 3
            if self.physics_engine == gymapi.SIM_PHYSX:
                franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
                franka_dof_props['damping'][i] = franka_dof_damping[i]
                franka_dof_props['armature'][i] = franka_dof_armature[i]
                franka_dof_props['friction'][i] = franka_dof_friction[i]

            else:
                franka_dof_props['stiffness'][i] = 7000.0
                franka_dof_props['damping'][i] = 50.0

            self.franka_dof_lower_limits.append(franka_dof_props['lower'][i])
            self.franka_dof_upper_limits.append(franka_dof_props['upper'][i])
            self._franka_effort_limits.append(franka_dof_props['effort'][i])

        self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)

        # self._franka_effort_limits = to_torch(self._franka_effort_limits, device=self.device)
        # self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        # self.franka_dof_speed_scales[[7, 8]] = 0.1
        # franka_dof_props['effort'][7] = 200
        # franka_dof_props['effort'][8] = 200

        # Define start pose for franka
        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(-0.45, 0.0, 1.0 + table_thickness / 2 + table_stand_height)
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for table
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self._table_surface_pos = np.array(table_pos) + np.array([0, 0, table_thickness / 2])
        self.reward_settings["table_height"] = self._table_surface_pos[2]

        # Define start pose for table stand
        table_stand_start_pose = gymapi.Transform()
        table_stand_start_pose.p = gymapi.Vec3(*table_stand_pos)
        table_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for cubes (doesn't really matter since they're get overridden during reset() anyways)
        cubeA_start_pose = gymapi.Transform()
        cubeA_start_pose.p = gymapi.Vec3(-1.0, 0.0, 0.0)
        cubeA_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        cubeB_start_pose = gymapi.Transform()
        cubeB_start_pose.p = gymapi.Vec3(1.0, 0.0, 0.0)
        cubeB_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        max_agg_bodies = num_franka_bodies + 4     # 1 for table, table stand, cubeA, cubeB
        max_agg_shapes = num_franka_shapes + 4     # 1 for table, table stand, cubeA, cubeB

        self.frankas = []
        self.envs = []

        # Create environments
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: franka should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create franka
            # Potentially randomize start pose
            if self.franka_position_noise > 0:
                rand_xy = self.franka_position_noise * (-1. + np.random.rand(2) * 2.0)
                franka_start_pose.p = gymapi.Vec3(-0.45 + rand_xy[0], 0.0 + rand_xy[1],
                                                 1.0 + table_thickness / 2 + table_stand_height)
            if self.franka_rotation_noise > 0:
                rand_rot = torch.zeros(1, 3)
                rand_rot[:, -1] = self.franka_rotation_noise * (-1. + np.random.rand() * 2.0)
                new_quat = axisangle2quat(rand_rot).squeeze().numpy().tolist()
                franka_start_pose.r = gymapi.Quat(*new_quat)
            franka_actor = self.gym.create_actor(env_ptr, franka_asset, franka_start_pose, "franka", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create table
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)
            table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand",
                                                      i, 1, 0)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create cubes
            self._cubeA_id = self.gym.create_actor(env_ptr, cubeA_asset, cubeA_start_pose, "cubeA", i, 2, 0)
            self._cubeB_id = self.gym.create_actor(env_ptr, cubeB_asset, cubeB_start_pose, "cubeB", i+1, 4, 0)
            # Set colors
            self.gym.set_rigid_body_color(env_ptr, self._cubeA_id, 0, gymapi.MESH_VISUAL, cubeA_color)
            self.gym.set_rigid_body_color(env_ptr, self._cubeB_id, 0, gymapi.MESH_VISUAL, cubeB_color)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)

        # Setup init state buffer
        self._init_cubeA_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._init_cubeB_state = torch.zeros(self.num_envs, 13, device=self.device)

        # Setup data
        self.init_data()

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        franka_handle = 0
        self.handles = {
            "grip_site": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "right_gripper_ee"),
            "base_link": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "base_link"),
            "cubeA_body_handle": self.gym.find_actor_rigid_body_handle(self.envs[0], self._cubeA_id, "box"),
            "cubeB_body_handle": self.gym.find_actor_rigid_body_handle(self.envs[0], self._cubeB_id, "box"),
        }

        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]

        self._eef_state = self._rigid_body_state[:, self.handles["grip_site"], :]
        self._base_link_state = self._rigid_body_state[:, self.handles["base_link"], :]

        self._cubeA_state = self._root_state[:, self._cubeA_id, :]
        self._cubeB_state = self._root_state[:, self._cubeB_id, :]

        # Initialize states
        self.states.update({
            "cubeA_size": torch.ones_like(self._eef_state[:, 0]) * self.cubeA_size,
            "cubeB_size": torch.ones_like(self._eef_state[:, 0]) * self.cubeB_size,
            "previous_q": torch.zeros((self.num_envs, self.num_dofs)).to(self.device),

        })

        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)

        # Initialize control
        self._arm_control = self._effort_control[:, :7]
        self._gripper_control = self._pos_control[:, 7:9]

        # Initialize indices
        self._global_indices = torch.arange(self.num_envs * 5, dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)

    def _update_states(self):
        self.states.update({
            # Franka
            "q": self._q[:, :],
            # "q_gripper": self._q[:, -2:],
            "eef_pos": self._eef_state[:, :3],
            "eef_pos_normalized": self._eef_state[:, :3] - self._base_link_state[:, :3],

            "first_cube_pos" : self.current_cube_obs,
            "first_cube_pos_normalized": self.current_cube_obs - self._base_link_state[:, :3],

            "q_short" : self._q[:, :6],

            "eef_quat": self._eef_state[:, 3:7],
            "eef_vel": self._eef_state[:, 7:],

            "q_vel": (self._q[:, :] - self.states['previous_q']) / self.sim_params.dt,

            "targets": self._cubeA_state[:, :3] + torch.cat((self._cubeA_state[:, :2],
                                                             self.height_target), dim = 1)  - self._base_link_state[:, :3],
            "lower_limits": self.franka_dof_lower_limits,
            "upper_limits": self.franka_dof_upper_limits,
            # "eef_lf_pos": self._eef_lf_state[:, :3],
            # "eef_rf_pos": self._eef_rf_state[:, :3],
            # Cubes
            "cubeA_quat": self._cubeA_state[:, 3:7],
            "cubeA_pos": self._cubeA_state[:, :3],
            "cubeA_pos_normalized": self._cubeA_state[:, :3] - self._base_link_state[:, :3],

            "cubeA_pos_relative": self._cubeA_state[:, :3] - self._eef_state[:, :3],
            "cubeB_quat": self._cubeB_state[:, 3:7],
            "cubeB_pos": self._cubeB_state[:, :3],
            "cubeA_to_cubeB_pos": self._cubeB_state[:, :3] - self._cubeA_state[:, :3],

            "target_normalized": self.current_targets - self._base_link_state[:, :3],
            "target": self.current_targets,

        })

        # self.states.update({"eef_pos_normalized_torch": self.chain.forward_kinematics(self.states['q'][:, :5]).get_matrix()[:, :3, 3]})

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # Refresh states
        self._update_states()

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_franka_reward(
            self.reset_buf, self.progress_buf, self.actions, self.states, self.reward_settings, self.max_episode_length
        )

    def compute_observations(self):
        self._refresh()
        obs = []
        # print("HERE")
        # print(self.states['first_cube_pos_normalized'][:, :])
        obs +=  ["q_short", "target_normalized", "eef_pos_normalized"]
        # print(self.states['first_cube_pos_normalized'])
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)
        # print(self.obs_buf)
        maxs = {ob: torch.max(self.states[ob]).item() for ob in obs}
        return self.obs_buf


    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # Reset cubes, sampling cube B first, then A
        # if not self._i:
        # self._reset_init_cube_state(cube='B', env_ids=env_ids, check_valid=False)
        # self._reset_init_cube_state(cube='A', env_ids=env_ids, check_valid=False)
        # self._i = True

        self.height_target[env_ids] = torch.rand((len(env_ids), 1)).cuda() * 0.5 - 0.075

        # Write these new init states to the sim states
        # self._cubeA_state[env_ids] = self._init_cubeA_state[env_ids]
        # self._cubeB_state[env_ids] = self._init_cubeB_state[env_ids]

        # Reset agent
        reset_noise = torch.rand((len(env_ids), 9), device=self.device)
        # reset_noise[:, 7:] = 0.5
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0) +
            self.franka_dof_noise * 2.0 * (reset_noise - 0.5),
            self.franka_dof_lower_limits.unsqueeze(0), self.franka_dof_upper_limits)
        # pos[:, 7:] = 0.

        # Overwrite gripper init pos (no noise since these are always position controlled)
        # pos[:, -2:] = self.franka_default_dof_pos[-2:]

        init_sample_idx = torch.randint(low = 0, high = self.init_points.shape[0]-1, size=(len(env_ids),))
        init_q = self.init_points[init_sample_idx, :9]

        init_cube_pos = self.init_points[init_sample_idx, 9:12]
        init_cube_quat = self.init_points[init_sample_idx, 12:]


        position_to_init_cube = torch.cat((init_cube_pos, init_cube_quat), dim = -1)
        # position_to_init_cube = self.chain.forward_kinematics(pos[:, :7]).get_matrix()[:, :3, 3]

        self._reset_init_cube_state(cube='A', env_ids=env_ids, position_to_init_cube = position_to_init_cube, check_valid=False)
        self._reset_init_cube_state(cube='B', env_ids=env_ids, position_to_init_cube = position_to_init_cube, check_valid=False)

        self._cubeA_state[env_ids] = self._init_cubeA_state[env_ids]
        self._cubeB_state[env_ids] = self._init_cubeB_state[env_ids]

        # Reset the internal obs accordingly
        self._q[env_ids, :] = init_q

        self.actions[env_ids, :] = init_q[:, :6]
        # self.actions[env_ids, -1] = (init_q[:, 5] > 0).float()

        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])

        # Set any position control to the current position, and any vel / effort control to be 0
        # NOTE: Task takes care of actually propagating these controls in sim using the SimActions API
        self._pos_control[env_ids, :] = init_q
        self._effort_control[env_ids, :] = torch.zeros_like(pos)

        # Deploy updates
        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()
        # self.gym.set_dof_position_target_tensor_indexed(self.sim,
        #                                                 gymtorch.unwrap_tensor(self._pos_control),
        #                                                 gymtorch.unwrap_tensor(multi_env_ids_int32),
        #                                                 len(multi_env_ids_int32))
        # self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
        #                                                 gymtorch.unwrap_tensor(self._effort_control),
        #                                                 gymtorch.unwrap_tensor(multi_env_ids_int32),
        #                                                 len(multi_env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

        # Update cube states
        multi_env_ids_cubes_int32 = self._global_indices[env_ids, -2:].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0


    def _reset_init_cube_state(self, cube, env_ids, position_to_init_cube, check_valid=True):
        """
        Simple method to sample @cube's position based on self.startPositionNoise and self.startRotationNoise, and
        automaticlly reset the pose internally. Populates the appropriate self._init_cubeX_state

        If @check_valid is True, then this will also make sure that the sampled position is not in contact with the
        other cube.

        Args:
            cube(str): Which cube to sample location for. Either 'A' or 'B'
            env_ids (tensor or None): Specific environments to reset cube for
            check_valid (bool): Whether to make sure sampled position is collision-free with the other cube.
        """
        # If env_ids is None, we reset all the envs
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

        # Initialize buffer to hold sampled values
        num_resets = len(env_ids)
        sampled_cube_state = torch.zeros(num_resets, 13, device=self.device)

        # Get correct references depending on which one was selected
        if cube.lower() == 'a':
            this_cube_state_all = self._init_cubeA_state
            other_cube_state = self._init_cubeB_state[env_ids, :]
            cube_heights = self.states["cubeA_size"]
        elif cube.lower() == 'b':
            this_cube_state_all = self._init_cubeB_state
            other_cube_state = self._init_cubeA_state[env_ids, :]
            cube_heights = self.states["cubeA_size"]
        else:
            raise ValueError(f"Invalid cube specified, options are 'A' and 'B'; got: {cube}")

        # Minimum cube distance for guarenteed collision-free sampling is the sum of each cube's effective radius
        min_dists = (self.states["cubeA_size"] + self.states["cubeB_size"])[env_ids] * np.sqrt(2) / 2.0

        # We scale the min dist by 2 so that the cubes aren't too close together
        min_dists = min_dists * 2.0

        # Sampling is "centered" around middle of table
        centered_cube_xy_state = torch.tensor(self._table_surface_pos[:2], device=self.device, dtype=torch.float32)
        # centered_cube_xy_state[0] -= 0.05
        centered_cube_xy_state[1] -= 0.2


        # Set z value, which is fixed height
        sampled_cube_state[:, 2] = self._table_surface_pos[2] + cube_heights.squeeze(-1)[env_ids] / 2

        # Initialize rotation, which is no rotation (quat w = 1)
        sampled_cube_state[:, 6] = 1.0

        # If we're verifying valid sampling, we need to check and re-sample if any are not collision-free
        # We use a simple heuristic of checking based on cubes' radius to determine if a collision would occur
        if check_valid:
            success = False
            # Indexes corresponding to envs we're still actively sampling for
            active_idx = torch.arange(num_resets, device=self.device)
            num_active_idx = len(active_idx)
            for i in range(100):
                # Sample x y values
                sampled_cube_state[active_idx, :2] = centered_cube_xy_state + \
                                                     2.0 * self.start_position_noise * (
                                                             torch.rand_like(sampled_cube_state[active_idx, :2]) - 0.5)
                # Check if sampled values are valid
                cube_dist = torch.linalg.norm(sampled_cube_state[:, :2] - other_cube_state[:, :2], dim=-1)
                active_idx = torch.nonzero(cube_dist < min_dists, as_tuple=True)[0]
                num_active_idx = len(active_idx)
                # If active idx is empty, then all sampling is valid :D
                if num_active_idx == 0:
                    success = True
                    break
            # Make sure we succeeded at sampling
            assert success, "Sampling cube locations was unsuccessful! ):"
        else:
            # We just directly sample
            sampled_cube_state[:, :2] = centered_cube_xy_state.unsqueeze(0) + \
                                              2.0 * self.start_position_noise * (
                                                      torch.rand(num_resets, 2, device=self.device) - 0.5)

        # Sample rotation value
        if self.start_rotation_noise > 0:
            aa_rot = torch.zeros(num_resets, 3, device=self.device)
            aa_rot[:, 2] = 2.0 * self.start_rotation_noise * (torch.rand(num_resets, device=self.device) - 0.5)
            sampled_cube_state[:, 3:7] = quat_mul(axisangle2quat(aa_rot), sampled_cube_state[:, 3:7])

        # Lastly, set these sampled values as the new init state
        # this_cube_state_all[env_ids, :] = sampled_cube_state
        # position_to_init_cube[:, 2] -= 1
        this_cube_state_all[env_ids, :7] = position_to_init_cube[:, :]
        # this_cube_state_all[env_ids, 4:7] =
        # this_cube_state_all[env_ids, 0] -= 0.45
        # this_cube_state_all[env_ids, 2] += self._table_surface_pos[2] + 0.1
        self.current_targets[env_ids, 2] = self._table_surface_pos[2] + 0.025 #0.025
        self.current_targets[env_ids, :2] = centered_cube_xy_state.unsqueeze(0) + \
                                              2.0 * self.start_position_noise * (
                                                      torch.rand(num_resets, 2, device=self.device) - 0.5)
        if cube.lower() == 'a':
            self.current_cube_obs[env_ids] = sampled_cube_state[:, :3].clone()


        if cube.lower() == 'b':
            this_cube_state_all[env_ids, :3] = self.current_targets[env_ids, :3]
            #sampled_cube_state[:, :3] = self.current_targets[env_ids, :3]
            this_cube_state_all[env_ids, 3:7] = torch.tensor([0., 0. ,0. ,1.]).to(self.device)


    # def pre_physics_step(self, actions):
    #     # print(self.states['cubeA_pos_normalized'][-1])
    #     computed_actions = self.actions.clone() + (1/200) * actions * 20
    #     self.actions = computed_actions.clip(self.franka_dof_lower_limits, self.franka_dof_upper_limits)
    #     self._pos_control = self.actions


    def pre_physics_step(self, actions):
        # print(self.states['cubeA_pos_normalized'][-1])
        # print('____')
        # print(self._q[-1])
        # print('____')

        actual_actions = torch.zeros((self.actions.shape[0], 9), device = self.actions.device)
        computed_actions = self._pos_control[:, :5].clone() + (1/200) * actions[:, :-1] * 7.5

        last_action_positive = actions[:, -1] > 0
        last_action_negative = actions[:, -1] < 0

        actual_actions[:, :5] = computed_actions[:, :5].clone()

        actual_actions[last_action_positive, 5] = 0.87
        actual_actions[last_action_positive, 6] = 0.87
        actual_actions[last_action_positive, 7] = 0.87
        actual_actions[last_action_positive, 8] = 0.87

        actual_actions[last_action_negative, 5] = 0
        actual_actions[last_action_negative, 6] = 0
        actual_actions[last_action_negative, 7] = 0
        actual_actions[last_action_negative, 8] = 0

        actual_actions = actual_actions.clip(self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        self._pos_control = actual_actions

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            # Grab relevant states to visualize
            eef_pos = self.states["eef_pos"]
            eef_rot = self.states["eef_quat"]
            cubeA_pos = self.states["cubeA_pos"]
            cubeA_rot = self.states["cubeA_quat"]
            cubeB_pos = self.states["cubeB_pos"]
            cubeB_rot = self.states["cubeB_quat"]

            # Plot visualizations
            for i in range(self.num_envs):
                for pos, rot in zip((eef_pos, cubeA_pos, cubeB_pos), (eef_rot, cubeA_rot, cubeB_rot)):
                    px = (pos[i] + quat_apply(rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                    py = (pos[i] + quat_apply(rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                    pz = (pos[i] + quat_apply(rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                    p0 = pos[i].cpu().numpy()
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

#####################################################################
###=========================jit functions=========================###
#####################################################################


# @torch.jit.script
# def compute_franka_reward(
#     reset_buf, progress_buf, actions, states, reward_settings, max_episode_length
# ):
#     # type: (Tensor, Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float) -> Tuple[Tensor, Tensor]
#     d = torch.norm(states["targets"] - states["eef_pos_normalized"], dim=-1)
#     # print(d)
#     open_hands = torch.norm(states['q'][:, -4:], dim = -1)
#     dist_reward = 1 - torch.tanh((d))
#     rewards =  open_hands #+ dist_reward
#     reset_buf = torch.where((progress_buf >= max_episode_length - 1), torch.ones_like(reset_buf), reset_buf)
#     return rewards, reset_buf


#
# @torch.jit.script
# def compute_franka_reward(
#     reset_buf, progress_buf, actions, states, reward_settings, max_episode_length
# ):
#     # type: (Tensor, Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float) -> Tuple[Tensor, Tensor]
#
#     # Compute per-env physical parameters
#     target_height = states["cubeA_size"] / 2.0
#     # cubeA_size = states["cubeA_size"]
#     # cube1_size = states["cube1_size"]
#
#     d = torch.norm(states["cubeA_pos"] - states["eef_pos"], dim=-1)
#     # print(d)
#     close_to_goal = d < 0.01
#
#     dist_reward = 1 - torch.tanh((5 * d))
#     table_h = reward_settings["table_height"]
#     cubeA_height = states["cubeA_pos"][:, 2]  - table_h
#     cubeA_lifted = (cubeA_height) > 0.06
#
#     # positive_torque = states['mean_torque'][:, -1] > 2 #works with mean torques
#     # gripper_stuck = torch.abs((states['q'][:, -1] - states['previous_env_dt_q'].squeeze(1))) < 0.01
#     # contact_reward = positive_torque * gripper_stuck * close_to_goal
#     # contact_reward = positive_torque * gripper_stuck * close_to_goal
#     # from_side = dist_reward * torch.exp(- from_side - from_side_b)
#     #
#     # dist_from_target = torch.exp(-5*torch.norm(states['target_normalized'] - states['cubeA_pos_normalized'], dim = -1))
#     # bad_coordinate = ((states['cartesian_targets'][:, 2] + states['gripper_pos_normalized'][:, 2]) < 0.03).long()
#     # print(states['target_normalized'])
#
#     # no_rotation = torch.abs(states['q'][:, -2])
#     # obstacle_fell = states['cube1_first_pos'][:, 2] < 0.10.
#     close_hand = torch.exp(-torch.norm(states['q'][:, -4:], dim = -1))
#     open_hand = torch.norm(states['q'][:, -4:], dim = -1)
#
#     rewards = 1 * dist_reward + 5 * cubeA_lifted + close_to_goal * close_hand  #+ close_to_goal * #+ open_hands#+ 1 * contact_reward + 5 * contact_reward * dist_from_target #- obstacle_fell.long()  #- 5 * bad_coordinate#+ from_side #+ #+ torque_reward + action_diff #+ action_diff #no_rotation
#
#     reset_buf = torch.where((progress_buf >= max_episode_length - 1), torch.ones_like(reset_buf), reset_buf)
#     cube_fall = states['cubeA_pos'][:, 2] < 1
#     reset_buf |= cube_fall
#     reset_buf |= (cubeA_height) > 0.5
#
#     # reset_buf |= (states['cubeA_pos'][:, 0] < -0.40)
#     # reset_buf |= (states['cubeA_pos'][:, 0] > 0.15)
#     # reset_buf |= (states['cubeA_pos'][:, 1] < -0.6)
#     # reset_buf |= (states['cubeA_pos'][:, 1] > 0.6)
#
#     # reset_buf |= torch.any((states['q'] > states['upper_limits']) | (states['q'] < states['lower_limits']), dim=1)
#
#     # success = states['cubeA_pos'][:, 2] > 1.06
#     # reset_buf |= ((states['cube1_pos'][:, 2] < states['object_heights'][:, 1]) & (states['cube1_first_pos'][:, 2] > 0) )#/ 2
#     # reset_buf |= ((states['cube2_pos'][:, 2] < states['object_heights'][:, 2]) & (states['cube2_first_pos'][:, 2] > 0) )#/ 2
#     # reset_buf |= ((states['cube3_pos'][:, 2] < states['object_heights'][:, 3]) & (states['cube3_first_pos'][:, 2] > 0) )#/ 2
#     # reset_buf |= ((states['cube4_pos'][:, 2] < states['object_heights'][:, 4]) & (states['cube4_first_pos'][:, 2] > 0) )#/ 2
#     # reset_buf |= ((states['cube5_pos'][:, 2] < states['object_heights'][:, 5]) & (states['cube5_first_pos'][:, 2] > 0) )#/ 2
#     # reset_buf |= ((states['cube6_pos'][:, 2] < states['object_heights'][:, 6]) & (states['cube6_first_pos'][:, 2] > 0) )#/ 2
#     # reset_buf |= ((states['cube7_pos'][:, 2] < states['object_heights'][:, 7]) & (states['cube7_first_pos'][:, 2] > 0) )#/ 2
#     # reset_buf |= ((states['cube8_pos'][:, 2] < states['object_heights'][:, 8]) & (states['cube8_first_pos'][:, 2] > 0) )#/ 2
#     # reset_buf |= ((states['cube9_pos'][:, 2] < states['object_heights'][:, 9]) & (states['cube9_first_pos'][:, 2] > 0) )#/ 2
#
#     # print(states['cube1_pos'][:, 2])
#     # print(states['cube1_first_pos'][:, 2])
#     # print(torch.norm(states['eef_vel'][:, :3], dim = -1))
#     #TODO reset on too high torques
#     # print(states['mean_torque'].mean(dim = 0))
#
#     return rewards, reset_buf




# @torch.jit.script
# def compute_franka_reward(
#     reset_buf, progress_buf, actions, states, reward_settings, max_episode_length
# ):
#     # type: (Tensor, Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float) -> Tuple[Tensor, Tensor]
#
#     # Compute per-env physical parameters
#     target_height = states["cubeA_size"] / 2.0
#     # cubeA_size = states["cubeA_size"]
#     # cube1_size = states["cube1_size"]
#
#     d = torch.norm(states["cubeA_pos"] - states["eef_pos"], dim=-1)
#     # print(d)
#     close_to_goal = d < 0.01
#
#     dist_reward = 1 - torch.tanh((5 * d))
#     table_h = reward_settings["table_height"]
#     cubeA_height = states["cubeA_pos"][:, 2]  - table_h
#     cubeA_lifted = (cubeA_height) > 0.06
#
#     # positive_torque = states['mean_torque'][:, -1] > 2 #works with mean torques
#     # gripper_stuck = torch.abs((states['q'][:, -1] - states['previous_env_dt_q'].squeeze(1))) < 0.01
#     # contact_reward = positive_torque * gripper_stuck * close_to_goal
#     # contact_reward = positive_torque * gripper_stuck * close_to_goal
#     # from_side = dist_reward * torch.exp(- from_side - from_side_b)
#     #
#     # dist_from_target = torch.exp(-5*torch.norm(states['target_normalized'] - states['cubeA_pos_normalized'], dim = -1))
#     # bad_coordinate = ((states['cartesian_targets'][:, 2] + states['gripper_pos_normalized'][:, 2]) < 0.03).long()
#     # print(states['target_normalized'])
#
#     # no_rotation = torch.abs(states['q'][:, -2])
#     # obstacle_fell = states['cube1_first_pos'][:, 2] < 0.10.
#     close_hand = torch.exp(-torch.norm(states['q'][:, -4:], dim = -1))
#     open_hand = torch.norm(states['q'][:, -4:], dim = -1)
#
#     rewards = 1 * dist_reward + 5 * cubeA_lifted + close_to_goal * close_hand  #+ close_to_goal * #+ open_hands#+ 1 * contact_reward + 5 * contact_reward * dist_from_target #- obstacle_fell.long()  #- 5 * bad_coordinate#+ from_side #+ #+ torque_reward + action_diff #+ action_diff #no_rotation
#
#     reset_buf = torch.where((progress_buf >= max_episode_length - 1), torch.ones_like(reset_buf), reset_buf)
#     cube_fall = states['cubeA_pos'][:, 2] < 1
#     reset_buf |= cube_fall
#     reset_buf |= (cubeA_height) > 0.5
#
#     # reset_buf |= (states['cubeA_pos'][:, 0] < -0.40)
#     # reset_buf |= (states['cubeA_pos'][:, 0] > 0.15)
#     # reset_buf |= (states['cubeA_pos'][:, 1] < -0.6)
#     # reset_buf |= (states['cubeA_pos'][:, 1] > 0.6)
#
#     # reset_buf |= torch.any((states['q'] > states['upper_limits']) | (states['q'] < states['lower_limits']), dim=1)
#
#     # success = states['cubeA_pos'][:, 2] > 1.06
#     # reset_buf |= ((states['cube1_pos'][:, 2] < states['object_heights'][:, 1]) & (states['cube1_first_pos'][:, 2] > 0) )#/ 2
#     # reset_buf |= ((states['cube2_pos'][:, 2] < states['object_heights'][:, 2]) & (states['cube2_first_pos'][:, 2] > 0) )#/ 2
#     # reset_buf |= ((states['cube3_pos'][:, 2] < states['object_heights'][:, 3]) & (states['cube3_first_pos'][:, 2] > 0) )#/ 2
#     # reset_buf |= ((states['cube4_pos'][:, 2] < states['object_heights'][:, 4]) & (states['cube4_first_pos'][:, 2] > 0) )#/ 2
#     # reset_buf |= ((states['cube5_pos'][:, 2] < states['object_heights'][:, 5]) & (states['cube5_first_pos'][:, 2] > 0) )#/ 2
#     # reset_buf |= ((states['cube6_pos'][:, 2] < states['object_heights'][:, 6]) & (states['cube6_first_pos'][:, 2] > 0) )#/ 2
#     # reset_buf |= ((states['cube7_pos'][:, 2] < states['object_heights'][:, 7]) & (states['cube7_first_pos'][:, 2] > 0) )#/ 2
#     # reset_buf |= ((states['cube8_pos'][:, 2] < states['object_heights'][:, 8]) & (states['cube8_first_pos'][:, 2] > 0) )#/ 2
#     # reset_buf |= ((states['cube9_pos'][:, 2] < states['object_heights'][:, 9]) & (states['cube9_first_pos'][:, 2] > 0) )#/ 2
#
#     # print(states['cube1_pos'][:, 2])
#     # print(states['cube1_first_pos'][:, 2])
#     # print(torch.norm(states['eef_vel'][:, :3], dim = -1))
#     #TODO reset on too high torques
#     # print(states['mean_torque'].mean(dim = 0))
#
#     return rewards, reset_buf


#@torch.jit.script
def compute_franka_reward(
    reset_buf, progress_buf, actions, states, reward_settings, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float) -> Tuple[Tensor, Tensor]

    # Compute per-env physical parameters
    target_height = states["cubeB_size"] + states["cubeA_size"] / 2.0
    cubeA_size = states["cubeA_size"]
    cubeB_size = states["cubeB_size"]

    result = my_quat_rotate(states['eef_quat'], torch.tensor([0., 0., 1.], device=states['eef_quat'].device).repeat(
        states['eef_quat'].shape[0], 1))
    result_b = my_quat_rotate(states['eef_quat'], torch.tensor([1., 0., 0.], device=states['eef_quat'].device).repeat(
        states['eef_quat'].shape[0], 1))

    from_side = torch.sum(result * torch.tensor([0., 0., 1.], device=result.device), dim=-1).abs()
    from_side_b = torch.sum(result_b * torch.tensor([0., 0., 1.], device=result.device), dim=-1).abs()

    d = torch.norm(states["cubeA_pos"] - states["eef_pos"], dim=-1)

    close_to_goal = d < 0.05

    midpoint = (states['first_cube_pos'] + states['target']) / 2
    # torque_reward = torch.exp(-0.05 * torch.norm(states['mean_torque'], dim = -1))
    # action_diff = torch.exp(-torch.norm(actions - states['previous_action'], dim = -1))
    # eef_vel_reg = torch.exp(-torch.norm((states["eef_vel"]) , dim = -1))


    # # final reward for stacking successfully (only if cubeA is close to target height and corresponding location, and gripper is not grasping)
    # cubeA_align_cubeB = (torch.norm(states["cubeA_to_cubeB_pos"][:, :2], dim=-1) < 0.02)
    # cubeA_on_cubeB = torch.abs(cubeA_height - target_height) < 0.02
    # gripper_away_from_cubeA = (d > 0.04)
    #

    dist_reward = 1 - torch.tanh((5 * d))
    table_h = reward_settings["table_height"]
    cubeA_height = states["cubeA_pos"][:, 2]  - table_h
    cubeA_lifted = ((cubeA_height) > 0.02) + 3*((cubeA_height) > 0.04) + 6*((cubeA_height) > 0.1) #(cubeA_height) > 0.1  #0.05 is train2 0.03 is train

    d_ab = torch.norm(states["cubeA_pos"] - states["target"], dim=-1)
    # print(states['target'])
    align_reward = (1 - torch.tanh(2 * d_ab))
    align_reward2 = (1 - torch.tanh(5*d_ab))

    # print('D ab', d_ab)
    close_to_target = d_ab < 0.03

    dist_reward[close_to_target] = align_reward[close_to_target]

    gripper_away_from_cubeA = (d > 0.05)
    success = torch.logical_and(close_to_target, gripper_away_from_cubeA)
    open_hand = -torch.exp(-torch.norm(states['q'][:, -4:], dim = -1))
    if success.shape[0] < 120:
        print(torch.sum(success.long())/success.shape[0])
    # print(d_ab, "open hand ", open_hand[-1])
    # positive_torque = states['mean_torque'][:, -1] > 2
    # gripper_stuck = torch.abs((states['q'][:, -1] - states['previous_env_dt_q'].squeeze(1))) < 0.01
    # contact_reward = positive_torque * gripper_stuck * close_to_goal
    # from_side = dist_reward * torch.exp(- from_side - from_side_b)

    # no_rotation = torch.abs(states['q'][:, -2])
    # print("Dist ", dist_reward, " scaled ", 1 * dist_reward)
    # print("Lifted ", cubeA_lifted, " scaled ", 0.5 * cubeA_lifted)
    # print("Align ", align_reward, " scaled ", 3 * align_reward)
    # print("Success", success.long(), " scaled ", 8 * success.long())
    # print(d_ab)
    # rewards = align_reward

    rewards = torch.where(close_to_target,
                          200*success + 2*torch.tanh(torch.norm(states["eef_pos"] - states["target"], dim=-1)) + 2*torch.tanh(torch.norm(states["eef_pos"] - states["cubeA_pos"], dim=-1))+ 10 * align_reward2,
                          0 * dist_reward + 1 * ((cubeA_height) > 0.05) *5* align_reward)#+ 20*open_hand * close_to_target - 5*gripper_away_from_cubeA*~close_to_target- 10*gripper_away_from_cubeA*open_hand)  #+ #+ torque_reward + action_diff #+ action_diff #no_rotation

    #rewards = ((cubeA_height) > 0.02) + 6*((cubeA_height) > 0.04) + 15*((cubeA_height) > 0.1)


    reset_buf = torch.where((progress_buf >= max_episode_length - 1), torch.ones_like(reset_buf), reset_buf)
    cube_fall = states['cubeA_pos'][:, 2] < 1
    reset_buf |= cube_fall
    reset_buf |= (cubeA_height) > 0.5

    # reset_buf |= (states['cubeA_pos'][:, 0] < -0.40)
    # reset_buf |= (states['cubeA_pos'][:, 0] > 0.15)
    # reset_buf |= (states['cubeA_pos'][:, 1] < -0.6)
    # reset_buf |= (states['cubeA_pos'][:, 1] > 0.6)

    # reset_buf |= torch.any(torch.logical_or(states['q'][:, :-4] > states['upper_limits'][:-4] + 1e-3, states['q'][:, :-4] < states['lower_limits'][:-4] - 1e-3), dim = 1)
    # if torch.any(torch.any((states['q'] > (states['upper_limits'] + 1e-5)) | (states['q'] < (states['lower_limits'] - 1e-5)), dim=1)):
    #     print('f')
    # reset_buf |= torch.norm(states['eef_vel'][:, :3], dim = -1) > .3
    # print(torch.norm(states['eef_vel'][:, :3], dim = -1))
    #TODO reset on too high torques
    # print(states['mean_torque'].mean(dim = 0))
    return rewards, reset_buf


# @torch.jit.script
# def compute_franka_reward(
#     reset_buf, progress_buf, actions, states, reward_settings, max_episode_length
# ):
#     # type: (Tensor, Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float) -> Tuple[Tensor, Tensor]

#     # Compute per-env physical parameters
#     target_height = states["cubeB_size"] + states["cubeA_size"] / 2.0
#     cubeA_size = states["cubeA_size"]
#     cubeB_size = states["cubeB_size"]

#     result = my_quat_rotate(states['eef_quat'], torch.tensor([0., 0., 1.], device=states['eef_quat'].device).repeat(
#         states['eef_quat'].shape[0], 1))
#     result_b = my_quat_rotate(states['eef_quat'], torch.tensor([1., 0., 0.], device=states['eef_quat'].device).repeat(
#         states['eef_quat'].shape[0], 1))

#     from_side = torch.sum(result * torch.tensor([0., 0., 1.], device=result.device), dim=-1).abs()
#     from_side_b = torch.sum(result_b * torch.tensor([0., 0., 1.], device=result.device), dim=-1).abs()

#     d = torch.norm(states["cubeA_pos"] - states["eef_pos"], dim=-1)

#     close_to_goal = d < 0.05
#     # torque_reward = torch.exp(-0.05 * torch.norm(states['mean_torque'], dim = -1))
#     # action_diff = torch.exp(-torch.norm(actions - states['previous_action'], dim = -1))
#     # eef_vel_reg = torch.exp(-torch.norm((states["eef_vel"]) , dim = -1))


#     # # final reward for stacking successfully (only if cubeA is close to target height and corresponding location, and gripper is not grasping)
#     # cubeA_align_cubeB = (torch.norm(states["cubeA_to_cubeB_pos"][:, :2], dim=-1) < 0.02)
#     # cubeA_on_cubeB = torch.abs(cubeA_height - target_height) < 0.02
#     # gripper_away_from_cubeA = (d > 0.04)
#     #

#     dist_reward = 1 - torch.tanh((5 * d))
#     table_h = reward_settings["table_height"]
#     cubeA_height = states["cubeA_pos"][:, 2]  - table_h
#     cubeA_lifted = (cubeA_height) > 0.05#0.05 is train2 0.03 is train

#     d_ab = torch.norm(states["cubeA_pos"] - states["target"], dim=-1)
#     # print(states['target'])
#     align_reward = (1 - torch.tanh(10 * d_ab))

#     # print('D ab', d_ab)
#     close_to_target = d_ab < 0.03

#     dist_reward[close_to_target] = align_reward[close_to_target]

#     gripper_away_from_cubeA = (d > 0.05)
#     success = torch.logical_and(close_to_target, gripper_away_from_cubeA)
#     open_hand = -torch.exp(-torch.norm(states['q'][:, -4:], dim = -1))
#     #print(torch.sum(success.long())/success.shape[0])
#     # print(d_ab, "open hand ", open_hand[-1])
#     # positive_torque = states['mean_torque'][:, -1] > 2
#     # gripper_stuck = torch.abs((states['q'][:, -1] - states['previous_env_dt_q'].squeeze(1))) < 0.01
#     # contact_reward = positive_torque * gripper_stuck * close_to_goal
#     # from_side = dist_reward * torch.exp(- from_side - from_side_b)

#     # no_rotation = torch.abs(states['q'][:, -2])
#     # print("Dist ", dist_reward, " scaled ", 1 * dist_reward)
#     # print("Lifted ", cubeA_lifted, " scaled ", 0.5 * cubeA_lifted)
#     # print("Align ", align_reward, " scaled ", 3 * align_reward)
#     # print("Success", success.long(), " scaled ", 8 * success.long())
#     # print(d_ab)
#     # rewards = align_reward

#     rewards = torch.where(success,
#                           success.long() * 16,
#                           1 * dist_reward + 1 * cubeA_lifted + 6 * align_reward + open_hand * close_to_target)  #+ #+ torque_reward + action_diff #+ action_diff #no_rotation


#     reset_buf = torch.where((progress_buf >= max_episode_length - 1), torch.ones_like(reset_buf), reset_buf)
#     cube_fall = states['cubeA_pos'][:, 2] < 1
#     reset_buf |= cube_fall
#     reset_buf |= (cubeA_height) > 0.5

#     # reset_buf |= (states['cubeA_pos'][:, 0] < -0.40)
#     # reset_buf |= (states['cubeA_pos'][:, 0] > 0.15)
#     # reset_buf |= (states['cubeA_pos'][:, 1] < -0.6)
#     # reset_buf |= (states['cubeA_pos'][:, 1] > 0.6)

#     # reset_buf |= torch.any(torch.logical_or(states['q'][:, :-4] > states['upper_limits'][:-4] + 1e-3, states['q'][:, :-4] < states['lower_limits'][:-4] - 1e-3), dim = 1)
#     # if torch.any(torch.any((states['q'] > (states['upper_limits'] + 1e-5)) | (states['q'] < (states['lower_limits'] - 1e-5)), dim=1)):
#     #     print('f')
#     # reset_buf |= torch.norm(states['eef_vel'][:, :3], dim = -1) > .3
#     # print(torch.norm(states['eef_vel'][:, :3], dim = -1))
#     #TODO reset on too high torques
#     # print(states['mean_torque'].mean(dim = 0))
#     return rewards, reset_buf