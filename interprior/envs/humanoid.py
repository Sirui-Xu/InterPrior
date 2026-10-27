# Copyright (c) 2018-2022, NVIDIA Corporation
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

from enum import Enum
import numpy as np
import torch
import glob
import os
import random
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
import torch.utils
from . import torch_utils
import torch.nn.functional as F
from .base_task import BaseTask
import trimesh
from ..motions import load_motion_tensor

PERTURB_OBJS = [["small", 60]]


class SMPLHumanoid(BaseTask):

    def __init__(
        self, cfg, sim_params, physics_engine, device_type, device_id, headless
    ):
        self.cfg = cfg
        self.inference_mode = bool(cfg["env"].get("inference_mode", False))
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self._pd_control = self.cfg["env"]["pdControl"]
        self.power_scale = self.cfg["env"]["powerScale"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]
        self.ref_hoi_obs_size = (
            324 + len(self.cfg["env"]["keyBodies"]) * 9 + 52 + 3 + 52 * 4 + 52 * 3
        )
        self.ref_contact_obs_size = len(self.cfg["env"]["contactBodies"]) * 3
        self._local_root_obs = self.cfg["env"]["localRootObs"]
        self._root_height_obs = self.cfg["env"].get("rootHeightObs", True)
        self._enable_early_termination = self.cfg["env"]["enableEarlyTermination"]
        key_bodies = self.cfg["env"]["keyBodies"]
        self._setup_character_props(key_bodies)
        self.cfg["env"]["numObservations"] = self.get_obs_size()
        self.cfg["env"]["numActions"] = self.get_action_size()
        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless
        super().__init__(cfg=self.cfg)
        self.dt = self.control_freq_inv * sim_params.dt
        self.common_step_counter = 0
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(
            self.num_envs, self.num_dof
        )
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self._root_states = gymtorch.wrap_tensor(actor_root_state)
        num_actors = self.get_num_actors_per_env()
        self._humanoid_root_states = self._root_states.view(
            self.num_envs, num_actors, actor_root_state.shape[-1]
        )[..., 0, :]
        self._initial_humanoid_root_states = self._humanoid_root_states.clone()
        self._initial_humanoid_root_states[:] = 0
        self._initial_humanoid_root_states[:, 6:7] = 1
        self._humanoid_actor_ids = num_actors * torch.arange(
            self.num_envs, device=self.device, dtype=torch.int32
        )
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        dofs_per_env = self._dof_state.shape[0] // self.num_envs
        self._dof_pos = self._dof_state.view(self.num_envs, dofs_per_env, 2)[
            ..., : self.num_dof, 0
        ]
        self._dof_vel = self._dof_state.view(self.num_envs, dofs_per_env, 2)[
            ..., : self.num_dof, 1
        ]
        self._initial_dof_pos = torch.zeros_like(
            self._dof_pos, device=self.device, dtype=torch.float
        )
        self._initial_dof_vel = torch.zeros_like(
            self._dof_vel, device=self.device, dtype=torch.float
        )
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        rigid_body_state_reshaped = self._rigid_body_state.view(
            self.num_envs, bodies_per_env, 13
        )
        self._rigid_body_pos = rigid_body_state_reshaped[..., : self.num_bodies, 0:3]
        self._rigid_body_rot = rigid_body_state_reshaped[..., : self.num_bodies, 3:7]
        self._rigid_body_vel = rigid_body_state_reshaped[..., : self.num_bodies, 7:10]
        self._rigid_body_ang_vel = rigid_body_state_reshaped[
            ..., : self.num_bodies, 10:13
        ]
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._contact_forces = contact_force_tensor.view(
            self.num_envs, bodies_per_env, 3
        )[..., : self.num_bodies, :]
        self._terminate_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self._build_termination_heights()
        contact_bodies = self.cfg["env"]["contactBodies"]
        self._key_body_ids = self._build_key_body_ids_tensor(key_bodies)
        self._contact_body_ids = self._build_contact_body_ids_tensor(contact_bodies)
        if self.viewer is not None:
            self._init_camera()
        return

    def _generate_fall_states(self):
        print("#################### Generating Fall State ####################")
        max_steps = 200
        env_ids = to_torch(
            np.arange(self.num_envs), device=self.device, dtype=torch.long
        )
        self._reset_envs(env_ids)
        rand_actions = np.random.uniform(-0.5, 0.5, size=[self.num_envs, 153])
        rand_actions = to_torch(rand_actions, device=self.device)
        self.pre_physics_step(rand_actions)
        for i in range(max_steps):
            self.render()
            self.gym.simulate(self.sim)
        self._refresh_sim_tensors()
        _fall_root_states = self._humanoid_root_states.clone()
        _fall_root_states[:, 7:13] = 0
        _fall_dof_pos = self._dof_pos.clone()
        _fall_dof_vel = torch.zeros_like(
            self._dof_vel, device=self.device, dtype=torch.float
        )
        _fall_target_states = self._target_states.clone()
        _fall_target_states[:, 7:13] = 0
        hoi_ref = torch.cat(
            (
                _fall_root_states[:, 0:3],
                _fall_root_states[:, 3:7],
                _fall_dof_pos,
                _fall_dof_vel,
                _fall_root_states[:, 7:10],
                _fall_root_states[:, 10:13],
                _fall_target_states[env_ids, :3],
                _fall_target_states[env_ids, 3:7],
                _fall_target_states[env_ids, 7:10],
                _fall_target_states[env_ids, 10:13],
            ),
            dim=-1,
        )
        return hoi_ref

    def get_obs_size(self):
        return self._num_obs

    def get_action_size(self):
        return self._num_actions

    def get_num_actors_per_env(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        return num_actors

    def create_sim(self):
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, "z")
        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self._create_ground_plane()
        self._create_envs(
            self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs))
        )
        return

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = to_torch(
                np.arange(self.num_envs), device=self.device, dtype=torch.long
            )
        self._reset_envs(env_ids)
        return

    def set_char_color(self, col, env_ids):
        for env_id in env_ids:
            env_ptr = self.envs[env_id]
            handle = self.humanoid_handles[env_id]
            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr,
                    handle,
                    j,
                    gymapi.MESH_VISUAL,
                    gymapi.Vec3(col[0], col[1], col[2]),
                )
        return

    def _reset_envs(self, env_ids):
        if len(env_ids) > 0:
            self._reset_actors(env_ids)
            self._reset_env_tensors(env_ids)
            self._refresh_sim_tensors()
            self._compute_observations(env_ids)
            self._compute_observations_raw(env_ids)
        return

    def _reset_env_tensors(self, env_ids):
        env_ids_int32 = self._humanoid_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        self.reset_buf_count[env_ids] = 0
        return

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)
        return

    def _setup_character_props(self, key_bodies):
        asset_file = self.robot_type
        num_key_bodies = len(key_bodies)
        self._dof_obs_size = 51 * 3
        self._num_actions = 51 * 3
        obj_obs_size = 15
        self._num_obs = self.cfg["env"]["numObs"]
        return

    def _build_termination_heights(self):
        self._termination_heights = 0.3
        self._termination_heights = to_torch(
            self._termination_heights, device=self.device
        )
        return

    def get_num_amp_obs(self):
        return self.ref_hoi_obs_size

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        asset_root = self.cfg["env"]["asset"]["assetRoot"]
        asset_file = self.robot_type
        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)
        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        humanoid_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )
        self.num_humanoid_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_humanoid_shapes = self.gym.get_asset_rigid_shape_count(humanoid_asset)
        actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]
        right_foot_idx = self.gym.find_asset_rigid_body_index(
            humanoid_asset, "right_foot"
        )
        left_foot_idx = self.gym.find_asset_rigid_body_index(
            humanoid_asset, "left_foot"
        )
        sensor_pose = gymapi.Transform()
        self.gym.create_asset_force_sensor(humanoid_asset, right_foot_idx, sensor_pose)
        self.gym.create_asset_force_sensor(humanoid_asset, left_foot_idx, sensor_pose)
        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = to_torch(motor_efforts, device=self.device)
        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_joints = self.gym.get_asset_joint_count(humanoid_asset)
        self.humanoid_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []
        max_agg_bodies = (
            self.num_humanoid_bodies
            + 2
            + getattr(getattr(self, "goal_spheres", None), "body_count", 0)
        )
        max_agg_shapes = self.num_humanoid_shapes + 65
        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
            self._build_env(i, env_ptr, humanoid_asset)
            self.gym.end_aggregate(env_ptr)
            self.envs.append(env_ptr)
        dof_prop = self.gym.get_actor_dof_properties(
            self.envs[0], self.humanoid_handles[0]
        )
        for j in range(self.num_dof):
            if dof_prop["lower"][j] > dof_prop["upper"][j]:
                self.dof_limits_lower.append(dof_prop["upper"][j])
                self.dof_limits_upper.append(dof_prop["lower"][j])
            else:
                self.dof_limits_lower.append(dof_prop["lower"][j])
                self.dof_limits_upper.append(dof_prop["upper"][j])
        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)
        if self._pd_control:
            self._build_pd_action_offset_scale()
        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        col_group = env_id
        col_filter = self._get_humanoid_collision_filter()
        segmentation_id = 0
        start_pose = gymapi.Transform()
        asset_file = self.robot_type
        char_h = 0.89
        start_pose.p = gymapi.Vec3(*get_axis_params(char_h, self.up_axis_idx))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        humanoid_handle = self.gym.create_actor(
            env_ptr,
            humanoid_asset,
            start_pose,
            "humanoid",
            col_group,
            col_filter,
            segmentation_id,
        )
        self.gym.enable_actor_dof_force_sensors(env_ptr, humanoid_handle)
        for j in range(self.num_bodies):
            self.gym.set_rigid_body_color(
                env_ptr,
                humanoid_handle,
                j,
                gymapi.MESH_VISUAL,
                gymapi.Vec3(0.75, 0.54, 0.3),
            )
        if self._pd_control:
            dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
            dof_prop["driveMode"] = gymapi.DOF_MODE_POS
            self.gym.set_actor_dof_properties(env_ptr, humanoid_handle, dof_prop)
        self.humanoid_handles.append(humanoid_handle)
        return

    def _build_pd_action_offset_scale(self):
        lim_low = self.dof_limits_lower.cpu().numpy()
        lim_high = self.dof_limits_upper.cpu().numpy()
        self._pd_action_offset = 0.5 * (lim_high + lim_low)
        self._pd_action_scale = 0.5 * (lim_high - lim_low)
        self._pd_action_offset = to_torch(self._pd_action_offset, device=self.device)
        self._pd_action_scale = to_torch(self._pd_action_scale, device=self.device)
        self._pd_action_scale[self._pd_action_offset.abs() > 0.1] = (
            self._pd_action_scale[0].clone()
        )
        self._pd_action_offset[self._pd_action_offset.abs() > 0.1] = 0
        self._L_knee_dof_idx = 5
        self._R_knee_dof_idx = 17
        self._pd_action_scale[self._L_knee_dof_idx] = 5
        self._pd_action_scale[self._R_knee_dof_idx] = 5
        return

    def _get_humanoid_collision_filter(self):
        return 1

    def _compute_reward(self, actions):
        hist_dof_vel = self._hist_obs[:, 160 : 160 + 51 * 3]
        local_vel = (
            self._curr_obs[:, 160 : 160 + 51 * 3] - hist_dof_vel
        ) * self.fps_data
        dof_diffacc = (
            local_vel.view(-1, 51 * 3)
            * (self.progress_buf - self.start_times > 2).float().unsqueeze(dim=-1)
        ).clone()
        hist_obj_vel = self._hist_obs[:, 320:323]
        obj_diffacc = (self._curr_obs[:, 320:323] - hist_obj_vel) * self.fps_data
        obj_diffacc = obj_diffacc * (
            self.progress_buf - self.start_times > 2
        ).float().unsqueeze(dim=-1)
        hist_obj_rot_vel = self._hist_obs[:, 323:326]
        local_vel = (self._curr_obs[:, 323:326] - hist_obj_rot_vel) * self.fps_data
        obj_rot_diffacc = local_vel.view(-1, 3) * (
            self.progress_buf - self.start_times > 2
        ).float().unsqueeze(dim=-1)
        ts = self.progress_buf_ref.clone()
        next_ts = torch.clamp(
            ts + self.long_term_t,
            min=self.max_episode_length[self.data_id]
            - self.max_episode_length[self.data_id],
            max=self.max_episode_length[self.data_id] - 1,
        )
        if self.inference_mode:
            goal_ref_obs = self.hoi_data[self.data_id, next_ts].clone()
        else:
            goal_ref_obs = self.hoi_data_envs[
                torch.arange(self.num_envs), next_ts
            ].clone()
        com_pos = self.compute_com_pos()
        (
            self.rew_buf[:],
            ig_reset,
            contact_reset,
            kinematic_reset,
            metric_1,
            metric_2,
        ) = compute_humanoid_reward(
            self._curr_ref_obs,
            self._curr_obs,
            self._contact_forces,
            self._tar_contact_forces,
            len(self._key_body_ids),
            self.reward_weights,
            dof_diffacc,
            self.object_points[self.object_id[self.data_id]],
            obj_diffacc,
            obj_rot_diffacc,
            self.mask[:, :628],
            self.mask[:, -628:],
            next_ts - ts,
            self._contact_body_ids,
            goal_ref_obs,
            com_pos,
            self.fall_state,
        )
        self.rew_buf -= self.enter_fall.float() * 50
        self._task_accomplish[metric_1 & metric_2] = True
        self.rew_buf += (
            metric_1 & metric_2 & torch.logical_not(self._task_accomplish)
        ) * 20
        if self.inference_mode:
            self.rew_buf[:] = 0
        self.contact_reset = (self.contact_reset + contact_reset) * contact_reset
        if self.inference_mode:
            self.contact_reset[:] = 0
        self._reset_ig = torch.logical_or(ig_reset, kinematic_reset)
        return

    def _compute_reset(self):
        reset_buf, self._terminate_buf[:] = compute_humanoid_reset(
            self.reset_buf,
            self.progress_buf,
            self.obs_buf,
            self._contact_forces,
            self._rigid_body_pos,
            self.max_episode_length[self.data_id],
            self._enable_early_termination,
            self._termination_heights,
            self._curr_ref_obs,
            self._curr_obs,
            self.start_times,
            self.rollout_length,
            self._reset_ig,
            self.contact_reset > 20,
            1 if self.inference_mode else self.device_id,
        )
        env_ids = to_torch(
            np.arange(self.num_envs), device=self.device, dtype=torch.long
        )
        if not self.inference_mode:
            self._terminate_buf = self.enter_fall
            continue_cond = torch.logical_and(
                reset_buf, torch.logical_not(self._terminate_buf)
            )
            continue_cond[self.num_envs // 2 + self.num_envs // 4 :] = torch.logical_or(
                continue_cond, self._task_accomplish & (self.long_term_t <= 1)
            )[self.num_envs // 2 + self.num_envs // 4 :]
            if continue_cond.any():
                self._reset_fake(env_ids[continue_cond])
                self.reset_buf_count += continue_cond
            self.reset_buf = torch.logical_or(
                self._terminate_buf, self.reset_buf_count > 2
            )
        else:
            self.reset_buf = reset_buf
        return

    def _refresh_sim_tensors(self):
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        return

    def _compute_task_obs_raw(self, env_ids=None, ref_obs=None):
        if env_ids is None:
            root_states = self._humanoid_root_states
            tar_states = self._target_states
        else:
            root_states = self._humanoid_root_states[env_ids]
            tar_states = self._target_states[env_ids]
        obs, obj_points = compute_obj_observations(
            root_states,
            tar_states,
            self.object_points[self.object_id[self.data_id[env_ids]]],
            ref_obs,
        )
        return (obs, obj_points)

    def _compute_task_obs(self, env_ids=None, ref_obs=None, long_term=False):
        if env_ids is None:
            env_ids = to_torch(
                np.arange(self.num_envs), device=self.device, dtype=torch.long
            )
            root_states = self._humanoid_root_states
            tar_states = self._target_states
        else:
            root_states = self._humanoid_root_states[env_ids]
            tar_states = self._target_states[env_ids]
        obs, goal, obj_points, obtained_mask = compute_obj_observations_decouple(
            root_states,
            tar_states,
            self.object_points[self.object_id[self.data_id[env_ids]]],
            ref_obs,
            long_term,
            not self.inference_mode,
        )
        return (obs, goal, obj_points, obtained_mask)

    def _prepare_goal_reference(self, reference, env_ids, delta_t):
        return reference

    def _compute_observations_iter(
        self, env_ids=None, delta_t=1, index=0, new_episode=None
    ):
        if env_ids is None:
            env_ids = to_torch(
                np.arange(self.num_envs), device=self.device, dtype=torch.long
            )
            ts = self.progress_buf_ref.clone()
            ts = torch.clamp(
                ts,
                min=self.max_episode_length[self.data_id[env_ids]]
                - self.max_episode_length[self.data_id[env_ids]],
                max=self.max_episode_length[self.data_id[env_ids]] - 1,
            )
            if self.inference_mode:
                self._curr_ref_obs = self.hoi_data[self.data_id[env_ids], ts].clone()
            else:
                self._curr_ref_obs = self.hoi_data_envs[
                    torch.arange(self.num_envs), ts
                ].clone()
            next_ts = torch.clamp(
                ts + delta_t,
                min=self.max_episode_length[self.data_id[env_ids]]
                - self.max_episode_length[self.data_id[env_ids]],
                max=self.max_episode_length[self.data_id[env_ids]] - 1,
            )
            if self.inference_mode:
                ref_obs = self.hoi_data[self.data_id[env_ids], next_ts].clone()
            else:
                ref_obs = self.hoi_data_envs[
                    torch.arange(self.num_envs), next_ts
                ].clone()
            ref_obs = self._prepare_goal_reference(ref_obs, env_ids, delta_t)
            require_mask = (torch.rand(env_ids.shape) < 0.01).to(self.device)
            if index == 0:
                self.require_mask[env_ids] = (
                    torch.logical_or(new_episode, require_mask)
                    .float()
                    .unsqueeze(1)
                    .to(self.device)
                )
                mask_mask = (
                    (torch.rand(env_ids.shape) < 0.3).unsqueeze(1).to(self.device)
                )
                mask_mask_human = (
                    (torch.rand(env_ids.shape) < 0.7).unsqueeze(1).to(self.device)
                )
                self.mask_mask[env_ids] = (
                    self.require_mask[env_ids]
                    * torch.logical_and(mask_mask, mask_mask_human)
                    + (1 - self.require_mask[env_ids]) * self.mask_mask[env_ids]
                )
                self.task_mask_mask[env_ids] = (
                    self.require_mask[env_ids]
                    * torch.logical_and(mask_mask, torch.logical_not(mask_mask_human))
                    + (1 - self.require_mask[env_ids]) * self.task_mask_mask[env_ids]
                )
            require_mask = require_mask.unsqueeze(1).to(self.device)
            obs, goal, mask = self._compute_humanoid_obs(
                env_ids, ref_obs, next_ts, index == 3
            )
            task_obs, task_goal, obj_points, task_mask = self._compute_task_obs(
                env_ids, ref_obs, index == 3
            )
            obs = torch.cat([obs, task_obs], dim=-1)
            key_body_pose = self._rigid_body_pos.clone()
            ig, normal = compute_sdf_normal(
                key_body_pose,
                obj_points,
                self.object_normals[self.object_id[self.data_id[env_ids]]],
            )
            obj_affordance = (
                (key_body_pose - ig)[:, self._key_body_ids, :].detach().clone()
            )
            heading_rot = torch_utils.calc_heading_quat_inv(
                self._rigid_body_rot[:, 0, :]
            )
            heading_rot_extend = (
                heading_rot.unsqueeze(1)
                .repeat(1, key_body_pose.shape[1], 1)
                .view(-1, 4)
            )
            ig = quat_rotate(heading_rot_extend, ig.view(-1, 3)).view(
                env_ids.shape[0], -1, 3
            )
            ig_norm = ig.norm(dim=-1, keepdim=True)
            ig_all = ig / (ig_norm + 1e-06) * (-5 * ig_norm).exp()
            ig = ig_all[:, self._key_body_ids, :].view(env_ids.shape[0], -1)
            ig_all = ig_all.view(env_ids.shape[0], -1)
            len_key_body_ids = len(self._key_body_ids)
            ref_ig = (
                ref_obs[
                    :,
                    326
                    + len_key_body_ids * 3
                    + 1
                    + 52 : 326
                    + len_key_body_ids * 3
                    + 1
                    + 52
                    + len_key_body_ids * 3,
                ]
                .contiguous()
                .view(env_ids.shape[0] * len_key_body_ids, 3)
            )
            heading_rot = torch_utils.calc_heading_quat(ref_obs[:, 3 : 3 + 4])
            heading_rot_extend = (
                heading_rot.unsqueeze(1).repeat(1, len_key_body_ids, 1).view(-1, 4)
            )
            ref_ig = quat_rotate(heading_rot_extend, ref_ig).view(
                env_ids.shape[0], len_key_body_ids, 3
            )
            _ref_body_pos = ref_obs[:, 326 : 326 + len_key_body_ids * 3].view(
                env_ids.shape[0], len_key_body_ids, 3
            )
            ref_obj_affordance = (_ref_body_pos - ref_ig).detach().clone()
            ref_obj_pos = ref_obs[:, 313 : 313 + 3]
            ref_obj_rot = ref_obs[:, 316 : 316 + 4]
            ref_obj_affordance = ref_obj_affordance - ref_obj_pos.unsqueeze(1)
            heading_rot = torch_utils.calc_heading_quat_inv(ref_obj_rot)
            heading_rot_extend = (
                heading_rot.unsqueeze(1).repeat(1, len_key_body_ids, 1).view(-1, 4)
            )
            ref_obj_affordance = quat_rotate(
                heading_rot_extend, ref_obj_affordance.view(-1, 3)
            )
            obj_pos = self._target_states[:, 0:3]
            obj_rot = self._target_states[:, 3:7]
            heading_rot = torch_utils.calc_heading_quat(obj_rot)
            heading_rot_extend = (
                heading_rot.unsqueeze(1).repeat(1, len_key_body_ids, 1).view(-1, 4)
            )
            ref_obj_affordance = quat_rotate(heading_rot_extend, ref_obj_affordance)
            ref_obj_affordance = ref_obj_affordance.view(-1, 21, 3) + obj_pos.unsqueeze(
                1
            )
            ig_mask = torch.zeros_like(ref_obj_affordance - obj_affordance)
            ig_mask = (
                (torch.rand([ig_mask.shape[0], ig_mask.shape[1]]) < 0.2)
                .float()
                .unsqueeze(2)
                .repeat(1, 1, 3)
                .to(self.device)
                .view(env_ids.shape[0], -1)
            )
            affordance_goal = (
                (ref_obj_affordance - obj_affordance)
                * (ref_ig.norm(dim=-1, keepdim=True) < 0.1)
            ).view(env_ids.shape[0], -1)
            obs_output = torch.cat(
                (obs, ig_all, normal.view(ig_all.shape[0], -1)), dim=-1
            )
            goal_all = torch.cat((goal, task_goal, affordance_goal), dim=-1)
            mask = mask * (1 - self.mask_mask)
            task_mask = task_mask * (1 - self.task_mask_mask)
            mask_all = torch.cat((mask, task_mask, ig_mask), dim=-1)
            if index > 0:
                self.mask_candidate[
                    env_ids,
                    index * mask_all.shape[-1] : (index + 1) * mask_all.shape[-1],
                ] = mask_all.to(self.mask.device) * require_mask.float().to(
                    self.mask.device
                ) + self.mask_candidate[
                    env_ids,
                    (index - 1) * mask_all.shape[-1] : index * mask_all.shape[-1],
                ] * (
                    1 - require_mask.float().to(self.mask.device)
                )
            else:
                self.mask_candidate[
                    env_ids,
                    index * mask_all.shape[-1] : (index + 1) * mask_all.shape[-1],
                ] = mask_all.to(self.mask.device)
            if index == 3:
                self.mask[env_ids] = (
                    self.mask_candidate[env_ids] * self.require_mask[env_ids]
                    + (1 - self.require_mask[env_ids]) * self.mask[env_ids]
                )
                long_term_goal = (
                    (torch.rand(env_ids.shape) < 0.35)
                    .unsqueeze(1)
                    .float()
                    .to(self.mask.device)
                )
                if not self.inference_mode:
                    long_term_goal[:] = 1
                long_term_goal = torch.logical_or(
                    long_term_goal, self.progress_buf_ref[env_ids].unsqueeze(1) < 0
                )
                self.long_term_goal[env_ids] = (
                    self.require_mask[env_ids] * long_term_goal
                    + (1 - self.require_mask[env_ids]) * self.long_term_goal[env_ids]
                )
                self.mask[env_ids, : index * mask_all.shape[-1]] = (
                    self.long_term_goal[env_ids] * 0
                    + (1 - self.long_term_goal[env_ids])
                    * self.mask[env_ids, : index * mask_all.shape[-1]]
                )
                all_0 = (
                    (torch.rand(env_ids.shape) < 0.005)
                    .float()
                    .unsqueeze(1)
                    .to(self.mask.device)
                )
                if not self.inference_mode:
                    all_0[:] = 0
                self.all_0[env_ids] = (
                    self.require_mask[env_ids] * all_0
                    + (1 - self.require_mask[env_ids]) * self.all_0[env_ids]
                )
                self.mask[env_ids] = (
                    self.all_0[env_ids] * 0
                    + (1 - self.all_0[env_ids]) * self.mask[env_ids]
                )
            return (obs_output, goal_all)
        else:
            ts = self.progress_buf_ref[env_ids].clone()
            ts = torch.clamp(
                ts,
                min=self.max_episode_length[self.data_id[env_ids]]
                - self.max_episode_length[self.data_id[env_ids]],
                max=self.max_episode_length[self.data_id[env_ids]] - 1,
            )
            if self.inference_mode:
                self._curr_ref_obs[env_ids] = self.hoi_data[
                    self.data_id[env_ids], ts
                ].clone()
            else:
                self._curr_ref_obs[env_ids] = self.hoi_data_envs[env_ids, ts].clone()
            next_ts = torch.clamp(
                ts + delta_t,
                min=self.max_episode_length[self.data_id[env_ids]]
                - self.max_episode_length[self.data_id[env_ids]],
                max=self.max_episode_length[self.data_id[env_ids]] - 1,
            )
            if self.inference_mode:
                ref_obs = self.hoi_data[self.data_id[env_ids], next_ts].clone()
            else:
                ref_obs = self.hoi_data_envs[env_ids, next_ts].clone()
            ref_obs = self._prepare_goal_reference(ref_obs, env_ids, delta_t)
            require_mask = (torch.rand(env_ids.shape) < 0.01).to(self.device)
            if index == 0:
                self.require_mask[env_ids] = (
                    torch.logical_or(new_episode, require_mask)
                    .float()
                    .unsqueeze(1)
                    .to(self.device)
                )
                mask_mask = (
                    (torch.rand(env_ids.shape) < 0.3).unsqueeze(1).to(self.device)
                )
                mask_mask_human = (
                    (torch.rand(env_ids.shape) < 0.7).unsqueeze(1).to(self.device)
                )
                self.mask_mask[env_ids] = (
                    self.require_mask[env_ids]
                    * torch.logical_and(mask_mask, mask_mask_human)
                    + (1 - self.require_mask[env_ids]) * self.mask_mask[env_ids]
                )
                self.task_mask_mask[env_ids] = (
                    self.require_mask[env_ids]
                    * torch.logical_and(mask_mask, torch.logical_not(mask_mask_human))
                    + (1 - self.require_mask[env_ids]) * self.task_mask_mask[env_ids]
                )
            require_mask = require_mask.unsqueeze(1).to(self.device)
            obs, goal, mask = self._compute_humanoid_obs(
                env_ids, ref_obs, next_ts, index == 3
            )
            task_obs, task_goal, obj_points, task_mask = self._compute_task_obs(
                env_ids, ref_obs, index == 3
            )
            obs = torch.cat([obs, task_obs], dim=-1)
            key_body_pose = self._rigid_body_pos[env_ids].to(self.device).clone()
            ig, normal = compute_sdf_normal(
                key_body_pose,
                obj_points,
                self.object_normals[self.object_id[self.data_id[env_ids]]],
            )
            obj_affordance = (
                (key_body_pose - ig)[:, self._key_body_ids, :].detach().clone()
            )
            heading_rot = torch_utils.calc_heading_quat_inv(
                self._rigid_body_rot[env_ids][:, 0, :]
            )
            heading_rot_extend = (
                heading_rot.unsqueeze(1)
                .repeat(1, key_body_pose.shape[1], 1)
                .view(-1, 4)
            )
            ig = quat_rotate(heading_rot_extend, ig.view(-1, 3)).view(
                env_ids.shape[0], -1, 3
            )
            ig_norm = ig.norm(dim=-1, keepdim=True)
            ig_all = ig / (ig_norm + 1e-06) * (-5 * ig_norm).exp()
            ig = ig_all[:, self._key_body_ids, :].view(env_ids.shape[0], -1)
            ig_all = ig_all.view(env_ids.shape[0], -1)
            len_key_body_ids = len(self._key_body_ids)
            ref_ig = (
                ref_obs[
                    :,
                    326
                    + len_key_body_ids * 3
                    + 1
                    + 52 : 326
                    + len_key_body_ids * 3
                    + 1
                    + 52
                    + len_key_body_ids * 3,
                ]
                .contiguous()
                .view(env_ids.shape[0] * len_key_body_ids, 3)
            )
            heading_rot = torch_utils.calc_heading_quat(ref_obs[:, 3 : 3 + 4])
            heading_rot_extend = (
                heading_rot.unsqueeze(1).repeat(1, len_key_body_ids, 1).view(-1, 4)
            )
            ref_ig = quat_rotate(heading_rot_extend, ref_ig).view(
                env_ids.shape[0], len_key_body_ids, 3
            )
            _ref_body_pos = ref_obs[:, 326 : 326 + len_key_body_ids * 3].view(
                env_ids.shape[0], len_key_body_ids, 3
            )
            ref_obj_affordance = (_ref_body_pos - ref_ig).detach().clone()
            ref_obj_pos = ref_obs[:, 313 : 313 + 3]
            ref_obj_rot = ref_obs[:, 316 : 316 + 4]
            ref_obj_affordance = ref_obj_affordance - ref_obj_pos.unsqueeze(1)
            heading_rot = torch_utils.calc_heading_quat_inv(ref_obj_rot)
            heading_rot_extend = (
                heading_rot.unsqueeze(1).repeat(1, len_key_body_ids, 1).view(-1, 4)
            )
            ref_obj_affordance = quat_rotate(
                heading_rot_extend, ref_obj_affordance.view(-1, 3)
            )
            obj_pos = self._target_states[env_ids, 0:3]
            obj_rot = self._target_states[env_ids, 3:7]
            heading_rot = torch_utils.calc_heading_quat(obj_rot)
            heading_rot_extend = (
                heading_rot.unsqueeze(1).repeat(1, len_key_body_ids, 1).view(-1, 4)
            )
            ref_obj_affordance = quat_rotate(heading_rot_extend, ref_obj_affordance)
            ref_obj_affordance = ref_obj_affordance.view(-1, 21, 3) + obj_pos.unsqueeze(
                1
            )
            ig_mask = torch.zeros_like(ref_obj_affordance - obj_affordance)
            ig_mask = (
                (torch.rand([ig_mask.shape[0], ig_mask.shape[1]]) < 0.2)
                .float()
                .unsqueeze(2)
                .repeat(1, 1, 3)
                .to(self.device)
                .view(env_ids.shape[0], -1)
            )
            affordance_goal = (
                (ref_obj_affordance - obj_affordance)
                * (ref_ig.norm(dim=-1, keepdim=True) < 0.1)
            ).view(env_ids.shape[0], -1)
            obs_output = torch.cat(
                (obs, ig_all, normal.view(ig_all.shape[0], -1)), dim=-1
            )
            goal_all = torch.cat((goal, task_goal, affordance_goal), dim=-1)
            mask = mask * (1 - self.mask_mask[env_ids])
            task_mask = task_mask * (1 - self.task_mask_mask[env_ids])
            mask_all = torch.cat((mask, task_mask, ig_mask), dim=-1)
            if index > 0:
                self.mask_candidate[
                    env_ids,
                    index * mask_all.shape[-1] : (index + 1) * mask_all.shape[-1],
                ] = mask_all.to(self.mask.device) * require_mask.float().to(
                    self.mask.device
                ) + self.mask_candidate[
                    env_ids,
                    (index - 1) * mask_all.shape[-1] : index * mask_all.shape[-1],
                ] * (
                    1 - require_mask.float().to(self.mask.device)
                )
            else:
                self.mask_candidate[
                    env_ids,
                    index * mask_all.shape[-1] : (index + 1) * mask_all.shape[-1],
                ] = mask_all.to(self.mask.device)
            if index == 3:
                self.mask[env_ids] = (
                    self.mask_candidate[env_ids] * self.require_mask[env_ids]
                    + (1 - self.require_mask[env_ids]) * self.mask[env_ids]
                )
                long_term_goal = (
                    (torch.rand(env_ids.shape) < 0.35)
                    .unsqueeze(1)
                    .float()
                    .to(self.mask.device)
                )
                if not self.inference_mode:
                    long_term_goal[:] = 1
                long_term_goal = torch.logical_or(
                    long_term_goal, self.progress_buf_ref[env_ids].unsqueeze(1) < 0
                )
                self.long_term_goal[env_ids] = (
                    self.require_mask[env_ids] * long_term_goal
                    + (1 - self.require_mask[env_ids]) * self.long_term_goal[env_ids]
                )
                self.mask[env_ids, : index * mask_all.shape[-1]] = (
                    self.long_term_goal[env_ids] * 0
                    + (1 - self.long_term_goal[env_ids])
                    * self.mask[env_ids, : index * mask_all.shape[-1]]
                )
                all_0 = (
                    (torch.rand(env_ids.shape) < 0.005)
                    .float()
                    .unsqueeze(1)
                    .to(self.mask.device)
                )
                if not self.inference_mode:
                    all_0[:] = 0
                self.all_0[env_ids] = (
                    self.require_mask[env_ids] * all_0
                    + (1 - self.require_mask[env_ids]) * self.all_0[env_ids]
                )
                self.mask[env_ids] = (
                    self.all_0[env_ids] * 0
                    + (1 - self.all_0[env_ids]) * self.mask[env_ids]
                )
            return (obs_output, goal_all)
        return

    def _compute_observations(self, env_ids=None):
        new_episode = self.progress_buf[env_ids] == self.start_times[env_ids]
        new_long_term_t = torch.logical_or(self.long_term_t[env_ids] < 1, new_episode)
        self.long_term_t[env_ids[new_long_term_t]] = (
            torch.randint_like(env_ids[new_long_term_t], 64) + 64
        )
        self._task_accomplish[env_ids[new_long_term_t]] = 0
        long_term_t = self.long_term_t.clone()
        long_term_t[self.long_term_t + self.progress_buf_ref < 0] = (
            -self.progress_buf_ref[self.long_term_t + self.progress_buf_ref < 0] + 1
        )
        self.long_term_t[env_ids[new_long_term_t]] = long_term_t[
            env_ids[new_long_term_t]
        ]
        if env_ids is None:
            obs1, goal1 = self._compute_observations_iter(None, 1, 0, new_episode)
            obs2, goal2 = self._compute_observations_iter(None, 4, 1, new_episode)
            obs3, goal3 = self._compute_observations_iter(None, 16, 2, new_episode)
            obs4, goal4 = self._compute_observations_iter(
                None, self.long_term_t, 3, new_episode
            )
        else:
            obs1, goal1 = self._compute_observations_iter(env_ids, 1, 0, new_episode)
            obs2, goal2 = self._compute_observations_iter(env_ids, 4, 1, new_episode)
            obs3, goal3 = self._compute_observations_iter(env_ids, 16, 2, new_episode)
            obs4, goal4 = self._compute_observations_iter(
                env_ids, self.long_term_t[env_ids], 3, new_episode
            )
        self.obs_buf[env_ids] = torch.cat(
            [
                self._hist_obs_all[env_ids].view(env_ids.shape[0], -1),
                obs1,
                goal1,
                goal2,
                goal3,
                goal4,
            ],
            dim=1,
        )
        self._hist_obs_all[env_ids, :-1] = self._hist_obs_all[env_ids, 1:].clone()
        self._hist_obs_all[env_ids, -1] = obs1.clone()
        return

    def _compute_observations_raw_iter(self, env_ids=None, delta_t=1):
        if env_ids is None:
            env_ids = to_torch(
                np.arange(self.num_envs), device=self.device, dtype=torch.long
            )
            ts = self.progress_buf_ref.clone()
            ts = torch.clamp(
                ts,
                min=self.max_episode_length[self.data_id[env_ids]]
                - self.max_episode_length[self.data_id[env_ids]],
                max=self.max_episode_length[self.data_id[env_ids]] - 1,
            )
            self._curr_ref_obs = self.hoi_data[self.data_id[env_ids], ts].clone()
            next_ts = torch.clamp(
                self.progress_buf_ref + delta_t,
                min=self.max_episode_length[self.data_id[env_ids]]
                - self.max_episode_length[self.data_id[env_ids]],
                max=self.max_episode_length[self.data_id[env_ids]] - 1,
            )
            ref_obs = self.hoi_data[self.data_id[env_ids], next_ts].clone()
            obs = self._compute_humanoid_obs_raw(env_ids, ref_obs, next_ts)
            task_obs, obj_points = self._compute_task_obs_raw(env_ids, ref_obs)
            obs = torch.cat([obs, task_obs], dim=-1)
            key_body_pose = self._rigid_body_pos.clone()
            ig = compute_sdf(key_body_pose, obj_points).view(-1, 3)
            heading_rot = torch_utils.calc_heading_quat_inv(
                self._rigid_body_rot[:, 0, :]
            )
            heading_rot_extend = (
                heading_rot.unsqueeze(1)
                .repeat(1, key_body_pose.shape[1], 1)
                .view(-1, 4)
            )
            ig = quat_rotate(heading_rot_extend, ig).view(env_ids.shape[0], -1, 3)
            ig_norm = ig.norm(dim=-1, keepdim=True)
            ig_all = ig / (ig_norm + 1e-06) * (-5 * ig_norm).exp()
            ig = ig_all[:, self._key_body_ids, :].view(env_ids.shape[0], -1)
            ig_all = ig_all.view(env_ids.shape[0], -1)
            len_key_body_ids = len(self._key_body_ids)
            ref_ig = ref_obs[
                :,
                326
                + len_key_body_ids * 3
                + 1
                + 52 : 326
                + len_key_body_ids * 3
                + 1
                + 52
                + len_key_body_ids * 3,
            ].view(env_ids.shape[0], len_key_body_ids, 3)
            ref_ig_norm = ref_ig.norm(dim=-1, keepdim=True)
            ref_ig = ref_ig / (ref_ig_norm + 1e-06) * (-5 * ref_ig_norm).exp()
            ref_ig = ref_ig.view(env_ids.shape[0], -1)
            return torch.cat((obs, ig_all, ref_ig - ig), dim=-1)
        else:
            ts = self.progress_buf_ref[env_ids].clone()
            ts = torch.clamp(
                ts,
                min=self.max_episode_length[self.data_id[env_ids]]
                - self.max_episode_length[self.data_id[env_ids]],
                max=self.max_episode_length[self.data_id[env_ids]] - 1,
            )
            self._curr_ref_obs[env_ids] = self.hoi_data[
                self.data_id[env_ids], ts
            ].clone()
            next_ts = torch.clamp(
                self.progress_buf_ref[env_ids] + delta_t,
                min=self.max_episode_length[self.data_id[env_ids]]
                - self.max_episode_length[self.data_id[env_ids]],
                max=self.max_episode_length[self.data_id[env_ids]] - 1,
            )
            ref_obs = self.hoi_data[self.data_id[env_ids], next_ts].clone()
            obs = self._compute_humanoid_obs_raw(env_ids, ref_obs, next_ts)
            task_obs, obj_points = self._compute_task_obs_raw(env_ids, ref_obs)
            obs = torch.cat([obs, task_obs], dim=-1)
            key_body_pose = self._rigid_body_pos[env_ids].clone()
            ig = compute_sdf(key_body_pose, obj_points).view(-1, 3)
            heading_rot = torch_utils.calc_heading_quat_inv(
                self._rigid_body_rot[env_ids][:, 0, :]
            )
            heading_rot_extend = (
                heading_rot.unsqueeze(1)
                .repeat(1, key_body_pose.shape[1], 1)
                .view(-1, 4)
            )
            ig = quat_rotate(heading_rot_extend, ig).view(env_ids.shape[0], -1, 3)
            ig_norm = ig.norm(dim=-1, keepdim=True)
            ig_all = ig / (ig_norm + 1e-06) * (-5 * ig_norm).exp()
            ig = ig_all[:, self._key_body_ids, :].view(env_ids.shape[0], -1)
            ig_all = ig_all.view(env_ids.shape[0], -1)
            len_key_body_ids = len(self._key_body_ids)
            ref_ig = ref_obs[
                :,
                326
                + len_key_body_ids * 3
                + 1
                + 52 : 326
                + len_key_body_ids * 3
                + 1
                + 52
                + len_key_body_ids * 3,
            ].view(env_ids.shape[0], len_key_body_ids, 3)
            ref_ig_norm = ref_ig.norm(dim=-1, keepdim=True)
            ref_ig = ref_ig / (ref_ig_norm + 1e-06) * (-5 * ref_ig_norm).exp()
            ref_ig = ref_ig.view(env_ids.shape[0], -1)
            return torch.cat((obs, ig_all, ref_ig - ig), dim=-1)
        return

    def _compute_observations_raw(self, env_ids=None):
        if env_ids is None:
            self.obs_buf_raw[:] = torch.cat(
                (
                    self._compute_observations_raw_iter(None, 1),
                    self._compute_observations_raw_iter(None, 16),
                ),
                dim=-1,
            )
        else:
            self.obs_buf_raw[env_ids] = torch.cat(
                (
                    self._compute_observations_raw_iter(env_ids, 1),
                    self._compute_observations_raw_iter(env_ids, 16),
                ),
                dim=-1,
            )
        return

    def _compute_humanoid_obs_raw(self, env_ids=None, ref_obs=None, next_ts=None):
        if env_ids is None:
            body_pos = self._rigid_body_pos
            body_rot = self._rigid_body_rot
            body_vel = self._rigid_body_vel
            body_ang_vel = self._rigid_body_ang_vel
            contact_forces = self._contact_forces
        else:
            body_pos = self._rigid_body_pos[env_ids]
            body_rot = self._rigid_body_rot[env_ids]
            body_vel = self._rigid_body_vel[env_ids]
            body_ang_vel = self._rigid_body_ang_vel[env_ids]
            contact_forces = self._contact_forces[env_ids]
        obs = compute_humanoid_observations_max(
            body_pos,
            body_rot,
            body_vel,
            body_ang_vel,
            self._local_root_obs,
            self._root_height_obs,
            contact_forces,
            self._contact_body_ids,
            ref_obs,
            self._key_body_ids,
        )
        return obs

    def _compute_humanoid_obs(
        self, env_ids=None, ref_obs=None, next_ts=None, long_term=False
    ):
        if env_ids is None:
            env_ids = to_torch(
                np.arange(self.num_envs), device=self.device, dtype=torch.long
            )
            body_pos = self._rigid_body_pos
            body_rot = self._rigid_body_rot
            body_vel = self._rigid_body_vel
            body_ang_vel = self._rigid_body_ang_vel
            contact_forces = self._contact_forces
            fall_state = self.fall_state
        else:
            body_pos = self._rigid_body_pos[env_ids]
            body_rot = self._rigid_body_rot[env_ids]
            body_vel = self._rigid_body_vel[env_ids]
            body_ang_vel = self._rigid_body_ang_vel[env_ids]
            contact_forces = self._contact_forces[env_ids]
            fall_state = self.fall_state[env_ids]
        obs, goal, obtained_mask = compute_humanoid_observations_max_decouple(
            body_pos,
            body_rot,
            body_vel,
            body_ang_vel,
            fall_state,
            self._root_height_obs,
            contact_forces,
            self._contact_body_ids,
            ref_obs,
            self._key_body_ids,
            long_term,
            not self.inference_mode,
        )
        return (obs, goal, obtained_mask)

    def _reset_actors(self, env_ids):
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[
            env_ids
        ]
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        return

    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone()
        if self._pd_control:
            pd_tar = self._action_to_pd_targets(self.actions)
            pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
            self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)
        else:
            forces = self.actions * self.motor_efforts.unsqueeze(0) * self.power_scale
            force_tensor = gymtorch.unwrap_tensor(forces)
            self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)
        return

    def align_ref_seq_to_current(
        self,
        cur_root_rot,
        ref_root_rot,
        ref_joints,
        ref_obj,
        ref_root_pos_t,
        cur_root_pos,
        sim_joints_t,
        len_keypos: int,
    ):
        """
        Align the entire reference sequence by yaw (about z) and translation.
        Uses self.torch_utils.quat_to_matrix() if torch_utils is registered on self.
        Returns:
            J_flat:    (B,T,K*3)
            O_aligned: (B,T,3)
            dpsi:      (B,) yaw delta in radians
        """
        B, T = ref_joints.shape[:2]
        K = len_keypos
        ref_rot_t = ref_root_rot
        h_ref = torch_utils.calc_heading_quat(ref_rot_t)
        h_ref_i = torch_utils.calc_heading_quat_inv(ref_rot_t)
        h_cur = torch_utils.calc_heading_quat(cur_root_rot)
        dQ = torch_utils.quat_mul(h_cur, h_ref_i)
        J = ref_joints.view(B, T, K, 3)
        rel = J - ref_root_pos_t.view(B, 1, 1, 3)
        rel_flat = rel.view(B * T * K, 3)
        dQ_expand = dQ[:, None, :].expand(B, T * K, 4).contiguous().view(B * T * K, 4)
        rel_rot = torch_utils.quat_rotate(dQ_expand, rel_flat)
        rel_rot = rel_rot.view(B, T, K, 3)
        root = cur_root_pos.view(B, 1, 1, 3)
        J_aligned = rel_rot + root
        J_aligned[..., 2:3] = J[..., 2:3]
        J_flat = J_aligned.view(B, T, K * 3).contiguous()
        Orel = (ref_obj - ref_root_pos_t.view(B, 1, 3)).view(B * T, 3)
        dQ_expand_obj = dQ[:, None, :].expand(B, T, 4).contiguous().view(B * T, 4)
        Orel_rot = torch_utils.quat_rotate(dQ_expand_obj, Orel)
        O_aligned = Orel_rot.view(B, T, 3) + cur_root_pos.view(B, 1, 3)
        O_aligned[..., 2:3] = ref_obj[..., 2:3]
        return (J_flat, O_aligned)

    def post_physics_step(self):
        self._refresh_sim_tensors()
        self.common_step_counter += 1
        env_ids = to_torch(
            np.arange(self.num_envs), device=self.device, dtype=torch.long
        )
        self._compute_hoi_observations(env_ids)
        self.progress_buf += 1
        move_on = torch.logical_or(self._task_accomplish, self.long_term_t > 1)
        move_on = torch.logical_or(move_on, self.not_move_on > 30)
        self.not_move_on[torch.logical_not(move_on)] += 1
        if self.inference_mode:
            move_on[:] = True
        self.not_move_on[move_on] = 0
        self.progress_buf_ref[move_on] += 1
        self.long_term_t[move_on] -= 1
        self._compute_observations(env_ids)
        self._compute_observations_raw(env_ids)
        self.update_fall_state()
        self._compute_reward(self.actions)
        self._compute_reset()
        if not self.inference_mode:
            self._terminate_buf = torch.logical_or(
                self.not_move_on > 29, self._terminate_buf
            )
        self.extras["terminate"] = self._terminate_buf.clone()
        if self.viewer and self.debug_viz:
            self._update_debug_viz()
        return

    def render(self, sync_frame_time=False):
        if self.viewer:
            self._update_camera()
        super().render(sync_frame_time)
        return

    def _build_key_body_ids_tensor(self, key_body_names):
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        body_ids = []
        for body_name in key_body_names:
            body_id = self.gym.find_actor_rigid_body_handle(
                env_ptr, actor_handle, body_name
            )
            assert body_id != -1
            body_ids.append(body_id)
        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _build_contact_body_ids_tensor(self, contact_body_names):
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        body_ids = []
        for body_name in contact_body_names:
            body_id = self.gym.find_actor_rigid_body_handle(
                env_ptr, actor_handle, body_name
            )
            assert body_id != -1
            body_ids.append(body_id)
        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _action_to_pd_targets(self, action):
        pd_tar = self._pd_action_offset + self._pd_action_scale * action
        return pd_tar

    def _init_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._cam_prev_char_pos = self._humanoid_root_states[0, 0:3].cpu().numpy()
        cam_pos = gymapi.Vec3(
            self._cam_prev_char_pos[0], self._cam_prev_char_pos[1] - 3.0, 1.0
        )
        cam_target = gymapi.Vec3(
            self._cam_prev_char_pos[0], self._cam_prev_char_pos[1], 1.0
        )
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        return

    def _update_camera(self):
        return

    def _update_debug_viz(self):
        self.gym.clear_lines(self.viewer)
        return


class InterPriorTask(SMPLHumanoid):

    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    def __init__(
        self, cfg, sim_params, physics_engine, device_type, device_id, headless
    ):
        state_init = cfg["env"]["stateInit"]
        self._state_init = InterPriorTask.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []
        self.motion_file_retarget = cfg["env"]["motion_file_retarget"]
        self.play_dataset = cfg["env"]["playdataset"]
        self.projtype = cfg["env"]["projtype"]
        self.reward_weights = cfg["env"]["rewardWeights"]
        self.save_images = cfg["env"]["saveImages"]
        self.init_vel = cfg["env"]["initVel"]
        self.ball_size = cfg["env"]["ballSize"]
        self.more_rigid = cfg["env"]["moreRigid"]
        self.rollout_length = cfg["env"]["rolloutLength"]
        if cfg["env"].get("inference_mode", False):
            self.rollout_length = cfg["env"].get("rolloutLength", 2000)
        self.motion_file = cfg["env"]["motionFiles"]
        self.dataset_index = to_torch(
            [
                int(data_path.split("/")[-1].split("_")[0][3:])
                for data_path in self.motion_file
            ],
            dtype=torch.long,
            device="cuda:" + str(device_id),
        )
        self.object_name = [
            motion_example.split("/")[-1].split("_")[1]
            for motion_example in self.motion_file
        ]
        object_name_set = sorted(list(set(self.object_name)))
        print(
            object_name_set,
            set(
                [
                    int(data_path.split("/")[-1].split("_")[0][3:])
                    for data_path in self.motion_file
                ]
            ),
        )
        self.object_id = to_torch(
            [object_name_set.index(name) for name in self.object_name],
            dtype=torch.long,
            device="cuda:" + str(device_id),
        )
        self.obj2motion = torch.stack(
            [self.object_id == k for k in range(len(object_name_set))], dim=0
        )
        self.object_name = object_name_set
        self.t = 0
        self.robot_type = "smplx/omomo.xml"
        print(self.robot_type)
        self.num_motions = len(self.motion_file)
        self.motion_ids = to_torch(
            cfg["env"]["motionIds"], dtype=torch.long, device="cuda:" + str(device_id)
        )
        self.env_object_ids = self.object_id[self.motion_ids]

        super().__init__(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=physics_engine,
            device_type=device_type,
            device_id=device_id,
            headless=headless,
        )
        self._load_motion(self.motion_file)
        self._curr_ref_obs = torch.zeros(
            (self.num_envs, self.ref_hoi_obs_size),
            device=self.device,
            dtype=torch.float,
        )
        self._hist_ref_obs = torch.zeros(
            (self.num_envs, self.ref_hoi_obs_size),
            device=self.device,
            dtype=torch.float,
        )
        self._curr_obs = torch.zeros(
            (self.num_envs, self.ref_hoi_obs_size),
            device=self.device,
            dtype=torch.float,
        )
        self._hist_obs = torch.zeros(
            (self.num_envs, self.ref_hoi_obs_size),
            device=self.device,
            dtype=torch.float,
        )
        self._tar_pos = torch.zeros(
            [self.num_envs, 3], device=self.device, dtype=torch.float
        )
        self._reset_ig = torch.zeros(
            [self.num_envs], device=self.device, dtype=torch.bool
        )
        self.mask = torch.ones(
            [self.num_envs, 628 * 4], device=self.device, dtype=torch.float
        )
        self.mask_candidate = torch.ones(
            [self.num_envs, 628 * 4], device=self.device, dtype=torch.float
        )
        self.mask_mask = torch.zeros(
            [self.num_envs, 1], device=self.device, dtype=torch.float
        )
        self.task_mask_mask = torch.zeros(
            [self.num_envs, 1], device=self.device, dtype=torch.float
        )
        self._hist_obs_all = torch.zeros(
            (self.num_envs, 4, 1127), device=self.device, dtype=torch.float
        )
        self.long_term_t = torch.zeros(
            [self.num_envs], device=self.device, dtype=torch.long
        )
        self.long_term_goal = torch.zeros(
            [self.num_envs, 1], device=self.device, dtype=torch.float
        )
        self.all_0 = torch.zeros(
            [self.num_envs, 1], device=self.device, dtype=torch.float
        )
        self.require_mask = torch.zeros(
            [self.num_envs, 1], device=self.device, dtype=torch.float
        )
        self.fall_state = torch.zeros(
            [self.num_envs], device=self.device, dtype=torch.bool
        )
        self._fall_on_ctr = torch.zeros(
            [self.num_envs], dtype=torch.int32, device=self.device
        )
        self._fall_off_ctr = torch.zeros(
            [self.num_envs], dtype=torch.int32, device=self.device
        )
        self._task_accomplish = torch.zeros(
            [self.num_envs], dtype=torch.int32, device=self.device
        )
        self.not_move_on = torch.zeros(
            [self.num_envs], dtype=torch.int32, device=self.device
        )
        self.progress_buf_ref = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self._build_target_tensors()
        if self.projtype == "Mouse" or self.projtype == "Auto":
            self._build_proj_tensors()
        self._cache_rb_mass_and_com()
        return

    def _compute_reward(self, actions):
        super()._compute_reward(actions)
        return

    def _compute_reset(self):
        super()._compute_reset()

    def post_physics_step(self):
        if self.projtype == "Mouse" or self.projtype == "Auto":
            self._update_proj()
        super().post_physics_step()
        env_ids = to_torch(
            np.arange(self.num_envs), device=self.device, dtype=torch.long
        )
        self._update_hist_hoi_obs()
        self._compute_hoi_observations(env_ids)
        return

    def update_fall_state(self):
        """
        Maintains per-env fall state with hysteresis.
        Requires these tensors already exist (as in your class):
        - self._rigid_body_pos:   [N, B, 3]
        - self._rigid_body_rot:   [N, B, 4]
        - self._contact_forces:   [N, B, 3]
        - self._dof_pos:          [N, D]
        And these attributes set once:
        - self.l_ank_id, self.l_toe_id, self.r_ank_id, self.r_toe_id  (foot body indices)
        Creates/updates:
        - self.fall_state:        Bool[N]
        - self._fall_on_ctr:      Int[N]
        - self._fall_off_ctr:     Int[N]
        """
        device = self._rigid_body_pos.device
        N, B = self._rigid_body_pos.shape[:2]
        z_floor = 0.03
        contact_thres = 20.0
        z_stand = 0.7
        upright_thresh = 0.85
        dof_neutral = 0.15
        on_steps = 3
        off_steps = 6
        root_z = self._rigid_body_pos[:, 0, 2]
        root_q = self._rigid_body_rot[:, 0, :]
        q = F.normalize(root_q, dim=-1)
        qx, qy, qz, qw = q.unbind(-1)
        q_vec = torch.stack([qx, qy, qz], dim=-1)
        v = q.new_zeros(N, 3)
        v[:, 2] = 1.0
        t = 2.0 * torch.cross(q_vec, v, dim=-1)
        up_w = v + qw.unsqueeze(-1) * t + torch.cross(q_vec, t, dim=-1)
        dot_up = up_w[:, 2].clamp(-1.0, 1.0)
        dof_mean_abs = self._dof_pos.abs().max(dim=-1)[0]
        standing_now = (
            (root_z > z_stand)
            & (dot_up > upright_thresh)
            & (dof_mean_abs < dof_neutral)
        )
        detect_enabled = ~standing_now
        foot_ids = torch.tensor([2, 3, 4, 6, 7, 8], device=device, dtype=torch.long)
        all_ids = torch.arange(B, device=device)
        nonfoot_mask = torch.ones(B, dtype=torch.bool, device=device)
        nonfoot_mask[foot_ids] = False
        near_ground_nonfoot = (
            self._rigid_body_pos[:, :, 2] < z_floor
        ) & nonfoot_mask.unsqueeze(0)
        contact_mag = self._contact_forces.abs().amax(dim=-1)
        contact_nonfoot = (contact_mag > contact_thres) & nonfoot_mask.unsqueeze(0)
        fall_trigger_now = (near_ground_nonfoot & contact_nonfoot).any(dim=1)
        fall_trigger_now = fall_trigger_now | (root_z < 0.3)
        fall_trigger_now = fall_trigger_now & detect_enabled
        self._fall_on_ctr = torch.where(
            fall_trigger_now, self._fall_on_ctr + 1, torch.zeros_like(self._fall_on_ctr)
        )
        self._fall_off_ctr = torch.where(
            standing_now, self._fall_off_ctr + 1, torch.zeros_like(self._fall_off_ctr)
        )
        enter_fall = ~self.fall_state & (self._fall_on_ctr >= on_steps)
        self.enter_fall = enter_fall.clone()
        recover = self.fall_state & (self._fall_off_ctr >= off_steps)
        self.recover = recover.clone()
        self.fall_state = (self.fall_state | enter_fall) & ~recover
        self._fall_on_ctr[enter_fall] = 0
        self._fall_off_ctr[recover] = 0

    def _update_hist_hoi_obs(self, env_ids=None):
        self._hist_obs = self._curr_obs.clone()
        return

    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)
        return

    def _load_motion(self, motion_file):
        self.hoi_data_dict = []
        hoi_datas = []
        hoi_refs = []
        if not isinstance(motion_file, list):
            motion_file = [motion_file]
        self.max_episode_length = []
        for idx, data_path in enumerate(motion_file):
            loaded_dict = {}
            hoi_data = load_motion_tensor(data_path)
            loaded_dict["hoi_data"] = hoi_data[1:].detach().to(self.device)
            if idx == 0:
                print(loaded_dict["hoi_data"].shape, data_path)
            self.max_episode_length.append(loaded_dict["hoi_data"].shape[0])
            self.fps_data = 30.0
            loaded_dict["root_pos"] = loaded_dict["hoi_data"][:, 0:3].clone()
            loaded_dict["root_pos_vel"] = (
                loaded_dict["root_pos"][1:, :].clone()
                - loaded_dict["root_pos"][:-1, :].clone()
            ) * self.fps_data
            loaded_dict["root_pos_vel"] = torch.cat(
                (
                    torch.zeros((1, loaded_dict["root_pos_vel"].shape[-1])).to(
                        self.device
                    ),
                    loaded_dict["root_pos_vel"],
                ),
                dim=0,
            )
            loaded_dict["root_rot"] = loaded_dict["hoi_data"][:, 3:7].clone()
            root_rot_exp_map = torch_utils.quat_to_exp_map(loaded_dict["root_rot"])
            loaded_dict["root_rot_vel"] = (
                root_rot_exp_map[1:, :].clone() - root_rot_exp_map[:-1, :].clone()
            ) * self.fps_data
            loaded_dict["root_rot_vel"] = torch.cat(
                (
                    torch.zeros((1, loaded_dict["root_rot_vel"].shape[-1])).to(
                        self.device
                    ),
                    loaded_dict["root_rot_vel"],
                ),
                dim=0,
            )
            loaded_dict["dof_pos"] = loaded_dict["hoi_data"][:, 9 : 9 + 153].clone()
            loaded_dict["dof_pos_vel"] = []
            loaded_dict["dof_pos_vel"] = (
                loaded_dict["dof_pos"][1:, :].clone()
                - loaded_dict["dof_pos"][:-1, :].clone()
            ) * self.fps_data
            loaded_dict["dof_pos_vel"] = torch.cat(
                (
                    torch.zeros((1, loaded_dict["dof_pos_vel"].shape[-1])).to(
                        self.device
                    ),
                    loaded_dict["dof_pos_vel"],
                ),
                dim=0,
            )
            loaded_dict["body_pos"] = (
                loaded_dict["hoi_data"][:, 162 : 162 + 52 * 3]
                .clone()
                .view(self.max_episode_length[-1], 52, 3)
            )
            loaded_dict["key_body_pos"] = (
                loaded_dict["body_pos"][:, self._key_body_ids, :]
                .view(self.max_episode_length[-1], -1)
                .clone()
            )
            loaded_dict["key_body_pos_vel"] = (
                loaded_dict["key_body_pos"][1:, :].clone()
                - loaded_dict["key_body_pos"][:-1, :].clone()
            ) * self.fps_data
            loaded_dict["key_body_pos_vel"] = torch.cat(
                (
                    torch.zeros((1, loaded_dict["key_body_pos_vel"].shape[-1])).to(
                        self.device
                    ),
                    loaded_dict["key_body_pos_vel"],
                ),
                dim=0,
            )
            loaded_dict["obj_pos"] = loaded_dict["hoi_data"][:, 318:321].clone()
            loaded_dict["obj_pos_vel"] = (
                loaded_dict["obj_pos"][1:, :].clone()
                - loaded_dict["obj_pos"][:-1, :].clone()
            ) * self.fps_data
            if self.init_vel:
                loaded_dict["obj_pos_vel"] = torch.cat(
                    (loaded_dict["obj_pos_vel"][:1], loaded_dict["obj_pos_vel"]), dim=0
                )
            else:
                loaded_dict["obj_pos_vel"] = torch.cat(
                    (
                        torch.zeros((1, loaded_dict["obj_pos_vel"].shape[-1])).to(
                            self.device
                        ),
                        loaded_dict["obj_pos_vel"],
                    ),
                    dim=0,
                )
            loaded_dict["obj_rot"] = loaded_dict["hoi_data"][:, 321:325].clone()
            obj_rot_exp_map = torch_utils.quat_to_exp_map(loaded_dict["obj_rot"])
            loaded_dict["obj_rot_vel"] = (
                obj_rot_exp_map[1:, :].clone() - obj_rot_exp_map[:-1, :].clone()
            ) * self.fps_data
            loaded_dict["obj_rot_vel"] = torch.cat(
                (
                    torch.zeros((1, loaded_dict["obj_rot_vel"].shape[-1])).to(
                        self.device
                    ),
                    loaded_dict["obj_rot_vel"],
                ),
                dim=0,
            )
            obj_rot_extend = (
                loaded_dict["obj_rot"]
                .unsqueeze(1)
                .repeat(1, self.object_points[self.object_id[idx]].shape[0], 1)
                .view(-1, 4)
            )
            object_points_extend = (
                self.object_points[self.object_id[idx]]
                .unsqueeze(0)
                .repeat(loaded_dict["obj_rot"].shape[0], 1, 1)
                .view(-1, 3)
            )
            obj_points = torch_utils.quat_rotate(
                obj_rot_extend, object_points_extend
            ).view(
                loaded_dict["obj_rot"].shape[0],
                self.object_points[self.object_id[idx]].shape[0],
                3,
            ) + loaded_dict[
                "obj_pos"
            ].unsqueeze(
                1
            )
            key_body_pose = loaded_dict["key_body_pos"][:, :].clone()
            ref_ig = compute_sdf(
                key_body_pose.view(loaded_dict["obj_rot"].shape[0], -1, 3), obj_points
            ).view(-1, 3)
            heading_rot = torch_utils.calc_heading_quat_inv(loaded_dict["root_rot"])
            heading_rot_extend = (
                heading_rot.unsqueeze(1)
                .repeat(1, key_body_pose.shape[1] // 3, 1)
                .view(-1, 4)
            )
            ref_ig = quat_rotate(heading_rot_extend, ref_ig).view(
                loaded_dict["obj_rot"].shape[0], -1
            )
            loaded_dict["contact"] = torch.round(
                loaded_dict["hoi_data"][:, 330:331].clone()
            )
            loaded_dict["contact_parts"] = torch.round(
                loaded_dict["hoi_data"][:, 331 : 331 + 52].clone()
            )
            loaded_dict["human_rot"] = loaded_dict["hoi_data"][
                :, 331 + 52 : 331 + 52 + 52 * 4
            ].clone()
            human_rot_exp_map = torch_utils.quat_to_exp_map(
                loaded_dict["human_rot"].view(-1, 4)
            ).view(-1, 52 * 3)
            loaded_dict["human_rot_vel"] = (
                human_rot_exp_map[1:, :].clone() - human_rot_exp_map[:-1, :].clone()
            ) * self.fps_data
            loaded_dict["human_rot_vel"] = torch.cat(
                (
                    torch.zeros((1, loaded_dict["human_rot_vel"].shape[-1])).to(
                        self.device
                    ),
                    loaded_dict["human_rot_vel"],
                ),
                dim=0,
            )
            loaded_dict["hoi_data"] = torch.cat(
                (
                    loaded_dict["root_pos"].clone(),
                    loaded_dict["root_rot"].clone(),
                    loaded_dict["dof_pos"].clone(),
                    loaded_dict["dof_pos_vel"].clone(),
                    loaded_dict["obj_pos"].clone(),
                    loaded_dict["obj_rot"].clone(),
                    loaded_dict["obj_pos_vel"].clone(),
                    loaded_dict["obj_rot_vel"].clone(),
                    loaded_dict["key_body_pos"][:, :].clone(),
                    loaded_dict["contact"].clone(),
                    loaded_dict["contact_parts"].clone(),
                    ref_ig.clone(),
                    loaded_dict["human_rot"].clone(),
                    loaded_dict["key_body_pos_vel"].clone(),
                    loaded_dict["human_rot_vel"],
                ),
                dim=-1,
            )
            assert self.ref_hoi_obs_size == loaded_dict["hoi_data"].shape[-1]
            hoi_datas.append(loaded_dict["hoi_data"])
            hoi_ref = torch.cat(
                (
                    loaded_dict["root_pos"].clone(),
                    loaded_dict["root_rot"].clone(),
                    loaded_dict["dof_pos"].clone(),
                    loaded_dict["dof_pos_vel"].clone(),
                    loaded_dict["root_pos_vel"].clone(),
                    loaded_dict["root_rot_vel"].clone(),
                    loaded_dict["obj_pos"].clone(),
                    loaded_dict["obj_rot"].clone(),
                    loaded_dict["obj_pos_vel"].clone(),
                    loaded_dict["obj_rot_vel"].clone(),
                ),
                dim=-1,
            )
            hoi_refs.append(hoi_ref)
        max_length = max(self.max_episode_length)
        self.num_motions = len(hoi_refs)
        self.max_episode_length = to_torch(
            self.max_episode_length, dtype=torch.long, device=self.device
        )
        self.hoi_data = []
        self.hoi_refs = []
        for i, data in enumerate(hoi_datas):
            pad_size = (0, 0, 0, max_length - data.size(0))
            padded_data = F.pad(data, pad_size, "constant", 0)
            self.hoi_data.append(padded_data)
            self.hoi_refs.append(F.pad(hoi_refs[i], pad_size, "constant", 0))
        self.hoi_data = torch.stack(self.hoi_data, dim=0)
        self.hoi_data_envs = torch.zeros(
            [self.num_envs, self.hoi_data.shape[1], self.hoi_data.shape[2]]
        ).to(self.hoi_data.device)
        self.hoi_refs = torch.stack(self.hoi_refs, dim=0).unsqueeze(1)
        self.ref_reward = torch.zeros(
            (self.hoi_refs.shape[0], self.hoi_refs.shape[1], self.hoi_refs.shape[2])
        ).to(self.hoi_refs.device)
        self.ref_reward[:, 0, :] = 1.0
        self.ref_index = torch.zeros((self.num_envs,)).long().to(self.hoi_refs.device)
        return

    def _load_motion_raw(self, motion_file):
        hoi_data_dict = []
        hoi_datas = []
        hoi_refs = []
        if not isinstance(motion_file, list):
            motion_file = [motion_file]
        for idx, data_path in enumerate(motion_file):
            loaded_dict = {}
            hoi_data = load_motion_tensor(data_path)
            loaded_dict["hoi_data"] = hoi_data[1:].detach().to(self.device)
            if idx == 0:
                print(loaded_dict["hoi_data"].shape, data_path)
            max_episode_length = loaded_dict["hoi_data"].shape[0]
            loaded_dict["root_pos"] = loaded_dict["hoi_data"][:, 0:3].clone()
            loaded_dict["root_pos_vel"] = (
                loaded_dict["root_pos"][1:, :].clone()
                - loaded_dict["root_pos"][:-1, :].clone()
            ) * self.fps_data
            loaded_dict["root_pos_vel"] = torch.cat(
                (
                    torch.zeros((1, loaded_dict["root_pos_vel"].shape[-1])).to(
                        self.device
                    ),
                    loaded_dict["root_pos_vel"],
                ),
                dim=0,
            )
            loaded_dict["root_rot"] = loaded_dict["hoi_data"][:, 3:7].clone()
            root_rot_exp_map = torch_utils.quat_to_exp_map(loaded_dict["root_rot"])
            loaded_dict["root_rot_vel"] = (
                root_rot_exp_map[1:, :].clone() - root_rot_exp_map[:-1, :].clone()
            ) * self.fps_data
            loaded_dict["root_rot_vel"] = torch.cat(
                (
                    torch.zeros((1, loaded_dict["root_rot_vel"].shape[-1])).to(
                        self.device
                    ),
                    loaded_dict["root_rot_vel"],
                ),
                dim=0,
            )
            loaded_dict["dof_pos"] = loaded_dict["hoi_data"][:, 9 : 9 + 153].clone()
            loaded_dict["dof_pos_vel"] = []
            loaded_dict["dof_pos_vel"] = (
                loaded_dict["dof_pos"][1:, :].clone()
                - loaded_dict["dof_pos"][:-1, :].clone()
            ) * self.fps_data
            loaded_dict["dof_pos_vel"] = torch.cat(
                (
                    torch.zeros((1, loaded_dict["dof_pos_vel"].shape[-1])).to(
                        self.device
                    ),
                    loaded_dict["dof_pos_vel"],
                ),
                dim=0,
            )
            loaded_dict["body_pos"] = (
                loaded_dict["hoi_data"][:, 162 : 162 + 52 * 3]
                .clone()
                .view(max_episode_length, 52, 3)
            )
            loaded_dict["key_body_pos"] = (
                loaded_dict["body_pos"][:, self._key_body_ids, :]
                .view(max_episode_length, -1)
                .clone()
            )
            loaded_dict["key_body_pos_vel"] = (
                loaded_dict["key_body_pos"][1:, :].clone()
                - loaded_dict["key_body_pos"][:-1, :].clone()
            ) * self.fps_data
            loaded_dict["key_body_pos_vel"] = torch.cat(
                (
                    torch.zeros((1, loaded_dict["key_body_pos_vel"].shape[-1])).to(
                        self.device
                    ),
                    loaded_dict["key_body_pos_vel"],
                ),
                dim=0,
            )
            loaded_dict["obj_pos"] = loaded_dict["hoi_data"][:, 318:321].clone()
            loaded_dict["obj_pos_vel"] = (
                loaded_dict["obj_pos"][1:, :].clone()
                - loaded_dict["obj_pos"][:-1, :].clone()
            ) * self.fps_data
            if self.init_vel:
                loaded_dict["obj_pos_vel"] = torch.cat(
                    (loaded_dict["obj_pos_vel"][:1], loaded_dict["obj_pos_vel"]), dim=0
                )
            else:
                loaded_dict["obj_pos_vel"] = torch.cat(
                    (
                        torch.zeros((1, loaded_dict["obj_pos_vel"].shape[-1])).to(
                            self.device
                        ),
                        loaded_dict["obj_pos_vel"],
                    ),
                    dim=0,
                )
            loaded_dict["obj_rot"] = loaded_dict["hoi_data"][:, 321:325].clone()
            obj_rot_exp_map = torch_utils.quat_to_exp_map(loaded_dict["obj_rot"])
            loaded_dict["obj_rot_vel"] = (
                obj_rot_exp_map[1:, :].clone() - obj_rot_exp_map[:-1, :].clone()
            ) * self.fps_data
            loaded_dict["obj_rot_vel"] = torch.cat(
                (
                    torch.zeros((1, loaded_dict["obj_rot_vel"].shape[-1])).to(
                        self.device
                    ),
                    loaded_dict["obj_rot_vel"],
                ),
                dim=0,
            )
            obj_rot_extend = (
                loaded_dict["obj_rot"]
                .unsqueeze(1)
                .repeat(1, self.object_points[self.object_id[idx]].shape[0], 1)
                .view(-1, 4)
            )
            object_points_extend = (
                self.object_points[self.object_id[idx]]
                .unsqueeze(0)
                .repeat(loaded_dict["obj_rot"].shape[0], 1, 1)
                .view(-1, 3)
            )
            obj_points = torch_utils.quat_rotate(
                obj_rot_extend, object_points_extend
            ).view(
                loaded_dict["obj_rot"].shape[0],
                self.object_points[self.object_id[idx]].shape[0],
                3,
            ) + loaded_dict[
                "obj_pos"
            ].unsqueeze(
                1
            )
            key_body_pose = loaded_dict["key_body_pos"][:, :].clone()
            ref_ig = compute_sdf(
                key_body_pose.view(loaded_dict["obj_rot"].shape[0], -1, 3), obj_points
            ).view(-1, 3)
            heading_rot = torch_utils.calc_heading_quat_inv(loaded_dict["root_rot"])
            heading_rot_extend = (
                heading_rot.unsqueeze(1)
                .repeat(1, key_body_pose.shape[1] // 3, 1)
                .view(-1, 4)
            )
            ref_ig = quat_rotate(heading_rot_extend, ref_ig).view(
                loaded_dict["obj_rot"].shape[0], -1
            )
            loaded_dict["contact"] = torch.round(
                loaded_dict["hoi_data"][:, 330:331].clone()
            )
            loaded_dict["contact_parts"] = torch.round(
                loaded_dict["hoi_data"][:, 331 : 331 + 52].clone()
            )
            loaded_dict["human_rot"] = loaded_dict["hoi_data"][
                :, 331 + 52 : 331 + 52 + 52 * 4
            ].clone()
            human_rot_exp_map = torch_utils.quat_to_exp_map(
                loaded_dict["human_rot"].view(-1, 4)
            ).view(-1, 52 * 3)
            loaded_dict["human_rot_vel"] = (
                human_rot_exp_map[1:, :].clone() - human_rot_exp_map[:-1, :].clone()
            ) * self.fps_data
            loaded_dict["human_rot_vel"] = torch.cat(
                (
                    torch.zeros((1, loaded_dict["human_rot_vel"].shape[-1])).to(
                        self.device
                    ),
                    loaded_dict["human_rot_vel"],
                ),
                dim=0,
            )
            loaded_dict["hoi_data"] = torch.cat(
                (
                    loaded_dict["root_pos"].clone(),
                    loaded_dict["root_rot"].clone(),
                    loaded_dict["dof_pos"].clone(),
                    loaded_dict["dof_pos_vel"].clone(),
                    loaded_dict["obj_pos"].clone(),
                    loaded_dict["obj_rot"].clone(),
                    loaded_dict["obj_pos_vel"].clone(),
                    loaded_dict["obj_rot_vel"].clone(),
                    loaded_dict["key_body_pos"][:, :].clone(),
                    loaded_dict["contact"].clone(),
                    loaded_dict["contact_parts"].clone(),
                    ref_ig.clone(),
                    loaded_dict["human_rot"].clone(),
                    loaded_dict["key_body_pos_vel"].clone(),
                    loaded_dict["human_rot_vel"],
                ),
                dim=-1,
            )
            hoi_data_dict.append(loaded_dict)
            hoi_datas.append(loaded_dict["hoi_data"])
        self.hoi_data_raw = []
        max_length = max(self.max_episode_length)
        for i, data in enumerate(hoi_datas):
            pad_size = (0, 0, 0, max_length - data.size(0))
            padded_data = F.pad(data, pad_size, "constant", 0)
            self.hoi_data_raw.append(padded_data)
        self.hoi_data_raw = torch.stack(self.hoi_data_raw, dim=0)
        return

    def _create_envs(self, num_envs, spacing, num_per_row):
        self._target_handles = []
        self._load_target_asset()
        if self.projtype == "Mouse" or self.projtype == "Auto":
            self._proj_handles = []
            self._load_proj_asset()
        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        self._build_target(env_id, env_ptr)
        if self.projtype == "Mouse" or self.projtype == "Auto":
            self._build_proj(env_id, env_ptr)
        return

    def _build_proj(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0
        for i, obj in enumerate(PERTURB_OBJS):
            default_pose = gymapi.Transform()
            default_pose.p.x = 200 + i
            default_pose.p.z = 1
            obj_type = obj[0]
            if obj_type == "small":
                proj_asset = self._small_proj_asset
            elif obj_type == "large":
                proj_asset = self._large_proj_asset
            proj_handle = self.gym.create_actor(
                env_ptr,
                proj_asset,
                default_pose,
                "proj{:d}".format(i),
                col_group,
                col_filter,
                segmentation_id,
            )
            self._proj_handles.append(proj_handle)
            self.gym.set_actor_scale(env_ptr, proj_handle, 1)
        return

    def _build_proj_tensors(self):
        self._proj_dist_min = 4
        self._proj_dist_max = 5
        self._proj_h_min = 0.25
        self._proj_h_max = 2
        self._proj_steps = 150
        self._proj_warmup_steps = 1
        self._proj_speed_min = 30
        self._proj_speed_max = 40
        num_actors = self.get_num_actors_per_env()
        num_objs = len(PERTURB_OBJS)
        self._proj_states = self._root_states.view(
            self.num_envs, num_actors, self._root_states.shape[-1]
        )[..., num_actors - num_objs :, :]
        self._proj_actor_ids = num_actors * np.arange(self.num_envs)
        self._proj_actor_ids = np.expand_dims(self._proj_actor_ids, axis=-1)
        self._proj_actor_ids = self._proj_actor_ids + np.reshape(
            np.array(self._proj_handles), [self.num_envs, num_objs]
        )
        self._proj_actor_ids = self._proj_actor_ids.flatten()
        self._proj_actor_ids = to_torch(
            self._proj_actor_ids, device=self.device, dtype=torch.int32
        )
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._proj_contact_forces = contact_force_tensor.view(
            self.num_envs, bodies_per_env, 3
        )[..., num_actors - num_objs :, :]
        self._calc_perturb_times()
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_SPACE, "space_shoot"
        )
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "reset")
        self.gym.subscribe_viewer_mouse_event(
            self.viewer, gymapi.MOUSE_LEFT_BUTTON, "mouse_shoot"
        )
        return

    def _load_proj_asset(self):
        asset_root = self.cfg["env"]["asset"]["objectRoot"] + os.sep
        small_asset_file = "block_projectile.urdf"
        small_asset_options = gymapi.AssetOptions()
        small_asset_options.angular_damping = 0.01
        small_asset_options.linear_damping = 0.01
        small_asset_options.max_angular_velocity = 100.0
        small_asset_options.density = 200.0
        small_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        self._small_proj_asset = self.gym.load_asset(
            self.sim, asset_root, small_asset_file, small_asset_options
        )
        return

    def _load_target_asset(self):
        asset_root = self.cfg["env"]["asset"]["objectRoot"] + os.sep
        self._target_asset = []
        points_num = []
        self.object_points = []
        self.object_normals = []
        for i, object_name in enumerate(self.object_name):
            asset_file = object_name + ".urdf"
            obj_file = (
                asset_root + "objects/" + object_name + "/" + object_name + ".obj"
            )
            max_convex_hulls = 64
            density = 200
            asset_options = gymapi.AssetOptions()
            asset_options.angular_damping = 0.01
            asset_options.linear_damping = 0.01
            asset_options.density = density
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            asset_options.vhacd_enabled = True
            asset_options.vhacd_params.max_convex_hulls = max_convex_hulls
            asset_options.vhacd_params.max_num_vertices_per_ch = 64
            asset_options.vhacd_params.resolution = 300000
            self._target_asset.append(
                self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
            )
            mesh_obj = trimesh.load(obj_file, process=False, force="mesh")
            obj_verts = mesh_obj.vertices
            center = np.mean(obj_verts, 0)
            object_points, object_faces = trimesh.sample.sample_surface_even(
                mesh_obj, count=512, seed=2024
            )
            object_normals = to_torch(
                mesh_obj.face_normals[object_faces], device=self.device
            )
            object_points = to_torch(object_points - center, device=self.device)
            while object_points.shape[0] < 512:
                object_points = torch.cat(
                    [object_points, object_points[: 512 - object_points.shape[0]]],
                    dim=0,
                )
                object_normals = torch.cat(
                    [object_normals, object_normals[: 512 - object_normals.shape[0]]],
                    dim=0,
                )
            self.object_points.append(object_points)
            self.object_normals.append(object_normals)
        self.object_points = torch.stack(self.object_points, dim=0)
        self.object_normals = torch.stack(self.object_normals, dim=0)
        return

    def _build_target(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 2
        segmentation_id = 0
        default_pose = gymapi.Transform()
        target_handle = self.gym.create_actor(
            env_ptr,
            self._target_asset[int(self.env_object_ids[env_id])],
            default_pose,
            self.object_name[int(self.env_object_ids[env_id])],
            col_group,
            col_filter,
            segmentation_id,
        )
        props = self.gym.get_actor_rigid_shape_properties(env_ptr, target_handle)
        for p_idx in range(len(props)):
            props[p_idx].restitution = 0.6
            props[p_idx].friction = 0.8
            props[p_idx].rolling_friction = 0.01
            props[p_idx].torsion_friction = 0.8
        self.gym.set_actor_rigid_shape_properties(env_ptr, target_handle, props)
        self._target_handles.append(target_handle)
        self.gym.set_actor_scale(env_ptr, target_handle, self.ball_size)
        return

    def _build_target_tensors(self):
        num_actors = self.get_num_actors_per_env()
        self._target_states = self._root_states.view(
            self.num_envs, num_actors, self._root_states.shape[-1]
        )[..., 1, :]
        self._tar_actor_ids = (
            to_torch(
                num_actors * np.arange(self.num_envs),
                device=self.device,
                dtype=torch.int32,
            )
            + 1
        )
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._tar_contact_forces = contact_force_tensor.view(
            self.num_envs, bodies_per_env, 3
        )[..., self.num_bodies, :]
        return

    def _push_robots(self):
        """Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
        max_vel = 1.0
        self._humanoid_root_states[:, 7:9] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device
        )
        self._target_states[:, 7:9] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device
        )
        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self._root_states)
        )

    def _reset_target(self, env_ids):
        self._target_states[env_ids, :3] = self.hoi_refs[
            self.data_id[env_ids],
            self.ref_index[env_ids],
            self.progress_buf[env_ids],
            319:322,
        ]
        self._target_states[env_ids, 3:7] = self.hoi_refs[
            self.data_id[env_ids],
            self.ref_index[env_ids],
            self.progress_buf[env_ids],
            322:326,
        ]
        self._target_states[env_ids, 7:10] = self.hoi_refs[
            self.data_id[env_ids],
            self.ref_index[env_ids],
            self.progress_buf[env_ids],
            326:329,
        ]
        self._target_states[env_ids, 10:13] = self.hoi_refs[
            self.data_id[env_ids],
            self.ref_index[env_ids],
            self.progress_buf[env_ids],
            329:332,
        ]
        if not self.inference_mode:
            rand_pos = 0.1 * (2 * torch.rand((len(env_ids), 3), device=self.device) - 1)
            rand_linear_vel = 0.1 * (
                2 * torch.rand((len(env_ids), 3), device=self.device) - 1
            )
            rand_angular_vel = 0.1 * (
                2 * torch.rand((len(env_ids), 3), device=self.device) - 1
            )
            self._target_states[env_ids, 0:3] += rand_pos
            self._target_states[env_ids, 7:10] += rand_linear_vel
            self._target_states[env_ids, 10:13] += rand_angular_vel
        return

    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)
        env_ids_int32 = self._tar_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )
        return

    def _reset_envs(self, env_ids):
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []
        super()._reset_envs(env_ids)
        return

    def _reset_actors(self, env_ids):
        if self._state_init == InterPriorTask.StateInit.Default:
            self._reset_default(env_ids)
        elif (
            self._state_init == InterPriorTask.StateInit.Start
            or self._state_init == InterPriorTask.StateInit.Random
        ):
            self._reset_ref_state_init(env_ids)
        elif self._state_init == InterPriorTask.StateInit.Hybrid:
            self._reset_hybrid_state_init(env_ids)
        else:
            assert False, "Unsupported state initialization strategy: {:s}".format(
                str(self._state_init)
            )
        self._reset_target(env_ids)
        return

    def _reset_default(self, env_ids):
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[
            env_ids
        ]
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        self._reset_default_env_ids = env_ids
        return

    def _reset_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        i = (
            self.motion_ids[env_ids]
            if self.inference_mode and hasattr(self, "motion_ids")
            else to_torch(
                [
                    torch.where(self.obj2motion[i % len(self.object_name)] == 1)[0][
                        torch.randint(
                            self.obj2motion[i % len(self.object_name)].sum(), ()
                        )
                    ]
                    for i in env_ids
                ],
                device=self.device,
                dtype=torch.long,
            )
        )
        if (
            self._state_init == InterPriorTask.StateInit.Random
            or self._state_init == InterPriorTask.StateInit.Hybrid
        ):
            motion_times = torch.cat(
                [
                    torch.randint(
                        0,
                        max(1, self.max_episode_length[i[e]]),
                        (1,),
                        device=self.device,
                        dtype=torch.long,
                    )
                    for e in range(num_envs)
                ]
            )
        elif self._state_init == InterPriorTask.StateInit.Start:
            motion_times = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        if self.inference_mode and hasattr(self, "reference_tracker"):
            motion_times = self.reference_tracker.start_frames[env_ids].clone()
        ref_reward = self.ref_reward[i, :, motion_times]
        prob = ref_reward / ref_reward.sum(1, keepdim=True)
        cdf = torch.cumsum(prob, dim=1)
        idx = torch.searchsorted(
            cdf, torch.rand((cdf.shape[0], 1)).to(cdf.device)
        ).squeeze(1)
        self.ref_index[env_ids] = idx
        self.progress_buf[env_ids] = motion_times.clone()
        self.progress_buf_ref[env_ids] = motion_times.clone()
        self.start_times[env_ids] = motion_times.clone()
        self.data_id[env_ids] = i
        self.dataset_id[env_ids] = self.dataset_index[self.data_id[env_ids]]
        self._hist_obs[env_ids] = 0
        self._hist_obs_all[env_ids] = 0
        self.contact_reset[env_ids] = 0
        self.mask[env_ids] = 1
        self.mask_candidate[env_ids] = 1
        self.mask_mask[env_ids] = 0
        self.task_mask_mask[env_ids] = 0
        self.long_term_t[env_ids] = 0
        self.long_term_goal[env_ids] = 0
        self.all_0[env_ids] = 0
        self.require_mask[env_ids] = 0
        self.fall_state[env_ids] = 0
        self._fall_on_ctr[env_ids] = 0
        self._fall_off_ctr[env_ids] = 0
        self._task_accomplish[env_ids] = 0
        self.hoi_data_envs[env_ids, :, :] = self.hoi_data[
            self.data_id[env_ids], :, :
        ].clone()
        if not self.inference_mode:
            ii = self.data_id[env_ids]
            idxi = self.ref_index[env_ids]
            tti = self.progress_buf[env_ids]
            base_pos = self.hoi_refs[ii, idxi, tti, 0:3]
            base_rot = self.hoi_refs[ii, idxi, tti, 3:7]
            base_dpos = self.hoi_refs[ii, idxi, tti, 7:160]
            base_rv = self.hoi_refs[ii, idxi, tti, 313:316]
            base_wv = self.hoi_refs[ii, idxi, tti, 316:319]
            base_dv = self.hoi_refs[ii, idxi, tti, 160:313]
            R = env_ids.shape[0]
            xy_jit = 0.2 * (2 * torch.rand((R, 2), device=self.device) - 1.0)
            z_lift = 0.0 * torch.rand((R, 1), device=self.device)
            yaw = np.pi / 6 * (2 * torch.rand((R,), device=self.device) - 1.0)
            pert_pos = base_pos.clone()
            pert_pos[:, :2] = pert_pos[:, :2] + xy_jit
            pert_pos[:, 2:3] = (pert_pos[:, 2:3] + z_lift).clamp_min(0.01)
            half = 0.5 * yaw
            yaw_q = torch.stack(
                [
                    torch.zeros_like(half),
                    torch.zeros_like(half),
                    torch.sin(half),
                    torch.cos(half),
                ],
                dim=-1,
            )
            pert_rot = torch_utils.quat_mul(yaw_q, base_rot)
            small = 0.05
            pert_rv = small * (2 * torch.rand_like(base_rv) - 1.0)
            pert_wv = small * (2 * torch.rand_like(base_wv) - 1.0)
            pert_dv = small * (2 * torch.rand_like(base_dv) - 1.0)
            self._set_env_state(
                env_ids=env_ids,
                root_pos=pert_pos,
                root_rot=pert_rot,
                dof_pos=base_dpos,
                root_vel=pert_rv,
                root_ang_vel=pert_wv,
                dof_vel=pert_dv,
            )
        else:
            self._set_env_state(
                env_ids=env_ids,
                root_pos=self.hoi_refs[i, idx, motion_times, 0:3],
                root_rot=self.hoi_refs[i, idx, motion_times, 3:7],
                dof_pos=self.hoi_refs[i, idx, motion_times, 7:160],
                root_vel=self.hoi_refs[i, idx, motion_times, 313:316],
                root_ang_vel=self.hoi_refs[i, idx, motion_times, 316:319],
                dof_vel=self.hoi_refs[i, idx, motion_times, 160:313],
            )
        return

    def cal_cdf(self, i, e):
        rewards = self.ref_reward[
            i[e], :, : max(1, self.max_episode_length[i[e]])
        ].clone()
        ref_reward_sum = 1 / rewards.sum(dim=0)
        prob = ref_reward_sum / ref_reward_sum.sum()
        cdf = torch.cumsum(prob, 0)
        return cdf

    def _reset_fake(self, env_ids):
        num_envs = env_ids.shape[0]
        i = to_torch(
            [
                torch.where(self.obj2motion[i % len(self.object_name)] == 1)[0][
                    torch.randint(self.obj2motion[i % len(self.object_name)].sum(), ())
                ]
                for i in env_ids
            ],
            device=self.device,
            dtype=torch.long,
        )
        motion_times = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.ref_index[env_ids] = 0
        self.progress_buf[env_ids] = motion_times.clone()
        self.progress_buf_ref[env_ids] = motion_times.clone() - torch.randint_like(
            env_ids, 32
        )
        self.start_times[env_ids] = motion_times.clone()
        self.data_id[env_ids] = i
        self.dataset_id[env_ids] = self.dataset_index[self.data_id[env_ids]]
        self._hist_obs[env_ids] = 0
        self._hist_obs_all[env_ids] = 0
        self.contact_reset[env_ids] = 0
        self.mask[env_ids] = 1
        self.mask_candidate[env_ids] = 1
        self.mask_mask[env_ids] = 0
        self.task_mask_mask[env_ids] = 0
        self.long_term_t[env_ids] = 0
        self.long_term_goal[env_ids] = 0
        self.all_0[env_ids] = 0
        self.require_mask[env_ids] = 0
        self.fall_state[env_ids] = 0
        self._fall_on_ctr[env_ids] = 0
        self._fall_off_ctr[env_ids] = 0
        self._task_accomplish[env_ids] = 0
        ref_joints = self.hoi_data[
            self.data_id[env_ids], :, 326 : 326 + len(self._key_body_ids) * 3
        ].clone()
        ref_obj = self.hoi_data[self.data_id[env_ids], :, 313:316].clone()
        ref_root_pos_t = self.hoi_data[self.data_id[env_ids], 0, 326:329].clone()
        cur_root_pos = self._rigid_body_pos[env_ids, 0]
        sim_joints_t = self._rigid_body_pos[:, self._key_body_ids, :][env_ids]
        cur_root_rot = self._rigid_body_rot[env_ids, 0]
        ref_root_rot = self.hoi_data[
            self.data_id[env_ids],
            0,
            326
            + len(self._key_body_ids) * 3
            + 1
            + 52
            + len(self._key_body_ids) * 3 : 326
            + len(self._key_body_ids) * 3
            + 1
            + 52
            + len(self._key_body_ids) * 3
            + 4,
        ].clone()
        aligned_joints, aligned_obj = self.align_ref_seq_to_current(
            cur_root_rot,
            ref_root_rot,
            ref_joints,
            ref_obj,
            ref_root_pos_t,
            cur_root_pos,
            sim_joints_t,
            len_keypos=len(self._key_body_ids),
        )
        self.hoi_data_envs[env_ids] = self.hoi_data[self.data_id[env_ids]].clone()
        self.hoi_data_envs[env_ids, :, 326 : 326 + len(self._key_body_ids) * 3] = (
            aligned_joints
        )
        self.hoi_data_envs[env_ids, :, 313:316] = aligned_obj
        max_err = (
            (
                aligned_joints.view(1, -1, len(self._key_body_ids), 3)[:, 0]
                - sim_joints_t
            )
            .abs()
            .max()
            .item()
        )
        return

    def _reset_hybrid_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        i = to_torch(
            [
                torch.where(self.obj2motion[i % len(self.object_name)] == 1)[0][
                    torch.randint(self.obj2motion[i % len(self.object_name)].sum(), ())
                ]
                for i in env_ids
            ],
            device=self.device,
            dtype=torch.long,
        )
        ref_probs = to_torch(
            np.array([self._hybrid_init_prob] * num_envs), device=self.device
        )
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0
        ref_reset_ids = env_ids[ref_init_mask]
        motion_times = torch.cat(
            [
                (
                    torch.searchsorted(
                        self.cal_cdf(i, e), torch.rand(1).to(self.device)
                    )
                    if env_ids[e] not in ref_reset_ids
                    else torch.zeros((1,), device=self.device, dtype=torch.long)
                )
                for e in range(num_envs)
            ]
        )
        ref_reward = self.ref_reward[i, :, motion_times]
        prob = ref_reward / ref_reward.sum(1, keepdim=True)
        cdf = torch.cumsum(prob, dim=1)
        idx = torch.searchsorted(
            cdf, torch.rand((cdf.shape[0], 1)).to(cdf.device)
        ).squeeze(1)
        self.ref_index[env_ids] = idx
        self.progress_buf[env_ids] = motion_times.clone()
        self.progress_buf_ref[env_ids] = motion_times.clone()
        self.start_times[env_ids] = motion_times.clone()
        self.data_id[env_ids] = i
        self.dataset_id[env_ids] = self.dataset_index[self.data_id[env_ids]]
        self._hist_obs[env_ids] = 0
        self._hist_obs_all[env_ids] = 0
        self.contact_reset[env_ids] = 0
        self.mask[env_ids] = 1
        self.mask_candidate[env_ids] = 1
        self.mask_mask[env_ids] = 0
        self.task_mask_mask[env_ids] = 0
        self.long_term_t[env_ids] = 0
        self.long_term_goal[env_ids] = 0
        self.all_0[env_ids] = 0
        self.require_mask[env_ids] = 0
        self.fall_state[env_ids] = 0
        self._fall_on_ctr[env_ids] = 0
        self._fall_off_ctr[env_ids] = 0
        self._task_accomplish[env_ids] = 0
        self.hoi_data_envs[env_ids, :, :] = self.hoi_data[
            self.data_id[env_ids], :, :
        ].clone()
        self._set_env_state(
            env_ids=env_ids,
            root_pos=self.hoi_refs[i, idx, motion_times, 0:3],
            root_rot=self.hoi_refs[i, idx, motion_times, 3:7],
            dof_pos=self.hoi_refs[i, idx, motion_times, 7:160],
            root_vel=self.hoi_refs[i, idx, motion_times, 313:316],
            root_ang_vel=self.hoi_refs[i, idx, motion_times, 316:319],
            dof_vel=self.hoi_refs[i, idx, motion_times, 160:313],
        )
        explore_ids = env_ids
        fall_probs = to_torch(np.full(explore_ids.shape[0], 0.0), device=self.device)
        fall_init_mask = torch.bernoulli(fall_probs) == 1.0
        self.fall_reset_ids = explore_ids[fall_init_mask]
        self.rand_reset_ids = explore_ids[~fall_init_mask]
        if hasattr(self, "fall_hoi_ref"):
            max_k = self.fall_hoi_ref.shape[0] - 1
            self.fall_reset_k = random.randint(0, max(0, max_k))
        if self.fall_reset_ids.numel() > 0 and hasattr(self, "fall_hoi_ref"):
            self.progress_buf_ref[self.fall_reset_ids] -= (
                torch.randint_like(self.fall_reset_ids, 32) + 32
            )
            self._set_env_state(
                env_ids=self.fall_reset_ids,
                root_pos=self.fall_hoi_ref[self.fall_reset_k, self.fall_reset_ids, 0:3],
                root_rot=self.fall_hoi_ref[self.fall_reset_k, self.fall_reset_ids, 3:7],
                dof_pos=self.fall_hoi_ref[
                    self.fall_reset_k, self.fall_reset_ids, 7:160
                ],
                root_vel=self.fall_hoi_ref[
                    self.fall_reset_k, self.fall_reset_ids, 313:316
                ],
                root_ang_vel=self.fall_hoi_ref[
                    self.fall_reset_k, self.fall_reset_ids, 316:319
                ],
                dof_vel=self.fall_hoi_ref[
                    self.fall_reset_k, self.fall_reset_ids, 160:313
                ],
            )
        if self.rand_reset_ids.numel() > 0:
            ii = self.data_id[self.rand_reset_ids]
            idxi = self.ref_index[self.rand_reset_ids]
            tti = self.progress_buf[self.rand_reset_ids]
            base_pos = self.hoi_refs[ii, idxi, tti, 0:3]
            base_rot = self.hoi_refs[ii, idxi, tti, 3:7]
            base_dpos = self.hoi_refs[ii, idxi, tti, 7:160]
            base_rv = self.hoi_refs[ii, idxi, tti, 313:316]
            base_wv = self.hoi_refs[ii, idxi, tti, 316:319]
            base_dv = self.hoi_refs[ii, idxi, tti, 160:313]
            R = self.rand_reset_ids.shape[0]
            xy_jit = 0.05 * (2 * torch.rand((R, 2), device=self.device) - 1.0)
            z_lift = 0.02 * torch.rand((R, 1), device=self.device)
            yaw = np.pi / 12 * (2 * torch.rand((R,), device=self.device) - 1.0)
            pert_pos = base_pos.clone()
            pert_pos[:, :2] = pert_pos[:, :2] + xy_jit
            pert_pos[:, 2:3] = (pert_pos[:, 2:3] + z_lift).clamp_min(0.05)
            half = 0.5 * yaw
            yaw_q = torch.stack(
                [
                    torch.zeros_like(half),
                    torch.zeros_like(half),
                    torch.sin(half),
                    torch.cos(half),
                ],
                dim=-1,
            )
            pert_rot = torch_utils.quat_mul(yaw_q, base_rot)
            small = 0.05
            pert_rv = small * (2 * torch.rand_like(base_rv) - 1.0)
            pert_wv = small * (2 * torch.rand_like(base_wv) - 1.0)
            pert_dv = small * (2 * torch.rand_like(base_dv) - 1.0)
            self._set_env_state(
                env_ids=self.rand_reset_ids,
                root_pos=pert_pos,
                root_rot=pert_rot,
                dof_pos=base_dpos,
                root_vel=pert_rv,
                root_ang_vel=pert_wv,
                dof_vel=pert_dv,
            )
        return

    def _set_env_state(
        self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel
    ):
        self._humanoid_root_states[env_ids, 0:3] = root_pos
        self._humanoid_root_states[env_ids, 3:7] = root_rot
        self._humanoid_root_states[env_ids, 7:10] = root_vel
        self._humanoid_root_states[env_ids, 10:13] = root_ang_vel
        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel
        return

    def _compute_hoi_observations(self, env_ids=None):
        key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
        key_body_vel = self._rigid_body_vel[:, self._key_body_ids, :]
        if env_ids is None:
            self._curr_obs[:] = build_hoi_observations(
                self._rigid_body_pos[:, 0, :],
                self._rigid_body_rot[:, 0, :],
                self._rigid_body_vel[:, 0, :],
                self._rigid_body_ang_vel[:, 0, :],
                self._dof_pos,
                self._dof_vel,
                key_body_pos,
                self._local_root_obs,
                self._root_height_obs,
                self._dof_obs_size,
                self._target_states,
                self._tar_contact_forces,
                self._contact_forces,
                self.object_points[self.object_id[self.data_id]],
                self._rigid_body_rot,
                key_body_vel,
                self._rigid_body_ang_vel,
            )
        else:
            self._curr_obs[env_ids] = build_hoi_observations(
                self._rigid_body_pos[env_ids][:, 0, :],
                self._rigid_body_rot[env_ids][:, 0, :],
                self._rigid_body_vel[env_ids][:, 0, :],
                self._rigid_body_ang_vel[env_ids][:, 0, :],
                self._dof_pos[env_ids],
                self._dof_vel[env_ids],
                key_body_pos[env_ids],
                self._local_root_obs,
                self._root_height_obs,
                self._dof_obs_size,
                self._target_states[env_ids],
                self._tar_contact_forces[env_ids],
                self._contact_forces[env_ids],
                self.object_points[self.object_id[self.data_id[env_ids]]],
                self._rigid_body_rot[env_ids],
                key_body_vel[env_ids],
                self._rigid_body_ang_vel[env_ids],
            ).float()
        return

    def _calc_perturb_times(self):
        self._perturb_timesteps = []
        total_steps = 0
        for i, obj in enumerate(PERTURB_OBJS):
            curr_time = obj[1]
            total_steps += curr_time
            self._perturb_timesteps.append(total_steps)
        self._perturb_timesteps = np.array(self._perturb_timesteps)
        return

    def _update_proj(self):
        if self.projtype == "Auto":
            curr_timestep = self.progress_buf.cpu().numpy()[0]
            curr_timestep = curr_timestep % (self._perturb_timesteps[-1] + 1)
            perturb_step = np.where(self._perturb_timesteps == curr_timestep)[0]
            if len(perturb_step) > 0:
                perturb_id = perturb_step[0]
                n = self.num_envs
                humanoid_root_pos = self._humanoid_root_states[..., 0:3]
                rand_theta = torch.rand(
                    [n], dtype=self._proj_states.dtype, device=self._proj_states.device
                )
                rand_theta *= 2 * np.pi
                rand_dist = (self._proj_dist_max - self._proj_dist_min) * torch.rand(
                    [n], dtype=self._proj_states.dtype, device=self._proj_states.device
                ) + self._proj_dist_min
                pos_x = rand_dist * torch.cos(rand_theta)
                pos_y = -rand_dist * torch.sin(rand_theta)
                pos_z = (self._proj_h_max - self._proj_h_min) * torch.rand(
                    [n], dtype=self._proj_states.dtype, device=self._proj_states.device
                ) + self._proj_h_min
                self._proj_states[..., perturb_id, 0] = (
                    humanoid_root_pos[..., 0] + pos_x
                )
                self._proj_states[..., perturb_id, 1] = (
                    humanoid_root_pos[..., 1] + pos_y
                )
                self._proj_states[..., perturb_id, 2] = pos_z
                self._proj_states[..., perturb_id, 3:6] = 0.0
                self._proj_states[..., perturb_id, 6] = 1.0
                tar_body_idx = np.random.randint(self.num_bodies)
                tar_body_idx = 1
                launch_tar_pos = self._rigid_body_pos[..., tar_body_idx, :]
                launch_dir = launch_tar_pos - self._proj_states[..., perturb_id, 0:3]
                launch_dir += 0.1 * torch.randn_like(launch_dir)
                launch_dir = torch.nn.functional.normalize(launch_dir, dim=-1)
                launch_speed = (
                    self._proj_speed_max - self._proj_speed_min
                ) * torch.rand_like(launch_dir[:, 0:1]) + self._proj_speed_min
                launch_vel = launch_speed * launch_dir
                launch_vel[..., 0:2] += self._rigid_body_vel[..., tar_body_idx, 0:2]
                self._proj_states[..., perturb_id, 7:10] = launch_vel
                self._proj_states[..., perturb_id, 10:13] = 0.0
                self.gym.set_actor_root_state_tensor_indexed(
                    self.sim,
                    gymtorch.unwrap_tensor(self._root_states),
                    gymtorch.unwrap_tensor(self._proj_actor_ids),
                    len(self._proj_actor_ids),
                )
        elif self.projtype == "Mouse":
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "reset" and evt.value > 0:
                    self.gym.set_sim_rigid_body_states(
                        self.sim, self._proj_states, gymapi.STATE_ALL
                    )
                elif (
                    evt.action == "space_shoot" or evt.action == "mouse_shoot"
                ) and evt.value > 0:
                    if evt.action == "mouse_shoot":
                        pos = self.gym.get_viewer_mouse_position(self.viewer)
                        window_size = self.gym.get_viewer_size(self.viewer)
                        xcoord = round(pos.x * window_size.x)
                        ycoord = round(pos.y * window_size.y)
                        print(
                            f"Fired projectile with mouse at coords: {xcoord} {ycoord}"
                        )
                    cam_pose = self.gym.get_viewer_camera_transform(self.viewer, None)
                    cam_fwd = cam_pose.r.rotate(gymapi.Vec3(0, 0, 1))
                    spawn = cam_pose.p
                    speed = 25
                    vel = cam_fwd * speed
                    angvel = 1.57 - 3.14 * np.random.random(3)
                    self._proj_states[..., 0] = spawn.x
                    self._proj_states[..., 1] = spawn.y
                    self._proj_states[..., 2] = spawn.z
                    self._proj_states[..., 7] = vel.x
                    self._proj_states[..., 8] = vel.y
                    self._proj_states[..., 9] = vel.z
                    self.gym.set_actor_root_state_tensor_indexed(
                        self.sim,
                        gymtorch.unwrap_tensor(self._root_states),
                        gymtorch.unwrap_tensor(self._proj_actor_ids),
                        len(self._proj_actor_ids),
                    )
        return

    def play_dataset_step(self, time):
        t = time
        if t == 0:
            self.data_id = to_torch(
                [
                    torch.where(self.obj2motion[i % len(self.object_name)] == 1)[0][
                        torch.randint(
                            self.obj2motion[i % len(self.object_name)].sum(), ()
                        )
                    ]
                    for i in range(self.num_envs)
                ],
                device=self.device,
                dtype=torch.long,
            )
        env_ids = to_torch(
            [
                i
                for i in range(self.num_envs)
                if t < self.max_episode_length[self.data_id[i]]
            ],
            device=self.device,
            dtype=torch.long,
        )
        self._target_states[env_ids, :3] = self.hoi_refs[
            self.data_id[env_ids], 0, t, 319:322
        ]
        self._target_states[env_ids, 3:7] = self.hoi_refs[
            self.data_id[env_ids], 0, t, 322:326
        ]
        self._target_states[env_ids, 7:10] = torch.zeros_like(
            self._target_states[env_ids, 7:10]
        )
        self._target_states[env_ids, 10:13] = torch.zeros_like(
            self._target_states[env_ids, 10:13]
        )
        _humanoid_root_pos = self.hoi_refs[self.data_id[env_ids], 0, t, 0:3]
        _humanoid_root_rot = self.hoi_refs[self.data_id[env_ids], 0, t, 3:7]
        self._humanoid_root_states[env_ids, 0:3] = _humanoid_root_pos
        self._humanoid_root_states[env_ids, 3:7] = _humanoid_root_rot
        self._humanoid_root_states[:, 7:10] = torch.zeros_like(
            self._humanoid_root_states[:, 7:10]
        )
        self._humanoid_root_states[:, 10:13] = torch.zeros_like(
            self._humanoid_root_states[:, 10:13]
        )
        self._dof_pos[env_ids] = self.hoi_refs[self.data_id[env_ids], 0, t, 7:160]
        self._dof_vel[env_ids] = self.hoi_refs[self.data_id[env_ids], 0, t, 160:313]
        env_ids_int32 = self._humanoid_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )
        env_ids_int32 = self._tar_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )
        self._refresh_sim_tensors()
        for env_id, env_ptr in enumerate(self.envs):
            if env_id in env_ids:
                contact = self.hoi_data_dict[self.data_id[env_id]]["contact"][t, :]
                obj_contact = torch.any(contact > 0.1, dim=-1)
                env_ptr = self.envs[env_id]
                handle = self._target_handles[env_id]
                if obj_contact:
                    self.gym.set_rigid_body_color(
                        env_ptr,
                        handle,
                        0,
                        gymapi.MESH_VISUAL,
                        gymapi.Vec3(1.0, 0.0, 0.0),
                    )
                else:
                    self.gym.set_rigid_body_color(
                        env_ptr,
                        handle,
                        0,
                        gymapi.MESH_VISUAL,
                        gymapi.Vec3(0.0, 0.0, 1.0),
                    )
        self.render(t=t)
        self.gym.simulate(self.sim)
        return

    def _draw_task_play(self, t):
        cols = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        self.gym.clear_lines(self.viewer)
        starts = self.hoi_data_dict[0]["hoi_data"][t, :3]
        for i, env_ptr in enumerate(self.envs):
            for j in range(len(self._key_body_ids)):
                vec = self.hoi_data_dict[0]["key_body_pos"][t, j * 3 : j * 3 + 3]
                vec = torch.cat([starts, vec], dim=-1).cpu().numpy().reshape([1, 6])
                self.gym.add_lines(self.viewer, env_ptr, 1, vec, cols)
        return

    def render(self, sync_frame_time=False, t=0):
        super().render(sync_frame_time)
        if self.viewer:
            self._draw_task()
            if self.save_images:
                env_ids = 0
                if self.play_dataset:
                    frame_id = t
                else:
                    frame_id = self.t
                self.t += 1
                dataname = self.motion_file[-1][
                    len("interprior/data/motions/BallPlay/") :
                ]
                rgb_filename = (
                    "interprior/data/images/"
                    + dataname
                    + "/rgb_env%d_frame%05d.png" % (env_ids, frame_id)
                )
                os.makedirs("interprior/data/images/" + dataname, exist_ok=True)
                self.gym.write_viewer_image_to_file(self.viewer, rgb_filename)
        return

    def _draw_task(self):
        return

    def _cache_rb_mass_and_com(self):
        env0, actor0 = (self.envs[0], self.humanoid_handles[0])
        props = self.gym.get_actor_rigid_body_properties(env0, actor0)
        rb_mass = torch.tensor(
            [p.mass for p in props], device=self.device, dtype=torch.float
        )
        rb_comL = torch.tensor(
            [[p.com.x, p.com.y, p.com.z] for p in props],
            device=self.device,
            dtype=torch.float,
        )
        self._rb_mass_exp = rb_mass.view(1, -1, 1).expand(self.num_envs, -1, -1)
        self._rb_comL_exp = rb_comL.view(1, -1, 3).expand(self.num_envs, -1, -1)
        self._rb_mass_sum = rb_mass.sum().clamp_min(1e-08)

    def compute_com_pos(self):
        com_off_w = quat_rotate(
            self._rigid_body_rot.reshape(-1, 4), self._rb_comL_exp.reshape(-1, 3)
        ).view(self.num_envs, self.num_bodies, 3)
        com_pts = self._rigid_body_pos + com_off_w
        com_pos = (com_pts * self._rb_mass_exp).sum(dim=1) / self._rb_mass_sum
        return com_pos


def build_hoi_observations(
    root_pos,
    root_rot,
    root_vel,
    root_ang_vel,
    dof_pos,
    dof_vel,
    key_body_pos,
    local_root_obs,
    root_height_obs,
    dof_obs_size,
    target_states,
    target_contact_buf,
    contact_buf,
    object_points,
    body_rot,
    body_vel,
    body_rot_vel,
):
    contact = torch.any(torch.abs(contact_buf) > 0.1, dim=-1).float()
    target_contact = (
        torch.any(torch.abs(target_contact_buf) > 0.1, dim=-1).float().unsqueeze(1)
    )
    tar_pos = target_states[:, 0:3]
    tar_rot = target_states[:, 3:7]
    obj_rot_extend = (
        tar_rot.unsqueeze(1).repeat(1, object_points.shape[1], 1).view(-1, 4)
    )
    object_points_extend = object_points.view(-1, 3)
    obj_points = torch_utils.quat_rotate(obj_rot_extend, object_points_extend).view(
        tar_rot.shape[0], object_points.shape[1], 3
    ) + tar_pos.unsqueeze(1)
    ig = compute_sdf(key_body_pos, obj_points).view(-1, 3)
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    heading_rot_extend = (
        heading_rot.unsqueeze(1).repeat(1, key_body_pos.shape[1], 1).view(-1, 4)
    )
    ig = quat_rotate(heading_rot_extend, ig).view(tar_pos.shape[0], -1)
    obs = torch.cat(
        (
            root_pos,
            root_rot,
            dof_pos,
            dof_vel,
            target_states,
            key_body_pos.contiguous().view(
                -1, key_body_pos.shape[1] * key_body_pos.shape[2]
            ),
            target_contact,
            contact,
            ig,
            body_rot.view(-1, 52 * 4),
            body_vel.view(-1, key_body_pos.shape[1] * key_body_pos.shape[2]),
            body_rot_vel.view(-1, 52 * 3),
        ),
        dim=-1,
    )
    return obs


def compute_obj_observations(root_states, tar_states, object_points, ref_obs):

    # type: (Tensor, Tensor, Tensor, Tensor) -> (Tensor, Tensor)
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    tar_pos = tar_states[:, 0:3]
    tar_rot = tar_states[:, 3:7]
    tar_vel = tar_states[:, 7:10]
    tar_ang_vel = tar_states[:, 10:13]
    obj_rot_extend = (
        tar_rot.unsqueeze(1).repeat(1, object_points.shape[1], 1).view(-1, 4)
    )
    object_points_extend = object_points.view(-1, 3)
    obj_points = torch_utils.quat_rotate(obj_rot_extend, object_points_extend).view(
        tar_rot.shape[0], object_points.shape[1], 3
    ) + tar_pos.unsqueeze(1)
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    heading_inv_rot = torch_utils.calc_heading_quat(root_rot)
    local_tar_pos = tar_pos - root_pos
    local_tar_pos[..., -1] = tar_pos[..., -1]
    local_tar_pos = quat_rotate(heading_rot, local_tar_pos)
    local_tar_vel = quat_rotate(heading_rot, tar_vel)
    local_tar_ang_vel = quat_rotate(heading_rot, tar_ang_vel)
    local_tar_rot = quat_mul(heading_rot, tar_rot)
    local_tar_rot_obs = torch_utils.quat_to_tan_norm(local_tar_rot)
    _ref_obj_pos = ref_obs[:, 313:316]
    diff_global_obj_pos = _ref_obj_pos - tar_pos
    diff_local_obj_pos_flat = torch_utils.quat_rotate(heading_rot, diff_global_obj_pos)
    local_ref_obj_pos = _ref_obj_pos - root_pos
    local_ref_obj_pos = torch_utils.quat_rotate(heading_rot, local_ref_obj_pos)
    ref_obj_rot = ref_obs[:, 316:320]
    diff_global_obj_rot = torch_utils.quat_mul_norm(
        torch_utils.quat_inverse(ref_obj_rot), tar_rot
    )
    diff_local_obj_rot_flat = torch_utils.quat_mul(
        torch_utils.quat_mul(heading_rot, diff_global_obj_rot.view(-1, 4)),
        heading_inv_rot,
    )
    diff_local_obj_rot_obs = torch_utils.quat_to_tan_norm(diff_local_obj_rot_flat)
    local_ref_obj_rot = torch_utils.quat_mul(heading_rot, ref_obj_rot)
    local_ref_obj_rot = torch_utils.quat_to_tan_norm(local_ref_obj_rot)
    ref_obj_vel = ref_obs[:, 320:323]
    diff_global_vel = ref_obj_vel - tar_vel
    diff_local_vel = torch_utils.quat_rotate(heading_rot, diff_global_vel)
    ref_obj_ang_vel = ref_obs[:, 323:326]
    diff_global_ang_vel = ref_obj_ang_vel - tar_ang_vel
    diff_local_ang_vel = torch_utils.quat_rotate(heading_rot, diff_global_ang_vel)
    obs = torch.cat(
        [
            local_tar_vel,
            local_tar_ang_vel,
            diff_local_obj_pos_flat,
            diff_local_obj_rot_obs,
            diff_local_vel,
            diff_local_ang_vel,
        ],
        dim=-1,
    )
    return (obs, obj_points)


@torch.jit.script
def compute_humanoid_observations_max(
    body_pos,
    body_rot,
    body_vel,
    body_ang_vel,
    local_root_obs,
    root_height_obs,
    contact_forces,
    contact_body_ids,
    ref_obs,
    key_body_ids,
):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, bool, Tensor, Tensor, Tensor, Tensor) -> Tensor
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]
    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    heading_inv_rot = torch_utils.calc_heading_quat(root_rot)
    if not root_height_obs:
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h
    len_keypos = len(key_body_ids)
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand_2 = heading_rot_expand.repeat((1, len_keypos, 1))
    flat_heading_rot_2 = heading_rot_expand_2.reshape(
        heading_rot_expand_2.shape[0] * heading_rot_expand_2.shape[1],
        heading_rot_expand_2.shape[2],
    )
    heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot = heading_rot_expand.reshape(
        heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
        heading_rot_expand.shape[2],
    )
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand_no_hand = heading_rot_expand.repeat((1, 22, 1))
    flat_heading_rot_no_hand = heading_rot_expand_no_hand.reshape(
        heading_rot_expand_no_hand.shape[0] * heading_rot_expand_no_hand.shape[1],
        heading_rot_expand_no_hand.shape[2],
    )
    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2)
    heading_inv_rot_expand = heading_inv_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_inv_rot = heading_inv_rot_expand.reshape(
        heading_inv_rot_expand.shape[0] * heading_inv_rot_expand.shape[1],
        heading_inv_rot_expand.shape[2],
    )
    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2)
    heading_inv_rot_expand_no_hand = heading_inv_rot_expand.repeat((1, 22, 1))
    flat_heading_inv_rot_no_hand = heading_inv_rot_expand_no_hand.reshape(
        heading_inv_rot_expand_no_hand.shape[0]
        * heading_inv_rot_expand_no_hand.shape[1],
        heading_inv_rot_expand_no_hand.shape[2],
    )
    _ref_body_pos = ref_obs[:, 326 : 326 + len_keypos * 3].view(-1, len_keypos, 3)
    _body_pos = body_pos[:, key_body_ids, :]
    diff_global_body_pos = _ref_body_pos - _body_pos
    diff_local_body_pos_flat = torch_utils.quat_rotate(
        flat_heading_rot_2, diff_global_body_pos.view(-1, 3)
    ).view(-1, len_keypos * 3)
    local_ref_body_pos = _body_pos - root_pos.unsqueeze(1)
    local_ref_body_pos = torch_utils.quat_rotate(
        flat_heading_rot_2, local_ref_body_pos.view(-1, 3)
    ).view(-1, len_keypos * 3)
    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(
        local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2]
    )
    flat_local_body_pos = quat_rotate(flat_heading_rot, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(
        local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2]
    )
    local_body_pos = local_body_pos[..., 3:]
    flat_body_rot = body_rot.reshape(
        body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2]
    )
    flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot)
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(
        body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1]
    )
    ref_body_rot = ref_obs[
        :,
        326
        + len_keypos * 3
        + 1
        + 52
        + len_keypos * 3 : 326
        + len_keypos * 3
        + 1
        + 52
        + len_keypos * 3
        + 52 * 4,
    ]
    ref_body_rot_no_hand = torch.cat(
        (ref_body_rot[:, : 18 * 4], ref_body_rot[:, 33 * 4 : 37 * 4]), dim=-1
    )
    body_rot_no_hand = torch.cat((body_rot[:, :18], body_rot[:, 33:37]), dim=1)
    diff_global_body_rot = torch_utils.quat_mul_norm(
        torch_utils.quat_inverse(ref_body_rot_no_hand.reshape(-1, 4)),
        body_rot_no_hand.reshape(-1, 4),
    )
    diff_local_body_rot_flat = torch_utils.quat_mul(
        torch_utils.quat_mul(
            flat_heading_rot_no_hand, diff_global_body_rot.view(-1, 4)
        ),
        flat_heading_inv_rot_no_hand,
    )
    diff_local_body_rot_obs = torch_utils.quat_to_tan_norm(diff_local_body_rot_flat)
    diff_local_body_rot_obs = diff_local_body_rot_obs.view(
        body_rot_no_hand.shape[0],
        body_rot_no_hand.shape[1] * diff_local_body_rot_obs.shape[-1],
    )
    local_ref_body_rot = torch_utils.quat_mul(
        flat_heading_rot_no_hand, ref_body_rot_no_hand.reshape(-1, 4)
    )
    local_ref_body_rot = torch_utils.quat_to_tan_norm(local_ref_body_rot).view(
        ref_body_rot_no_hand.shape[0], -1
    )
    ref_body_vel = ref_obs[
        :,
        326
        + len_keypos * 3
        + 1
        + 52
        + len_keypos * 3
        + 52 * 4 : 326
        + len_keypos * 3
        + 1
        + 52
        + len_keypos * 3
        + 52 * 4
        + len_keypos * 3,
    ].view(-1, len_keypos, 3)
    _body_vel = body_vel[:, key_body_ids, :]
    diff_global_vel = ref_body_vel - _body_vel
    diff_local_vel = torch_utils.quat_rotate(
        flat_heading_rot_2, diff_global_vel.view(-1, 3)
    ).view(-1, len_keypos * 3)
    ref_body_ang_vel = ref_obs[
        :,
        326
        + len_keypos * 3
        + 1
        + 52
        + len_keypos * 3
        + 52 * 4
        + len_keypos * 3 : 326
        + len_keypos * 3
        + 1
        + 52
        + len_keypos * 3
        + 52 * 4
        + len_keypos * 3
        + 52 * 3,
    ]
    ref_body_ang_vel_no_hand = torch.cat(
        (ref_body_ang_vel[:, : 18 * 3], ref_body_ang_vel[:, 33 * 3 : 37 * 3]), dim=-1
    )
    body_ang_vel_no_hand = torch.cat(
        (body_ang_vel[:, :18], body_ang_vel[:, 33:37]), dim=1
    )
    diff_global_ang_vel = (
        ref_body_ang_vel_no_hand.view(-1, 22, 3) - body_ang_vel_no_hand
    )
    diff_local_ang_vel = torch_utils.quat_rotate(
        flat_heading_rot_no_hand, diff_global_ang_vel.view(-1, 3)
    ).view(-1, 22 * 3)
    if local_root_obs:
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot)
        local_body_rot_obs[..., 0:6] = root_rot_obs
    flat_body_vel = body_vel.reshape(
        body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2]
    )
    flat_local_body_vel = quat_rotate(flat_heading_rot, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(
        body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2]
    )
    flat_body_ang_vel = body_ang_vel.reshape(
        body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2]
    )
    flat_local_body_ang_vel = quat_rotate(flat_heading_rot, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(
        body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2]
    )
    body_contact_buf = contact_forces[:, contact_body_ids, :].clone()
    contact = torch.any(torch.abs(body_contact_buf) > 0.1, dim=-1).float()
    ref_body_contact = ref_obs[
        :, 326 + len_keypos * 3 + 1 : 326 + len_keypos * 3 + 1 + 52
    ][:, contact_body_ids]
    diff_body_contact = ref_body_contact * ((ref_body_contact + 1) / 2 - contact)
    obs = torch.cat(
        (
            root_h_obs,
            local_body_pos,
            local_body_rot_obs,
            local_body_vel,
            local_body_ang_vel,
            contact,
            diff_local_body_pos_flat,
            diff_local_body_rot_obs,
            diff_body_contact,
            local_ref_body_pos,
            local_ref_body_rot,
            diff_local_vel,
            diff_local_ang_vel,
        ),
        dim=-1,
    )
    return obs


def compute_obj_observations_decouple(
    root_states, tar_states, object_points, ref_obs, long_term, ft_idx
):
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    tar_pos = tar_states[:, 0:3]
    tar_pos_original = tar_pos.detach().clone()
    local_tar_pos = tar_pos - root_pos
    if long_term:
        r_xy = local_tar_pos[..., :2].norm(dim=-1, keepdim=True)
        scale = torch.clamp(1.5 / (r_xy + 1e-08), max=1.0)
        local_tar_pos_xy = local_tar_pos[..., :2] * scale
        local_tar_pos = torch.cat([local_tar_pos_xy, local_tar_pos[..., 2:3]], dim=-1)
        tar_pos = local_tar_pos + root_pos
    local_tar_pos[..., -1] = tar_pos[..., -1]
    tar_rot = tar_states[:, 3:7]
    tar_vel = tar_states[:, 7:10]
    tar_ang_vel = tar_states[:, 10:13]
    obj_rot_extend = (
        tar_rot.unsqueeze(1).repeat(1, object_points.shape[1], 1).view(-1, 4)
    )
    object_points_extend = object_points.view(-1, 3)
    obj_points = torch_utils.quat_rotate(obj_rot_extend, object_points_extend).view(
        tar_rot.shape[0], object_points.shape[1], 3
    ) + tar_pos.unsqueeze(1)
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    heading_inv_rot = torch_utils.calc_heading_quat(root_rot)
    local_tar_pos = quat_rotate(heading_rot, local_tar_pos)
    local_tar_vel = quat_rotate(heading_rot, tar_vel)
    local_tar_ang_vel = quat_rotate(heading_rot, tar_ang_vel)
    local_tar_rot = quat_mul(heading_rot, tar_rot)
    local_tar_rot_obs = torch_utils.quat_to_tan_norm(local_tar_rot)
    _ref_obj_pos = ref_obs[:, 313:316]
    diff_global_obj_pos = _ref_obj_pos - tar_pos_original
    if long_term:
        r_xy = diff_global_obj_pos[..., :2].norm(dim=-1, keepdim=True)
        scale = torch.clamp(1.0 / (r_xy + 1e-08), max=1.0)
        diff_global_obj_pos_xy = diff_global_obj_pos[..., :2] * scale
        diff_global_obj_pos = torch.cat(
            [diff_global_obj_pos_xy, diff_global_obj_pos[..., 2:3]], dim=-1
        )
    diff_local_obj_pos_flat = torch_utils.quat_rotate(heading_rot, diff_global_obj_pos)
    local_ref_obj_pos = _ref_obj_pos - root_pos
    local_ref_obj_pos = torch_utils.quat_rotate(heading_rot, local_ref_obj_pos)
    ref_obj_rot = ref_obs[:, 316:320]
    diff_global_obj_rot = torch_utils.quat_mul_norm(
        torch_utils.quat_inverse(ref_obj_rot), tar_rot
    )
    diff_local_obj_rot_flat = torch_utils.quat_mul(
        torch_utils.quat_mul(heading_rot, diff_global_obj_rot.view(-1, 4)),
        heading_inv_rot,
    )
    diff_local_obj_rot_obs = torch_utils.quat_to_tan_norm(diff_local_obj_rot_flat)
    local_ref_obj_rot = torch_utils.quat_mul(heading_rot, ref_obj_rot)
    local_ref_obj_rot = torch_utils.quat_to_tan_norm(local_ref_obj_rot)
    ref_obj_vel = ref_obs[:, 320:323]
    diff_global_vel = ref_obj_vel - tar_vel
    diff_local_vel = torch_utils.quat_rotate(heading_rot, diff_global_vel)
    ref_obj_ang_vel = ref_obs[:, 323:326]
    diff_global_ang_vel = ref_obj_ang_vel - tar_ang_vel
    diff_local_ang_vel = torch_utils.quat_rotate(heading_rot, diff_global_ang_vel)
    obs = torch.cat([local_tar_vel, local_tar_ang_vel], dim=-1)
    goal = torch.cat(
        [
            diff_local_obj_pos_flat,
            diff_local_obj_rot_obs,
            diff_local_vel,
            diff_local_ang_vel,
        ],
        dim=-1,
    )
    diff_local_obj_pos_flat_mask = (
        (torch.rand([obs.shape[0], 1]) < 0.5)
        .float()
        .repeat(1, diff_local_obj_pos_flat.shape[1])
        .to(diff_local_obj_pos_flat.device)
    )
    diff_local_obj_rot_obs_mask = (
        (torch.rand([obs.shape[0], 1]) < 0.5)
        .float()
        .repeat(1, diff_local_obj_rot_obs.shape[1])
        .to(diff_local_obj_pos_flat.device)
    )
    diff_local_vel_mask = (torch.rand([obs.shape[0], 1]) < 0.1).float().to(
        diff_local_obj_pos_flat.device
    ) * diff_local_obj_pos_flat_mask
    diff_local_ang_vel_mask = (torch.rand([obs.shape[0], 1]) < 0.1).float().to(
        diff_local_obj_pos_flat.device
    ) * diff_local_obj_rot_obs_mask[..., :3]
    mask = torch.cat(
        [
            diff_local_obj_pos_flat_mask,
            diff_local_obj_rot_obs_mask,
            diff_local_vel_mask,
            diff_local_ang_vel_mask,
        ],
        dim=-1,
    )
    if ft_idx:
        diff_local_obj_pos_flat_mask[:] = 1
        diff_local_obj_rot_obs_mask[:] = 0
        diff_local_vel_mask[:] = 0
        diff_local_ang_vel_mask[:] = 0
    assert goal.shape == mask.shape
    return (obs, goal, obj_points, mask)


def compute_humanoid_observations_max_decouple(
    body_pos,
    body_rot,
    body_vel,
    body_ang_vel,
    local_root_obs,
    root_height_obs,
    contact_forces,
    contact_body_ids,
    ref_obs,
    key_body_ids,
    long_term,
    ft_idx,
):
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]
    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    heading_inv_rot = torch_utils.calc_heading_quat(root_rot)
    root_h_obs = root_h
    len_keypos = len(key_body_ids)
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand_2 = heading_rot_expand.repeat((1, len_keypos, 1))
    flat_heading_rot_2 = heading_rot_expand_2.reshape(
        heading_rot_expand_2.shape[0] * heading_rot_expand_2.shape[1],
        heading_rot_expand_2.shape[2],
    )
    heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot = heading_rot_expand.reshape(
        heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
        heading_rot_expand.shape[2],
    )
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand_no_hand = heading_rot_expand.repeat((1, 22, 1))
    flat_heading_rot_no_hand = heading_rot_expand_no_hand.reshape(
        heading_rot_expand_no_hand.shape[0] * heading_rot_expand_no_hand.shape[1],
        heading_rot_expand_no_hand.shape[2],
    )
    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2)
    heading_inv_rot_expand = heading_inv_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_inv_rot = heading_inv_rot_expand.reshape(
        heading_inv_rot_expand.shape[0] * heading_inv_rot_expand.shape[1],
        heading_inv_rot_expand.shape[2],
    )
    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2)
    heading_inv_rot_expand_no_hand = heading_inv_rot_expand.repeat((1, 22, 1))
    flat_heading_inv_rot_no_hand = heading_inv_rot_expand_no_hand.reshape(
        heading_inv_rot_expand_no_hand.shape[0]
        * heading_inv_rot_expand_no_hand.shape[1],
        heading_inv_rot_expand_no_hand.shape[2],
    )
    _ref_body_pos = ref_obs[:, 326 : 326 + len_keypos * 3].view(-1, len_keypos, 3)
    _body_pos = body_pos[:, key_body_ids, :]
    diff_global_body_pos = _ref_body_pos - _body_pos
    if long_term:
        r_xy = diff_global_body_pos.norm(dim=-1, keepdim=True)
        scale = torch.clamp(1.0 / (r_xy + 1e-08), max=1.0).max(dim=1, keepdim=True)[0]
        diff_global_body_pos = diff_global_body_pos * scale
    diff_local_body_pos_flat = torch_utils.quat_rotate(
        flat_heading_rot_2, diff_global_body_pos.view(-1, 3)
    ).view(-1, len_keypos * 3)
    local_ref_body_pos = _ref_body_pos - root_pos.unsqueeze(1)
    local_ref_body_pos = torch_utils.quat_rotate(
        flat_heading_rot_2, local_ref_body_pos.view(-1, 3)
    ).view(-1, len_keypos * 3)
    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(
        local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2]
    )
    flat_local_body_pos = quat_rotate(flat_heading_rot, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(
        local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2]
    )
    local_body_pos = local_body_pos[..., 3:]
    flat_body_rot = body_rot.reshape(
        body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2]
    )
    flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot)
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(
        body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1]
    )
    ref_body_rot = ref_obs[
        :,
        326
        + len_keypos * 3
        + 1
        + 52
        + len_keypos * 3 : 326
        + len_keypos * 3
        + 1
        + 52
        + len_keypos * 3
        + 52 * 4,
    ]
    ref_body_rot_no_hand = torch.cat(
        (ref_body_rot[:, : 18 * 4], ref_body_rot[:, 33 * 4 : 37 * 4]), dim=-1
    )
    body_rot_no_hand = torch.cat((body_rot[:, :18], body_rot[:, 33:37]), dim=1)
    diff_global_body_rot = torch_utils.quat_mul_norm(
        torch_utils.quat_inverse(ref_body_rot_no_hand.reshape(-1, 4)),
        body_rot_no_hand.reshape(-1, 4),
    )
    diff_local_body_rot_flat = torch_utils.quat_mul(
        torch_utils.quat_mul(
            flat_heading_rot_no_hand, diff_global_body_rot.view(-1, 4)
        ),
        flat_heading_inv_rot_no_hand,
    )
    diff_local_body_rot_obs = torch_utils.quat_to_tan_norm(diff_local_body_rot_flat)
    diff_local_body_rot_obs = diff_local_body_rot_obs.view(
        body_rot_no_hand.shape[0],
        body_rot_no_hand.shape[1] * diff_local_body_rot_obs.shape[-1],
    )
    local_ref_body_rot = torch_utils.quat_mul(
        flat_heading_rot_no_hand, ref_body_rot_no_hand.reshape(-1, 4)
    )
    local_ref_body_rot = torch_utils.quat_to_tan_norm(local_ref_body_rot).view(
        ref_body_rot_no_hand.shape[0], -1
    )
    ref_body_vel = ref_obs[
        :,
        326
        + len_keypos * 3
        + 1
        + 52
        + len_keypos * 3
        + 52 * 4 : 326
        + len_keypos * 3
        + 1
        + 52
        + len_keypos * 3
        + 52 * 4
        + len_keypos * 3,
    ].view(-1, len_keypos, 3)
    _body_vel = body_vel[:, key_body_ids, :]
    diff_global_vel = ref_body_vel - _body_vel
    diff_local_vel = torch_utils.quat_rotate(
        flat_heading_rot_2, diff_global_vel.view(-1, 3)
    ).view(-1, len_keypos * 3)
    ref_body_ang_vel = ref_obs[
        :,
        326
        + len_keypos * 3
        + 1
        + 52
        + len_keypos * 3
        + 52 * 4
        + len_keypos * 3 : 326
        + len_keypos * 3
        + 1
        + 52
        + len_keypos * 3
        + 52 * 4
        + len_keypos * 3
        + 52 * 3,
    ]
    ref_body_ang_vel_no_hand = torch.cat(
        (ref_body_ang_vel[:, : 18 * 3], ref_body_ang_vel[:, 33 * 3 : 37 * 3]), dim=-1
    )
    body_ang_vel_no_hand = torch.cat(
        (body_ang_vel[:, :18], body_ang_vel[:, 33:37]), dim=1
    )
    diff_global_ang_vel = (
        ref_body_ang_vel_no_hand.view(-1, 22, 3) - body_ang_vel_no_hand
    )
    diff_local_ang_vel = torch_utils.quat_rotate(
        flat_heading_rot_no_hand, diff_global_ang_vel.view(-1, 3)
    ).view(-1, 22 * 3)
    flat_body_vel = body_vel.reshape(
        body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2]
    )
    flat_local_body_vel = quat_rotate(flat_heading_rot, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(
        body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2]
    )
    flat_body_ang_vel = body_ang_vel.reshape(
        body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2]
    )
    flat_local_body_ang_vel = quat_rotate(flat_heading_rot, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(
        body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2]
    )
    body_contact_buf = contact_forces[:, contact_body_ids, :].clone()
    contact = torch.any(torch.abs(body_contact_buf) > 0.1, dim=-1).float()
    ref_body_contact = ref_obs[
        :, 326 + len_keypos * 3 + 1 : 326 + len_keypos * 3 + 1 + 52
    ][:, contact_body_ids]
    diff_body_contact = ref_body_contact * ((ref_body_contact + 1) / 2 - contact)
    obs = torch.cat(
        (
            root_h_obs,
            local_body_pos,
            local_body_rot_obs,
            local_body_vel,
            local_body_ang_vel,
            contact,
        ),
        dim=-1,
    )
    goal = torch.cat(
        (
            diff_local_body_pos_flat,
            diff_local_body_rot_obs,
            diff_body_contact,
            local_ref_body_pos,
            local_ref_body_rot,
            diff_local_vel,
            diff_local_ang_vel,
        ),
        dim=-1,
    )
    human_mask = (torch.rand([obs.shape[0], len_keypos]) < 0.1).float() * ft_idx + (
        1 - ft_idx
    ) * (torch.rand([obs.shape[0], len_keypos]) < 0.1).float()
    vel_mask = (torch.rand([obs.shape[0], len_keypos]) < 0.0).float() * ft_idx + (
        1 - ft_idx
    ) * (torch.rand([obs.shape[0], len_keypos]) < 0.1).float()
    contact_mask = (
        torch.rand([obs.shape[0], diff_body_contact.shape[1]]) < 0.1
    ).float() * ft_idx + (1 - ft_idx) * (
        torch.rand([obs.shape[0], diff_body_contact.shape[1]]) < 0.1
    ).float()
    diff_local_body_pos_flat_mask = (
        human_mask.unsqueeze(2)
        .repeat(1, 1, 3)
        .view(-1, len_keypos * 3)
        .to(diff_local_body_pos_flat.device)
    )
    diff_local_body_rot_obs_mask = torch.zeros_like(diff_local_body_rot_obs).float()
    diff_body_contact_mask = contact_mask.to(diff_local_body_pos_flat.device)
    local_ref_body_pos_mask = (
        diff_local_body_pos_flat_mask.detach()
        .clone()
        .to(diff_local_body_pos_flat.device)
    )
    local_ref_body_rot_mask = torch.zeros_like(local_ref_body_rot).float()
    diff_local_vel_mask = (
        (human_mask * vel_mask)
        .unsqueeze(2)
        .repeat(1, 1, 3)
        .view(-1, len_keypos * 3)
        .to(diff_local_body_pos_flat.device)
    )
    diff_local_ang_vel_mask = torch.zeros_like(diff_local_ang_vel).float()
    mask = torch.cat(
        [
            diff_local_body_pos_flat_mask,
            diff_local_body_rot_obs_mask,
            diff_body_contact_mask,
            local_ref_body_pos_mask,
            local_ref_body_rot_mask,
            diff_local_vel_mask,
            diff_local_ang_vel_mask,
        ],
        dim=-1,
    )
    assert goal.shape == mask.shape
    return (obs, goal, mask)


def huber_loss(diff, sigma=1.0):
    beta = 1.0 / sigma**2
    diff = torch.abs(diff)
    cond = diff < beta
    loss = torch.where(cond, 0.5 * diff**2 / beta, diff - 0.5 * beta)
    return loss


def compute_humanoid_reward(
    hoi_ref,
    hoi_obs,
    contact_buf,
    tar_contact_forces,
    len_keypos,
    w,
    actions,
    object_points,
    object_pos_action,
    object_rot_action,
    masked_repre,
    masked_goal,
    time_goal,
    _contact_body_ids,
    hoi_ref_goal,
    com_pos,
    fall_state,
):
    z_target = 0.6
    z_thresh = 0.1
    v_target = 0.5
    alpha = 1.0
    beta = 0.4
    tau_t = 20.0
    body_pos_mask = masked_repre[:, :63].view(-1, 21, 3)
    contact_mask = masked_repre[:, 195:226]
    object_pos_mask = masked_repre[:, 550:551]
    object_rot_mask = masked_repre[:, 553:554]
    ig_mask = masked_repre[:, 565:628].view(-1, 21, 3)[:, :, 0]
    body_pos_mask_goal = masked_goal[:, :63].view(-1, 21, 3)
    object_pos_mask_goal = masked_goal[:, 550:551]
    object_rot_mask_goal = masked_goal[:, 553:554]
    obj_pos = hoi_obs[:, 313 : 313 + 3]
    obj_rot = hoi_obs[:, 316 : 316 + 4]
    obj_vel = hoi_obs[:, 320:323]
    obj_rot_extend = (
        obj_rot.unsqueeze(1).repeat(1, object_points.shape[1], 1).view(-1, 4)
    )
    object_points_extend = object_points.view(-1, 3)
    obj_points = torch_utils.quat_rotate(obj_rot_extend, object_points_extend).view(
        obj_rot.shape[0], object_points.shape[1], 3
    ) + obj_pos.unsqueeze(1)
    key_pos = hoi_obs[:, 326 : 326 + len_keypos * 3]
    ig = key_pos.view(-1, len_keypos, 3).unsqueeze(2) - obj_points.unsqueeze(1)
    not_in_contact = ig.norm(dim=-1).min(dim=-1)[0].min(dim=-1)[0] > 0.05
    obj_move = obj_vel.norm(dim=-1) > 0.05
    ref_obj_pos = hoi_ref[:, 313 : 313 + 3]
    ref_obj_rot = hoi_ref[:, 316 : 316 + 4]
    ref_obj_rot_extend = (
        ref_obj_rot.unsqueeze(1).repeat(1, object_points.shape[1], 1).view(-1, 4)
    )
    ref_obj_points = torch_utils.quat_rotate(
        ref_obj_rot_extend, object_points_extend
    ).view(obj_rot.shape[0], object_points.shape[1], 3) + ref_obj_pos.unsqueeze(1)
    ref_key_pos = hoi_ref[:, 326 : 326 + len_keypos * 3]
    ref_ig = ref_key_pos.view(-1, len_keypos, 3).unsqueeze(
        2
    ) - ref_obj_points.unsqueeze(1)
    ref_human_contact = hoi_ref[
        :, 326 + len_keypos * 3 + 1 : 326 + len_keypos * 3 + 1 + 52
    ]
    goal_obj_pos = hoi_ref_goal[:, 313 : 313 + 3]
    goal_obj_rot = hoi_ref_goal[:, 316 : 316 + 4]
    goal_key_pos = hoi_ref_goal[:, 326 : 326 + len_keypos * 3]
    human_reset = (
        (ref_key_pos - key_pos).view(-1, len_keypos, 3) * body_pos_mask
    ).norm(dim=-1).max(dim=-1)[0] > 0.5
    object_reset = (
        (obj_points - ref_obj_points).norm(dim=-1) * object_pos_mask * object_rot_mask
    ).max(dim=-1)[0] > 0.5
    kinematic_reset = torch.logical_or(human_reset, object_reset)
    reset_ig_1 = (
        ((ig - ref_ig) ** 2).sum(dim=-1).sqrt()
        / torch.clamp((ref_ig**2).sum(dim=-1).sqrt(), min=1)
    ).max(dim=-1)[0].max(dim=-1)[0] > 0.5
    reset_ig_2 = (
        ((ig - ref_ig) ** 2).sum(dim=-1).sqrt()
        / torch.clamp((ig**2).sum(dim=-1).sqrt(), min=1)
    ).max(dim=-1)[0].max(dim=-1)[0] > 0.5
    reset_ig = torch.logical_or(reset_ig_1, reset_ig_2)
    vxyz = hoi_obs[
        :,
        326
        + len_keypos * 3
        + 1
        + 52
        + len_keypos * 3
        + 52 * 4 : 326
        + len_keypos * 3
        + 1
        + 52
        + len_keypos * 3
        + 52 * 4
        + len_keypos * 3,
    ]
    ep_goal = torch.sum(
        (
            (goal_key_pos.view(-1, len_keypos, 3) - key_pos.view(-1, len_keypos, 3))
            ** 2
            * body_pos_mask_goal
        )
        .sum(dim=-1)
        .sqrt(),
        dim=-1,
    ) / (body_pos_mask_goal.sum(dim=-1).sum(dim=-1) / 3 + 1e-05)
    G = goal_key_pos.view(-1, len_keypos, 3)
    K = key_pos.view(-1, len_keypos, 3)
    V = vxyz.view(-1, len_keypos, 3)
    pos_dist = (G - K) / ((G - K).norm(dim=-1, keepdim=True) + 1e-05)
    v_dist = V / (V.norm(dim=-1, keepdim=True) + 1e-05)
    v_0_cond = (V.norm(dim=-1).max(dim=-1)[0] > 0.5).float()
    mask_body = body_pos_mask_goal.to(pos_dist.dtype)
    weighted_sum = (pos_dist * v_dist * mask_body).sum(dim=(1, 2))
    denom = body_pos_mask_goal[:, :, 0].sum(dim=-1) + 1e-05
    score = (1.0 - weighted_sum / denom * v_0_cond) / 8 * denom
    ep_goal = ep_goal.clone()
    rp_goal = torch.exp(-score)
    joint_speed_thresh = 0.1
    joint_vel = hoi_obs[:, 160:313]
    joint_mean = joint_vel.abs().mean(dim=-1)
    almost_static = joint_mean < joint_speed_thresh
    stuck_now = almost_static
    energy = actions.pow(2).mean(dim=-1).mul(-w["eg1"]).exp()
    rb = energy * rp_goal
    eps = 1e-08
    pos_err = (goal_obj_pos - obj_pos) ** 2 * object_pos_mask_goal
    pos_term = pos_err.mean(dim=-1)
    has_obj_goal = pos_term > 0.2
    to_goal = goal_obj_pos - obj_pos
    to_goal_hat = to_goal / (to_goal.norm(dim=-1, keepdim=True) + eps)
    v_par = (obj_vel * to_goal_hat).sum(dim=-1, keepdim=True)
    v_perp = obj_vel - v_par * to_goal_hat
    gamma_perp = 0.5
    vel_term = (
        torch.relu(-v_par).squeeze(-1) ** 2 + gamma_perp * v_perp.norm(dim=-1) ** 2
    )
    w_pos = 1.0
    w_vel = 1.0
    eop_goal = w_vel * vel_term * object_pos_mask_goal.squeeze(dim=1)
    rop_goal = torch.exp(-eop_goal)
    obj_energy = (
        object_pos_action.pow(2).mean(dim=-1).mul(-w["eg2"]).exp()
        * object_rot_action.pow(2).mean(dim=-1).mul(-w["eg2"]).exp()
    )
    ro = obj_energy * rop_goal
    reward = ro / 2
    metric_1 = ep_goal < 0.1
    metric_2 = pos_term < 0.1
    return (reward, reset_ig, stuck_now, kinematic_reset, metric_1, metric_2)


def compute_humanoid_reset(
    reset_buf,
    progress_buf,
    obs_buf,
    contact_buf,
    rigid_body_pos,
    max_episode_length,
    enable_early_termination,
    termination_heights,
    hoi_ref,
    hoi_obs,
    start_times,
    rollout_length,
    reset_ig,
    contact_reset,
    device_id,
):
    terminated = torch.zeros_like(reset_buf)
    terminated_sup = torch.zeros_like(reset_buf)
    if enable_early_termination:
        body_height = rigid_body_pos[:, 0, 2]
        body_fall = body_height < termination_heights
        has_failed = body_fall.clone()
        has_failed *= progress_buf > 1
        reset_ig *= progress_buf > 1 + start_times
        contact_reset *= progress_buf > 1 + start_times
        invalid_obs = ~torch.isfinite(obs_buf)
        invalid_batches = torch.any(invalid_obs, dim=1)
        if torch.any(invalid_obs):
            print("invalid observation")
            raise Exception("invalid observation")
        terminated = torch.where(
            torch.logical_or(invalid_batches, torch.logical_or(has_failed, reset_ig)),
            torch.ones_like(reset_buf),
            terminated,
        )
        if not device_id:
            terminated[:] = 0
    reset = torch.where(
        torch.logical_or(
            progress_buf >= max_episode_length - 1,
            progress_buf - start_times >= rollout_length - 1,
        ),
        torch.ones_like(reset_buf),
        terminated,
    )
    return (reset, terminated)


@torch.jit.script
def compute_sdf(points1, points2):
    # type: (Tensor, Tensor) -> Tensor
    dis_mat = points1.unsqueeze(2) - points2.unsqueeze(1)
    dis_mat_lengths = torch.norm(dis_mat, dim=-1)
    min_length_indices = torch.argmin(dis_mat_lengths, dim=-1)
    B_indices, N_indices = torch.meshgrid(
        torch.arange(points1.shape[0]), torch.arange(points1.shape[1]), indexing="ij"
    )
    min_dis_mat = dis_mat[B_indices, N_indices, min_length_indices].contiguous()
    return min_dis_mat


def compute_sdf_normal(points1, points2, normals2):

    # type: (Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    dis_mat = points1.unsqueeze(2) - points2.unsqueeze(1)
    dis_mat_lengths = torch.norm(dis_mat, dim=-1)
    min_length_indices = torch.argmin(dis_mat_lengths, dim=-1)
    B_indices, N_indices = torch.meshgrid(
        torch.arange(points1.shape[0]), torch.arange(points1.shape[1]), indexing="ij"
    )
    min_dis_mat = dis_mat[B_indices, N_indices, min_length_indices].contiguous()
    selected_normals = normals2[B_indices, min_length_indices].contiguous()
    return (min_dis_mat, selected_normals)
