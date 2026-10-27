"""Goal spheres rendered as collision-free actors inside the Isaac Gym scene."""

from pathlib import Path

import torch

from .geometry import rotate


SPHERE_COUNT = 28  # 21 body targets, object center, 3 axes, 3 future centers.


def sphere_targets(reference, mask, future=None):
    """Return environment-space positions and visibility for native sphere actors."""
    positions = reference.new_full((len(reference), SPHERE_COUNT, 3), -1000)
    visible = torch.zeros(
        positions.shape[:2], device=reference.device, dtype=torch.bool
    )
    positions[:, :21] = reference[:, 326:389].reshape(-1, 21, 3)
    visible[:, :21] = mask[:, :63].reshape(-1, 21, 3).any(-1)
    positions[:, 21] = reference[:, 313:316]
    visible[:, 21] = mask[:, 550:553].any(-1)
    axes = torch.eye(3, device=reference.device)[None] * 0.18
    positions[:, 22:25] = (
        rotate(reference[:, None, 316:320], axes.expand(len(reference), -1, -1))
        + reference[:, None, 313:316]
    )
    visible[:, 22:25] = mask[:, 553:559].any(-1, keepdim=True)
    if future is not None:
        for i, (future_reference, future_mask) in enumerate(future):
            positions[:, 25 + i] = future_reference[:, 313:316]
            visible[:, 25 + i] = future_mask[:, 550:553].any(-1)
    positions[~visible] = -1000
    return positions, visible


def current_goals(task):
    """Show the exact long-term goal supplied to the policy, not the current pose."""
    program = task.reference_tracker
    ids = torch.arange(task.num_envs, device=task.device)
    reference = program.observation_reference(None, ids, task.long_term_t)
    return reference, program.base_mask


class GoalSpheres:
    """Create visual-only sphere geometry before prepare_sim, then move by tensor.

    The URDF has no collision element. Its zero rigid shapes are checked at load
    time, so spheres cannot touch the character, object, or ground. All actors
    have fixed bases and no DOFs; only their root poses are updated.
    """

    body_count = SPHERE_COUNT

    def __init__(self, gym, sim):
        from isaacgym import gymapi

        self.gym, self.sim = gym, sim
        options = gymapi.AssetOptions()
        options.fix_base_link = True
        options.disable_gravity = True
        folder = Path(__file__).parent / "assets"
        self.asset = gym.load_asset(sim, str(folder), "goal_sphere.urdf", options)
        if gym.get_asset_rigid_shape_count(self.asset) != 0:
            raise RuntimeError("Goal spheres must have zero collision shapes")
        self.handles = []

    def build(self, env, env_id):
        from isaacgym import gymapi

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(-1000, -1000, -1000)
        handles = []
        colors = [(0.05, 0.8, 1.0)] * 21 + [(1.0, 0.05, 0.5)]
        colors += [(1, 0.1, 0.1), (0.1, 1, 0.1), (0.1, 0.25, 1)]
        colors += [(0.65, 0.2, 0.75)] * 3
        scales = [0.8] * 21 + [1.35] + [0.55] * 3 + [0.5] * 3
        for i, (color, scale) in enumerate(zip(colors, scales)):
            handle = self.gym.create_actor(
                env, self.asset, pose, "goal_sphere_%d" % i, env_id, 0, 100 + i
            )
            self.gym.set_actor_scale(env, handle, scale)
            self.gym.set_rigid_body_color(
                env, handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*color)
            )
            handles.append(handle)
        self.handles.append(handles)

    def bind(self, task):
        self.task = task
        self.states = task._root_states.view(task.num_envs, -1, 13)[:, 2:]
        if self.states.shape[1] != SPHERE_COUNT:
            raise RuntimeError("Unexpected goal sphere actor layout")
        stride = task.get_num_actors_per_env()
        self.actor_ids = (
            (
                torch.arange(task.num_envs, device=task.device)[:, None] * stride
                + torch.arange(2, 2 + SPHERE_COUNT, device=task.device)
            )
            .to(torch.int32)
            .flatten()
        )
        self.states.zero_()
        self.states[..., 6] = 1

    def update_current(self):
        reference, mask = current_goals(self.task)
        program = getattr(self.task, "reference_tracker", None)
        future = None
        if program is not None and program.mode == "trajectory":
            ids = torch.arange(self.task.num_envs, device=self.task.device)
            future = [
                (program.observation_reference(None, ids, delta), program.base_mask)
                for delta in (1, 4, 16)
            ]
        return self.update(reference, mask, future)

    def update(self, reference, mask, future=None):
        from isaacgym import gymtorch

        positions, visible = sphere_targets(reference, mask, future)
        self.states[..., :3] = positions
        self.states[..., 7:] = 0
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.task._root_states),
            gymtorch.unwrap_tensor(self.actor_ids),
            len(self.actor_ids),
        )
        contacts = mask[:, 565:628].reshape(-1, 21, 3).any(-1)
        previous = getattr(self, "contact_colors", torch.zeros_like(contacts))
        changed = (contacts != previous).nonzero().tolist()
        if changed:
            from isaacgym import gymapi

            for env_id, joint in changed:
                color = (
                    (1.0, 0.45, 0.02) if contacts[env_id, joint] else (0.05, 0.8, 1.0)
                )
                self.gym.set_rigid_body_color(
                    self.task.envs[env_id],
                    self.handles[env_id][joint],
                    0,
                    gymapi.MESH_VISUAL,
                    gymapi.Vec3(*color),
                )
        self.contact_colors = contacts.clone()
        self.visible = visible
        return positions, visible
