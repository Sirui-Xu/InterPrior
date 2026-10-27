"""Sparse tracking targets extracted exclusively from prepared reference motion."""

from dataclasses import asdict, dataclass, fields
from pathlib import Path
import torch

from .goals import KEY_BODIES, make_goal_mask


@dataclass(frozen=True)
class ReferenceSpec:
    """Select reference timing and visible fields; target coordinates are read-only."""

    mode: str = "trajectory"
    profile: str = "combined"
    start_frame: int = 0
    snapshot_frame: object = None
    joints: tuple = ()

    def __post_init__(self):
        if self.mode not in ("snapshot", "trajectory"):
            raise ValueError("Reference mode must be snapshot or trajectory")
        if self.profile not in ("object_position", "sparse_human", "combined"):
            raise ValueError(
                "Reference profile must be object_position, sparse_human, or combined"
            )
        if type(self.start_frame) is not int or self.start_frame < 0:
            raise ValueError("start_frame must be a nonnegative integer")
        if self.snapshot_frame is not None and (
            type(self.snapshot_frame) is not int
            or not 1 <= self.snapshot_frame - self.start_frame <= 127
        ):
            raise ValueError("snapshot_frame must be 1..127 frames after start_frame")
        if self.mode == "trajectory" and self.snapshot_frame is not None:
            raise ValueError("snapshot_frame is only valid in snapshot mode")
        if isinstance(self.joints, str) or not isinstance(self.joints, (list, tuple)):
            raise ValueError("joints must be a list of unique reference joint names")
        names = tuple(self.joints)
        if self.profile == "object_position" and names:
            raise ValueError("The object_position profile does not reveal human joints")
        if self.profile != "object_position" and not names:
            names = ("Head", "L_Wrist", "R_Wrist")
        if any(not isinstance(n, str) or n not in KEY_BODIES for n in names) or len(
            set(names)
        ) != len(names):
            raise ValueError("joints contain unknown or duplicate names")
        if len(names) > 3:
            raise ValueError(
                "Select at most three reference joints for sparse tracking"
            )
        object.__setattr__(self, "joints", names)

    @classmethod
    def from_dict(cls, value):
        if isinstance(value, cls):
            value = value.to_dict()
        if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
            raise ValueError("Reference configuration must be a JSON object")
        unknown = set(value) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(
                "Unsupported reference fields: "
                + ", ".join(sorted(unknown))
                + ". Only reference sparse tracking is provided; custom targets are not accepted."
            )
        return cls(**value)

    def to_dict(self):
        return asdict(self)


class SparseReference:
    """Keep all target features from the same frame of the initialized motion.

    No coordinates, poses, contacts, or velocities are synthesized or edited.
    Snapshot mode holds a selected frame in the long-term slot. Trajectory mode
    follows the original sequence at offsets 1, 4, 16 and the native countdown.
    """

    def __init__(self, task, specification):
        self.task = task
        self.spec = ReferenceSpec.from_dict(specification)
        self.mode, self.device, self.batch = self.spec.mode, task.device, task.num_envs
        self.start_frames = torch.full_like(task.motion_ids, self.spec.start_frame)
        self.lengths = task.max_episode_length[task.motion_ids]
        if (self.start_frames + 1 >= self.lengths).any():
            raise ValueError("start_frame leaves no future reference frames")
        frame = self.spec.snapshot_frame
        if self.mode == "snapshot" and frame is None:
            frame = self.spec.start_frame + 96
        self.target_frames = torch.full_like(
            task.motion_ids, frame or self.spec.start_frame
        )
        if self.mode == "snapshot" and (self.target_frames >= self.lengths).any():
            raise ValueError(
                "snapshot_frame is outside a selected motion; it is never silently clamped"
            )
        self.anchor = task.hoi_data[task.motion_ids, self.target_frames].clone()
        self.references = self.anchor[None]
        self.base_mask = (
            make_goal_mask(
                self.batch,
                self.spec.joints,
                self.spec.profile in ("object_position", "combined"),
                (3,),
                self.device,
            )
            .view(self.batch, 4, 628)[:, 3]
            .clone()
        )
        self.masks = self.base_mask[None]
        self.support_report = dict(
            task="reference_sparse_tracking",
            specification=self.spec.to_dict(),
            horizon_slots=[3] if self.mode == "snapshot" else [0, 1, 2, 3],
            short_preview_frames=[1, 4, 16],
            long_horizon="native 64..127-frame countdown",
            source_motions=[
                Path(task.motion_file[int(i)]).name for i in task.motion_ids
            ],
            reference_lengths=self.lengths.tolist(),
            target_frames=(
                self.target_frames.tolist() if self.mode == "snapshot" else None
            ),
            interpretation="Targets are copied from prepared references without target editing. Sparse fields alone are exposed to the policy. This is a reference tracking demonstration, not a custom-goal feasibility claim.",
        )

    def elapsed_seconds(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.batch, device=self.device)
        return (
            self.task.progress_buf_ref[env_ids] - self.start_frames[env_ids]
        ).float() / 30

    def frame_indices(self, seconds, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.batch, device=self.device)
        if self.mode == "snapshot":
            return self.target_frames[env_ids]
        seconds = torch.as_tensor(
            seconds, device=self.device, dtype=torch.float32
        ).expand(len(env_ids))
        frames = self.start_frames[env_ids] + (seconds * 30).round().long()
        # Preview clamping matches the native constructor at the end of a clip.
        return torch.minimum(frames.clamp_min(0), self.lengths[env_ids] - 1)

    def sample(self, seconds, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.batch, device=self.device)
        frames = self.frame_indices(seconds, env_ids)
        reference = self.task.hoi_data[self.task.motion_ids[env_ids], frames].clone()
        return reference, self.base_mask[env_ids].clone()

    def observation_reference(self, reference, env_ids, delta_t):
        if self.mode == "trajectory" and reference is not None:
            return reference
        frames = self.observation_frames(delta_t, env_ids)
        return self.task.hoi_data[self.task.motion_ids[env_ids], frames].clone()

    def observation_frames(self, delta_t, env_ids=None):
        """Use the native integer reference clock and preview clamping order."""
        if env_ids is None:
            env_ids = torch.arange(self.batch, device=self.device)
        if self.mode == "snapshot":
            return self.target_frames[env_ids]
        last = self.lengths[env_ids] - 1
        current = torch.minimum(self.task.progress_buf_ref[env_ids].clamp_min(0), last)
        return torch.minimum((current + delta_t).clamp_min(0), last)

    def observation_mask(self, env_ids):
        mask = self.base_mask[env_ids, None].expand(-1, 4, -1).clone()
        if self.mode == "snapshot":
            mask[:, :3] = 0
        return mask.flatten(1)

    def conditioned_frames(self):
        return torch.stack(
            [
                self.observation_frames(offset)
                for offset in (1, 4, 16, self.task.long_term_t)
            ],
            dim=-1,
        )
