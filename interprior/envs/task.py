"""Sparse-goal inference in the InterPrior SMPL simulation."""

from pathlib import Path
import random

from isaacgym import gymapi, gymutil
import numpy as np
import torch
import yaml

from .humanoid import InterPriorTask
from ..reference import ReferenceSpec, SparseReference
from ..motions import select_motions
from ..visualization import GoalSpheres


class InferenceTask(InterPriorTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reference_tracker = SparseReference(
            self, self.cfg["env"]["referenceTracking"]
        )
        if self.cfg["env"].get("showGoals", False):
            self.goal_spheres.bind(self)

    def _create_envs(self, num_envs, spacing, num_per_row):
        if self.cfg["env"].get("showGoals", False):
            self.goal_spheres = GoalSpheres(self.gym, self.sim)
        super()._create_envs(num_envs, spacing, num_per_row)

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        if hasattr(self, "goal_spheres"):
            self.goal_spheres.build(env_ptr, env_id)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if hasattr(self, "reference_tracker") and hasattr(self, "goal_spheres"):
            self.goal_spheres.update_current()

    def _prepare_goal_reference(self, reference, env_ids, delta_t):
        if (
            hasattr(self, "reference_tracker")
            and self.reference_tracker.mode == "snapshot"
        ):
            return self.reference_tracker.observation_reference(
                reference, env_ids, delta_t
            )
        # Trajectory targets are selected by the inherited reference constructor.
        return reference

    def _compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._compute_observations(env_ids)
        if hasattr(self, "reference_tracker"):
            self.mask[env_ids] = self.reference_tracker.observation_mask(env_ids)

    def _compute_reset(self):
        fallen = (
            self._rigid_body_pos[:, 0, 2] < self.cfg["env"]["terminationHeight"]
        ) & (self.progress_buf > 1)
        self._terminate_buf[:] = fallen
        if hasattr(self, "reference_tracker"):
            self.reference_end_buf = (
                (self.progress_buf >= self.max_episode_length[self.motion_ids] - 1)
                if self.reference_tracker.mode == "trajectory"
                else torch.zeros_like(fallen)
            )
            self.reset_buf[:] = fallen | self.reference_end_buf
        else:
            self.reset_buf[:] = fallen | (
                self.progress_buf >= self.max_episode_length[self.data_id] - 1
            )

    def step(self, actions):
        self.pre_physics_step(actions)
        self._physics_step()
        self.post_physics_step()
        if hasattr(self, "goal_spheres"):
            self.goal_spheres.update_current()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_assets(asset_root=None, intermimic_root="external/InterMimic"):
    root = (
        Path(asset_root).resolve()
        if asset_root
        else Path(intermimic_root).resolve() / "isaacgym/src/intermimic/data/assets"
    )
    if not (root / "smplx/omomo.xml").is_file():
        raise FileNotFoundError(
            "SMPL assets not found; pass --intermimic-root or --asset-root"
        )
    for name in ("objects", "mjcf"):
        if any((root / name).glob("*.urdf")):
            return root, root / name
    raise FileNotFoundError("Object URDF assets were not found")


def create_task(
    motion_dir,
    asset_root=None,
    intermimic_root="external/InterMimic",
    num_envs=16,
    seed=0,
    motion_file=None,
    object_name=None,
    reference=None,
    headless=True,
    cameras=False,
    show_goals=False,
    threads=4,
):
    specification = ReferenceSpec.from_dict(reference or {})
    if not torch.cuda.is_available():
        raise RuntimeError("Isaac Gym inference requires a CUDA GPU")
    selection = select_motions(motion_dir, num_envs, seed, motion_file, object_name)
    assets, objects = resolve_assets(asset_root, intermimic_root)
    for p in selection.files:
        name = p.stem.split("_")[1]
        if not (objects / (name + ".urdf")).is_file():
            raise FileNotFoundError("No object asset for motion: " + p.name)
    set_seed(seed)
    torch.set_num_threads(threads)
    cfg = yaml.safe_load(
        (Path(__file__).parents[1] / "configs/simulation.yaml").read_text()
    )
    cfg["env"].update(
        numEnvs=num_envs,
        numObs=8147,
        numObsStudent=8147,
        inference_mode=True,
        referenceTracking=specification.to_dict(),
        stateInit="Start",
        rolloutLength=100000,
        motion_file_retarget=str(Path(motion_dir).resolve()),
        motionFiles=[str(p) for p in selection.files],
        motionIds=list(selection.environment_motion_ids),
        render_cameras=cameras,
        showGoals=show_goals,
    )
    cfg["env"]["asset"].update(assetRoot=str(assets), objectRoot=str(objects))
    cfg["task"]["randomize"] = False
    params = gymapi.SimParams()
    params.dt = 1.0 / 60.0
    params.use_gpu_pipeline = True
    params.physx.use_gpu = True
    gymutil.parse_sim_config(cfg["sim"], params)
    params.physx.num_threads = threads
    return InferenceTask(cfg, params, gymapi.SIM_PHYSX, "cuda", 0, headless)
