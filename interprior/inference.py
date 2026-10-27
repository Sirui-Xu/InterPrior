"""Run the released InterPrior SMPL policy in Isaac Gym."""

import argparse
from pathlib import Path
import json

# Isaac Gym must be imported before PyTorch.
from isaacgym import gymapi
import numpy as np
from PIL import Image
import torch

from .envs.task import create_task, set_seed
from .goals import KEY_BODIES
from .reference import ReferenceSpec
from .metrics import summarize_goals
from .policy import InterPriorPolicy
from .visualization import current_goals, SPHERE_COUNT


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/interprior_smpl.pth")
    )
    parser.add_argument(
        "--motion-dir", type=Path, default=Path("InterAct/OMOMO_retarget")
    )
    parser.add_argument(
        "--motion-file", help="Use one named .pt clip from the motion directory"
    )
    parser.add_argument(
        "--intermimic-root", type=Path, default=Path("external/InterMimic")
    )
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument(
        "--object", dest="object_name", help="Select one object category"
    )
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument(
        "--goal-mode", choices=["human", "object", "combined"], default=None
    )
    parser.add_argument("--task", choices=["trajectory", "snapshot"], default=None)
    parser.add_argument(
        "--goal-frame",
        type=int,
        default=None,
        help="Exact reference target frame; never silently clamped",
    )
    parser.add_argument(
        "--reference-config",
        type=Path,
        help="Reference timing and sparse visibility configuration; no custom targets",
    )
    parser.add_argument(
        "--joints",
        nargs="+",
        choices=KEY_BODIES,
        default=None,
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--latent-noise",
        type=float,
        default=0.0,
        help="Scale of persistent episode latent noise",
    )
    parser.add_argument(
        "--record-frames",
        type=Path,
        help="Record environment 0 as a 30 fps PNG sequence",
    )
    parser.add_argument(
        "--show-goals",
        action="store_true",
        help="Render collision-free 3D goal sphere actors in the simulation",
    )
    parser.add_argument(
        "--camera-azimuth",
        type=float,
        default=-55,
        help="Camera azimuth in degrees around the scene",
    )
    parser.add_argument(
        "--camera-distance",
        type=float,
        default=3.2,
        help="Minimum horizontal camera distance in meters",
    )
    parser.add_argument("--save-rollout", type=Path)
    args = parser.parse_args()
    if args.steps < 1 or args.latent_noise < 0:
        parser.error("steps must be positive; latent-noise must be nonnegative")
    if not args.checkpoint.is_file():
        parser.error(
            "Place the released model at checkpoints/interprior_smpl.pth, or pass --checkpoint"
        )
    try:
        if args.reference_config:
            if any(
                value is not None
                for value in (args.task, args.goal_mode, args.goal_frame, args.joints)
            ):
                parser.error(
                    "Use a reference-config or CLI reference selectors, not both"
                )
            args.reference_spec = ReferenceSpec.from_dict(
                json.loads(args.reference_config.read_text())
            )
        else:
            args.reference_spec = ReferenceSpec(
                mode=args.task or "trajectory",
                profile={
                    "human": "sparse_human",
                    "object": "object_position",
                    "combined": "combined",
                }[args.goal_mode or "combined"],
                snapshot_frame=args.goal_frame,
                joints=tuple(args.joints or ()),
            )
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return args


def create_camera(task, args):
    properties = gymapi.CameraProperties()
    properties.width, properties.height = 800, 600
    camera = task.gym.create_camera_sensor(task.envs[0], properties)
    if camera < 0:
        raise RuntimeError("Camera sensor creation failed")
    initial = task.hoi_data[
        int(task.motion_ids[0]), task.reference_tracker.spec.start_frame
    ]
    points = [initial[None, :3], initial[None, 313:316]]
    tracker = task.reference_tracker
    if tracker.mode == "trajectory":
        data = task.hoi_data[
            int(task.motion_ids[0]), tracker.spec.start_frame : int(tracker.lengths[0])
        ]
        points.extend((data[:, :3], data[:, 313:316]))
    if hasattr(task, "reference_tracker"):
        for reference, mask in zip(
            task.reference_tracker.references, task.reference_tracker.masks
        ):
            human = reference[0, 326:389].reshape(21, 3)
            points.append(human[mask[0, :63].reshape(21, 3).any(-1)])
            if mask[0, 550:553].any():
                points.append(reference[0:1, 313:316])
    points = torch.cat(points).detach().cpu().numpy()
    low, high = points[:, :2].min(0), points[:, :2].max(0)
    center = (low + high) / 2
    distance = max(args.camera_distance, float((high - low).max()) * 1.3 + 1.8)
    angle = np.deg2rad(args.camera_azimuth)
    eye = gymapi.Vec3(
        float(center[0] + distance * np.cos(angle)),
        float(center[1] + distance * np.sin(angle)),
        2.15,
    )
    target = gymapi.Vec3(float(center[0]), float(center[1]), 0.85)
    task.gym.set_camera_location(camera, task.envs[0], eye, target)
    return camera


def main():
    args = parse_args()
    task = create_task(
        args.motion_dir,
        asset_root=args.asset_root,
        intermimic_root=args.intermimic_root,
        num_envs=args.num_envs,
        seed=args.seed,
        motion_file=args.motion_file,
        object_name=args.object_name,
        reference=args.reference_spec,
        headless=args.headless,
        cameras=args.record_frames is not None,
        show_goals=args.show_goals,
    )
    try:
        policy = InterPriorPolicy(args.checkpoint, device=task.device)
        set_seed(args.seed)
        task.reset()
        noise = torch.randn((task.num_envs, 64), device=task.device) * args.latent_noise
        camera = create_camera(task, args) if args.record_frames else None
        if args.record_frames:
            args.record_frames.mkdir(parents=True, exist_ok=True)
        history = {
            "actions": [],
            "human_root_states": [],
            "object_states": [],
            "reset_flags": [],
            "termination_flags": [],
            "reference_end_flags": [],
            "reference_frames": [],
            "remaining_goal_frames": [],
            "conditioned_reference_frames": [],
            "displayed_goal_human_positions": [],
            "displayed_goal_object_pose": [],
        }
        program = getattr(task, "reference_tracker", None)
        if args.show_goals:
            history.update(
                goal_sphere_positions=[],
                goal_sphere_visible=[],
                goal_sphere_contact_forces=[],
            )
        if program is not None:
            history.update(
                {
                    key: []
                    for key in (
                        "human_positions",
                        "goal_human_positions",
                        "goal_object_pose",
                        "goal_human_mask",
                        "goal_object_mask",
                        "goal_times",
                    )
                }
            )
        for step in range(args.steps):
            actions = policy(task.obs_buf, task.mask, noise)
            if not torch.isfinite(actions).all():
                raise RuntimeError("Policy produced a nonfinite action")
            task.step(actions)
            goal_reference, goal_mask = program.sample(program.elapsed_seconds())
            if args.show_goals:
                if args.save_rollout:
                    history["goal_sphere_positions"].append(
                        task.goal_spheres.states[..., :3].detach().cpu().clone()
                    )
                    history["goal_sphere_visible"].append(
                        task.goal_spheres.visible.cpu()
                    )
                    forces = task.gym.acquire_net_contact_force_tensor(task.sim)
                    from isaacgym import gymtorch

                    forces = gymtorch.wrap_tensor(forces).view(task.num_envs, -1, 3)[
                        :, -SPHERE_COUNT:
                    ]
                    history["goal_sphere_contact_forces"].append(forces.cpu().clone())
            if args.save_rollout:
                displayed, _ = current_goals(task)
                history["reference_frames"].append(
                    task.progress_buf_ref.detach().cpu().clone()
                )
                history["remaining_goal_frames"].append(
                    task.long_term_t.detach().cpu().clone()
                )
                history["conditioned_reference_frames"].append(
                    program.conditioned_frames().cpu()
                )
                history["displayed_goal_human_positions"].append(
                    displayed[:, 326:389].view(-1, 21, 3).cpu()
                )
                history["displayed_goal_object_pose"].append(
                    displayed[:, 313:320].cpu()
                )
                if program is not None:
                    history["human_positions"].append(
                        task._rigid_body_pos[:, task._key_body_ids]
                        .detach()
                        .cpu()
                        .clone()
                    )
                    history["goal_human_positions"].append(
                        goal_reference[:, 326:389].view(-1, 21, 3).cpu()
                    )
                    history["goal_object_pose"].append(goal_reference[:, 313:320].cpu())
                    history["goal_human_mask"].append(
                        goal_mask[:, :63].view(-1, 21, 3).any(-1).cpu()
                    )
                    history["goal_object_mask"].append(
                        goal_mask[:, 550:553].any(-1).cpu()
                    )
                    history["goal_times"].append(
                        program.elapsed_seconds().cpu().clone()
                    )
                history["actions"].append(actions.cpu())
                history["human_root_states"].append(
                    task._humanoid_root_states.detach().cpu().clone()
                )
                history["object_states"].append(
                    task._target_states.detach().cpu().clone()
                )
                history["reset_flags"].append(task.reset_buf.detach().cpu().clone())
                history["reference_end_flags"].append(
                    task.reference_end_buf.detach().cpu().clone()
                )
                history["termination_flags"].append(
                    task._terminate_buf.detach().cpu().clone()
                )
            if camera is not None:
                task.gym.fetch_results(task.sim, True)
                task.gym.step_graphics(task.sim)
                task.gym.render_all_camera_sensors(task.sim)
                rgb = np.asarray(
                    task.gym.get_camera_image(
                        task.sim, task.envs[0], camera, gymapi.IMAGE_COLOR
                    )
                ).reshape(600, 800, 4)[:, :, :3]
                Image.fromarray(rgb).save(args.record_frames / ("%05d.png" % step))
            if not args.headless:
                task.render(sync_frame_time=True)
            if program.mode == "trajectory" and task.reference_end_buf.any():
                break
            reset_ids = task.reset_buf.nonzero().flatten()
            if len(reset_ids):
                task.reset(reset_ids)
                noise[reset_ids] = (
                    torch.randn((len(reset_ids), 64), device=task.device)
                    * args.latent_noise
                )
        if args.save_rollout:
            payload = {key: torch.stack(value) for key, value in history.items()}
            payload.update(
                fps=30,
                model="InterPrior-SMPL",
                goal_mode=program.spec.profile,
                task=program.mode,
                seed=args.seed,
                latent_noise=args.latent_noise,
                goal_visualization="native_3d_spheres" if args.show_goals else "none",
                motions=[Path(p).name for p in task.motion_file],
                environment_motion_ids=task.motion_ids.detach().cpu(),
            )
            args.save_rollout.parent.mkdir(parents=True, exist_ok=True)
            if program is not None:
                payload["reference_specification"] = program.spec.to_dict()
                payload["reference_provenance"] = program.support_report
                args.save_rollout.with_suffix(".reference.json").write_text(
                    json.dumps(program.support_report, indent=2)
                )
                metrics = summarize_goals(payload)
                args.save_rollout.with_suffix(".json").write_text(
                    json.dumps(metrics, indent=2)
                )
            torch.save(payload, args.save_rollout)
        print(
            "InterPrior inference completed: %d steps, %d environments"
            % (step + 1, task.num_envs)
        )
    except KeyboardInterrupt:
        pass
    finally:
        if task.viewer is not None:
            task.gym.destroy_viewer(task.viewer)
        task.gym.destroy_sim(task.sim)


if __name__ == "__main__":
    main()
