# Reference Tracking Examples

Run commands from the repository root. Goals are selected from the prepared reference motions; human and object targets share the same reference frame.

## Record Examples

Install `ffmpeg`, then record all eight cases in [suite.json](suite.json):

```bash
python -m interprior.examples --output output/reference_examples
```

To record a subset:

```bash
python -m interprior.examples --only object_trajectory human_snapshot \
  --output output/selected_examples
```

Each case saves `video.mp4`, an unmodified camera-frame poster, reference configuration, rollout tensors, metrics, and a `.reference.json` provenance record. Intermediate PNGs are removed after encoding; use `--keep-frames` to retain them. Choose a new output directory when recording again.

| Configuration | Reference fields |
| --- | --- |
| [object_snapshot.json](object_snapshot.json) | Object center from frame 96. |
| [human_snapshot.json](human_snapshot.json) | Head and wrists from frame 96. |
| [combined_snapshot.json](combined_snapshot.json) | Human and object positions from the same frame. |
| [object_trajectory.json](object_trajectory.json) | Object center along the reference sequence. |
| [human_trajectory.json](human_trajectory.json) | Head and wrists along the reference sequence. |
| [combined_trajectory.json](combined_trajectory.json) | Sparse human and object reference tracking. |
| [wrist_trajectory.json](wrist_trajectory.json) | One wrist along the reference sequence. |
| [head_feet_trajectory.json](head_feet_trajectory.json) | Head and ankles along the reference sequence. |

## Record a Single Motion

```bash
sh scripts/test_interprior.sh --headless --num-envs 1 --steps 240 \
  --motion-dir external/InterMimic/InterAct/OMOMO_retarget \
  --motion-file sub2_largetable_000.pt \
  --reference-config examples/combined_trajectory.json --show-goals \
  --record-frames output/frames --save-rollout output/rollout.pt

ffmpeg -framerate 30 -i output/frames/%05d.png \
  -c:v libx264 -pix_fmt yuv420p output/reference_tracking.mp4
```

Use either `--reference-config` or the CLI selectors (`--task`, `--goal-mode`, `--goal-frame`, `--joints`). Configuration files accept only `mode`, `profile`, `start_frame`, `snapshot_frame`, and `joints`.

## Reference Timing

**Trajectory** follows the reference sequence at 30 Hz. This checkpoint uses short previews at 1, 4, and 16 frames and a long-term countdown initialized in the range 64–127 frames. Previews clamp to the final reference frame. The run stops when the first selected reference reaches its end; use one clip for a consistent rollout length.

The inherited simulator advances `progress_buf_ref` by one and decreases `long_term_t` by one each step. Their sum selects the same long-term target until the countdown reaches zero, when a new horizon is sampled. Saved rollouts include both clocks in `reference_frames` and `remaining_goal_frames`.

**Snapshot** holds one reference frame in the long-term slot, with short previews masked. For example:

```bash
sh scripts/test_interprior.sh --headless --num-envs 1 \
  --motion-dir external/InterMimic/InterAct/OMOMO_retarget \
  --motion-file sub8_largebox_003.pt \
  --task snapshot --goal-mode object --goal-frame 96 --show-goals
```

Snapshot defaults to frame 96 and accepts frames 1–127 after `start_frame`, within the selected motion. Out-of-range frames are rejected. Indices refer to the loaded sequence after its initial frame is discarded by the motion loader.

## Spheres and Metrics

`--show-goals` renders collision-free sphere actors inside Isaac Gym. Cyan and pink mark the human and object positions of the exact long-term reference goal supplied to the policy. Small purple spheres show trajectory object previews at 1, 4, and 16 frames. Adjust framing with `--camera-azimuth` and `--camera-distance`.

Trajectory metrics compare the simulated state with the current reference frame; spheres show future conditioning. Rollouts save both frame indices and displayed targets. Metrics stop at the first fall or reference boundary. Position tolerance is 0.1 m for every conditioned joint and the object center. Snapshot diagnostics report one-frame reaching and holding all selected targets within tolerance for 15 consecutive frames (0.5 seconds).

These diagnostics are separate from the paper benchmark. For comparisons, use the same reference, visible fields, seed, and visualization settings. GPU PhysX repeats can vary with the same seed. Latent noise defaults to zero; `--latent-noise 1.0` samples a persistent episode latent.
