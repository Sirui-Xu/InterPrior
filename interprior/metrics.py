"""Sparse-reference tracking diagnostics, separate from paper benchmark scores."""

import torch


def summarize_goals(rollout, position_tolerance=0.1):
    distances = (rollout["human_positions"] - rollout["goal_human_positions"]).norm(
        dim=-1
    )
    active = rollout["goal_human_mask"].bool()
    human_mean = (distances * active).sum(-1) / active.sum(-1).clamp_min(1)
    human_max = distances.masked_fill(~active, 0).amax(-1)
    object_error = (
        rollout["object_states"][..., :3] - rollout["goal_object_pose"][..., :3]
    ).norm(dim=-1)
    object_active = rollout["goal_object_mask"].bool()
    satisfied = ((human_max <= position_tolerance) | ~active.any(-1)) & (
        (object_error <= position_tolerance) | ~object_active
    )
    falls = rollout["termination_flags"].bool()
    ends = rollout.get("reference_end_flags", torch.zeros_like(falls)).bool()
    valid = (falls.long().cumsum(0) == 0) & ((ends.long().cumsum(0) - ends.long()) == 0)
    results = []
    for env in range(distances.shape[1]):

        def optional(values, reduction):
            if not len(values):
                return None
            return float(getattr(values, reduction)())

        longest = current = 0
        for reached in (satisfied[:, env] & valid[:, env]).tolist():
            current = current + 1 if reached else 0
            longest = max(longest, current)
        human_values = human_mean[:, env][valid[:, env] & active[:, env].any(-1)]
        object_values = object_error[:, env][valid[:, env] & object_active[:, env]]
        results.append(
            dict(
                environment=env,
                reached_before_first_fall=(
                    bool((satisfied[:, env] & valid[:, env]).any())
                    if rollout["task"] == "snapshot"
                    else None
                ),
                sustained_goal_reached_before_first_fall=(
                    longest >= 15 if rollout["task"] == "snapshot" else None
                ),
                longest_goal_hold_seconds=longest / 30,
                reference_episode_completed=bool(
                    (ends[:, env] & (falls[:, env].long().cumsum(0) == 0)).any()
                ),
                fraction_within_tolerance_before_first_fall=float(
                    (satisfied[:, env] & valid[:, env]).sum()
                    / valid[:, env].sum().clamp_min(1)
                ),
                minimum_human_error_m=optional(human_values, "min"),
                mean_human_error_m=optional(human_values, "mean"),
                maximum_human_error_m=optional(
                    human_max[:, env][valid[:, env] & active[:, env].any(-1)], "max"
                ),
                minimum_object_error_m=optional(object_values, "min"),
                mean_object_error_m=optional(object_values, "mean"),
                falls=int(falls[:, env].sum()),
            )
        )
    return dict(
        task=rollout["task"],
        position_tolerance_m=position_tolerance,
        snapshot_hold_frames=15,
        human_threshold="maximum Euclidean error across conditioned reference joints",
        interpretation="Reference tracking diagnostics before the first fall/reference boundary. Snapshot sustained reaching requires 15 consecutive frames; one-frame reaching is reported separately. These are not paper benchmark scores.",
        environments=results,
    )
