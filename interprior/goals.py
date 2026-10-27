"""Sparse world-space human and object position goals."""

import torch

KEY_BODIES = [
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "L_Toe",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "R_Toe",
    "Torso",
    "Spine",
    "Chest",
    "Neck",
    "Head",
    "L_Thorax",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "R_Thorax",
    "R_Shoulder",
    "R_Elbow",
    "R_Wrist",
]


def make_goal_mask(
    batch, human_joints=(), object_position=False, horizons=(3,), device="cpu"
):
    """Return [B,2512]; slot 3 is the original long-term goal slot.

    Human position features have two matching representations: relative
    position (0:63) and heading-local reference position (226:289).
    Object center offset occupies 550:553. Rotations and velocities remain hidden.
    """
    mask = torch.zeros((batch, 4, 628), device=device)
    for slot in horizons:
        if slot not in range(4):
            raise ValueError("horizons must contain indices 0..3")
        for name in human_joints:
            j = KEY_BODIES.index(name)
            mask[:, slot, 3 * j : 3 * j + 3] = 1
            mask[:, slot, 226 + 3 * j : 226 + 3 * j + 3] = 1
        if object_position:
            mask[:, slot, 550:553] = 1
    return mask.flatten(1)
