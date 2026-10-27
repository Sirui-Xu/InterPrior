"""Quaternion vector rotation used by native sphere visualization."""

import torch


def rotate(quaternion, points):
    """Rotate vectors by xyzw unit quaternions, with broadcasting."""
    xyz = quaternion[..., :3].expand_as(points)
    cross = 2 * torch.cross(xyz, points, dim=-1)
    return points + quaternion[..., 3:] * cross + torch.cross(xyz, cross, dim=-1)
