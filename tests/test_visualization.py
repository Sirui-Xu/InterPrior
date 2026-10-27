"""Marker coordinates, actor layout, and collision-free geometry checks."""

from pathlib import Path
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

try:
    import torch
    from interprior.reference import SparseReference
    from interprior.visualization import GoalSpheres, sphere_targets
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required")
class SphereTests(unittest.TestCase):
    def test_masked_targets_hidden_and_declared_world_positions_exact(self):
        reference = torch.zeros(2, 932)
        reference[:, 319] = 1
        reference[:, 326:389] = torch.arange(63)
        reference[:, 313:316] = torch.tensor([1, 2, 3])
        mask = torch.zeros(2, 628)
        mask[:, 36:39] = 1
        mask[:, 550:553] = 1
        positions, visible = sphere_targets(reference, mask)
        self.assertTrue(torch.equal(positions[:, 12], reference[:, 362:365]))
        self.assertTrue(torch.equal(positions[:, 21], reference[:, 313:316]))
        self.assertTrue((positions[~visible] == -1000).all())
        self.assertEqual(int(visible.sum()), 4)

    def test_quaternion_axes_and_future_object_goals(self):
        reference = torch.zeros(1, 932)
        reference[:, 318:320] = 2**-0.5
        mask = torch.ones(1, 628)
        future = [(reference.clone(), mask)] * 3
        positions, visible = sphere_targets(reference, mask, future)
        self.assertTrue(
            torch.allclose(positions[0, 22], torch.tensor([0.0, 0.18, 0.0]), atol=1e-6)
        )
        self.assertTrue(torch.equal(positions[:, 25], reference[:, 313:316]))
        self.assertTrue(visible[:, 25:].all())

    def test_binding_preserves_humanoid_and_object_states(self):
        roots = torch.randn(2, 30, 13)
        before = roots[:, :2].clone()
        task = SimpleNamespace(
            _root_states=roots.flatten(0, 1),
            num_envs=2,
            device="cpu",
            get_num_actors_per_env=lambda: 30,
        )
        spheres = object.__new__(GoalSpheres)
        spheres.bind(task)
        self.assertTrue(torch.equal(roots[:, :2], before))
        self.assertEqual(
            spheres.actor_ids.tolist(), list(range(2, 30)) + list(range(32, 60))
        )
        self.assertTrue((roots[:, 2:, 6] == 1).all())

    def test_all_spheres_use_the_native_reference_clock(self):
        task = SimpleNamespace(
            num_envs=1,
            device="cpu",
            motion_ids=torch.tensor([0]),
            max_episode_length=torch.tensor([200]),
            motion_file=["reference.pt"],
            hoi_data=torch.arange(200 * 932).reshape(1, 200, 932).float(),
            progress_buf=torch.tensor([100]),
            progress_buf_ref=torch.tensor([-10]),
            long_term_t=torch.tensor([96]),
        )
        task.reference_tracker = SparseReference(task, {})
        spheres = object.__new__(GoalSpheres)
        spheres.task = task
        spheres.update = lambda reference, mask, future: (reference, future)
        reference, future = spheres.update_current()
        self.assertTrue(torch.equal(reference, task.hoi_data[:, 96]))
        for (preview, _), frame in zip(future, (1, 4, 16)):
            self.assertTrue(torch.equal(preview, task.hoi_data[:, frame]))

    def test_visual_sphere_asset_has_no_collision_or_joint(self):
        path = Path(__file__).parents[1] / "interprior/assets/goal_sphere.urdf"
        root = ET.parse(path).getroot()
        self.assertEqual(root.findall(".//collision"), [])
        self.assertEqual(root.findall(".//joint"), [])
        self.assertIsNotNone(root.find(".//visual/geometry/sphere"))


if __name__ == "__main__":
    unittest.main()
