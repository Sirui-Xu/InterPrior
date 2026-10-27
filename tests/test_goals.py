"""Reference provenance, time alignment, visibility, and diagnostic boundaries."""

from types import SimpleNamespace
import unittest

try:
    import torch
    from interprior.reference import ReferenceSpec, SparseReference
    from interprior.metrics import summarize_goals
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Reference tensor tests require PyTorch")
class ReferenceTests(unittest.TestCase):
    def scene(self):
        data = torch.arange(200 * 932, dtype=torch.float32).reshape(1, 200, 932) * 1e-6
        data[..., 2] = 1
        data[..., 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        data[..., 316:320] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        return SimpleNamespace(
            device="cpu",
            num_envs=2,
            motion_ids=torch.tensor([0, 0]),
            max_episode_length=torch.tensor([200]),
            hoi_data=data,
            motion_file=["sub8_largebox_003.pt"],
            progress_buf=torch.tensor([0, 0]),
            progress_buf_ref=torch.tensor([0, 0]),
            long_term_t=torch.tensor([96, 96]),
        )

    def test_snapshot_copies_complete_state_and_only_long_term_is_visible(self):
        scene = self.scene()
        original = scene.hoi_data.clone()
        p = SparseReference(
            scene, dict(mode="snapshot", profile="combined", snapshot_frame=96)
        )
        ref, _ = p.sample(7)
        self.assertTrue(torch.equal(ref, scene.hoi_data[scene.motion_ids, 96]))
        self.assertTrue(torch.equal(scene.hoi_data, original))
        mask = p.observation_mask(torch.arange(2)).view(2, 4, 628)
        self.assertFalse(mask[:, :3].any())
        self.assertTrue(mask[:, 3, 550:553].all())
        self.assertFalse(mask[:, 3, 553:].any())

    def test_trajectory_uses_exact_reference_frames_and_phase(self):
        scene = self.scene()
        scene.progress_buf[:] = 20
        scene.progress_buf_ref[:] = 20
        p = SparseReference(
            scene, dict(mode="trajectory", profile="sparse_human", start_frame=20)
        )
        ref = p.observation_reference(None, torch.arange(2), 16)
        self.assertTrue(torch.equal(ref, scene.hoi_data[scene.motion_ids, 36]))
        self.assertTrue(
            torch.equal(p.sample(0.5)[0], scene.hoi_data[scene.motion_ids, 35])
        )
        self.assertEqual(p.conditioned_frames().tolist(), [[21, 24, 36, 116]] * 2)
        mask = p.observation_mask(torch.arange(2)).view(2, 4, 628)
        self.assertTrue(torch.equal(mask[:, 0], mask[:, 3]))

    def test_trajectory_preserves_the_native_reference_tensor(self):
        scene = self.scene()
        p = SparseReference(scene, {})
        native = scene.hoi_data[scene.motion_ids, 37].clone()
        self.assertIs(p.observation_reference(native, torch.arange(2), 16), native)

    def test_reference_clock_and_countdown_keep_the_long_target_fixed(self):
        scene = self.scene()
        scene.progress_buf[:] = 100
        scene.progress_buf_ref[:] = 10
        scene.long_term_t[:] = torch.tensor([96, 32])
        p = SparseReference(scene, {})
        for step in range(5):
            self.assertEqual(
                p.conditioned_frames().tolist(),
                [
                    [11 + step, 14 + step, 26 + step, 106],
                    [11 + step, 14 + step, 26 + step, 42],
                ],
            )
            self.assertTrue(
                torch.equal(p.elapsed_seconds(), scene.progress_buf_ref.float() / 30)
            )
            scene.progress_buf += 1
            scene.progress_buf_ref += 1
            scene.long_term_t -= 1

    def test_paused_reference_clock_does_not_follow_simulation_time(self):
        scene = self.scene()
        scene.progress_buf_ref[:] = 20
        p = SparseReference(scene, {})
        before = p.conditioned_frames()
        scene.progress_buf += 5
        self.assertTrue(torch.equal(p.conditioned_frames(), before))
        reference = p.observation_reference(None, torch.arange(2), 16)
        self.assertTrue(torch.equal(reference, scene.hoi_data[scene.motion_ids, 36]))

    def test_native_preview_clamps_reference_clock_before_adding_offset(self):
        scene = self.scene()
        scene.progress_buf_ref[:] = -10
        p = SparseReference(scene, {})
        self.assertEqual(p.observation_frames(16).tolist(), [16, 16])

    def test_snapshot_is_an_explicit_fixed_frame_override(self):
        scene = self.scene()
        p = SparseReference(scene, dict(mode="snapshot", snapshot_frame=96))
        for frame in (0, 95, 96, 140):
            scene.progress_buf_ref[:] = frame
            self.assertEqual(p.conditioned_frames().tolist(), [[96] * 4] * 2)

    def test_snapshot_out_of_range_is_rejected_not_clamped(self):
        scene = self.scene()
        scene.max_episode_length[:] = 80
        with self.assertRaisesRegex(ValueError, "outside"):
            SparseReference(scene, dict(mode="snapshot", snapshot_frame=96))

    def test_only_preview_sampling_clamps_at_reference_boundary(self):
        scene = self.scene()
        scene.progress_buf[:] = 190
        scene.progress_buf_ref[:] = 190
        p = SparseReference(scene, {})
        self.assertEqual(p.conditioned_frames().tolist(), [[191, 194, 199, 199]] * 2)

    def test_arbitrary_targets_and_user_goal_interfaces_are_not_accepted(self):
        for spec in (
            {"query": {}},
            {"object_position": [0, 0, 1]},
            {"human_positions": {}},
            {"mode": "chain"},
            {"profile": "contact"},
            {"object_rotation": [0, 0, 0, 1]},
        ):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                ReferenceSpec.from_dict(spec)

    def test_invalid_reference_selectors_are_rejected(self):
        for spec in (
            {"start_frame": True},
            {"snapshot_frame": 128, "mode": "snapshot"},
            {"joints": ["Head", "Head"]},
            {"joints": ["missing"]},
            {"profile": "object_position", "joints": ["Head"]},
        ):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                ReferenceSpec.from_dict(spec)

    def rollout(self, frames):
        return dict(
            human_positions=torch.zeros(frames, 1, 21, 3),
            goal_human_positions=torch.zeros(frames, 1, 21, 3),
            goal_human_mask=torch.zeros(frames, 1, 21, dtype=torch.bool),
            goal_object_pose=torch.zeros(frames, 1, 7),
            object_states=torch.zeros(frames, 1, 13),
            goal_object_mask=torch.ones(frames, 1, dtype=torch.bool),
            termination_flags=torch.zeros(frames, 1),
            reference_end_flags=torch.zeros(frames, 1),
            task="snapshot",
        )

    def test_one_frame_reaching_is_not_a_sustained_success(self):
        data = self.rollout(20)
        data["object_states"][1:, :, 0] = 1
        e = summarize_goals(data)["environments"][0]
        self.assertTrue(e["reached_before_first_fall"])
        self.assertFalse(e["sustained_goal_reached_before_first_fall"])
        data["object_states"].zero_()
        self.assertTrue(
            summarize_goals(data)["environments"][0][
                "sustained_goal_reached_before_first_fall"
            ]
        )

    def test_success_after_a_fall_is_not_counted(self):
        data = self.rollout(20)
        data["termination_flags"][0] = 1
        self.assertFalse(
            summarize_goals(data)["environments"][0]["reached_before_first_fall"]
        )

    def test_trajectory_metrics_stop_at_first_reference_end(self):
        data = self.rollout(20)
        data["task"] = "trajectory"
        data["reference_end_flags"][9] = 1
        data["object_states"][10:, :, 0] = 10
        e = summarize_goals(data)["environments"][0]
        self.assertEqual(e["mean_object_error_m"], 0)
        self.assertTrue(e["reference_episode_completed"])


if __name__ == "__main__":
    unittest.main()
