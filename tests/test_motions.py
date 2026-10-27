"""Motion routing must handle unequal object counts without wrong actors."""

from pathlib import Path
import tempfile
import unittest

from interprior.motions import select_motions


class MotionSelectionTests(unittest.TestCase):
    def test_unequal_counts_and_repeated_environments(self):
        with tempfile.TemporaryDirectory() as folder:
            names = [
                "sub2_largetable_000.pt",
                "sub2_largetable_001.pt",
                "sub2_woodchair_000.pt",
            ]
            for name in names:
                (Path(folder) / name).touch()
            selected = select_motions(folder, num_envs=8)
            routed = [selected.files[i].name for i in selected.environment_motion_ids]
            self.assertEqual(routed, names + names + names[:2])
            self.assertEqual(
                select_motions(folder, num_envs=4, object_name="woodchair")
                .files[0]
                .name,
                names[-1],
            )

    def test_large_library_has_bounded_deterministic_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            for i in range(100):
                (Path(folder) / ("sub2_largetable_%03d.pt" % i)).touch()
            a = select_motions(folder, num_envs=4, seed=7)
            b = select_motions(folder, num_envs=4, seed=7)
            self.assertEqual(a, b)
            self.assertEqual(len(a.files), 4)
            self.assertEqual(len(set(a.environment_motion_ids)), 4)


if __name__ == "__main__":
    unittest.main()
