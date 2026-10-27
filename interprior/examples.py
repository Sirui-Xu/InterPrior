"""Record reference sparse tracking examples with native Isaac Gym spheres."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=Path("examples/suite.json"))
    parser.add_argument("--output", type=Path, default=Path("output/examples"))
    parser.add_argument(
        "--motion-dir", default="external/InterMimic/InterAct/OMOMO_retarget"
    )
    parser.add_argument("--checkpoint", default="checkpoints/interprior_smpl.pth")
    parser.add_argument("--intermimic-root", default="external/InterMimic")
    parser.add_argument("--only", nargs="+", help="Run only named cases")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="Encoder executable")
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()
    if shutil.which(args.ffmpeg) is None:
        parser.error("Install ffmpeg or pass its executable with --ffmpeg")
    cases = json.loads(args.suite.read_text())["cases"]
    if args.only:
        names = {case["name"] for case in cases}
        if set(args.only) - names:
            parser.error("Unknown cases: " + ", ".join(sorted(set(args.only) - names)))
        cases = [case for case in cases if case["name"] in args.only]
    report = {"visualization": "native_3d_spheres", "runs": []}
    for case in cases:
        folder = args.output / case["name"]
        if (folder / "video.mp4").exists():
            raise FileExistsError("Choose a new output folder: " + str(folder))
        command = [
            sys.executable,
            "-m",
            "interprior.inference",
            "--headless",
            "--show-goals",
            "--num-envs",
            "1",
            "--steps",
            str(case.get("steps", 300)),
            "--motion-dir",
            args.motion_dir,
            "--checkpoint",
            args.checkpoint,
            "--intermimic-root",
            args.intermimic_root,
            "--motion-file",
            case["motion_file"],
            "--reference-config",
            case["reference_config"],
            "--camera-azimuth",
            str(case.get("camera_azimuth", -55)),
            "--record-frames",
            str(folder / "frames"),
            "--save-rollout",
            str(folder / "rollout.pt"),
        ]
        # Each native simulator owns its process and graphics context.
        subprocess.run(command, check=True)
        subprocess.run(
            [
                args.ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                "30",
                "-i",
                str(folder / "frames/%05d.png"),
                "-c:v",
                "libx264",
                "-crf",
                "19",
                "-threads",
                "2",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(folder / "video.mp4"),
            ],
            check=True,
        )
        shutil.copy2(folder / "frames/00030.png", folder / "poster.png")
        shutil.copy2(case["reference_config"], folder / "reference.json")
        if not args.keep_frames:
            shutil.rmtree(folder / "frames")
        metrics = json.loads((folder / "rollout.json").read_text())
        report["runs"].append(dict(case=case, metrics=metrics))
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "results.json").write_text(json.dumps(report, indent=2))
        print("Recorded reference sparse tracking: " + case["name"], flush=True)


if __name__ == "__main__":
    main()
