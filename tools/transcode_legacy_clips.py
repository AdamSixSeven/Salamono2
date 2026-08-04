"""Create browser-compatible copies of legacy mp4v evidence clips once."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("clips_dir", nargs="?", default="data/event_clips")
    parser.add_argument("--ffmpeg", default=None)
    args = parser.parse_args()
    ffmpeg = args.ffmpeg or shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("FFmpeg not found in PATH; pass --ffmpeg")
    clips_dir = Path(args.clips_dir)
    converted = 0
    for source in sorted(clips_dir.glob("*.mp4")):
        if source.name.endswith(".browser.mp4"):
            continue
        destination = source.with_name(source.stem + ".browser.mp4")
        if destination.is_file() and destination.stat().st_size > 0:
            continue
        temporary = destination.with_suffix(".tmp.mp4")
        result = subprocess.run([
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(temporary),
        ], check=False)
        if result.returncode == 0 and temporary.is_file() and temporary.stat().st_size > 0:
            temporary.replace(destination)
            converted += 1
        else:
            temporary.unlink(missing_ok=True)
            print(f"FAILED {source.name}")
    print(f"Converted {converted} legacy clip(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
