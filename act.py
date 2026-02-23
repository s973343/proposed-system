import glob
import importlib.util
import os
import subprocess
import sys
import argparse

VIDEO_OUTPUT_DIR = os.path.join("data", "activitynetqa", "videos")


def _ensure_yt_dlp_installed():
    if importlib.util.find_spec("yt_dlp") is None:
        raise RuntimeError(
            "yt-dlp is not installed in this Python environment. "
            "Install it with: pip install yt-dlp"
        )


def _open_in_default_player(file_path):
    if sys.platform.startswith("win"):
        os.startfile(file_path)  # type: ignore[attr-defined]
        return
    if sys.platform == "darwin":
        subprocess.run(["open", file_path], check=False)
        return
    subprocess.run(["xdg-open", file_path], check=False)


def fetch_video(video_id, proxy=None):
    _ensure_yt_dlp_installed()
    os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)

    expected_mp4 = os.path.join(VIDEO_OUTPUT_DIR, f"v_{video_id}.mp4")
    if os.path.exists(expected_mp4):
        return expected_mp4

    outtmpl = os.path.join(VIDEO_OUTPUT_DIR, f"v_{video_id}.%(ext)s")
    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "-f",
        "bv*[height<=480]+ba/b[height<=480]/b",
        "--merge-output-format",
        "mp4",
        "--no-part",
        url,
        "-o",
        outtmpl,
    ]
    if proxy:
        cmd.extend(["--proxy", proxy])

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "yt-dlp failed to download from YouTube. "
            "If you see WinError 10013, Windows firewall/antivirus/VPN/proxy is blocking access. "
            "Allow this Python executable in firewall or run with --proxy."
        ) from exc

    matches = glob.glob(os.path.join(VIDEO_OUTPUT_DIR, f"v_{video_id}.*"))
    if not matches:
        raise FileNotFoundError(
            f"Download finished but no file found for video id '{video_id}' in {VIDEO_OUTPUT_DIR}"
        )
    mp4_matches = [p for p in matches if p.lower().endswith(".mp4")]
    return mp4_matches[0] if mp4_matches else matches[0]


def main():
    parser = argparse.ArgumentParser(description="Download and open ActivityNet video")
    parser.add_argument("--video-id", default="1QIUV7WYKXg", help="YouTube video id")
    parser.add_argument("--open", action="store_true", help="Open in default video player")
    parser.add_argument("--proxy", default=None, help="Proxy URL, e.g. http://127.0.0.1:7890")
    args = parser.parse_args()

    try:
        video_path = fetch_video(args.video_id, proxy=args.proxy)
    except RuntimeError as err:
        print(f"Error: {err}")
        sys.exit(1)

    print(f"Saved video: {video_path}")
    print(f"Absolute path: {os.path.abspath(video_path)}")

    if args.open:
        _open_in_default_player(video_path)
        print("Opened in your default video player.")


if __name__ == "__main__":
    main()
