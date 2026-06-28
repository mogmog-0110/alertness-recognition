"""画面録画つきデモ起動。

ffmpeg(gdigrab) でデスクトップを録画しながら `python -m alertness` を起動する。
アプリが終了（q キーかウィンドウを閉じる）したら ffmpeg を正常終了させ、mp4 を閉じる。
未知の引数はそのままアプリへ渡す（例: --video, --record, --config）。

  python -m alertness.record_demo                  # 画面全体を録画しつつデモ起動
  python -m alertness.record_demo --video clip.mp4 # 動画入力でデモ＋録画
  python -m alertness.record_demo --out demo.mp4   # 保存先を指定
  python -m alertness.record_demo --region title=Alertness  # デモ窓だけ録画
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

_TITLE_PREFIX = "title="


def _ffmpeg_cmd(ffmpeg: str, region: str, fps: int, out: Path) -> list[str]:
    # gdigrab は Windows のデスクトップ/ウィンドウを取り込む。title=... で窓単位も可。
    return [
        ffmpeg, "-y",
        "-f", "gdigrab", "-framerate", str(fps), "-i", region,
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        str(out),
    ]


def _window_title(region: str) -> str | None:
    """region が `title=...` のときウィンドウ名を返す。デスクトップ録画なら None。"""
    if region.startswith(_TITLE_PREFIX):
        return region[len(_TITLE_PREFIX):]
    return None


def _wait_for_window(title: str, timeout: float) -> bool:
    """指定タイトルのウィンドウが現れるまで待つ（見つかれば True）。

    窓が無い状態で gdigrab を起動すると "Can't find window" で即終了するため、
    ウィンドウ単位の録画ではこれで生成を待ってから ffmpeg を起動する。
    """
    try:
        import ctypes  # Windows 専用。gdigrab 自体が Windows 限定なので問題ない。

        find_window = ctypes.windll.user32.FindWindowW
    except (ImportError, AttributeError):
        # 判定手段が無い環境では少しだけ待って続行する。
        time.sleep(1.0)
        return True

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if find_window(None, title):
            time.sleep(0.3)  # 窓の初期化が落ち着くまで一拍おく。
            return True
        time.sleep(0.2)
    return False


def _start_recording(ffmpeg: str, region: str, fps: int, out: Path) -> subprocess.Popen:
    rec = subprocess.Popen(
        _ffmpeg_cmd(ffmpeg, region, fps, out), stdin=subprocess.PIPE
    )
    print(f"[record] 録画開始 -> {out}")
    return rec


def _stop_recording(rec: subprocess.Popen | None) -> None:
    if rec is None:
        return
    # ffmpeg の stdin に 'q' を送って正常終了させ、mp4 を確定する。
    try:
        rec.communicate(input=b"q", timeout=10)
    except Exception:
        rec.terminate()
        rec.wait(timeout=5)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="デモを起動し、画面を録画して mp4 に保存する（ffmpeg が必要）"
    )
    parser.add_argument("--out", default=None, help="保存先 mp4（既定: recordings/demo_<時刻>.mp4）")
    parser.add_argument("--fps", type=int, default=30, help="録画フレームレート（既定: 30）")
    parser.add_argument(
        "--region",
        default="desktop",
        help="gdigrab の入力。既定は画面全体。デモ窓だけなら title=Alertness",
    )
    args, passthrough = parser.parse_known_args(argv)

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("[record] ffmpeg が見つかりません。PATH に通してから再実行してください。")
        return 1

    out = Path(args.out) if args.out else Path("recordings") / f"demo_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    title = _window_title(args.region)
    app = [sys.executable, "-m", "alertness", *passthrough]
    rec: subprocess.Popen | None = None

    # デスクトップ録画は順序非依存なので、先に録画を始めて起動の瞬間から取り込む。
    if title is None:
        rec = _start_recording(ffmpeg, args.region, args.fps, out)

    print(f"[record] アプリ起動: {' '.join(app)}")
    app_proc = subprocess.Popen(app)

    # 窓単位の録画は、アプリ窓が出来てから ffmpeg を起動する。
    if title is not None:
        if _wait_for_window(title, timeout=20.0):
            rec = _start_recording(ffmpeg, args.region, args.fps, out)
        else:
            print(
                f"[record] ウィンドウ '{title}' が現れませんでした。"
                "録画なしでデモを続けます。"
            )

    try:
        app_proc.wait()  # q かウィンドウを閉じるまで待つ
    finally:
        _stop_recording(rec)

    if rec is not None:
        print(f"[record] 保存しました -> {out}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
