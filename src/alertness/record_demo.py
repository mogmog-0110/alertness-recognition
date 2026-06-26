"""画面録画つきデモ起動。

ffmpeg(gdigrab) でデスクトップを録画しながら `python -m alertness` を起動する。
アプリが終了（q キーかウィンドウを閉じる）したら ffmpeg を正常終了させ、mp4 を閉じる。
未知の引数はそのままアプリへ渡す（例: --video, --record, --config）。

  python -m alertness.record_demo                 # 画面全体を録画しつつデモ起動
  python -m alertness.record_demo --video clip.mp4 # 動画入力でデモ＋録画
  python -m alertness.record_demo --out demo.mp4   # 保存先を指定
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _ffmpeg_cmd(ffmpeg: str, region: str, fps: int, out: Path) -> list[str]:
    # gdigrab は Windows のデスクトップ/ウィンドウを取り込む。title=... で窓単位も可。
    return [
        ffmpeg, "-y",
        "-f", "gdigrab", "-framerate", str(fps), "-i", region,
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        str(out),
    ]


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

    rec = subprocess.Popen(
        _ffmpeg_cmd(ffmpeg, args.region, args.fps, out), stdin=subprocess.PIPE
    )
    print(f"[record] 録画開始 -> {out}")

    app = [sys.executable, "-m", "alertness", *passthrough]
    print(f"[record] アプリ起動: {' '.join(app)}")
    try:
        subprocess.run(app)  # q かウィンドウを閉じるまで待つ
    finally:
        # ffmpeg の stdin に 'q' を送って正常終了させ、mp4 を確定する。
        try:
            rec.communicate(input=b"q", timeout=10)
        except Exception:
            rec.terminate()
            rec.wait(timeout=5)

    print(f"[record] 保存しました -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
