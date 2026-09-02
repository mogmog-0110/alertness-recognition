"""姿勢の飛びを捨てる仕組みのテスト。

solvePnP は時折もう一方の解へ飛び、pitch が 1 フレームで 100 度以上変わる。
1 フレームの飛びでも、うなずき判定は「8 度以上の上下動」を数えるので偽の
うなずきが立ち、60 秒の窓のあいだ眠気が最大に張り付く。
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from alertness.contracts import FaceLandmarks, Pose
from alertness.features.extractor import FaceFeatureExtractor


def _landmarks() -> FaceLandmarks:
    points = np.full((478, 3), 0.5)
    # 幾何は使わない (estimate_pose を差し替えるため)。点数だけ満たす。
    return FaceLandmarks(points=points, image_size=(1280, 720), detected=True, blendshapes={})


def _pitches(poses: list[Pose], timestamps: list[float]) -> list[float]:
    extractor = FaceFeatureExtractor()
    out = []
    with patch("alertness.features.extractor.estimate_pose") as fake:
        for pose, t in zip(poses, timestamps):
            fake.return_value = pose
            out.append(extractor.extract(_landmarks(), t).get("pitch"))
    return out


def test_an_impossible_jump_is_dropped() -> None:
    # 30fps で 1 フレーム 120 度は 3600 度/秒。人間の頭では出せない。
    poses = [Pose(0.0, 0.0, 0.0), Pose(120.0, 0.0, 0.0), Pose(2.0, 0.0, 0.0)]
    stamps = [0.0, 1 / 30, 2 / 30]
    values = _pitches(poses, stamps)
    assert values[1] == 0.0, "飛びは捨てて直前を使う"
    assert values[2] == 2.0, "次のフレームでは普通に追従する"


def test_a_fast_but_possible_turn_is_kept() -> None:
    # 30fps で 1 フレーム 8 度 = 240 度/秒。素早い振り向きの範囲。
    poses = [Pose(0.0, 0.0, 0.0), Pose(0.0, 8.0, 0.0)]
    stamps = [0.0, 1 / 30]
    values = _pitches(poses, stamps)
    assert values[1] == 0.0  # pitch は動いていない
    # yaw が採用されていること
    extractor = FaceFeatureExtractor()
    with patch("alertness.features.extractor.estimate_pose") as fake:
        fake.return_value = poses[0]
        extractor.extract(_landmarks(), 0.0)
        fake.return_value = poses[1]
        assert extractor.extract(_landmarks(), 1 / 30).get("yaw") == 8.0


def test_the_limit_scales_with_elapsed_time() -> None:
    # 間隔が空けば、同じ角度差でも実際の動きとして通る。
    poses = [Pose(0.0, 0.0, 0.0), Pose(120.0, 0.0, 0.0)]
    values = _pitches(poses, [0.0, 1.0])  # 1 秒あれば 120 度は動ける
    assert values[1] == 120.0
