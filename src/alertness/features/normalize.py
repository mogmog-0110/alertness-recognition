"""生特徴量を基準値（キャリブ結果）で相対化する。

学習(Colab)と推論(ローカル)で必ず同じ変換を使うため、version を持たせる。
正規化済みの値は元の値と同じ袋に追記して返す（生の値も残す）。
"""

from __future__ import annotations

from ..contracts import CalibrationProfile, Features


def _wrap_deg(deg: float) -> float:
    # 角度を [-180, 180] に畳む。
    return ((deg + 180.0) % 360.0) - 180.0


def normalize_features(
    raw: Features, profile: CalibrationProfile, version: int = 1
) -> Features:
    if not raw.face_present:
        return raw

    v = dict(raw.values)
    if "ear" in v and profile.ear_open_baseline > 1e-6:
        # 1.0=基準の開度。小さいほど閉じ気味。
        v["ear_norm"] = v["ear"] / profile.ear_open_baseline
    if "mar" in v:
        v["mar_rel"] = v["mar"] - profile.mar_neutral
    if "pitch" in v:
        # 角度の差は ±180 を跨ぐと 360 近く跳ねるので、差そのものを畳む。
        v["pitch_rel"] = _wrap_deg(v["pitch"] - profile.head_pose_neutral.pitch)
    if "yaw" in v:
        v["yaw_rel"] = _wrap_deg(v["yaw"] - profile.head_pose_neutral.yaw)
    if "gaze_x" in v:
        # 画面中心を見たときの基準位置からのズレ。
        v["gaze_off"] = abs(v["gaze_x"] - profile.gaze_center[0])

    v["normalize_version"] = float(version)
    return Features(values=v, timestamp=raw.timestamp, face_present=True)
