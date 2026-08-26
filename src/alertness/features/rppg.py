"""カメラ映像から心拍を推定する rPPG（遠隔光電脈波）。stress 判定の特徴源。

顔の額あたりの肌の平均色は、心拍に合わせて微妙に変化する。その時系列から脈波を取り出し
（POS法, Wang et al. 2017）、周波数解析で心拍[bpm]を推定する。numpy だけで動く素朴な実装で、
まず動くことを優先。より高精度が要るなら pos_signal / estimate_hr を pyVHR 等に差し替える。

状態（過去フレームの肌色バッファ）を持つので、1フレーム独立の特徴抽出とは別の口にする。
Pipeline から augment を呼び、既存の特徴量へ hr_bpm / rppg_quality を足す。
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from ..bio.hrv import plausible_rr, rmssd, rr_intervals_ms
from ..bio.peaks import peak_times
from ..contracts import FaceLandmarks, Features, Frame
from ..features import landmark_ids as ids


def pos_signal(rgb: np.ndarray) -> np.ndarray:
    """RGB 時系列(N,3) → 脈波(N,)。POS法。平均色の時間変化から拍成分を取り出す。"""
    c = np.asarray(rgb, dtype=float)
    if c.ndim != 2 or c.shape[0] < 2 or c.shape[1] != 3:
        return np.zeros(max(0, c.shape[0]))

    # 各チャンネルを時間平均で割って、照明の絶対値に依存しないようにする。
    mean = np.mean(c, axis=0)
    mean[mean < 1e-8] = 1e-8
    cn = c / mean

    # 肌色平面に直交する2方向へ射影し、分散比で足し合わせる（POSの肝）。
    s1 = cn[:, 1] - cn[:, 2]
    s2 = cn[:, 1] + cn[:, 2] - 2.0 * cn[:, 0]
    std2 = np.std(s2)
    alpha = np.std(s1) / std2 if std2 > 1e-8 else 0.0
    h = s1 + alpha * s2
    return h - np.mean(h)


def estimate_hr(
    signal: np.ndarray,
    fs: float,
    min_bpm: float = 42.0,
    max_bpm: float = 180.0,
) -> tuple[float, float]:
    """脈波と標本化周波数[Hz] → (心拍[bpm], 品質0..1)。

    FFT のビン幅は fs/N で決まり、10秒窓・30fps だと 6 bpm もある。そのままビンの中心を
    返すと心拍が 6 bpm 刻みでしか出ず、安静時からの上振れ（span 25 bpm）を測るには粗すぎる。
    ピーク周辺3点に放物線を当てて、ビンの間を補間する。

    品質はピークが雑音の底からどれだけ突出しているか。帯域全体の電力と比べると、窓を
    長くするほどビン数が増えて値が下がり、窓長ごとに意味が変わってしまう（同じ 0.3 が
    8秒窓では悪い推定、30秒窓では良い推定になる）。そこで帯域の中央ビン電力＝雑音の底と
    比べる。こちらは窓を長くすると素直に上がり、しきい値を窓長によらず使える。
    帯域内に成分が無い・標本が足りない場合は (nan, 0)。
    """
    x = np.asarray(signal, dtype=float).ravel()
    if x.size < 8 or fs <= 0:
        return float("nan"), 0.0

    x = (x - np.mean(x)) * np.hanning(x.size)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    power = np.abs(np.fft.rfft(x)) ** 2

    band = (freqs >= min_bpm / 60.0) & (freqs <= max_bpm / 60.0)
    if not np.any(band) or np.sum(power[band]) < 1e-12:
        return float("nan"), 0.0

    band_power = power[band]
    band_freqs = freqs[band]
    peak = int(np.argmax(band_power))

    spacing = float(freqs[1] - freqs[0]) if freqs.size > 1 else 0.0
    hr = float((band_freqs[peak] + _parabolic_offset(band_power, peak) * spacing) * 60.0)
    hr = min(max(hr, min_bpm), max_bpm)

    lo, hi = max(0, peak - 1), min(band_power.size, peak + 2)
    quality = _prominence_quality(band_power, lo, hi)
    return hr, quality


def estimate_respiration(
    signal: np.ndarray,
    fs: float,
    min_rpm: float = 6.0,
    max_rpm: float = 30.0,
) -> tuple[float, float]:
    """脈波の低周波成分 → (呼吸数[回/分], 品質0..1)。

    呼吸は脈波を心拍よりずっと低い周波数で揺らす（胸郭の動きが顔の位置と陰影を動かし、
    加えて呼吸性洞性不整脈が拍の間隔を周期的に変える）。帯域が心拍と重ならないので、
    同じ「帯域内の最強成分を探す」処理をそのまま低い帯域に当てれば取り出せる。

    直線の傾きを先に抜く。照明のゆっくりした変化や姿勢の沈み込みは呼吸帯域の下端に
    大きな電力を作り、抜かないと「呼吸 6 回/分」が常に最強のピークになってしまう。

    窓は心拍より長く要る。30 回/分でも 2 秒に 1 周期しかないので、10 秒窓では
    数周期しか入らず、周波数の分解能が呼吸数の刻みとして粗すぎる。
    """
    x = np.asarray(signal, dtype=float).ravel()
    if x.size < 16 or fs <= 0:
        return float("nan"), 0.0
    return estimate_hr(x - _linear_trend(x), fs, min_rpm, max_rpm)


def _linear_trend(x: np.ndarray) -> np.ndarray:
    """最小二乗で当てた直線。ゆっくりしたドリフトを抜くのに使う。"""
    t = np.arange(x.size, dtype=float)
    slope, intercept = np.polyfit(t, x, 1)
    return slope * t + intercept


# ピークが雑音の底の何倍あれば「見えている」とみなすか。合成波での実測で、これ未満の
# 推定は誤差が二桁 bpm に跳ね、これを境に急に安定する。
_NOISE_FLOOR_RATIO = 4.5


def _prominence_quality(band_power: np.ndarray, lo: int, hi: int) -> float:
    """ピークの突出度を 0..1 にする。0＝雑音の底と区別がつかない。"""
    floor = float(np.median(band_power))
    if floor <= 0:
        return 0.0
    peak = float(np.sum(band_power[lo:hi]))
    bins = max(1, hi - lo)
    prominence = peak / (bins * floor)
    if prominence <= _NOISE_FLOOR_RATIO:
        return 0.0
    return float(min(1.0, 1.0 - _NOISE_FLOOR_RATIO / prominence))


def _parabolic_offset(power: np.ndarray, peak: int) -> float:
    """ピーク前後3点に放物線を当て、頂点のビンからのずれ(-0.5..0.5)を返す。

    対数電力で当てるのが定石（窓関数のピーク形状が対数側で放物線に近いため）。
    端にあるときや形が凹んでいるときは補間しない。
    """
    if peak <= 0 or peak >= power.size - 1:
        return 0.0
    y0, y1, y2 = (math.log(max(float(v), 1e-30)) for v in power[peak - 1 : peak + 2])
    denom = y0 - 2.0 * y1 + y2
    if denom >= -1e-12:  # 上に凸でなければ頂点が定まらない
        return 0.0
    return max(-0.5, min(0.5, 0.5 * (y0 - y2) / denom))


def forehead_roi_box(
    landmarks: FaceLandmarks, width: int, height: int
) -> tuple[int, int, int, int] | None:
    """rPPG が肌色を測る額の矩形 (x0, y0, x1, y1)。取れなければ None。

    目尻2点から目幅を出し、その上（額）に矩形を置く。画面表示（どこを測っているかの
    可視化）と平均色サンプリングの両方が同じ矩形を使えるよう、計算をここに集約する。
    """
    lx, ly = landmarks.pixel(ids.LEFT_EYE_OUTER)
    rx, ry = landmarks.pixel(ids.RIGHT_EYE_OUTER)
    eye_span = float(np.hypot(rx - lx, ry - ly))
    if eye_span < 1.0:
        return None

    cx = (lx + rx) / 2.0
    cy = (ly + ry) / 2.0 - 0.4 * eye_span  # 目の上＝額
    half_w = 0.25 * eye_span
    half_h = 0.1 * eye_span

    x0 = max(0, int(cx - half_w))
    x1 = min(width, int(cx + half_w))
    y0 = max(0, int(cy - half_h))
    y1 = min(height, int(cy + half_h))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return x0, y0, x1, y1


def _with(features: Features, extra: dict[str, float]) -> Features:
    """特徴量に値を足した新しい Features を返す。足すものが無ければそのまま返す。"""
    if not extra:
        return features
    return Features(
        values={**features.values, **extra},
        timestamp=features.timestamp,
        face_present=features.face_present,
    )


def _forehead_roi_mean(image: np.ndarray, landmarks: FaceLandmarks) -> np.ndarray | None:
    """額あたりの肌領域の平均色(RGB)を返す。取れなければ None。"""
    h, w = image.shape[:2]
    box = forehead_roi_box(landmarks, w, h)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    patch = image[y0:y1, x0:x1].reshape(-1, image.shape[2])[:, :3]
    mean_bgr = patch.mean(axis=0)
    return mean_bgr[::-1].astype(float)  # OpenCV は BGR。RGB 順にして返す。


class RppgEstimator:
    """肌色バッファを持ち、貯まったら心拍を推定して特徴量へ足す。"""

    def __init__(
        self,
        fps: float = 30.0,
        window_seconds: float = 20.0,
        min_bpm: float = 42.0,
        max_bpm: float = 180.0,
        hrv_min_quality: float = 0.5,
        hrv_min_beats: int = 8,
        hrv_enabled: bool = False,
        hrv_max_ratio: float = 0.15,
        hrv_max_ms: float = 150.0,
        hrv_upsample: int = 16,
        resp_enabled: bool = True,
        resp_window_seconds: float = 30.0,
        resp_min_rpm: float = 6.0,
        resp_max_rpm: float = 30.0,
    ) -> None:
        self._fps = fps
        self._min_bpm = min_bpm
        self._max_bpm = max_bpm
        # 窓はフレーム数ではなく時間で切る。要求 fps が出ない機械だと、フレーム数で持つと
        # 窓が何倍にも伸びる（60fps 設定で実測 10fps なら 10 秒のつもりが 56 秒になる）。
        # 窓が伸びると心拍の変動と照明のドリフトが入り込み、ピークが潰れて品質が落ちる。
        self._window_seconds = window_seconds
        self._min_span = window_seconds / 2  # これだけの長さが貯まるまで推定しない
        self._max_samples = max(16, int(window_seconds * 240))  # 暴走時の上限だけ設ける
        self._buf: deque[tuple[float, np.ndarray]] = deque()
        # HRV は拍の時刻の精度で決まる。フレーム間隔そのままだと 30fps で RMSSD の下限が
        # 22ms（人の安静時 20〜50ms と同じ桁）になり測定にならないので、帯域制限補間で
        # 標本の間を埋める。x16 で下限は 30fps でも 3ms、60fps なら 2ms まで下がる。
        self._hrv_enabled = hrv_enabled
        self._hrv_upsample = hrv_upsample
        # 生理的にありえない値を弾く。実測では、拍検出が崩れると RMSSD が拍間隔の 24%
        # （225ms）といった値を出す。人の安静時は 20〜50ms、拍間隔の数%に収まる。
        self._hrv_max_ratio = hrv_max_ratio
        self._hrv_max_ms = hrv_max_ms
        self._hrv_min_quality = hrv_min_quality
        self._hrv_min_beats = hrv_min_beats
        # 呼吸は心拍より長い窓を要る（30回/分でも2秒に1周期）。心拍用の窓を伸ばすと
        # 心拍の側が変動とドリフトを拾って品質を落とすので、別のバッファを持つ。
        self._resp_enabled = resp_enabled
        self._resp_window = resp_window_seconds
        self._resp_min_rpm = resp_min_rpm
        self._resp_max_rpm = resp_max_rpm
        self._resp_buf: deque[tuple[float, np.ndarray]] = deque()
        self._resp_max_samples = max(16, int(resp_window_seconds * 240))

    def reset(self) -> None:
        """肌色バッファを捨てる。人が替わると額の色も脈も別物になる。"""
        self._buf.clear()
        self._resp_buf.clear()

    def augment(self, frame: Frame, landmarks: FaceLandmarks, features: Features) -> Features:
        if not landmarks.detected:
            return features
        rgb = _forehead_roi_mean(frame.image, landmarks)
        if rgb is None:
            return features

        self._buf.append((features.timestamp, rgb))
        cutoff = features.timestamp - self._window_seconds
        while self._buf and self._buf[0][0] < cutoff:
            self._buf.popleft()
        while len(self._buf) > self._max_samples:
            self._buf.popleft()
        respiration = self._respiration(features.timestamp, rgb)
        if len(self._buf) < 8 or self._span() < self._min_span:
            return _with(features, respiration)

        times = np.array([t for t, _ in self._buf])
        series = np.array([c for _, c in self._buf])
        fs = self._effective_fs(times)
        pulse = pos_signal(series)
        hr, quality = estimate_hr(pulse, fs, self._min_bpm, self._max_bpm)

        values = dict(features.values)
        values["hr_bpm"] = hr
        values["rppg_quality"] = quality
        hrv = self._hrv(pulse, fs, quality)
        if hrv is not None:
            values["hrv_rmssd"] = hrv
        values.update(respiration)
        return Features(values=values, timestamp=features.timestamp, face_present=True)

    def _respiration(self, now: float, rgb: np.ndarray) -> dict[str, float]:
        """呼吸数を推定して {resp_rpm, resp_quality} を返す。出せなければ空。"""
        if not self._resp_enabled:
            return {}
        self._resp_buf.append((now, rgb))
        cutoff = now - self._resp_window
        while self._resp_buf and self._resp_buf[0][0] < cutoff:
            self._resp_buf.popleft()
        while len(self._resp_buf) > self._resp_max_samples:
            self._resp_buf.popleft()

        times = np.array([t for t, _ in self._resp_buf])
        # 窓の8割は埋まっていること。呼吸1周期が最長10秒あるので、半端な窓では
        # 一番低い帯域にピークが立つだけで、呼吸数を読んだことにならない。
        if times.size < 16 or float(times[-1] - times[0]) < self._resp_window * 0.8:
            return {}
        series = np.array([c for _, c in self._resp_buf])
        fs = (times.size - 1) / float(times[-1] - times[0])
        rpm, quality = estimate_respiration(
            pos_signal(series), fs, self._resp_min_rpm, self._resp_max_rpm
        )
        if math.isnan(rpm):
            return {}
        return {"resp_rpm": float(rpm), "resp_quality": float(quality)}

    def _hrv(self, pulse: np.ndarray, fs: float, quality: float) -> float | None:
        # 品質が高く窓が満杯（＝安定して長い）ときだけ HRV(RMSSD) を返す。それ以外は None。
        if not self._hrv_enabled:
            return None
        if quality < self._hrv_min_quality or self._span() < self._window_seconds * 0.9:
            return None
        times = peak_times(pulse, fs, self._min_bpm, self._max_bpm, self._hrv_upsample)
        rr = rr_intervals_ms(times)
        if rr.size < self._hrv_min_beats:
            return None
        # 取りこぼした拍が作る 2 倍の間隔を先に外す。外さずに RMSSD を出すと、1 回の
        # 取りこぼしだけで値が跳ね、「変動が大きい＝リラックス」と正反対に読める。
        valid = plausible_rr(rr)
        if int(np.count_nonzero(valid)) < self._hrv_min_beats:
            return None  # 妥当な拍が足りない＝この窓は測れていない
        value = rmssd(rr, valid)
        if math.isnan(value) or not self._plausible(value, rr[valid]):
            return None
        return float(value)

    def _plausible(self, value: float, rr: np.ndarray) -> bool:
        # 拍検出が崩れると RMSSD は拍間隔と同じ桁まで跳ねる。心臓由来ならありえない。
        mean_rr = float(np.mean(rr))
        if mean_rr <= 0:
            return False
        return value <= self._hrv_max_ms and value <= self._hrv_max_ratio * mean_rr

    def _span(self) -> float:
        return self._buf[-1][0] - self._buf[0][0] if len(self._buf) > 1 else 0.0

    def _effective_fs(self, times: np.ndarray) -> float:
        # 実フレーム間隔から標本化周波数を出す。取れなければ公称 fps。
        span = float(times[-1] - times[0])
        if span <= 0:
            return self._fps
        return (times.size - 1) / span
