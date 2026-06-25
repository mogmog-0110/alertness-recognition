"""MediaPipe FaceLandmarker のランドマーク番号。

EAR は片目6点で計算する。並びは (p1, p2, p3, p4, p5, p6) で、
p1/p4 が目頭・目尻（横幅）、p2/p3 が上まぶた、p5/p6 が下まぶた。
"""

from __future__ import annotations

# 左目（画像の左側＝本人の右目）
LEFT_EYE_EAR = (33, 160, 158, 133, 153, 144)
# 右目（画像の右側＝本人の左目）
RIGHT_EYE_EAR = (362, 385, 387, 263, 373, 380)

# 口（縦＝開き、横＝口幅）
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291

# 虹彩中心（478点モデルで有効）
LEFT_IRIS = 468
RIGHT_IRIS = 473

# 目頭・目尻（視線の水平比に使う）
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263

# 頭部姿勢推定(solvePnP)に使う代表点
NOSE_TIP = 1
CHIN = 152
LEFT_EYE_CORNER = 33
RIGHT_EYE_CORNER = 263
LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291
