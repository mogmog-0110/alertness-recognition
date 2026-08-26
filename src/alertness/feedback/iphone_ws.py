"""判定結果を iPhone へ返す出力先。映像を送ってきたのと同じ接続を使う。

端末は運転者の手元にあるので、ここが実際の警告の出口になる。画面の文言と、音・振動を
出すかどうかだけを送る。特徴量は開発中の切り分け用で、無くても端末は動く。

軸の表示名は config から受け取る。OpenCV の窓は日本語を描けないので軸名は英語のままに
してあるが、端末の画面は日本語で出せる。名前の対応をここに埋め込むと軸を増やすたびに
コードを触ることになるので、対応表は設定に置く。
"""

from __future__ import annotations

from collections.abc import Mapping

from ..contracts import Assessment, Level, Observation

_LEVEL_NAME = {
    Level.NONE: "none",
    Level.LOW: "low",
    Level.MEDIUM: "medium",
    Level.HIGH: "high",
}


class IPhoneSink:
    def __init__(
        self,
        link,
        alert_from: Level = Level.MEDIUM,
        features: tuple[str, ...] = (),
        names: Mapping[str, str] | None = None,
    ) -> None:
        self._link = link
        self._alert_from = alert_from
        self._features = features
        self._names = dict(names or {})

    def emit(self, obs: Observation, assessment: Assessment) -> None:
        level = assessment.alert_level()
        payload: dict = {
            "timestamp": assessment.timestamp,
            "level": _LEVEL_NAME.get(level, "none"),
            "alert": level >= self._alert_from,
        }
        head = assessment.headline()
        if head is not None:
            # 対応表の見出しは画面に出る名前（alert_name があればそちら）。集中の軸は
            # 「集中」ではなく「注意散漫」として警告するので、警告名の側で引く。
            shown = self._names.get(head.display_name, head.display_name)
            payload["dimension"] = shown
            payload["message"] = self._message(shown, level)
        if self._features:
            payload["features"] = self._selected(obs)
        self._link.send(payload)

    def _selected(self, obs: Observation) -> dict[str, float]:
        # NaN は JSON では表せない（json は NaN を吐くが受け側の JSONDecoder が拒む）。
        # 測れていない特徴はキーごと落とし、端末側では「欠けている」と扱わせる。
        return {
            name: value
            for name, value in obs.features.values.items()
            if name in self._features and value == value
        }

    @staticmethod
    def _message(name: str, level: Level) -> str:
        if level >= Level.HIGH:
            return f"{name}が強いです"
        if level == Level.MEDIUM:
            return f"{name}の兆候があります"
        return name

    def close(self) -> None:
        pass  # 接続は IPhoneLink の持ち物。source 側が閉じる
