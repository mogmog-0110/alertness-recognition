"""学習側が素のデータとして書き出した時系列モデルを、推論できる形に復元する。

## なぜ必要か

pickle はオブジェクトを「クラスのモジュールパス」で記録する。学習側(alertness-colab)が
自作の推定器クラスをそのまま model.pkl に漬け込むと、アプリ側で読むときに
alertness-colab のコードが import できないと復元できない（実際 ModuleNotFoundError:
No module named 'algorithms' で落ちた）。SVM や Random Forest は scikit-learn のクラス
なので、両側に scikit-learn が入っていれば読める。自作クラスはそうはいかない。

そこで時系列モデルは、クラスではなく**素のデータ**として bundle に載せる。中身は
TorchScript にしたネットワーク（バイト列）と、スケーラの統計量とクラス名だけ。
joblib からは numpy 配列と dict と bytes にしか見えないので、復元に学習側のコードは要らない。

## なぜ TorchScript か

「重みだけ保存してアプリ側で同じネットワークを組み直す」でも動くが、それだと同じ
アーキテクチャの定義が両リポジトリに重複し、片方だけ変えたときに静かにズレる。
TorchScript は構造ごと1つのアーカイブに入るので、アプリ側は形を知らなくても動かせる。
学習側がアーキテクチャを変えても、アプリ側は何もしなくてよい。

torch 2.13 は torch.jit を非推奨にし torch.export を勧めているが、ここでは TorchScript を
使い続ける。学習は Colab、推論は手元と、torch の版が揃わない前提で動かすため。
torch.export の保存形式は版をまたいだ互換性が保証されておらず、TorchScript のアーカイブの
ほうが実績がある。非推奨は警告であって削除ではないので、torch.export の形式が安定したら
移ればよい（この関数の中だけで完結する）。

torch はこの経路でだけ要るので、遅延 import にする（SVM/RandomForest だけ使う構成では
torch を入れずに済ませたい）。
"""

from __future__ import annotations

import io
from collections.abc import Mapping
from typing import Any

import numpy as np

# bundle の models の値が「素のデータで書かれた時系列モデル」であることを示す印。
# 学習側 (alertness-colab の algorithms/lstm.py) が同じ文字列を書き込む。
SEQUENCE_KIND = "lstm"


class SequenceModel:
    """TorchScript のネットワークを scikit-learn 互換の口で包む。

    MLClassifier が使うのは predict / predict_proba / classes_ の3つだけなので、
    それだけを備える。入力は (件数, 窓, 特徴量) の3次元。
    """

    def __init__(self, spec: Mapping[str, Any]) -> None:
        script = spec.get("torchscript")
        if not script:
            raise ValueError("時系列モデルに torchscript が入っていません。")
        self.classes_ = [str(c) for c in spec.get("classes", ())]
        if not self.classes_:
            raise ValueError("時系列モデルにクラス名(classes)が入っていません。")
        self.window = max(1, int(spec.get("window") or 1))
        self._center = np.asarray(spec.get("center", 0.0), dtype=np.float32)
        self._scale = np.asarray(spec.get("scale", 1.0), dtype=np.float32)

        import torch  # この経路でだけ要る重い依存なので、ここで読む。

        self._torch = torch
        self._net = torch.jit.load(io.BytesIO(script))
        self._net.eval()

    def predict(self, x) -> np.ndarray:
        probabilities = self.predict_proba(x)
        return np.array([self.classes_[i] for i in probabilities.argmax(axis=1)])

    def predict_proba(self, x) -> np.ndarray:
        features = np.asarray(x, dtype=np.float32)
        if features.ndim != 3:
            raise ValueError(
                f"時系列モデルは (件数, 窓, 特徴量) の3次元入力を取ります。渡されたのは"
                f" {features.shape}。model.pkl の window に合わせて窓を作っているか確認。"
            )
        scaled = (features - self._center) / self._scale
        with self._torch.no_grad():
            logits = self._net(self._torch.from_numpy(scaled.astype(np.float32)))
            return self._torch.softmax(logits, dim=1).numpy()


def is_sequence_spec(value: Any) -> bool:
    return isinstance(value, Mapping) and value.get("kind") == SEQUENCE_KIND


def load_model(value: Any) -> Any:
    """bundle の models の値を、predict を持つオブジェクトにする。

    scikit-learn 由来のモデルはそのまま通す（既に predict を持っている）。素のデータで
    書かれた時系列モデルだけ、ここで復元する。
    """
    return SequenceModel(value) if is_sequence_spec(value) else value


def load_models(models: Mapping[str, Any]) -> dict[str, Any]:
    return {axis: load_model(model) for axis, model in models.items()}


def window_of(models: Mapping[str, Any], fallback: int = 1) -> int:
    """models が要求する窓の長さ。bundle 直下に window を持たない成果物への保険。

    窓の長さは本来 bundle 直下の window が正で、こちらは無いときだけ使う。軸ごとに
    違う長さを要求されたら、1つの Observation から両方を作れないので止める。
    """
    windows = {int(getattr(model, "window", 1)) for model in models.values()}
    windows.discard(1)
    if len(windows) > 1:
        raise ValueError(f"軸ごとに窓の長さが違います: {sorted(windows)}")
    return windows.pop() if windows else fallback
