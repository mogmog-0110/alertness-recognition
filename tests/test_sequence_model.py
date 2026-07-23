"""素のデータで書かれた時系列モデルの復元。

要点は「学習側(alertness-colab)のコードを import せずに読めること」なので、
このテストでは torch だけを使い、学習側のクラスには一切触れずに成果物を組み立てる。
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from alertness.classifier.sequence_model import (
    SequenceModel,
    is_sequence_spec,
    load_model,
    load_models,
    window_of,
)


class _Plain:
    """scikit-learn 由来のモデルの代役。predict を既に持っている。"""

    window = 1

    def predict(self, x):
        return ["none"] * len(x)


def _spec(window: int = 4, features: int = 3, classes=("none", "high")) -> dict:
    """学習側と同じ鍵を持つ成果物を、torch だけで組み立てる。"""
    torch = pytest.importorskip("torch")

    class _Net(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = torch.nn.LSTM(features, 8, 1, batch_first=True)
            self.fc = torch.nn.Linear(8, len(classes))

        def forward(self, x):
            output, _ = self.lstm(x)
            return self.fc(output[:, -1, :])

    buffer = io.BytesIO()
    torch.jit.save(torch.jit.script(_Net().eval()), buffer)
    return {
        "kind": "lstm",
        "window": window,
        "classes": list(classes),
        "center": np.zeros(features, dtype=np.float32),
        "scale": np.ones(features, dtype=np.float32),
        "torchscript": buffer.getvalue(),
    }


def test_plain_models_pass_through():
    # scikit-learn 由来のモデルは触らずそのまま返す。
    model = _Plain()
    assert load_model(model) is model
    assert not is_sequence_spec(model)


def test_sequence_spec_is_recognised():
    spec = _spec()
    assert is_sequence_spec(spec)
    assert isinstance(load_model(spec), SequenceModel)


def test_sequence_model_predicts_from_window():
    model = load_model(_spec(window=4, features=3))
    x = np.zeros((2, 4, 3), dtype=np.float32)

    probabilities = model.predict_proba(x)
    assert probabilities.shape == (2, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)  # softmax なので各行が1

    predicted = model.predict(x)
    assert list(predicted) == [model.classes_[i] for i in probabilities.argmax(axis=1)]


def test_sequence_model_rejects_two_dimensional_input():
    model = load_model(_spec(window=4, features=3))
    with pytest.raises(ValueError, match="3次元"):
        model.predict(np.zeros((2, 3), dtype=np.float32))


def test_scaler_statistics_are_applied():
    # center/scale が効いていることを、同じ入力を二通りの統計量で通して確かめる。
    spec = _spec(window=4, features=3)
    plain = load_model(dict(spec))
    shifted = load_model({**spec, "center": np.full(3, 5.0, dtype=np.float32)})
    x = np.full((1, 4, 3), 5.0, dtype=np.float32)

    assert not np.allclose(plain.predict_proba(x), shifted.predict_proba(x))


def test_missing_torchscript_is_rejected():
    with pytest.raises(ValueError, match="torchscript"):
        SequenceModel({"kind": "lstm", "classes": ["none"], "window": 2})


def test_missing_classes_is_rejected():
    spec = _spec()
    with pytest.raises(ValueError, match="クラス名"):
        SequenceModel({**spec, "classes": []})


def test_window_of_reads_models():
    models = load_models({"label_stress": _spec(window=7)})
    assert window_of(models) == 7


def test_window_of_defaults_when_all_are_frame_wise():
    assert window_of({"label_stress": _Plain()}) == 1


def test_window_of_rejects_mixed_windows():
    models = load_models({"a": _spec(window=4), "b": _spec(window=9)})
    with pytest.raises(ValueError, match="窓の長さ"):
        window_of(models)
