"""UBFC-Phys の自動処理ドライバ。ディスクを食わない順序（ゲート先・展開後）を確かめる。"""

from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(_EXAMPLES))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


proc = _load("process_ubfc_phys")


def _anx_csv(cognitive: tuple[float, float]) -> str:
    # 3行2列: 認知不安 / 身体不安 / 自信、列=前,後。
    return f"{cognitive[0]},{cognitive[1]}\n1.0,1.0\n3.0,3.0\n"


def _make_zip(path: Path, subject: str, scenario: str, cognitive: tuple[float, float]) -> None:
    """小さいファイルと、ダミーの(中身は小さい).avi を持つ zip を作る。"""
    info = f"{subject}\nm\n{scenario}\n2019_01_01\n10_00_00"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"{subject}/info_{subject}.txt", info)
        z.writestr(f"{subject}/selfReportedAnx_{subject}.csv", _anx_csv(cognitive))
        for task in ("T1", "T2", "T3"):
            z.writestr(f"{subject}/bvp_{subject}_{task}.csv", "\n".join(["70.0"] * 200))
            z.writestr(f"{subject}/vid_{subject}_{task}.avi", b"x" * 1024)  # ダミー動画


def test_rejected_subject_never_extracts_video(tmp_path, monkeypatch):
    # 誘発が効かなかった被験者（認知不安が下がる）は不採用。動画を展開しないこと。
    root = tmp_path / "UBFC-Phys"
    root.mkdir()
    zip_path = root / "s9.zip"
    _make_zip(zip_path, "s9", "ctrl", cognitive=(2.0, 1.5))  # 下降 → スコア負

    result = proc.process_subject(zip_path, {}, root, tmp_path / "out", keep_videos=False)

    assert "不採用" in result
    assert not list((root / "s9").glob("*.avi"))  # 動画は1本も展開されていない
    assert (root / "s9" / "info_s9.txt").exists()  # 小さいファイルは展開済み


def test_already_ingested_subject_is_skipped(tmp_path):
    root = tmp_path / "UBFC-Phys"
    root.mkdir()
    zip_path = root / "s9.zip"
    _make_zip(zip_path, "s9", "test", cognitive=(2.0, 3.5))
    out = tmp_path / "out"
    (out / "s9__vid_s9_T1").mkdir(parents=True)  # 取り込み済みの跡

    result = proc.process_subject(zip_path, {}, root, out, keep_videos=False)
    assert "飛ばす" in result
    assert not (root / "s9").exists()  # 触っていない（展開すらしない）


def test_accepted_subject_ingests_and_removes_video(tmp_path, monkeypatch):
    # 採用被験者は動画を展開して取り込み、済んだら動画を消す。
    root = tmp_path / "UBFC-Phys"
    root.mkdir()
    zip_path = root / "s9.zip"
    _make_zip(zip_path, "s9", "test", cognitive=(2.0, 3.5))  # 上昇 → 採用

    # 重い取り込み・manifest生成は差し替え、ドライバの制御フローだけを見る。
    monkeypatch.setattr(proc, "run_ingest", lambda *a, **k: None)
    monkeypatch.setattr(proc, "load_manifest", lambda p: p)
    monkeypatch.chdir(tmp_path)  # convert_subject は data/manifests に書くので作業場所を移す

    def fake_convert(root_, subject, out_dir, *, force):
        return [Path("m1"), Path("m2"), Path("m3")]

    monkeypatch.setattr(proc.conv, "convert_subject", fake_convert)

    result = proc.process_subject(zip_path, {}, root, tmp_path / "out", keep_videos=False)

    assert "取り込み 3 タスク" in result
    assert not list((root / "s9").glob("*.avi"))  # 取り込み後、動画は削除された


def test_accepted_subject_can_keep_video(tmp_path, monkeypatch):
    root = tmp_path / "UBFC-Phys"
    root.mkdir()
    zip_path = root / "s9.zip"
    _make_zip(zip_path, "s9", "test", cognitive=(2.0, 3.5))
    monkeypatch.setattr(proc, "run_ingest", lambda *a, **k: None)
    monkeypatch.setattr(proc, "load_manifest", lambda p: p)
    monkeypatch.setattr(proc.conv, "convert_subject", lambda *a, **k: [Path("m1")])
    monkeypatch.chdir(tmp_path)

    proc.process_subject(zip_path, {}, root, tmp_path / "out", keep_videos=True)

    assert len(list((root / "s9").glob("*.avi"))) == 3  # 残っている
