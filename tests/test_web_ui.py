"""ブラウザ版の画面と翻訳カタログの静的な契約を検証する。

表示処理は web/ の中で完結する。ここでは日英のキーずれによって一部だけ日本語へ戻る事故と、
サーバが送る名前（cue・音・待ちの理由）に端末側の受け皿が無いまま増える変更を検出する。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from alertness.guided import PROTOCOLS

_WEB = Path(__file__).parents[1] / "web"
_CONFIG = Path(__file__).parents[1] / "config" / "browser.yaml"


def _page_and_catalog() -> tuple[str, dict]:
    page = (_WEB / "index.html").read_text(encoding="utf-8")
    match = re.search(
        r'<script id="translations" type="application/json">\s*(.*?)\s*</script>',
        page,
        re.DOTALL,
    )
    assert match is not None
    return page, json.loads(match.group(1))


def _script(name: str) -> str:
    return (_WEB / name).read_text(encoding="utf-8")


def _scripts() -> str:
    return "\n".join(_script(p.name) for p in sorted(_WEB.glob("*.js")))


def test_japanese_is_the_default_and_both_language_controls_are_visible():
    page, _ = _page_and_catalog()

    assert '<html lang="ja">' in page
    assert 'data-locale="ja" aria-pressed="true">JP</button>' in page
    assert 'data-locale="en" aria-pressed="false">EN</button>' in page
    assert 'return "ja";' in _script("view.js")


def test_the_language_choice_survives_a_reload_but_storage_is_optional():
    # 再接続でページを開き直すたびに日本語へ戻ると、英語で見せている最中に困る。
    # 保存が使えない環境（プライベートブラウズ等）でも落ちないよう、必ず try で包む。
    view = _script("view.js")
    assert "try {\n    const saved = localStorage.getItem(LOCALE_KEY)" in view
    assert "try { localStorage.setItem(LOCALE_KEY, next); }" in view


def test_translation_catalogs_have_the_same_nonempty_shape():
    _, catalog = _page_and_catalog()
    japanese, english = catalog["ja"], catalog["en"]
    nested = {"dimensions", "causes", "prompts"}

    assert japanese.keys() == english.keys()
    for key in nested - {"prompts"}:
        assert japanese[key].keys() == english[key].keys()
        assert all(japanese[key].values()) and all(english[key].values())
    assert japanese["prompts"].keys() == english["prompts"].keys()
    for language in (japanese, english):
        assert all(value for key, value in language.items() if key not in nested)
        for prompt in language["prompts"].values():
            assert prompt["title"]
            assert prompt["instruction"]


def test_every_built_in_guided_prompt_can_be_shown_in_both_languages():
    _, catalog = _page_and_catalog()
    prompt_keys = {prompt.key for prompts in PROTOCOLS.values() for prompt in prompts}
    prompt_keys.add("done")

    assert prompt_keys == set(catalog["ja"]["prompts"])
    assert prompt_keys == set(catalog["en"]["prompts"])


def test_every_configured_cue_has_a_cause_to_show():
    # サーバは警告の原因を cue 名で送る。受け皿が無い cue は原因の行が出ないだけで、
    # 壊れていることに誰も気づかない。
    _, catalog = _page_and_catalog()
    config = yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))
    cues = {c for d in config["assessment"]["dimensions"] for c in d["cues"]}

    assert cues <= set(catalog["ja"]["causes"])
    assert cues <= set(catalog["en"]["causes"])


def test_every_configured_sound_can_be_played():
    config = yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))
    patterns = set(re.findall(r"^\s+(\w+): \{", _script("sound.js"), re.MULTILINE))

    assert set(config["feedback"]["sounds"].values()) <= patterns
    assert {"ready", "done"} <= patterns


def test_every_calibration_wait_reason_has_a_message():
    # calibrator.waiting_for が返す理由。無い理由は黙って空行になる。
    view = _script("view.js")
    for reason in ("hr_bpm", "face", "steady"):
        assert re.search(rf"\b{reason}: w\.\w+", view), reason


def test_web_client_keeps_legacy_payload_fallbacks():
    view = _script("view.js")

    assert "shown.message || w.warning" in view
    assert 'g.title || ""' in view
    assert 'g.instruction || ""' in view


def test_every_state_label_is_a_catalog_key():
    # setState はカタログを直に引くので、キーではなく表示文言を渡すと textContent が
    # undefined になる。日本語表示でも英語表示でも画面に "undefined" と出る。
    _, catalog = _page_and_catalog()
    keys = set(re.findall(r'setState\(\s*"([^"]+)"', _scripts()))

    assert keys
    assert keys <= set(catalog["ja"])
    assert keys <= set(catalog["en"])


def test_every_icon_the_view_refers_to_exists():
    page, _ = _page_and_catalog()
    view = _script("view.js")
    icons = set(re.findall(r'"(\w+)"', view.split("const ICONS = ", 1)[1].split(";", 1)[0]))
    icons.add("face")

    for icon in icons:
        assert f'<symbol id="icon-{icon}"' in page, icon
