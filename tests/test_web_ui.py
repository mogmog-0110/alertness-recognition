"""ブラウザ版の言語選択と翻訳カタログの静的な契約を検証する。

表示処理は web/index.html 内で完結する。ここでは日英のキーずれによって一部だけ
日本語へ戻る事故と、初期言語・互換フォールバックが失われる変更を検出する。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from alertness.guided import PROTOCOLS

_PAGE = Path(__file__).parents[1] / "web" / "index.html"


def _page_and_catalog() -> tuple[str, dict]:
    page = _PAGE.read_text(encoding="utf-8")
    match = re.search(
        r'<script id="translations" type="application/json">\s*(.*?)\s*</script>',
        page,
        re.DOTALL,
    )
    assert match is not None
    return page, json.loads(match.group(1))


def test_japanese_is_the_default_and_both_language_controls_are_visible():
    page, _ = _page_and_catalog()

    assert '<html lang="ja">' in page
    assert 'data-locale="ja" aria-pressed="true">JP</button>' in page
    assert 'data-locale="en" aria-pressed="false">EN</button>' in page
    assert "localStorage" not in page
    assert "sessionStorage" not in page


def test_translation_catalogs_have_the_same_nonempty_shape():
    _, catalog = _page_and_catalog()
    japanese, english = catalog["ja"], catalog["en"]

    assert japanese.keys() == english.keys()
    assert japanese["dimensions"].keys() == english["dimensions"].keys()
    assert japanese["prompts"].keys() == english["prompts"].keys()
    for language in (japanese, english):
        assert all(value for key, value in language.items() if key not in {"dimensions", "prompts"})
        assert all(language["dimensions"].values())
        for prompt in language["prompts"].values():
            assert prompt["title"]
            assert prompt["instruction"]


def test_every_built_in_guided_prompt_can_be_shown_in_both_languages():
    _, catalog = _page_and_catalog()
    prompt_keys = {prompt.key for prompts in PROTOCOLS.values() for prompt in prompts}
    prompt_keys.add("done")

    assert prompt_keys == set(catalog["ja"]["prompts"])
    assert prompt_keys == set(catalog["en"]["prompts"])


def test_web_client_keeps_legacy_payload_fallbacks():
    page, _ = _page_and_catalog()

    assert "result.message || words.warning" in page
    assert 'g.title || ""' in page
    assert 'g.instruction || ""' in page


def test_every_state_label_is_a_catalog_key():
    # setState はカタログを直に引くので、キーではなく表示文言を渡すと textContent が
    # undefined になる。日本語表示でも英語表示でも画面に "undefined" と出る。
    page, catalog = _page_and_catalog()
    keys = set(re.findall(r'setState\(\s*"([^"]+)"', page))

    assert keys
    assert keys <= set(catalog["ja"])
    assert keys <= set(catalog["en"])
