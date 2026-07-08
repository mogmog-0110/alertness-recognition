"""行動クラス名の注意逸脱アノテを manifest に写す変換器の例（運転監視データセット等の想定）。

核の外に置く使い捨て想定。KSS例（convert_kss_example.py）は数値スケールを ordinal_bin で
写したが、こちらは「クラス名 → 段階」の対応表(lookup)で写す形。アノテが行動ラベル
（脇見・携帯操作…）で与えられるデータセットはこちら。どの行動をどの段階にするか＝アノテ規約を、
対応表として1か所に明示する。注意逸脱の意味は用途で変わるので context も必ず付ける。
実データが無いのでダミーのセクションを manifest 化する。
"""

from alertness.ingest.mapping import lookup, segment, write_manifest

# アノテ規約: 行動クラス → distraction 段階（運転文脈での例）。
DISTRACTION_BY_ACTIVITY = {
    "safe_driving": "none",
    "talking_passenger": "low",
    "adjusting_radio": "low",
    "talking_phone": "medium",
    "reaching_behind": "medium",
    "texting": "high",
}

# データセット固有: 生アノテの読み取り（ここではダミー）。(start秒, end秒, 行動クラス)。
sections = [
    (0.0, 30.0, "safe_driving"),
    (30.0, 45.0, "talking_phone"),
    (45.0, 60.0, "texting"),
]

# 対応表に無いクラスは none（＝注意逸脱なし）に落とす。
segments = [
    segment(start, end, distraction=lookup(activity, DISTRACTION_BY_ACTIVITY, default="none"))
    for (start, end, activity) in sections
]

path = write_manifest(
    "data/manifests/example_driver01.json",
    video="driver01.mp4",
    subject="driver01",
    context="driving",
    segments=segments,
)
print(f"manifest を書き出しました: {path}")
