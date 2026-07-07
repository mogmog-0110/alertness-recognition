"""データセット→manifest 変換器のテンプレート。コピーして examples/convert_<名前>.py を作る。

核の外・使い捨て前提（データセットごとに手書きする）。やることは3つだけ:
  1. 生アノテを (start秒, end秒, 元ラベル) の並びに読み出す ← データセット固有。ここを書く
  2. 元ラベルを段階(none/low/medium/high)へ写す ← ここが「アノテ規約」。しきい値/対応表で明示
  3. segment / write_manifest で manifest(JSON) にする ← 定型

正準ラベルは2軸（drowsiness / distraction）。片方しか情報が無いデータセットは、その軸だけ
写せばよい（もう片方は none 既定）。数値/順序尺度は ordinal_bin、クラス名は lookup で写す
（クラス名の写し方は convert_activity_example.py を参照）。
各段階(none/low/medium/high)の意味は docs/annotation-guide.md（アノテ規約）に従う。
"""

from alertness.ingest.mapping import ordinal_bin, segment, write_manifest

# 1. 生アノテの読み取り（TODO: 実データの配布形式に合わせて書き換える）。今はダミー。
#    ここでは (start秒, end秒, 元ラベル値) の並びとする。
raw_sections = [
    (0.0, 60.0, 2),
    (60.0, 120.0, 7),
]

# 2. アノテ規約（TODO: このデータセットでの写像を決めて明示する）。
#    下は「元値<4=none / 4-5=low / 6-7=medium / 8以上=high」という規約の例。
segments = [
    segment(start, end, drowsiness=ordinal_bin(raw, [4, 6, 8]))
    for (start, end, raw) in raw_sections
]

# 3. manifest に書き出す（TODO: 出力名・video・subject・context を埋める）。
path = write_manifest(
    "data/manifests/CHANGE_ME.json",
    video="CHANGE_ME.mp4",
    subject="s01",
    context="",  # 用途を分けるなら study / driving 等を入れる
    segments=segments,
)
print(f"manifest を書き出しました: {path}")
