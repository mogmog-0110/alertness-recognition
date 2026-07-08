"""KSS(1-9)の眠気データを manifest に写す変換器の例。核の外に置く使い捨て想定。

実際は「生ラベルを (start, end, kss) の形に読み出す部分」を、各データセットの
配布フォーマットに合わせて書き換える。写像の判断（KSSのしきい値など）もここに明示する。
実データが無いのでダミーのセクションを1本 manifest 化する。
"""

from alertness.ingest.mapping import ordinal_bin, segment, write_manifest

# データセット固有: 生ラベルの読み取り（ここではダミー）。(start秒, end秒, KSS)。
sections = [(0.0, 240.0, 3), (240.0, 480.0, 7), (480.0, 720.0, 9)]

segments = [
    segment(start, end, drowsiness=ordinal_bin(kss, [4, 6, 8]))
    for (start, end, kss) in sections
]

path = write_manifest("data/manifests/example_s01.json", "s01.mp4", "s01", "driving", segments)
print(f"manifest を書き出しました: {path}")
