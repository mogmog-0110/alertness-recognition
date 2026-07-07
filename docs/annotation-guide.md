# アノテーション規約（テンプレ）

眠気・注意逸脱のラベルを「誰が付けても同じ」にするための判断基準を書き留める場所。
ライブ収録のラベラーも、外部データセットの変換器(`examples/convert_*.py`)も、最終的に
ここの定義に従う。空欄(TODO)を埋めて運用する。

正準ラベルは2軸 × 4段階（none / low / medium / high）。ラベルは区間 (start, end) ごとに
付ける。状態が続く範囲をひとまとめにする。

## 眠気 drowsiness（用途非依存）

用途で意味が変わらないので、全用途で共通の定義を1つ持つ。観測できる兆候で書く。
システムが出せる手がかり: 閉眼/PERCLOS(`eye_closure`)、瞬き(`blink`)、あくび(`yawn`)、
うつむき(`head_down`)。

| 段階 | 定義 | 目安の手がかり |
|---|---|---|
| none | TODO | TODO |
| low | TODO | TODO |
| medium | TODO | TODO |
| high | TODO | TODO |

## 注意逸脱 distraction（用途依存）

意味が用途で変わるため、用途(context)ごとに定義する。用途を足すたびに節を増やし、
節名は録画/manifest の `context`（study / driving 等）と一致させる。
システムが出せる手がかり: 視線外れ(`gaze_off`)、よそ向き(`head_turn`)。

### context: study（自習）

| 段階 | 定義 | 目安の手がかり |
|---|---|---|
| none | TODO | TODO |
| low | TODO | TODO |
| medium | TODO | TODO |
| high | TODO | TODO |

### context: driving（運転）

用途を扱うときに study の表をコピーして埋める。

## メモ

- 判断に迷う短い遷移は、前後どちらかの段階に寄せる（宙ぶらりんにしない）。
- 片軸しか情報が無いデータ（例: 運転データに眠気ラベルが無い）は、その軸を none と
  断定せず未アノテ（空）にできるのが理想。現状 manifest は区間内で軸別 unknown 未対応なので、
  そのデータは対象軸だけを持つ manifest として扱う。
