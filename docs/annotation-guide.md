# アノテーション規約（テンプレ）

各評価軸のラベルを「誰が付けても同じ」にするための判断基準を書き留める場所。
ライブ収録のラベラーも、外部データセットの変換器(`examples/convert_*.py`)も、最終的に
ここの定義に従う。空欄(TODO)を埋めて運用する。

正準ラベルは4軸（drowsiness / distraction / concentration / stress）× 4段階
（none / low / medium / high）。ラベルは区間 (start, end) ごとに付け、状態が続く範囲を
ひとまとめにする。**区間には情報のある軸だけを付ければよい**（付けなかった軸は未アノテ＝
none とは断定しない）。だから「眠気だけ」「ストレスだけ」のデータも、他軸を誤って none と
埋めずに取り込める。データセットごとに得意な軸だけ供給する運用を前提にする。

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

## 集中 concentration（用途依存）

注意逸脱の裏返しに近いが、厳密な補数ではない（どちらでもない中立がある）ので独立した軸として
持つ。engagement 系データセット（DAiSEE 等）の段階ラベルを写して使う想定。意味が用途で変わる
ため、注意逸脱と同じく用途(context)ごとに定義する。観測できる兆候: 対象への視線が載り続ける、
頭部姿勢が安定、瞬きの規則性。

### context: driving（運転）

| 段階 | 定義 | 目安の手がかり |
|---|---|---|
| none | TODO | TODO |
| low | TODO | TODO |
| medium | TODO | TODO |
| high | TODO | TODO |

## ストレス stress（用途非依存）

映像だけでは客観的に判断しにくいため、**人手アノテではなく生体信号から算出したラベルを写す**
のが基本（例: PPG/ECG の HRV からストレス指標を出し段階化）。用途で意味は変わらないので定義は
1つ。cue を持たないため rule 経路では常に none、ML 経路（生体信号ラベルで学習したモデル）でのみ
値が出る。

| 段階 | 定義（生体指標のしきい値で書く） | 備考 |
|---|---|---|
| none | TODO | TODO |
| low | TODO | TODO |
| medium | TODO | TODO |
| high | TODO | TODO |

## メモ

- 判断に迷う短い遷移は、前後どちらかの段階に寄せる（宙ぶらりんにしない）。
- 片軸しか情報が無いデータ（例: 運転データに眠気ラベルが無い）は、その軸を付けずに区間へ
  写せばよい。付けなかった軸は未アノテ（空）として扱われ、採点・学習から自動で外れる。
