# alertness-recognition

PCのカメラに映った顔から、眠気とよそ見（注意の逸れ）を判定して画面に出すデモ。

## セットアップ（Windows）

Python 3.10〜3.12が必要。入っていなければ <https://www.python.org/downloads/> から。

このリポジトリをクローン後、

```bat
scripts\setup.bat
```
を実行。
venv の作成から依存のインストール、モデルの取得まで自動化するので、初回だけ実行すればいい。

消したいときは、

```bat
scripts\clean.bat      :: ライブラリ・モデル・キャッシュを削除（コードは残す）
scripts\uninstall.bat  :: ライブラリだけ消すか丸ごと消すかをメニューで選ぶ
```

## 動かす

```bat
scripts\run.bat
```

カメラが立ち上がる。最初に数秒のキャリブレーションが入るので、正面を向いて目を開けたまま待つ。
基準が取れたら判定モードに切り替わり、眠気とよそ見のレベルとスコアが映像に重なって出る。

操作はキー2つ。

- `q` … 終了
- `c` … キャリブレーションのやり直し。席を立った後や、別の人に替わったときに押す

カメラがなくても動画ファイルで試せる。`--record` を付けると、判定しながら各フレームの
特徴量を `runs\` に CSV で残せる（そのまま `report.bat` で採点できる）。

```bat
scripts\run.bat --video path\to\clip.mp4   :: カメラの代わりに動画を流す
scripts\run.bat --record                   :: 判定の裏で特徴量CSVを runs\ に保存
```

しきい値やキャリブの有無などの設定は `config\default.yaml`。`--config` で別ファイルも渡せる。

## 画面を録画する

デモの様子を動画で残したいときは `record.bat`。画面を録画しながらデモを起動し、
終了すると `recordings\` に mp4 が残る。前述の `--record`（特徴量CSV）とは別で、こちらは映像そのもの。
ffmpeg が要る（無ければ `winget install Gyan.FFmpeg`）。

```bat
scripts\record.bat                            :: 画面を録画しながらデモ起動
scripts\record.bat --video clip.mp4           :: 引数はそのままデモへ渡る
scripts\record.bat --region title=Alertness   :: デモ窓だけ録る
```

## データ収集と採点

ルールの閾値はラベル付きデータで詰める。録る → 採点する、の2ステップ。

### 1. 録る

`collect.bat` を叩くと、まず数秒のキャリブレーション（正面・開眼）が入り、続けて
「○○の状態にしてください」という指示が順に出る。目・口・頭を指示どおりに動かすだけで、
ラベル付きの CSV が `runs/` に溜まっていく。既定は3周、区切りで合図音が鳴る。

```bat
scripts\collect.bat                 :: ガイドに従って録る
scripts\collect.bat --subject taro  :: 複数人で録るときは被験者IDを付ける
```

録画中は `q` で中断、`c` でキャリブレーションのやり直し。

### 2. 採点する

`runs/` に溜めた CSV をまとめて採点する。引数なしなら `runs/` の全ファイルが対象。

```bat
scripts\report.bat                   :: 分布と採点表を表示
scripts\report.bat --out report.txt  :: 結果をファイルに保存（UTF-8）
```

採点表には accuracy、macro-F1、誤警告率、見逃し率、クラス別の成績、混同行列が出る。

## 外部データセットを取り込む

自前収録（前節）のほかに、顔動画の公開データセットからも学習用CSVを作れる。アノテは
区間（または動画1本）単位の粗いラベルでよい。

まず、データセット固有の変換器で「動画＋区間ラベル」を manifest(JSON) にする。
`examples\_convert_template.py` をコピーして写像を書く（数値尺度は `ordinal_bin`、クラス名は
`lookup`。運転行動アノテの例は `examples\convert_activity_example.py`）。次に manifest から
特徴量CSVを作る（`.venv` を有効にして実行）。

```bat
python -m alertness.ingest --manifests data\manifests   :: フォルダ内の *.json をまとめて取り込む
python -m alertness.ingest --manifests clip.json         :: 単体のJSONでも可
```

CSVは `runs\ingested\` に出る（`--out` で変更可）。列は自前収録と同じなので、以降の採点や学習は
まったく同じに扱える。正準ラベルは2軸（drowsiness / distraction）× 4段階（none/low/medium/high）。
どのラベルをどの段階に写すか＝アノテ規約は `docs\annotation-guide.md` に集約する（自前収録の
ラベリングもこの規約に従う）。

### DROZYのPSGから眠気ラベルを生成する

DROZY変換だけはEDF/信号処理用の追加依存を使う。通常のWebカメラ実行には不要。

```bat
pip install -e ".[drozy]"
python examples\convert_drozy.py path\to\DROZY --out data\manifests
python -m alertness.ingest --manifests data\manifests --out runs\ingested
```

1段目はPSGを主な教師情報、PVTを境界校正、KSSを検証情報として眠気区間manifestを作る。
2段目は既存ingestで動画特徴量と `label_drowsiness` を持つCanonical CSVへ変換する。
変換パラメータは `config\default.yaml` の `drozy`、既存manifestの上書きは `--force`、
被験者の限定は `--subject ID` で指定する。実データでPVT/KSSとの整合を確認するまでは、
生成ラベルを生理学的な正解として確定しないこと。

## 学習モデル（ML）で判定する

既定はルールベース判定だが、学習済みモデルに差し替えられる。モデル（`model.pkl`）の学習は
別リポジトリ **[alertness-colab](https://github.com/mogmog-0110/alertness-colab)** が担当する。
役割分担は：

- このリポジトリ … カメラ/動画 → 特徴量、ルール判定、特徴量CSVの書き出し、ML判定
- alertness-colab … 特徴量CSV → `model.pkl`（Colab でノートを実行するだけ）

手順：

1. 特徴量CSV（自前収録 or 取り込み）を alertness-colab に渡して `model.pkl` を作る。
2. できた `model.pkl` を `models\` に置く。
3. ML用の依存を入れる。

   ```bat
   .venv\Scripts\activate
   pip install -e ".[ml]"
   ```

4. `config\default.yaml` の policy を切り替える。

   ```yaml
   policy:
     type: ml
     model_path: models/model.pkl
   ```

あとは `run.bat` などいつもどおり動かせば、ルールの代わりに `model.pkl` で判定する。用途別
モデル（自習用・運転用など）は `model_path` を `models/model_study.pkl` のように向けるだけで
切り替わる。眠気は用途に依存しないので全データ、注意逸脱だけが用途別に学習される（詳細は
alertness-colab 側）。
