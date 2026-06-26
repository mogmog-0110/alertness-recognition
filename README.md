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
