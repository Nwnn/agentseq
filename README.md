# LLM パイプライン実験プロジェクト

このプロジェクトは、実験計画メモに基づいたマルチステップLLMパイプラインの実験を実施します。局所最強戦略とネットワーク最適戦略を比較し、タスク精度と埋め込み距離のトレードオフを分析します。

## 特徴

- **柔軟な設定**: nステップのパイプラインをYAML設定ファイルで自由に定義可能。各ステップごとにモデルとプロンプトをカスタマイズ。
- **データセット対応**: AG NewsとYelp Polarityの両方をサポート。
- **OpenRouter統合**: OpenRouter API経由でさまざまなLLMモデルを使用。
- **埋め込み分析**: Sentence-BERTによる表現の相性分析。
- **自動解析**: 戦略比較とパレートフロンティアの可視化。

## セットアップ

1. 依存関係をインストール:
   ```bash
   pip install -r requirements.txt
   ```

2. OpenRouter APIキーを環境変数に設定:
   ```bash
   export OPENROUTER_API_KEY="your_api_key_here"
   ```

3. 設定ファイルを編集: `config.yaml`
   - データセット選択
   - ステップ定義（モデル、プロンプト）
   - パラメータ調整

## 使用方法

1. 実験実行:
   ```bash
   cd src
   python run_experiment.py
   ```

2. ログ集計:
   ```bash
   cd analysis
   python aggregate_logs.py
   ```

3. 結果分析:
   ```bash
   python analyze_results.py
   ```

## 設定ファイルの構造

`config.yaml` で以下を定義:

- `experiment`: データセット、埋め込みモデル、λ値など
- `steps`: 各ステップのエージェント（id, model, prompt）

プロンプト内の `{input}` と `{categories}` が自動置換されます。

## 出力

- `logs/experiment_logs.csv`: 詳細ログ
- `analysis/aggregated_results.csv`: 集計結果
- `analysis/tradeoff_plot.png`: トレードオフ可視化

## 注意

- 大規模実験時はAPIコストに注意。
- ラベル解析は簡易実装のため、必要に応じて改善。