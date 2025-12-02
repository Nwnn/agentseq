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

2. OpenRouter APIキーを設定:
   - `.env.example` をコピーして `.env` を作成し、APIキーを記載:
     ```bash
     cp .env.example .env
     ```
     `.env` ファイルを編集:
     ```
     OPENROUTER_API_KEY=your_api_key_here
     ```
     ※ `.env` は `.gitignore` で無視されるため、Gitにコミットされません。

   - または、環境変数に直接設定（上記方法も使用可能）:
     - Linux/Mac:
       ```bash
       export OPENROUTER_API_KEY="your_api_key_here"
       ```
     - Windows (コマンドプロンプト):
       ```cmd
       set OPENROUTER_API_KEY=your_api_key_here
       ```
     - Windows (PowerShell):
       ```powershell
       $env:OPENROUTER_API_KEY="your_api_key_here"
       ```

3. 設定ファイルを編集: `config.yaml`
   - データセット選択
   - ステップ定義（モデル、プロンプト）
   - パラメータ調整

## 使用方法

1. `.env`ファイルを作成し、OpenRouter APIキーを設定:
   ```bash
   cp .env.example .env
   # .envファイルを編集してAPIキーを入力
   ```

2. 実験実行:
   ```bash
   cd src
   python run_experiment.py
   ```

3. ログ集計:
   ```bash
   cd analysis
   python aggregate_logs.py
   ```

4. 結果分析:
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
- `logs/`: config.yamlの環境設定に基づいて生成されたオフライン実験環境として利用可能なデータセット
- `analysis/aggregated_results.csv`: 集計結果
- `analysis/tradeoff_plot.png`: トレードオフ可視化

## 注意

- 大規模実験時はAPIコストに注意。
- ラベル解析は簡易実装のため、必要に応じて改善。