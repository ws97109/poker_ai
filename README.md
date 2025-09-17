# Poker AI 專案

這是一個使用深度強化學習訓練的撲克牌AI專案，包含了AI模型訓練和網頁遊戲介面。
可以透過以下網站使用
poker-ai-one.vercel.app
## 專案結構

```
poker-ai-main/
├── poker-RL/                    # 強化學習訓練模組
│   ├── game/                    # 遊戲邏輯
│   │   ├── poker_game.py       # 主要遊戲流程
│   │   ├── player_class.py     # 玩家類別定義
│   │   ├── poker_round.py      # 單局遊戲邏輯
│   │   ├── poker_score.py      # 牌型評分
│   │   ├── bot.py              # 機器人玩家
│   │   ├── deepAI.py           # AI玩家邏輯
│   │   ├── auction.py          # 下注邏輯
│   │   └── split_pot.py        # 分池邏輯
│   ├── models/                  # 訓練好的模型
│   │   └── 7_players/          # 7玩家模型
│   ├── models_new/             # 新訓練模型
│   ├── html/                   # 網頁版遊戲
│   ├── test/                   # 測試檔案
│   ├── train.py                # 主要訓練腳本
│   ├── train_RL.py            # 強化學習訓練
│   ├── train_RL_7.py          # 7玩家訓練
│   ├── requirements.txt        # Python依賴
│   └── setting.py              # 遊戲設定
├── poker game/                 # 網頁遊戲版本
│   ├── game_final.html        # 主要遊戲介面
│   ├── model.js               # JavaScript模型
│   ├── poker-ai.js            # AI邏輯
│   ├── images/                # 撲克牌圖片
│   └── finish_model/          # 完成的模型檔案
└── finish_model/              # 最終訓練模型
```

## 系統需求

- Python 3.8+
- TensorFlow 2.13.1
- Keras 2.13.1
- NumPy 1.24.3
- Conda (推薦)

## 安裝步驟

### 1. 建立虛擬環境

```bash
# 使用 conda 建立虛擬環境
conda create -n poker python=3.8
conda activate poker
```

### 2. 安裝依賴套件

```bash
# 進入 poker-RL 目錄
cd poker-RL

# 安裝所需套件
pip install -r requirements.txt
```

### 3. 驗證安裝

```python
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
```

## 使用方法

### 訓練 AI 模型

```bash
# 啟動虛擬環境
conda activate poker

# 進入訓練目錄
cd poker-RL

# 開始訓練 (預設10,000個epoch)
python train.py
```

訓練參數說明：
- `n_epoch = 10_000`: 訓練輪數
- `epsilon = 0.9`: 探索率（初始值）
- `epsilon_decay = 0.999`: 探索率衰減
- `min_epsilon = 0.1`: 最小探索率
- `discount = 0.9`: 折扣因子

### 訓練不同玩家數量的模型

```bash
# 訓練7玩家模型
python train_RL_7.py

# 使用多進程訓練（更快）
python "train RL multiprocess.py"
```

### 執行網頁遊戲

1. 開啟 `poker game/game_final.html`
2. 在瀏覽器中載入頁面
3. 開始與AI對戰

### 測試模型

```bash
cd poker-RL/test

# 測試AI算法
python "test alg ai.py"

# 測試預測功能
python "test predict.py"

# 測試概率計算
python "test probability.py"
```

## 模型檔案

- **訓練中的模型**: `poker-RL/models_new/`
- **7玩家模型**: `poker-RL/models/7_players/`
- **完成的模型**: `finish_model/`
- **網頁版模型**: `poker game/model.json` 和 `model.js`

每個模型包含：
- `.h5` 檔案：模型權重
- `param_*.txt`：模型參數
- `stats_*.txt`：訓練統計
- `.npy` 檔案：獎勵和勝負記錄

## 遊戲設定

在 `poker-RL/setting.py` 中可以調整：

```python
BB = 50          # 大盲注
SB = 25          # 小盲注
show_game = False # 是否顯示遊戲過程
```

## 故障排除

### 1. GPU 設定
如果有 NVIDIA GPU，程式會自動偵測並使用：
```python
physical_devices = tensorflow.config.list_physical_devices("GPU")
if len(physical_devices) >= 1:
    tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)
```

### 2. 記憶體不足
- 減少 batch size
- 使用較小的神經網路
- 關閉不必要的程式

### 3. 訓練速度慢
- 使用 GPU 加速
- 使用多進程版本：`python "train RL multiprocess.py"`
- 調整 `n_epoch_save_model` 參數

### 4. 模型載入錯誤
確保模型檔案路徑正確：
```python
# 載入現有模型
model = load_model('models/xxx.h5')
```

## 進階使用

### 自訂訓練參數
修改 `train.py` 中的參數：
- 調整網路架構（Dense層數和節點數）
- 修改學習率和優化器
- 改變獎勵函數

### 分析訓練結果
訓練完成後會產生統計檔案：
- 勝率統計
- 遊戲長度分析
- 獎勵變化趨勢

### 網頁版本自訂
- 修改 `poker-ai.js` 調整AI行為
- 更新 `game_final.html` 改變介面
- 替換 `images/` 中的撲克牌圖片

## 技術特色

- **深度Q學習 (DQN)**: 使用experience replay和target network
- **多玩家支援**: 支援2-7個玩家的德州撲克
- **網頁介面**: 提供友善的HTML/JavaScript遊戲界面
- **模型可視化**: 支援訓練過程的統計分析
- **GPU加速**: 自動偵測和使用GPU進行訓練

## 聯絡資訊

如有問題或建議，歡迎提出issue或pull request。
