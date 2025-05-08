// poker-ai.js

class PokerAI {
    constructor() {
        this.model = null;
        this.initialized = false;
        console.log('PokerAI 建立完成');
    }

    // 初始化 AI，載入模型
    async initialize() {
        console.log('開始初始化 AI...');
        try {
            // 讀取模型檔案
            const response = await fetch('model.json');
            const modelData = await response.json();
            
            // 從模型資料中提取權重
            const weights = this.extractWeights(modelData);
            
            // 建立模型
            this.model = await this.createModel(weights);
            this.initialized = true;
            console.log('AI 模型載入成功');
        } catch (error) {
            console.error('AI 模型載入失敗:', error);
            console.error('錯誤詳情:', {
                message: error.message,
                stack: error.stack
            });
        }
    }

    // 從模型資料中提取權重
    extractWeights(modelData) {
        const weights = {
            dense: {
                kernel: modelData.model_weights.dense.dense["kernel:0"],
                bias: modelData.model_weights.dense.dense["bias:0"]
            },
            dense_1: {
                kernel: modelData.model_weights.dense_1.dense_1["kernel:0"],
                bias: modelData.model_weights.dense_1.dense_1["bias:0"]
            },
            dense_2: {
                kernel: modelData.model_weights.dense_2.dense_2["kernel:0"],
                bias: modelData.model_weights.dense_2.dense_2["bias:0"]
            }
        };
        return weights;
    }

    // 建立模型架構
    async createModel(weights) {
        const model = tf.sequential();
        
        // 第一層 Dense (7 -> 32)
        model.add(tf.layers.dense({
            units: 32,
            activation: 'relu',
            inputShape: [7],
            weights: [
                tf.tensor2d(weights.dense.kernel, [7, 32]),
                tf.tensor1d(weights.dense.bias)
            ]
        }));
        
        // 第二層 Dense (32 -> 32)
        model.add(tf.layers.dense({
            units: 32,
            activation: 'relu',
            weights: [
                tf.tensor2d(weights.dense_1.kernel, [32, 32]),
                tf.tensor1d(weights.dense_1.bias)
            ]
        }));
        
        // 輸出層 (32 -> 41)
        model.add(tf.layers.dense({
            units: 41,
            activation: 'linear',
            weights: [
                tf.tensor2d(weights.dense_2.kernel, [32, 41]),
                tf.tensor1d(weights.dense_2.bias)
            ]
        }));

        return model;
    }

    // 將遊戲狀態轉換為模型輸入格式
    preprocessState(gameState) {
        // 將輸入標準化到合適的範圍
        const input = [
            gameState.potSize / 1000,         // 底池大小標準化
            gameState.playerStack / 1000,     // 玩家籌碼標準化
            gameState.botStack / 1000,        // AI 籌碼標準化
            gameState.currentBet / 100,       // 當前下注標準化
            gameState.position,               // 位置 (0: 小盲, 1: 大盲)
            gameState.stage,                  // 遊戲階段 (0: preflop, 1: flop, 2: turn, 3: river)
            gameState.handStrength           // 手牌強度 (0-1)
        ];
        
        return tf.tensor2d([input]);
    }

    // AI 決策
    async makeDecision(gameState) {
        if (!this.initialized) {
            console.log('AI 尚未初始化');
            throw new Error('AI 尚未初始化');
        }

        console.log('目前遊戲狀態:', gameState);

        // 預處理輸入數據
        const input = this.preprocessState(gameState);
        
        // 使用模型進行預測
        const prediction = await this.model.predict(input).array();
        console.log('AI 決策結果:', prediction[0]);
        
        // 解釋預測結果
        const decision = this.interpretAction(prediction[0]);
        console.log('AI 最終決定:', decision);
        
        return decision;
    }

    // 解釋模型輸出的行動
    interpretAction(actionVector) {
        const maxValue = Math.max(...actionVector);
        const actionIndex = actionVector.indexOf(maxValue);
        
        // 基本行動
        if (actionIndex === 0) return 'fold';    // 棄牌
        if (actionIndex === 1) return 'check';   // 過牌
        if (actionIndex === 2) return 'call';    // 跟注
        
        // 加注行動（剩餘的 index 對應不同的加注量）
        const raiseAmount = (actionIndex - 2) * 50;
        return {
            action: 'raise',
            amount: raiseAmount
        };
    }
}

// 確保全域只有一個 AI 實例
if (!window.pokerAI) {
    window.pokerAI = new PokerAI();
}