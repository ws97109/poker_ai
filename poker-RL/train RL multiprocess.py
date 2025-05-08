from keras.models import Sequential, load_model  # 引入 Sequential 類和 load_model 函數以建立和加載模型
from keras.layers import Dense  # 引入 Dense 層以構建神經網絡的全連接層
import numpy as np  # 引入 numpy 模組以進行數據處理
from keras.optimizers import Adam  # 引入 Adam 優化器以進行模型訓練
import tensorflow  # 引入 TensorFlow
import time  # 引入 time 模組以計算時間
from multiprocessing import Process, Queue  # 引入 Process 和 Queue 用於多進程操作

from game.player_class import Player  # 從 player_class 模組中引入 Player 類
from game.poker_game import game  # 從 poker_game 模組中引入 game 類

# 檢查是否有可用的 GPU 設備
physical_devices = tensorflow.config.list_physical_devices("GPU")
if len(physical_devices) >= 1:
    tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)  # 設置 GPU 記憶體增長模式


def create_model():
    # 創建並編譯深度學習模型
    model = Sequential([
        Dense(32, activation='relu', input_shape=(7,)),  # 第一層，全連接層，輸入為 7 個神經元
        Dense(32, activation='relu'),  # 第二層，全連接層
        Dense(41)  # 輸出層，有 41 個輸出神經元
    ])
    model.compile(optimizer=Adam(), loss='mse')  # 使用 Adam 優化器和均方誤差損失進行編譯
    return model


def one_game(epsilon, weight_model, weight_model_target, queue):
    """
    :param epsilon: epsilon-greedy 探索參數
    :param weight_model: 主模型的權重
    :param weight_model_target: 目標模型的權重
    :param queue: 多進程中的 Queue 對象
    :return: 將遊戲結果存入隊列中
    """

    discount = 0.9  # 折扣因子，用於計算未來回報的現值
    model = create_model()  # 創建主模型
    model.set_weights(weight_model)  # 設置主模型的權重

    model_target = create_model()  # 創建目標模型
    model_target.set_weights(weight_model_target)  # 設置目標模型的權重

    Player('Alice', 1000, 'AI')  # 創建 AI 玩家 Alice
    Player('Bob', 1000, 'deepAI')  # 創建深度學習 AI 玩家 Bob

    game_instance = game()  # 創建遊戲實例
    result = []  # 初始化結果列表
    states = []  # 初始化狀態列表
    actions = []  # 初始化動作列表
    rewards = []  # 初始化回報列表
    next_states = []  # 初始化下一狀態列表
    targets = []  # 初始化目標 Q 值列表

    state, _, _, _ = next(game_instance)  # 獲取遊戲的初始狀態

    done = False  # 遊戲是否結束標誌

    while not done:
        # 生成行動向量並插入 epsilon 值，以便進行 epsilon-greedy 探索
        action_vector = model.predict(np.array(state).reshape(1, 7))
        targets.append(action_vector)
        action_vector = np.insert(action_vector, 0, epsilon)

        # 將行動向量發送至遊戲，獲取新狀態、回報和所執行的行動
        new_state, reward, done, action_used = game_instance.send(action_vector)

        states.append(state)  # 將當前狀態存入列表
        actions.append(action_used)  # 將執行的行動存入列表
        rewards.append(reward)  # 將獲得的回報存入列表
        next_states.append(new_state)  # 將新狀態存入列表

        state = new_state  # 更新當前狀態

    # 將列表轉換為 numpy 陣列
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    targets = np.squeeze(np.array(targets), axis=1)
    next_states = np.array(next_states)

    # 計算目標 Q 值
    if next_states.shape[0] > 1:
        next_q_values = model_target.predict(next_states[:-1])  # 預測下一狀態的 Q 值
        max_next_q_values = np.max(next_q_values, axis=1)  # 獲取最大 Q 值
        target_q_values = rewards[:-1] + discount * max_next_q_values  # 計算目標 Q 值
        target_q_values = np.append(target_q_values, rewards[-1])  # 將最後一個回報添加至目標 Q 值
    else:
        target_q_values = rewards  # 如果只有一個狀態，目標 Q 值即為回報

    for i in range(len(targets)):
        targets[i][actions[i]] = target_q_values[i]  # 更新目標 Q 值

    total_reward = sum(rewards)  # 計算總回報

    result.append(states)  # 將狀態添加至結果列表
    result.append(targets)  # 將目標 Q 值添加至結果列表

    if total_reward > 0:
        result.append(1)  # 如果總回報為正，表示贏
    else:
        result.append(0)  # 否則表示輸

    queue.put(result)  # 將結果存入隊列


if __name__ == '__main__':

    path_save_model = r"models"  # 模型保存路徑

    # 學習參數
    epsilon = 0.1  # epsilon-greedy 的初始探索參數
    epsilon_decay = 0.9996  # epsilon 衰減率
    min_epsilon = 0.1  # epsilon 最小值

    epochs_info = []  # 儲存每一輪訓練的信息
    n_win_games = 0  # 計算贏得遊戲的次數
    game_result_list = []  # 儲存遊戲結果
    game_length_list = []  # 儲存每局遊戲的長度
    list_win_ratio = []  # 儲存贏的比例
    history = []  # 儲存模型訓練過程的歷史

    # 加載已訓練的模型
    model = load_model('C:/Users/ws971/OneDrive/桌面/Texas-Holdem-Poker-Reinforcement-Learning-master/models/model_9400epoch-1729480546.9694104.h5')
    # 或加載另一個模型
    # model = load_model('models/xxx.h5')

    model_target = create_model()  # 創建目標模型
    model_target.set_weights(model.get_weights())  # 設置目標模型的權重為主模型的權重

    weight_model = model.get_weights()  # 獲取主模型的權重
    weight_model_target = model_target.get_weights()  # 獲取目標模型的權重

    n_process = 5  # 定義進程數量
    processes = []  # 儲存所有進程
    queue_list = []  # 儲存所有進程的隊列

    # 創建進程和隊列
    for i in range(n_process):
        queue_list.append(Queue())
        processes.append(Process(target=one_game, args=(epsilon, weight_model, weight_model_target, queue_list[i],)))

    # 啟動進程
    for j in range(n_process):
        processes[j].start()

    total_iterations = 30_000  # 設置總迭代次數
    completed_iterations = 0  # 已完成的迭代次數，若訓練繼續可更改
    X = []  # 初始化輸入數據集
    y = []  # 初始化目標數據集

    run_train = False  # 是否進行訓練標誌
    update_model = False  # 是否更新模型標誌
    save_file = False  # 是否保存模型標誌

    n_epoch_train_model = 5  # 設置每隔多少次迭代進行訓練
    n_epoch_update_model = 20  # 設置每隔多少次迭代
