from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam

import numpy as np
from game.player_class import Player  
from game.poker_game import game  
from keras.optimizers import Adam
import time
import tensorflow

# 檢查是否有可用的 GPU
physical_devices = tensorflow.config.list_physical_devices("GPU")
if len(physical_devices) >= 1:
    tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)

# 建立模型
def create_model():
    model = Sequential([
        Dense(32, activation='relu', input_shape=(7,)), 
        Dense(32, activation='relu'),  
        Dense(41)]) 
    model.compile(optimizer=Adam(), loss='mse')
    return model

# 函數教學模型如何玩撲克，並將訓練好的模型保存為 .h5 格式
def train(model):
    """
    The function teaches the model how to play heads up poker
    :param model: model to train
    :return: saves the learned model as file .h5
    """
    model_target = create_model()  
    model_target.set_weights(model.get_weights()) 

    path_save_model = r"models_new" 

  
    epsilon = 0.9  
    epoch_start = 0  
    n_epoch = 10_000 
    epsilon_decay = 0.999 
    min_epsilon = 0.1  
    discount = 0.9  

    # 建立玩家
    Player('Alice', 1000, 'AI')
    Player('Bob', 1000, 'deepAI')
    Player('Charlie', 1000, 'AI')
    Player('Dave', 1000, 'AI')
    Player('Eve', 1000, 'AI')
    Player('Frank', 1000, 'AI')
    Player('Grace', 1000, 'AI')
    n_win_games = 0  
    game_result_list = []  
    game_length_list = [] 

    n_epoch_update_model = 20  
    n_epoch_save_model = 300 

    for epoch in range(epoch_start, n_epoch):

        start = time.time()  # 開始計時

        # 重置所有玩家的籌碼為 1000
        for player in Player.player_list:
            player.stack = 1000

        game_instance = game()  

        # 更新目標模型的權重
        if epoch % n_epoch_update_model == 0:
            model_target.set_weights(model.get_weights())

        # 初始化狀態、動作、回報和目標等列表
        states = []
        actions = []
        rewards = []
        next_states = []
        targets = []

        # 獲取遊戲的初始狀態
        state, _, _, _ = next(game_instance)

        done = False  
        while not done:
            # 使用模型預測當前狀態下的動作向量
            action_vector = model.predict(np.array(state).reshape(1, 7))

      
            targets.append(action_vector)
            action_vector = np.insert(action_vector, 0, epsilon)

            # action_used 表示上一回合實際採取的動作
            new_state, reward, done, action_used = game_instance.send(action_vector)

            # 將當前狀態、動作、回報和新狀態加入列表中
            states.append(state)
            actions.append(action_used)
            rewards.append(reward)
            next_states.append(new_state)

            # 更新狀態為新的狀態
            state = new_state

        # 衰減探索率
        if epsilon > min_epsilon:
            epsilon *= epsilon_decay

        # 將列表轉換為 numpy 數組
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        targets = np.squeeze(np.array(targets), axis=1)
        next_states = np.array(next_states)

        # 計算目標 Q 值
        if next_states.shape[0] > 1:
            next_q_values = model_target.predict(next_states[:-1])
            max_next_q_values = np.max(next_q_values, axis=1)
            target_q_values = rewards[:-1] + discount * max_next_q_values

            target_q_values = np.append(target_q_values, rewards[-1])
        else:
            target_q_values = rewards

        # 更新 targets 中每個動作的 Q 值
        for i in range(len(targets)):
            targets[i][actions[i]] = target_q_values[i]

        total_reward = sum(rewards)  

        # 更新遊戲結果列表和勝場數量
        if total_reward > 0:
            game_result_list.append(1)
            n_win_games += 1
        else:
            game_result_list.append(0)

        win_ratio = n_win_games / (epoch + 1) * 100  

        game_length_list.append(len(rewards)) 

        # 在批次上訓練模型
        model.train_on_batch(states, targets)

        # 打印當前回合的資訊
        print("epoch {}, rewards: {}, win games {} %".format(epoch, np.sum(rewards), win_ratio))
        
        # 保存模型
        if epoch % n_epoch_save_model == 0 and epoch != 0:
            model_name = "model_{}epoch-{}".format(epoch, time.time())
            model.save(str(path_save_model) + r"/{}.h5".format(model_name))
            # 創建模型信息的文件
            with open(path_save_model + r"/param_{}.txt".format(model_name), 'w') as file:
                file.write("Architecture model: \n")
                file.write("Number epoch: {}\n".format(epoch))
                file.write("Epsilon: {}\n".format(epsilon))
                model.summary(print_fn=lambda x: file.write(x + '\n'))

            # 保存統計信息
            with open(path_save_model + r"/stats_{}.txt".format(model_name), 'w') as file:
                file.write("Win/lose: {}\n".format(game_result_list))
                file.write("Game length: \n{}".format(game_length_list))

        stop = time.time() 
        # 打印回合數、動作數量和花費的時間
        print('number action: {}, time: {}'.format(len(rewards), stop - start))


if __name__ == '__main__':
    model = create_model()
    # model = load_model('models/xxx.h5')
    train(model) 
