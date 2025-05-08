from keras.models import Sequential, load_model
from keras.layers import Dense
import numpy as np
from game.player_class import Player
from game.poker_game import game
from keras.optimizers import Adam
import time
import tensorflow
import os


physical_devices = tensorflow.config.list_physical_devices("GPU")
if len(physical_devices) >= 1:
    tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)


def create_model():
    """
    Create a neural network model for the 7-player poker game
    The input shape is increased to accommodate for more player information
    """
    model = Sequential([
        Dense(64, activation='relu', input_shape=(7,)),  # Keeping same input shape as we use same observation format
        Dense(64, activation='relu'),
        Dense(41)])  # Output size remains 41 for the different betting actions
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model


def train_7_players(model):
    """
    The function teaches the model how to play 7-player poker
    :param model: model to train
    :return: saves the learned model as file .h5
    """
    model_target = create_model()
    model_target.set_weights(model.get_weights())

    path_save_model = r"models/7_players"
    
    # Create directory if it doesn't exist
    os.makedirs(path_save_model, exist_ok=True)

    # Learning parameters
    epsilon = 0.9
    epoch_start = 0  # change if retrain
    n_epoch = 15_000  # More epochs for 7 players as the game is more complex
    epsilon_decay = 0.9995  # Slower decay for more exploration in 7-player game
    min_epsilon = 0.1
    discount = 0.9

    # Create 7 players (1 deepAI agent + 6 regular AI opponents)
    Player('Alice', 1000, 'AI')
    Player('Bob', 1000, 'AI')
    Player('Charlie', 1000, 'AI')
    Player('David', 1000, 'AI')
    Player('Eve', 1000, 'AI')
    Player('Frank', 1000, 'AI')
    Player('Grace', 1000, 'deepAI')  # Our training agent

    n_win_games = 0
    game_result_list = []
    game_length_list = []
    rewards_per_epoch = []

    n_epoch_update_model = 10  # Update target network more frequently for 7-player scenario
    n_epoch_save_model = 300

    # Stats tracking
    avg_reward_window = []
    win_rate_window = []
    window_size = 100

    for epoch in range(epoch_start, n_epoch):
        start = time.time()

        # Reset all players' stacks at the beginning of each game
        for player in Player.player_list:
            player.next_game()  # Using next_game to fully reset player status

        game_instance = game()

        # Update target model
        if epoch % n_epoch_update_model == 0:
            model_target.set_weights(model.get_weights())

        states = []
        actions = []
        rewards = []
        next_states = []
        targets = []

        state, _, _, _ = next(game_instance)

        done = False
        while not done:
            # Get action probabilities from the model
            action_vector = model.predict(np.array(state).reshape(1, 7))

            targets.append(action_vector)
            action_vector = np.insert(action_vector, 0, epsilon)

            # Send action vector to game and get back the next state
            new_state, reward, done, action_used = game_instance.send(action_vector)

            states.append(state)
            actions.append(action_used)
            rewards.append(reward)
            next_states.append(new_state)

            state = new_state

        # Decay epsilon
        if epsilon > min_epsilon:
            epsilon *= epsilon_decay

        # Conversion lists to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        targets = np.squeeze(np.array(targets), axis=1)
        next_states = np.array(next_states)

        # Calculate target Q-values
        if next_states.shape[0] > 1:
            next_q_values = model_target.predict(next_states[:-1])
            max_next_q_values = np.max(next_q_values, axis=1)
            target_q_values = rewards[:-1] + discount * max_next_q_values

            target_q_values = np.append(target_q_values, rewards[-1])
        else:
            target_q_values = rewards

        for i in range(len(targets)):
            targets[i][actions[i]] = target_q_values[i]

        total_reward = sum(rewards)
        rewards_per_epoch.append(total_reward)

        # Track win/loss
        if total_reward > 0:
            game_result_list.append(1)
            n_win_games += 1
        else:
            game_result_list.append(0)

        win_ratio = n_win_games / (epoch + 1) * 100
        game_length_list.append(len(rewards))

        # Moving average tracking
        avg_reward_window.append(total_reward)
        win_rate_window.append(1 if total_reward > 0 else 0)
        
        if len(avg_reward_window) > window_size:
            avg_reward_window.pop(0)
            win_rate_window.pop(0)
            
        avg_reward = np.mean(avg_reward_window)
        recent_win_rate = np.mean(win_rate_window) * 100

        # Train model on batch
        model.train_on_batch(states, targets)

        print(f"Epoch {epoch}, Total Reward: {total_reward:.2f}, Game Length: {len(rewards)}")
        print(f"Win Rate: {win_ratio:.2f}%, Recent Win Rate: {recent_win_rate:.2f}%, Epsilon: {epsilon:.4f}")
        
        # Extra logging every 50 epochs
        if epoch % 50 == 0:
            print(f"Last {window_size} games - Avg Reward: {avg_reward:.2f}, Win Rate: {recent_win_rate:.2f}%")

        # Save model
        if epoch % n_epoch_save_model == 0 and epoch != 0:
            model_name = f"model_7players_{epoch}epoch-{int(time.time())}"
            model.save(f"{path_save_model}/{model_name}.h5")
            
            # Save training parameters
            with open(f"{path_save_model}/param_{model_name}.txt", 'w') as file:
                file.write("Seven-Player Poker Training\n")
                file.write(f"Number epoch: {epoch}\n")
                file.write(f"Epsilon: {epsilon}\n")
                file.write(f"Current win rate: {win_ratio:.2f}%\n")
                file.write(f"Recent win rate: {recent_win_rate:.2f}%\n")
                model.summary(print_fn=lambda x: file.write(x + '\n'))

            # Save statistics
            with open(f"{path_save_model}/stats_{model_name}.txt", 'w') as file:
                file.write(f"Win/lose streak (last 20): {game_result_list[-20:]}\n")
                file.write(f"Recent game lengths: {game_length_list[-20:]}\n")
                file.write(f"Recent rewards: {rewards_per_epoch[-20:]}\n")
                
            # Save numpy arrays for potential analysis
            np.save(f"{path_save_model}/winloss_{model_name}.npy", np.array(game_result_list))
            np.save(f"{path_save_model}/rewards_{model_name}.npy", np.array(rewards_per_epoch))

        stop = time.time()
        print(f'Epoch completed in {stop - start:.2f} seconds\n')

    # Final model save
    final_model_name = f"model_7players_final_{n_epoch}epoch-{int(time.time())}"
    model.save(f"{path_save_model}/{final_model_name}.h5")
    print(f"Training complete. Final model saved as {final_model_name}.h5")
    
    return model


def evaluate_model(model, n_games=100):
    """
    Evaluate the trained model against AI opponents
    :param model: model to evaluate
    :param n_games: number of games to play
    :return: win rate percentage
    """
    # Reset player list
    Player.player_list = []
    Player.player_list_chair = []
    Player._position = 0
    
    # Create 7 players (1 deepAI agent + 6 regular AI opponents)
    Player('Alice', 1000, 'AI')
    Player('Bob', 1000, 'AI')
    Player('Charlie', 1000, 'AI')
    Player('David', 1000, 'AI')
    Player('Eve', 1000, 'AI')
    Player('Frank', 1000, 'AI')
    Player('Grace', 1000, 'deepAI')  # Our trained agent
    
    wins = 0
    total_reward = 0
    
    print(f"Evaluating model over {n_games} games...")
    
    for i in range(n_games):
        # Reset all players' stacks
        for player in Player.player_list:
            player.next_game()
            
        game_instance = game()
        
        state, _, _, _ = next(game_instance)
        
        done = False
        game_reward = 0
        
        while not done:
            # Use the model to predict the best action (no exploration)
            action_vector = model.predict(np.array(state).reshape(1, 7))
            
            # No exploration during evaluation (epsilon = 0)
            action_vector = np.insert(action_vector, 0, 0)
            
            # Send action to game
            new_state, reward, done, _ = game_instance.send(action_vector)
            
            game_reward += reward
            state = new_state
            
        if game_reward > 0:
            wins += 1
            
        total_reward += game_reward
        
        if (i+1) % 10 == 0:
            print(f"Evaluated {i+1}/{n_games} games. Current win rate: {(wins/(i+1))*100:.2f}%")
    
    win_rate = (wins/n_games) * 100
    avg_reward = total_reward / n_games
    
    print(f"Evaluation complete.")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Average reward: {avg_reward:.4f}")
    
    return win_rate, avg_reward


if __name__ == '__main__':
    # Create or load a model
    if os.path.exists('models/7_players/pretrained_model.h5'):
        print("Loading existing model...")
        model = load_model('models/7_players/pretrained_model.h5')
    else:
        print("Creating new model...")
        model = create_model()
    
    # Train the model
    trained_model = train_7_players(model)
    
    # Evaluate the trained model
    print("\nEvaluating the trained model...")
    win_rate, avg_reward = evaluate_model(trained_model)