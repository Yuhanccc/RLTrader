import random
import numpy as np
import tensorflow as tf
from collections import deque
from tqdm import tqdm

class DQN(tf.keras.Model):
    def __init__(self, input_shape, output_dim):
        super(DQN, self).__init__()
        self.conv1 = tf.keras.layers.Conv1D(filters=32, kernel_size=8, strides=4, activation='relu', input_shape=input_shape)
        self.conv2 = tf.keras.layers.Conv1D(filters=64, kernel_size=4, strides=2, activation='relu')
        self.conv3 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.fc2 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.fc3 = tf.keras.layers.Dense(output_dim)
    
    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, state_dim, action_dim, replay_buffer_size=10000, batch_size=128, gamma=0.995, lr=0.001, target_update=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.target_update = target_update
        
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        
        self.target_net.set_weights(self.policy_net.get_weights())
        self.steps_done = 0
    
    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        else:
            state = np.expand_dims(state, axis=0).astype(np.float32)
            q_values = self.policy_net(state)
            return np.argmax(q_values.numpy())
    
    def store_experience(self, state, action, reward, next_state, done):
        state = np.array(state)
        next_state = np.array(next_state)
        action = np.array(action)
        reward = np.array(reward)
        done = np.array(done)
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def sample_experiences(self):
        experiences = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        return np.asarray(states), np.asarray(actions), np.asarray(rewards), np.asarray(next_states), np.asarray(dones)
    
    def update_policy(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.sample_experiences()
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int64)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            current_q_values = tf.reduce_sum(self.policy_net(states) * tf.one_hot(actions, self.action_dim), axis=1)
            next_q_values = tf.reduce_max(self.target_net(next_states), axis=1)
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
            loss = self.loss_fn(target_q_values, current_q_values)
        
        grads = tape.gradient(loss, self.policy_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy_net.trainable_variables))
        
        if self.steps_done % self.target_update == 0:
            self.target_net.set_weights(self.policy_net.get_weights())
    
    def train(self, envs, num_episodes, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        """
        Train the DQN agent.
        
        Parameters:
        - envs: List of environments, each initialized with a different dataframe.
        - num_episodes: Number of episodes to train.
        - epsilon_start: Initial value of epsilon for epsilon-greedy policy.
        - epsilon_end: Minimum value of epsilon.
        - epsilon_decay: Decay rate of epsilon per episode.
        """
        rewards_log = {}
        for episode in range(num_episodes):
            reward_log = []
            epsilon = epsilon_start
            for env in tqdm(envs, desc=f"Episode {episode + 1}/{num_episodes} Progress"):
                state = env.reset()
                done = False
                action = self.select_action(state, epsilon)  # Initialize the first action
                while not done:
                    next_state, reward, done, _ = env.step(action)  # Perform the stored action
                    next_action = self.select_action(next_state, epsilon)  # Decide the next action
                    self.store_experience(state, action, reward, next_state, done)
                    self.update_policy()
                    state = next_state
                    action = next_action  # Store the next action to be performed in the next step
                    self.steps_done += 1
                pnl = env.pnl - env.total_commission
                pnl_exclude_commission = env.pnl_exclude_commission
                reward_log.append({'pnl': pnl, 'pnl_exclude_commission': pnl_exclude_commission, 'commission': env.total_commission,
                                    'total_trading_actions': env.total_action_count, 'shares_traded': env.total_commission/env.commission})
                # Print PnL and total reward for each environment
                print(f"Environment Total Reward: {env.pnl}, Pure PnL: {env.pnl_exclude_commission}, Commission: {env.total_commission}")
                epsilon = max(epsilon_end, epsilon_decay * epsilon)
            rewards_log['Episode ' + str(episode + 1)] = reward_log

        return rewards_log