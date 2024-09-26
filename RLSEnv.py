import gym
from gym import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    def __init__(self, data: pd.DataFrame, history_length: int = 180, commission: float = 0.0002):
        super(TradingEnv, self).__init__()
        
        self.data = data
        self.history_length = history_length
        self.initial_balance = 10000
        self.current_step = 0
        self.shares_held = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.pnl_log = []
        self.commission = commission
        self.total_commission = 0
        self.action_buffer = None  # Buffer to store the action to be applied in the next step
        self.pnl = 0
        self.pnl_exclude_commission = 0
        self.total_action_count = 0
        self.buy_count = 0
        self.sell_count = 0
        self.hold_count = 0

        self.action_space = spaces.Discrete(3)  # Hold, Buy, Sell
        self.observation_space = spaces.Box(low=0, high=1, shape=(history_length, len(data.columns) - 1), dtype=np.float32)

    def reset(self):
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.pnl_log = []
        self.pnl = 0
        self.pnl_exclude_commission = 0
        self.total_commission = 0
        self.current_step = 0
        self.total_action_count = 0
        self.buy_count = 0
        self.sell_count = 0
        self.hold_count = 0
        self.action_buffer = 0  # Reset the action buffer
        return self._next_observation()

    def _calculate_action_ratios(self):
        if self.total_action_count == 0:
            return 0, 0, 0, 0
        buy_ratio = self.buy_count / self.total_action_count
        sell_ratio = self.sell_count / self.total_action_count
        hold_ratio = self.hold_count / self.total_action_count
        pnl_ratio = self.pnl / self.initial_balance
        return buy_ratio, sell_ratio, hold_ratio, pnl_ratio


    def _next_observation(self):
        # Ensure the observation has the correct shape
        start = max(0, self.current_step - self.history_length + 1)
        obs = self.data.drop(columns=['close']).iloc[start:self.current_step + 1].values

        # Add realized PnL and action counts to the current step row
        buy_ratio, sell_ratio, hold_ratio, pnl_ratio = self._calculate_action_ratios()
        additional_features = np.array([
            buy_ratio,
            sell_ratio,
            hold_ratio,
            pnl_ratio
        ]).reshape(1, -1)

        # Append additional features to the last row of the observation
        obs = np.hstack((obs, np.zeros((obs.shape[0], additional_features.shape[1]))))
        obs[-1, -additional_features.shape[1]:] = additional_features

        if obs.shape[0] < self.history_length:
            # Pad the observation if it's shorter than the history length
            padding = np.zeros((self.history_length - obs.shape[0], obs.shape[1]))
            obs = np.vstack((padding, obs))
        return obs

    def step(self, action):
        if self.action_buffer is not None:
            self._take_action(self.action_buffer)  # Apply the buffered action

        self.action_buffer = action  # Store the current action in the buffer

        self.current_step += 1

        if self.current_step >= len(self.data) - 1:
            done = True
        else:
            done = False

        if action == 0:
            self.hold_count += 1
        elif action == 1:
            self.buy_count += 1
            self.total_action_count += 1
        elif action == 2:
            self.sell_count += 1
            self.total_action_count += 1

        # Calculate previous pnl
        previous_pnl = self.pnl

        # Use the 'close' column for reward calculation
        current_price = self.data.iloc[self.current_step]['close']
        self.pnl = self.balance + self.shares_held * current_price - self.initial_balance
        self.pnl_exclude_commission = self.pnl + self.total_commission
        
        reward = self.pnl - previous_pnl
        penalty = abs(reward) * 0.05 if action != 0 else 0
        reward -= penalty

        obs = self._next_observation()

        return obs, reward, done, {}

    def _take_action(self, action):
        current_price = self.data.iloc[self.current_step]['close']

        if action == 1:  # Buy
            if self.shares_held < 0:  # If in short position, buy back the shorted shares first
                shares_to_buy = min(abs(self.shares_held), self.balance // current_price)
                total_cost = shares_to_buy * current_price
                commission = shares_to_buy * self.commission
                self.total_commission += commission
                self.shares_held += shares_to_buy
                self.balance -= (total_cost + commission)
                # If there's remaining balance, buy more shares
                remaining_balance = self.balance
                shares_to_buy = remaining_balance // current_price
                if shares_to_buy > 0:
                    total_cost = shares_to_buy * current_price
                    commission = shares_to_buy * self.commission
                    self.total_commission += commission
                    self.shares_held += shares_to_buy
                    self.balance -= (total_cost + commission)
            else:  # Buy new shares
                max_investment = self.balance * 0.3
                shares_to_buy = max_investment // current_price
                if shares_to_buy > 0:
                    total_cost = shares_to_buy * current_price
                    commission = shares_to_buy * self.commission
                    self.total_commission += commission
                    self.shares_held += shares_to_buy
                    self.balance -= (total_cost + commission)

        elif action == 2:  # Sell
            if self.shares_held > 0:  # Sell held shares
                total_sale = self.shares_held * current_price
                commission = self.shares_held * self.commission
                self.total_commission += commission
                self.balance += (total_sale - commission)
                self.total_shares_sold += self.shares_held
                self.total_sales_value += total_sale
                self.shares_held = 0
            else:  # Short sell
                max_investment = self.balance * 0.3
                shares_to_sell = max_investment // current_price  # Limit short position to current balance
                total_sale = shares_to_sell * current_price
                commission = shares_to_sell * self.commission
                self.total_commission += commission
                self.balance += (total_sale - commission)
                self.shares_held -= shares_to_sell

    def render(self, mode='human', close=False):
        profit = self.balance + self.shares_held * self.data.iloc[self.current_step]['close'] - self.initial_balance
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held}')
        print(f'Total shares sold: {self.total_shares_sold}')
        print(f'Total sales value: {self.total_sales_value}')
        print(f'Profit: {profit}')