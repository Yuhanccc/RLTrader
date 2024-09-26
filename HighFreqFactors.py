import pandas as pd
import numpy as np


class Factors:

    # Calculate the price
    @staticmethod
    def calculate_median_price(self, data: pd.DataFrame):
        data.loc[:, 'median_price'] = data[['open', 'high', 'low', 'close']].median(axis=1)

    # Calculate the return
    @staticmethod
    def calculate_return(data: pd.DataFrame, price_col: str):
        data.loc[:, 'return'] = np.log(data[price_col] / data[price_col].shift(1))

    # Calculate realized volatility
    @staticmethod
    def calculate_realized_volatility(data: pd.DataFrame):
        data.loc[:, 'squared_return'] = data['return'] ** 2
        data.loc[:, 'realized_volatility'] = data['squared_return'].expanding().sum()

    # Calculate realized skewness with expanding window
    @staticmethod
    def calculate_realized_skewness_expanding(data: pd.DataFrame, output_col: str):
        N = np.arange(1, len(data) + 1)
        data.loc[:, 'squared_return'] = data['return'] ** 2
        data.loc[:, 'cubed_return'] = data['return'] ** 3
        
        # Calculate cumulative sums
        cum_sum_squared = data['squared_return'].expanding().sum()
        cum_sum_cubed = data['cubed_return'].expanding().sum()
        
        # Calculate realized skewness
        data.loc[:, output_col] = (np.sqrt(N) * cum_sum_cubed) / (cum_sum_squared ** (3/2))

    # Calculate realized kurtosis with expanding window
    @staticmethod
    def calculate_realized_kurtosis_expanding(data: pd.DataFrame, output_col: str):
        N = np.arange(1, len(data) + 1)
        data.loc[:, 'squared_return'] = data['return'] ** 2
        data.loc[:, 'fourth_return'] = data['return'] ** 4
        
        # Calculate cumulative sums
        cum_sum_squared = data['squared_return'].expanding().sum()
        cum_sum_fourth = data['fourth_return'].expanding().sum()

        # Calculate realized kurtosis
        data.loc[:, output_col] = (N * cum_sum_fourth) / (cum_sum_squared ** 2)

    # Calculate momentum
    @staticmethod
    def calculate_momentum(data: pd.DataFrame, output_col: str):
        data.loc[:, output_col] = data['return'].expanding().sum()
