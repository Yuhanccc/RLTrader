# Reinforcement Learning Trading Agent

This project implements a reinforcement learning (RL) trading agent using Deep Q-Networks (DQN). The agent is trained to trade stocks based on historical minute-level data. The project includes data preprocessing, feature engineering, environment setup, agent training, and performance visualization.

## Project Structure

- `TrainAgent.ipynb`: Jupyter notebook for data preprocessing, feature calculation, normalization, and agent training.
- `HighFreqFactors.py`: Python module containing functions for calculating various financial factors.
- `RLSEnv.py`: Python module defining the trading environment using OpenAI's Gym.
- `DQNAgent.py`: Python module implementing the DQN agent.
- `plot.ipynb`: Jupyter notebook for visualizing the performance of the trained agent.

## Usage

### Data Preprocessing and Training

1. Open `TrainAgent.ipynb` in Jupyter Notebook or Jupyter Lab.
2. Follow the steps in the notebook to preprocess the data, calculate features, normalize them, and train the DQN agent.

### Visualizing Performance

1. After training, open `plot.ipynb` in Jupyter Notebook or Jupyter Lab.
2. Follow the steps in the notebook to load the training logs and visualize the performance of the trained agent.

### Example Performance
Historical 1-minute frequency data from 2000 to 2018 is used to train an agent for high-frequency intraday trading with the goal of making a profit. A total of 300 trading days are used for training the agent. The agent appears to start making steady profits around the 100th day. However, it becomes highly unstable (more specifically, goes MADDDD!!!) after 250 trading days, so I have only plotted the data for the first 240 days.

For anyone attempting to replicate the results, please note that I created 25 high-frequency factors to construct observations at each step, which go beyond the factors provided in `HighFreqFactors.py` (after all, I can't do all the work for you, right?). It is recommended to conduct some research on high-frequency factors before cloning this project.

The data used is provided as a link (Google Drive) in `.data/link.txt`. Just take it anyway as it costed me 5 bucks.

![output](https://github.com/user-attachments/assets/1daad57f-6136-48bc-8169-55430d1a31cf)


## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [OpenAI Gym](https://gym.openai.com/)
- [TensorFlow](https://www.tensorflow.org/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)

---

Feel free to customize this `README.md` to better fit your project and add any additional information you find necessary.
