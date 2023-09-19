![FML.png](Images%2FFML.png)

## FinancialMachineLearning

### 1. Financial Machine Learning Library

금융 머신러닝 프로젝트를 위한 다양한 기능을 지원합니다

#### Bar sampling

- Standard Bar : Tick Bar, Volume Bar, Dollar Bar
- Imbalance Bar : Tick Imbalance Bar, Volume Imbalance Bar, Dollar Imbalance Bar
- Run Bar : Tick Run Bar, Volume Run Bar, Dollar Run Bar

#### Filter

- CUSUM Filtering
- ETF-trick Filtering
- Kalman Filtering
- Kalman Smoothering

#### Generator
Stochastic Process generator. 아래와 같은 확률 과정을 생성합니다

- Geometric Brownian Motion
- Ornstein Uhlenbeck Process
- Jump Diffusion Model
- Auto Regressive Process
- Microstructural Process Generator

#### Labeling
Machine Learning 학습을 위한 label분류를 진행합니다

- Triple Barrier Method
- Vertical Barrier
- Meta Labeling

#### Portfolio
portfolio 최적화를 위한 기능을 지원합니다. Machine Learning for Asset Manager(2019, de prado)

- Denoising
- Detoning

#### Regime Change
시장 국면 전환 사후 검정 모형입니다

- Chow type dickey fuller test
- CUSUM test
- Supremum augmented dickey fuller test

#### Feature Importance

- Mean Decrease Impurity (MDI)
- Mean Decrease Accuracy (MDA)
- Single Feature Importance (SFI)

#### Useful Features

- Concurrency
- Volatility
  - Auto Regressive Conditional Heteroscedasticity Model
  - General Auto Regressive Conditional Heteroscedasticity Model
- Discrete Entropy
- Approximate Entropy
- Fractionally Differentiated features
- Dynamic Z Score

#### Microstructure

- Roll Model
- Tick Rule
- Corwin Schultz
- Market Lambda : Kyle, Amihud, Hasbrouck
- Volume-Synchronized Probability of Informed Trading Model

#### Modeling

- Purged K Fold Cross Validation
- Embargo Lookback
- Hyper Parameter Tuning
- Log Uniform function

#### Backtesting

- Betting size
- Backtest Statistics

#### Technical Feature

- RSI
- MACD
- Moving Average