![FML.png](Images%2FFML.png)

## FinancialMachineLearning

### 1. Financial Machine Learning Library

금융 머신러닝 프로젝트를 위한 다양한 기능을 지원합니다

#### Bar sampling
![bar.png](Images%2Fbar.png)

- Standard Bar : Tick Bar, Volume Bar, Dollar Bar
- Imbalance Bar : Tick Imbalance Bar, Volume Imbalance Bar, Dollar Imbalance Bar
- Run Bar : Tick Run Bar, Volume Run Bar, Dollar Run Bar

#### Filter

- CUSUM Filtering
- ETF-trick Filtering
- Kalman Filtering
- Kalman Smoothering
- Denoising
![denoising.png](Images%2Fdenoising.png)

#### Generator

Stochastic Process generator. 아래와 같은 확률 과정을 생성합니다

- Geometric Brownian Motion
![GBM.png](Images%2FGBM.png)
- Ornstein Uhlenbeck Process
![OUprocess.png](Images%2FOUprocess.png)
- Jump Diffusion Model
![jdprocess.png](Images%2Fjdprocess.png)
- Auto Regressive Process
![ar.png](Images%2Far.png)
- Microstructural Process Generator
- de Prado Synthetic Process
![prado.png](Images%2Fprado.png)

#### Labeling
Machine Learning 학습을 위한 label분류를 진행합니다

- Triple Barrier Method
- Vertical Barrier
- Meta Labeling

#### Portfolio
portfolio 최적화를 위한 기능을 지원합니다. Machine Learning for Asset Manager(2019, de prado)

- Historical Risk Parity
![hrp.png](Images%2Fhrp.png)
- Critical Line Algorithms
- Mean Variance Optimization
- Black Litterman Optimization

#### Regime Change
시장 국면 전환 사후 검정 모형입니다

- Chow type dickey fuller test
- CUSUM test
- Supremum augmented dickey fuller test

#### Feature Importance

Machine Learning 특성 분석을 위한 Feature Importance 계산 기능을 지원합니다 (AFML Chapter 8)

- Mean Decrease Impurity (MDI)
- Mean Decrease Accuracy (MDA)
![feature_importance.png](Images%2Ffeature_importance.png)
- Single Feature Importance (SFI)

#### Useful Features
![ffd.png](Images%2Fffd.png)

- Concurrency
- Volatility
  - Auto Regressive Conditional Heteroscedasticity Model
  - General Auto Regressive Conditional Heteroscedasticity Model
- Discrete Entropy
![etp_vpin.png](Images%2Fetp_vpin.png)
![vpin_etp100.png](Images%2Fvpin_etp100.png)
- Approximate Entropy
- Fractionally Differentiated features
- Dynamic Z Score

#### Microstructure

시장미시구조적 특성 (Lopez de Prado, 2018)
- Roll Model
![roll_model.png](Images%2Froll_model.png)
- Tick Rule
- Corwin Schultz
- Market Lambda : Kyle, Amihud, Hasbrouck
- Becker Parkinson Volatility
![bpvol.png](Images%2Fbpvol.png)
- Volume-Synchronized Probability of Informed Trading Model
![vip.png](Images%2Fvip.png)

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

### 2. Example Notes

library 주요 기능을 사용하는 jupyter notebook 예제를 제공합니다