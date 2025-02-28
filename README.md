 # 🌟 欢迎访问我的项目：[DeepTrafficSense]

## 项目简介
Latest Research From JLU Transportation College
A comprehensive RTTE framework that takes into account the heterogeneity and temporal variability of road segment conditions. It integrates a Heterogeneous Dynamic Graph (HDG) and a Trajectory Motion Transformer (TMT) to model macro traffic dynamics and micro vehicle motion patterns. 

---

## 功能特点

- **1**: The HDG captures the temporal evolution of traffic states across road segments using asynchronous memory updates and adaptive feature aggregation, effectively addressing challenges such as irregular trajectory sampling and stale memory states.
- **2**: The TMT, designed for variable-length multivariate sequences, employs self- and cross-attention mechanisms to fuse endogenous (e.g., speed) and exogenous (e.g., acceleration) motion features.
- **3**: A type-aware gating mechanism dynamically adjusts the contributions of the HDG and TMT, enhancing prediction accuracy for both intersections and regular road segments. 


## 安装与使用

### Usage
1. pip install -r requirements.txt
2. The datasets can be obtained [Baidu Cloud]([https://pan.baidu.com/s/4muPzAg4](https://pan.baidu.com/s/1Z3_YPdF4YiwMSD2CC_26qg?pwd=hqwe)).
3. Train and evaluate the model. We provide all the above tasks under the folder ./DTS_train.py. 
