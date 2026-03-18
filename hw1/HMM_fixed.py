import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp


def compute_state_log_likelihoods(observations, mus, sigmas):
    """计算每个时刻各状态下的对数似然"""
    n_states = len(mus)
    T = len(observations)
    log_likelihoods = np.zeros((T, n_states))
    for t in range(T):
        for i in range(n_states):
            log_likelihoods[t][i] = -0.5 * np.log(2 * np.pi * sigmas[i]**2) \
                                     - (observations[t] - mus[i])**2 / (2 * sigmas[i]**2)
    return log_likelihoods


def compute_variational_free_energy(observations, gamma, pi, A, mus, sigmas):
    """计算变分自由能 F = E_q[log q(S)] - E_q[log p(O,S|θ)]"""
    n_states = len(mus)
    T = len(observations)
    log_likelihoods = compute_state_log_likelihoods(observations, mus, sigmas)
    
    entropy = sum(
        gamma[t][i] * np.log(gamma[t][i] + 1e-10)
        for t in range(T) for i in range(n_states)
    )
    
    term_initial = sum(gamma[0][i] * np.log(pi[i] + 1e-10) for i in range(n_states))
    
    term_transition = sum(
        gamma[t-1][i] * gamma[t][j] * np.log(A[i][j] + 1e-10)
        for t in range(1, T) for i in range(n_states) for j in range(n_states)
    )
    
    term_emission = sum(
        gamma[t][i] * log_likelihoods[t][i]
        for t in range(T) for i in range(n_states)
    )
    
    F = entropy - (term_initial + term_transition + term_emission)
    return F


def update_gamma_variational(gamma, log_likelihoods, pi, A):
    """E步：使用局部平均场近似更新 gamma"""
    T, n_states = gamma.shape
    new_gamma = np.zeros_like(gamma)
    
    for t in range(T):
        log_msg = np.zeros(n_states)
        
        # 来自 t-1 的消息
        if t > 0:
            for j in range(n_states):
                log_msg[j] += np.sum(gamma[t-1] * np.log(A[:, j] + 1e-10))
        
        # 来自 t+1 的消息
        if t < T - 1:
            for i in range(n_states):
                log_msg[i] += np.sum(gamma[t+1] * np.log(A[i, :] + 1e-10))
        
        # 加上观测对数似然
        log_msg += log_likelihoods[t]
        
        # 边界处理 t=0: 使用初始概率 pi
        if t == 0:
            log_msg += np.log(pi + 1e-10)
        
        # softmax 归一化
        log_msg -= np.max(log_msg)
        new_gamma[t] = np.exp(log_msg)
        new_gamma[t] = new_gamma[t] / np.sum(new_gamma[t])
    
    return new_gamma


def update_parameters(gamma, observations):
    """M步：更新 HMM 参数"""
    T, n_states = gamma.shape
    
    # 更新初始概率
    pi = gamma[0] / np.sum(gamma[0])
    
    # 更新转移概率
    A = np.zeros((n_states, n_states))
    for i in range(n_states):
        for j in range(n_states):
            A[i, j] = np.sum(gamma[:-1, i] * gamma[1:, j])
    A = A / (A.sum(axis=1, keepdims=True) + 1e-10)
    
    # 更新发射概率（高斯均值和方差）
    mus = np.zeros(n_states)
    sigmas = np.zeros(n_states)
    for i in range(n_states):
        weight = gamma[:, i]
        weight_sum = np.sum(weight)
        mus[i] = np.sum(weight * observations) / (weight_sum + 1e-10)
        sigmas[i] = np.sqrt(
            np.sum(weight * (observations - mus[i])**2) / (weight_sum + 1e-10)
        )
    
    return pi, A, mus, sigmas


np.random.seed(42)

n_points = 200
states = []
current_state = 1

for i in range(n_points):
    if np.random.random() < 0.1:
        current_state = 2 if current_state == 1 else 1
    states.append(current_state)

data = []
for state in states:
    if state == 1:
        data.append(np.random.normal(0, 1))
    else:
        data.append(np.random.normal(0, 5))

# ===== HMM 参数初始化 =====
n_states = 2
T = len(data)

pi = np.array([1.0, 0.0])
A = np.array([[0.9, 0.1], 
              [0.1, 0.9]])
mus = np.array([0.0, 0.0])
sigmas = np.array([1.0, 5.0])

gamma = np.random.rand(T, n_states)
gamma = gamma / gamma.sum(axis=1, keepdims=True)

# ===== 固定参数，只对q(gamma)进行平均场近似迭代 =====
log_liks = compute_state_log_likelihoods(data, mus, sigmas)

F_history = []
gamma_history = []

for iteration in range(20):
    gamma = update_gamma_variational(gamma, log_liks, pi, A)
    F = compute_variational_free_energy(data, gamma, pi, A, mus, sigmas)
    
    F_history.append(F)
    gamma_history.append(gamma.copy())
    
    print(f"Iteration {iteration+1}: F = {F:.4f}")

print("State log-likelihoods shape:", log_liks.shape)
print("Variational Free Energy:", F)

plt.figure(figsize=(14, 6))
plt.plot(data, 'k-', linewidth=0.8, label='Data')

ax = plt.gca()
for i in range(n_points - 1):
    if states[i] == 1:
        ax.axvspan(i, i + 1, alpha=0.3, color='lightblue', linewidth=0)
    else:
        ax.axvspan(i, i + 1, alpha=0.3, color='lightyellow', linewidth=0)

plt.plot(data, 'k-', linewidth=0.8)
plt.xlabel('Time Index')
plt.ylabel('Value')
plt.title('Time Series with Two States (State1: std=1, State2: std=5)')

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='lightblue', alpha=0.5, label='State 1 (std=1)'),
    Patch(facecolor='lightyellow', alpha=0.5, label='State 2 (std=5)'),
    plt.Line2D([0], [0], color='black', linewidth=1, label='Data')
]
plt.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig('time_series_plot.png', dpi=150)
plt.show()
plt.close()

# ===== 可视化 =====
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 图1：变分自由能下降曲线
axes[0].plot(range(1, 21), F_history, 'b-o', linewidth=2, markersize=6)
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Variational Free Energy F')
axes[0].set_title('Variational Free Energy Decrease (Fixed Parameters)')
axes[0].grid(True, alpha=0.3)

# 图2：gamma变化过程（热力图）
gamma_array = np.array(gamma_history)
im = axes[1].imshow(gamma_array[:, :, 0], aspect='auto', cmap='Blues', 
                     extent=[0, T, 20, 1])
axes[1].set_xlabel('Time Index')
axes[1].set_ylabel('Iteration')
axes[1].set_title('Gamma (State 1 Probability) Evolution')
plt.colorbar(im, ax=axes[1], label='P(State 1)')

plt.tight_layout()
plt.savefig('variational_evolution.png', dpi=150)
plt.show()
plt.close()

print("States:", states)
print("Data:", data)
