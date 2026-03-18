import numpy as np
import matplotlib.pyplot as plt

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

# compute emission log-probabilities for each point under each state
# state 1: N(0,1); state 2: N(0,5)
def log_emission(x, sigma):
    # log density of N(0, sigma^2)
    return -0.5 * ((x / sigma) ** 2 + np.log(2 * np.pi * sigma ** 2))

log_probs_state1 = [log_emission(x, 1) for x in data]
log_probs_state2 = [log_emission(x, 5) for x in data]

print("States:", states)
print("Data:", data)
print("\nFirst 10 emission log-probabilities under state 1:")
print(log_probs_state1[:10])
print("First 10 emission log-probabilities under state 2:")
print(log_probs_state2[:10])


# ---------- variational free energy ----------

def variational_free_energy(q, log_likelihoods, logA, log_pi=None):
    """Compute variational free energy (negative ELBO) for a set of beliefs.

    Parameters
    ----------
    q : array-like, shape (T, K)
        Belief distributions over K states at each of the T time points.  Each
        row should sum to 1.
    log_likelihoods : array-like, shape (T, K)
        Log probability of the observation at time t under each state k,
        i.e. log p(x_t | s_k).
    logA : array-like, shape (K, K)
        Log transition matrix, logA[j,k] = log p(s_{t+1}=k | s_t=j).
    log_pi : array-like, shape (K,), optional
        Log initial state probabilities, log_pi[k] = log p(s_1 = k).
        If None, assumes uniform.

    Returns
    -------
    float
        The variational free energy, defined as

            F = - E_q[log p(x|s)] - E_q[log p(s|s_prev)] - E_q[log p(s1)] + H[q]

        where the terms are the expected log-likelihood, expected log-transition,
        expected log-initial, and entropy of the approximate posterior q.
    """
    q = np.asarray(q)
    log_likelihoods = np.asarray(log_likelihoods)
    logA = np.asarray(logA)
    T, K = q.shape

    if log_pi is None:
        log_pi = np.log(np.ones(K) / K)  # uniform

    # expected log-likelihood term
    expected_log_like = np.sum(q * log_likelihoods)

    # expected log-transition term (for t=2 to T)
    expected_log_trans = 0.0
    for t in range(1, T):
        # q[t-1] @ logA gives sum_j q_{t-1}(j) log A_{j,k} for each k
        # then dot with q[t] gives sum_k [sum_j q_{t-1}(j) log A_{j,k}] q_t(k)
        expected_log_trans += np.dot(q[t-1] @ logA, q[t])

    # expected log-initial term
    expected_log_init = np.dot(q[0], log_pi)

    # entropy term (add small epsilon to avoid log(0))
    eps = 1e-12
    entropy = -np.sum(q * np.log(q + eps))

    # free energy as negative ELBO
    return -expected_log_like - expected_log_trans - expected_log_init + entropy


T = len(data)
K = 2
rng = np.random.RandomState(42)  # 42 as the seed
q = rng.rand(T, K)
q = q / q.sum(axis=1, keepdims=True)  # keep normality
log_likes = np.vstack([log_probs_state1, log_probs_state2]).T
# perform a single coordinate-ascent sweep over time points
# transition log-probs (log A)
logA = np.log(np.array([[0.9, 0.1], [0.1, 0.9]]))


def mean_field(qt, log_matrix):
    global log_likes
    for t in range(1, T-1):
        # compute unnormalized log q_t for each state k
        lnq = np.empty(K)
        # emission term:
        lnq += log_likes[t]
        # contribution from previous belief
        lnq += qt[t-1] @ log_matrix
        # contribution from next belief
        lnq += log_matrix.T @ qt[t+1]
        # normalize
        lnq -= np.max(lnq)  # for numerical stability
        qt[t] = np.exp(lnq)
        qt[t] /= qt[t].sum()
    return qt

# run multiple mean-field sweeps and track free energy
n_iter = 20
vfe_history = []
q_history = []

for it in range(n_iter):
    q = mean_field(q, logA)
    vfe_history.append(variational_free_energy(q, log_likes, logA))
    q_history.append(q.copy())

# print history and show simple plot
print("\nVariational free energy history:")
for it, val in enumerate(vfe_history, 1):
    print(f" iter {it:2d}: {val:.6f}")

plt.figure()
plt.plot(range(1, n_iter+1), vfe_history, '-o')
plt.xlabel('Iteration')
plt.ylabel('Variational free energy')
plt.title('Free energy over iterations')
plt.grid(True)
plt.tight_layout()
plt.savefig('free_energy_history.png', dpi=150)
plt.show()

# animate the beliefs over iterations
from matplotlib import animation
fig, ax = plt.subplots(figsize=(10,4))
line1, = ax.plot(q_history[0][:,0], label='P(state=1)')
line2, = ax.plot(q_history[0][:,1], label='P(state=2)')
ax.set_ylim(0,1)
ax.set_xlim(0, T-1)
ax.set_xlabel('Time index')
ax.set_ylabel('Belief')
ax.legend(loc='upper right')

def update(frame):
    line1.set_ydata(q_history[frame][:,0])
    line2.set_ydata(q_history[frame][:,1])
    ax.set_title(f'Iteration {frame+1}')
    return line1, line2

ani = animation.FuncAnimation(fig, update, frames=n_iter, blit=True)
try:
    ani.save('belief_evolution.gif', writer='pillow', fps=2)
except Exception:
    ani.save('belief_evolution.mp4', fps=2)

plt.show()




