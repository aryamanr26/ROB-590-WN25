import gymnasium as gym
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# Linearized Pendulum Dynamics at upright (theta ≈ 0)
# State: [theta, theta_dot], Control: [torque]
A = np.array([[1, 0.05],
              [9.8 * 0.05, 1]])
B = np.array([[0],
              [0.05]])

# Cost weights
Q = np.diag([10, 1])  # Penalize angle more
R = np.array([[0.1]])  # Penalize torque

# MPC horizon
N = 20

def mpc_controller(x0):
    x = cp.Variable((2, N + 1))
    u = cp.Variable((1, N))

    cost = 0
    constraints = [x[:, 0] == x0]

    for t in range(N):
        cost += cp.quad_form(x[:, t], Q) + cp.quad_form(u[:, t], R)
        constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t],
                        cp.abs(u[:, t]) <= 2.0]  # torque limits

    cost += cp.quad_form(x[:, N], Q)  # terminal cost

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP)

    return u[:, 0].value if u[:, 0].value is not None else np.array([0.0])

# Normalize angle to [-pi, pi]
def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)

# --- Run MPC in Pendulum-v1 ---
env = gym.make("Pendulum-v1", render_mode="human")
obs, _ = env.reset()

theta_history = []
timestep = 0

for _ in range(200):
    cos_theta, sin_theta, theta_dot = obs
    theta = np.arctan2(sin_theta, cos_theta)  # Recover theta from cos/sin

    state = np.array([angle_normalize(theta), theta_dot])
    action = mpc_controller(state).reshape(-1)
    obs, reward, terminated, truncated, _ = env.step(action)
    
    theta_history.append(angle_normalize(theta))
    timestep += 1

    if terminated or truncated:
        break

env.close()

# --- Plot theta over time ---
plt.plot(theta_history)
plt.title("Pendulum Angle Over Time (MPC Control)")
plt.xlabel("Timestep")
plt.ylabel("Angle (rad)")
plt.grid()
plt.show()
