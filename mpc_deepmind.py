from dm_control import suite
import numpy as np
import matplotlib.pyplot as plt

# --- Load Ball-in-Cup environment ---
env = suite.load(domain_name="ball_in_cup", task_name="catch")
time_step = env.reset()

action_dim = env.action_spec().shape[0]

def random_shooting_mpc(state, horizon=10, num_samples=100):
    best_action_seq = None
    best_reward = -np.inf

    for _ in range(num_samples):
        action_seq = np.random.uniform(low=-1, high=1, size=(horizon, action_dim))
        reward = rollout_action_sequence(state, action_seq)
        
        if reward > best_reward:
            best_reward = reward
            best_action_seq = action_seq

    return best_action_seq[0]  # Return only the first action

def rollout_action_sequence(state, action_seq):
    # Reset to the current state
    env.physics.set_state(state)

    total_reward = 0
    for action in action_seq:
        time_step = env.step(action)
        total_reward += time_step.reward or 0

    return total_reward

# --- MPC Loop ---
num_steps = 200
reward_history = []

for step in range(num_steps):
    # Save current full physics state
    state = env.physics.get_state()

    # Solve MPC (choose best first action)
    action = random_shooting_mpc(state)

    # Step environment
    time_step = env.step(action)

    reward_history.append(time_step.reward)

# --- Plot reward over time ---
plt.plot(reward_history)
plt.xlabel("Timestep")
plt.ylabel("Reward")
plt.title("Ball-in-Cup with Random Shooting MPC")
plt.grid()
plt.show()
