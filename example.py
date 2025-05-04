import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise

# Simple transition storage (in place of ReplayBuffer for demo)
class RolloutStorage:
    def __init__(self):
        self.transitions = []

    def add(self, obs, next_obs, action, reward, done):
        self.transitions.append({
            "obs": obs,
            "next_obs": next_obs,
            "action": action,
            "reward": reward,
            "done": done
        })

    def print_summary(self, max_transitions=10):
        print(f"\nCollected {len(self.transitions)} transitions.")
        print("Sample transitions:")
        for i, t in enumerate(self.transitions[:max_transitions]):
            print(f"Step {i + 1}:")
            print(f"  Obs: {t['obs']}")
            print(f"  Action: {t['action']}")
            print(f"  Reward: {t['reward']}")
            print(f"  Done: {t['done']}")
            print()

# Custom rollout collection function
def collect_rollouts(env, model, n_steps=100, action_noise=None, learning_starts=0):
    storage = RolloutStorage()
    obs, _ = env.reset()
    for step in range(n_steps):
        if step > learning_starts:
            action, _ = model.predict(obs, deterministic=False)
            if action_noise is not None:
                action += action_noise()
        else:
            action = env.action_space.sample()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        storage.add(obs, next_obs, action, reward, done)

        if done:
            obs, _ = env.reset()
        else:
            obs = next_obs

    storage.print_summary()
    return storage


# ===== Main =====
env = gym.make("Pendulum-v1", render_mode="human")

n_actions = env.action_space.shape[-1]
print("Action space: ", env.action_space)
print("State space: ", env.observation_space)

action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

print("Training starts!")
model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)
model.learn(total_timesteps=10000, log_interval=10)
print("Training done!")

# Collect and print rollouts
print("\nCollecting rollouts after training...\n")
rollout_data = collect_rollouts(env, model, n_steps=20, action_noise=action_noise, learning_starts=0)

model.save("td3_pendulum")

model = TD3.load("td3_pendulum")

# Visualize the policy
env = gym.make("Pendulum-v1", render_mode="human")
obs, info = env.reset(seed = 42)
for _ in range(200):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()

env.close()
