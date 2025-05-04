import numpy as np
from PIL import Image
from dm_control import suite
import dm_control2gym
import DMControl2Gymnasium
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
import math
# from dm_control.suite.ball_in_cup as bic
from dm_control.suite.ball_in_cup import BallInCup

# 1) stash the originals
_orig_init      = BallInCup.initialize_episode
_orig_after    = BallInCup.after_step

# 2) patch initialize to reset our “left‐cup” flag each episode
def _patched_init(self, physics):
    self._has_left_target = False
    return _orig_init(self, physics)

# 3) patch after_step to set the flag once the ball leaves the cup
def _patched_after(self, physics):
    # once time>0, if the ball is outside target, mark “left”
    if physics.data.time > 0.0 and not physics.in_target():
        self._has_left_target = True
        print("patched after function being called")
    return _orig_after(self, physics)

# 4) custom termination: only end when we’ve left *and* re‐entered
def _custom_termination(self, physics):
    value = getattr(self, '_has_left_target', False) and physics.in_target()
    print("Going to print value:", value)
    return value

def custom_termination(self, physics):
    # physics.data.time is the elapsed sim time in seconds.
    # At reset, physics.data.time == 0.0, so we skip termination then.
    if physics.data.time > 0.002 and physics.in_target():
        
        return True

    # after the very first step, we allow termination if caught
    print("Going to print false (outside if case)")
    return False

# Custom Reward function:
def custom_reward(self, physics):
    """
    Dense shaping reward for Kendama “ball in cup”:
      - +10 bonus on successful catch
      - Quadratic penalty on horizontal displacement (x)
      - Quadratic penalty on angular misalignment (θ)
      - Quadratic penalty on cup's horizontal velocity
      - Penalty on downward ball velocity
      - Penalty on ball height squared
      - Large penalty if the ball strays too far horizontally
    """
    # 1) Early bonus & termination
    if physics.in_target():
        return 10.0
    print(physics.data.time)
    # 2) Positions (x, y, z)
    ball_pos = physics.named.data.xpos['ball']
    cup_pos  = physics.named.data.xpos['cup']
    x_ball, z_ball = ball_pos[0], ball_pos[2]
    x_cup,  z_cup  = cup_pos[0],  cup_pos[2]

    # 3) Linear velocities (vx, vy, vz)
    v_ball = physics.named.data.cvel['ball']
    v_cup  = physics.named.data.cvel['cup']
    v_z_ball = v_ball[2]
    v_x_cup  = v_cup[0]

    # 4) Relative geometry
    dx, dz = x_ball - x_cup, z_ball - z_cup
    theta  = math.atan2(dz, dx) - math.pi/2

    # 5) Reward coefficients (tune these!)
    X_LIMIT              = 1.3
    A, B, C              = 0.01, 0.1, 0.001
    D                    = -0.01
    OUT_OF_BOUNDS_PENALTY= -5000.0
    VEL_DOWN_PENALTY     = -5000.0
    HEIGHT_PENALTY_SCALE = 5000.0
    SCALE                = 1e4

    # 6) Shaping & penalties
    shaping        = A * x_ball**2 + B * theta**2 + C * v_x_cup**2
    vel_penalty    = VEL_DOWN_PENALTY * np.sign(z_ball) * v_z_ball
    height_penalty = -HEIGHT_PENALTY_SCALE * z_ball**2

    reward = D * shaping + vel_penalty + height_penalty

    # 7) Out‑of‑bounds penalty
    if abs(x_ball) > X_LIMIT:
        reward += OUT_OF_BOUNDS_PENALTY

    # 8) Scale down
    return reward / SCALE
# ========================
# 1. Create the Environment
# ========================
# Wrap dm_control env as gym.Env using dm_control2gym
#env = DMControl2Gymnasium.make(domain_name="ball_in_cup", task_name="catch")
env = dm_control2gym.make(domain_name="ball_in_cup", task_name="catch")
# env.reset(seed=42)

def make_env(seed: int):
    def _init():
        # rebuild the patched Task internally
        env = dm_control2gym.make(domain_name="ball_in_cup", task_name="catch")
        env.seed(seed)
        return env
    return _init

# 5) apply the patches
BallInCup.initialize_episode = _patched_init
BallInCup.after_step        = _patched_after
BallInCup.get_termination   = _custom_termination
BallInCup.get_reward = custom_reward
#BallInCup.get_termination = custom_termination

# n_envs = 4
# env_fns = [make_env(i) for i in range(n_envs)]
# vec_env = SubprocVecEnv(env_fns)

# 3) wrap with running‐mean/std normalization
#vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
# ========================
# 2. Prepare TD3 Training
# ========================
version = 7
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

print("Starting TD3 training...")
model = TD3("MlpPolicy", env, learning_rate= 0.005, action_noise=action_noise, verbose=1)
eval_cb  = EvalCallback(
    env,
    best_model_save_path="./logs/",
    log_path="./logs/",
    eval_freq=5_000,       # evaluate every 5k steps
    deterministic=True,
    render=True
)
model.learn(total_timesteps=100000, callback= eval_cb, log_interval=1)
print("Training complete!")

# Save model
model.save("td3_ball_in_cup")

# Delete original env (not needed anymore)
env.close()

# ========================
# 3. Reload Trained Model & Visualize
# ========================
model = TD3.load("td3_ball_in_cup")

# Recreate environment with rendering enabled
env = dm_control2gym.make(domain_name="ball_in_cup", task_name="catch")

# Reset and collect frames using trained policy
obs = env.reset()
done = False
frames = []

print("Capturing rollout from trained policy...")

for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    # Render RGB frame
    frame = env.render(mode="rgb_array")
    img = Image.fromarray(frame)
    frames.append(img)

    if done:
        obs = env.reset()

env.close()

# ========================
# 4. Save as Animated GIF
# ========================
frames[0].save(
    "gifs/ball_in_cup_training_v{}.gif".format(version),
    save_all=True,
    append_images=frames[1:],
    duration=50,
    loop=0
)

print("GIF saved as ball_in_cup.gif!")
