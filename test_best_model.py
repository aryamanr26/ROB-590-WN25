from PIL import Image
from stable_baselines3 import TD3
import dm_control2gym

# 1. Load your best model
model = TD3.load("logs/best_model.zip")  # or "td3_ball_in_cup.zip", wherever you saved it

# 2. Recreate the env
env = dm_control2gym.make(domain_name="ball_in_cup", task_name="catch")

# 2) Collect frames
obs    = env.reset()
done   = False
frames = []
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    # render as rgb array
    frame = env.render(mode="rgb_array")
    frames.append(Image.fromarray(frame))

env.close()

# 3) Save as animated GIF
frames[0].save(
    "gifs/best_policy1.gif",
    save_all=True,
    append_images=frames[1:],
    duration=50,  # ms per frame
    loop=0
)
print("✅ Saved best_policy.gif")