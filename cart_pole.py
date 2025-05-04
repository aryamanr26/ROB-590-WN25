from dm_control import suite
import numpy as np
from PIL import Image

# Load one task:
env = suite.load(domain_name="cartpole", task_name="swingup")

# # Iterate over a task set:
# for domain_name, task_name in suite.BENCHMARKING:
#   env = suite.load(domain_name, task_name, visualize_reward=True)

# Step through an episode and print out reward, discount and observation.
action_spec = env.action_spec()
time_step = env.reset()
print(action_spec, time_step)
# List to store rendered frames
frames = []

# Run for a number of steps and collect frames
for step in range(100):
    # Sample a random action within the valid action space
    action = np.random.uniform(action_spec.minimum, action_spec.maximum, size=action_spec.shape)

    # Take a step in the environment
    time_step = env.step(action)

    # Render the frame as an RGB array
    frame = env.physics.render(height=480, width=640, camera_id=0)

    # Convert to PIL Image and store
    img = Image.fromarray(frame)
    frames.append(img)

    # Reset if the episode ends
    if time_step.last():
        time_step = env.reset()

# Save all frames as an animated GIF
frames[0].save(
    "gifs/cart_pole.gif",
    save_all=True,
    append_images=frames[1:],
    duration=50,   # milliseconds per frame
    loop=0         # infinite loop
)

print("Saved simulation to cartpole.gif ✅")