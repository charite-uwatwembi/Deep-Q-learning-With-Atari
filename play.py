import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.policies import GreedyQPolicy

# Load the trained model
model = DQN.load(
    "dqn_assault_model",
    env=gym.make("ALE/Assault-v5", render_mode="human"),
    policy=GreedyQPolicy
)

# Environment setup
env = model.get_env()
obs = env.reset()

# Run 5 episodes
for episode in range(5):
    done = False
    obs = env.reset()
    total_reward = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
        
    print(f"Episode {episode + 1} | Total Reward: {total_reward}")

# Close environment
env.close()