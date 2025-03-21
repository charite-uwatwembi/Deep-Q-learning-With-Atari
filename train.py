import gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecFrameStack

env_id = "AssaultNoFrameskip-v4"
env = make_atari_env(env_id, n_envs=1, seed=42)
env = VecFrameStack(env, n_stack=4)

policy = "CnnPolicy"

model = DQN(
    policy,
    env,
    learning_rate=1e-4,
    gamma=0.99,
    batch_size=32,
    buffer_size=100000,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.01,
    exploration_fraction=0.1,
    target_update_interval=1000,
    train_freq=4,
    gradient_steps=1,
    verbose=1,
    tensorboard_log="./dqn_tensorboard/"
)

logger = configure("./dqn_logs/", ["stdout", "csv", "tensorboard"])
model.set_logger(logger)
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path="./checkpoints/")

model.learn(
    total_timesteps=1_000_000,
    callback=checkpoint_callback,
    tb_log_name="dqn_assault"
)

model.save("dqn_assault_model")

env.close()