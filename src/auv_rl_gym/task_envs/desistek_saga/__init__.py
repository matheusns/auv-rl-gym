from gymnasium.envs.registration import register

register(
     id="DesistekSagaAutoDocking-v0",
     entry_point="auv_rl_gym.task_envs.desistek_saga.auto_docking:AutoDocking",
     max_episode_steps=300,
)
