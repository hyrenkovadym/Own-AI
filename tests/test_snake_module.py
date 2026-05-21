from __future__ import annotations

from pathlib import Path

import pytest

from modules.snake.agent import QLearningSnakeAgent
from modules.snake.entry import SnakeModule
from modules.snake.env import SnakeEnv


def test_snake_env_rejects_invalid_action() -> None:
    env = SnakeEnv(width=6, height=6, seed=1)
    env.reset()
    with pytest.raises(ValueError):
        env.step(99)


def test_snake_env_wall_collision() -> None:
    env = SnakeEnv(width=6, height=6, seed=1, goal_score=3)
    env.reset()
    env.snake = [(5, 2), (4, 2), (3, 2)]
    env.dir_idx = 1  # moving right
    env.food = (0, 0)

    _, reward, done, info = env.step(0)
    assert done is True
    assert reward == -10.0
    assert info.done_reason == "wall"


def test_snake_env_goal_completion_reward() -> None:
    env = SnakeEnv(width=6, height=6, seed=2, goal_score=1)
    env.reset()
    env.snake = [(2, 2), (1, 2), (0, 2)]
    env.dir_idx = 1  # moving right
    env.food = (3, 2)
    env.score = 0
    env.steps = 0
    env.steps_since_food = 0
    env.recent_heads.clear()
    for part in env.snake:
        env.recent_heads.append(part)

    _, reward, done, info = env.step(0)
    assert done is True
    assert info.done_reason == "goal"
    assert info.score == 1
    assert reward >= 40.0


def test_snake_agent_save_and_load(tmp_path: Path) -> None:
    path = tmp_path / "snake_agent.pkl"
    agent = QLearningSnakeAgent(seed=4)
    state = (0,) * 21
    _ = agent.choose_action(state)
    agent.save(path)

    restored = QLearningSnakeAgent.load(path)
    assert restored.state_dim == 21
    assert restored.model_size() > 0


def test_snake_module_train_and_play(tmp_path: Path) -> None:
    module = SnakeModule()
    model_path = tmp_path / "snake_q.pkl"

    train_stats = module.train(
        episodes=3,
        max_steps=30,
        width=6,
        height=6,
        model_path=str(model_path),
        seed=3,
        log_every=0,
        resume=False,
        goal_score=3,
    )
    assert train_stats.episodes > 0
    assert model_path.exists()
    assert train_stats.model_path.endswith("snake_q.pkl")

    eval_stats = module.play(
        episodes=2,
        max_steps=30,
        width=6,
        height=6,
        model_path=str(model_path),
        seed=5,
        goal_score=3,
    )
    assert eval_stats.episodes == 2
