from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .agent import QLearningSnakeAgent
from .env import SnakeEnv


@dataclass
class TrainStats:
    episodes: int
    avg_score_last_100: float
    best_score: int
    q_states: int
    total_episodes: int
    wins: int
    goal_score_requested: int | None
    goal_score_effective: int
    best_fill_percent: float
    model_path: str
    resumed_from_model: bool
    epsilon_start: float
    epsilon_end: float


@dataclass
class EvalStats:
    episodes: int
    avg_score: float
    best_score: int
    wins: int
    goal_score_requested: int | None
    goal_score_effective: int
    best_fill_percent: float
    model_path: str


@dataclass
class RolloutFrame:
    snake: list[tuple[int, int]]
    food: tuple[int, int]
    score: int
    step: int
    done: bool


@dataclass
class LearnFrame:
    episode: int
    snake: list[tuple[int, int]]
    food: tuple[int, int]
    score: int
    step: int
    done: bool
    action: int | None
    reward: float
    epsilon: float
    done_reason: str | None


@dataclass
class ModelInfo:
    exists: bool
    model_path: str
    q_states: int
    total_episodes: int


class SnakeModule:
    name = "snake"

    def train(
        self,
        episodes: int = 3000,
        max_steps: int = 250,
        width: int = 10,
        height: int = 10,
        model_path: str = "models/snake_q.pkl",
        seed: int | None = None,
        log_every: int = 250,
        progress_callback: Callable[[str], None] | None = None,
        resume: bool = True,
        goal_score: int | None = 100,
        stop_predicate: Callable[[], bool] | None = None,
    ) -> TrainStats:
        env = SnakeEnv(width=width, height=height, seed=seed, goal_score=goal_score)
        agent, resumed = self._build_train_agent(model_path=model_path, seed=seed, resume=resume)
        scores: list[int] = []
        best_score = 0
        wins = 0
        episodes_done = 0
        epsilon_start = agent.epsilon
        step_limit = self._resolve_step_limit(max_steps=max_steps, env=env)

        def emit(message: str) -> None:
            if progress_callback is not None:
                progress_callback(message)
            else:
                print(message)

        expected_state_dim = len(env.reset())
        if self._ensure_state_compatibility(agent, expected_state_dim=expected_state_dim):
            resumed = False
            emit(
                f"[train] model state format changed -> memory reset "
                f"(expected_state_dim={expected_state_dim})"
            )

        if resumed:
            emit(f"[train] resuming from model: {model_path} (epsilon={agent.epsilon:.4f})")
        else:
            emit(f"[train] starting fresh model (epsilon={agent.epsilon:.4f})")

        for ep in range(1, episodes + 1):
            if stop_predicate is not None and stop_predicate():
                emit(f"[train] stopped before episode {ep}.")
                break

            state = env.reset()
            done = False
            steps = 0
            info = None

            while not done and steps < step_limit:
                if stop_predicate is not None and stop_predicate():
                    done = True
                    info = None
                    break
                action = agent.choose_action(state)
                next_state, reward, done, info = env.step(action)
                agent.update(state, action, reward, next_state, done)
                state = next_state
                steps += 1

            if info is None:
                emit(f"[train] stopped during episode {ep}.")
                break

            scores.append(info.score)
            best_score = max(best_score, info.score)
            if info.done_reason == "goal":
                wins += 1
            episodes_done += 1
            agent.episodes_trained += 1
            agent.decay_epsilon()

            if log_every > 0 and ep % log_every == 0:
                recent = scores[-100:] if len(scores) >= 100 else scores
                avg = sum(recent) / len(recent)
                emit(
                    f"[train] episode={ep}/{episodes} "
                    f"avg_score_last_{len(recent)}={avg:.2f} "
                    f"best={best_score} epsilon={agent.epsilon:.4f}"
                )

        agent.save(model_path)
        recent = scores[-100:] if len(scores) >= 100 else scores
        avg_last = sum(recent) / len(recent) if recent else 0.0
        best_fill_percent = self._calc_fill_percent(best_score, width=width, height=height)
        return TrainStats(
            episodes=episodes_done,
            avg_score_last_100=avg_last,
            best_score=best_score,
            q_states=agent.model_size(),
            total_episodes=agent.episodes_trained,
            wins=wins,
            goal_score_requested=goal_score,
            goal_score_effective=env.effective_goal_score,
            best_fill_percent=best_fill_percent,
            model_path=str(Path(model_path)),
            resumed_from_model=resumed,
            epsilon_start=epsilon_start,
            epsilon_end=agent.epsilon,
        )

    def train_live(
        self,
        episodes: int = 300,
        max_steps: int = 250,
        width: int = 10,
        height: int = 10,
        model_path: str = "models/snake_q.pkl",
        seed: int | None = None,
        frame_callback: Callable[[LearnFrame], None] | None = None,
        episode_callback: Callable[[str], None] | None = None,
        stop_predicate: Callable[[], bool] | None = None,
        resume: bool = True,
        goal_score: int | None = 100,
    ) -> TrainStats:
        env = SnakeEnv(width=width, height=height, seed=seed, goal_score=goal_score)
        agent, resumed = self._build_train_agent(model_path=model_path, seed=seed, resume=resume)
        scores: list[int] = []
        best_score = 0
        episodes_done = 0
        wins = 0
        epsilon_start = agent.epsilon
        step_limit = self._resolve_step_limit(max_steps=max_steps, env=env)

        def emit_episode(message: str) -> None:
            if episode_callback is not None:
                episode_callback(message)
            else:
                print(message)

        expected_state_dim = len(env.reset())
        if self._ensure_state_compatibility(agent, expected_state_dim=expected_state_dim):
            resumed = False
            emit_episode(
                f"[live] model state format changed -> memory reset "
                f"(expected_state_dim={expected_state_dim})"
            )

        if resumed:
            emit_episode(f"[live] resuming from model: {model_path} (epsilon={agent.epsilon:.4f})")
        else:
            emit_episode(f"[live] starting fresh model (epsilon={agent.epsilon:.4f})")

        for ep in range(1, episodes + 1):
            if stop_predicate is not None and stop_predicate():
                emit_episode(f"[live] stopped before episode {ep}.")
                break

            state = env.reset()
            done = False
            step_count = 0
            info = None

            if frame_callback is not None:
                frame_callback(
                    LearnFrame(
                        episode=ep,
                        snake=list(env.snake),
                        food=env.food,
                        score=env.score,
                        step=0,
                        done=False,
                        action=None,
                        reward=0.0,
                        epsilon=agent.epsilon,
                        done_reason=None,
                    )
                )

            while not done and step_count < step_limit:
                if stop_predicate is not None and stop_predicate():
                    done = True
                    info = None
                    break

                action = agent.choose_action(state)
                next_state, reward, done, info = env.step(action)
                agent.update(state, action, reward, next_state, done)
                state = next_state
                step_count += 1

                if frame_callback is not None:
                    frame_callback(
                        LearnFrame(
                            episode=ep,
                            snake=list(env.snake),
                            food=env.food,
                            score=env.score,
                            step=step_count,
                            done=done,
                            action=action,
                            reward=reward,
                            epsilon=agent.epsilon,
                            done_reason=info.done_reason if info is not None else None,
                        )
                    )

            if stop_predicate is not None and stop_predicate():
                emit_episode(f"[live] stopped during episode {ep}.")
                break

            if info is None:
                continue

            scores.append(info.score)
            episodes_done += 1
            best_score = max(best_score, info.score)
            if info.done_reason == "goal":
                wins += 1
            agent.episodes_trained += 1
            agent.decay_epsilon()
            recent = scores[-100:] if len(scores) >= 100 else scores
            avg = sum(recent) / len(recent)
            emit_episode(
                (
                    f"[live] ep={ep}/{episodes} score={info.score} "
                    f"steps={info.steps} done={info.done_reason or 'ok'} "
                    f"avg_last_{len(recent)}={avg:.2f} best={best_score} "
                    f"epsilon={agent.epsilon:.4f}"
                )
            )

        agent.save(model_path)
        if scores:
            recent = scores[-100:] if len(scores) >= 100 else scores
            avg_last = sum(recent) / len(recent)
        else:
            avg_last = 0.0
        return TrainStats(
            episodes=episodes_done,
            avg_score_last_100=avg_last,
            best_score=best_score,
            q_states=agent.model_size(),
            total_episodes=agent.episodes_trained,
            wins=wins,
            goal_score_requested=goal_score,
            goal_score_effective=env.effective_goal_score,
            best_fill_percent=self._calc_fill_percent(best_score, width=width, height=height),
            model_path=str(Path(model_path)),
            resumed_from_model=resumed,
            epsilon_start=epsilon_start,
            epsilon_end=agent.epsilon,
        )

    def play(
        self,
        episodes: int = 10,
        max_steps: int = 250,
        width: int = 10,
        height: int = 10,
        model_path: str = "models/snake_q.pkl",
        seed: int | None = None,
        goal_score: int | None = 100,
    ) -> EvalStats:
        env = SnakeEnv(width=width, height=height, seed=seed, goal_score=goal_score)
        agent = QLearningSnakeAgent.load(model_path)
        agent.epsilon = 0.0
        step_limit = self._resolve_step_limit(max_steps=max_steps, env=env)

        scores: list[int] = []
        best_score = 0
        wins = 0

        for _ in range(episodes):
            state = env.reset()
            done = False
            steps = 0
            info = None

            while not done and steps < step_limit:
                action = agent.choose_action(state, exploit_only=True)
                state, _, done, info = env.step(action)
                steps += 1

            if info is None:
                raise RuntimeError("Episode did not produce final info.")
            scores.append(info.score)
            best_score = max(best_score, info.score)
            if info.done_reason == "goal":
                wins += 1

        avg_score = sum(scores) / len(scores)
        return EvalStats(
            episodes=episodes,
            avg_score=avg_score,
            best_score=best_score,
            wins=wins,
            goal_score_requested=goal_score,
            goal_score_effective=env.effective_goal_score,
            best_fill_percent=self._calc_fill_percent(best_score, width=width, height=height),
            model_path=str(Path(model_path)),
        )

    def rollout(
        self,
        max_steps: int = 250,
        width: int = 10,
        height: int = 10,
        model_path: str = "models/snake_q.pkl",
        seed: int | None = None,
        goal_score: int | None = 100,
    ) -> list[RolloutFrame]:
        env = SnakeEnv(width=width, height=height, seed=seed, goal_score=goal_score)
        agent = QLearningSnakeAgent.load(model_path)
        agent.epsilon = 0.0
        step_limit = self._resolve_step_limit(max_steps=max_steps, env=env)

        state = env.reset()
        frames: list[RolloutFrame] = [
            RolloutFrame(
                snake=list(env.snake),
                food=env.food,
                score=env.score,
                step=0,
                done=False,
            )
        ]

        done = False
        step = 0
        while not done and step < step_limit:
            action = agent.choose_action(state, exploit_only=True)
            state, _, done, _ = env.step(action)
            step += 1
            frames.append(
                RolloutFrame(
                    snake=list(env.snake),
                    food=env.food,
                    score=env.score,
                    step=step,
                    done=done,
                )
            )

        return frames

    def get_model_info(self, model_path: str = "models/snake_q.pkl") -> ModelInfo:
        path = Path(model_path)
        if not path.exists():
            return ModelInfo(
                exists=False,
                model_path=str(path),
                q_states=0,
                total_episodes=0,
            )

        agent = QLearningSnakeAgent.load(path)
        return ModelInfo(
            exists=True,
            model_path=str(path),
            q_states=agent.model_size(),
            total_episodes=agent.episodes_trained,
        )

    @staticmethod
    def _build_train_agent(model_path: str, seed: int | None, resume: bool) -> tuple[QLearningSnakeAgent, bool]:
        path = Path(model_path)
        if resume and path.exists():
            agent = QLearningSnakeAgent.load(path)
            if seed is not None:
                agent.rng.seed(seed)
            resumed = not (agent.state_dim is None and agent.model_size() == 0)
            return agent, resumed
        return QLearningSnakeAgent(seed=seed), False

    @staticmethod
    def _resolve_step_limit(max_steps: int, env: SnakeEnv) -> int:
        # max_steps <= 0 means "soft unlimited": rely mostly on game-over/starvation.
        if max_steps <= 0:
            return max(1000, env.max_starve_steps * 10)
        return max_steps

    @staticmethod
    def _calc_fill_percent(score: int, width: int, height: int) -> float:
        max_possible = width * height - 3
        if max_possible <= 0:
            return 0.0
        return 100.0 * score / max_possible

    @staticmethod
    def _ensure_state_compatibility(agent: QLearningSnakeAgent, expected_state_dim: int) -> bool:
        if agent.state_dim is None:
            return False
        if agent.state_dim_compatible(expected_state_dim):
            return False
        agent.reset_for_state_dim(expected_state_dim)
        return True


def create_module() -> SnakeModule:
    return SnakeModule()
