from __future__ import annotations

import pickle
import random
from collections import deque
from pathlib import Path

import numpy as np


class QLearningSnakeAgent:
    """
    Backward-compatible name, but implementation is Deep Q-Network (DQN).
    """

    def __init__(
        self,
        alpha: float = 0.0008,
        gamma: float = 0.98,
        epsilon: float = 1.0,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.9992,
        planning_steps: int = 1,
        replay_capacity: int = 120_000,
        replay_batch_size: int = 128,
        hidden_size_1: int = 128,
        hidden_size_2: int = 128,
        target_sync_every: int = 300,
        warmup_transitions: int = 1_000,
        seed: int | None = None,
    ) -> None:
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.planning_steps = max(1, int(planning_steps))
        self.replay_capacity = max(1_000, int(replay_capacity))
        self.replay_batch_size = max(16, int(replay_batch_size))
        self.hidden_size_1 = max(16, int(hidden_size_1))
        self.hidden_size_2 = max(16, int(hidden_size_2))
        self.target_sync_every = max(20, int(target_sync_every))
        self.warmup_transitions = max(self.replay_batch_size, int(warmup_transitions))
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.episodes_trained = 0
        self.total_gradient_steps = 0
        self.state_dim: int | None = None

        self.replay_buffer: deque[tuple[np.ndarray, int, float, np.ndarray, float]] = deque(
            maxlen=self.replay_capacity
        )

        self.online_params: dict[str, np.ndarray] = {}
        self.target_params: dict[str, np.ndarray] = {}

    def choose_action(self, state: tuple[int, ...], exploit_only: bool = False) -> int:
        state_vec = self._state_to_array(state)
        self._ensure_network(state_vec.shape[0])

        if not exploit_only and self.rng.random() < self.epsilon:
            return self.rng.choice([0, 1, 2])

        q_values = self._forward(self.online_params, state_vec[None, :])[0]
        return int(np.argmax(q_values))

    def update(
        self,
        state: tuple[int, ...],
        action: int,
        reward: float,
        next_state: tuple[int, ...],
        done: bool,
    ) -> None:
        state_vec = self._state_to_array(state)
        next_state_vec = self._state_to_array(next_state)
        self._ensure_network(state_vec.shape[0])

        done_float = 1.0 if done else 0.0
        self.replay_buffer.append((state_vec, int(action), float(reward), next_state_vec, done_float))

        # Repeat critical events so the model learns from mistakes/wins faster.
        if done and reward < 0:
            self.replay_buffer.append((state_vec, int(action), float(reward), next_state_vec, done_float))
            self.replay_buffer.append((state_vec, int(action), float(reward), next_state_vec, done_float))
        elif reward >= 10:
            self.replay_buffer.append((state_vec, int(action), float(reward), next_state_vec, done_float))

        for _ in range(self.planning_steps):
            self._replay_train_step()

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def model_size(self) -> int:
        if not self.online_params:
            return 0
        return int(sum(int(v.size) for v in self.online_params.values()))

    def reset_for_state_dim(self, expected_state_dim: int) -> None:
        self.state_dim = int(expected_state_dim)
        self.replay_buffer.clear()
        self.total_gradient_steps = 0
        self.online_params = self._init_network(self.state_dim)
        self.target_params = {k: v.copy() for k, v in self.online_params.items()}
        self.episodes_trained = 0
        self.epsilon = max(self.epsilon, 0.90)

    def state_dim_compatible(self, expected_state_dim: int) -> bool:
        if self.state_dim is None:
            return True
        return self.state_dim == int(expected_state_dim)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "agent_type": "dqn_v1",
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "planning_steps": self.planning_steps,
            "replay_capacity": self.replay_capacity,
            "replay_batch_size": self.replay_batch_size,
            "hidden_size_1": self.hidden_size_1,
            "hidden_size_2": self.hidden_size_2,
            "target_sync_every": self.target_sync_every,
            "warmup_transitions": self.warmup_transitions,
            "episodes_trained": self.episodes_trained,
            "total_gradient_steps": self.total_gradient_steps,
            "state_dim": self.state_dim,
            "online_params": self.online_params,
            "target_params": self.target_params,
        }
        with path.open("wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str | Path) -> "QLearningSnakeAgent":
        path = Path(path)
        with path.open("rb") as f:
            data = pickle.load(f)

        agent_type = data.get("agent_type")
        if agent_type != "dqn_v1":
            # Legacy q-table model: create fresh DQN but keep run metadata where possible.
            agent = cls(
                epsilon=float(data.get("epsilon", 1.0)),
                epsilon_min=float(data.get("epsilon_min", 0.02)),
                epsilon_decay=float(data.get("epsilon_decay", 0.9992)),
            )
            agent.episodes_trained = int(data.get("episodes_trained", 0))
            agent.epsilon = max(agent.epsilon, 0.90)
            return agent

        agent = cls(
            alpha=float(data.get("alpha", 0.0008)),
            gamma=float(data.get("gamma", 0.98)),
            epsilon=float(data.get("epsilon", 1.0)),
            epsilon_min=float(data.get("epsilon_min", 0.02)),
            epsilon_decay=float(data.get("epsilon_decay", 0.9992)),
            planning_steps=int(data.get("planning_steps", 1)),
            replay_capacity=int(data.get("replay_capacity", 120_000)),
            replay_batch_size=int(data.get("replay_batch_size", 128)),
            hidden_size_1=int(data.get("hidden_size_1", 128)),
            hidden_size_2=int(data.get("hidden_size_2", 128)),
            target_sync_every=int(data.get("target_sync_every", 300)),
            warmup_transitions=int(data.get("warmup_transitions", 1_000)),
        )
        agent.episodes_trained = int(data.get("episodes_trained", 0))
        agent.total_gradient_steps = int(data.get("total_gradient_steps", 0))
        state_dim = data.get("state_dim")
        agent.state_dim = int(state_dim) if state_dim is not None else None
        agent.online_params = {
            k: np.array(v, dtype=np.float32) for k, v in data.get("online_params", {}).items()
        }
        if data.get("target_params"):
            agent.target_params = {
                k: np.array(v, dtype=np.float32) for k, v in data.get("target_params", {}).items()
            }
        elif agent.online_params:
            agent.target_params = {k: v.copy() for k, v in agent.online_params.items()}
        return agent

    def _ensure_network(self, state_dim: int) -> None:
        if self.state_dim is None:
            self.state_dim = int(state_dim)
            self.online_params = self._init_network(self.state_dim)
            self.target_params = {k: v.copy() for k, v in self.online_params.items()}
            return
        if self.state_dim != int(state_dim):
            raise ValueError(
                f"State dimension mismatch: model={self.state_dim}, input={state_dim}. "
                "Reset memory or keep same state representation."
            )
        if not self.online_params:
            self.online_params = self._init_network(self.state_dim)
            self.target_params = {k: v.copy() for k, v in self.online_params.items()}

    def _init_network(self, state_dim: int) -> dict[str, np.ndarray]:
        scale = 0.05
        return {
            "W1": self.np_rng.normal(0.0, scale, size=(state_dim, self.hidden_size_1)).astype(np.float32),
            "b1": np.zeros((self.hidden_size_1,), dtype=np.float32),
            "W2": self.np_rng.normal(0.0, scale, size=(self.hidden_size_1, self.hidden_size_2)).astype(np.float32),
            "b2": np.zeros((self.hidden_size_2,), dtype=np.float32),
            "W3": self.np_rng.normal(0.0, scale, size=(self.hidden_size_2, 3)).astype(np.float32),
            "b3": np.zeros((3,), dtype=np.float32),
        }

    def _forward(
        self,
        params: dict[str, np.ndarray],
        x: np.ndarray,
    ) -> np.ndarray:
        z1 = x @ params["W1"] + params["b1"]
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ params["W2"] + params["b2"]
        a2 = np.maximum(z2, 0.0)
        q = a2 @ params["W3"] + params["b3"]
        return q

    def _forward_cache(
        self,
        params: dict[str, np.ndarray],
        x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        z1 = x @ params["W1"] + params["b1"]
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ params["W2"] + params["b2"]
        a2 = np.maximum(z2, 0.0)
        q = a2 @ params["W3"] + params["b3"]
        return z1, a1, z2, a2, q

    def _replay_train_step(self) -> None:
        if len(self.replay_buffer) < self.warmup_transitions:
            return
        batch_size = min(self.replay_batch_size, len(self.replay_buffer))
        indices = self.np_rng.integers(0, len(self.replay_buffer), size=batch_size)
        transitions = [self.replay_buffer[int(i)] for i in indices]

        states = np.stack([t[0] for t in transitions]).astype(np.float32)
        actions = np.array([t[1] for t in transitions], dtype=np.int64)
        rewards = np.array([t[2] for t in transitions], dtype=np.float32)
        next_states = np.stack([t[3] for t in transitions]).astype(np.float32)
        dones = np.array([t[4] for t in transitions], dtype=np.float32)

        z1, a1, z2, a2, q_pred = self._forward_cache(self.online_params, states)

        q_next_online = self._forward(self.online_params, next_states)
        next_actions = np.argmax(q_next_online, axis=1)
        q_next_target = self._forward(self.target_params, next_states)
        next_values = q_next_target[np.arange(batch_size), next_actions]
        targets = rewards + (1.0 - dones) * self.gamma * next_values

        pred_selected = q_pred[np.arange(batch_size), actions]
        td = pred_selected - targets
        # Huber derivative for robustness.
        abs_td = np.abs(td)
        grad_selected = np.where(abs_td <= 1.0, td, np.sign(td)).astype(np.float32) / batch_size

        d_q = np.zeros_like(q_pred, dtype=np.float32)
        d_q[np.arange(batch_size), actions] = grad_selected

        dW3 = a2.T @ d_q
        db3 = np.sum(d_q, axis=0)

        da2 = d_q @ self.online_params["W3"].T
        dz2 = da2 * (z2 > 0).astype(np.float32)
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        da1 = dz2 @ self.online_params["W2"].T
        dz1 = da1 * (z1 > 0).astype(np.float32)
        dW1 = states.T @ dz1
        db1 = np.sum(dz1, axis=0)

        self._apply_gradients(dW1, db1, dW2, db2, dW3, db3)

        self.total_gradient_steps += 1
        if self.total_gradient_steps % self.target_sync_every == 0:
            self.target_params = {k: v.copy() for k, v in self.online_params.items()}

    def _apply_gradients(
        self,
        dW1: np.ndarray,
        db1: np.ndarray,
        dW2: np.ndarray,
        db2: np.ndarray,
        dW3: np.ndarray,
        db3: np.ndarray,
    ) -> None:
        # Gradient clipping by global norm to keep training stable.
        grads = [dW1, db1, dW2, db2, dW3, db3]
        global_norm = float(np.sqrt(sum(float(np.sum(g * g)) for g in grads)))
        clip = 5.0
        if global_norm > clip and global_norm > 0:
            scale = clip / global_norm
            grads = [g * scale for g in grads]
            dW1, db1, dW2, db2, dW3, db3 = grads

        self.online_params["W1"] -= self.alpha * dW1
        self.online_params["b1"] -= self.alpha * db1
        self.online_params["W2"] -= self.alpha * dW2
        self.online_params["b2"] -= self.alpha * db2
        self.online_params["W3"] -= self.alpha * dW3
        self.online_params["b3"] -= self.alpha * db3

    @staticmethod
    def _state_to_array(state: tuple[int, ...]) -> np.ndarray:
        return np.asarray(state, dtype=np.float32)
