from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass


# Clockwise order is important for relative turns.
DIRECTIONS = [
    (0, -1),  # up
    (1, 0),   # right
    (0, 1),   # down
    (-1, 0),  # left
]


@dataclass
class StepInfo:
    score: int
    ate_food: bool
    steps: int
    done_reason: str | None = None


class SnakeEnv:
    """
    Headless Snake environment for simple RL experiments.
    Actions (relative): 0=straight, 1=right turn, 2=left turn
    """

    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        seed: int | None = None,
        goal_score: int | None = 100,
        loop_penalty: float = 0.20,
    ) -> None:
        self.width = width
        self.height = height
        self.rng = random.Random(seed)
        self.max_starve_steps = width * height * 2
        self.goal_score = goal_score
        self.loop_penalty = loop_penalty

        self.snake: list[tuple[int, int]] = []
        self.food: tuple[int, int] = (0, 0)
        self.dir_idx = 1
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.max_possible_score = 0
        self.effective_goal_score = 0
        self.recent_heads: deque[tuple[int, int]] = deque(maxlen=max(12, width + height))

    def reset(self) -> tuple[int, ...]:
        cx = self.width // 2
        cy = self.height // 2
        self.dir_idx = self.rng.choice([0, 1, 2, 3])
        dx, dy = DIRECTIONS[self.dir_idx]

        self.snake = [(cx, cy), (cx - dx, cy - dy), (cx - 2 * dx, cy - 2 * dy)]
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.max_possible_score = self.width * self.height - len(self.snake)
        if self.goal_score is None:
            self.effective_goal_score = self.max_possible_score
        else:
            self.effective_goal_score = max(1, min(self.goal_score, self.max_possible_score))
        self.recent_heads.clear()
        for part in self.snake:
            self.recent_heads.append(part)
        self._spawn_food()
        return self.get_state()

    def step(self, action: int) -> tuple[tuple[int, ...], float, bool, StepInfo]:
        if action not in (0, 1, 2):
            raise ValueError("Action must be 0 (straight), 1 (right), or 2 (left).")

        old_food = self.food
        old_head = self.snake[0]
        old_dist = self._manhattan(old_head, old_food)

        self._apply_action(action)
        dx, dy = DIRECTIONS[self.dir_idx]
        new_head = (old_head[0] + dx, old_head[1] + dy)
        self.steps += 1
        will_grow = new_head == self.food

        collision_reason = self._collision_reason(new_head, will_grow)
        if collision_reason is not None:
            state = self.get_state()
            return state, -10.0, True, StepInfo(
                score=self.score,
                ate_food=False,
                steps=self.steps,
                done_reason=collision_reason,
            )
        if self.steps_since_food > self.max_starve_steps:
            state = self.get_state()
            return state, -10.0, True, StepInfo(
                score=self.score,
                ate_food=False,
                steps=self.steps,
                done_reason="starvation",
            )

        self.snake.insert(0, new_head)
        ate_food = will_grow

        if ate_food:
            self.score += 1
            self.steps_since_food = 0
            reward = 15.0
            if self.score >= self.effective_goal_score:
                self.recent_heads.append(new_head)
                state = self.get_state()
                return state, reward + 25.0, True, StepInfo(
                    score=self.score,
                    ate_food=True,
                    steps=self.steps,
                    done_reason="goal",
                )
            self._spawn_food()
        else:
            self.snake.pop()
            self.steps_since_food += 1
            new_dist = self._manhattan(new_head, old_food)
            reward = -0.05
            reward += 0.20 if new_dist < old_dist else -0.10
            # Penalize short loops so the agent stops repeating "safe but useless" cycles.
            if new_head in self.recent_heads:
                reward -= self.loop_penalty

        self.recent_heads.append(new_head)
        state = self.get_state()
        return state, reward, False, StepInfo(
            score=self.score,
            ate_food=ate_food,
            steps=self.steps,
            done_reason=None,
        )

    def get_state(self) -> tuple[int, ...]:
        head = self.snake[0]

        dir_up = self.dir_idx == 0
        dir_right = self.dir_idx == 1
        dir_down = self.dir_idx == 2
        dir_left = self.dir_idx == 3

        right_dir = (self.dir_idx + 1) % 4
        left_dir = (self.dir_idx - 1) % 4
        straight_point = self._next_point(head, self.dir_idx)
        right_point = self._next_point(head, right_dir)
        left_point = self._next_point(head, left_dir)

        straight_point_2 = self._next_point(straight_point, self.dir_idx)
        right_point_2 = self._next_point(right_point, right_dir)
        left_point_2 = self._next_point(left_point, left_dir)

        food_dx = self.food[0] - head[0]
        food_dy = self.food[1] - head[1]
        food_left = False
        food_right = False
        food_up = False
        food_down = False
        food_ahead = False
        food_back = False
        food_right_rel = False
        food_left_rel = False

        if food_dx < 0:
            food_left = True
        elif food_dx > 0:
            food_right = True
        if food_dy < 0:
            food_up = True
        elif food_dy > 0:
            food_down = True

        if dir_up:
            food_ahead = food_dy < 0
            food_back = food_dy > 0
            food_right_rel = food_dx > 0
            food_left_rel = food_dx < 0
        elif dir_right:
            food_ahead = food_dx > 0
            food_back = food_dx < 0
            food_right_rel = food_dy > 0
            food_left_rel = food_dy < 0
        elif dir_down:
            food_ahead = food_dy > 0
            food_back = food_dy < 0
            food_right_rel = food_dx < 0
            food_left_rel = food_dx > 0
        else:  # dir_left
            food_ahead = food_dx < 0
            food_back = food_dx > 0
            food_right_rel = food_dy < 0
            food_left_rel = food_dy > 0

        state = (
            int(self._is_collision(straight_point)),
            int(self._is_collision(right_point)),
            int(self._is_collision(left_point)),
            int(self._is_collision(straight_point_2)),
            int(self._is_collision(right_point_2)),
            int(self._is_collision(left_point_2)),
            int(dir_up),
            int(dir_right),
            int(dir_down),
            int(dir_left),
            int(food_left),
            int(food_right),
            int(food_up),
            int(food_down),
            int(food_ahead),
            int(food_back),
            int(food_right_rel),
            int(food_left_rel),
            int(self._is_trap(straight_point)),
            int(self._is_trap(right_point)),
            int(self._is_trap(left_point)),
        )
        return state

    def _apply_action(self, action: int) -> None:
        if action == 1:
            self.dir_idx = (self.dir_idx + 1) % 4
        elif action == 2:
            self.dir_idx = (self.dir_idx - 1) % 4

    def _spawn_food(self) -> None:
        while True:
            candidate = (self.rng.randrange(self.width), self.rng.randrange(self.height))
            if candidate not in self.snake:
                self.food = candidate
                return

    def _next_point(self, point: tuple[int, int], dir_idx: int) -> tuple[int, int]:
        dx, dy = DIRECTIONS[dir_idx]
        return point[0] + dx, point[1] + dy

    def _is_collision(self, point: tuple[int, int]) -> bool:
        x, y = point
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        return point in self.snake

    def _is_trap(self, point: tuple[int, int]) -> bool:
        if self._is_collision(point):
            return True
        return self._free_neighbors_count(point) <= 1

    def _free_neighbors_count(self, point: tuple[int, int]) -> int:
        if self._is_collision(point):
            return 0
        free = 0
        for direction in range(4):
            neighbor = self._next_point(point, direction)
            if not self._is_collision(neighbor):
                free += 1
        return free

    def _would_collide(self, point: tuple[int, int], will_grow: bool) -> bool:
        x, y = point
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        body = self.snake if will_grow else self.snake[:-1]
        return point in body

    def _collision_reason(self, point: tuple[int, int], will_grow: bool) -> str | None:
        x, y = point
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return "wall"
        body = self.snake if will_grow else self.snake[:-1]
        if point in body:
            return "self"
        return None

    @staticmethod
    def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
