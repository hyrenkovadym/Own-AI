from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk

from .common import apply_dark_style, build_kernel


class SnakeApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Local AI - Змійка")
        self.geometry("1160x760")
        self.minsize(940, 620)

        self.kernel = build_kernel()
        self._ui_queue: queue.Queue[tuple[Callable, tuple, dict]] = queue.Queue()
        self._busy = False
        self._anim_after_id: str | None = None
        self._anim_frames = []
        self._anim_index = 0
        self._mode: str | None = None
        self._live_cancel_requested = False
        self._cell_size = 26
        self._grid_w = 10
        self._grid_h = 10
        self._model_path = "models/snake_q.pkl"

        apply_dark_style(self)
        self._build_layout()
        self.after(80, self._drain_ui_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_layout(self) -> None:
        shell = ttk.Frame(self, padding=14)
        shell.pack(fill=tk.BOTH, expand=True)

        ttk.Label(shell, text="Модуль Змійки", style="Header.TLabel").pack(anchor="w")
        ttk.Label(shell, text="Окремий застосунок для змійки: тренування, оцінка та жива візуалізація.").pack(
            anchor="w", pady=(2, 12)
        )

        controls = ttk.LabelFrame(shell, text="Керування", style="Card.TLabelframe", padding=10)
        controls.pack(fill=tk.X, pady=(0, 10))

        for idx in range(8):
            controls.columnconfigure(idx, weight=1 if idx % 2 else 0)

        self.episodes_var = tk.StringVar(value="25")
        self.max_steps_var = tk.StringVar(value="0")
        self.width_var = tk.StringVar(value="10")
        self.height_var = tk.StringVar(value="10")
        self.seed_var = tk.StringVar(value="42")
        self.log_every_var = tk.StringVar(value="100")
        self.eval_episodes_var = tk.StringVar(value="20")
        self.goal_score_var = tk.StringVar(value="100")
        self.speed_ms_var = tk.StringVar(value="30")
        self.resume_var = tk.BooleanVar(value=True)
        self.total_episodes_var = tk.StringVar(value="0")

        ttk.Label(controls, text="Сцени").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(controls, textvariable=self.episodes_var, width=8).grid(row=0, column=1, sticky="w", pady=4)
        ttk.Label(controls, text="Макс кроків").grid(row=0, column=2, sticky="w", padx=(12, 8), pady=4)
        ttk.Entry(controls, textvariable=self.max_steps_var, width=8).grid(row=0, column=3, sticky="w", pady=4)
        ttk.Label(controls, text="Сітка W").grid(row=0, column=4, sticky="w", padx=(12, 8), pady=4)
        ttk.Entry(controls, textvariable=self.width_var, width=8).grid(row=0, column=5, sticky="w", pady=4)
        ttk.Label(controls, text="Сітка H").grid(row=0, column=6, sticky="w", padx=(12, 8), pady=4)
        ttk.Entry(controls, textvariable=self.height_var, width=8).grid(row=0, column=7, sticky="w", pady=4)

        ttk.Label(controls, text="Сід").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(controls, textvariable=self.seed_var, width=8).grid(row=1, column=1, sticky="w", pady=4)
        ttk.Label(controls, text="Лог кожні").grid(row=1, column=2, sticky="w", padx=(12, 8), pady=4)
        ttk.Entry(controls, textvariable=self.log_every_var, width=8).grid(row=1, column=3, sticky="w", pady=4)
        ttk.Label(controls, text="Епізоди оцінки").grid(row=1, column=4, sticky="w", padx=(12, 8), pady=4)
        ttk.Entry(controls, textvariable=self.eval_episodes_var, width=8).grid(row=1, column=5, sticky="w", pady=4)
        ttk.Label(controls, text="Цільовий рахунок").grid(row=1, column=6, sticky="w", padx=(12, 8), pady=4)
        ttk.Entry(controls, textvariable=self.goal_score_var, width=8).grid(row=1, column=7, sticky="w", pady=4)

        ttk.Label(controls, text="Швидкість візуалізації (мс)").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(controls, textvariable=self.speed_ms_var, width=8).grid(row=2, column=1, sticky="w", pady=4)
        self.resume_check = ttk.Checkbutton(
            controls,
            text="Продовжити модель",
            variable=self.resume_var,
            command=self._update_resume_ui,
        )
        self.resume_check.grid(row=2, column=2, sticky="w", padx=(12, 8), pady=4)
        self.resume_state_label = ttk.Label(controls, text="")
        self.resume_state_label.grid(row=2, column=3, sticky="w", padx=(2, 8), pady=4)
        ttk.Label(controls, text="Усього епізодів").grid(row=2, column=4, sticky="w", padx=(12, 8), pady=4)
        ttk.Label(controls, textvariable=self.total_episodes_var).grid(row=2, column=5, sticky="w", pady=4)
        ttk.Label(controls, text="Макс кроків: 0 = мʼяко без ліміту").grid(
            row=2, column=6, columnspan=2, sticky="w", padx=(12, 8), pady=4
        )

        actions = ttk.Frame(controls)
        actions.grid(row=3, column=0, columnspan=8, sticky="e", pady=(8, 0))
        self.train_btn = ttk.Button(actions, text="Тренувати", command=self._on_train)
        self.train_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.auto_btn = ttk.Button(actions, text="Автонавчання", command=self._on_auto_learn)
        self.auto_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.live_btn = ttk.Button(actions, text="Вчитись наживо", command=self._on_learn_live)
        self.live_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.eval_btn = ttk.Button(actions, text="Оцінити", command=self._on_eval)
        self.eval_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.visual_btn = ttk.Button(actions, text="Візуалізувати", command=self._on_visualize)
        self.visual_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.reset_model_btn = ttk.Button(actions, text="Скинути памʼять", command=self._on_reset_model)
        self.reset_model_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.stop_btn = ttk.Button(
            actions,
            text="Стоп",
            command=self._on_stop_clicked,
            state=tk.DISABLED,
        )
        self.stop_btn.pack(side=tk.LEFT)

        body = ttk.Frame(shell)
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        vis = ttk.LabelFrame(body, text="Візуалізація", style="Card.TLabelframe", padding=10)
        vis.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        vis.rowconfigure(0, weight=1)
        vis.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(vis, width=520, height=520, bg="#020617", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.visual_meta = ttk.Label(vis, text="Епізод: - | Рахунок: 0 | Крок: 0")
        self.visual_meta.grid(row=1, column=0, sticky="w", pady=(8, 0))

        logs = ttk.LabelFrame(body, text="Логи", style="Card.TLabelframe", padding=10)
        logs.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        logs.rowconfigure(0, weight=1)
        logs.columnconfigure(0, weight=1)

        self.log_text = tk.Text(
            logs,
            bg="#0B1222",
            fg="#E2E8F0",
            insertbackground="#E2E8F0",
            relief=tk.FLAT,
            wrap=tk.WORD,
            font=("Consolas", 11),
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_scroll = ttk.Scrollbar(logs, orient="vertical", command=self.log_text.yview)
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=log_scroll.set)

        self._append_log("Застосунок змійки готовий.")
        self._append_log("Порада: натисни 'Вчитись наживо', щоб бачити помилки (стіна/себе) і прогрес по епізодах.")
        self._append_log("Порада: 'Тренувати' запускає рівно вказану кількість сцен без анімації.")
        self._append_log("Порада: постав 'Макс кроків' = 0 для мʼяко необмеженої довжини епізоду.")
        self._update_resume_ui()
        self._refresh_total_episodes()
        self._draw_empty_grid(self._grid_w, self._grid_h)

    def _snake_module(self):
        return self.kernel.load("snake")

    def _on_train(self) -> None:
        try:
            episodes = int(self.episodes_var.get().strip())
        except ValueError:
            messagebox.showerror("Сцени", "Сцени мають бути цілим числом.")
            return
        goal_score = self._parse_goal_score()
        if goal_score is None:
            return
        self._start_train(episodes=episodes, goal_score=goal_score)

    def _start_train(self, episodes: int, goal_score: int) -> None:
        if self._busy:
            return
        if episodes <= 0:
            messagebox.showerror("Сцени", "Сцени мають бути > 0.")
            return
        self._mode = "train"
        self._live_cancel_requested = False
        self._set_busy(True)
        self.stop_btn.configure(state=tk.NORMAL)
        self._append_log(f"Тренування запущено... сцен={episodes}")

        def worker() -> None:
            try:
                stats = self._snake_module().train(
                    episodes=episodes,
                    max_steps=int(self.max_steps_var.get().strip()),
                    width=int(self.width_var.get().strip()),
                    height=int(self.height_var.get().strip()),
                    model_path=self._model_path,
                    seed=int(self.seed_var.get().strip()),
                    log_every=int(self.log_every_var.get().strip()),
                    progress_callback=self._append_log_threadsafe,
                    resume=self.resume_var.get(),
                    goal_score=goal_score,
                    stop_predicate=lambda: self._live_cancel_requested,
                )
                self._append_log_threadsafe(
                    (
                        "Тренування завершено: "
                        f"episodes={stats.episodes}, avg_last_100={stats.avg_score_last_100:.2f}, "
                        f"best={stats.best_score} ({stats.best_fill_percent:.1f}%), wins={stats.wins}, "
                        f"goal={stats.goal_score_requested}->{stats.goal_score_effective}, "
                        f"q_states={stats.q_states}, "
                        f"total_episodes={stats.total_episodes}, resumed={stats.resumed_from_model}, "
                        f"eps={stats.epsilon_start:.3f}->{stats.epsilon_end:.3f}"
                    )
                )
                self._queue_ui(self.total_episodes_var.set, str(stats.total_episodes))
            except Exception as exc:  # noqa: BLE001
                self._append_log_threadsafe(f"Помилка: {exc}")
            finally:
                self._queue_ui(self._set_busy, False)

        threading.Thread(target=worker, daemon=True).start()

    def _on_auto_learn(self) -> None:
        if self._busy:
            return
        goal_score = self._parse_goal_score()
        if goal_score is None:
            return
        try:
            cycle_episodes = int(self.episodes_var.get().strip())
        except ValueError:
            messagebox.showerror("Автонавчання", "Сцени мають бути цілим числом.")
            return
        if cycle_episodes <= 0:
            messagebox.showerror("Автонавчання", "Сцени мають бути > 0.")
            return

        self._mode = "auto_learn"
        self._live_cancel_requested = False
        self._set_busy(True)
        self.stop_btn.configure(state=tk.NORMAL)
        self._append_log(
            f"Автонавчання запущено... сцен у циклі={cycle_episodes}. "
            "Агент тренуватиметься безперервно, доки не натиснеш Стоп."
        )

        def worker() -> None:
            cycle = 0
            best_overall = 0
            wins_total = 0
            epsilon_start = None
            epsilon_end = None
            try:
                while not self._live_cancel_requested:
                    cycle += 1
                    self._append_log_threadsafe(f"[auto] цикл={cycle} старт")
                    stats = self._snake_module().train(
                        episodes=cycle_episodes,
                        max_steps=int(self.max_steps_var.get().strip()),
                        width=int(self.width_var.get().strip()),
                        height=int(self.height_var.get().strip()),
                        model_path=self._model_path,
                        seed=int(self.seed_var.get().strip()) + cycle - 1,
                        log_every=int(self.log_every_var.get().strip()),
                        progress_callback=self._append_log_threadsafe,
                        resume=self.resume_var.get() if cycle == 1 else True,
                        goal_score=goal_score,
                        stop_predicate=lambda: self._live_cancel_requested,
                    )
                    if stats.episodes <= 0:
                        break

                    if epsilon_start is None:
                        epsilon_start = stats.epsilon_start
                    epsilon_end = stats.epsilon_end
                    wins_total += stats.wins
                    best_overall = max(best_overall, stats.best_score)
                    self._queue_ui(self.total_episodes_var.set, str(stats.total_episodes))
                    self._append_log_threadsafe(
                        (
                            f"[auto] цикл={cycle} завершено: натреновано={stats.episodes}, "
                            f"best_cycle={stats.best_score}, best_overall={best_overall}, "
                            f"wins_total={wins_total}, total_episodes={stats.total_episodes}, "
                            f"eps={stats.epsilon_start:.3f}->{stats.epsilon_end:.3f}"
                        )
                    )

                    if cycle == 1 and not self.resume_var.get():
                        self._queue_ui(self.resume_var.set, True)
                        self._queue_ui(self._update_resume_ui)
                        self._append_log_threadsafe("[auto] Продовження моделі увімкнено для безперервного навчання.")

                eps_start_text = "-" if epsilon_start is None else f"{epsilon_start:.3f}"
                eps_end_text = "-" if epsilon_end is None else f"{epsilon_end:.3f}"
                self._append_log_threadsafe(
                    (
                        f"Автонавчання зупинено. Циклів={cycle}, best_overall={best_overall}, "
                        f"wins_total={wins_total}, eps={eps_start_text}->{eps_end_text}"
                    )
                )
            except Exception as exc:  # noqa: BLE001
                self._append_log_threadsafe(f"Помилка: {exc}")
            finally:
                self._queue_ui(self._set_busy, False)

        threading.Thread(target=worker, daemon=True).start()

    def _on_eval(self) -> None:
        if self._busy:
            return
        goal_score = self._parse_goal_score()
        if goal_score is None:
            return
        self._mode = "eval"
        self._set_busy(True)
        self._append_log("Оцінювання запущено...")

        def worker() -> None:
            try:
                stats = self._snake_module().play(
                    episodes=int(self.eval_episodes_var.get().strip()),
                    max_steps=int(self.max_steps_var.get().strip()),
                    width=int(self.width_var.get().strip()),
                    height=int(self.height_var.get().strip()),
                    model_path=self._model_path,
                    seed=int(self.seed_var.get().strip()),
                    goal_score=goal_score,
                )
                self._append_log_threadsafe(
                    (
                        "Оцінювання завершено: "
                        f"episodes={stats.episodes}, avg_score={stats.avg_score:.2f}, "
                        f"best={stats.best_score} ({stats.best_fill_percent:.1f}%), "
                        f"wins={stats.wins}, goal={stats.goal_score_requested}->{stats.goal_score_effective}"
                    )
                )
            except Exception as exc:  # noqa: BLE001
                self._append_log_threadsafe(f"Помилка: {exc}")
            finally:
                self._queue_ui(self._set_busy, False)

        threading.Thread(target=worker, daemon=True).start()

    def _on_visualize(self) -> None:
        if self._busy:
            return
        goal_score = self._parse_goal_score()
        if goal_score is None:
            return
        self._mode = "visualize"
        self._live_cancel_requested = False
        self._stop_visualization()
        self._set_busy(True)
        self.stop_btn.configure(state=tk.NORMAL)
        self._append_log("Готую прогін для візуалізації...")

        def worker() -> None:
            try:
                width = int(self.width_var.get().strip())
                height = int(self.height_var.get().strip())
                frames = self._snake_module().rollout(
                    max_steps=int(self.max_steps_var.get().strip()),
                    width=width,
                    height=height,
                    model_path=self._model_path,
                    seed=int(self.seed_var.get().strip()),
                    goal_score=goal_score,
                )
                self._queue_ui(self._start_animation, frames, width, height)
            except Exception as exc:  # noqa: BLE001
                self._queue_ui(self._append_log, f"Помилка: {exc}")
                self._queue_ui(self._set_busy, False)

        threading.Thread(target=worker, daemon=True).start()

    def _start_animation(self, frames, width: int, height: int) -> None:
        if not frames:
            self._append_log("Немає кадрів для візуалізації.")
            self._set_busy(False)
            return

        self._anim_frames = frames
        self._anim_index = 0
        self._grid_w = width
        self._grid_h = height

        size = max(10, min(42, int(560 / max(width, height))))
        self._cell_size = size
        canvas_w = width * size + 2
        canvas_h = height * size + 2
        self.canvas.configure(width=canvas_w, height=canvas_h)
        self.stop_btn.configure(state=tk.NORMAL)
        self._append_log(f"Візуалізую {len(frames)} кадрів...")
        self._animate_step()

    def _animate_step(self) -> None:
        self._anim_after_id = None
        if self._anim_index >= len(self._anim_frames):
            self._append_log("Візуалізацію завершено.")
            self.stop_btn.configure(state=tk.DISABLED)
            self._set_busy(False)
            return

        frame = self._anim_frames[self._anim_index]
        self._draw_frame(frame)
        self._anim_index += 1

        delay = int(self.speed_ms_var.get().strip() or "70")
        delay = max(10, delay)
        self._anim_after_id = self.after(delay, self._animate_step)

    def _draw_empty_grid(self, width: int, height: int) -> None:
        self.canvas.delete("all")
        wpx = width * self._cell_size
        hpx = height * self._cell_size
        self.canvas.create_rectangle(1, 1, wpx + 1, hpx + 1, outline="#1E293B", width=2)
        for x in range(width + 1):
            px = x * self._cell_size + 1
            self.canvas.create_line(px, 1, px, hpx + 1, fill="#17243C")
        for y in range(height + 1):
            py = y * self._cell_size + 1
            self.canvas.create_line(1, py, wpx + 1, py, fill="#17243C")

    def _draw_frame(self, frame) -> None:
        self._draw_empty_grid(self._grid_w, self._grid_h)

        fx, fy = frame.food
        self._draw_cell(fx, fy, "#EF4444")

        for i, (sx, sy) in enumerate(frame.snake):
            color = "#22D3EE" if i == 0 else "#22C55E"
            self._draw_cell(sx, sy, color)

        episode = getattr(frame, "episode", None)
        action = getattr(frame, "action", None)
        reward = getattr(frame, "reward", None)
        epsilon = getattr(frame, "epsilon", None)
        reason = getattr(frame, "done_reason", None)

        action_text = "-"
        if action is not None:
            action_text = self._action_label(action)
        reward_text = "-" if reward is None else f"{reward:+.2f}"
        epsilon_text = "-" if epsilon is None else f"{epsilon:.3f}"
        episode_text = "-" if episode is None else str(episode)
        max_possible = max(1, self._grid_w * self._grid_h - 3)
        fill_percent = 100.0 * frame.score / max_possible
        done_text = " | DONE" if frame.done else ""
        reason_text = f" | причина={reason}" if reason else ""
        self.visual_meta.configure(
            text=(
                f"Епізод: {episode_text} | Рахунок: {frame.score} ({fill_percent:.1f}%) | Крок: {frame.step} "
                f"| action={action_text} | reward={reward_text} | eps={epsilon_text}"
                f"{done_text}{reason_text}"
            )
        )

    def _draw_cell(self, x: int, y: int, color: str) -> None:
        pad = 2
        x0 = x * self._cell_size + 1 + pad
        y0 = y * self._cell_size + 1 + pad
        x1 = (x + 1) * self._cell_size + 1 - pad
        y1 = (y + 1) * self._cell_size + 1 - pad
        self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")

    def _stop_visualization(self, release_busy: bool = False) -> None:
        if self._anim_after_id is not None:
            try:
                self.after_cancel(self._anim_after_id)
            except tk.TclError:
                pass
            self._anim_after_id = None
        self.stop_btn.configure(state=tk.DISABLED)
        if release_busy and self._busy:
            self._set_busy(False)
            self._append_log("Візуалізацію зупинено.")

    def _on_stop_clicked(self) -> None:
        if not self._busy:
            return
        if self._mode == "learn_live":
            self._live_cancel_requested = True
            self._append_log("Зупинка запитана для навчання наживо...")
            self.stop_btn.configure(state=tk.DISABLED)
            return
        if self._mode == "auto_learn":
            self._live_cancel_requested = True
            self._append_log("Зупинка запитана для автонавчання...")
            self.stop_btn.configure(state=tk.DISABLED)
            return
        if self._mode == "train":
            self._live_cancel_requested = True
            self._append_log("Зупинка запитана для тренування...")
            self.stop_btn.configure(state=tk.DISABLED)
            return
        if self._mode == "visualize":
            self._stop_visualization(release_busy=True)
            return
        self._live_cancel_requested = True
        self._append_log("Зупинка запитана...")
        self.stop_btn.configure(state=tk.DISABLED)

    def _on_reset_model(self) -> None:
        if self._busy:
            messagebox.showinfo("Скинути памʼять", "Спочатку зупини поточний запуск.")
            return

        model_path = Path(self._model_path)
        if not model_path.exists():
            messagebox.showinfo("Скинути памʼять", f"Файл моделі не знайдено:\n{model_path}")
            return

        ok = messagebox.askyesno(
            "Скинути памʼять",
            "Це назавжди видалить вивчену памʼять змійки.\nПродовжити?",
        )
        if not ok:
            self._append_log("Скидання памʼяті скасовано.")
            return

        token = simpledialog.askstring("Підтвердження скидання", "Введи DELETE для підтвердження:")
        if token != "DELETE":
            self._append_log("Скидання памʼяті скасовано (підтвердження не пройдено).")
            return

        try:
            model_path.unlink()
            self._append_log(f"Файл моделі видалено: {model_path}")
            self.resume_var.set(False)
            self._update_resume_ui()
            self.total_episodes_var.set("0")
            self._append_log("Продовження моделі вимкнено для наступного запуску.")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Помилка скидання памʼяті", str(exc))

    def _on_learn_live(self) -> None:
        if self._busy:
            return
        goal_score = self._parse_goal_score()
        if goal_score is None:
            return
        self._mode = "learn_live"
        self._live_cancel_requested = False
        self._stop_visualization()
        self._set_busy(True)
        self.stop_btn.configure(state=tk.NORMAL)
        self._append_log(
            f"Навчання наживо запущено... сцен={self.episodes_var.get().strip()} "
            "Ти побачиш дії, помилки й покращення в реальному часі."
        )

        try:
            width = int(self.width_var.get().strip())
            height = int(self.height_var.get().strip())
            delay_ms = max(10, int(self.speed_ms_var.get().strip() or "70"))
        except Exception as exc:  # noqa: BLE001
            self._append_log(f"Помилка: {exc}")
            self._set_busy(False)
            return

        size = max(10, min(42, int(560 / max(width, height))))
        self._cell_size = size
        self._grid_w = width
        self._grid_h = height
        self.canvas.configure(width=width * size + 2, height=height * size + 2)
        self._draw_empty_grid(width, height)

        def worker() -> None:
            try:
                module = self._snake_module()

                def on_frame(frame) -> None:
                    self._queue_ui(self._draw_frame, frame)
                    if frame.done and frame.done_reason:
                        self._queue_ui(
                            self._append_log,
                            (
                                f"[live] епізод={frame.episode} завершився через {frame.done_reason}; "
                                f"рахунок={frame.score}; крок={frame.step}"
                            ),
                        )
                    time.sleep(delay_ms / 1000.0)

                stats = module.train_live(
                    episodes=int(self.episodes_var.get().strip()),
                    max_steps=int(self.max_steps_var.get().strip()),
                    width=width,
                    height=height,
                    model_path=self._model_path,
                    seed=int(self.seed_var.get().strip()),
                    frame_callback=on_frame,
                    episode_callback=self._append_log_threadsafe,
                    stop_predicate=lambda: self._live_cancel_requested,
                    resume=self.resume_var.get(),
                    goal_score=goal_score,
                )
                self._append_log_threadsafe(
                    (
                        "Навчання наживо завершено: "
                        f"episodes={stats.episodes}, avg_last_100={stats.avg_score_last_100:.2f}, "
                        f"best={stats.best_score} ({stats.best_fill_percent:.1f}%), wins={stats.wins}, "
                        f"goal={stats.goal_score_requested}->{stats.goal_score_effective}, "
                        f"q_states={stats.q_states}, "
                        f"total_episodes={stats.total_episodes}, resumed={stats.resumed_from_model}, "
                        f"eps={stats.epsilon_start:.3f}->{stats.epsilon_end:.3f}"
                    )
                )
                self._queue_ui(self.total_episodes_var.set, str(stats.total_episodes))
            except Exception as exc:  # noqa: BLE001
                self._append_log_threadsafe(f"Помилка: {exc}")
            finally:
                self._queue_ui(self._set_busy, False)

        threading.Thread(target=worker, daemon=True).start()

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        self.train_btn.configure(state=state)
        self.auto_btn.configure(state=state)
        self.live_btn.configure(state=state)
        self.eval_btn.configure(state=state)
        self.visual_btn.configure(state=state)
        self.reset_model_btn.configure(state=state)
        self.resume_check.configure(state=state)
        if not busy:
            self._mode = None
            self._live_cancel_requested = False
            self.stop_btn.configure(state=tk.DISABLED)

    def _append_log(self, text: str) -> None:
        self.log_text.insert(tk.END, f"{text}\n")
        self.log_text.see(tk.END)

    def _append_log_threadsafe(self, text: str) -> None:
        self._queue_ui(self._append_log, text)

    def _queue_ui(self, fn, *args, **kwargs) -> None:
        self._ui_queue.put((fn, args, kwargs))

    def _drain_ui_queue(self) -> None:
        while True:
            try:
                fn, args, kwargs = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            fn(*args, **kwargs)
        self.after(80, self._drain_ui_queue)

    def _on_close(self) -> None:
        self._stop_visualization()
        self._live_cancel_requested = True
        try:
            self.kernel.unload_all()
        except Exception:
            pass
        self.destroy()

    @staticmethod
    def _action_label(action: int) -> str:
        if action == 0:
            return "прямо"
        if action == 1:
            return "вправо"
        if action == 2:
            return "вліво"
        return f"дія({action})"

    def _update_resume_ui(self) -> None:
        if self.resume_var.get():
            self.resume_state_label.configure(text="ON")
        else:
            self.resume_state_label.configure(text="OFF")

    def _refresh_total_episodes(self) -> None:
        try:
            info = self._snake_module().get_model_info(model_path=self._model_path)
            self.total_episodes_var.set(str(info.total_episodes))
        except Exception:
            self.total_episodes_var.set("?")

    def _parse_goal_score(self) -> int | None:
        raw = self.goal_score_var.get().strip()
        try:
            value = int(raw)
        except ValueError:
            messagebox.showerror("Цільовий рахунок", "Цільовий рахунок має бути цілим числом.")
            return None
        if value <= 0:
            messagebox.showerror("Цільовий рахунок", "Цільовий рахунок має бути > 0.")
            return None
        return value


def run_snake_app() -> None:
    app = SnakeApp()
    app.mainloop()


if __name__ == "__main__":
    run_snake_app()

