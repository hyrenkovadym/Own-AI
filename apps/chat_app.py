from __future__ import annotations

import queue
import threading
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import messagebox, ttk

from .common import apply_dark_style, build_kernel


class ChatApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Local AI - Чат")
        self.geometry("980x700")
        self.minsize(860, 560)

        self.kernel = build_kernel()
        self._ui_queue: queue.Queue[tuple[Callable, tuple, dict]] = queue.Queue()
        self._busy = False
        self._developer_mode = tk.BooleanVar(value=False)
        self._role_var = tk.StringVar(value="programmer")

        apply_dark_style(self)
        self._build_layout()
        self.after(80, self._drain_ui_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_layout(self) -> None:
        shell = ttk.Frame(self, padding=14)
        shell.pack(fill=tk.BOTH, expand=True)

        ttk.Label(shell, text="Модуль Чату", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            shell,
            text="Окремий застосунок чату. Навчання, діалог і донавчання в одному вікні.",
        ).pack(anchor="w", pady=(2, 12))

        mode_row = ttk.Frame(shell)
        mode_row.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(mode_row, text="Роль:").pack(side=tk.LEFT)
        role = ttk.Combobox(
            mode_row,
            textvariable=self._role_var,
            values=["programmer", "chat"],
            state="readonly",
            width=14,
        )
        role.pack(side=tk.LEFT, padx=(8, 12))
        ttk.Checkbutton(
            mode_row,
            text="Режим розробника",
            variable=self._developer_mode,
            command=self._toggle_developer_mode,
        ).pack(side=tk.RIGHT)

        self.cfg = ttk.LabelFrame(shell, text="Налаштування Моделі", style="Card.TLabelframe", padding=10)
        self.cfg.pack(fill=tk.X, pady=(0, 10))
        self.cfg.columnconfigure(1, weight=1)
        self.cfg.columnconfigure(3, weight=1)

        self.dataset_var = tk.StringVar(value="modules/chat/data/intents.json")
        self.model_var = tk.StringVar(value="models/chat_intent.pkl")
        self.threshold_var = tk.StringVar(value="0.25")
        self.session_var = tk.StringVar(value="desktop")

        ttk.Label(self.cfg, text="Датасет").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(self.cfg, textvariable=self.dataset_var).grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Label(self.cfg, text="Модель").grid(row=0, column=2, sticky="w", padx=(12, 8), pady=4)
        ttk.Entry(self.cfg, textvariable=self.model_var).grid(row=0, column=3, sticky="ew", pady=4)

        ttk.Label(self.cfg, text="Поріг").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(self.cfg, textvariable=self.threshold_var, width=8).grid(row=1, column=1, sticky="w", pady=4)
        ttk.Label(self.cfg, text="Сесія").grid(row=1, column=2, sticky="w", padx=(12, 8), pady=4)
        ttk.Entry(self.cfg, textvariable=self.session_var, width=18).grid(row=1, column=3, sticky="w", pady=4)

        actions = ttk.Frame(self.cfg)
        actions.grid(row=2, column=0, columnspan=4, sticky="e", pady=(8, 0))
        self.train_btn = ttk.Button(actions, text="Навчити", command=self._on_train)
        self.train_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.reset_btn = ttk.Button(actions, text="Скинути Сесію", command=self._on_reset_session)
        self.reset_btn.pack(side=tk.LEFT)

        chat_box = ttk.LabelFrame(shell, text="Діалог", style="Card.TLabelframe", padding=10)
        chat_box.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        chat_box.columnconfigure(0, weight=1)
        chat_box.rowconfigure(0, weight=1)

        self.chat_log = tk.Text(
            chat_box,
            bg="#0B1222",
            fg="#E2E8F0",
            insertbackground="#E2E8F0",
            relief=tk.FLAT,
            wrap=tk.WORD,
            font=("Consolas", 11),
        )
        self.chat_log.grid(row=0, column=0, sticky="nsew")
        chat_scroll = ttk.Scrollbar(chat_box, orient="vertical", command=self.chat_log.yview)
        chat_scroll.grid(row=0, column=1, sticky="ns")
        self.chat_log.configure(yscrollcommand=chat_scroll.set)

        input_row = ttk.Frame(chat_box)
        input_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        input_row.columnconfigure(0, weight=1)

        self.input_var = tk.StringVar()
        input_entry = ttk.Entry(input_row, textvariable=self.input_var)
        input_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        input_entry.bind("<Return>", lambda _: self._on_send())
        self.send_btn = ttk.Button(input_row, text="Надіслати", command=self._on_send)
        self.send_btn.grid(row=0, column=1)

        self.teach = ttk.LabelFrame(shell, text="Навчити Прикладом", style="Card.TLabelframe", padding=10)
        self.teach.pack(fill=tk.X)
        self.teach.columnconfigure(1, weight=1)
        self.teach.columnconfigure(3, weight=1)
        self.teach.columnconfigure(5, weight=1)

        self.intent_var = tk.StringVar()
        self.example_var = tk.StringVar()
        self.response_var = tk.StringVar()
        self.fallback_status_var = tk.StringVar(value="Fallback-черга: 0 (всього 0)")

        ttk.Label(self.teach, text="Інтент").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(self.teach, textvariable=self.intent_var).grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Label(self.teach, text="Приклад").grid(row=0, column=2, sticky="w", padx=(12, 8), pady=4)
        ttk.Entry(self.teach, textvariable=self.example_var).grid(row=0, column=3, sticky="ew", pady=4)
        ttk.Label(self.teach, text="Відповідь").grid(row=0, column=4, sticky="w", padx=(12, 8), pady=4)
        ttk.Entry(self.teach, textvariable=self.response_var).grid(row=0, column=5, sticky="ew", pady=4)

        ttk.Label(self.teach, textvariable=self.fallback_status_var).grid(row=1, column=0, columnspan=3, sticky="w", pady=(8, 0))
        self.pick_fallback_btn = ttk.Button(self.teach, text="Взяти Fallback", command=self._on_pick_fallback)
        self.pick_fallback_btn.grid(row=1, column=3, sticky="w", pady=(8, 0))
        self.teach_fallback_btn = ttk.Button(
            self.teach,
            text="Навчити з Fallback",
            command=self._on_teach_from_fallback,
        )
        self.teach_fallback_btn.grid(row=1, column=4, sticky="w", padx=(8, 0), pady=(8, 0))
        self.teach_btn = ttk.Button(self.teach, text="Додати + Перенавчити", command=self._on_teach)
        self.teach_btn.grid(row=1, column=5, sticky="e", pady=(8, 0))

        self._append("система", "Чат-застосунок готовий.")
        self._append("система", "Порада: якщо відповідь неточна, додай приклад через «Додати + Перенавчити».")
        self._refresh_fallback_status()
        self._toggle_developer_mode()

    def _chat_module(self):
        return self.kernel.load("chat")

    def _on_train(self) -> None:
        if self._busy:
            return
        self._set_busy(True)
        self._append("система", "Навчаю чат-модель...")

        def worker() -> None:
            try:
                stats = self._chat_module().train(
                    dataset_path=self.dataset_var.get().strip(),
                    model_path=self.model_var.get().strip(),
                    seed=42,
                )
                self._queue_ui(
                    self._append,
                    "система",
                    (
                        "Навчання завершено: "
                        f"інтентів={stats.intents}, прикладів={stats.examples}, "
                        f"словник={stats.vocab_size}, модель={stats.model_path}"
                    ),
                )
                self._queue_ui(self._refresh_fallback_status)
            except Exception as exc:  # noqa: BLE001
                self._queue_ui(messagebox.showerror, "Помилка Навчання Чату", str(exc))
            finally:
                self._queue_ui(self._set_busy, False)

        threading.Thread(target=worker, daemon=True).start()

    def _on_send(self) -> None:
        text = self.input_var.get().strip()
        if not text or self._busy:
            return

        self.input_var.set("")
        self._append("ти", text)
        model_path = Path(self.model_var.get().strip())

        def worker() -> None:
            try:
                module = self._chat_module()
                if not model_path.exists():
                    stats = module.train(
                        dataset_path=self.dataset_var.get().strip(),
                        model_path=str(model_path),
                        seed=42,
                    )
                    self._queue_ui(
                        self._append,
                        "система",
                        f"Модель навчено автоматично ({stats.intents} інтентів, {stats.examples} прикладів).",
                    )

                if self._role_var.get().strip() == "programmer":
                    reply = module.reply_programmer(
                        user_text=text,
                        model_path=str(model_path),
                        confidence_threshold=float(self.threshold_var.get().strip()),
                        session_id=self.session_var.get().strip() or "desktop",
                    )
                else:
                    reply = module.reply(
                        user_text=text,
                        model_path=str(model_path),
                        confidence_threshold=float(self.threshold_var.get().strip()),
                        session_id=self.session_var.get().strip() or "desktop",
                    )
                self._queue_ui(
                    self._append,
                    "бот",
                    f"{reply.text} (intent={reply.intent}, conf={reply.confidence:.2f})",
                )
                self._queue_ui(self._refresh_fallback_status)
            except Exception as exc:  # noqa: BLE001
                self._queue_ui(messagebox.showerror, "Помилка Чату", str(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _on_teach(self) -> None:
        if self._busy:
            return
        intent = self.intent_var.get().strip()
        example = self.example_var.get().strip()
        response = self.response_var.get().strip()
        if not intent or not example or not response:
            messagebox.showwarning("Навчити", "Заповни інтент, приклад і відповідь.")
            return

        self._set_busy(True)
        self._append("система", f"Додаю інтент '{intent}'...")

        def worker() -> None:
            try:
                module = self._chat_module()
                module.teach(
                    intent=intent,
                    example=example,
                    response=response,
                    dataset_path=self.dataset_var.get().strip(),
                )
                module.train(
                    dataset_path=self.dataset_var.get().strip(),
                    model_path=self.model_var.get().strip(),
                    seed=42,
                )
                self._queue_ui(self._append, "система", f"Інтент '{intent}' вивчено.")
                self._queue_ui(self.intent_var.set, "")
                self._queue_ui(self.example_var.set, "")
                self._queue_ui(self.response_var.set, "")
                self._queue_ui(self._refresh_fallback_status)
            except Exception as exc:  # noqa: BLE001
                self._queue_ui(messagebox.showerror, "Помилка Донавчання", str(exc))
            finally:
                self._queue_ui(self._set_busy, False)

        threading.Thread(target=worker, daemon=True).start()

    def _on_reset_session(self) -> None:
        session_id = self.session_var.get().strip() or "desktop"
        try:
            self._chat_module().reset_session(session_id)
            self._append("система", f"Сесію '{session_id}' скинуто.")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Помилка Скидання Сесії", str(exc))

    def _on_pick_fallback(self) -> None:
        try:
            top = self._chat_module().fallback_peek()
            if not top:
                self._append("система", "Fallback-черга порожня.")
                self._refresh_fallback_status()
                return
            self.example_var.set(top)
            self._append("система", f"Підставлено fallback: {top}")
            self._refresh_fallback_status()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Помилка Fallback-Черги", str(exc))

    def _on_teach_from_fallback(self) -> None:
        if self._busy:
            return

        intent = self.intent_var.get().strip()
        response = self.response_var.get().strip()
        if not intent or not response:
            messagebox.showwarning("Навчити з Fallback", "Заповни інтент і відповідь.")
            return

        self._set_busy(True)
        self._append("система", f"Донавчаю з fallback для інтенту '{intent}'...")

        def worker() -> None:
            try:
                module = self._chat_module()
                example = module.fallback_peek()
                if not example:
                    self._queue_ui(self._append, "система", "Fallback-черга порожня.")
                    return

                module.teach(
                    intent=intent,
                    example=example,
                    response=response,
                    dataset_path=self.dataset_var.get().strip(),
                )
                module.fallback_consume(example)
                module.train(
                    dataset_path=self.dataset_var.get().strip(),
                    model_path=self.model_var.get().strip(),
                    seed=42,
                )
                self._queue_ui(self._append, "система", f"Додано приклад з fallback: {example}")
                self._queue_ui(self.example_var.set, "")
                self._queue_ui(self._refresh_fallback_status)
            except Exception as exc:  # noqa: BLE001
                self._queue_ui(messagebox.showerror, "Помилка Навчання з Fallback", str(exc))
            finally:
                self._queue_ui(self._set_busy, False)

        threading.Thread(target=worker, daemon=True).start()

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        self.train_btn.configure(state=state)
        self.teach_btn.configure(state=state)
        self.pick_fallback_btn.configure(state=state)
        self.teach_fallback_btn.configure(state=state)
        self.send_btn.configure(state=state)
        self.reset_btn.configure(state=state)

    def _append(self, role: str, text: str) -> None:
        self.chat_log.insert(tk.END, f"{role}> {text}\n")
        self.chat_log.see(tk.END)

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

    def _refresh_fallback_status(self) -> None:
        try:
            unique_items, total_items = self._chat_module().fallback_queue_stats()
            self.fallback_status_var.set(f"Fallback-черга: {unique_items} (всього {total_items})")
        except Exception:
            self.fallback_status_var.set("Fallback-черга: n/a")

    def _toggle_developer_mode(self) -> None:
        dev = bool(self._developer_mode.get())
        if dev:
            if not self.cfg.winfo_ismapped():
                self.cfg.pack(fill=tk.X, pady=(0, 10), before=self.chat_log.master)
            if not self.teach.winfo_ismapped():
                self.teach.pack(fill=tk.X)
        else:
            if self.cfg.winfo_ismapped():
                self.cfg.pack_forget()
            if self.teach.winfo_ismapped():
                self.teach.pack_forget()

    def _on_close(self) -> None:
        try:
            self.kernel.unload_all()
        except Exception:
            pass
        self.destroy()


def run_chat_app() -> None:
    app = ChatApp()
    app.mainloop()


if __name__ == "__main__":
    run_chat_app()
