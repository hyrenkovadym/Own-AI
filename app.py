from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from apps.common import apply_dark_style


class LauncherApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Local AI Лаунчер")
        self.geometry("560x320")
        self.minsize(520, 280)
        apply_dark_style(self)
        self._build_layout()

    def _build_layout(self) -> None:
        shell = ttk.Frame(self, padding=18)
        shell.pack(fill=tk.BOTH, expand=True)

        ttk.Label(shell, text="Local AI Лаунчер", style="Header.TLabel").pack(anchor="w")
        ttk.Label(shell, text="Відкрий легкий окремий застосунок під поточну задачу.").pack(
            anchor="w", pady=(2, 14)
        )

        card = ttk.LabelFrame(shell, text="Оберіть Застосунок", style="Card.TLabelframe", padding=12)
        card.pack(fill=tk.BOTH, expand=True)

        ttk.Button(card, text="Відкрити Чат", command=self._open_chat).pack(fill=tk.X, pady=(4, 8))
        ttk.Button(card, text="Відкрити Змійку", command=self._open_snake).pack(fill=tk.X, pady=(0, 8))
        ttk.Button(card, text="Закрити Лаунчер", command=self.destroy).pack(fill=tk.X)

    def _open_chat(self) -> None:
        self.destroy()
        from apps.chat_app import run_chat_app

        run_chat_app()

    def _open_snake(self) -> None:
        self.destroy()
        from apps.snake_app import run_snake_app

        run_snake_app()


def run_app() -> None:
    app = LauncherApp()
    app.mainloop()


if __name__ == "__main__":
    run_app()
