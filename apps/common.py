from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from core import Kernel, ModuleSpec


def build_kernel() -> Kernel:
    specs = [
        ModuleSpec(name="chat", entrypoint="modules.chat.entry"),
        ModuleSpec(name="snake", entrypoint="modules.snake.entry"),
    ]
    return Kernel(specs=specs)


def apply_dark_style(root: tk.Tk) -> None:
    root.configure(bg="#0F172A")
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    style.configure("TFrame", background="#0F172A")
    style.configure("TLabel", background="#0F172A", foreground="#E2E8F0")
    style.configure("TButton", background="#0EA5E9", foreground="#0B1020", padding=6)
    style.map("TButton", background=[("active", "#38BDF8")])
    style.configure("Header.TLabel", font=("Segoe UI Semibold", 16), foreground="#F8FAFC")
    style.configure("Card.TLabelframe", background="#0B1222", foreground="#E2E8F0")
    style.configure("Card.TLabelframe.Label", background="#0B1222", foreground="#E2E8F0")

