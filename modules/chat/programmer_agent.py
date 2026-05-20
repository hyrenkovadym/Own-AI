from __future__ import annotations

import re
import subprocess
from pathlib import Path


class LocalProgrammerAgent:
    """Fully local rule-based coding helper (no external model/runtime)."""

    def __init__(self, workspace_root: str | Path) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self._skip_dirs = {".git", "__pycache__", ".venv", "venv", "env", "models"}
        self._search_ext = {
            ".py",
            ".md",
            ".txt",
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".ini",
            ".cfg",
            ".js",
            ".ts",
            ".tsx",
            ".jsx",
            ".html",
            ".css",
            ".sql",
            ".sh",
            ".bat",
            ".ps1",
        }

    def reply(self, user_text: str) -> tuple[str, float, str]:
        text = user_text.strip()
        low = text.lower()
        if not text:
            return "programmer", 0.30, "Опиши задачу по коду, і я почну."

        # Explicit commands first.
        if low.startswith("/help"):
            return "programmer_help", 1.0, self._help_text()
        if low.startswith("/ls"):
            arg = text[3:].strip()
            return "programmer_ls", 0.98, self._list_files(arg or ".")
        if low.startswith("/read"):
            arg = text[5:].strip()
            if not arg:
                return "programmer_read", 0.2, "Формат: /read шлях/до/файлу"
            return "programmer_read", 0.98, self._read_file(arg)
        if low.startswith("/find"):
            arg = text[5:].strip()
            if not arg:
                return "programmer_find", 0.2, "Формат: /find текст_для_пошуку"
            return "programmer_find", 0.98, self._find_text(arg)
        if low.startswith("/tests"):
            return "programmer_tests", 0.98, self._run_tests()
        if low.startswith("/plan"):
            arg = text[5:].strip() or "Задача без уточнення"
            return "programmer_plan", 0.95, self._make_plan(arg)

        # Natural-language routing.
        if self._looks_like_list_request(low):
            path_hint = self._extract_path(text) or "."
            return "programmer_ls", 0.90, self._list_files(path_hint)

        if self._looks_like_read_request(low):
            path_hint = self._extract_path(text)
            if path_hint:
                return "programmer_read", 0.90, self._read_file(path_hint)
            return "programmer_read", 0.30, "Вкажи шлях до файлу, наприклад: /read modules/chat/entry.py"

        if self._looks_like_find_request(low):
            query = self._extract_find_query(text)
            if query:
                return "programmer_find", 0.90, self._find_text(query)
            return "programmer_find", 0.30, "Що саме шукати? Приклад: /find reply_programmer"

        if self._looks_like_tests_request(low):
            return "programmer_tests", 0.90, self._run_tests()

        if self._looks_like_plan_request(low):
            return "programmer_plan", 0.80, self._make_plan(text)

        if self._looks_like_code_request(low):
            return "programmer_code", 0.75, self._draft_code_answer(text)

        return (
            "programmer_help",
            0.40,
            "Я локальний програміст-помічник без зовнішніх API.\n"
            "Спробуй одну з команд:\n"
            "- /ls [path]\n"
            "- /read <file>\n"
            "- /find <text>\n"
            "- /tests\n"
            "- /plan <task>\n"
            "Або опиши задачу по коду конкретніше.",
        )

    def _help_text(self) -> str:
        return (
            "Режим локального програміста (без зовнішніх API):\n"
            "1. /ls [path] - список файлів\n"
            "2. /read <file> - прочитати файл\n"
            "3. /find <text> - пошук тексту в коді\n"
            "4. /tests - запустити тести\n"
            "5. /plan <task> - розбити задачу на кроки\n"
            "6. Пиши звичайною мовою: 'покажи файли', 'знайди reply', 'запусти тести'."
        )

    def _list_files(self, path_hint: str) -> str:
        target = self._resolve_path(path_hint)
        if target is None:
            return "Шлях поза робочою папкою або невалідний."
        if not target.exists():
            return f"Не знайдено: {path_hint}"
        if target.is_file():
            rel = self._rel(target)
            return f"Це файл: {rel}"

        out: list[str] = []
        count = 0
        max_items = 80
        for p in sorted(target.rglob("*")):
            if any(part in self._skip_dirs for part in p.parts):
                continue
            rel = self._rel(p)
            if p.is_dir():
                continue
            out.append(rel)
            count += 1
            if count >= max_items:
                break

        if not out:
            return f"Порожньо: {self._rel(target)}"

        head = f"Файли в {self._rel(target)} (перші {len(out)}):"
        return head + "\n" + "\n".join(f"- {x}" for x in out)

    def _read_file(self, path_hint: str) -> str:
        target = self._resolve_path(path_hint)
        if target is None:
            return "Шлях поза робочою папкою або невалідний."
        if not target.exists() or not target.is_file():
            return f"Файл не знайдено: {path_hint}"

        text = target.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        max_lines = 140
        clipped = lines[:max_lines]
        numbered = "\n".join(f"{i+1}: {line}" for i, line in enumerate(clipped))
        suffix = ""
        if len(lines) > max_lines:
            suffix = f"\n... (показано {max_lines} з {len(lines)} рядків)"
        return f"Файл: {self._rel(target)}\n{numbered}{suffix}"

    def _find_text(self, query: str) -> str:
        q = query.strip()
        if not q:
            return "Порожній запит пошуку."

        matches: list[str] = []
        max_hits = 40
        pattern = re.compile(re.escape(q), flags=re.IGNORECASE)

        for p in self.workspace_root.rglob("*"):
            if any(part in self._skip_dirs for part in p.parts):
                continue
            if not p.is_file():
                continue
            if p.suffix and p.suffix.lower() not in self._search_ext:
                continue

            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            for ln, line in enumerate(text.splitlines(), 1):
                if pattern.search(line):
                    snippet = line.strip()
                    if len(snippet) > 140:
                        snippet = snippet[:137] + "..."
                    matches.append(f"{self._rel(p)}:{ln} | {snippet}")
                    if len(matches) >= max_hits:
                        break
            if len(matches) >= max_hits:
                break

        if not matches:
            return f"Збігів не знайдено для: {q}"
        return "Збіги:\n" + "\n".join(f"- {m}" for m in matches)

    def _run_tests(self) -> str:
        cmd = ["python", "-m", "pytest", "-q"]
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.workspace_root),
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        except FileNotFoundError:
            return "Не можу запустити `python` у поточному середовищі."
        except subprocess.TimeoutExpired:
            return "Тести перевищили ліміт часу (120с)."

        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()

        if proc.returncode == 0:
            tail = out.splitlines()[-12:]
            return "Тести пройшли успішно.\n" + ("\n".join(tail) if tail else "(без виводу)")

        # pytest code 5 often means "no tests collected"
        if proc.returncode == 5:
            return "Pytest не знайшов тестів у цьому проєкті."

        merged = (out + "\n" + err).strip()
        tail = merged.splitlines()[-18:]
        return "Тести впали:\n" + "\n".join(tail)

    @staticmethod
    def _make_plan(task_text: str) -> str:
        return (
            "План задачі:\n"
            "1. Уточнити файл(и)/модуль і очікуваний результат.\n"
            "2. Знайти поточну реалізацію (`/find` + `/read`).\n"
            "3. Внести мінімальні зміни в код.\n"
            "4. Запустити перевірку (`/tests`).\n"
            "5. Показати короткий звіт змін.\n"
            f"\nЗадача: {task_text}"
        )

    @staticmethod
    def _draft_code_answer(task_text: str) -> str:
        return (
            "Прийняв coding-задачу. Щоб зробити її точно, дай:\n"
            "1. Шлях до файлу.\n"
            "2. Що має бути на вході/виході.\n"
            "3. Обмеження (Python версія, стиль, тести).\n"
            f"\nЧернетка запиту: {task_text}"
        )

    @staticmethod
    def _looks_like_list_request(low: str) -> bool:
        keys = ("покажи файли", "список файлів", "що в папці", "структура проєкту", "list files")
        return any(k in low for k in keys)

    @staticmethod
    def _looks_like_read_request(low: str) -> bool:
        keys = ("прочитай файл", "покажи файл", "відкрий файл", "що в файлі", "/read")
        return any(k in low for k in keys)

    @staticmethod
    def _looks_like_find_request(low: str) -> bool:
        keys = ("знайди", "пошукай", "search", "grep", "/find")
        return any(k in low for k in keys)

    @staticmethod
    def _looks_like_tests_request(low: str) -> bool:
        keys = ("запусти тести", "проганяй тести", "run tests", "/tests", "pytest")
        return any(k in low for k in keys)

    @staticmethod
    def _looks_like_plan_request(low: str) -> bool:
        keys = ("план", "розбий задачу", "кроки", "/plan")
        return any(k in low for k in keys)

    @staticmethod
    def _looks_like_code_request(low: str) -> bool:
        keys = ("напиши код", "зроби функцію", "рефактор", "додай", "виправ", "fix", "implement")
        return any(k in low for k in keys)

    @staticmethod
    def _extract_find_query(text: str) -> str:
        m = re.search(r"['\"](.+?)['\"]", text)
        if m:
            return m.group(1).strip()

        low = text.lower()
        for marker in ("знайди", "пошукай", "search", "grep", "/find"):
            pos = low.find(marker)
            if pos >= 0:
                tail = text[pos + len(marker) :].strip(" :")
                if tail:
                    return tail
        return ""

    @staticmethod
    def _extract_path(text: str) -> str | None:
        # Absolute Windows path, relative path or file name with extension.
        patterns = (
            r"([A-Za-z]:[\\/][^\s\"']+)",
            r"([./\\][^\s\"']+)",
            r"([A-Za-z0-9_\-/\\]+\.[A-Za-z0-9_]{1,10})",
        )
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                return m.group(1)
        return None

    def _resolve_path(self, raw_path: str) -> Path | None:
        p = Path(raw_path.strip().strip("\"'"))
        try:
            resolved = p.resolve() if p.is_absolute() else (self.workspace_root / p).resolve()
        except Exception:
            return None

        try:
            resolved.relative_to(self.workspace_root)
        except ValueError:
            return None
        return resolved

    def _rel(self, p: Path) -> str:
        try:
            return str(p.resolve().relative_to(self.workspace_root))
        except Exception:
            return str(p)
