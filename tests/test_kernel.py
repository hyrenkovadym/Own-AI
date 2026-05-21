from core import Kernel, ModuleSpec


def _build_kernel() -> Kernel:
    specs = [
        ModuleSpec(name="chat", entrypoint="modules.chat.entry"),
        ModuleSpec(name="snake", entrypoint="modules.snake.entry"),
    ]
    return Kernel(specs=specs)


def test_kernel_lists_modules_and_status() -> None:
    kernel = _build_kernel()

    assert kernel.list_modules() == ["chat", "snake"]
    statuses = kernel.list_module_status()
    assert [s.name for s in statuses] == ["chat", "snake"]
    assert all(not s.loaded for s in statuses)


def test_kernel_load_and_unload_module() -> None:
    kernel = _build_kernel()

    chat_module = kernel.load("chat")
    assert getattr(chat_module, "name", None) == "chat"
    assert kernel.is_loaded("chat")

    assert kernel.unload("chat") is True
    assert kernel.is_loaded("chat") is False
    assert kernel.unload("chat") is False
