import builtins

import nanobot.cli.commands as commands


def test_read_interactive_input_plain_prompt_when_no_readline(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def fake_input(prompt: str = "") -> str:
        captured["prompt"] = prompt
        return "hello"

    monkeypatch.setattr(commands, "_READLINE", None)
    monkeypatch.setattr(builtins, "input", fake_input)

    value = commands._read_interactive_input()

    assert value == "hello"
    assert captured["prompt"] == "You: "


def test_read_interactive_input_colored_prompt_with_readline(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def fake_input(prompt: str = "") -> str:
        captured["prompt"] = prompt
        return "hello"

    monkeypatch.setattr(commands, "_READLINE", object())
    monkeypatch.setattr(builtins, "input", fake_input)

    value = commands._read_interactive_input()

    assert value == "hello"
    assert "You:" in captured["prompt"]
    assert "\x1b[" in captured["prompt"]


def test_remember_input_history_dedupes_adjacent(monkeypatch) -> None:
    class FakeReadline:
        def __init__(self) -> None:
            self.items: list[str] = []

        def get_current_history_length(self) -> int:
            return len(self.items)

        def get_history_item(self, index: int) -> str | None:
            if index <= 0 or index > len(self.items):
                return None
            return self.items[index - 1]

        def add_history(self, text: str) -> None:
            self.items.append(text)

    fake = FakeReadline()
    monkeypatch.setattr(commands, "_READLINE", fake)

    commands._remember_input_history("abc")
    commands._remember_input_history("abc")
    commands._remember_input_history("xyz")

    assert fake.items == ["abc", "xyz"]
