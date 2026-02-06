import builtins

import pytest

import nanobot.cli.commands as commands


def test_read_interactive_input_prefers_prompt_session(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakePromptSession:
        def prompt(self, label: object) -> str:
            captured["label"] = label
            return "hello"

    monkeypatch.setattr(commands, "_PROMPT_SESSION", FakePromptSession())
    monkeypatch.setattr(commands, "_PROMPT_SESSION_LABEL", "LABEL")

    value = commands._read_interactive_input()

    assert value == "hello"
    assert captured["label"] == "LABEL"


def test_read_interactive_input_converts_prompt_session_eof(monkeypatch) -> None:
    class FakePromptSession:
        def prompt(self, _label: object) -> str:
            raise EOFError

    monkeypatch.setattr(commands, "_PROMPT_SESSION", FakePromptSession())
    monkeypatch.setattr(commands, "_PROMPT_SESSION_LABEL", "LABEL")

    try:
        commands._read_interactive_input()
    except KeyboardInterrupt:
        assert True
    else:
        assert False, "EOF should be converted to KeyboardInterrupt"


@pytest.mark.asyncio
async def test_read_interactive_input_async_prefers_prompt_session(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakePromptSession:
        async def prompt_async(self, label: object) -> str:
            captured["label"] = label
            return "hello"

    monkeypatch.setattr(commands, "_PROMPT_SESSION", FakePromptSession())
    monkeypatch.setattr(commands, "_PROMPT_SESSION_LABEL", "LABEL")

    value = await commands._read_interactive_input_async()

    assert value == "hello"
    assert captured["label"] == "LABEL"


@pytest.mark.asyncio
async def test_read_interactive_input_async_converts_prompt_session_eof(monkeypatch) -> None:
    class FakePromptSession:
        async def prompt_async(self, _label: object) -> str:
            raise EOFError

    monkeypatch.setattr(commands, "_PROMPT_SESSION", FakePromptSession())
    monkeypatch.setattr(commands, "_PROMPT_SESSION_LABEL", "LABEL")

    with pytest.raises(KeyboardInterrupt):
        await commands._read_interactive_input_async()


def test_read_interactive_input_plain_prompt_when_no_readline(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def fake_input(prompt: str = "") -> str:
        captured["prompt"] = prompt
        return "hello"

    monkeypatch.setattr(commands, "_PROMPT_SESSION", None)
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

    monkeypatch.setattr(commands, "_PROMPT_SESSION", None)
    monkeypatch.setattr(commands, "_READLINE", object())
    monkeypatch.setattr(commands, "_USING_LIBEDIT", False)
    monkeypatch.setattr(builtins, "input", fake_input)

    value = commands._read_interactive_input()

    assert value == "hello"
    assert "You:" in captured["prompt"]
    assert "\x1b[" in captured["prompt"]


def test_read_interactive_input_colored_prompt_with_libedit(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def fake_input(prompt: str = "") -> str:
        captured["prompt"] = prompt
        return "hello"

    monkeypatch.setattr(commands, "_PROMPT_SESSION", None)
    monkeypatch.setattr(commands, "_READLINE", object())
    monkeypatch.setattr(commands, "_USING_LIBEDIT", True)
    monkeypatch.setattr(builtins, "input", fake_input)

    value = commands._read_interactive_input()

    assert value == "hello"
    assert captured["prompt"] == "\x1b[1;34mYou:\x1b[0m "


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
