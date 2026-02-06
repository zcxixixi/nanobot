import builtins

from nanobot.cli.commands import _read_interactive_input


def test_read_interactive_input_passes_prompt_to_input(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def fake_input(prompt: str = "") -> str:
        captured["prompt"] = prompt
        return "hello"

    monkeypatch.setattr(builtins, "input", fake_input)

    value = _read_interactive_input()

    assert value == "hello"
    assert captured["prompt"] == "You: "
