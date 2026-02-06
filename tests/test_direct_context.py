from nanobot.agent.loop import AgentLoop


def test_resolve_direct_context_uses_session_key_channel_and_chat() -> None:
    channel, chat_id = AgentLoop._resolve_direct_context("telegram:12345", "cli", "direct")
    assert channel == "telegram"
    assert chat_id == "12345"


def test_resolve_direct_context_keeps_explicit_context() -> None:
    channel, chat_id = AgentLoop._resolve_direct_context("telegram:12345", "cron", "job-1")
    assert channel == "cron"
    assert chat_id == "job-1"


def test_resolve_direct_context_supports_shorthand_session_key() -> None:
    channel, chat_id = AgentLoop._resolve_direct_context("my-session", "cli", "direct")
    assert channel == "cli"
    assert chat_id == "my-session"
