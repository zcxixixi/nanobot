from nanobot.config.loader import (
    camel_to_snake,
    convert_keys,
    convert_to_camel,
    snake_to_camel,
)


def test_camel_to_snake_basic() -> None:
    assert camel_to_snake("apiKey") == "api_key"
    assert camel_to_snake("workspacePath") == "workspace_path"


def test_snake_to_camel_basic() -> None:
    assert snake_to_camel("api_key") == "apiKey"
    assert snake_to_camel("workspace_path") == "workspacePath"


def test_convert_keys_nested() -> None:
    data = {
        "apiKey": "xxx",
        "agents": {"maxToolIterations": 5},
        "channels": [{"allowFrom": ["123"]}],
    }
    out = convert_keys(data)
    assert out["api_key"] == "xxx"
    assert out["agents"]["max_tool_iterations"] == 5
    assert out["channels"][0]["allow_from"] == ["123"]


def test_convert_to_camel_nested() -> None:
    data = {
        "api_key": "xxx",
        "agents": {"max_tool_iterations": 5},
        "channels": [{"allow_from": ["123"]}],
    }
    out = convert_to_camel(data)
    assert out["apiKey"] == "xxx"
    assert out["agents"]["maxToolIterations"] == 5
    assert out["channels"][0]["allowFrom"] == ["123"]
