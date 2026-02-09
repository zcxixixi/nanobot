---
name: game-autoplay-demo
description: Generate stable Snake/Tetris autoplay terminal demos from simple natural language (Chinese/English), with concise result output.
metadata: {"nanobot":{"emoji":"🎮","always":true,"requires":{"bins":["opencode","python3"]}}}
---

# Game Autoplay Demo

Use this skill when the user says simple requests like:
- `我要玩贪吃蛇，但是要让他自己展示`
- `能不能做一个自己玩的俄罗斯方块`
- `make an auto-play snake/tetris demo`

## Required behavior

1. Prefer `exec` + `opencode run` to generate files quickly.
2. Default targets:
- Snake autoplay: `snake_bot.py`
- Tetris autoplay: `tetris_bot.py`
3. Keep output concise:
- Success: only report file + compile check result.
- Do not print full source code unless explicitly requested.
4. For curses apps:
- Do **not** run interactive game inside non-TTY tool environment.
- Validate with `python3 -m py_compile <file>` only.
5. UI quality constraints:
- ASCII-only rendering.
- Safe draw helpers (no out-of-bound writes).
- Show `terminal too small` message instead of crashing.
- Support `manual + auto` toggle key.

## Fast response format

- Success:
`已生成 + 编译通过`

- Failure:
`生成失败（exit code: N）` + last log lines.
