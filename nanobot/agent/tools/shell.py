"""Shell execution tool."""

import asyncio
import os
import re
import sys
from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool


class ExecTool(Tool):
    """Tool to execute shell commands."""
    
    def __init__(
        self,
        timeout: int = 60,
        working_dir: str | None = None,
        deny_patterns: list[str] | None = None,
        allow_patterns: list[str] | None = None,
        restrict_to_workspace: bool = False,
    ):
        self.timeout = timeout
        self.working_dir = working_dir
        self.deny_patterns = deny_patterns or [
            r"\brm\s+-[rf]{1,2}\b",          # rm -r, rm -rf, rm -fr
            r"\bdel\s+/[fq]\b",              # del /f, del /q
            r"\brmdir\s+/s\b",               # rmdir /s
            r"\b(format|mkfs|diskpart)\b",   # disk operations
            r"\bdd\s+if=",                   # dd
            r">\s*/dev/sd",                  # write to disk
            r"\b(shutdown|reboot|poweroff)\b",  # system power
            r":\(\)\s*\{.*\};\s*:",          # fork bomb
        ]
        self.allow_patterns = allow_patterns or []
        self.restrict_to_workspace = restrict_to_workspace
    
    @property
    def name(self) -> str:
        return "exec"
    
    @property
    def description(self) -> str:
        return "Execute a shell command and return its output. Use with caution."
    
    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                },
                "working_dir": {
                    "type": "string",
                    "description": "Optional working directory for the command"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Optional timeout override in seconds for this command"
                }
            },
            "required": ["command"]
        }

    async def execute(
        self,
        command: str,
        working_dir: str | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> str:
        cwd = working_dir or self.working_dir or os.getcwd()
        exec_timeout = timeout if isinstance(timeout, int) and timeout > 0 else self.timeout
        guard_error = self._guard_command(command, cwd)
        if guard_error:
            return guard_error
        
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )

            stdout_parts: list[str] = []
            stderr_parts: list[str] = []
            stream_live = self._stream_live_enabled()

            stdout_task = asyncio.create_task(
                self._drain_stream(process.stdout, stdout_parts, stream_live, to_stderr=False)
            )
            stderr_task = asyncio.create_task(
                self._drain_stream(process.stderr, stderr_parts, stream_live, to_stderr=True)
            )

            timed_out = False
            try:
                await asyncio.wait_for(process.wait(), timeout=exec_timeout)
            except asyncio.TimeoutError:
                timed_out = True
                process.kill()
                await process.wait()

            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)

            output_parts = []
            stdout_text = "".join(stdout_parts)
            stderr_text = "".join(stderr_parts)

            if stdout_text:
                output_parts.append(stdout_text)
            if stderr_text.strip():
                output_parts.append(f"STDERR:\n{stderr_text}")
            if process.returncode != 0:
                output_parts.append(f"\nExit code: {process.returncode}")

            if timed_out:
                output_parts.append(f"\nError: Command timed out after {exec_timeout} seconds")

            result = "\n".join(output_parts) if output_parts else "(no output)"

            # Truncate very long output
            max_len = 10000
            if len(result) > max_len:
                result = result[:max_len] + f"\n... (truncated, {len(result) - max_len} more chars)"

            return result

        except Exception as e:
            return f"Error executing command: {str(e)}"

    @staticmethod
    def _stream_live_enabled() -> bool:
        """
        Enable live streaming by default on TTY, configurable via NANOBOT_EXEC_STREAM.
        """
        flag = os.getenv("NANOBOT_EXEC_STREAM", "1").strip().lower()
        if flag in {"0", "false", "no", "off"}:
            return False
        return sys.stdout.isatty()

    async def _drain_stream(
        self,
        stream: asyncio.StreamReader | None,
        collector: list[str],
        stream_live: bool,
        *,
        to_stderr: bool,
    ) -> None:
        if stream is None:
            return

        while True:
            chunk = await stream.readline()
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace")
            collector.append(text)
            if stream_live:
                target = sys.stderr if to_stderr else sys.stdout
                target.write(text)
                target.flush()

    def _guard_command(self, command: str, cwd: str) -> str | None:
        """Best-effort safety guard for potentially destructive commands."""
        cmd = command.strip()
        lower = cmd.lower()

        for pattern in self.deny_patterns:
            if re.search(pattern, lower):
                return "Error: Command blocked by safety guard (dangerous pattern detected)"

        if self.allow_patterns:
            if not any(re.search(p, lower) for p in self.allow_patterns):
                return "Error: Command blocked by safety guard (not in allowlist)"

        if self.restrict_to_workspace:
            if "..\\" in cmd or "../" in cmd:
                return "Error: Command blocked by safety guard (path traversal detected)"

            cwd_path = Path(cwd).resolve()

            win_paths = re.findall(r"[A-Za-z]:\\[^\\\"']+", cmd)
            posix_paths = re.findall(r"/[^\s\"']+", cmd)

            for raw in win_paths + posix_paths:
                try:
                    p = Path(raw).resolve()
                except Exception:
                    continue
                if cwd_path not in p.parents and p != cwd_path:
                    return "Error: Command blocked by safety guard (path outside working dir)"

        return None
