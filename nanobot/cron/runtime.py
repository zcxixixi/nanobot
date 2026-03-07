"""Cron execution helpers."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING

from nanobot.bus.events import OutboundMessage

if TYPE_CHECKING:
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus
    from nanobot.cron.types import CronJob


async def run_command_payload(job: "CronJob") -> str:
    """Execute a direct command payload and return combined output."""
    argv = job.payload.argv or []
    if not argv:
        raise ValueError("command cron job requires payload.argv")

    env = os.environ.copy()
    env.update(job.payload.env or {})
    cwd = job.payload.cwd or None
    if cwd:
        Path(cwd).expanduser().resolve(strict=False)

    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=cwd,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    timeout_s = job.payload.timeout_s or None
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError as exc:
        process.kill()
        await process.communicate()
        raise TimeoutError(
            f"command timed out after {timeout_s}s: {' '.join(argv)}"
        ) from exc

    stdout_text = stdout.decode("utf-8", "ignore").strip()
    stderr_text = stderr.decode("utf-8", "ignore").strip()
    output = "\n".join(part for part in (stdout_text, stderr_text) if part).strip()

    if process.returncode != 0:
        tail = output[-1500:] if output else "(no output)"
        raise RuntimeError(
            f"command exited with {process.returncode}: {' '.join(argv)}\n{tail}"
        )

    return output


async def execute_cron_job(
    job: "CronJob",
    *,
    agent: "AgentLoop | None" = None,
    bus: "MessageBus | None" = None,
) -> str | None:
    """Dispatch a cron job to either direct command execution or the agent loop."""
    if job.payload.kind == "command":
        response = await run_command_payload(job)
    else:
        if agent is None:
            raise ValueError("agent is required for agent_turn cron jobs")
        reminder_note = (
            "[Scheduled Task] Timer finished.\n\n"
            f"Task '{job.name}' has been triggered.\n"
            f"Scheduled instruction: {job.payload.message}"
        )
        response = await agent.process_direct(
            reminder_note,
            session_key=f"cron:{job.id}",
            channel=job.payload.channel or "cli",
            chat_id=job.payload.to or "direct",
        )

    if job.payload.deliver and job.payload.to and response and bus is not None:
        await bus.publish_outbound(
            OutboundMessage(
                channel=job.payload.channel or "cli",
                chat_id=job.payload.to,
                content=response,
            )
        )

    return response
