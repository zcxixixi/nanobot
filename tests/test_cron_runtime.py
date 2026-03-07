from types import SimpleNamespace

import pytest

from nanobot.cron.runtime import execute_cron_job
from nanobot.cron.types import CronJob, CronPayload, CronSchedule


class _FakeBus:
    def __init__(self) -> None:
        self.messages = []

    async def publish_outbound(self, message) -> None:
        self.messages.append(message)


class _ExplodingAgent:
    async def process_direct(self, *_args, **_kwargs):
        raise AssertionError("agent loop should not be used for command cron jobs")


@pytest.mark.asyncio
async def test_execute_cron_job_uses_direct_command_runner(monkeypatch) -> None:
    observed = {}

    async def fake_runner(job):
        observed["argv"] = job.payload.argv
        return "pipeline complete"

    monkeypatch.setattr("nanobot.cron.runtime.run_command_payload", fake_runner)

    job = CronJob(
        id="job-1",
        name="sync",
        schedule=CronSchedule(kind="every", every_ms=1000),
        payload=CronPayload(kind="command", argv=["python3", "scripts/asset_pipeline.py", "run-cycle"]),
    )

    response = await execute_cron_job(job, agent=_ExplodingAgent(), bus=_FakeBus())

    assert response == "pipeline complete"
    assert observed["argv"] == ["python3", "scripts/asset_pipeline.py", "run-cycle"]


@pytest.mark.asyncio
async def test_execute_cron_job_delivers_command_output(monkeypatch) -> None:
    async def fake_runner(_job):
        return "done"

    monkeypatch.setattr("nanobot.cron.runtime.run_command_payload", fake_runner)

    bus = _FakeBus()
    job = CronJob(
        id="job-2",
        name="notify",
        schedule=CronSchedule(kind="every", every_ms=1000),
        payload=CronPayload(
            kind="command",
            argv=["echo", "done"],
            deliver=True,
            channel="telegram",
            to="12345",
        ),
    )

    await execute_cron_job(job, bus=bus)

    assert len(bus.messages) == 1
    assert bus.messages[0].channel == "telegram"
    assert bus.messages[0].chat_id == "12345"
    assert bus.messages[0].content == "done"
