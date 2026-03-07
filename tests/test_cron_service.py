import pytest

from nanobot.cron.service import CronService
from nanobot.cron.types import CronSchedule


def test_add_job_rejects_unknown_timezone(tmp_path) -> None:
    service = CronService(tmp_path / "cron" / "jobs.json")

    with pytest.raises(ValueError, match="unknown timezone 'America/Vancovuer'"):
        service.add_job(
            name="tz typo",
            schedule=CronSchedule(kind="cron", expr="0 9 * * *", tz="America/Vancovuer"),
            message="hello",
        )

    assert service.list_jobs(include_disabled=True) == []


def test_add_job_accepts_valid_timezone(tmp_path) -> None:
    service = CronService(tmp_path / "cron" / "jobs.json")

    job = service.add_job(
        name="tz ok",
        schedule=CronSchedule(kind="cron", expr="0 9 * * *", tz="America/Vancouver"),
        message="hello",
    )

    assert job.schedule.tz == "America/Vancouver"
    assert job.state.next_run_at_ms is not None


def test_add_command_job_persists_direct_command_payload(tmp_path) -> None:
    service = CronService(tmp_path / "cron" / "jobs.json")

    job = service.add_command_job(
        name="sync assets",
        schedule=CronSchedule(kind="every", every_ms=60000),
        argv=["python3", "scripts/asset_pipeline.py", "run-cycle"],
        cwd="/tmp/demo",
        env={"FOO": "bar"},
        timeout_s=30,
    )

    reloaded = CronService(tmp_path / "cron" / "jobs.json").list_jobs(include_disabled=True)[0]
    assert job.payload.kind == "command"
    assert reloaded.payload.kind == "command"
    assert reloaded.payload.argv == ["python3", "scripts/asset_pipeline.py", "run-cycle"]
    assert reloaded.payload.cwd == "/tmp/demo"
    assert reloaded.payload.env == {"FOO": "bar"}
    assert reloaded.payload.timeout_s == 30
