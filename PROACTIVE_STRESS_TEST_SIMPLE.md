# Proactive Stress Test (Simple Report)

## 1) Goal

We changed agent behavior to **outcome-first**:

- Finish the task first.
- Use tools when execution is actually needed.
- If model only “talks” for an execution-required task, force one retry.
- If still no tool call for execution-required task, return a clear failure status.

## 2) What Changed

File changed:

- `nanobot/agent/loop.py`

Main logic:

1. `prefer_tools`: action-oriented request => prefer tool route.
2. `require_tools`: request clearly needs real execution (run/test/list/create file/etc.).
3. Deflection detector: if model says “you run it manually”, trigger forced retry.
4. Fail-closed only when `require_tools=True` and still no tool call.

This avoids two bad cases:

- Fake success without execution evidence.
- Forcing tools for code-only / explanation-only requests.

## 3) Test Design

Stress test used a fake provider to simulate high-frequency requests.

### Case A: Execution required, model never calls tools
- Input example: “please fix and run tests in current directory”
- Expected: return `Status: fail`

### Case B: Execution required, first turn deflects, second turn calls tool
- Expected: forced retry happens, tool runs, final success

### Case C: Code-only request (explicitly says do not run/save)
- Expected: normal answer allowed, no forced tool call

## 4) Command

```bash
LOGURU_LEVEL=ERROR PYTHONPATH=$PWD .venv/bin/python <stress-script>
```

Also validated all tests:

```bash
PYTHONPATH=$PWD .venv/bin/pytest -q
```

## 5) Results

### Functional tests

- `20 passed`

### Stress test

- Iterations per case: `2000`
- Total cases: `6000`
- Elapsed: `9.68s`
- Throughput: `619.72 cases/s`

Per-case pass rate:

- Case A: `2000/2000` (100%)
- Case B: `2000/2000` (100%)
- Case C: `2000/2000` (100%)

Average model-call count:

- Case A: `2.00` (first reply + forced retry)
- Case B: `3.00` (deflect -> tool call -> final)
- Case C: `1.00` (direct answer)

## 6) Plain Conclusion

The new policy works:

- It stops “talk-only pretending” for real execution tasks.
- It still stays flexible for requests that do not need tool execution.
- It remains fast and stable under load.
