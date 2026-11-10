# Feedback task starting environment

Upstream: https://github.com/msgpack/msgpack-python
Upstream commit: `8a88e3df6b7663fc72d03aa63ab9c75b8b62fabd`
Prepared on macOS with CPython 3.12.11. No external database, queue, or cloud account is required.
The dependency file pins the installed runtime and test packages for the selected baseline. It is not a claim that all optional upstream test suites are installed.

From the repository root:

```sh
export NO_INTERNET=1 TZ=UTC MSGPACK_PUREPYTHON=1
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r .feedback/requirements.txt -e .
.venv/bin/python -m pytest -q -o addopts= test
```

Scope note: The scope is the command below, not all optional upstream integrations.

Observed baseline (2026-11-10): `175 passed, 4 skipped, 3 warnings in 1.50s`
Whole-command wall time: 1.828 seconds, exit code 0. This is one local measurement, not a runtime guarantee. The first install is excluded.

This seed adds preparation metadata only. Application sources and upstream tests are unchanged; there is no task solution in this commit.
