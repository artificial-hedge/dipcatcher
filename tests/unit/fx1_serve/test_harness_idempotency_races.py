"""SYNTHETIC scheduling barriers for the real harness HTTP handlers.

Only the command runner and the first job publication are held by events;
request validation, replay lookups, admission, and storage use the real app.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import Any

import anyio
import httpx
import pytest

from fx1.harness import Harness
from fx1.serve import api


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("FX1_API_KEY", "FX1_API_STATE_DIR", "FX1_API_CORS_ORIGINS"):
        monkeypatch.delenv(name, raising=False)


async def _event(event: threading.Event, timeout: float = 2.0) -> None:
    with anyio.fail_after(timeout):
        while not event.is_set():
            await anyio.sleep(0.005)


def _app(runner: Callable[[list[str], int], tuple[int, str, str]]) -> Any:
    return api.create_app(harness=Harness(runner=runner), max_inflight=32, rate_limit_rps=0)


@pytest.mark.parametrize("key_channel", ["header", "body"])
def test_concurrent_runs_execute_once(key_channel: str) -> None:
    entered, release = threading.Event(), threading.Event()
    calls: list[list[str]] = []

    def runner(argv: list[str], _timeout: int) -> tuple[int, str, str]:
        calls.append(argv)
        entered.set()
        assert release.wait(3), "test did not release runner"
        return 0, "one execution", ""

    async def exercise() -> None:
        app = _app(runner)
        body = {"command": "doctor"}
        headers = {}
        if key_channel == "header":
            headers["Idempotency-Key"] = "same"
        else:
            body["idempotency_key"] = "same"
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://localhost"
            ) as client,
        ):
            first = asyncio.create_task(client.post("/harness/runs", json=body, headers=headers))
            await _event(entered)
            second = asyncio.create_task(client.post("/harness/runs", json=body, headers=headers))
            try:
                await anyio.sleep(0.12)
            finally:
                release.set()
            responses = await asyncio.gather(first, second)
            assert [r.status_code for r in responses] == [200, 200]
            assert len(calls) == 1
            assert sorted(r.json()["replayed"] for r in responses) == [False, True]

    asyncio.run(exercise())


@pytest.mark.parametrize("contender_is_batch", [False, True])
def test_concurrent_job_submissions_return_one_job(
    monkeypatch: pytest.MonkeyPatch, contender_is_batch: bool
) -> None:
    publishing, release = threading.Event(), threading.Event()
    publications: list[str] = []
    calls: list[list[str]] = []
    original_put = api._JobStore.put

    def held_put(self: Any, job: Any, key: str | None, fp: str) -> None:
        publications.append(job.job_id)
        if len(publications) == 1:
            publishing.set()
            assert release.wait(3), "test did not release first publication"
        original_put(self, job, key, fp)

    def runner(argv: list[str], _timeout: int) -> tuple[int, str, str]:
        calls.append(argv)
        return 0, "ok", ""

    monkeypatch.setattr(api._JobStore, "put", held_put)

    async def exercise() -> None:
        app = _app(runner)
        body = {"command": "doctor", "idempotency_key": "job-key"}
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://localhost"
            ) as client,
        ):
            first = asyncio.create_task(client.post("/harness/jobs", json=body))
            await _event(publishing)
            if contender_is_batch:
                second = asyncio.create_task(
                    client.post("/harness/jobs/batch", json={"jobs": [body]})
                )
            else:
                second = asyncio.create_task(client.post("/harness/jobs", json=body))
            try:
                await anyio.sleep(0.12)
            finally:
                release.set()
            a, b = await asyncio.gather(first, second)
            assert (a.status_code, b.status_code) == (202, 202)
            second_item = b.json()["jobs"][0] if contender_is_batch else b.json()
            assert a.json()["job_id"] == second_item["job_id"]
            assert a.json()["replayed"] is False
            assert second_item["replayed"] is True
            assert len(publications) == len(calls) == 1
            assert a.headers["location"] == f"/harness/jobs/{a.json()['job_id']}"

    asyncio.run(exercise())


def test_changed_body_conflicts_after_concurrent_leader_finishes() -> None:
    entered, release = threading.Event(), threading.Event()
    calls: list[list[str]] = []

    def runner(argv: list[str], _timeout: int) -> tuple[int, str, str]:
        calls.append(argv)
        entered.set()
        assert release.wait(3)
        return 0, "ok", ""

    async def exercise() -> None:
        app = _app(runner)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://localhost"
            ) as client,
        ):
            a = asyncio.create_task(
                client.post(
                    "/harness/runs", json={"command": "doctor", "idempotency_key": "conflict"}
                )
            )
            await _event(entered)
            b = asyncio.create_task(
                client.post(
                    "/harness/runs",
                    json={
                        "command": "doctor",
                        "extra_args": ["--help"],
                        "idempotency_key": "conflict",
                    },
                )
            )
            try:
                await anyio.sleep(0.12)
            finally:
                release.set()
            first, second = await asyncio.gather(a, b)
            assert (first.status_code, second.status_code) == (200, 409)
            assert len(calls) == 1

    asyncio.run(exercise())


@pytest.mark.parametrize("keys", [("a", "b"), (None, None)])
def test_distinct_or_missing_keys_remain_parallel(keys: tuple[str | None, str | None]) -> None:
    both, release = threading.Event(), threading.Event()
    calls: list[list[str]] = []

    def runner(argv: list[str], _timeout: int) -> tuple[int, str, str]:
        calls.append(argv)
        if len(calls) == 2:
            both.set()
        assert release.wait(3)
        return 0, "ok", ""

    async def exercise() -> None:
        app = _app(runner)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://localhost"
            ) as client,
        ):
            tasks = [
                asyncio.create_task(
                    client.post("/harness/runs", json={"command": "doctor", "idempotency_key": key})
                )
                for key in keys
            ]
            try:
                await _event(both)
            finally:
                release.set()
            responses = await asyncio.gather(*tasks)
            assert [r.status_code for r in responses] == [200, 200]
            assert len(calls) == 2

    asyncio.run(exercise())


def test_waiting_retries_do_not_starve_control_plane() -> None:
    entered, release = threading.Event(), threading.Event()
    calls: list[list[str]] = []

    def runner(argv: list[str], _timeout: int) -> tuple[int, str, str]:
        calls.append(argv)
        entered.set()
        assert release.wait(3)
        return 0, "ok", ""

    async def exercise() -> None:
        limiter = anyio.to_thread.current_default_thread_limiter()
        limiter.total_tokens = 2
        app = _app(runner)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://localhost"
            ) as client,
        ):
            body = {"command": "doctor", "idempotency_key": "busy"}
            first = asyncio.create_task(client.post("/harness/runs", json=body))
            await _event(entered)
            retries = [
                asyncio.create_task(client.post("/harness/runs", json=body)) for _ in range(8)
            ]
            try:
                await anyio.sleep(0.05)
                with anyio.fail_after(1):
                    response = await client.get("/harness/commands")
                assert response.status_code == 200
            finally:
                release.set()
            responses = await asyncio.gather(first, *retries)
            assert all(r.status_code == 200 for r in responses)
            assert len(calls) == 1

    asyncio.run(exercise())


@pytest.mark.parametrize("route", ["/harness/runs", "/harness/jobs"])
def test_header_precedence_and_failed_lookup_release_claim(route: str) -> None:
    calls: list[list[str]] = []

    def runner(argv: list[str], _timeout: int) -> tuple[int, str, str]:
        calls.append(argv)
        return 0, "ok", ""

    async def exercise() -> None:
        app = _app(runner)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://localhost"
            ) as client,
        ):
            bad = await client.post(route, json={"command": "unknown", "idempotency_key": "reuse"})
            assert bad.status_code == 404
            good = await client.post(route, json={"command": "doctor", "idempotency_key": "reuse"})
            assert good.status_code == (202 if route.endswith("jobs") else 200)
            replay = await client.post(
                route, json={"command": "doctor", "idempotency_key": "reuse"}
            )
            assert replay.json()["replayed"] is True
            # A truthy whitespace header has always suppressed the body key.
            fresh = await client.post(
                route,
                json={"command": "doctor", "idempotency_key": "reuse"},
                headers={"Idempotency-Key": " "},
            )
            assert fresh.json()["replayed"] is False
            long_key = await client.post(
                route, json={"command": "doctor"}, headers={"Idempotency-Key": "x" * 257}
            )
            assert long_key.status_code == 400

    asyncio.run(exercise())
