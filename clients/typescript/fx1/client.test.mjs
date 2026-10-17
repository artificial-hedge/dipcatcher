import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";

const source = await readFile(new URL("./client.ts", import.meta.url), "utf8");
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ES2022, target: ts.ScriptTarget.ES2022 },
}).outputText;
const { HarnessApiClient, HarnessApiError, HarnessCompatError } = await import(
  `data:text/javascript;charset=utf-8,${encodeURIComponent(compiled)}`
);

test("version mismatch fails strict compatibility check", async () => {
  const client = new HarnessApiClient({
    fetch: async () => Response.json({ api_version: "99", fx1_version: "0.4.0" }),
  });

  await assert.rejects(client.checkCompat(), (error) => {
    assert.ok(error instanceof HarnessCompatError);
    assert.equal(error.report.compatible, false);
    assert.equal(error.report.serverApiVersion, "99");
    return true;
  });
});

test("keyed completion retries a declared transient refusal", async () => {
  const calls = [];
  const waits = [];
  const client = new HarnessApiClient({
    apiKey: "test-key",
    maxRetries: 1,
    sleep: async (ms) => waits.push(ms),
    fetch: async (url, init) => {
      calls.push({ url, init });
      return calls.length === 1
        ? Response.json({ detail: "busy" }, { status: 503, headers: { "Retry-After": "0" } })
        : Response.json({ id: "completion-1" });
    },
  });

  assert.deepEqual(await client.complete({ backend: "local_fx1" }, "request-1"), {
    id: "completion-1",
  });
  assert.equal(calls.length, 2);
  assert.deepEqual(waits, [0]);
  assert.ok(calls.every(({ url }) => url === "http://127.0.0.1:8011/harness/complete"));
  assert.ok(calls.every(({ init }) => init.headers["Idempotency-Key"] === "request-1"));
  assert.ok(calls.every(({ init }) => init.headers["X-API-Key"] === "test-key"));
});

test("unkeyed completion does not retry a transient refusal", async () => {
  let attempts = 0;
  const client = new HarnessApiClient({
    maxRetries: 2,
    fetch: async () => {
      attempts += 1;
      return Response.json(
        { detail: "busy", code: "overloaded" },
        { status: 503, headers: { "Retry-After": "0" } },
      );
    },
  });

  await assert.rejects(client.complete({ backend: "local_fx1" }), (error) => {
    assert.ok(error instanceof HarnessApiError);
    assert.equal(error.status, 503);
    assert.equal(error.code, "overloaded");
    return true;
  });
  assert.equal(attempts, 1);
});
