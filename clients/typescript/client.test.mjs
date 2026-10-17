import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";

const source = await readFile(new URL("./client.ts", import.meta.url), "utf8");
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ES2022, target: ts.ScriptTarget.ES2022 },
}).outputText;
const { ResearchApiClient, ResearchApiError } = await import(
  `data:text/javascript;charset=utf-8,${encodeURIComponent(compiled)}`
);

test("run listing preserves query, auth, and the response body", async () => {
  const calls = [];
  const body = { items: [], total: 0 };
  const client = new ResearchApiClient({
    baseUrl: "http://127.0.0.1:8010/",
    apiKey: "test-key",
    fetch: async (url, init) => {
      calls.push({ url, init });
      return Response.json(body);
    },
  });

  assert.deepEqual(await client.listRuns({ limit: 3, offset: 0 }), body);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, "http://127.0.0.1:8010/runs?limit=3&offset=0");
  assert.deepEqual(calls[0].init.headers, {
    Accept: "application/json",
    "X-API-Key": "test-key",
  });
});

test("path identifiers are encoded and API faults retain their status", async () => {
  let requested;
  const client = new ResearchApiClient({
    fetch: async (url) => {
      requested = url;
      return Response.json({ detail: "missing receipt" }, { status: 404 });
    },
  });

  await assert.rejects(client.getReceipt("a/b ?"), (error) => {
    assert.ok(error instanceof ResearchApiError);
    assert.equal(error.status, 404);
    assert.equal(error.message, "missing receipt");
    return true;
  });
  assert.equal(requested, "http://127.0.0.1:8010/receipts/a%2Fb%20%3F");
});
