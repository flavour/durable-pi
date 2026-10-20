import { test } from "node:test";
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { mkdir, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Store } from "../src/store.js";
import { LLAMA_PROVIDER, llamaModels } from "../src/llamacpp.js";
import { loadConfig } from "../src/config.js";

/** A stand-in for `llama-server`: the OpenAI-compatible chat endpoint plus llama.cpp's own /props. */
async function fakeLlamaServer() {
  const seen: { path: string; auth?: string; body?: Record<string, unknown> }[] = [];
  const server = createServer((req, res) => {
    const path = new URL(req.url ?? "/", "http://127.0.0.1");
    const auth = req.headers.authorization;
    if (path.pathname === "/v1/models") {
      res.writeHead(200, { "content-type": "application/json" }).end(
        JSON.stringify({
          data: [
            {
              id: "fake.gguf",
              status: { value: "loaded" },
              meta: { n_ctx: 32000 },
              architecture: { input_modalities: ["text"], output_modalities: ["text"] },
            },
            // A preset the router has not loaded: listing it would load it into a context nobody chose.
            { id: "sleeping-in-the-router.gguf", status: { value: "unloaded" }, meta: { n_ctx: 8192 } },
          ],
        }),
      );
      return;
    }
    if (path.pathname === "/props") {
      seen.push({ path: `${path.pathname}?${path.searchParams.get("model")}`, auth });
      res.writeHead(200, { "content-type": "application/json" }).end(
        JSON.stringify({ chat_template: "… {% if enable_thinking %} …" }),
      );
      return;
    }
    if (path.pathname === "/v1/chat/completions") {
      let body = "";
      req.on("data", (chunk) => (body += chunk));
      req.on("end", () => {
        seen.push({ path: path.pathname, auth, body: JSON.parse(body) as Record<string, unknown> });
        res.writeHead(200, { "content-type": "text/event-stream" });
        res.write(`data: ${JSON.stringify({ choices: [{ index: 0, delta: { role: "assistant", content: "from the local model" } }] })}\n\n`);
        res.write(`data: ${JSON.stringify({ choices: [{ index: 0, delta: {}, finish_reason: "stop" }], usage: { prompt_tokens: 5, completion_tokens: 5 } })}\n\n`);
        res.write("data: [DONE]\n\n");
        res.end();
      });
      return;
    }
    res.writeHead(404).end(JSON.stringify({ error: { message: "not found" } }));
  });
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  const port = (server.address() as { port: number }).port;
  return { url: `http://127.0.0.1:${port}`, seen, close: () => new Promise<void>((r) => server.close(() => r())) };
}

test("DPI_PROVIDER accepts the llama.cpp spellings, and the server URL honours LLAMA_BASE_URL", () => {
  const token = "x".repeat(40);
  for (const value of ["llamacpp", "llama.cpp", "llama-cpp", "local", "LLAMA.CPP"]) {
    const config = loadConfig(["--env-file", "/nonexistent.env"], { DPI_TOKEN: token, DPI_PROVIDER: value });
    assert.equal(config.provider, "llamacpp", `DPI_PROVIDER=${value}`);
  }
  const defaults = loadConfig(["--env-file", "/nonexistent.env"], { DPI_TOKEN: token });
  assert.equal(defaults.provider, "openrouter");
  assert.equal(defaults.llama.url, "http://127.0.0.1:8080");
  const configured = loadConfig(["--env-file", "/nonexistent.env"], {
    DPI_TOKEN: token,
    DPI_PROVIDER: "llamacpp",
    LLAMA_BASE_URL: "http://forge:9931/v1",
    LLAMA_API_KEY: "secret",
  });
  assert.equal(configured.llama.url, "http://forge:9931/v1");
  assert.equal(configured.llama.apiKey, "secret");
});

test("a llama.cpp server's own models become the device's models, costed at zero", async () => {
  const llama = await fakeLlamaServer();
  try {
    const models = await llamaModels({ url: `${llama.url}/v1` }, AbortSignal.timeout(5000));
    // Only loaded models: an unloaded router preset would be loaded by the request that listed it.
    assert.equal(models.length, 1);
    const [model] = models;
    assert.equal(model.provider, LLAMA_PROVIDER);
    assert.equal(model.id, "fake.gguf");
    assert.equal(model.baseUrl, `${llama.url}/v1`); // the /v1 suffix is implied by the api, not doubled
    assert.equal(model.contextWindow, 32000);
    assert.equal(model.reasoning, true); // /props reported a chat template that switches thinking on
    assert.deepEqual(model.cost, { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 });
    assert.equal(model.maxTokens, 32000);
  } finally {
    await llama.close();
  }
});

test("a device with provider=llamacpp answers a durable session from its own llama.cpp server", async () => {
  const llama = await fakeLlamaServer();
  const root = await mkdtemp(join(tmpdir(), "dpi-llama-"));
  try {
    await mkdir(join(root, "work"), { recursive: true });
    const store = await Store.open({
      dataDir: join(root, "data"),
      startDirs: [join(root, "work")],
      name: "test",
      provider: "llamacpp",
      llama: { url: llama.url },
    });
    try {
      const device = store.deviceInfo();
      assert.equal(device.defaultModel, "fake.gguf");
      assert.deepEqual(
        device.models.map((m) => [m.provider, m.id, m.tier, m.cost]),
        [[LLAMA_PROVIDER, "fake.gguf", "local", { input: 0, output: 0 }]],
      );

      const session = await store.create({ name: "local", cwd: ".", model: { provider: LLAMA_PROVIDER, modelId: "fake.gguf" } });
      const { h, c } = await store.resolveAgent(session.id, "main");
      const sent = await store.send(c, { content: "hello", requestId: "llamacpp-test-1" });
      const result = await store.settled(h, sent.id as number, 15000, AbortSignal.timeout(15000));
      assert.equal(result.status, "done");
      assert.equal(result.answer?.text, "from the local model");

      const request = llama.seen.find((r) => r.path === "/v1/chat/completions");
      assert.equal(request?.body?.model, "fake.gguf");
      // An unkeyed server still needs *a* key for the OpenAI client: pi sends this placeholder.
      assert.equal(request?.auth, "Bearer local");
    } finally {
      await store.close();
    }
  } finally {
    await llama.close();
    await rm(root, { recursive: true, force: true });
  }
});
