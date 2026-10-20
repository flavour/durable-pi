// A llama.cpp server as a device's model provider (DPI_PROVIDER=llamacpp): the GGUFs the server
// serves are the device's models, at zero cost. Works with single-model `llama-server -m …` and
// with the router (`--models-dir …`); loaded and sleeping models are selectable, unloaded router
// presets are not, because a request would load them into a context the device did not choose.
import { createProvider, type MutableModels } from "@earendil-works/pi-ai/models";
import { openAICompletionsApi } from "@earendil-works/pi-ai/api/openai-completions.lazy";
import type { Model } from "@earendil-works/pi-ai";

// Provider id matches pi's own llama.cpp extension, so a model id means the same thing on both sides.
export const LLAMA_PROVIDER = "llama.cpp";

/** Where a llama.cpp server lives. `url` may be written with or without its `/v1`. */
export interface LlamaServer {
  url: string;
  /** Only needed when the server was started with `--api-key`. */
  apiKey?: string;
}

/** The server root, without `/v1`: llama.cpp answers under both. */
export const serverRoot = (url: string) => url.trim().replace(/\/+$/, "").replace(/\/v1$/, "");
/** The OpenAI-compatible endpoint of the server. */
export const chatUrl = (url: string) => `${serverRoot(url)}/v1`;

const FALLBACK_CONTEXT_WINDOW = 128_000;

type RawModel = {
  id?: unknown;
  status?: { value?: unknown };
  meta?: { n_ctx?: unknown; n_ctx_train?: unknown };
  architecture?: { input_modalities?: unknown; output_modalities?: unknown };
};

const positiveInt = (v: unknown) => (typeof v === "number" && Number.isSafeInteger(v) && v > 0 ? v : undefined);
const has = (list: unknown, value: string) => Array.isArray(list) && list.includes(value);
const statusOf = (m: RawModel) => (typeof m.status?.value === "string" ? m.status.value : "loaded"); // older servers report no status

/** A model the server can actually answer with: loaded or woken on request, and text-producing. */
const usable = (m: RawModel) =>
  (statusOf(m) === "loaded" || statusOf(m) === "sleeping") && !has(m.architecture?.output_modalities, "decisions");

async function getJson(url: string, apiKey: string | undefined, signal: AbortSignal): Promise<unknown | undefined> {
  const response = await fetch(url, {
    headers: { accept: "application/json", ...(apiKey ? { authorization: `Bearer ${apiKey}` } : {}) },
    signal,
  });
  if (!response.ok) throw Error(`llama.cpp ${new URL(url).pathname}: ${response.status}`);
  return response.json();
}

/** The server's own model list, mapped onto pi chat models. */
export async function llamaModels(server: LlamaServer, signal: AbortSignal): Promise<Model<"openai-completions">[]> {
  const root = serverRoot(server.url);
  const base = `${root}/v1`;
  const list = await getJson(`${base}/models`, server.apiKey, signal);
  const raw = (list as { data?: unknown } | undefined)?.data;
  const models = (Array.isArray(raw) ? raw : [])
    .filter((m): m is RawModel => typeof m === "object" && m !== null)
    .filter(usable);
  return Promise.all(
    models.map(async (m) => {
      const id = typeof m.id === "string" ? m.id : "";
      if (!id) throw Error("llama.cpp /models entry without an id");
      // /props (llama.cpp's own endpoint, not under /v1) reports the chat template, which is what says
      // whether this model switches thinking on with an enable_thinking argument. Only loaded models are
      // asked: querying a sleeping model would wake it, and an unloaded router preset would load it.
      const props =
        statusOf(m) === "loaded"
          ? await getJson(`${root}/props?model=${encodeURIComponent(id)}`, server.apiKey, signal).catch(() => undefined)
          : undefined;
      return toModel(id, m, base, reasoningOf(props));
    }),
  );
}

const reasoningOf = (props: unknown) =>
  typeof (props as { chat_template?: unknown } | undefined)?.chat_template === "string" &&
  (props as { chat_template: string }).chat_template.includes("enable_thinking");

function toModel(id: string, m: RawModel, baseUrl: string, reasoning: boolean): Model<"openai-completions"> {
  const contextWindow =
    positiveInt(m.meta?.n_ctx) ?? positiveInt(m.meta?.n_ctx_train) ?? FALLBACK_CONTEXT_WINDOW;
  return {
    id,
    name: id,
    api: "openai-completions",
    provider: LLAMA_PROVIDER,
    baseUrl,
    reasoning,
    ...(reasoning && {
      thinkingLevelMap: { off: "off", minimal: null, low: null, medium: "medium", high: null, xhigh: null },
    }),
    input: has(m.architecture?.input_modalities, "image") ? ["text", "image"] : ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow,
    maxTokens: contextWindow,
    // llama.cpp's chat completions: no store field, no developer role, no reasoning_effort, `max_tokens`,
    // and usage in the stream. Thinking goes through the chat template's enable_thinking.
    compat: {
      supportsStore: false,
      supportsDeveloperRole: false,
      supportsReasoningEffort: false,
      supportsUsageInStreaming: true,
      supportsStrictMode: false,
      maxTokensField: "max_tokens",
      ...(reasoning && { thinkingFormat: "qwen-chat-template" }),
    },
  };
}

/** Register the server as this device's only model provider. */
export function installLlamaCpp(models: MutableModels, server: LlamaServer) {
  const baseUrl = chatUrl(server.url);
  models.setProvider(
    createProvider<"openai-completions">({
      id: LLAMA_PROVIDER,
      name: `llama.cpp ${serverRoot(server.url)}`,
      baseUrl,
      // A llama.cpp server only checks a bearer token when started with --api-key, so an unkeyed
      // server is still "configured": the device needs no secret to use its own hardware. pi-ai's
      // OpenAI client insists on a key, hence the "local" placeholder pi itself sends.
      auth: {
        apiKey: {
          name: "llama.cpp server",
          resolve: async () => ({
            auth: { apiKey: server.apiKey ?? "local" },
            source: server.apiKey ? "DPI_LLAMA_API_KEY" : "placeholder",
          }),
        },
      },
      models: [],
      fetchModels: ({ signal }) => llamaModels(server, signal),
      api: openAICompletionsApi(),
    }),
  );
  return baseUrl;
}
