// The agent view stream is the backend's own render of pi-durable 1.5.0's live state (src/view-events.ts): the wire's
// event set derived from a conversation view, covering what 1.5.0 stopped reporting. A stub Harness hands the renderer
// the views a real one would, so each behaviour is checked on its own.
import { test } from "node:test";
import assert from "node:assert/strict";
import type { ConversationView, EntryRecord, Harness, LiveState, JsonObject } from "@earendil-works/pi-durable";
import { agentViewEvents } from "../src/view-events.js";

const ident = (n: number) => n as never;

function entry(id: number, kind: string, model?: readonly unknown[], head?: number): EntryRecord {
  const e: Record<string, unknown> = { id: ident(id), conversationId: ident(1), ts: 0, kind };
  if (model !== undefined) e.model = model;
  if (head !== undefined) e.head = ident(head);
  return e as unknown as EntryRecord;
}
const user = (id: number, text?: string) =>
  entry(id, "pi.user", text === undefined ? undefined : [{ role: "user", content: text }]);
const round = (id: number, blocks: readonly unknown[], stopReason = "toolUse") =>
  entry(id, "pi.assistant", [{ role: "assistant", content: blocks, provider: "faux", model: "echo", stopReason }]);
const text = (t: string) => ({ type: "text" as const, text: t });
const call = { type: "toolCall" as const, id: "tc-1", name: "bash", arguments: { command: "echo hi" } };
const run = { taskId: ident(11), first: ident(1), inputs: [ident(3)], turns: 1 };
const streaming = (t: string, withCall = false) =>
  ({ attempt: 1, message: { role: "assistant", content: withCall ? [text(t), call] : [text(t)] } } as never);

/** A view of one conversation, as the Harness would hand it to an observer. */
function view(entries: readonly EntryRecord[] = [], live: Partial<LiveState> = {}, docs: Record<string, JsonObject> = {}): ConversationView {
  return {
    conversation: { id: ident(1), rootId: ident(1), label: "main", createdAt: 0 } as never,
    entries,
    docs: { "pi.live": live as JsonObject, ...docs },
  };
}

/** A Harness whose only conversation starts at `initial` and takes whatever the test pushes. */
function stubHarness(initial: ConversationView = view()) {
  let advance: ((call: { kind: "value"; value: ConversationView }) => void) | undefined;
  let end: (() => void) | undefined;
  const harness = {
    observeConversation: async (_id: unknown, observer: { advance: typeof advance; end: typeof end }) => {
      advance = observer.advance;
      end = observer.end;
      return { detach: () => {}, value: initial };
    },
  } as unknown as Harness;
  return {
    /** Push the next view, as an observer's `advance` would. */
    push: (next: ConversationView) => advance?.({ kind: "value", value: next }),
    /** End the view, as an observer's `end` would. */
    stop: () => end?.(),
    attach: async () => {
      const stream = await agentViewEvents(harness, 1);
      if (!stream) throw new Error("the view did not attach");
      return stream;
    },
  };
}

const types = (events: readonly { type: string }[]) => events.map((e) => e.type);

test("a run's start names its inputs, and its end names them again after 1.5.0 settled them away", async () => {
  const stub = stubHarness();
  const stream = await stub.attach();
  assert.deepEqual(stream.snapshot().entries, []);

  stub.push(view([user(1, "echo hi")], { run }));
  const started = await stream.next();
  assert.deepEqual(types(started), ["entry_appended", "run_start"]);
  assert.deepEqual(started[1].type === "run_start" ? started[1].inputs : null, [3]);

  // A run that answered holds no inputs while it waits on its afterRun tasks, and the state keeps no record of
  // a settled run: `run_end` names the inputs the run started with, and a failure names the ones it left unsettled.
  stub.push(view([user(1, "echo hi")], { run: { ...run, inputs: [] } }));
  assert.deepEqual(await stream.next(), [], "a repeat of what the client holds says nothing new");
  stub.push(view([], { runFailure: { taskId: ident(11), inputs: [ident(3)], reason: "model_error", detail: "429" } }));
  const ended = await stream.next();
  assert.deepEqual(types(ended), ["run_end", "task_failed"]);
  assert.deepEqual(ended[0].type === "run_end" ? ended[0].inputs : null, [3]);
  assert.ok(ended[1].type === "task_failed" && ended[1].message === "model_error: 429", "the failure reason reaches the client");

  // The ended view stops handing over events.
  stub.stop();
  assert.deepEqual(await stream.next(), []);
  assert.ok(stream.ended);
});

test("a model-issued tool call's arguments come from the entry that asked for it, and it ends on its result entry", async () => {
  const stub = stubHarness();
  const stream = await stub.attach();
  const question = user(1, "echo hi");
  const asked = round(2, [text("I will run it"), call]);
  const result = entry(3, "pi.tool_result", [{ role: "toolResult", toolCallId: "tc-1", content: [text("hi")], isError: false }]);

  // The answer streams before the round is committed.
  stub.push(view([question], { run, generation: streaming("I will") }));
  assert.deepEqual(types(await stream.next()), ["entry_appended", "run_start", "message_start"]);

  // The round lands with the call. Its slot names the call but holds no arguments, and its output starts flowing.
  stub.push(view([question, asked], { run, generation: streaming("I will run it", true), tools: [{ callId: "tc-1", name: "bash", status: "running", output: "hi" }] }));
  const grown = await stream.next();
  assert.deepEqual(types(grown), ["entry_appended", "message_update", "message_end", "tool_execution_start", "tool_execution_update"]);
  assert.deepEqual(grown.find((e) => e.type === "tool_execution_start")?.args, { command: "echo hi" });
  assert.ok(
    grown.find((e) => e.type === "message_update")?.changes.some((c) => c.type === "text_delta" && c.delta === " run it"),
    "the streamed answer grows by delta",
  );
  assert.ok(grown.some((e) => e.type === "message_end" && e.entry.id === 2), "the answer settles into the round's entry");

  // The result entry lands, and the slot settles onto it.
  stub.push(view([question, asked, result], { run, tools: [{ callId: "tc-1", name: "bash", status: "done", entry: ident(3) }] }));
  const ended = await stream.next();
  assert.deepEqual(types(ended), ["entry_appended", "tool_execution_end"]);
  assert.ok(ended[1].type === "tool_execution_end" && ended[1].entry?.id === 3, "the call ends on its result entry");

  // The next round's call is answered from the entry that asked for it, not from a stale map.
  const second = { type: "toolCall" as const, id: "tc-2", name: "read", arguments: { path: "a.md" } };
  stub.push(view([question, asked, result, round(4, [second])], { run, tools: [{ callId: "tc-2", name: "read", status: "running" }] }));
  const again = await stream.next();
  assert.deepEqual(again.find((e) => e.type === "tool_execution_start")?.args, { path: "a.md" });
});

test("an answer committed without ever being streamed still starts and ends a message", async () => {
  const stub = stubHarness();
  const stream = await stub.attach();
  stub.push(view([round(4, [text("Done")], "stop")]));
  const events = await stream.next();
  assert.deepEqual(types(events), ["entry_appended", "message_start", "message_end"]);
  assert.ok(events[2].type === "message_end" && events[2].entry.id === 4);
});

test("a compaction shows while its task is live and ends named by what started it", async () => {
  const stub = stubHarness();
  const stream = await stub.attach();
  stub.push(view([], { compactions: [{ taskId: ident(21), trigger: "threshold", blocking: true, attempt: 1 }] }));
  const started = await stream.next();
  assert.deepEqual(types(started), ["compaction_start"]);
  assert.ok(started[0].type === "compaction_start" && started[0].reason === "threshold" && started[0].blocking);

  stub.push(view([], { compactions: [] }));
  const ended = await stream.next();
  assert.deepEqual(types(ended), ["compaction_end"]);
  assert.ok(ended[0].type === "compaction_end" && ended[0].reason === "threshold", "it ends with its trigger");
});

test("the agent, usage and inbox documents reach the client as they change", async () => {
  const stub = stubHarness();
  const stream = await stub.attach();
  const docs = {
    "pi.agent": { model: { provider: "faux", modelId: "echo" } } as never,
    "pi.usage": { models: { echo: { runs: 1 } }, tools: {} } as never,
    "pi.inbox": { items: [{ id: 7, mode: "steer", content: "later" }] } as never,
  };
  stub.push(view([], {}, docs));
  const changed = await stream.next();
  assert.deepEqual(types(changed), ["agent_changed", "usage_changed", "inbox_update"]);
  assert.ok(changed[2].type === "inbox_update" && changed[2].items[0].id === 7);

  stub.push(view([], {}, docs));
  assert.deepEqual(await stream.next(), [], "the same documents say nothing new");
});

test("a snapshot starts where the model's context starts, at the newest head marker", async () => {
  // The window holds the whole fork-aware history, with the marker where it was written; the snapshot cuts the
  // entries the marker summarizes out, as the model's context does.
  const entries = [user(1), user(2), entry(3, "pi.compaction", undefined, 2), user(4)];
  const stub = stubHarness(view(entries));
  const stream = await stub.attach();
  assert.deepEqual(stream.snapshot().entries.map((e) => `${e.id}:${e.kind}`), ["3:pi.compaction", "4:pi.user"]);
  // Later appends arrive one by one, so the client's reducer applies a marker's cut as it lands.
  stub.push(view([...entries, user(5)]));
  assert.deepEqual(types(await stream.next()), ["entry_appended"]);
});
