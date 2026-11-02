// A live agent view, rendered from pi-durable 1.5.0's conversation view (spec §9.3).
//
// 1.5.0 removed `watchEvents()`: live agent state is read as state, not as an event log. This renders the event set
// the clients already speak (see dpi-sdk/src/api.ts) from that state, so the web UI, the TUI and the REST client are
// unchanged by the upgrade.
//
// Four properties of the view shape the rendering:
//
//  * `pi.live` is stored later than it is shown (deferred progress commits), so a serialized watch would hide a
//    streamed answer and running tool output until the next stored commit. The stream attaches a synchronous observer
//    instead, which sees every change as memory adopts it.
//  * A view's documents are the mount's live drafts, shared by every view it hands out, so they say what is true now,
//    not what was true when the change happened. Each change is therefore captured (its documents read as plain JSON,
//    its entry window as delivered) the moment it is seen, and the diff is taken between captures.
//  * The view's `entries` is a bounded window over the fork-aware history, in which head markers (`pi.compaction`,
//    `pi.reset`, `pi.handoff`) sit where they were written instead of cutting out the entries before them. The snapshot
//    applies the same head cut as the model's context, so a client starts at the active context; later appends arrive
//    one by one, and the client's reducer applies a marker's cut as it lands.
//  * The state says nothing about what the client already holds, so each event is derived against the state last sent:
//    entries at or after the newest one sent, a streamed answer as its blocks grow, a tool call as its slot appears,
//    gains output and settles, and the run as its task appears and drains.
//
// 1.5.0 also stopped reporting three things this stream used to carry, and the renderer supplies or omits them:
//  * a model-issued tool call's arguments are in its assistant entry, not its slot, so the start event takes them from
//    the entries the same stream carries;
//  * `run_end` carries the run's inputs from the run that was just live, since the settled list is not kept;
//  * the answer entry a streamed message settles into is found by matching the streamed blocks against the entries, as
//    the live state no longer marks the settled answer.
import { BACKGROUND_CONTEXT as ctx } from "@earendil-works/chord/context";
import type { TextContent, ThinkingContent, ToolCall as AiToolCall } from "@earendil-works/pi-ai";
import type { ConversationView, EntryRecord, Harness, LiveState, Observation, ToolSlot } from "@earendil-works/pi-durable";
import type {
  WireAgentEvent,
  WireAgentState,
  WireAssistantBlock,
  WireAssistantMessage,
  WireCompaction,
  WireEntry,
  WireMessage,
  WireMessageChange,
  WireQueued,
  WireSnapshotEvent,
  WireToolSlot,
  WireUsage,
  WireUsageState,
} from "../../dpi-sdk/src/api.js";

/** How much of the history a view window holds; its snapshot starts at the newest head marker. */
const WINDOW = { min: 2_000, max: 8_000 };
/** How far back from the window's tail to look for the entry a streamed answer settled into. */
const SETTLE_SCAN = 25;

/** One agent's live view as the wire's event set. */
export interface AgentViewEvents {
  /** The state as it was when the view attached, as the wire's snapshot: the first thing to send. */
  snapshot(): WireSnapshotEvent;
  /** Events for the changes seen since then; a change can say nothing new, and `[]` also means the view ended. */
  next(): Promise<WireAgentEvent[]>;
  /** True once the conversation's view has gone away, which ends `next()`. */
  readonly ended: boolean;
  /** Detach the observer; idempotent. */
  close(): void;
}

/** Attach to one conversation's view; `undefined` when the Harness has no such conversation. */
export async function agentViewEvents(harness: Harness, conversationId: number): Promise<AgentViewEvents | undefined> {
  const stitch = new ViewRender(harness, conversationId);
  return (await stitch.attach()) ? stitch : undefined;
}

/** A message with no usage yet; the wire's update frame always names one. */
const EMPTY_USAGE: WireUsage = {
  input: 0,
  output: 0,
  cacheRead: 0,
  cacheWrite: 0,
  totalTokens: 0,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

const json = <T>(value: unknown): T => JSON.parse(JSON.stringify(value)) as T;
const idOf = (id: unknown) => Number(id);
const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);

/** One change of the view, held as plain JSON so a later change cannot rewrite it. */
type Captured = {
  entries: readonly EntryRecord[];
  live: LiveState;
  agent?: WireAgentState;
  usage?: WireUsageState;
  inbox: WireQueued[];
};

function capture(view: ConversationView): Captured {
  const doc = <T>(kind: string) => {
    const value = view.docs[kind];
    return value === undefined ? undefined : json<T>(value);
  };
  return {
    entries: [...view.entries],
    live: doc<LiveState>("pi.live") ?? {},
    agent: doc<WireAgentState>("pi.agent"),
    usage: doc<WireUsageState>("pi.usage"),
    inbox: inboxOf(doc<{ items?: { id?: unknown; mode?: string }[] }>("pi.inbox")),
  };
}

/** The active context of a window: the newest head marker, then the non-marker entries from its head. */
function activeWindow(entries: readonly EntryRecord[]): readonly EntryRecord[] {
  let marker: EntryRecord | undefined;
  for (const entry of entries) if (entry.head !== undefined) marker = entry;
  if (marker === undefined) return [...entries];
  // A marker summarizes everything up to and including its head entry, as the model's context does.
  const from = idOf(marker.head);
  return [marker, ...entries.filter((e) => e.head === undefined && idOf(e.id) > from)];
}

const inboxOf = (doc: { items?: { id?: unknown; mode?: string }[] } | undefined): WireQueued[] =>
  (doc?.items ?? []).map((item) => ({ id: idOf(item.id), mode: item.mode ?? "" }));

/** The blocks of an entry's message, in the shape the wire carries them. */
const blocksOf = (message: WireMessage): readonly WireAssistantBlock[] =>
  (message as { content?: readonly WireAssistantBlock[] }).content ?? [];
const textOf = (blocks: readonly WireAssistantBlock[]) =>
  blocks.filter((b) => b.type === "text").map((b) => (b as TextContent).text ?? "").join("");
const thinkingOf = (blocks: readonly WireAssistantBlock[]) =>
  blocks.filter((b) => b.type === "thinking").map((b) => (b as ThinkingContent).thinking ?? "").join("");
const callIdsOf = (blocks: readonly WireAssistantBlock[]) =>
  blocks.filter((b) => b.type === "toolCall").map((b) => (b as AiToolCall).id).join(",");

/** The assistant message an entry contributes, if it contributes one. */
function assistantOf(entry: EntryRecord): WireAssistantMessage | undefined {
  for (const message of [...(entry.model ?? [])].reverse())
    if (message.role === "assistant") return json<WireAssistantMessage>(message);
  return undefined;
}

/** Whether `committed` is `streamed` finished: the same calls, and its text grown to hold the streamed text. */
function settledInto(streamed: readonly WireAssistantBlock[], committed: readonly WireAssistantBlock[]): boolean {
  if (streamed.length === 0 || committed.length === 0) return false;
  if (callIdsOf(streamed) !== callIdsOf(committed)) return false;
  return textOf(committed).startsWith(textOf(streamed)) && thinkingOf(committed).startsWith(thinkingOf(streamed));
}

/** The entry a streamed answer has settled into: the last one near the tail that holds the same answer, finished. */
function settledEntry(entries: readonly EntryRecord[], streamed: readonly WireAssistantBlock[]) {
  let found: EntryRecord | undefined;
  for (const entry of entries.slice(-SETTLE_SCAN)) {
    const message = assistantOf(entry);
    if (message !== undefined && settledInto(streamed, message.content)) found = entry;
  }
  return found;
}

/** Append-only change set that turns `previous` into `next`; a shrink replaces the whole message. */
function messageChanges(previous: WireAssistantMessage, next: WireAssistantMessage): readonly WireMessageChange[] {
  const changes: WireMessageChange[] = [];
  if (next.content.length < previous.content.length) return [{ type: "message", message: next }];
  for (let i = 0; i < next.content.length; i++) {
    const before = previous.content[i];
    const after = next.content[i];
    if (before === undefined || before.type !== after.type) {
      changes.push({ type: "block", contentIndex: i, block: after });
      continue;
    }
    if (after.type === "toolCall") {
      // Arguments grow key by key, which has no path to append at; setting the block gives the same view.
      if (!same((before as AiToolCall).arguments, (after as AiToolCall).arguments))
        changes.push({ type: "block", contentIndex: i, block: after });
      continue;
    }
    const field = after.type === "text" ? "text" : "thinking";
    const grown = String((after as TextContent & ThinkingContent)[field] ?? "");
    const soFar = String((before as TextContent & ThinkingContent)[field] ?? "");
    if (soFar === grown) continue;
    if (grown.startsWith(soFar))
      changes.push({
        type: after.type === "text" ? "text_delta" : "thinking_delta",
        contentIndex: i,
        delta: grown.slice(soFar.length),
      });
    else changes.push({ type: "block", contentIndex: i, block: after });
  }
  return changes;
}

const slotOf = (slot: ToolSlot): WireToolSlot => ({
  callId: slot.callId,
  name: slot.name,
  ...(slot.taskId !== undefined ? { taskId: idOf(slot.taskId) } : {}),
  status: slot.status,
  ...(slot.output !== undefined ? { output: slot.output } : {}),
  ...(slot.droppedBytes !== undefined ? { droppedBytes: slot.droppedBytes } : {}),
  ...(slot.droppedLines !== undefined ? { droppedLines: slot.droppedLines } : {}),
  ...(slot.details !== undefined ? { details: json(slot.details) } : {}),
  ...(slot.entry !== undefined ? { entry: idOf(slot.entry) } : {}),
});

const statusOf = (status: NonNullable<LiveState["compactions"]>[number]): WireCompaction => ({
  taskId: idOf(status.taskId),
  reason: String(status.trigger),
  blocking: status.blocking,
  attempt: status.attempt,
  ...(status.retry ? { retry: { at: idOf(status.retry.at), error: status.retry.error } } : {}),
});

/** What has been sent for one conversation, so the next capture can be handed over as a diff. */
class ViewRender {
  #pending: Captured[] = [];
  #ended = false;
  #wake: (() => void) | undefined;
  #observation: Observation<ConversationView> | undefined;
  /** Newest entry the client holds. */
  #seenEntry = 0;
  /** Tool call arguments from the assistant entries it holds. */
  #args = new Map<string, unknown>();
  /** The live run: its task, and the inputs it started with. */
  #run?: { taskId: number; inputs: readonly number[] };
  /** The streamed answer as the client holds it. */
  #message?: WireAssistantMessage;
  #slots = new Map<string, { name: string; output?: string }>();
  /** Compaction tasks the client holds, by the reason that started them. */
  #compactions = new Map<number, string>();
  #agent?: WireAgentState;
  #usage?: WireUsageState;
  #inbox: WireQueued[] = [];
  #failure?: string;
  /** The state as of `attach()`, until a later one is what the snapshot should show. */
  #initial!: Captured;

  constructor(
    readonly harness: Harness,
    readonly conversationId: number,
  ) {}

  /** Attach to the conversation's view; false when the Harness has no such conversation. */
  async attach(): Promise<boolean> {
    const observation = await this.harness.observeConversation(
      this.conversationId as never,
      {
        advance: (call) => {
          if (call.kind !== "value") return;
          this.#pending.push(capture(call.value));
          this.#wake?.();
        },
        end: () => {
          this.#ended = true;
          this.#wake?.();
        },
      },
      ctx,
      WINDOW,
    );
    if (observation === undefined) return false;
    this.#observation = observation;
    this.#initial = capture(observation.value);
    return true;
  }

  /** Detach the observer; idempotent. */
  close(): void {
    this.#observation?.detach();
    this.#observation = undefined;
    this.#ended = true;
    this.#wake?.();
  }

  snapshot(): WireSnapshotEvent {
    // Whatever has changed since attach is part of what the client has not seen yet, so the newest capture is the
    // one to show; the older ones describe states it never held.
    const state = this.#pending.pop() ?? this.#initial;
    this.#pending.length = 0;
    return this.#snapshot(state);
  }

  /** Events for the changes seen since the last call; a change can say nothing new. */
  async next(): Promise<WireAgentEvent[]> {
    for (;;) {
      if (this.#pending.length > 0) return this.#pending.splice(0, this.#pending.length).flatMap((state) => this.#render(state));
      if (this.#ended) return [];
      await new Promise<void>((resolve) => {
        this.#wake = resolve;
      });
      this.#wake = undefined;
    }
  }

  get ended(): boolean {
    return this.#ended;
  }

  #render(state: Captured): WireAgentEvent[] {
    const events: WireAgentEvent[] = [];
    const appended: EntryRecord[] = [];
    for (const entry of state.entries) {
      if (idOf(entry.id) <= this.#seenEntry) continue;
      this.#seenEntry = idOf(entry.id);
      for (const block of (entry.model ?? []).flatMap(blocksOf))
        if (block.type === "toolCall") this.#args.set(block.id, json((block as AiToolCall).arguments));
      events.push({ type: "entry_appended", entry: json<WireEntry>(entry) });
      appended.push(entry);
    }
    events.push(...this.#runEvents(state.live));
    events.push(...this.#generation(state, appended));
    events.push(...this.#toolEvents(state));
    events.push(...this.#compactionEvents(state.live));
    events.push(...this.#docEvents(state));
    events.push(...this.#failureEvents(state.live));
    return events;
  }

  #snapshot(state: Captured): WireSnapshotEvent {
    const entries = activeWindow(state.entries);
    for (const entry of entries) {
      this.#seenEntry = Math.max(this.#seenEntry, idOf(entry.id));
      for (const block of (entry.model ?? []).flatMap(blocksOf))
        if (block.type === "toolCall") this.#args.set(block.id, json((block as AiToolCall).arguments));
    }
    this.#agent = state.agent;
    this.#usage = state.usage;
    this.#inbox = state.inbox;
    this.#run = state.live.run ? { taskId: idOf(state.live.run.taskId), inputs: state.live.run.inputs.map(idOf) } : undefined;
    this.#message = state.live.generation?.message ? json<WireAssistantMessage>(state.live.generation.message) : undefined;
    for (const slot of state.live.tools ?? []) this.#slots.set(slot.callId, { name: slot.name, output: slot.output });
    this.#compactions = new Map((state.live.compactions ?? []).map((c) => [idOf(c.taskId), String(c.trigger)]));
    if (state.live.runFailure !== undefined)
      this.#failure = `${idOf(state.live.runFailure.taskId)}:${state.live.runFailure.reason}`;
    const generation = state.live.generation;
    return {
      type: "snapshot",
      entries: entries.map((e) => json<WireEntry>(e)),
      ...(this.#run ? { run: { inputs: this.#run.inputs } } : {}),
      ...(generation
        ? {
            generation: {
              attempt: generation.attempt,
              ...(this.#message ? { message: this.#message } : {}),
              ...(generation.retry ? { retry: { at: idOf(generation.retry.at), error: generation.retry.error } } : {}),
              ...(generation.deferred ? { deferred: { pollAt: idOf(generation.deferred.pollAt) } } : {}),
            },
          }
        : {}),
      tools: (state.live.tools ?? []).map(slotOf),
      compactions: (state.live.compactions ?? []).map(statusOf),
      inbox: this.#inbox,
      agent: this.#agent ?? {},
      usage: this.#usage ?? { models: {}, tools: {} },
    };
  }

  /** The streamed answer: its start, how its blocks grew, and the entry it settled into. */
  #generation(state: Captured, appended: readonly EntryRecord[]): WireAgentEvent[] {
    const events: WireAgentEvent[] = [];
    const current = state.live.generation?.message ? json<WireAssistantMessage>(state.live.generation.message) : undefined;
    if (current !== undefined) {
      if (this.#message === undefined) {
        this.#message = current;
        events.push({ type: "message_start", message: current });
      } else {
        const changes = messageChanges(this.#message, current);
        if (changes.length > 0) {
          this.#message = current;
          events.push({ type: "message_update", usage: json<WireUsage>(current.usage ?? EMPTY_USAGE), changes });
        }
      }
    }
    // A committed answer does not remove the live copy, so match the streamed blocks against the entries.
    let settled: EntryRecord | undefined;
    if (this.#message !== undefined) settled = settledEntry(state.entries, this.#message.content);
    if (settled !== undefined) {
      this.#message = undefined;
      events.push({ type: "message_end", entry: json<WireEntry>(settled) });
    }
    // An answer committed without ever being held live (a short or cached round) still starts and ends a message.
    if (this.#message === undefined)
      for (const entry of appended) {
        if (entry === settled) continue;
        const message = assistantOf(entry);
        if (message === undefined) continue;
        events.push({ type: "message_start", message }, { type: "message_end", entry: json<WireEntry>(entry) });
      }
    return events;
  }

  #runEvents(live: LiveState): WireAgentEvent[] {
    const events: WireAgentEvent[] = [];
    if (live.run !== undefined && this.#run === undefined) {
      const inputs = live.run.inputs.map(idOf);
      this.#run = { taskId: idOf(live.run.taskId), inputs };
      events.push({ type: "run_start", inputs });
    } else if (live.run !== undefined && this.#run !== undefined) {
      // A run that has answered holds no inputs while it waits on its `afterRun` tasks; keep naming the ones it
      // started with, which is what `run_end` carries.
      const inputs = live.run.inputs.map(idOf);
      if (idOf(live.run.taskId) === this.#run.taskId && inputs.length > 0) this.#run = { ...this.#run, inputs };
    } else if (live.run === undefined && this.#run !== undefined) {
      // 1.5.0 keeps no settled list of what a run took: an unanswered run names its inputs in the failure it left,
      // and a run that answered has the ones this stream saw it start with.
      const failure = live.runFailure;
      const ended = this.#run;
      const inputs = failure !== undefined && idOf(failure.taskId) === ended.taskId ? failure.inputs.map(idOf) : ended.inputs;
      this.#run = undefined;
      this.#message = undefined;
      events.push({ type: "run_end", inputs });
    }
    return events;
  }

  #toolEvents(state: Captured): WireAgentEvent[] {
    const events: WireAgentEvent[] = [];
    const held = new Set<string>();
    for (const slot of state.live.tools ?? []) {
      held.add(slot.callId);
      const before = this.#slots.get(slot.callId);
      if (before === undefined)
        events.push({
          type: "tool_execution_start",
          toolCallId: slot.callId,
          toolName: slot.name,
          // A model-issued slot carries no arguments: they are in the assistant entry that asked for the call.
          args: this.#args.get(slot.callId),
        });
      const output = slot.output;
      if (output !== undefined && output !== before?.output) {
        const soFar = before?.output ?? "";
        events.push({
          type: "tool_execution_update",
          toolCallId: slot.callId,
          toolName: slot.name,
          output: output.startsWith(soFar) ? { append: output.slice(soFar.length) } : { set: output },
        });
      }
      if (slot.status === "done") {
        this.#slots.delete(slot.callId);
        this.#args.delete(slot.callId);
        // The result entry went out with the entries above when the window still held it.
        const entry = slot.entry === undefined ? undefined : entryIn(state.entries, idOf(slot.entry));
        events.push({
          type: "tool_execution_end",
          toolCallId: slot.callId,
          toolName: slot.name,
          ...(entry !== undefined ? { entry: json<WireEntry>(entry) } : {}),
        });
      } else this.#slots.set(slot.callId, { name: slot.name, output });
    }
    // A slot that leaves the registry without being done ended anyway: a rerun replaces the round, or its task faulted.
    for (const [callId, slot] of [...this.#slots])
      if (!held.has(callId)) {
        this.#slots.delete(callId);
        events.push({ type: "tool_execution_end", toolCallId: callId, toolName: slot.name });
      }
    return events;
  }

  #compactionEvents(live: LiveState): WireAgentEvent[] {
    const events: WireAgentEvent[] = [];
    const current = new Map((live.compactions ?? []).map((c) => [idOf(c.taskId), statusOf(c)]));
    for (const [taskId, compaction] of current)
      if (!this.#compactions.has(taskId)) {
        this.#compactions.set(taskId, compaction.reason);
        events.push({ type: "compaction_start", taskId, reason: compaction.reason, blocking: compaction.blocking });
      }
    for (const [taskId, reason] of [...this.#compactions])
      if (!current.has(taskId)) {
        this.#compactions.delete(taskId);
        // How it ended is not in the state once the task leaves; its result entry follows as an entry, and the
        // next snapshot shows the new context.
        events.push({ type: "compaction_end", taskId, reason });
      }
    return events;
  }

  #docEvents(state: Captured): WireAgentEvent[] {
    const events: WireAgentEvent[] = [];
    if (state.agent !== undefined && !same(state.agent, this.#agent)) {
      this.#agent = state.agent;
      events.push({ type: "agent_changed", agent: state.agent });
    }
    if (state.usage !== undefined && !same(state.usage, this.#usage)) {
      this.#usage = state.usage;
      events.push({ type: "usage_changed", usage: state.usage });
    }
    if (!same(state.inbox, this.#inbox)) {
      this.#inbox = state.inbox;
      events.push({ type: "inbox_update", items: state.inbox });
    }
    return events;
  }

  /** A run that ended without an answer: `pi.live` keeps the reason until a later run starts. */
  #failureEvents(live: LiveState): WireAgentEvent[] {
    const failure = live.runFailure;
    if (failure === undefined) {
      this.#failure = undefined;
      return [];
    }
    const key = `${idOf(failure.taskId)}:${failure.reason}`;
    if (key === this.#failure) return [];
    this.#failure = key;
    return [
      {
        type: "task_failed",
        taskId: idOf(failure.taskId),
        kind: "pi.generation",
        message: `${failure.reason}${failure.detail === undefined ? "" : `: ${failure.detail}`}`,
      },
    ];
  }
}

/** An entry by ID, while the view's window still holds it. */
function entryIn(entries: readonly EntryRecord[], id: number): EntryRecord | undefined {
  return entries.find((e) => idOf(e.id) === id);
}
