// Compile-time proof that pi-durable's live state still fits the SDK's wire contract.
// 1.5.0 no longer exports an event type to compare against: the agent stream renders the wire's event set out of
// pi-durable's state (src/view-events.ts), and `Store.taskNode` maps a task graph node into the SDK's task view. Both
// read pi-durable's types directly, so rename or reshape one of them and this file, or that code, stops compiling.
import type { BlockedReason, ConversationView, LiveState } from "@earendil-works/pi-durable";
import type { TaskBlockedReason, WireEntry } from "../../dpi-sdk/src/api.js";

/** A conversation view's entries are the wire's entries. */
export const entryFits = (e: ConversationView["entries"][number]): WireEntry => e;
/** The live document the agent stream renders is a conversation document with the shape it reads. */
export const liveStateFits = (live: LiveState): { run?: { inputs: readonly unknown[] } } => live;
// Why a task cannot run here is what the task tree carries to the client, in both directions: a reason pi-durable
// stops producing would be dead in the wire type, and one it adds would have nowhere to go.
export const blockedFits = (blocked: BlockedReason): TaskBlockedReason => blocked;
export const blockedBack = (blocked: TaskBlockedReason): BlockedReason => blocked;
