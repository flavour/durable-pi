// pi-durable 1.4.0 stores a task's wait instead of holding an invocation: the task turns `sleeping` and carries the
// moment it wakes, so a waiting job must surface that to clients and must wake by itself.
import { test } from "node:test";
import assert from "node:assert/strict";
import type { Store } from "../src/store.js";
import type { TaskNode } from "../../dpi-sdk/src/api.js";
import { JOB_RUNS_DOC, type JobRunsDoc } from "../../dpi-sdk/src/jobs.js";
import { faux, tempStore, until } from "./helpers.js";

const run = async (store: Store, session: string, taskId: number) => {
  const { h, c } = await store.resolveAgent(session, "main");
  return ((await store.agentDoc(h, c, JOB_RUNS_DOC)) as JobRunsDoc).items[String(taskId)];
};

test("a job waiting between checks parks as a sleeping task, then wakes by itself", async () => {
  const t = await tempStore();
  const store = await t.open();
  try {
    const session = await store.create({ name: "Sleep", cwd: "project", model: faux });
    const m = await store.resolveAgent(session.id, "main");
    const taskId = (await store.startJob(m.row, m.h, m.c, { job: "change-watch", input: { checks: 2, interval_seconds: 4 } })).taskId;

    let parked: TaskNode | undefined;
    await until(
      async () => {
        parked = (await store.liveTasks(m.h)).find((task) => task.id === taskId && task.state.status === "sleeping");
        return parked !== undefined;
      },
      20000,
      "the job run to park as sleeping",
    );
    // A stored wait: no invocation held, and the wake moment is part of the task.
    const wakes = Number(parked!.state.until);
    assert.ok(wakes > Date.now(), `sleeping task wakes in the future, not ${new Date(wakes).toISOString()}`);

    await until(async () => (await run(store, session.id, taskId))?.status === "completed", 30000, "watch completion");
    const done = await run(store, session.id, taskId);
    assert.deepEqual(done.phases.map((p) => p.status), ["done", "done", "done"]);
    assert.equal((await store.liveTasks(m.h)).some((task) => task.id === taskId), false);
  } finally {
    await store.close();
    await t.cleanup();
  }
});

test("a task the device cannot run carries its blocked reason to the client", async () => {
  const t = await tempStore();
  const store = await t.open();
  try {
    // The crash-loop guard parks a task restarted too often; the reason has to reach the task tree.
    const node = store.taskNode({
      id: 7,
      kind: "demo.job.change-watch",
      conversationId: 1,
      background: true,
      abortRequested: false,
      blocked: "restart_limit",
      state: { status: "sleeping", phase: "watch", until: 1_800_000_000_000 },
      conversations: [],
    });
    assert.equal(node.blocked, "restart_limit");
    assert.equal(node.state.status, "sleeping");
    assert.equal(node.state.until, 1_800_000_000_000);
  } finally {
    await store.close();
    await t.cleanup();
  }
});
