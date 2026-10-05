import type { ChatHistoryItem, ModelDescription } from "core";
import { describe, expect, it, vi } from "vitest";
import { MockIdeMessenger } from "../../context/MockIdeMessenger";
import {
  getCompactionIndex,
  splitTranscript,
  summarizeForContinuation,
} from "./automaticCompaction";
const model = {
  title: "Local",
  model: "qwen",
  contextLength: 16384,
} as ModelDescription;
const item = (role: "user" | "assistant" | "tool", content = "text") =>
  ({ message: { role, content }, contextItems: [] }) as ChatHistoryItem;

describe("automatic compaction", () => {
  it("preserves unanswered user input and skips the empty streaming placeholder", () => {
    expect(
      getCompactionIndex([
        item("user"),
        item("assistant"),
        item("user"),
        item("assistant", ""),
      ]),
    ).toBe(1);
    expect(getCompactionIndex([item("user"), item("assistant", "")])).toBe(-1);
  });
  it("compacts a completed tool sequence but never pending approvals or running tools", () => {
    const assistant = {
      ...item("assistant"),
      toolCallStates: [{ status: "done" }],
    } as ChatHistoryItem;
    expect(getCompactionIndex([item("user"), assistant, item("tool")])).toBe(2);
    for (const status of ["generating", "generated", "calling"]) {
      expect(
        getCompactionIndex([
          item("user"),
          { ...assistant, toolCallStates: [{ status }] } as ChatHistoryItem,
        ]),
      ).toBe(-1);
    }
  });
  it("does not recompact an unchanged summary", () => {
    expect(
      getCompactionIndex([
        { ...item("assistant"), conversationSummary: "done" },
      ]),
    ).toBe(-1);
  });
  it("bounds UTF-8 chunks without dropping non-ASCII text", () => {
    const text = "🦀漢字abc".repeat(1000),
      chunks = splitTranscript(text, 100);
    expect(chunks.join("")).toBe(text);
    expect(chunks.every((c) => new TextEncoder().encode(c).length <= 100)).toBe(
      true,
    );
  });
  it("summarizes oversized tool text in bounded, tool-free requests", async () => {
    const messenger = new MockIdeMessenger();
    const spy = vi
      .spyOn(messenger, "llmStreamChat")
      .mockImplementation(async function* () {
        yield [
          {
            role: "assistant",
            content: "Preserved user goal and completed actions.",
          },
        ];
        return undefined;
      });
    const summary = await summarizeForContinuation(
      [{ role: "tool", toolCallId: "t", content: "x".repeat(40000) }],
      model,
      messenger,
      new AbortController().signal,
    );
    expect(summary).toContain("user goal");
    expect(spy.mock.calls.length).toBeGreaterThan(1);
    for (const [request] of spy.mock.calls) {
      expect(request.completionOptions?.tools).toEqual([]);
      expect(request.messageOptions?.precompiled).toBe(true);
      expect(JSON.stringify(request.messages).length).toBeLessThan(10000);
    }
  });
  it("rejects empty summaries and stops on cancellation", async () => {
    const messenger = new MockIdeMessenger();
    vi.spyOn(messenger, "llmStreamChat").mockImplementation(async function* () {
      return undefined;
    });
    await expect(
      summarizeForContinuation(
        [{ role: "user", content: "Goal" }],
        model,
        messenger,
        new AbortController().signal,
      ),
    ).rejects.toThrow("empty summary");
    const controller = new AbortController();
    controller.abort();
    await expect(
      summarizeForContinuation(
        [{ role: "user", content: "Goal" }],
        model,
        messenger,
        controller.signal,
      ),
    ).rejects.toThrow();
  });
});
