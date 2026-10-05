import { describe, expect, it, vi } from "vitest";
import { createMockStore, getEmptyRootState } from "../../util/test/mockStore";
import { streamNormalInput } from "./streamNormalInput";
vi.mock("../util/getBaseSystemMessage", () => ({
  getBaseSystemMessage: () => "System rules",
}));

function setup() {
  const state = getEmptyRootState();
  state.config.config.selectedModelByRole.chat = {
    title: "Local",
    model: "gpt-4",
    provider: "openai",
    underlyingProviderName: "openai",
    contextLength: 16384,
  };
  state.session.history = [
    {
      message: { id: "u1", role: "user", content: "Original goal" },
      contextItems: [],
    },
    {
      message: { id: "a1", role: "assistant", content: "Completed work" },
      contextItems: [],
    },
    {
      message: { id: "u2", role: "user", content: "Continue with the tests" },
      contextItems: [],
    },
    { message: { id: "a2", role: "assistant", content: "" }, contextItems: [] },
  ];
  const store = createMockStore(state);
  return { store, messenger: store.mockIdeMessenger };
}
const compiled = (percentage = 0.9) =>
  ({
    status: "success",
    content: {
      compiledChatMessages: [{ role: "user", content: "Continue" }],
      contextPercentage: percentage,
      didPrune: false,
    },
  }) as const;

describe("stream auto compaction", () => {
  it.each(["threshold", "overflow"])(
    "recovers from %s, keeps newest request, and resumes",
    async (reason) => {
      const { store, messenger } = setup();
      const request = vi
        .spyOn(messenger, "request")
        .mockResolvedValueOnce(
          (reason === "overflow"
            ? { status: "error", error: "Not enough context" }
            : compiled()) as any,
        )
        .mockResolvedValueOnce(compiled(0.2) as any);
      const stream = vi
        .spyOn(messenger, "llmStreamChat")
        .mockImplementation(async function* () {
          yield [
            { role: "assistant", content: "Summary retaining original goal" },
          ];
          return undefined;
        });
      const result = await store.dispatch(streamNormalInput({}) as any);
      expect(result.type).toBe("chat/streamNormalInput/fulfilled");
      expect(request).toHaveBeenCalledTimes(2);
      expect(stream).toHaveBeenCalledTimes(2);
      expect(
        (store.getState() as any).session.history[1].conversationSummary,
      ).toContain("original goal");
      const messages = (request.mock.calls[1][1] as any).messages;
      expect(
        messages.some((m: any) =>
          JSON.stringify(m.content).includes("Continue with the tests"),
        ),
      ).toBe(true);
      expect((store.getState() as any).session.compactionLoading).toEqual({});
    },
  );
  it("does not compact below threshold", async () => {
    const { store, messenger } = setup();
    vi.spyOn(messenger, "request").mockResolvedValue(compiled(0.5) as any);
    const stream = vi.spyOn(messenger, "llmStreamChat");
    await store.dispatch(streamNormalInput({}) as any);
    expect(stream).toHaveBeenCalledTimes(1);
    expect(
      (store.getState() as any).session.history[1].conversationSummary,
    ).toBeUndefined();
  });
  it("stops after one failed budget recovery", async () => {
    const { store, messenger } = setup();
    const request = vi
      .spyOn(messenger, "request")
      .mockResolvedValue({
        status: "error",
        error: "Not enough context",
      } as any);
    const stream = vi
      .spyOn(messenger, "llmStreamChat")
      .mockImplementation(async function* () {
        yield [{ role: "assistant", content: "Summary" }];
        return undefined;
      });
    await store.dispatch(streamNormalInput({}) as any);
    expect(request).toHaveBeenCalledTimes(2);
    expect(stream).toHaveBeenCalledTimes(1);
    expect((store.getState() as any).session.inlineErrorMessage).toBe(
      "out-of-context",
    );
  });
  it("does not apply a canceled summary or resume generation", async () => {
    const { store, messenger } = setup();
    vi.spyOn(messenger, "request").mockResolvedValue(compiled() as any);
    const stream = vi
      .spyOn(messenger, "llmStreamChat")
      .mockImplementation(async function* () {
        (store.getState() as any).session.streamAborter.abort();
        yield [{ role: "assistant", content: "Stale summary" }];
        return undefined;
      });
    await store.dispatch(streamNormalInput({}) as any);
    expect(stream).toHaveBeenCalledTimes(1);
    expect(
      (store.getState() as any).session.history[1].conversationSummary,
    ).toBeUndefined();
  });
});
