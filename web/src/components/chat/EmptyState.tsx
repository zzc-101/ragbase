const EXAMPLES = [
  "挣值分析中 CPI 小于 1 说明什么？",
  "TOGAF 的 ADM 包括哪些阶段？",
  "关键路径法如何计算总浮动时间？",
  "什么是软件架构风格？",
];

export function EmptyState({
  onAsk,
  disabled,
}: {
  onAsk: (question: string) => void;
  disabled?: boolean;
}) {
  return (
    <div className="@container flex min-h-0 flex-1 overflow-y-auto px-4 pb-10">
      <div className="m-auto flex w-full flex-col items-center py-4 text-center">
        <h1 className="text-2xl font-semibold tracking-tight">
          想了解点什么？
        </h1>
        <p className="mt-2 max-w-md text-[15px] leading-relaxed text-muted-foreground">
          我会先在已入库的教材里检索，再据此回答，并标出每句话的出处。
        </p>
        <div className="mt-8 grid w-full max-w-xl gap-2 @md:grid-cols-2">
          {EXAMPLES.map((question) => (
            <button
              key={question}
              type="button"
              disabled={disabled}
              onClick={() => onAsk(question)}
              className="rounded-xl border bg-card px-4 py-3 text-left text-sm leading-relaxed transition-colors duration-150 hover:border-primary/40 hover:bg-accent focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none disabled:opacity-50"
            >
              {question}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
