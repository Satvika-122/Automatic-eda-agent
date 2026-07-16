# 🔍 Autonomous EDA Agent

A multi-agent system that automates exploratory data analysis end-to-end — from schema inspection to a grounded, LLM-written narrative summary with an automated QA pass to catch inaccuracies in that narrative.

Built as a Jupyter/Colab notebook using a class-based agent architecture, with **Microsoft's Phi-3-mini-4k-instruct** as the reasoning engine for narrative generation.

---

## What It Does

Given a raw CSV, the system runs a pipeline of specialized agents that each own one part of the analysis, coordinated by a central orchestrator:

| Agent | Responsibility |
|---|---|
| **SchemaAgent** | Inspects rows, columns, dtypes, missing-value %, unique counts, and auto-detects datetime columns |
| **PlotAgent** | Generates histograms for numeric columns, bar charts for top categorical values, a missing-data heatmap, and a time-series plot if datetime columns exist |
| **TextAgent** | Finds long-form text columns and extracts word-frequency insights |
| **TimeSeriesAgent** | Analyzes datetime columns and summarizes event counts over time |
| **NarrativeAgent** | Feeds all structured findings into Phi-3 and generates a grounded, human-readable EDA summary — explicitly prompted to use *only* the collected data, not to speculate |
| **QAAgent** | Cross-checks the generated narrative against the structured findings and flags contradictions (e.g. narrative claims "no numeric columns" when numeric columns exist) |

An **OrchestratorAgent** runs the pipeline and supports goal-based branching — e.g. `"text_focus"` skips time-series analysis, `"time_series_focus"` skips text analysis — so the pipeline only does the work relevant to the dataset and the user's goal.

---

## Architecture

```
                    ┌─────────────────┐
                    │ OrchestratorAgent│
                    └────────┬─────────┘
                             │
        ┌──────────┬─────────┼─────────┬────────────┐
        ▼          ▼         ▼         ▼            ▼
   SchemaAgent  PlotAgent  TextAgent  TimeSeriesAgent │
        │          │         │         │             │
        └──────────┴─────────┴─────────┘             │
                             │                        │
                             ▼                        │
                     NarrativeAgent (Phi-3) ──────────┘
                             │
                             ▼
                        QAAgent
```

All agents read from and write to a shared `AnalysisState` dataclass, which also keeps a running log of every agent action for traceability.

---

## Tech Stack

`Python` · `Microsoft Phi-3-mini-4k-instruct` · `Transformers` · `PyTorch` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn`

---

## Key Design Choices

- **Grounded generation:** the narrative prompt explicitly instructs the model to use *only* the structured findings passed to it, reducing hallucinated claims about the dataset.
- **Self-validating QA:** rather than trusting the LLM's output blindly, a separate QA agent checks it against the actual schema/insights and reports specific issues plus a pass/needs-improvement status.
- **JSON-safe serialization:** a recursive `_make_json_safe` helper converts NumPy types, Timestamps, and other non-serializable objects before they're passed into the LLM prompt.
- **Goal-based branching:** the orchestrator only runs the agents relevant to the selected goal, avoiding unnecessary computation on datasets that don't need certain analyses.

---

## Running It

This is a notebook designed for Google Colab or Jupyter.

```bash
pip install --upgrade pandas numpy matplotlib seaborn transformers torch
```

```python
csv_path = "/content/your_dataset.csv"   # update to your CSV path
df = pd.read_csv(csv_path)

orchestrator = OrchestratorAgent(
    SchemaAgent(), PlotAgent(), TextAgent(),
    TimeSeriesAgent(), NarrativeAgent(), QAAgent()
)

state = orchestrator.run(df, goal="full_eda")

print(state.narrative)   # LLM-generated EDA summary
print(state.qa_report)   # QA pass/fail + flagged issues
```

**Available goals:** `full_eda`, `text_focus`, `time_series_focus`, `bias_analysis`

**Note:** Phi-3-mini is loaded in `float16` with `device_map="auto"` — a CUDA GPU is recommended (e.g. Colab's free GPU runtime); CPU-only execution will be slow.

---

## Output

- Inline plots (histograms, bar charts, missing-data heatmap, time-series chart)
- A structured `AnalysisState` object containing the full schema summary, insights, and logs
- A natural-language EDA narrative
- A QA report flagging any detected inconsistencies between the narrative and the underlying data
