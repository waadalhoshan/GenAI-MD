# ChatGPT Prompt — Generate Replit Coding Prompt (Multi‑Agent “Learn and Read as a Scientist”)

**Role**  
You are an expert Python developer building a small interactive web app on **Replit**. The app uses a **multi‑agent framework** powered by **LLMs** to help users *read like a scientist*. Use **Streamlit** for the user interface and to orchestrate agent runs.

---

## Goal
Build a web app where several specialized agents collaborate to process scientific papers (PDF or plain text). The system should:
1) Identify **what** the study is about, **how** it was conducted, and **what** it found.  
2) Analyze the **strength of evidence** and highlight **limitations/assumptions**.  
3) Transform understanding into **teachable insights**: concise explanations, learning questions, and practical steps to apply findings.

The outcome moves users from passive reading to active comprehension and reflection.

---

## Multi‑Agent Context (no Mesa)
Design an **LLM‑based multi‑agent architecture** with explicit, typed message passing. Each agent is a Python class with a `run(input: dict) -> dict` interface and clear responsibilities. Agents exchange **JSON messages** over a minimal in‑process message bus.

### Agent 1 — ReaderAgent
- **Input:** uploaded PDF or text.  
- **Tasks:** extract text; split into logical sections (Abstract, Intro, Methods, Results, Discussion, Conclusion); detect entities (topic, variables, outcomes).  
- **Output JSON:** `{ "sections": [...], "entities": {...}, "paper_meta": {...} }`

### Agent 2 — AnalyzerAgent
- **Input:** ReaderAgent output.  
- **Tasks:** identify research questions/hypotheses; summarize methodology (design, sample size, metrics); summarize key findings.  
- **Evidence & limits:** score strength (e.g., scale 0–5) using heuristics (sample adequacy, controls, transparency, effect sizes reported); list assumptions/limitations and potential biases.  
- **Output JSON:** `{ "about": "...", "methods": {...}, "findings": {...}, "evidence_score": n, "limitations": [...], "assumptions": [...] }`

### Agent 3 — InsightAgent
- **Input:** AnalyzerAgent output.  
- **Tasks:** produce **teaching artifacts**:  
  - `explain_it` — layperson-friendly explanation (200–300 words).  
  - `question_it` — 5–8 reflection/quiz questions.  
  - `apply_it` — 3–5 actionable steps or scenarios to apply/test the findings.  
- **Output JSON:** `{ "explain_it": "...", "question_it": [...], "apply_it": [...] }`

### Agent 4 — EvaluatorAgent
- **Input:** Analyzer + Insight outputs.  
- **Tasks:** check coherence and factual consistency against the source sections; flag contradictions; rank clarity/educational value; suggest edits.  
- **Output JSON:** `{ "consistency_report": {...}, "quality_scores": {...}, "suggested_edits": {...} }`

### Agent 5 — Orchestrator/Coordinator
- **Tasks:** pipe messages among agents; maintain state; allow re-runs of individual agents; merge final payload for display/export.  
- **Output JSON (final):** combines Reader, Analyzer, Insight, and Evaluator artifacts.

---

## Architecture & Requirements
- **Language/stack:** Python 3.10+, Streamlit UI, pydantic for schemas, pdfminer or pypdf for PDF text extraction.  
- **LLM Provider Abstraction:** create a `LLMClient` interface (e.g., OpenAI-compatible) with rate‑limit handling and simple caching (on‑disk JSON cache keyed by prompt hash).  
- **Typed Schemas:** Define pydantic models for all agent inputs/outputs to guarantee structure and validation.  
- **Message Bus:** simple in‑memory pub/sub or sequential orchestration class; log all messages to a timeline for transparency.  
- **Determinism:** set temperature in prompts; store prompts and system roles alongside outputs.  
- **Safety:** strip PII; reject encrypted/protected PDFs; handle long docs with chunking + section headers.

---

## Streamlit UI Spec
- **Main layout:** sidebar for controls; main area with tabs:
  - **Upload**: file uploader (PDF, .txt) and/or URL fetch (optional).  
  - **Study Summary**: *about, methods, findings* (from Analyzer).  
  - **Critical Review**: evidence score, limitations, assumptions, evaluator consistency report.  
  - **Learn & Apply**: Explain It / Question It / Apply It cards; copy buttons.  
  - **Timeline**: chronological agent messages (collapsible JSON).  
- **Controls:** Run all, Re‑run selected agent, Clear state.  
- **Export:** download final report as **Markdown** and **JSON**; optional PDF export using `reportlab` or HTML→PDF.  
- **Status indicators:** spinners per agent; success/warning badges based on Evaluator scores.

---

## Prompts (examples to include in code)
- **ReaderAgent system prompt:** “You segment scholarly articles into standard sections and extract entities (topic, variables, outcomes). Return JSON matching the ReaderOutput model.”  
- **AnalyzerAgent system prompt:** “You analyze research rigor. Identify the research question, methods, findings; evaluate evidence strength (0–5) with brief justification; list limitations and assumptions. Return AnalyzerOutput JSON.”  
- **InsightAgent system prompt:** “You convert analyses into educational artifacts: Explain It (200–300 words), Question It (5–8 items), Apply It (3–5 steps). Return InsightOutput JSON.”  
- **EvaluatorAgent system prompt:** “You check coherence and factual alignment with source excerpts. Produce a consistency report, quality scores, and suggested edits. Return EvaluatorOutput JSON.”

---

## Deliverables (Replit project structure)
```
/app
  main.py                # Streamlit entry point
  agents/
    __init__.py
    reader.py            # ReaderAgent
    analyzer.py          # AnalyzerAgent
    insight.py           # InsightAgent
    evaluator.py         # EvaluatorAgent
    orchestrator.py      # Coordinator + message bus
  llm/
    client.py            # LLMClient abstraction + caching
  models/
    schemas.py           # pydantic models for IO
  ui/
    components.py        # reusable Streamlit components
  utils/
    pdf.py               # PDF/text loaders & chunkers
    io.py                # export to md/json/pdf
  prompts/
    reader.txt
    analyzer.txt
    insight.txt
    evaluator.txt
requirements.txt         # streamlit, pydantic, pypdf, pdfminer.six (optional), tiktoken, openai-compatible client
README.md
```

**README.md** should explain setup, environment variables for the LLM key, and example usage.

---

## “Your Task” (to Replit’s code model)
> Implement the multi‑agent system and Streamlit UI exactly as specified above.  
> Provide runnable code with stub LLM calls that can be swapped with a real provider.  
> Include unit tests for schema validation and a mock PDF to demonstrate end‑to‑end flow.  
> Ensure all agents read/write the defined JSON schemas and that the Orchestrator can re‑run any agent without resetting the entire pipeline.

---

## Acceptance Criteria
- Upload a PDF and obtain structured **Study Summary**, **Critical Review**, and **Learn & Apply** outputs.  
- Evidence score and limitations are present and justified.  
- Exports work (Markdown + JSON).  
- Timeline shows agent messages; re-run of a single agent updates downstream panels.  
- Code passes basic tests and runs locally via `streamlit run app/main.py`.
