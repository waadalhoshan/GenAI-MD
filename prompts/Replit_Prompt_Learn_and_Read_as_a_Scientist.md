# Replit Coding Prompt — “Learn and Read as a Scientist” (Multi-Agent LLM + Streamlit)

You are an expert Python developer. Generate a small, production-ready **Streamlit** web app that uses a **multi-agent LLM framework** to read, analyze, and teach from scientific papers (PDF or text). Build clean, modular Python with a reproducible workflow. Do not output explanations—just create the project files and code.

## Objectives
- Implement five specialized LLM agents that collaborate to help users critically understand a research paper beyond summarization.
- Provide explainable artifacts: sections, entities, analysis, insights, and an evaluation pass.
- Offer a simple, transparent Streamlit UI that runs the pipeline, shows intermediate artifacts, and exports results.

## Project Structure (files to create)
- `app.py` — Streamlit UI and run orchestration.
- `agents/base.py` — abstract Agent base class with a single run entrypoint.
- `agents/reader.py`, `agents/analyzer.py`, `agents/insight.py`, `agents/evaluator.py`, `agents/visualizer.py` — concrete agents.
- `core/messaging.py` — pydantic data models for messages, artifacts, sections, entities, analysis, insights.
- `core/orchestrator.py` — deterministic pipeline controller with caching and logging.
- `core/llm.py` — provider-agnostic LLM client wrapper (supports OpenAI/Anthropic/Azure via env vars) plus a deterministic FakeLLM for tests.
- `core/pdf_utils.py` — PDF/text ingestion and lightweight sectioning.
- `core/logging_utils.py` — structured JSON logging and run-ID utilities.
- `prompts/reader.md`, `prompts/analyzer.md`, `prompts/insight.md`, `prompts/evaluator.md`, `prompts/visualizer.md` — concise prompt templates that instruct models to return strictly valid JSON for the corresponding schema.
- `tests/test_pipeline.py` — minimal test using FakeLLM.
- `requirements.txt`, `README.md`.

## Agents (functional requirements)
1) **ReaderAgent**
   - Inputs: uploaded PDF or plain text.
   - Tasks: load text; detect common sections (Abstract, Introduction, Methods, Results, Discussion/Conclusion); extract main entities (topics, methods, outcomes, variables).
   - Outputs: a single message containing both section list and entity list as structured artifacts.
   - Failure handling: if section detection is weak, still return chunked text with best-effort headings.

2) **AnalyzerAgent**
   - Inputs: sections + entities.
   - Tasks: determine the research question, methodology, dataset/sample (if any), key findings, metrics, assumptions, limitations, threats to validity, and an overall evidence-strength rating (weak/moderate/strong).
   - Outputs: a structured analysis artifact suitable for downstream use.
   - Constraints: avoid verbatim copying; be concise and neutral.

3) **InsightAgent**
   - Inputs: analysis.
   - Tasks: produce three educational outputs:  
     • Explain — a short, non-technical conceptual explanation.  
     • Question — 3–7 reflection/learning questions.  
     • Apply — 3–7 short, practical ways a learner could use or test the insight.
   - Outputs: a structured insights artifact optimized for readability by non-experts.

4) **EvaluatorAgent**
   - Inputs: analysis + insights.
   - Tasks: review coherence, factual consistency, educational value; propose edits where needed; rank the most valuable points.
   - Outputs: a review artifact with scalar scores (0–1), ranked bullets, and optional suggested rewrites.

5) **VisualizerAgent**
   - Role: formatting helper for final markdown/JSON export and display in the UI tabs.

## Orchestration
- Implement a `Pipeline` that executes: Reader → Analyzer → Insight → Evaluator (with optional refinement applying suggested edits) → Visualizer.
- Provide deterministic behavior via a user-settable seed propagated to all components that accept randomness or temperature.
- Implement content-hash caching keyed by input file + model + temperature; reuse artifacts on reruns.
- Log each step with timestamps, token/size stats (if available), and status.

## LLM Wrapper
- Environment variables: `LLM_PROVIDER` (`openai`, `anthropic`, `azure_openai`, `fake`), `LLM_MODEL`, `LLM_API_KEY`, `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`.
- Unified method to obtain completions; retries with backoff on transient errors.
- Include a `FakeLLM` that returns deterministic canned responses to enable tests and demos without external calls.

## Data & Messaging (describe, don’t embed code)
Define pydantic models for:
- **Section**: name, text, character start/end.
- **Entities**: lists of topics, methods, outcomes, variables.
- **Analysis**: research question, methodology, optional dataset/sample, key findings list, metrics list, assumptions, limitations, threats to validity, evidence-strength category.
- **Insights**: explain (short text), question (list), apply (list).
- **Artifact**: kind identifier and payload dictionary.
- **Message**: id, timestamp, sender, receiver, role, natural-language content, optional artifact, and metadata.

## Streamlit UI (app behavior)
- Sidebar controls: provider, model, temperature, max tokens, seed, “save run” toggle.
- Main flow:
  1. File uploader for PDF or text.
  2. “Run pipeline” button with per-agent progress/status indicators.
  3. Tabs:
     - **Study Summary**: detected sections, entities, concise summary (question, methodology, key findings).
     - **Critical Review**: assumptions, limitations, threats, evidence-strength, and evaluator scores.
     - **Learn & Apply**: the Explain/Question/Apply outputs as clear cards.
- Export buttons:
  - **Download Markdown**: single composed document of analysis + insights + review.
  - **Download JSON**: raw artifacts.
- Extras: show run metadata (run ID, model, seed, elapsed times), “Rerun Analyzer only” toggle to demonstrate modularity, and a tiny built-in sample text for demo when no file is uploaded.

## Prompt Templates (prompts/*.md)
- Each template succinctly states the role, task, and required output schema fields.
- Each ends with a strict instruction to return only valid JSON conforming to the schema, with no commentary.
- Include 1–2 minimal, generic few-shot exemplars directly in the template text (keep very short).

## Non-Functional Requirements
- Clean, typed Python; small cohesive functions; friendly error messages in the UI.
- No secrets in logs or UI. Respect environment variables.
- Minimal dependencies: Streamlit, Pydantic (v2), a PDF parser, dotenv, retries. No vector DB.
- Caching directory `.cache/` and a `run_log.json` alongside artifacts.
- Works offline with `FakeLLM`; gracefully degrades when no API key is present.

## Testing
- `tests/test_pipeline.py` uses `FakeLLM` and a tiny two-paragraph sample to assert:
  - Sections parsed and non-empty.
  - Analysis includes non-empty research question and at least one key finding.
  - Insights include multiple questions and apply items.
  - Evaluator returns scores between 0 and 1.
  - Pipeline returns a run dictionary with run ID and all artifacts.

## README
- Clear run instructions (`pip install -r requirements.txt`, `streamlit run app.py`).
- Environment variable setup, provider switching, caching/logging notes, and limitations (e.g., PDF heuristics, no citation extraction).
- Short “Prompt Trace” section explaining where to find sanitized prompts and artifacts.

## Acceptance Criteria
- Launching `streamlit run app.py` provides the described UI and completes a full run on both: a) provided sample text with `FakeLLM`, b) a small PDF with a real provider when keys are set.
- Artifacts are deterministic for the same seed, model, and temperature.
- Exports produce valid Markdown and JSON that faithfully reflect the displayed artifacts.
- Tests pass using `FakeLLM`.

---

**Generate all files and code that satisfy the above, without including any explanatory text in your output.**
