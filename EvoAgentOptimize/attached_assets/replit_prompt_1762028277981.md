# Replit Prompt — Materials Discovery Lab (Mesa + Streamlit)

**Instructions:**
1. Link your Replit project to your GitHub repository (use GitHub as storage).
2. Open Replit’s **Agent 3 chat area**.
3. Copy and paste this prompt completely.
4. Run the agent to generate the Mesa + Streamlit code automatically.
5. Commit and push your generated files to GitHub.

---

You are an expert Python engineer. Generate a **working Streamlit app** that uses **Mesa** for an agent-based simulation called **“Materials Discovery Lab”**. The app must run end-to-end in Replit with one-click “Run” and render a live dashboard in Streamlit.

### Functional Overview (must implement exactly)
We simulate a population of **ScientistAgents** that iteratively mutate material candidates to improve a weighted performance score. A **ValidatorAgent** cleans/normalizes data, an **AnalyzerAgent** tracks metrics and Top-K, a **ParameterAgent** syncs runtime parameters with Streamlit sliders, and a **VisualizerAgent** handles rendering and UI controls.

#### Data schema + bounds (hard requirements)
A material has four numeric features **with fixed value boundaries**:
- `density`: 2.0 – 15.0 (lower is better)
- `hardness`: 1.0 – 10.0 (higher is better)
- `conductivity`: 0.0 – 100.0 (higher is better)
- `cost`: 5.0 – 200.0 (lower is better)

Normalization: **min–max** to `[0, 1]` per feature using the fixed bounds above (not dataset min/max).

#### Performance score (use normalized values)
```
score = 0.35*hardness_n + 0.35*conductivity_n + 0.20*(1 - density_n) + 0.10*(1 - cost_n)
```

#### Mutation behavior
- Each step, each ScientistAgent mutates **1–2 randomly chosen features** by adding Gaussian noise with sigma = `mutation_sigma` (≈ 0.07 default), scaled to ~±7% of the feature’s range.
- After mutation: **clip to bounds**, re-normalize with the same fixed min–max, recompute score.
- Keep the improved candidate if new score ≥ old score (simple hill-climb); otherwise revert.

### Architecture & Classes (exact names)
- `materials_lab/model.py`
  - `class MaterialsLabModel(Model)`
  - `class ValidatorAgent(Agent)`
  - `class ScientistAgent(Agent)`
  - `class AnalyzerAgent(Agent)`
  - `class ParameterAgent(Agent)`
  - `class VisualizerAgent(Agent)`

### Streamlit Integration
- Sidebar: sliders for parameters, file uploader, and run controls.
- Main page: live charts (best score, mean score, diversity), top-10 materials table, and simulation status.
- Buttons: **Run**, **Pause**, **Step Once**, **Reset**.

### Output Expectation
- `app.py` for Streamlit UI.
- `materials_lab/` module containing agent and model code.
- Complete `requirements.txt` with: `mesa`, `streamlit`, `pandas`, `numpy`.

Generate the full codebase now.
