# ChatGPT Prompt — Generate Replit Coding Prompt

Role: You are an expert Python developer building a small interactive web app using Replit platform and by using Mesa (for agent-based simulation) and Streamlit (for user interaction and visualization).

Goal: Create a Materials Discovery Lab simulator — a web app where a population of digital scientists (agents) explores and improves materials through mutation and evaluation.
The app should let users upload or generate a dataset of candidate materials and observe how the population’s performance evolves over time.

Context: a Mesa agent-based modeling system with the following agents

Agent 1: ValidatorAgent
• Loads a CSV file (or generates data) with four material features and their fixed value boundaries:
  o density: 2.0 – 15.0 (lower is better)
  o hardness: 1.0 – 10.0 (higher is better)
  o conductivity: 0.0 – 100.0 (higher is better)
  o cost: 5.0 – 200.0 (lower is better)
• Ensures all values are within valid bounds.
• Applies min–max normalization to scale all features to [0, 1].
• Prepares the initial population of candidate materials for simulation.

Agent 2: ScientistAgent (N agents)
• Each agent owns one candidate material.
• On each simulation step:
  1. Computes the performance score:
     score = 0.35*hardness_n + 0.35*conductivity_n + 0.20*(1 - density_n) + 0.10*(1 - cost_n)
  2. Mutates 1–2 features by applying small Gaussian noise (≈ ±7%).
  3. Clips new values to the feature bounds, re-normalizes using min–max scaling, and re-evaluates its score.
• Collectively, the ScientistAgents perform a distributed Evolutionary Strategy (ES) optimization process.

Agent 3: AnalyzerAgent
• Aggregates results from all ScientistAgents each step.
• Calculates best score, mean score, and diversity (standard deviation) for each property.
• Maintains a Top-K table of the highest-performing materials.
• Records results for visualization and performance tracking.

Agent 4: ParameterAgent
• Reads and updates runtime parameters from the Streamlit sliders, including:
  o mutation_rate (0.0–0.6, default 0.2)
  o agent_count (50–500, default 200)
  o steps (50–1000, default 200)
  o (Optional) mutation_sigma (0.02–0.15, default 0.07)
• Allows dynamic changes without restarting the simulation.
• Synchronizes parameters across the model and other agents each cycle.

Agent 5: VisualizerAgent
• Integrates with Streamlit to render dynamic results:
  o Line charts of best and average performance scores over time.
  o Diversity charts for property variation.
  o A Top-10 table of best-performing materials.
• Provides user controls: Run, Pause, Step once, and Reset.
• Displays current parameter settings from the ParameterAgent.

Your Task:
Write a clear prompt to Replit’s model to generate proper coding for a multi-agent system using the Mesa package in this context.
The prompt must address the high-level architecture of the proposed system, including these agents, data handling, and real-time Streamlit interface.
I prefer the final system to be implemented in Streamlit for visualization and interactivity.
