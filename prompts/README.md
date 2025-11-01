# Materials Discovery Lab — Prompt Package

This repository contains markdown prompts for generating the **Mesa + Streamlit** agent-based Materials Discovery Lab simulator using **Replit** and **ChatGPT**.

---

## 📁 Files Included

- **chatgpt_prompt.md** → Prompt for ChatGPT to generate the detailed Replit coding prompt.  
- **replit_prompt.md** → High-level system prompt to paste into Replit’s *Agent 3 chat area* to build the app.  
- **dataset_prompt.md** → Prompt for generating a synthetic materials dataset for use with the simulation.  
- **README.md** → This documentation file.

---

## 🚀 Usage Steps

1. **Prepare GitHub**
   - Create a new GitHub repository (e.g., `materials-discovery-lab`).
   - Upload all `.md` files to it.

2. **Link GitHub to Replit**
   - Open Replit → Link to your GitHub repository (acts as storage).

3. **Generate Code**
   - Open *Agent 3 Chat* in Replit.
   - Copy the full content from `replit_prompt.md` and paste it there.
   - Run the agent to generate the full Mesa + Streamlit codebase automatically.

4. **Generate Dataset**
   - Before running the app, open a new Replit tab or ChatGPT session.
   - Use the content of `dataset_prompt.md` to generate `materials_dataset.csv` (500 rows).

5. **Run Simulation**
   - Upload `materials_dataset.csv` in the Streamlit interface (ValidatorAgent input).
   - Run and visualize your simulation.

---

## 🧠 Notes

- The **dataset_prompt.md** is optional; the app can also generate random materials internally.
- Keep all prompts version-controlled for reproducibility.
