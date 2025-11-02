import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from materials_lab import MaterialsLabModel
import io

# Page configuration
st.set_page_config(
    page_title="Materials Discovery Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.running = False
    st.session_state.step_count = 0
    st.session_state.initialized = False

# Helper function to create model
def create_model(num_scientists, mutation_sigma, initial_materials=None, custom_weights=None, mutation_strategy='gaussian'):
    model = MaterialsLabModel(
        num_scientists=num_scientists,
        mutation_sigma=mutation_sigma,
        initial_materials=initial_materials,
        custom_weights=custom_weights,
        mutation_strategy=mutation_strategy
    )
    # Sync parameters with ParameterAgent
    model.parameter_agent.update_parameters({
        'mutation_sigma': mutation_sigma,
        'population_size': num_scientists
    })
    return model

# Title
st.title("🔬 Materials Discovery Lab")
st.markdown("**Agent-Based Simulation with Evolutionary Strategy Optimization**")

# Sidebar controls
st.sidebar.header("⚙️ Simulation Controls")

# Parameter sliders
st.sidebar.subheader("Parameters")
num_scientists = st.sidebar.slider(
    "Population Size",
    min_value=5,
    max_value=50,
    value=20,
    step=1,
    help="Number of ScientistAgents optimizing materials"
)

mutation_sigma = st.sidebar.slider(
    "Mutation Sigma",
    min_value=0.01,
    max_value=0.20,
    value=0.07,
    step=0.01,
    help="Standard deviation for Gaussian mutation (~7% of feature range)"
)

mutation_strategy = st.sidebar.selectbox(
    "Mutation Strategy",
    options=['gaussian', 'adaptive', 'crossover'],
    index=0,
    help="Gaussian: Fixed noise | Adaptive: Decreasing noise | Crossover: Blend top 2"
)

num_steps_to_run = st.sidebar.slider(
    "Steps per Run",
    min_value=1,
    max_value=100,
    value=10,
    step=1,
    help="Number of evolutionary steps to execute when 'Run' is pressed"
)

# Performance weight customization
st.sidebar.subheader("Performance Weights")
st.sidebar.caption("Adjust optimization priorities (auto-normalized to sum=1.0)")

weight_hardness = st.sidebar.slider(
    "Hardness Weight",
    min_value=0.0,
    max_value=1.0,
    value=0.35,
    step=0.05,
    help="Weight for hardness (higher is better)"
)

weight_conductivity = st.sidebar.slider(
    "Conductivity Weight",
    min_value=0.0,
    max_value=1.0,
    value=0.35,
    step=0.05,
    help="Weight for conductivity (higher is better)"
)

weight_density = st.sidebar.slider(
    "Density Weight",
    min_value=0.0,
    max_value=1.0,
    value=0.20,
    step=0.05,
    help="Weight for density (lower is better)"
)

weight_cost = st.sidebar.slider(
    "Cost Weight",
    min_value=0.0,
    max_value=1.0,
    value=0.10,
    step=0.05,
    help="Weight for cost (lower is better)"
)

# Normalize weights to sum to 1.0
total_weight = weight_hardness + weight_conductivity + weight_density + weight_cost
if total_weight > 0:
    weights = {
        'hardness': weight_hardness / total_weight,
        'conductivity': weight_conductivity / total_weight,
        'density': weight_density / total_weight,
        'cost': weight_cost / total_weight
    }
    st.sidebar.caption(f"Normalized: H={weights['hardness']:.2f}, C={weights['conductivity']:.2f}, D={weights['density']:.2f}, $={weights['cost']:.2f}")
else:
    weights = {'hardness': 0.35, 'conductivity': 0.35, 'density': 0.20, 'cost': 0.10}

# Sync parameters and weights with model if it exists
if st.session_state.initialized and st.session_state.model:
    st.session_state.model.parameter_agent.update_parameters({
        'mutation_sigma': mutation_sigma,
        'population_size': num_scientists,
        'num_steps': num_steps_to_run
    })
    # Update mutation strategy dynamically
    st.session_state.model.mutation_strategy = mutation_strategy
    # Update weights dynamically and recalculate all scores
    old_weights = st.session_state.model.weights.copy()
    st.session_state.model.weights = weights
    # Only recalculate if weights actually changed
    if old_weights != weights:
        st.session_state.model.recalculate_all_scores()

# File uploader
st.sidebar.subheader("Initial Materials")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV (optional)",
    type=['csv'],
    help="CSV with columns: density, hardness, conductivity, cost"
)

initial_materials = None
if uploaded_file is not None:
    try:
        df_upload = pd.read_csv(uploaded_file)
        required_cols = ['density', 'hardness', 'conductivity', 'cost']
        if all(col in df_upload.columns for col in required_cols):
            initial_materials = df_upload[required_cols].to_dict('records')
            st.sidebar.success(f"✅ Loaded {len(initial_materials)} materials")
        else:
            st.sidebar.error(f"❌ CSV must have columns: {', '.join(required_cols)}")
    except Exception as e:
        st.sidebar.error(f"❌ Error reading CSV: {str(e)}")

# Control buttons
st.sidebar.subheader("Actions")

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("▶️ Run", use_container_width=True):
        if not st.session_state.initialized:
            st.session_state.model = create_model(num_scientists, mutation_sigma, initial_materials, weights, mutation_strategy)
            st.session_state.initialized = True
        st.session_state.running = True

with col2:
    if st.button("⏸️ Pause", use_container_width=True):
        st.session_state.running = False

col3, col4 = st.sidebar.columns(2)
with col3:
    if st.button("⏭️ Step", use_container_width=True):
        if not st.session_state.initialized:
            st.session_state.model = create_model(num_scientists, mutation_sigma, initial_materials, weights, mutation_strategy)
            st.session_state.initialized = True
        if st.session_state.model:
            st.session_state.model.step()
            st.session_state.step_count = st.session_state.model.steps_count
        st.rerun()

with col4:
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.model = create_model(num_scientists, mutation_sigma, initial_materials, weights, mutation_strategy)
        st.session_state.running = False
        st.session_state.step_count = 0
        st.session_state.initialized = True
        st.rerun()

# Run simulation if running
if st.session_state.running and st.session_state.initialized:
    for _ in range(num_steps_to_run):
        st.session_state.model.step()
    st.session_state.step_count = st.session_state.model.steps_count
    st.session_state.running = False
    st.rerun()

# Playback controls
st.sidebar.divider()
st.sidebar.subheader("📽️ Historical Playback")

if 'playback_mode' not in st.session_state:
    st.session_state.playback_mode = False
if 'playback_step' not in st.session_state:
    st.session_state.playback_step = 0
if 'playback_playing' not in st.session_state:
    st.session_state.playback_playing = False

# Playback mode toggle
playback_enabled = st.sidebar.checkbox(
    "Enable Playback Mode",
    value=st.session_state.playback_mode,
    help="Review material evolution through time"
)
st.session_state.playback_mode = playback_enabled

if st.session_state.playback_mode and st.session_state.initialized and st.session_state.model:
    snapshots = st.session_state.model.get_all_snapshots()
    if snapshots:
        max_step = len(snapshots) - 1
        playback_step = st.sidebar.slider(
            "Review Step",
            min_value=0,
            max_value=max_step,
            value=min(st.session_state.playback_step, max_step),
            help="Navigate through simulation history"
        )
        st.session_state.playback_step = playback_step
        
        # Playback control buttons
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("▶️ Play", use_container_width=True):
                st.session_state.playback_playing = True
                st.rerun()
        with col2:
            if st.button("⏸️ Stop", use_container_width=True):
                st.session_state.playback_playing = False
        
        # Auto-advance if playing
        if st.session_state.playback_playing:
            if st.session_state.playback_step < max_step:
                st.session_state.playback_step += 1
                st.rerun()
            else:
                st.session_state.playback_playing = False
    else:
        st.sidebar.info("No history yet. Run simulation to generate snapshots.")

# Main dashboard
if st.session_state.initialized and st.session_state.model:
    model = st.session_state.model
    
    # Determine if using playback data or live data
    if st.session_state.playback_mode and model.get_all_snapshots():
        snapshot = model.get_all_snapshots()[st.session_state.playback_step]
        display_step = snapshot['step']
        display_mode = "📽️ Playback"
    else:
        snapshot = None
        display_step = st.session_state.step_count
        display_mode = "🔴 Live"
    
    # Status bar
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Mode", display_mode)
    with col2:
        st.metric("Step", display_step)
    with col3:
        strategy_display = model.mutation_strategy.capitalize()
        st.metric("Mutation Strategy", strategy_display)
    with col4:
        if snapshot:
            # In playback mode, only use snapshot data
            if snapshot['metrics']:
                current_best = snapshot['metrics']['best_score']
                st.metric("Best Score", f"{current_best:.4f}")
            else:
                st.metric("Best Score", "N/A")
        elif model.get_metrics():
            # In live mode, use current metrics
            current_best = model.get_metrics()[-1]['best_score']
            st.metric("Best Score", f"{current_best:.4f}")
        else:
            st.metric("Best Score", "N/A")
    with col5:
        if snapshot:
            # In playback mode, only use snapshot data
            if snapshot['metrics']:
                current_mean = snapshot['metrics']['mean_score']
                st.metric("Mean Score", f"{current_mean:.4f}")
            else:
                st.metric("Mean Score", "N/A")
        elif model.get_metrics():
            # In live mode, use current metrics
            current_mean = model.get_metrics()[-1]['mean_score']
            st.metric("Mean Score", f"{current_mean:.4f}")
        else:
            st.metric("Mean Score", "N/A")
    
    st.divider()
    
    # Charts section
    if model.get_metrics():
        # Use full metrics history for charts (up to playback step if in playback mode)
        if snapshot:
            # In playback mode: show metrics up to selected step
            all_snapshots = model.get_all_snapshots()
            metrics_up_to_step = [s['metrics'] for s in all_snapshots[:st.session_state.playback_step + 1] if s['metrics']]
            metrics_df = pd.DataFrame(metrics_up_to_step)
        else:
            # Live mode: show all metrics
            metrics_df = pd.DataFrame(model.get_metrics())
        
        # Create three charts: Best/Mean Score, Diversity, Score Distribution
        st.subheader("📊 Evolution Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Best and Mean Score Evolution
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=metrics_df['step'],
                y=metrics_df['best_score'],
                mode='lines+markers',
                name='Best Score',
                line=dict(color='green', width=2)
            ))
            fig1.add_trace(go.Scatter(
                x=metrics_df['step'],
                y=metrics_df['mean_score'],
                mode='lines+markers',
                name='Mean Score',
                line=dict(color='blue', width=2)
            ))
            
            # Add marker for current playback position
            if snapshot:
                fig1.add_vline(x=display_step, line_dash="dash", line_color="red", 
                              annotation_text="Current", annotation_position="top")
            
            fig1.update_layout(
                title="Score Evolution",
                xaxis_title="Step",
                yaxis_title="Performance Score",
                hovermode='x unified',
                height=350
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Diversity Evolution
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=metrics_df['step'],
                y=metrics_df['diversity'],
                mode='lines+markers',
                name='Diversity',
                line=dict(color='purple', width=2),
                fill='tozeroy'
            ))
            
            # Add marker for current playback position
            if snapshot:
                fig2.add_vline(x=display_step, line_dash="dash", line_color="red",
                              annotation_text="Current", annotation_position="top")
            
            fig2.update_layout(
                title="Population Diversity",
                xaxis_title="Step",
                yaxis_title="Mean Std Deviation",
                hovermode='x unified',
                height=350
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Feature distributions
        st.subheader("📈 Feature Distributions")
        
        # Use snapshot data if in playback mode
        if snapshot and snapshot['materials']:
            densities = [m['density'] for m in snapshot['materials']]
            hardnesses = [m['hardness'] for m in snapshot['materials']]
            conductivities = [m['conductivity'] for m in snapshot['materials']]
            costs = [m['cost'] for m in snapshot['materials']]
        else:
            scientists = model.get_scientists()
            densities = [s.material['density'] for s in scientists]
            hardnesses = [s.material['hardness'] for s in scientists]
            conductivities = [s.material['conductivity'] for s in scientists]
            costs = [s.material['cost'] for s in scientists]
        
        if densities:  # Check if we have data to display
            fig3 = make_subplots(
                rows=1, cols=4,
                subplot_titles=('Density', 'Hardness', 'Conductivity', 'Cost')
            )
            
            fig3.add_trace(
                go.Histogram(x=densities, name='Density', marker_color='lightblue', nbinsx=15),
                row=1, col=1
            )
            fig3.add_trace(
                go.Histogram(x=hardnesses, name='Hardness', marker_color='lightgreen', nbinsx=15),
                row=1, col=2
            )
            fig3.add_trace(
                go.Histogram(x=conductivities, name='Conductivity', marker_color='lightyellow', nbinsx=15),
                row=1, col=3
            )
            fig3.add_trace(
                go.Histogram(x=costs, name='Cost', marker_color='lightcoral', nbinsx=15),
                row=1, col=4
            )
            
            fig3.update_layout(
                showlegend=False,
                height=300
            )
            fig3.update_xaxes(title_text="Density", row=1, col=1)
            fig3.update_xaxes(title_text="Hardness", row=1, col=2)
            fig3.update_xaxes(title_text="Conductivity", row=1, col=3)
            fig3.update_xaxes(title_text="Cost", row=1, col=4)
            
            st.plotly_chart(fig3, use_container_width=True)
    
    st.divider()
    
    # Top-10 Materials Table
    st.subheader("🏆 Top 10 Materials")
    
    # Use snapshot data if in playback mode
    if snapshot and snapshot['materials']:
        # Create DataFrame from snapshot materials
        materials_list = snapshot['materials']
        # Sort by score and take top 10
        sorted_materials = sorted(materials_list, key=lambda x: x['score'], reverse=True)[:10]
        top_materials = pd.DataFrame(sorted_materials)
    else:
        top_materials = model.get_top_materials()
    
    if not top_materials.empty:
        # Format the dataframe for display
        display_df = top_materials.copy()
        display_df['density'] = display_df['density'].round(3)
        display_df['hardness'] = display_df['hardness'].round(3)
        display_df['conductivity'] = display_df['conductivity'].round(3)
        display_df['cost'] = display_df['cost'].round(3)
        display_df['score'] = display_df['score'].round(4)
        
        # Highlight the best material (first row)
        def highlight_best(s):
            return ['background-color: lightgreen' if s.name == display_df.index[0] else '' for _ in s]
        
        st.dataframe(
            display_df.style.apply(highlight_best, axis=1),
            use_container_width=True,
            hide_index=False
        )
        
        # Export functionality
        st.subheader("💾 Export Results")
        col1, col2 = st.columns(2)
        
        with col1:
            # Export top materials
            csv_buffer = io.StringIO()
            top_materials.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Top Materials (CSV)",
                data=csv_buffer.getvalue(),
                file_name=f"top_materials_step_{st.session_state.step_count}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Export metrics
            if model.get_metrics():
                metrics_csv = io.StringIO()
                metrics_df.to_csv(metrics_csv, index=False)
                st.download_button(
                    label="📥 Download Metrics (CSV)",
                    data=metrics_csv.getvalue(),
                    file_name=f"metrics_step_{st.session_state.step_count}.csv",
                    mime="text/csv"
                )
    else:
        st.info("No materials to display yet. Click 'Run' or 'Step' to start the simulation.")

else:
    # Welcome screen
    st.info("👈 Configure parameters and click **Run** or **Step** to start the simulation")
    
    st.markdown("""
    ### About This Simulation
    
    This agent-based model simulates evolutionary optimization of materials using **evolutionary strategy**.
    
    **Agents:**
    - **ScientistAgents**: Evolve materials using Gaussian mutation (±7% sigma) on 1-2 random features per step
    - **ValidatorAgent**: Normalizes data using fixed min-max bounds
    - **AnalyzerAgent**: Tracks metrics, diversity, and Top-K materials
    - **ParameterAgent**: Synchronizes runtime parameters with UI controls
    - **VisualizerAgent**: Manages rendering and UI state
    
    **Material Properties:**
    - **Density** (2.0 - 15.0): Lower is better
    - **Hardness** (1.0 - 10.0): Higher is better
    - **Conductivity** (0.0 - 100.0): Higher is better
    - **Cost** (5.0 - 200.0): Lower is better
    
    **Performance Score** (on normalized values):
    ```
    score = 0.35×hardness + 0.35×conductivity + 0.20×(1-density) + 0.10×(1-cost)
    ```
    
    **Optimization Strategy:**
    - Each scientist applies Gaussian mutation to 1-2 random features
    - Hill-climbing selection: Keep mutated material if score improves
    - Population evolves toward optimal material properties over time
    """)

# Footer
st.sidebar.divider()
st.sidebar.markdown("**Materials Discovery Lab v1.0**")
st.sidebar.caption("Powered by Mesa + Streamlit")
