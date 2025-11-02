# Materials Discovery Lab

## Overview

This is an agent-based simulation system built with Mesa and Streamlit that models a materials science research laboratory. The simulation uses evolutionary strategies to discover optimal material compositions through iterative mutation and selection. Scientists (agents) mutate material candidates to improve a weighted performance score across four key properties: density, hardness, conductivity, and cost. The system demonstrates how multi-agent collaboration can optimize complex multi-objective problems in materials discovery.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application
- **Rationale**: Streamlit provides rapid prototyping for data science applications with minimal code, making it ideal for interactive scientific simulations
- **Key Components**:
  - Interactive sidebar with parameter controls (sliders, file upload)
  - Real-time visualization dashboard using Plotly for live charts
  - Control buttons for simulation state management (Run, Pause, Step, Reset)
  - Data tables displaying top-performing materials
- **State Management**: Uses Streamlit's session_state to persist model instance and simulation status across reruns

### Backend Architecture
- **Framework**: Mesa agent-based modeling framework
- **Design Pattern**: Multi-agent system with specialized agent roles
- **Agent Types**:
  1. **ScientistAgent**: Performs evolutionary mutations on material candidates using Gaussian noise
  2. **ValidatorAgent**: Handles data normalization and validation with fixed min-max bounds
  3. **AnalyzerAgent**: Computes performance metrics and analyzes material properties
  4. **ParameterAgent**: Manages and synchronizes simulation parameters
  5. **VisualizerAgent**: Prepares data for visualization components
  
- **Optimization Strategy**: Hill-climbing evolutionary algorithm where agents mutate 1-2 random features per step and keep improvements (greedy selection)
- **Performance Evaluation**: Multi-objective weighted scoring function:
  - 35% hardness (normalized, higher better)
  - 35% conductivity (normalized, higher better)
  - 20% inverted density (normalized, lower better)
  - 10% inverted cost (normalized, lower better)

### Data Architecture
- **Material Representation**: Dictionary-based with four numeric features
- **Feature Bounds** (hard constraints):
  - Density: 2.0 - 15.0
  - Hardness: 1.0 - 10.0
  - Conductivity: 0.0 - 100.0
  - Cost: 5.0 - 200.0
- **Normalization**: Fixed min-max scaling to [0, 1] using predetermined bounds (not data-dependent)
- **Rationale**: Fixed bounds ensure consistent normalization across different datasets and simulation runs, preventing score drift

### Mutation Mechanism
- **Algorithm**: Gaussian noise addition with configurable sigma (default ~0.07)
- **Mutation Scope**: Randomly selects 1-2 features per mutation event
- **Noise Scale**: Sigma represents ~7% of feature range for controlled exploration
- **Constraint Handling**: Post-mutation clipping to enforce hard bounds
- **Selection Pressure**: Greedy acceptance (keep if score ≥ previous)

### Architectural Trade-offs
- **Pros of Mesa Framework**:
  - Built-in agent scheduling and model structure
  - Clean separation of concerns between agent types
  - Easy to extend with new agent behaviors
- **Cons**:
  - Overhead for simple simulations
  - Less performance than pure NumPy for large populations
- **Pros of Streamlit**:
  - Rapid development with automatic UI reactivity
  - Built-in widget library for controls
  - Easy integration with Python data science stack
- **Cons**:
  - Full page reruns can be inefficient
  - Limited customization compared to traditional web frameworks

## External Dependencies

### Python Libraries
- **mesa**: Agent-based modeling framework - core simulation engine
- **streamlit**: Web application framework - user interface
- **pandas**: Data manipulation and analysis - material data handling
- **numpy**: Numerical computing - mathematical operations and mutations
- **plotly**: Interactive visualization library - real-time charts and graphs

### No External Services
This application runs entirely self-contained with no external APIs, databases, or cloud services. All data is maintained in-memory during simulation runtime with optional CSV file import/export through Streamlit's file upload widget.

### Development Environment
- Designed to run in Replit with nix environment
- GitHub integration for version control and deployment
- No build process or compilation required (pure Python)