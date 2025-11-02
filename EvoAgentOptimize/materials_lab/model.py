import numpy as np
import pandas as pd
from mesa import Agent, Model
from typing import Dict, List, Tuple


# Fixed bounds for material properties
BOUNDS = {
    'density': (2.0, 15.0),      # lower is better
    'hardness': (1.0, 10.0),     # higher is better
    'conductivity': (0.0, 100.0), # higher is better
    'cost': (5.0, 200.0)         # lower is better
}

# Performance score weights (applied to normalized values)
WEIGHTS = {
    'hardness': 0.35,
    'conductivity': 0.35,
    'density': 0.20,    # inverted (lower is better)
    'cost': 0.10        # inverted (lower is better)
}


class ValidatorAgent(Agent):
    """Handles data validation and normalization using fixed min-max bounds."""
    
    def __init__(self, model):
        super().__init__(model)
    
    def normalize_value(self, value: float, feature: str) -> float:
        """Normalize a single feature value using fixed bounds."""
        min_val, max_val = BOUNDS[feature]
        # Clip to bounds first
        value = np.clip(value, min_val, max_val)
        # Min-max normalization to [0, 1]
        normalized = (value - min_val) / (max_val - min_val)
        return normalized
    
    def normalize_material(self, material: Dict[str, float]) -> Dict[str, float]:
        """Normalize all features of a material."""
        normalized = {}
        for feature in ['density', 'hardness', 'conductivity', 'cost']:
            normalized[feature + '_n'] = self.normalize_value(material[feature], feature)
        return normalized
    
    def denormalize_value(self, norm_value: float, feature: str) -> float:
        """Convert normalized value back to original scale."""
        min_val, max_val = BOUNDS[feature]
        return norm_value * (max_val - min_val) + min_val
    
    def step(self):
        """Validation step - ensure all materials are properly normalized."""
        pass


class ScientistAgent(Agent):
    """
    Implements evolutionary strategy optimization through Gaussian mutation.
    Each scientist maintains and evolves a single material candidate.
    """
    
    def __init__(self, model, initial_material: Dict[str, float], material_id: int = None):
        super().__init__(model)
        self.material = initial_material.copy()
        self.material_id = material_id if material_id is not None else self.unique_id
        self.validator = model.validator
        
        # Normalize initial material
        normalized = self.validator.normalize_material(self.material)
        self.material.update(normalized)
        
        # Calculate initial score
        self.score = self._calculate_score()
        self.best_score = self.score
        self.mutation_history = []
    
    def _calculate_score(self) -> float:
        """Calculate weighted performance score using normalized values."""
        # Use model's custom weights (or defaults)
        weights = self.model.weights
        score = (
            weights['hardness'] * self.material['hardness_n'] +
            weights['conductivity'] * self.material['conductivity_n'] +
            weights['density'] * (1 - self.material['density_n']) +
            weights['cost'] * (1 - self.material['cost_n'])
        )
        return score
    
    def mutate(self) -> bool:
        """
        Apply evolutionary strategy mutation based on selected strategy:
        - 'gaussian': Basic Gaussian noise with fixed sigma
        - 'adaptive': Gaussian noise with sigma that decreases as score improves
        - 'crossover': Blend features from top 2 candidates
        
        Returns True if mutation was accepted, False otherwise.
        """
        strategy = self.model.mutation_strategy
        
        if strategy == 'crossover':
            return self._mutate_crossover()
        elif strategy == 'adaptive':
            return self._mutate_adaptive()
        else:  # gaussian (default)
            return self._mutate_gaussian()
    
    def _mutate_gaussian(self) -> bool:
        """Apply basic Gaussian mutation with fixed sigma."""
        # Get mutation sigma from model parameters
        mutation_sigma = self.model.mutation_sigma
        
        # Store current state
        old_material = self.material.copy()
        old_score = self.score
        
        # Randomly select 1 or 2 features to mutate
        features = ['density', 'hardness', 'conductivity', 'cost']
        num_features = np.random.choice([1, 2])
        features_to_mutate = np.random.choice(features, size=num_features, replace=False)
        
        # Apply Gaussian mutation to selected features
        for feature in features_to_mutate:
            min_val, max_val = BOUNDS[feature]
            feature_range = max_val - min_val
            
            # Add Gaussian noise scaled to ~7% of feature range
            noise = np.random.normal(0, mutation_sigma * feature_range)
            new_value = self.material[feature] + noise
            
            # Clip to bounds
            new_value = np.clip(new_value, min_val, max_val)
            self.material[feature] = new_value
        
        # Re-normalize mutated features
        normalized = self.validator.normalize_material(self.material)
        self.material.update(normalized)
        
        # Recalculate score
        new_score = self._calculate_score()
        
        # Hill-climbing: accept if improved or equal
        if new_score >= old_score:
            self.score = new_score
            if new_score > self.best_score:
                self.best_score = new_score
            self.mutation_history.append({
                'step': self.model.steps_count,
                'features': list(features_to_mutate),
                'old_score': old_score,
                'new_score': new_score,
                'accepted': True
            })
            return True
        else:
            # Revert mutation
            self.material = old_material
            self.score = old_score
            self.mutation_history.append({
                'step': self.model.steps_count,
                'features': list(features_to_mutate),
                'old_score': old_score,
                'new_score': new_score,
                'accepted': False
            })
            return False
    
    def _mutate_adaptive(self) -> bool:
        """Apply adaptive Gaussian mutation where sigma decreases as score improves."""
        base_sigma = self.model.mutation_sigma
        
        # Adaptive sigma: decreases as score approaches 1.0 (perfect score)
        # At score=0, sigma=base_sigma; at score=1.0, sigma=base_sigma/10
        adaptive_sigma = base_sigma * (1.0 - 0.9 * self.score)
        
        # Store current state
        old_material = self.material.copy()
        old_score = self.score
        
        # Randomly select 1 or 2 features to mutate
        features = ['density', 'hardness', 'conductivity', 'cost']
        num_features = np.random.choice([1, 2])
        features_to_mutate = np.random.choice(features, size=num_features, replace=False)
        
        # Apply adaptive Gaussian mutation
        for feature in features_to_mutate:
            min_val, max_val = BOUNDS[feature]
            feature_range = max_val - min_val
            
            # Add Gaussian noise with adaptive sigma
            noise = np.random.normal(0, adaptive_sigma * feature_range)
            new_value = self.material[feature] + noise
            
            # Clip to bounds
            new_value = np.clip(new_value, min_val, max_val)
            self.material[feature] = new_value
        
        # Re-normalize mutated features
        normalized = self.validator.normalize_material(self.material)
        self.material.update(normalized)
        
        # Recalculate score
        new_score = self._calculate_score()
        
        # Hill-climbing: accept if improved or equal
        if new_score >= old_score:
            self.score = new_score
            if new_score > self.best_score:
                self.best_score = new_score
            self.mutation_history.append({
                'step': self.model.steps_count,
                'features': list(features_to_mutate),
                'old_score': old_score,
                'new_score': new_score,
                'accepted': True,
                'adaptive_sigma': adaptive_sigma
            })
            return True
        else:
            # Revert mutation
            self.material = old_material
            self.score = old_score
            self.mutation_history.append({
                'step': self.model.steps_count,
                'features': list(features_to_mutate),
                'old_score': old_score,
                'new_score': new_score,
                'accepted': False,
                'adaptive_sigma': adaptive_sigma
            })
            return False
    
    def _mutate_crossover(self) -> bool:
        """Apply crossover mutation by blending features from top 2 candidates."""
        # Get top 2 scientists by score
        scientists = self.model.get_scientists()
        if len(scientists) < 2:
            # Fall back to Gaussian if not enough scientists
            return self._mutate_gaussian()
        
        # Sort by score and get top 2
        sorted_scientists = sorted(scientists, key=lambda s: s.score, reverse=True)
        parent1 = sorted_scientists[0]
        parent2 = sorted_scientists[1]
        
        # Store current state
        old_material = self.material.copy()
        old_score = self.score
        
        # Crossover: randomly blend features from top 2
        features = ['density', 'hardness', 'conductivity', 'cost']
        features_modified = []
        
        for feature in features:
            # 50% chance to take from parent1, 50% from parent2
            if np.random.random() < 0.5:
                self.material[feature] = parent1.material[feature]
                features_modified.append(feature + '_p1')
            else:
                self.material[feature] = parent2.material[feature]
                features_modified.append(feature + '_p2')
        
        # Add small Gaussian noise to avoid exact copies
        mutation_sigma = self.model.mutation_sigma * 0.3  # Smaller noise for crossover
        for feature in features:
            min_val, max_val = BOUNDS[feature]
            feature_range = max_val - min_val
            noise = np.random.normal(0, mutation_sigma * feature_range)
            new_value = self.material[feature] + noise
            new_value = np.clip(new_value, min_val, max_val)
            self.material[feature] = new_value
        
        # Re-normalize
        normalized = self.validator.normalize_material(self.material)
        self.material.update(normalized)
        
        # Recalculate score
        new_score = self._calculate_score()
        
        # Hill-climbing: accept if improved or equal
        if new_score >= old_score:
            self.score = new_score
            if new_score > self.best_score:
                self.best_score = new_score
            self.mutation_history.append({
                'step': self.model.steps_count,
                'features': features_modified,
                'old_score': old_score,
                'new_score': new_score,
                'accepted': True,
                'crossover': True
            })
            return True
        else:
            # Revert mutation
            self.material = old_material
            self.score = old_score
            self.mutation_history.append({
                'step': self.model.steps_count,
                'features': features_modified,
                'old_score': old_score,
                'new_score': new_score,
                'accepted': False,
                'crossover': True
            })
            return False
    
    def step(self):
        """Perform one evolutionary step."""
        self.mutate()


class AnalyzerAgent(Agent):
    """Tracks metrics, diversity measures, and maintains Top-K materials."""
    
    def __init__(self, model):
        super().__init__(model)
        self.metrics_history = []
        self.top_k = 10
        self.diversity_history = []
    
    def calculate_diversity(self, scientists: List[ScientistAgent]) -> Dict[str, float]:
        """Calculate diversity metrics across the population."""
        if not scientists:
            return {'std_density': 0, 'std_hardness': 0, 'std_conductivity': 0, 'std_cost': 0, 'mean_std': 0}
        
        # Get all material values
        densities = [s.material['density'] for s in scientists]
        hardnesses = [s.material['hardness'] for s in scientists]
        conductivities = [s.material['conductivity'] for s in scientists]
        costs = [s.material['cost'] for s in scientists]
        
        diversity = {
            'std_density': np.std(densities),
            'std_hardness': np.std(hardnesses),
            'std_conductivity': np.std(conductivities),
            'std_cost': np.std(costs)
        }
        diversity['mean_std'] = np.mean(list(diversity.values()))
        
        return diversity
    
    def get_top_k_materials(self, scientists: List[ScientistAgent]) -> pd.DataFrame:
        """Get top K materials by score."""
        materials_data = []
        for s in scientists:
            materials_data.append({
                'material_id': s.material_id,
                'scientist_id': s.unique_id,
                'density': s.material['density'],
                'hardness': s.material['hardness'],
                'conductivity': s.material['conductivity'],
                'cost': s.material['cost'],
                'score': s.score
            })
        
        df = pd.DataFrame(materials_data)
        df = df.sort_values('score', ascending=False).head(self.top_k)
        df = df.reset_index(drop=True)
        df.index = df.index + 1  # Start index at 1
        return df
    
    def step(self):
        """Analyze current population state."""
        scientists = [agent for agent in self.model.agents 
                     if isinstance(agent, ScientistAgent)]
        
        if not scientists:
            return
        
        scores = [s.score for s in scientists]
        diversity = self.calculate_diversity(scientists)
        
        metrics = {
            'step': self.model.steps_count,
            'best_score': max(scores),
            'mean_score': np.mean(scores),
            'min_score': min(scores),
            'std_score': np.std(scores),
            'diversity': diversity['mean_std']
        }
        
        self.metrics_history.append(metrics)
        self.diversity_history.append(diversity)


class ParameterAgent(Agent):
    """Synchronizes runtime parameters with Streamlit controls."""
    
    def __init__(self, model):
        super().__init__(model)
        self.parameters = {
            'mutation_sigma': 0.07,
            'population_size': 20,
            'num_steps': 100
        }
    
    def update_parameters(self, new_params: Dict):
        """Update parameters from Streamlit UI."""
        self.parameters.update(new_params)
        # Sync with model
        if 'mutation_sigma' in new_params:
            self.model.mutation_sigma = new_params['mutation_sigma']
    
    def step(self):
        """Parameter synchronization step."""
        pass


class VisualizerAgent(Agent):
    """Handles rendering state and UI control flags."""
    
    def __init__(self, model):
        super().__init__(model)
        self.render_data = {
            'charts_ready': False,
            'table_ready': False,
            'status': 'initialized'
        }
    
    def prepare_render_data(self):
        """Prepare data for visualization."""
        self.render_data['charts_ready'] = True
        self.render_data['table_ready'] = True
        self.render_data['status'] = 'ready'
    
    def step(self):
        """Visualization preparation step."""
        self.prepare_render_data()


class MaterialsLabModel(Model):
    """
    Main Mesa model for Materials Discovery Lab.
    Orchestrates evolutionary optimization through specialized agents.
    """
    
    def __init__(self, num_scientists: int = 20, mutation_sigma: float = 0.07, 
                 initial_materials: List[Dict] = None, custom_weights: Dict[str, float] = None,
                 mutation_strategy: str = 'gaussian'):
        super().__init__()
        self.num_scientists = num_scientists
        self.mutation_sigma = mutation_sigma
        self.mutation_strategy = mutation_strategy  # 'gaussian', 'adaptive', or 'crossover'
        self.running = True
        self.steps_count = 0
        
        # Use custom weights if provided, otherwise use defaults
        if custom_weights is not None:
            self.weights = custom_weights
        else:
            self.weights = WEIGHTS.copy()
        
        # Historical playback storage
        self.history_snapshots = []
        
        # Create validator agent (single instance)
        self.validator = ValidatorAgent(self)
        
        # Create parameter agent
        self.parameter_agent = ParameterAgent(self)
        
        # Create analyzer agent
        self.analyzer = AnalyzerAgent(self)
        
        # Create visualizer agent
        self.visualizer = VisualizerAgent(self)
        
        # Create scientist agents with initial materials
        if initial_materials is None:
            initial_materials = self._generate_random_materials(num_scientists)
        
        for i, material in enumerate(initial_materials[:num_scientists]):
            scientist = ScientistAgent(self, material, material_id=i+1)
        
        # Trigger initial analysis to populate metrics
        self.analyzer.step()
        
        # Capture initial snapshot (step 0)
        self.capture_snapshot()
    
    def _generate_random_materials(self, count: int) -> List[Dict[str, float]]:
        """Generate random initial materials within bounds."""
        materials = []
        for _ in range(count):
            material = {
                'density': np.random.uniform(*BOUNDS['density']),
                'hardness': np.random.uniform(*BOUNDS['hardness']),
                'conductivity': np.random.uniform(*BOUNDS['conductivity']),
                'cost': np.random.uniform(*BOUNDS['cost'])
            }
            materials.append(material)
        return materials
    
    def step(self):
        """Advance the model by one step."""
        self.steps_count += 1
        self.agents.shuffle_do("step")
        # Capture snapshot after each step
        self.capture_snapshot()
    
    def get_scientists(self) -> List[ScientistAgent]:
        """Get all scientist agents."""
        return [agent for agent in self.agents 
                if isinstance(agent, ScientistAgent)]
    
    def get_metrics(self) -> List[Dict]:
        """Get metrics history from analyzer."""
        return self.analyzer.metrics_history
    
    def get_top_materials(self) -> pd.DataFrame:
        """Get top K materials."""
        scientists = self.get_scientists()
        return self.analyzer.get_top_k_materials(scientists)
    
    def recalculate_all_scores(self):
        """
        Recalculate scores for all scientists with current weights.
        Resets best_score since old maxima are meaningless under new weights.
        Refreshes analyzer metrics to reflect new scores.
        """
        for scientist in self.get_scientists():
            scientist.score = scientist._calculate_score()
            scientist.best_score = scientist.score
        # Update analyzer metrics to reflect new scores
        self.analyzer.step()
    
    def capture_snapshot(self):
        """Capture current state for historical playback."""
        scientists = self.get_scientists()
        snapshot = {
            'step': self.steps_count,
            'materials': [],
            'metrics': self.get_metrics()[-1] if self.get_metrics() else None
        }
        
        # Capture each scientist's material state
        for s in scientists:
            snapshot['materials'].append({
                'material_id': s.material_id,
                'density': s.material['density'],
                'hardness': s.material['hardness'],
                'conductivity': s.material['conductivity'],
                'cost': s.material['cost'],
                'score': s.score,
                'best_score': s.best_score
            })
        
        self.history_snapshots.append(snapshot)
    
    def get_snapshot(self, step: int) -> Dict:
        """Retrieve snapshot at given step."""
        for snapshot in self.history_snapshots:
            if snapshot['step'] == step:
                return snapshot
        return None
    
    def get_all_snapshots(self) -> List[Dict]:
        """Retrieve all snapshots."""
        return self.history_snapshots
    
    def clear_history(self):
        """Clear all historical snapshots."""
        self.history_snapshots = []
    
    def reset_simulation(self, num_scientists: int = None):
        """Reset the simulation with new random materials."""
        if num_scientists is None:
            num_scientists = self.num_scientists
        
        # Remove all scientist agents
        scientists = self.get_scientists()
        for scientist in scientists:
            scientist.remove()
        
        # Reset step counter
        self.steps_count = 0
        
        # Reset analyzer history
        self.analyzer.metrics_history = []
        self.analyzer.diversity_history = []
        # Clear historical snapshots
        self.clear_history()
        
        # Create new scientist agents
        initial_materials = self._generate_random_materials(num_scientists)
        for i, material in enumerate(initial_materials):
            scientist = ScientistAgent(self, material, material_id=i+1)
        
        # Trigger initial analysis to populate metrics
        self.analyzer.step()
        
        # Capture initial snapshot
        self.capture_snapshot()
