# python optimize_k_means.py --input /home/dev/work_main/random/assesment/all_criteria_scored/gemma/cv_scores_google_gemma-3-12b-it_merged.csv --output_dir /home/dev/work_main/random/assesment/k_means_selected --experiment_name gemma3-12B_cv --n_rounds 100 --k_categories 10 --feature_mode concat

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, precision_score, recall_score, roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
import os
from datetime import datetime
import random
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Professional metric naming mapping (underscore-free, publication-ready)
METRIC_LABELS = {
    'silhouette': 'Silhouette Score',
    'school_roc': 'School ROC AUC',
    'project_roc': 'Project ROC AUC',
    'avg_roc': 'Avg. ROC AUC',
    'school_f1': 'School F1 Score',
    'project_f1': 'Project F1 Score',
    'avg_f1': 'Avg. F1 Score'
}

METRIC_COLORS = {
    'silhouette': 'steelblue',
    'school_roc': 'seagreen',
    'project_roc': 'coral',
    'avg_roc': 'darkorchid',
    'school_f1': 'seagreen',
    'project_f1': 'coral',
    'avg_f1': 'darkorchid'
}

class CategorySamplingExperiment:
    def __init__(self, input_csv: str, output_dir: str, experiment_name: str, 
                 n_rounds: int = 100, k_categories: int = 10, feature_mode: str = 'mean',
                 evolutionary_search=False,elite_size=10, offspring_per_elite=10,mutation_rate=0.5):
        """
        Initialize experiment focused on unsupervised-supervised correlation analysis.
        """
        self.input_csv = input_csv
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.n_rounds = n_rounds
        self.k_categories = k_categories
        self.feature_mode = feature_mode

        self.best_silhouette = -1.0

        self.evolutionary_search = evolutionary_search
        self.elite_size = elite_size
        self.offspring_per_elite = offspring_per_elite
        self.mutation_rate = mutation_rate  # 0.5 = keep 50%, randomize 50%
        self.elite_pool = []  # List of (categories, silhouette_score)
        
        os.makedirs(output_dir, exist_ok=True)
        self.setup_logging()
        self.data = pd.read_csv(input_csv)
        
        # Validate targets
        required_targets = ['school_participation_flag', 'project_participation_flag']
        for target in required_targets:
            if target not in self.data.columns:
                raise ValueError(f"Required target column '{target}' not found in CSV")
        
        # Get category columns
        self.all_categories = [col for col in self.data.columns 
                              if col not in ['document_name'] + required_targets]
        
        if len(self.all_categories) < self.k_categories:
            raise ValueError(f"Need at least {self.k_categories} categories, but only found {len(self.all_categories)}")
        
        # Diagnose class imbalance
        self.diagnose_class_imbalance()
        
        # Initialize results with professional metric names
        self.results = {
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'k_categories': self.k_categories,
            'feature_mode': self.feature_mode,
            'class_distribution': {
                'school_participation_flag': self.get_class_distribution('school_participation_flag'),
                'project_participation_flag': self.get_class_distribution('project_participation_flag')
            },
            'rounds': [],
            'metric_definitions': {
                'silhouette_score': 'Clustering quality metric (unsupervised)',
                'school_roc_auc': 'ROC AUC for school participation prediction',
                'project_roc_auc': 'ROC AUC for project participation prediction',
                'avg_roc_auc': 'Average ROC AUC across both targets',
                'school_f1': 'F1 score for school participation prediction',
                'project_f1': 'F1 score for project participation prediction',
                'avg_f1': 'Average F1 score across both targets'
            }
        }
        
        # Track best performers
        self.best_metrics = {
            'silhouette': {'score': -1.0, 'categories': [], 'round': -1},
            'school_roc_auc': {'score': -1.0, 'categories': [], 'round': -1},
            'project_roc_auc': {'score': -1.0, 'categories': [], 'round': -1},
            'avg_roc_auc': {'score': -1.0, 'categories': [], 'round': -1},
            'school_f1': {'score': -1.0, 'categories': [], 'round': -1},
            'project_f1': {'score': -1.0, 'categories': [], 'round': -1},
            'avg_f1': {'score': -1.0, 'categories': [], 'round': -1}
        }
        
        self.prev_best = {metric: -1.0 for metric in self.best_metrics.keys()}
        self.logger.info(f"Initialized experiment: K={self.k_categories}, feature_mode='{self.feature_mode}'")

    def setup_logging(self):
        log_file = os.path.join(self.output_dir, f"{self.experiment_name}_experiment.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def diagnose_class_imbalance(self):
        """Diagnose class distribution for both targets."""
        self.logger.info("\n" + "="*70)
        self.logger.info("CLASS DISTRIBUTION DIAGNOSTICS")
        self.logger.info("="*70)
        
        for target in ['school_participation_flag', 'project_participation_flag']:
            dist = self.get_class_distribution(target)
            total = dist.get(0, 0) + dist.get(1, 0)
            pct_pos = (dist.get(1, 0) / total * 100) if total > 0 else 0
            
            self.logger.info(f"\n{target}:")
            self.logger.info(f"  Class 0 (negative): {dist.get(0, 0)} samples ({100-pct_pos:.2f}%)")
            self.logger.info(f"  Class 1 (positive): {dist.get(1, 0)} samples ({pct_pos:.2f}%)")
            
            if pct_pos < 1.0:
                self.logger.warning(f"  ⚠️  EXTREME IMBALANCE: Positive class < 1% ({pct_pos:.2f}%)")
            elif pct_pos < 5.0:
                self.logger.warning(f"  ⚠️  HIGH IMBALANCE: Positive class < 5% ({pct_pos:.2f}%)")
        
        self.logger.info("="*70 + "\n")

    def get_class_distribution(self, target: str) -> Dict[int, int]:
        """Get class distribution for a target variable."""
        y = self.data[target].astype(int).values
        unique, counts = np.unique(y, return_counts=True)
        # CRITICAL FIX: Convert numpy int64 keys to Python native int for JSON serialization
        return {int(u): int(c) for u, c in zip(unique, counts)}

    def sample_categories(self) -> List[str]:
        """Sample categories using evolutionary strategy if enabled."""
        # Early rounds: pure random until we have elites
        if not self.evolutionary_search or len(self.elite_pool) == 0:
            return random.sample(self.all_categories, self.k_categories)
        
        # Select best elite (could extend to tournament selection)
        elite_categories, _ = max(self.elite_pool, key=lambda x: x[1])
        
        # Generate offspring: keep 50%, randomize 50%
        n_keep = int(self.k_categories * (1 - self.mutation_rate))
        n_mutate = self.k_categories - n_keep
        
        offspring = []
        attempts = 0
        while len(offspring) < self.offspring_per_elite and attempts < 100:
            attempts += 1
            # Keep top-performing portion (could also shuffle which indices to keep)
            kept = random.sample(elite_categories, n_keep)
            
            # Sample new categories not already in kept
            available = [c for c in self.all_categories if c not in kept]
            if len(available) < n_mutate:
                continue  # Skip if not enough diversity
            
            mutated = random.sample(available, n_mutate)
            candidate = kept + mutated
            
            if len(candidate) == self.k_categories and candidate not in offspring:
                offspring.append(candidate)
        
        # Cycle through offspring, then fall back to random if exhausted
        if not hasattr(self, '_offspring_idx'):
            self._offspring_idx = 0
        
        if self._offspring_idx < len(offspring):
            candidate = offspring[self._offspring_idx]
            self._offspring_idx += 1
            return candidate
        else:
            self._offspring_idx = 0  # Reset cycle
            return random.sample(self.all_categories, self.k_categories)  # Exploration fallback

    def perform_kmeans(self, categories: List[str]) -> Tuple[float, np.ndarray]:
        X = self.data[categories].values
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        score = silhouette_score(X, cluster_labels)
        return score, cluster_labels

    def prepare_features(self, categories: List[str], target_flag: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features using selected strategy."""
        y = self.data[target_flag].astype(int).values
        
        if self.feature_mode == 'mean':
            X = self.data[categories].mean(axis=1).values.reshape(-1, 1)
        elif self.feature_mode == 'concat':
            X = self.data[categories].values
        else:
            raise ValueError(f"Unknown feature_mode: {self.feature_mode}")
        
        # Filter samples to ensure both classes exist
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            return None, None
        
        # Require minimum positive samples
        n_pos = np.sum(y == 1)
        if n_pos < 5:
            return None, None
        
        return X, y

    def train_classification_model(self, categories: List[str], target_flag: str) -> Dict[str, float]:
        """Train classification model with class weighting for imbalanced targets."""
        X, y = self.prepare_features(categories, target_flag)
        
        if X is None or y is None:
            return {'precision': 0.0, 'recall': 0.0, 'roc_auc': 0.5, 'f1': 0.0, 'valid': False}
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
        
        # CRITICAL: Class weighting for imbalanced targets
        model = LogisticRegression(
            random_state=42, 
            max_iter=1000,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except ValueError:
            roc_auc = 0.5
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'roc_auc': float(roc_auc),
            'f1': float(f1),
            'valid': True,
            'n_test_samples': len(y_test),
            'n_test_positives': int(np.sum(y_test)),
            'n_predicted_positives': int(np.sum(y_pred))
        }

    def run_experiment_round(self, round_num: int) -> Tuple[Dict[str, Any], bool]:
        self.logger.info(f"Round {round_num + 1}/{self.n_rounds}")
        
        sampled_categories = self.sample_categories()


        silhouette_score_val, _ = self.perform_kmeans(sampled_categories)  

        improvement_detected = False
        if silhouette_score_val > self.best_silhouette:
            self.best_silhouette = silhouette_score_val
            improvement_detected = True
            
            if self.evolutionary_search:
                # Replace elite pool with this new best solution
                self.elite_pool = [(sampled_categories.copy(), silhouette_score_val)]
        
        # if self.evolutionary_search:
        #     # Add to elite pool (keep only top-N)
        #     self.elite_pool.append((sampled_categories.copy(), silhouette_score_val))
        #     self.elite_pool = sorted(self.elite_pool, key=lambda x: x[1], reverse=True)[:self.elite_size]
    

        school_metrics = self.train_classification_model(sampled_categories, 'school_participation_flag')
        project_metrics = self.train_classification_model(sampled_categories, 'project_participation_flag')
        
        round_result = {
            'round': round_num,
            'categories': sampled_categories,
            'silhouette_score': float(silhouette_score_val),
            'school_roc_auc': float(school_metrics['roc_auc']),
            'school_f1': float(school_metrics['f1']),
            'project_roc_auc': float(project_metrics['roc_auc']),
            'project_f1': float(project_metrics['f1']),
            'avg_roc_auc': float((school_metrics['roc_auc'] + project_metrics['roc_auc']) / 2),
            'avg_f1': float((school_metrics['f1'] + project_metrics['f1']) / 2),
            'school_valid': school_metrics.get('valid', False),
            'project_valid': project_metrics.get('valid', False),
            'school_participation_metrics': school_metrics,
            'project_participation_metrics': project_metrics
        }
        
        # Check for improvements
        improvement_detected = False
        silhouette_improved = False  # NEW FLAG to track silhouette-specific improvement
        
        metrics_to_check = [
            ('silhouette', silhouette_score_val),
            ('school_roc_auc', school_metrics['roc_auc']),
            ('project_roc_auc', project_metrics['roc_auc']),
            ('avg_roc_auc', round_result['avg_roc_auc']),
            ('school_f1', school_metrics['f1']),
            ('project_f1', project_metrics['f1']),
            ('avg_f1', round_result['avg_f1'])
        ]

        for metric_name, current_score in metrics_to_check:
            if current_score > self.prev_best[metric_name] + 1e-6:
                improvement_detected = True
                if metric_name == 'silhouette':
                    silhouette_improved = True  # Mark silhouette improvement specifically
                
                self.best_metrics[metric_name] = {
                    'score': current_score,
                    'categories': sampled_categories.copy(),
                    'round': round_num
                }
                self.prev_best[metric_name] = current_score
                        # === NEW: Enhanced diagnostics when Silhouette improves ===

        # Main log line (unchanged)
        self.logger.info(
            f"  Silhouette: {silhouette_score_val:.4f} | "
            f"School ROC: {school_metrics['roc_auc']:.4f} F1: {school_metrics['f1']:.4f} | "
            f"Project ROC: {project_metrics['roc_auc']:.4f} F1: {project_metrics['f1']:.4f} "
            f"{'✓' if improvement_detected else ''}"
        )
        
        if silhouette_improved:
            avg_roc = round_result['avg_roc_auc']
            avg_f1 = round_result['avg_f1']
            ratio = avg_f1 / avg_roc if avg_roc > 1e-6 else 0.0
            
            # Compute ratio interpretation
            ratio_status = "balanced" if 0.9 <= ratio <= 1.1 else \
                        "F1 < ROC (conservative predictions)" if ratio < 0.9 else \
                        "F1 > ROC (aggressive predictions)"
            
            self.logger.info(
                f"    📊 Silhouette improvement → Avg ROC: {avg_roc:.4f} | "
                f"Avg F1: {avg_f1:.4f} | F1/ROC ratio: {ratio:.4f} ({ratio_status})"
            )
        # =========================================================

        return round_result, improvement_detected
                        
  

    def run(self):
        self.logger.info(f"Starting experiment: {self.experiment_name}")
        self.logger.info("Research focus: Correlation between unsupervised clustering (Silhouette) and supervised metrics")
        
        for round_num in range(self.n_rounds):
            round_result, improvement = self.run_experiment_round(round_num)
            self.results['rounds'].append(round_result)
            
            if improvement:
                self.results['best_performers'] = self.best_metrics.copy()
                self.save_results(incremental=True)
        
        # Final save
        self.results['best_performers'] = self.best_metrics.copy()

        # Final save with enhanced structure
   
        # ADD: Best categories per metric with explicit naming
        self.results['best_categories'] = {
            'silhouette': {
                'score': self.best_metrics['silhouette']['score'],
                'round': self.best_metrics['silhouette']['round'],
                'categories': self.best_metrics['silhouette']['categories']
            },
            'school_roc_auc': {
                'score': self.best_metrics['school_roc_auc']['score'],
                'round': self.best_metrics['school_roc_auc']['round'],
                'categories': self.best_metrics['school_roc_auc']['categories']
            },
            'school_f1': {
                'score': self.best_metrics['school_f1']['score'],
                'round': self.best_metrics['school_f1']['round'],
                'categories': self.best_metrics['school_f1']['categories']
            },
            'project_roc_auc': {
                'score': self.best_metrics['project_roc_auc']['score'],
                'round': self.best_metrics['project_roc_auc']['round'],
                'categories': self.best_metrics['project_roc_auc']['categories']
            },
            'project_f1': {
                'score': self.best_metrics['project_f1']['score'],
                'round': self.best_metrics['project_f1']['round'],
                'categories': self.best_metrics['project_f1']['categories']
            },
            'avg_roc_auc': {
                'score': self.best_metrics['avg_roc_auc']['score'],
                'round': self.best_metrics['avg_roc_auc']['round'],
                'categories': self.best_metrics['avg_roc_auc']['categories']
            },
            'avg_f1': {
                'score': self.best_metrics['avg_f1']['score'],
                'round': self.best_metrics['avg_f1']['round'],
                'categories': self.best_metrics['avg_f1']['categories']
            }
        }
        
        # ADD: Summary statistics
        self.results['summary_statistics'] = self.compute_summary_statistics()
        
        self.save_results(final=True)
        
        # Generate specialized plots
        self.generate_convergence_plots()
        self.generate_silhouette_correlation_analysis()  # FOCUSED CORRELATION ANALYSIS
  
        
        self.logger.info("Experiment completed!")
        self.print_final_summary()

    def save_results(self, incremental=False, final=False):
        results_file = os.path.join(self.output_dir, f"{self.experiment_name}_results.json")
    
        def convert_to_serializable(obj):
            """Recursively convert numpy types INCLUDING dictionary keys."""
            if isinstance(obj, dict):
                new_dict = {}
                for k, v in obj.items():
                    if isinstance(k, (np.integer, np.floating, np.bool_)):
                        k = k.item()
                    new_dict[k] = convert_to_serializable(v)
                return new_dict
            elif isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        results_to_save = self.results.copy()
        
        if final:
            # 1. Best categories per metric (only the 4 core metrics + silhouette)
            results_to_save['best_categories'] = {
                'silhouette': {
                    'score': self.best_metrics['silhouette']['score'],
                    'round': self.best_metrics['silhouette']['round'],
                    'categories': self.best_metrics['silhouette']['categories']
                },
                'school_roc_auc': {
                    'score': self.best_metrics['school_roc_auc']['score'],
                    'round': self.best_metrics['school_roc_auc']['round'],
                    'categories': self.best_metrics['school_roc_auc']['categories']
                },
                'school_f1': {
                    'score': self.best_metrics['school_f1']['score'],
                    'round': self.best_metrics['school_f1']['round'],
                    'categories': self.best_metrics['school_f1']['categories']
                },
                'project_roc_auc': {
                    'score': self.best_metrics['project_roc_auc']['score'],
                    'round': self.best_metrics['project_roc_auc']['round'],
                    'categories': self.best_metrics['project_roc_auc']['categories']
                },
                'project_f1': {
                    'score': self.best_metrics['project_f1']['score'],
                    'round': self.best_metrics['project_f1']['round'],
                    'categories': self.best_metrics['project_f1']['categories']
                }
            }
            
            # 2. GLOBAL AVERAGES ACROSS ALL TRIALS (per metric, NOT school+project averaged)
            if self.results['rounds']:
                school_roc_vals = [r['school_roc_auc'] for r in self.results['rounds']]
                school_f1_vals = [r['school_f1'] for r in self.results['rounds']]
                project_roc_vals = [r['project_roc_auc'] for r in self.results['rounds']]
                project_f1_vals = [r['project_f1'] for r in self.results['rounds']]
                
                results_to_save['global_averages_across_trials'] = {
                    'school_roc_auc': {
                        'mean': float(np.mean(school_roc_vals)),
                        'std': float(np.std(school_roc_vals))
                    },
                    'school_f1': {
                        'mean': float(np.mean(school_f1_vals)),
                        'std': float(np.std(school_f1_vals))
                    },
                    'project_roc_auc': {
                        'mean': float(np.mean(project_roc_vals)),
                        'std': float(np.std(project_roc_vals))
                    },
                    'project_f1': {
                        'mean': float(np.mean(project_f1_vals)),
                        'std': float(np.std(project_f1_vals))
                    }
                }
            
            # 3. Silhouette-associated performance (RAW metrics only - no school+project averaging)
            best_sil_round = self.best_metrics['silhouette']['round']
            if best_sil_round >= 0 and best_sil_round < len(self.results['rounds']):
                sil_round_data = self.results['rounds'][best_sil_round]
                
                # RAW metrics at best silhouette round (no averaging school+project)
                metrics_at_sil = {
                    'school_roc_auc': sil_round_data['school_roc_auc'],
                    'school_f1': sil_round_data['school_f1'],
                    'project_roc_auc': sil_round_data['project_roc_auc'],
                    'project_f1': sil_round_data['project_f1']
                }
                
                # Best possible metrics (individual maxima)
                best_possible = {
                    'school_roc_auc': self.best_metrics['school_roc_auc']['score'],
                    'school_f1': self.best_metrics['school_f1']['score'],
                    'project_roc_auc': self.best_metrics['project_roc_auc']['score'],
                    'project_f1': self.best_metrics['project_f1']['score']
                }
                
                # Global baseline (mean across all trials)
                baseline = results_to_save['global_averages_across_trials']
                
                # Compute gaps to best possible
                gaps_to_best = {
                    'school_roc_auc': best_possible['school_roc_auc'] - metrics_at_sil['school_roc_auc'],
                    'school_f1': best_possible['school_f1'] - metrics_at_sil['school_f1'],
                    'project_roc_auc': best_possible['project_roc_auc'] - metrics_at_sil['project_roc_auc'],
                    'project_f1': best_possible['project_f1'] - metrics_at_sil['project_f1']
                }
                
                # Compute deltas vs baseline (positive = better than random sampling)
                deltas_vs_baseline = {
                    'school_roc_auc': metrics_at_sil['school_roc_auc'] - baseline['school_roc_auc']['mean'],
                    'school_f1': metrics_at_sil['school_f1'] - baseline['school_f1']['mean'],
                    'project_roc_auc': metrics_at_sil['project_roc_auc'] - baseline['project_roc_auc']['mean'],
                    'project_f1': metrics_at_sil['project_f1'] - baseline['project_f1']['mean']
                }
                
                # Gap assessment helper
                def gap_label(gap):
                    if abs(gap) < 0.03:
                        return "NEGLIGIBLE"
                    elif abs(gap) < 0.08:
                        return "MODERATE"
                    elif abs(gap) < 0.15:
                        return "SUBSTANTIAL"
                    else:
                        return "LARGE"
                
                # Overall assessment
                avg_gap_to_best = np.mean([abs(v) for v in gaps_to_best.values()])
                if avg_gap_to_best < 0.05:
                    corr_strength = "STRONG"
                    verdict = "✓✓ Best clustering round achieves near-optimal classification performance"
                elif avg_gap_to_best < 0.10:
                    corr_strength = "MODERATE"
                    verdict = "✓ Best clustering round achieves reasonably good classification performance"
                else:
                    corr_strength = "WEAK"
                    verdict = "⚠️ Best clustering round does NOT reliably yield best classification performance"
                
                results_to_save['silhouette_associated_performance'] = {
                    'best_silhouette_round': best_sil_round + 1,  # 1-indexed for readability
                    'silhouette_score': self.best_metrics['silhouette']['score'],
                    'metrics_at_silhouette_round': metrics_at_sil,
                    'global_baseline_means': {
                        'school_roc_auc': baseline['school_roc_auc']['mean'],
                        'school_f1': baseline['school_f1']['mean'],
                        'project_roc_auc': baseline['project_roc_auc']['mean'],
                        'project_f1': baseline['project_f1']['mean']
                    },
                    'best_possible_metrics': best_possible,
                    'gaps_to_best_possible': gaps_to_best,
                    'improvement_vs_baseline': deltas_vs_baseline,  # Positive = better than random
                    'gap_assessment': {
                        'average_gap_to_best': float(avg_gap_to_best),
                        'correlation_strength': corr_strength,
                        'verdict': verdict
                    }
                }
            
            # 4. Simplified summary statistics (mean/std/min/max per metric)
            if self.results['rounds']:
                metrics = {
                    'silhouette': [r['silhouette_score'] for r in self.results['rounds']],
                    'school_roc_auc': school_roc_vals,
                    'school_f1': school_f1_vals,
                    'project_roc_auc': project_roc_vals,
                    'project_f1': project_f1_vals
                }
                results_to_save['summary_statistics'] = {
                    metric: {
                        'mean': float(np.mean(vals)),
                        'std': float(np.std(vals)),
                        'min': float(np.min(vals)),
                        'max': float(np.max(vals))
                    }
                    for metric, vals in metrics.items()
                }
        
        # Serialize with robust converter
        serializable_results = convert_to_serializable(results_to_save)
        
        if incremental:
            serializable_results['last_save_reason'] = 'incremental_improvement'
        elif final:
            serializable_results['last_save_reason'] = 'final_save'
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        if incremental or final:
            self.logger.info(f"Results saved to {results_file}")


    ####

    def generate_convergence_plots(self):
        """Generate convergence plots with professional labeling."""
        if not self.results['rounds']:
            return
        
        rounds = np.arange(len(self.results['rounds']))
        
        # Extract metrics
        metrics = {
            'silhouette': np.array([r['silhouette_score'] for r in self.results['rounds']]),
            'school_roc': np.array([r['school_roc_auc'] for r in self.results['rounds']]),
            'project_roc': np.array([r['project_roc_auc'] for r in self.results['rounds']]),
            'avg_roc': np.array([r['avg_roc_auc'] for r in self.results['rounds']]),
            'school_f1': np.array([r['school_f1'] for r in self.results['rounds']]),
            'project_f1': np.array([r['project_f1'] for r in self.results['rounds']]),
            'avg_f1': np.array([r['avg_f1'] for r in self.results['rounds']])
        }
        
        # Compute best-so-far
        best_so_far = {name: np.maximum.accumulate(values) for name, values in metrics.items()}
        
        # Create figure with professional layout
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.28)
        
        # Plot 1: Silhouette Score (main unsupervised metric)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(rounds, best_so_far['silhouette'], color=METRIC_COLORS['silhouette'], 
                linewidth=2.8, label='Best-so-far', zorder=3)
        ax1.plot(rounds, metrics['silhouette'], '.', color=METRIC_COLORS['silhouette'], 
                alpha=0.35, markersize=5, label='Current value', zorder=2)
        ax1.axhline(y=best_so_far['silhouette'][-1], color='red', linestyle='--', 
                   linewidth=2.2, alpha=0.8, label=f'Final best: {best_so_far["silhouette"][-1]:.4f}', zorder=1)
        ax1.set_title('Clustering Quality: Silhouette Score Convergence', 
                     fontsize=15, fontweight='bold', pad=12)
        ax1.set_xlabel('Sampling Round', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
        ax1.legend(loc='lower right', framealpha=0.95)
        ax1.grid(True, alpha=0.35, linestyle='--', linewidth=0.7)
        ax1.set_ylim([-0.15, 1.02])
        
        # Plot 2: ROC AUC variants
        ax2 = fig.add_subplot(gs[1, 0])
        for metric in ['school_roc', 'project_roc', 'avg_roc']:
            ax2.plot(rounds, best_so_far[metric], color=METRIC_COLORS[metric], 
                    linewidth=2.3, label=METRIC_LABELS[metric])
        ax2.set_title('Classification Performance: ROC AUC Convergence', 
                     fontsize=14, fontweight='bold', pad=10)
        ax2.set_xlabel('Sampling Round', fontsize=11)
        ax2.set_ylabel('ROC AUC', fontsize=11, fontweight='bold')
        ax2.legend(loc='lower right', fontsize=9.5, framealpha=0.95)
        ax2.grid(True, alpha=0.35, linestyle='--', linewidth=0.7)
        ax2.set_ylim([0.48, 1.01])
        
        # Plot 3: F1 Score variants
        ax3 = fig.add_subplot(gs[1, 1])
        for metric in ['school_f1', 'project_f1', 'avg_f1']:
            ax3.plot(rounds, best_so_far[metric], color=METRIC_COLORS[metric], 
                    linewidth=2.3, label=METRIC_LABELS[metric])
        ax3.set_title('Classification Performance: F1 Score Convergence', 
                     fontsize=14, fontweight='bold', pad=10)
        ax3.set_xlabel('Sampling Round', fontsize=11)
        ax3.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
        ax3.legend(loc='lower right', fontsize=9.5, framealpha=0.95)
        ax3.grid(True, alpha=0.35, linestyle='--', linewidth=0.7)
        ax3.set_ylim([-0.05, 1.02])
        
        # Plot 4: School vs Project ROC AUC
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.plot(rounds, best_so_far['school_roc'], color=METRIC_COLORS['school_roc'], 
                linewidth=2.6, label='School (best)', zorder=3)
        ax4.plot(rounds, best_so_far['project_roc'], color=METRIC_COLORS['project_roc'], 
                linewidth=2.6, label='Project (best)', zorder=2)
        ax4.fill_between(rounds, best_so_far['school_roc'], best_so_far['project_roc'], 
                         alpha=0.22, color='gray', label='Performance gap', zorder=1)
        ax4.set_title('Target Comparison: School vs Project ROC AUC', 
                     fontsize=14, fontweight='bold', pad=10)
        ax4.set_xlabel('Sampling Round', fontsize=11)
        ax4.set_ylabel('ROC AUC', fontsize=11, fontweight='bold')
        ax4.legend(loc='lower right', framealpha=0.95)
        ax4.grid(True, alpha=0.35, linestyle='--', linewidth=0.7)
        ax4.set_ylim([0.48, 1.01])
        
        # Plot 5: School vs Project F1 Score
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.plot(rounds, best_so_far['school_f1'], color=METRIC_COLORS['school_f1'], 
                linewidth=2.6, label='School (best)', zorder=3)
        ax5.plot(rounds, best_so_far['project_f1'], color=METRIC_COLORS['project_f1'], 
                linewidth=2.6, label='Project (best)', zorder=2)
        ax5.fill_between(rounds, best_so_far['school_f1'], best_so_far['project_f1'], 
                         alpha=0.22, color='gray', label='Performance gap', zorder=1)
        
        # Annotate if project F1 is problematic
        final_proj_f1 = best_so_far['project_f1'][-1]
        if final_proj_f1 < 0.05:
            ax5.text(0.5, 0.88, f'⚠️ Project F1 = {final_proj_f1:.3f}\n(extreme imbalance)', 
                    ha='center', va='top', transform=ax5.transAxes,
                    fontsize=10, color='darkred', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.92))
        
        ax5.set_title('Target Comparison: School vs Project F1 Score', 
                     fontsize=14, fontweight='bold', pad=10)
        ax5.set_xlabel('Sampling Round', fontsize=11)
        ax5.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
        ax5.legend(loc='lower right', framealpha=0.95)
        ax5.grid(True, alpha=0.35, linestyle='--', linewidth=0.7)
        ax5.set_ylim([-0.05, 1.02])
        
        fig.suptitle(f'Convergence Analysis: {self.experiment_name}\n', 
                    fontsize=17, fontweight='bold', y=0.998)
        plt.tight_layout()
        
        plot_file = os.path.join(self.output_dir, f"{self.experiment_name}_convergence_plots.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Convergence plots saved to {plot_file}")

    def generate_silhouette_correlation_analysis(self):
        """
        FOCUSED CORRELATION ANALYSIS: Silhouette Score vs Supervised Metrics
        Directly addresses research question: "Do unsupervised clustering metrics correlate with supervised performance?"
        """
        if not self.results['rounds']:
            return
        
        # Create comprehensive metrics DataFrame
        metrics_df = pd.DataFrame({
            'Silhouette Score': [r['silhouette_score'] for r in self.results['rounds']],
            'School ROC AUC': [r['school_roc_auc'] for r in self.results['rounds']],
            'Project ROC AUC': [r['project_roc_auc'] for r in self.results['rounds']],
            'Avg. ROC AUC': [r['avg_roc_auc'] for r in self.results['rounds']],
            'School F1 Score': [r['school_f1'] for r in self.results['rounds']],
            'Project F1 Score': [r['project_f1'] for r in self.results['rounds']],
            'Avg. F1 Score': [r['avg_f1'] for r in self.results['rounds']]
        })
        
        # Compute correlations with silhouette
        silhouette_corr = metrics_df.corr()['Silhouette Score'].drop('Silhouette Score')
        
        # Create specialized visualization: 2x3 grid focusing on silhouette relationships
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.28)
        
        # Define supervised metrics to correlate with silhouette
        supervised_metrics = [
            ('School ROC AUC', 'seagreen', 0, 0),
            ('Project ROC AUC', 'coral', 0, 1),
            ('Avg. ROC AUC', 'darkorchid', 0, 2),
            ('School F1 Score', 'seagreen', 1, 0),
            ('Project F1 Score', 'coral', 1, 1),
            ('Avg. F1 Score', 'darkorchid', 1, 2)
        ]
        
        # Create scatter plots for each relationship
        for metric_name, color, row, col in supervised_metrics:
            ax = fig.add_subplot(gs[row, col])
            
            # Scatter plot with transparency
            ax.scatter(metrics_df['Silhouette Score'], metrics_df[metric_name], 
                      alpha=0.45, s=35, color=color, edgecolors='white', linewidth=0.3)
            
            # Regression line
            z = np.polyfit(metrics_df['Silhouette Score'], metrics_df[metric_name], 1)
            p = np.poly1d(z)
            x_range = np.linspace(metrics_df['Silhouette Score'].min(), 
                                metrics_df['Silhouette Score'].max(), 100)
            ax.plot(x_range, p(x_range), "r--", linewidth=2.5, alpha=0.85, 
                   label=f'Linear fit (R={silhouette_corr[metric_name]:.3f})')
            
            # Correlation coefficient as large annotation
            corr_val = silhouette_corr[metric_name]
            corr_color = 'darkgreen' if corr_val > 0.3 else 'darkred' if corr_val < -0.3 else 'gray'
            ax.text(0.05, 0.95, f'R = {corr_val:.3f}', 
                   transform=ax.transAxes, fontsize=16, fontweight='bold',
                   color=corr_color, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
            
            ax.set_xlabel('Silhouette Score', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
            ax.set_title(f'Silhouette vs {metric_name}', fontsize=13, fontweight='bold', pad=8)
            ax.grid(True, alpha=0.35, linestyle='--', linewidth=0.7)
            ax.legend(loc='lower right', framealpha=0.95)
            
            # Set consistent y-axis limits for ROC and F1 groups
            if 'ROC' in metric_name:
                ax.set_ylim([0.45, 1.02])
            else:
                ax.set_ylim([-0.05, 1.02])
        
        # Add research question as figure title
        fig.suptitle('Unsupervised-Supervised Correlation Analysis\n'
                    'Does Clustering Quality (Silhouette) Predict Classification Performance?', 
                    fontsize=18, fontweight='bold', y=0.995)
        
        # Add interpretation footnote
        # footnote = (
        #     "Interpretation: |R| > 0.3 suggests meaningful relationship. "
        #     "Positive R: Better clustering → Better classification. "
        #     "Negative R: Better clustering → Worse classification (rare)."
        # )
        # fig.text(0.5, 0.01, footnote, ha='center', fontsize=11, style='italic', 
        #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.25))
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Make room for footnote
        
        corr_plot_file = os.path.join(self.output_dir, f"{self.experiment_name}_silhouette_correlation.png")
        plt.savefig(corr_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save correlation values to JSON
        corr_json = {
            'silhouette_correlations': silhouette_corr.round(4).to_dict()}
        
        corr_json_file = os.path.join(self.output_dir, f"{self.experiment_name}_silhouette_correlations.json")
        with open(corr_json_file, 'w') as f:
            json.dump(corr_json, f, indent=2)
        
        self.logger.info(f"Focused correlation analysis saved to {corr_plot_file}")
        self.logger.info(f"Correlation values saved to {corr_json_file}")
        
        # Log key findings
        self.logger.info("Silhouette Correlation Summary:")
        for metric, corr in silhouette_corr.items():
            strength = "STRONG" if abs(corr) > 0.5 else "MODERATE" if abs(corr) > 0.3 else "WEAK"
            direction = "positive" if corr > 0 else "negative"
            self.logger.info(f"  • Silhouette vs {metric}: R={corr:.3f} ({strength} {direction})")


    def print_final_summary(self):
        print("\n" + "="*115)
        print(f"EXPERIMENT SUMMARY: {self.experiment_name}")
        print(f"Research Question: Does unsupervised clustering quality correlate with supervised performance?")
        print(f"Configuration: K={self.k_categories} categories | Rounds={self.n_rounds} | Feature mode={self.feature_mode}")
        print("="*115)
        
        # Class distribution
        print("\n📊 CLASS DISTRIBUTION:")
        for target in ['school_participation_flag', 'project_participation_flag']:
            dist = self.results['class_distribution'][target]
            total = sum(dist.values())
            pct_pos = dist.get(1, 0) / total * 100 if total > 0 else 0
            status = "⚠️ EXTREME IMBALANCE" if pct_pos < 1.0 else "⚠️ HIGH IMBALANCE" if pct_pos < 5.0 else "✓ Acceptable"
            print(f"  • {target.replace('_participation_flag', '').title()}: "
                f"Positive class = {pct_pos:.2f}% ({status})")
        
        # Best performers (only core 4 metrics + silhouette)
        print("\n🏆 BEST PERFORMERS (Individual Metric Maxima):")
        print(f"\n  Unsupervised Metric:")
        sil = self.best_metrics['silhouette']
        print(f"    • Silhouette Score: {sil['score']:.4f} (Round {sil['round']+1})")
        
        print(f"\n  School Participation (Supervised):")
        print(f"    • ROC AUC: {self.best_metrics['school_roc_auc']['score']:.4f} "
            f"(Round {self.best_metrics['school_roc_auc']['round']+1})")
        print(f"    • F1 Score: {self.best_metrics['school_f1']['score']:.4f} "
            f"(Round {self.best_metrics['school_f1']['round']+1})")
        
        print(f"\n  Project Participation (Supervised):")
        print(f"    • ROC AUC: {self.best_metrics['project_roc_auc']['score']:.4f} "
            f"(Round {self.best_metrics['project_roc_auc']['round']+1})")
        print(f"    • F1 Score: {self.best_metrics['project_f1']['score']:.4f} "
            f"(Round {self.best_metrics['project_f1']['round']+1})")
        
        # Compute global baselines (mean across ALL trials)
        school_roc_vals = [r['school_roc_auc'] for r in self.results['rounds']]
        school_f1_vals = [r['school_f1'] for r in self.results['rounds']]
        project_roc_vals = [r['project_roc_auc'] for r in self.results['rounds']]
        project_f1_vals = [r['project_f1'] for r in self.results['rounds']]
        
        global_baseline = {
            'school_roc_auc': np.mean(school_roc_vals),
            'school_f1': np.mean(school_f1_vals),
            'project_roc_auc': np.mean(project_roc_vals),
            'project_f1': np.mean(project_f1_vals)
        }
        
        # Silhouette-associated performance (CORE METRICS ONLY - no school/project averaging)
        print("\n" + "─"*115)
        print("🎯 KEY INSIGHT: Performance AT Best Silhouette Round vs. Baselines")
        print("─"*115)
        
        best_sil_round = sil['round']
        if best_sil_round >= 0 and best_sil_round < len(self.results['rounds']):
            sil_round_data = self.results['rounds'][best_sil_round]
            
            # Metrics at best silhouette round
            sil_school_roc = sil_round_data['school_roc_auc']
            sil_school_f1 = sil_round_data['school_f1']
            sil_project_roc = sil_round_data['project_roc_auc']
            sil_project_f1 = sil_round_data['project_f1']
            
            # Best possible metrics
            best_school_roc = self.best_metrics['school_roc_auc']['score']
            best_school_f1 = self.best_metrics['school_f1']['score']
            best_project_roc = self.best_metrics['project_roc_auc']['score']
            best_project_f1 = self.best_metrics['project_f1']['score']
            
            # Gaps and improvements
            gaps = {
                'school_roc': best_school_roc - sil_school_roc,
                'school_f1': best_school_f1 - sil_school_f1,
                'project_roc': best_project_roc - sil_project_roc,
                'project_f1': best_project_f1 - sil_project_f1
            }
            improvements = {
                'school_roc': sil_school_roc - global_baseline['school_roc_auc'],
                'school_f1': sil_school_f1 - global_baseline['school_f1'],
                'project_roc': sil_project_roc - global_baseline['project_roc_auc'],
                'project_f1': sil_project_f1 - global_baseline['project_f1']
            }
            
            # Gap interpretation helper
            def gap_label(gap):
                if abs(gap) < 0.03:
                    return "✓ NEGLIGIBLE"
                elif abs(gap) < 0.08:
                    return "⚠️ MODERATE"
                elif abs(gap) < 0.15:
                    return "✗ SUBSTANTIAL"
                else:
                    return "✗✗ LARGE"
            
            def improvement_arrow(imp):
                if imp > 0.03:
                    return "↑↑"
                elif imp > 0.01:
                    return "↑"
                elif imp < -0.01:
                    return "↓"
                else:
                    return "→"
            
            print(f"\n  Best Silhouette Round: {best_sil_round + 1} (Score: {sil['score']:.4f})")
            print(f"\n  ┌──────────────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐")
            print(f"  │ Metric               │ At Silhouette│ Global Mean  │ Best Possible│ Gap to Best  │ vs. Baseline │")
            print(f"  ├──────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤")
            print(f"  │ School ROC AUC       │ {sil_school_roc:12.4f} │ {global_baseline['school_roc_auc']:12.4f} │ {best_school_roc:12.4f} │ {gaps['school_roc']:6.4f} {gap_label(gaps['school_roc']):12s} │ {improvement_arrow(improvements['school_roc'])} {improvements['school_roc']:+6.4f} │")
            print(f"  │ School F1 Score      │ {sil_school_f1:12.4f} │ {global_baseline['school_f1']:12.4f} │ {best_school_f1:12.4f} │ {gaps['school_f1']:6.4f} {gap_label(gaps['school_f1']):12s} │ {improvement_arrow(improvements['school_f1'])} {improvements['school_f1']:+6.4f} │")
            print(f"  │ Project ROC AUC      │ {sil_project_roc:12.4f} │ {global_baseline['project_roc_auc']:12.4f} │ {best_project_roc:12.4f} │ {gaps['project_roc']:6.4f} {gap_label(gaps['project_roc']):12s} │ {improvement_arrow(improvements['project_roc'])} {improvements['project_roc']:+6.4f} │")
            print(f"  │ Project F1 Score     │ {sil_project_f1:12.4f} │ {global_baseline['project_f1']:12.4f} │ {best_project_f1:12.4f} │ {gaps['project_f1']:6.4f} {gap_label(gaps['project_f1']):12s} │ {improvement_arrow(improvements['project_f1'])} {improvements['project_f1']:+6.4f} │")
            print(f"  └──────────────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘")
            
            # Overall assessment
            avg_gap = np.mean([abs(v) for v in gaps.values()])
            avg_improvement = np.mean([v for v in improvements.values()])
            
            if avg_gap < 0.05 and avg_improvement > 0.01:
                correlation = "STRONG"
                verdict = "✓✓ Clustering optimization yields near-optimal classification with consistent improvement over random sampling"
            elif avg_gap < 0.10 and avg_improvement > 0.005:
                correlation = "MODERATE"
                verdict = "✓ Clustering optimization yields reasonably good classification performance"
            elif avg_improvement > 0:
                correlation = "WEAK_POSITIVE"
                verdict = "→ Clustering optimization shows slight improvement over random sampling, but not near-optimal"
            else:
                correlation = "NONE/NEGATIVE"
                verdict = "✗ Clustering optimization does NOT reliably improve classification performance"
            
            print(f"\n  💡 CORRELATION ASSESSMENT: {correlation}")
            print(f"     {verdict}")
            print(f"     • Average absolute gap to best possible: {avg_gap:.4f}")
            print(f"     • Average improvement vs. random baseline: {avg_improvement:+.4f}")
        
        # STATISTICAL CORRELATION (4 core metrics only)
        print("\n" + "─"*115)
        print("🔬 STATISTICAL CORRELATION: Silhouette vs. Core Supervised Metrics")
        print("─"*115)
        
        metrics_df = pd.DataFrame({
            'silhouette': [r['silhouette_score'] for r in self.results['rounds']],
            'school_roc': school_roc_vals,
            'school_f1': school_f1_vals,
            'project_roc': project_roc_vals,
            'project_f1': project_f1_vals
        })
        
        corr = metrics_df.corr()
        silhouette_corrs = {
            'School ROC AUC': corr.loc['silhouette', 'school_roc'],
            'School F1 Score': corr.loc['silhouette', 'school_f1'],
            'Project ROC AUC': corr.loc['silhouette', 'project_roc'],
            'Project F1 Score': corr.loc['silhouette', 'project_f1']
        }
        
        print(f"\n  ┌──────────────────────┬──────────┬──────────────────────────────────┬──────────────────┐")
        print(f"  │ Metric               │ R        │ Interpretation                   │ Practical Impact │")
        print(f"  ├──────────────────────┼──────────┼──────────────────────────────────┼──────────────────┤")
        for metric, r in silhouette_corrs.items():
            # Interpretation
            if abs(r) > 0.5:
                interp = "STRONG relationship"
                marker = "↑↑" if r > 0 else "↓↓"
            elif abs(r) > 0.3:
                interp = "MODERATE relationship"
                marker = "↑" if r > 0 else "↓"
            else:
                interp = "WEAK relationship"
                marker = "→"
            
            # Practical impact assessment
            if r > 0.4:
                impact = "HIGH"
            elif r > 0.2:
                impact = "MODERATE"
            elif r > -0.2:
                impact = "LOW"
            else:
                impact = "NEGATIVE"
            
            print(f"  │ {metric:20s} │ {r:7.3f}  │ {marker} {interp:28s} │ {impact:16s} │")
        print(f"  └──────────────────────┴──────────┴──────────────────────────────────┴──────────────────┘")
        
        # Final conclusion based on 4 core metrics
        avg_abs_corr = np.mean([abs(r) for r in silhouette_corrs.values()])
        
        print("\n" + "─"*115)
        print("✅ FINAL CONCLUSION")
        print("─"*115)
        if avg_abs_corr > 0.4:
            print(f"  ✓✓ STRONG EVIDENCE of correlation between clustering quality and classification performance")
            print(f"     (Average |R| = {avg_abs_corr:.3f} across 4 core metrics)")
        elif avg_abs_corr > 0.25:
            print(f"  ✓ MODERATE EVIDENCE of correlation")
            print(f"     (Average |R| = {avg_abs_corr:.3f} across 4 core metrics)")
        else:
            print(f"  ✗ WEAK/NO EVIDENCE of practical correlation")
            print(f"     (Average |R| = {avg_abs_corr:.3f} across 4 core metrics)")
        
        # Critical context for imbalanced Project target
        proj_pos_samples = self.results['class_distribution']['project_participation_flag'].get(1, 0)
        if proj_pos_samples < 10:
            print(f"\n  ⚠️  CRITICAL CONTEXT: Project metrics may be unstable due to extreme class imbalance")
            print(f"     (Only {proj_pos_samples} positive samples for Project Participation)")
            print(f"     → Project F1 correlations should be interpreted with caution")
        
        print("\n" + "="*115)
        print("📁 Results saved to:")
        print(f"   • JSON: {self.experiment_name}_results.json")
        print(f"     - Contains global averages across trials (not school/project averages)")
        print(f"     - Contains silhouette-associated metrics for all 4 core metrics")
        print(f"     - Contains best categories for silhouette, school ROC/F1, project ROC/F1")
        print(f"   • Convergence plots: {self.experiment_name}_convergence_plots.png")
        print(f"   • CORRELATION ANALYSIS: {self.experiment_name}_silhouette_correlation.png")
        print("="*115 + "\n")

    def compute_summary_statistics(self):
        """Compute comprehensive summary statistics for final JSON output."""
        if not self.results['rounds']:
            return {}
        
        # Extract all metric values across rounds
        silhouette_scores = [r['silhouette_score'] for r in self.results['rounds']]
        school_roc = [r['school_roc_auc'] for r in self.results['rounds']]
        school_f1 = [r['school_f1'] for r in self.results['rounds']]
        project_roc = [r['project_roc_auc'] for r in self.results['rounds']]
        project_f1 = [r['project_f1'] for r in self.results['rounds']]
        avg_roc = [r['avg_roc_auc'] for r in self.results['rounds']]
        avg_f1 = [r['avg_f1'] for r in self.results['rounds']]
        
        def stats(arr):
            return {
                'mean': float(np.mean(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
            }
        
        return {
            'silhouette': stats(silhouette_scores),
            'school_roc_auc': stats(school_roc),
            'school_f1': stats(school_f1),
            'project_roc_auc': stats(project_roc),
            'project_f1': stats(project_f1),
            'avg_roc_auc': stats(avg_roc),
            'avg_f1': stats(avg_f1),
            'total_valid_rounds': len(self.results['rounds']),
            'school_valid_rounds': sum(1 for r in self.results['rounds'] if r.get('school_valid', False)),
            'project_valid_rounds': sum(1 for r in self.results['rounds'] if r.get('project_valid', False))
        }

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Category sampling experiment focused on unsupervised-supervised correlation analysis'
    )
    parser.add_argument('--input', type=str, required=True, 
                        help='Path to input CSV file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for logs and results')
    parser.add_argument('--experiment_name', type=str, required=True,
                        help='Name of the experiment')
    parser.add_argument('--n_rounds', type=int, default=100,
                        help='Number of sampling rounds (default: 100)')
    parser.add_argument('--k_categories', type=int, default=10,
                        help='Number of categories to sample per round (default: 10)')
    parser.add_argument('--feature_mode', type=str, default='mean',
                        choices=['mean', 'concat'],
                        help="Feature engineering mode: 'mean' averages categories (default), "
                             "'concat' uses all categories as separate features (better for imbalanced targets)")
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    parser.add_argument('--evolutionary_search', action='store_true',
                    help='Enable evolutionary search instead of pure random sampling')
    parser.add_argument('--elite_size', type=int, default=2,
                        help='Number of elite solutions to maintain (default: 1)')
    parser.add_argument('--offspring_per_elite', type=int, default=5,
                        help='Number of mutated offspring to generate per elite (default: 10)')
    parser.add_argument('--mutation_rate', type=float, default=0.5,
                        help='Fraction of categories to randomize (default: 0.5 = 50%)')

    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    experiment = CategorySamplingExperiment(
        input_csv=args.input,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        n_rounds=args.n_rounds,
        k_categories=args.k_categories,
        feature_mode=args.feature_mode,
        evolutionary_search=args.evolutionary_search,
        offspring_per_elite=args.offspring_per_elite,
        elite_size=args.elite_size,
        mutation_rate=args.mutation_rate
    )
    
    experiment.run()

if __name__ == "__main__":
    main()



    # python evolve_k_means.py --input /home/dev/work_main/random/assesment/all_criteria_scored/mistral_nemo/cv_scores_mistralai_mistral-nemo_merged.csv --output_dir /home/dev/work_main/random/assesment/k_means_selected/miatral/ --experiment_name mistral_cv_evo --n_rounds 1000 --evolutionary_search