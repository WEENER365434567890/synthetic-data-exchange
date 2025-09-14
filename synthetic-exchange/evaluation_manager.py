"""
Evaluation and Quality Scoring for Synthetic Data
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class StatisticalEvaluator:
    """Evaluates statistical similarity between real and synthetic data"""
    
    def __init__(self):
        self.results = {}
    
    def kolmogorov_smirnov_test(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform KS test for each numerical column"""
        ks_results = {}
        
        for column in real_data.select_dtypes(include=[np.number]).columns:
            if column in synthetic_data.columns:
                real_values = real_data[column].dropna()
                synthetic_values = synthetic_data[column].dropna()
                
                if len(real_values) > 0 and len(synthetic_values) > 0:
                    ks_statistic, p_value = stats.ks_2samp(real_values, synthetic_values)
                    ks_results[column] = {
                        'ks_statistic': ks_statistic,
                        'p_value': p_value,
                        'similar': p_value > 0.05  # Not significantly different
                    }
        
        return ks_results
    
    def correlation_comparison(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> Dict[str, Any]:
        """Compare correlation matrices between real and synthetic data"""
        # Get numerical columns
        numerical_cols = real_data.select_dtypes(include=[np.number]).columns
        common_cols = [col for col in numerical_cols if col in synthetic_data.columns]
        
        if len(common_cols) < 2:
            return {"error": "Need at least 2 numerical columns for correlation analysis"}
        
        real_corr = real_data[common_cols].corr()
        synthetic_corr = synthetic_data[common_cols].corr()
        
        # Calculate correlation of correlations
        real_corr_values = real_corr.values[np.triu_indices_from(real_corr.values, k=1)]
        synthetic_corr_values = synthetic_corr.values[np.triu_indices_from(synthetic_corr.values, k=1)]
        
        correlation_similarity = np.corrcoef(real_corr_values, synthetic_corr_values)[0, 1]
        
        # Calculate mean absolute difference
        mean_abs_diff = np.mean(np.abs(real_corr.values - synthetic_corr.values))
        
        return {
            'correlation_similarity': correlation_similarity,
            'mean_absolute_difference': mean_abs_diff,
            'real_correlation_matrix': real_corr.to_dict(),
            'synthetic_correlation_matrix': synthetic_corr.to_dict()
        }
    
    def distribution_comparison(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> Dict[str, Any]:
        """Compare distributions using various statistical measures"""
        results = {}
        
        for column in real_data.columns:
            if column not in synthetic_data.columns:
                continue
            
            real_values = real_data[column].dropna()
            synthetic_values = synthetic_data[column].dropna()
            
            if len(real_values) == 0 or len(synthetic_values) == 0:
                continue
            
            column_results = {}
            
            if real_data[column].dtype in ['int64', 'float64']:
                # Numerical column
                column_results.update({
                    'mean_difference': abs(real_values.mean() - synthetic_values.mean()),
                    'std_difference': abs(real_values.std() - synthetic_values.std()),
                    'median_difference': abs(real_values.median() - synthetic_values.median()),
                    'real_mean': real_values.mean(),
                    'synthetic_mean': synthetic_values.mean(),
                    'real_std': real_values.std(),
                    'synthetic_std': synthetic_values.std()
                })
            else:
                # Categorical column
                real_counts = real_values.value_counts(normalize=True)
                synthetic_counts = synthetic_values.value_counts(normalize=True)
                
                # Calculate Jensen-Shannon divergence
                all_categories = set(real_counts.index) | set(synthetic_counts.index)
                real_probs = np.array([real_counts.get(cat, 0) for cat in all_categories])
                synthetic_probs = np.array([synthetic_counts.get(cat, 0) for cat in all_categories])
                
                # Avoid log(0) by adding small epsilon
                epsilon = 1e-10
                real_probs = real_probs + epsilon
                synthetic_probs = synthetic_probs + epsilon
                
                # Normalize
                real_probs = real_probs / real_probs.sum()
                synthetic_probs = synthetic_probs / synthetic_probs.sum()
                
                m = 0.5 * (real_probs + synthetic_probs)
                js_divergence = 0.5 * stats.entropy(real_probs, m) + 0.5 * stats.entropy(synthetic_probs, m)
                
                column_results.update({
                    'js_divergence': js_divergence,
                    'category_overlap': len(set(real_counts.index) & set(synthetic_counts.index)) / len(set(real_counts.index) | set(synthetic_counts.index)),
                    'real_unique_count': len(real_counts),
                    'synthetic_unique_count': len(synthetic_counts)
                })
            
            results[column] = column_results
        
        return results

class UtilityEvaluator:
    """Evaluates utility of synthetic data by training ML models"""
    
    def __init__(self):
        self.results = {}
    
    def train_test_comparison(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, 
                            target_column: str = None, task_type: str = 'auto') -> Dict[str, Any]:
        """Compare ML model performance on real vs synthetic data"""
        
        if target_column is None:
            # Try to infer target column (last column or most correlated)
            numerical_cols = real_data.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 1:
                target_column = numerical_cols[-1]  # Use last numerical column
            else:
                return {"error": "No suitable target column found for ML evaluation"}
        
        if target_column not in real_data.columns or target_column not in synthetic_data.columns:
            return {"error": f"Target column '{target_column}' not found in both datasets"}
        
        # Prepare data
        real_features = real_data.drop(columns=[target_column])
        real_target = real_data[target_column]
        synthetic_features = synthetic_data.drop(columns=[target_column])
        synthetic_target = synthetic_data[target_column]
        
        # Handle categorical variables
        categorical_cols = real_features.select_dtypes(include=['object']).columns
        label_encoders = {}
        
        for col in categorical_cols:
            if col in synthetic_features.columns:
                le = LabelEncoder()
                # Fit on combined data to handle unseen categories
                combined_values = pd.concat([real_features[col], synthetic_features[col]]).astype(str)
                le.fit(combined_values)
                
                real_features[col] = le.transform(real_features[col].astype(str))
                synthetic_features[col] = le.transform(synthetic_features[col].astype(str))
                label_encoders[col] = le
        
        # Determine task type
        if task_type == 'auto':
            if real_target.dtype == 'object' or real_target.nunique() < 10:
                task_type = 'classification'
            else:
                task_type = 'regression'
        
        results = {'task_type': task_type, 'target_column': target_column}
        
        try:
            if task_type == 'classification':
                # Encode target if categorical
                if real_target.dtype == 'object':
                    target_encoder = LabelEncoder()
                    combined_targets = pd.concat([real_target, synthetic_target]).astype(str)
                    target_encoder.fit(combined_targets)
                    real_target = target_encoder.transform(real_target.astype(str))
                    synthetic_target = target_encoder.transform(synthetic_target.astype(str))
                
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                metric_fn = accuracy_score
                metric_name = 'accuracy'
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                metric_fn = r2_score
                metric_name = 'r2_score'
            
            # Split real data for testing
            X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
                real_features, real_target, test_size=0.3, random_state=42
            )
            
            # Train on real, test on real (baseline)
            model.fit(X_real_train, y_real_train)
            real_pred = model.predict(X_real_test)
            real_score = metric_fn(y_real_test, real_pred)
            
            # Train on synthetic, test on real
            if len(synthetic_features) > 0:
                model.fit(synthetic_features, synthetic_target)
                synthetic_pred = model.predict(X_real_test)
                synthetic_score = metric_fn(y_real_test, synthetic_pred)
                
                # Calculate utility score (how close synthetic performance is to real)
                utility_score = synthetic_score / real_score if real_score > 0 else 0
            else:
                synthetic_score = 0
                utility_score = 0
            
            results.update({
                f'real_{metric_name}': real_score,
                f'synthetic_{metric_name}': synthetic_score,
                'utility_score': utility_score,
                'utility_percentage': utility_score * 100
            })
            
        except Exception as e:
            results['error'] = f"ML evaluation failed: {str(e)}"
        
        return results

class PrivacyEvaluator:
    """Evaluates privacy preservation in synthetic data"""
    
    def __init__(self):
        self.results = {}
    
    def nearest_neighbor_distance(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> Dict[str, Any]:
        """Check if any synthetic record is too similar to real records"""
        
        # Use only numerical columns for distance calculation
        numerical_cols = real_data.select_dtypes(include=[np.number]).columns
        common_cols = [col for col in numerical_cols if col in synthetic_data.columns]
        
        if len(common_cols) == 0:
            return {"error": "No numerical columns for distance calculation"}
        
        real_numerical = real_data[common_cols].fillna(0)
        synthetic_numerical = synthetic_data[common_cols].fillna(0)
        
        # Normalize data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        real_scaled = scaler.fit_transform(real_numerical)
        synthetic_scaled = scaler.transform(synthetic_numerical)
        
        # Calculate minimum distances
        min_distances = []
        for synthetic_row in synthetic_scaled:
            distances = np.sqrt(np.sum((real_scaled - synthetic_row) ** 2, axis=1))
            min_distances.append(np.min(distances))
        
        min_distances = np.array(min_distances)
        
        # Privacy metrics
        mean_min_distance = np.mean(min_distances)
        std_min_distance = np.std(min_distances)
        
        # Flag potentially problematic records (very close to real data)
        threshold = np.percentile(min_distances, 5)  # Bottom 5%
        risky_records = np.sum(min_distances < threshold)
        
        return {
            'mean_nearest_neighbor_distance': mean_min_distance,
            'std_nearest_neighbor_distance': std_min_distance,
            'min_distance': np.min(min_distances),
            'max_distance': np.max(min_distances),
            'risky_records_count': int(risky_records),
            'risky_records_percentage': (risky_records / len(min_distances)) * 100,
            'privacy_score': min(100, mean_min_distance * 100)  # Higher is better
        }
    
    def membership_inference_test(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> Dict[str, Any]:
        """Test if an attacker can determine if a record was in the training set"""
        
        try:
            # Prepare features (exclude target if exists)
            numerical_cols = real_data.select_dtypes(include=[np.number]).columns
            common_cols = [col for col in numerical_cols if col in synthetic_data.columns]
            
            if len(common_cols) < 2:
                return {"error": "Need at least 2 numerical columns for membership inference test"}
            
            # Create labels: 1 for real, 0 for synthetic
            real_features = real_data[common_cols].fillna(0)
            synthetic_features = synthetic_data[common_cols].fillna(0)
            
            # Balance the datasets
            min_size = min(len(real_features), len(synthetic_features))
            real_sample = real_features.sample(n=min_size, random_state=42)
            synthetic_sample = synthetic_features.sample(n=min_size, random_state=42)
            
            # Combine data
            X = pd.concat([real_sample, synthetic_sample])
            y = np.concatenate([np.ones(len(real_sample)), np.zeros(len(synthetic_sample))])
            
            # Train classifier to distinguish real from synthetic
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            classifier.fit(X_train, y_train)
            
            # Evaluate
            train_accuracy = classifier.score(X_train, y_train)
            test_accuracy = classifier.score(X_test, y_test)
            
            # Privacy score: lower accuracy means better privacy
            privacy_score = max(0, (0.5 - abs(test_accuracy - 0.5)) * 200)
            
            return {
                'membership_inference_accuracy': test_accuracy,
                'train_accuracy': train_accuracy,
                'privacy_score': privacy_score,
                'privacy_level': 'High' if test_accuracy < 0.6 else 'Medium' if test_accuracy < 0.75 else 'Low'
            }
            
        except Exception as e:
            return {"error": f"Membership inference test failed: {str(e)}"}

class QualityEvaluator:
    """Main class that orchestrates all evaluation methods"""
    
    def __init__(self):
        self.statistical_evaluator = StatisticalEvaluator()
        self.utility_evaluator = UtilityEvaluator()
        self.privacy_evaluator = PrivacyEvaluator()
    
    def evaluate_synthetic_data(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, 
                              target_column: str = None) -> Dict[str, Any]:
        """Comprehensive evaluation of synthetic data quality"""
        
        evaluation_report = {
            'metadata': {
                'real_data_shape': real_data.shape,
                'synthetic_data_shape': synthetic_data.shape,
                'evaluation_timestamp': pd.Timestamp.now().isoformat(),
                'target_column': target_column
            },
            'statistical_similarity': {},
            'utility_preservation': {},
            'privacy_protection': {},
            'overall_scores': {}
        }
        
        # Statistical similarity evaluation
        try:
            ks_results = self.statistical_evaluator.kolmogorov_smirnov_test(real_data, synthetic_data)
            correlation_results = self.statistical_evaluator.correlation_comparison(real_data, synthetic_data)
            distribution_results = self.statistical_evaluator.distribution_comparison(real_data, synthetic_data)
            
            evaluation_report['statistical_similarity'] = {
                'kolmogorov_smirnov': ks_results,
                'correlation_analysis': correlation_results,
                'distribution_comparison': distribution_results
            }
            
            # Calculate statistical similarity score
            if ks_results:
                ks_score = np.mean([result['similar'] for result in ks_results.values()]) * 100
            else:
                ks_score = 50
            
            if 'correlation_similarity' in correlation_results:
                corr_score = max(0, correlation_results['correlation_similarity'] * 100)
            else:
                corr_score = 50
            
            statistical_score = (ks_score + corr_score) / 2
            
        except Exception as e:
            evaluation_report['statistical_similarity']['error'] = str(e)
            statistical_score = 0
        
        # Utility evaluation
        try:
            utility_results = self.utility_evaluator.train_test_comparison(
                real_data, synthetic_data, target_column
            )
            evaluation_report['utility_preservation'] = utility_results
            
            if 'utility_percentage' in utility_results:
                utility_score = min(100, utility_results['utility_percentage'])
            else:
                utility_score = 0
                
        except Exception as e:
            evaluation_report['utility_preservation']['error'] = str(e)
            utility_score = 0
        
        # Privacy evaluation
        try:
            nn_results = self.privacy_evaluator.nearest_neighbor_distance(real_data, synthetic_data)
            mi_results = self.privacy_evaluator.membership_inference_test(real_data, synthetic_data)
            
            evaluation_report['privacy_protection'] = {
                'nearest_neighbor_analysis': nn_results,
                'membership_inference_test': mi_results
            }
            
            # Calculate privacy score
            nn_score = nn_results.get('privacy_score', 0) if 'error' not in nn_results else 0
            mi_score = mi_results.get('privacy_score', 0) if 'error' not in mi_results else 0
            privacy_score = (nn_score + mi_score) / 2
            
        except Exception as e:
            evaluation_report['privacy_protection']['error'] = str(e)
            privacy_score = 0
        
        # Calculate overall scores
        overall_score = (statistical_score * 0.4 + utility_score * 0.4 + privacy_score * 0.2)
        
        evaluation_report['overall_scores'] = {
            'statistical_similarity_score': round(statistical_score, 2),
            'utility_preservation_score': round(utility_score, 2),
            'privacy_protection_score': round(privacy_score, 2),
            'overall_quality_score': round(overall_score, 2),
            'grade': self._get_quality_grade(overall_score)
        }
        
        return evaluation_report
    
    def _get_quality_grade(self, score: float) -> str:
        """Convert numerical score to letter grade"""
        if score >= 90:
            return 'A+'
        elif score >= 85:
            return 'A'
        elif score >= 80:
            return 'A-'
        elif score >= 75:
            return 'B+'
        elif score >= 70:
            return 'B'
        elif score >= 65:
            return 'B-'
        elif score >= 60:
            return 'C+'
        elif score >= 55:
            return 'C'
        elif score >= 50:
            return 'C-'
        else:
            return 'F'