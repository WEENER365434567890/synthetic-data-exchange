"""
Intelligent Model Selection and Caching Manager
"""
import os
import pickle
import hashlib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import logging
from pathlib import Path

# SDV imports with fallbacks
try:
    from sdv.single_table import GaussianCopulaSynthesizer, TVAESynthesizer, CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata
except ImportError as e:
    logging.warning(f"SDV import error: {e}")

# Try to use Polars for faster preprocessing
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    logging.info("Polars not available, using pandas")

# PyTorch MPS backend for Apple Silicon - Disabled due to compatibility issues
try:
    import torch
    if torch.backends.mps.is_available():
        # Disable MPS for now due to SDV compatibility issues
        # torch.set_default_device('mps')
        logging.info("MPS available but disabled for SDV compatibility")
        # Force CPU usage for stability
        import os
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
except ImportError:
    logging.info("PyTorch not available")

logger = logging.getLogger(__name__)

class ModelManager:
    """Intelligent model selection, caching, and optimization"""
    
    def __init__(self, cache_dir: str = "model_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.model_cache = {}
        self.performance_stats = {}
        
    def detect_data_type(self, data: pd.DataFrame) -> str:
        """Detect the type of dataset to choose optimal model"""
        
        # Check for time series indicators
        datetime_cols = data.select_dtypes(include=['datetime64']).columns
        date_like_cols = [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower()]
        
        if len(datetime_cols) > 0 or len(date_like_cols) > 0:
            # Check if data is sorted by time (time series indicator)
            if len(datetime_cols) > 0:
                time_col = datetime_cols[0]
                if data[time_col].is_monotonic_increasing:
                    return "timeseries"
        
        # Check for geospatial indicators
        geo_indicators = ['lat', 'lon', 'latitude', 'longitude', 'coord', 'location']
        geo_cols = [col for col in data.columns if any(geo in col.lower() for geo in geo_indicators)]
        
        if len(geo_cols) >= 2:
            return "geospatial"
        
        # Default to tabular
        return "tabular"
    
    def choose_optimal_model(self, data: pd.DataFrame, data_type: str, speed_priority: bool = True) -> str:
        """Choose the best model based on data characteristics and speed requirements"""
        
        rows, cols = data.shape
        
        if data_type == "timeseries":
            if speed_priority or rows < 10000:
                return "statistical_timeseries"  # Fast statistical approach
            else:
                return "timegan"  # More accurate but slower
        
        elif data_type == "geospatial":
            return "noise_based_geo"  # Fast noise-based generation
        
        else:  # tabular
            if speed_priority:
                if rows < 1000 and cols < 20:
                    return "gaussian_copula"  # Fastest for small data
                elif rows < 50000:
                    return "tvae"  # Good balance
                else:
                    return "gaussian_copula"  # Scale better than GANs
            else:
                if rows < 10000:
                    return "tvae"  # Good quality
                else:
                    return "ctgan"  # Best quality for large data
    
    def get_model_cache_key(self, data: pd.DataFrame, model_type: str, constraints: list = None) -> str:
        """Generate a unique cache key for the model based on data characteristics"""
        
        # Create a hash based on data structure and constraints
        data_signature = {
            'columns': list(data.columns),
            'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()},
            'shape': data.shape,
            'model_type': model_type
        }
        
        if constraints:
            data_signature['constraints'] = [str(c) for c in constraints]
        
        signature_str = str(sorted(data_signature.items()))
        return hashlib.md5(signature_str.encode()).hexdigest()
    
    def save_model(self, model, cache_key: str, metadata: Dict[str, Any]):
        """Save trained model to cache"""
        try:
            cache_path = self.cache_dir / f"{cache_key}.pkl"
            metadata_path = self.cache_dir / f"{cache_key}_metadata.pkl"
            
            with open(cache_path, 'wb') as f:
                pickle.dump(model, f)
            
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Model cached: {cache_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to cache model: {e}")
            return False
    
    def load_model(self, cache_key: str) -> Tuple[Any, Dict[str, Any]]:
        """Load trained model from cache"""
        try:
            cache_path = self.cache_dir / f"{cache_key}.pkl"
            metadata_path = self.cache_dir / f"{cache_key}_metadata.pkl"
            
            if not cache_path.exists() or not metadata_path.exists():
                return None, None
            
            with open(cache_path, 'rb') as f:
                model = pickle.load(f)
            
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            logger.info(f"Model loaded from cache: {cache_key}")
            return model, metadata
        except Exception as e:
            logger.error(f"Failed to load cached model: {e}")
            return None, None
    
    def preprocess_data_fast(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fast data preprocessing using Polars if available"""
        
        if POLARS_AVAILABLE and len(data) > 10000:
            try:
                # Convert to Polars for faster processing
                pl_df = pl.from_pandas(data)
                
                # Fast preprocessing operations
                pl_df = pl_df.drop_nulls()  # Remove null rows
                
                # Convert back to pandas for SDV compatibility
                processed_data = pl_df.to_pandas()
                logger.info("Used Polars for fast preprocessing")
                return processed_data
            except Exception as e:
                logger.warning(f"Polars preprocessing failed, using pandas: {e}")
        
        # Fallback to pandas
        return data.dropna()
    
    def create_synthesizer(self, model_type: str, metadata: SingleTableMetadata, **kwargs) -> Any:
        """Create the appropriate synthesizer based on model type"""
        
        if model_type == "gaussian_copula":
            return GaussianCopulaSynthesizer(metadata, **kwargs)
        
        elif model_type == "tvae":
            # Optimized TVAE settings for speed
            tvae_kwargs = {
                'epochs': kwargs.get('epochs', 100),
                'batch_size': kwargs.get('batch_size', 500),
                'verbose': False
            }
            return TVAESynthesizer(metadata, **tvae_kwargs)
        
        elif model_type == "ctgan":
            # Optimized CTGAN settings
            ctgan_kwargs = {
                'epochs': kwargs.get('epochs', 100),
                'batch_size': kwargs.get('batch_size', 500),
                'verbose': False
            }
            return CTGANSynthesizer(metadata, **ctgan_kwargs)
        
        else:
            # Default to Gaussian Copula for unknown types
            logger.warning(f"Unknown model type {model_type}, using Gaussian Copula")
            return GaussianCopulaSynthesizer(metadata)
    
    def generate_synthetic_data(self, 
                              data: pd.DataFrame, 
                              num_rows: int,
                              speed_priority: bool = True,
                              constraints: list = None,
                              force_retrain: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Main method to generate synthetic data with optimal model selection and caching
        """
        start_time = datetime.now()
        
        # Step 1: Preprocess data for speed
        processed_data = self.preprocess_data_fast(data)
        
        # Step 2: Detect data type and choose model
        data_type = self.detect_data_type(processed_data)
        model_type = self.choose_optimal_model(processed_data, data_type, speed_priority)
        
        logger.info(f"Detected data type: {data_type}, chosen model: {model_type}")
        
        # Step 3: Check cache
        cache_key = self.get_model_cache_key(processed_data, model_type, constraints)
        
        if not force_retrain:
            cached_model, cached_metadata = self.load_model(cache_key)
            if cached_model is not None:
                try:
                    synthetic_data = cached_model.sample(num_rows)
                    generation_time = (datetime.now() - start_time).total_seconds()
                    
                    stats = {
                        'model_type': model_type,
                        'data_type': data_type,
                        'cached': True,
                        'generation_time': generation_time,
                        'rows_generated': len(synthetic_data)
                    }
                    
                    return synthetic_data, stats
                except Exception as e:
                    logger.warning(f"Cached model failed, retraining: {e}")
        
        # Step 4: Train new model
        try:
            # Create metadata
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(processed_data)
            
            # Create and train synthesizer
            if model_type == "gaussian_copula":
                synthesizer = self.create_synthesizer(model_type, metadata)
                train_start = datetime.now()
                synthesizer.fit(processed_data)
                train_time = (datetime.now() - train_start).total_seconds()
                
            elif model_type == "tvae":
                # Optimized for speed
                epochs = 50 if speed_priority else 100
                synthesizer = self.create_synthesizer(model_type, metadata, epochs=epochs)
                train_start = datetime.now()
                synthesizer.fit(processed_data)
                train_time = (datetime.now() - train_start).total_seconds()
                
            else:  # ctgan or fallback
                epochs = 50 if speed_priority else 100
                synthesizer = self.create_synthesizer("ctgan", metadata, epochs=epochs)
                train_start = datetime.now()
                synthesizer.fit(processed_data)
                train_time = (datetime.now() - train_start).total_seconds()
            
            # Step 5: Generate synthetic data
            synthetic_data = synthesizer.sample(num_rows)
            
            # Step 6: Cache the model
            model_metadata = {
                'model_type': model_type,
                'data_type': data_type,
                'train_time': train_time,
                'created_at': datetime.now().isoformat(),
                'data_shape': processed_data.shape
            }
            
            self.save_model(synthesizer, cache_key, model_metadata)
            
            generation_time = (datetime.now() - start_time).total_seconds()
            
            stats = {
                'model_type': model_type,
                'data_type': data_type,
                'cached': False,
                'train_time': train_time,
                'generation_time': generation_time,
                'rows_generated': len(synthetic_data)
            }
            
            return synthetic_data, stats
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            # Fallback to simple statistical generation
            return self._fallback_generation(processed_data, num_rows), {
                'model_type': 'fallback_statistical',
                'error': str(e),
                'generation_time': (datetime.now() - start_time).total_seconds()
            }
    
    def _fallback_generation(self, data: pd.DataFrame, num_rows: int) -> pd.DataFrame:
        """Simple statistical fallback when all else fails"""
        
        synthetic_data = pd.DataFrame()
        
        for column in data.columns:
            if data[column].dtype in ['int64', 'float64']:
                # Numerical: sample from normal distribution with same mean/std
                mean = data[column].mean()
                std = data[column].std()
                synthetic_data[column] = np.random.normal(mean, std, num_rows)
            else:
                # Categorical: sample with replacement
                synthetic_data[column] = np.random.choice(data[column].dropna(), num_rows, replace=True)
        
        return synthetic_data
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about cached models"""
        
        cache_files = list(self.cache_dir.glob("*.pkl"))
        model_files = [f for f in cache_files if not f.name.endswith("_metadata.pkl")]
        
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            'cached_models': len(model_files),
            'total_cache_size_mb': total_size / (1024 * 1024),
            'cache_directory': str(self.cache_dir)
        }
    
    def clear_cache(self, older_than_days: int = 30):
        """Clear old cached models"""
        
        cutoff_time = datetime.now().timestamp() - (older_than_days * 24 * 60 * 60)
        removed_count = 0
        
        for cache_file in self.cache_dir.glob("*.pkl"):
            if cache_file.stat().st_mtime < cutoff_time:
                cache_file.unlink()
                removed_count += 1
        
        logger.info(f"Removed {removed_count} old cache files")
        return removed_count