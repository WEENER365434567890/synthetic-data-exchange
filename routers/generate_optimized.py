"""
Optimized Generation API - Intelligent Model Selection & Caching
"""
from fastapi import APIRouter, File, UploadFile, Depends, HTTPException, Form, Query
from fastapi.responses import StreamingResponse, JSONResponse
import pandas as pd
import io
import os
from datetime import datetime
from dependencies import validate_csv_file, validate_csv_content
from model_manager import ModelManager
from typing import Optional
from sqlalchemy.orm import Session
from database import SessionLocal, UsageLog
from routers.auth import get_current_user

router = APIRouter()

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize optimized model manager
model_manager = ModelManager()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/generate/optimized")
async def generate_optimized(
    file: UploadFile = Depends(validate_csv_file), 
    num_rows: Optional[int] = Form(100),
    speed_priority: bool = Form(True),
    force_retrain: bool = Form(False),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Optimized generation with intelligent model selection and caching
    - Automatically chooses best model (Gaussian Copula, TVAE, CTGAN)
    - Caches trained models for instant reuse
    - Uses Polars for fast preprocessing when available
    - Apple MPS acceleration for neural models
    """
    try:
        # Validate parameters
        rows_to_generate = num_rows or 100
        if rows_to_generate <= 0:
            raise HTTPException(status_code=400, detail="Number of rows must be positive")
        if rows_to_generate > 50000:
            raise HTTPException(status_code=400, detail="Number of rows cannot exceed 50,000")

        contents = await file.read()
        
        # Fast CSV validation
        data = await validate_csv_content(contents)
        
        # Limit training data size for performance (sample if too large)
        if len(data) > 100000:
            data = data.sample(n=100000, random_state=42)
            print(f"Sampled data to 100k rows for faster training")

        # Generate synthetic data with intelligent model selection
        synthetic_data, generation_stats = model_manager.generate_synthetic_data(
            data=data,
            num_rows=rows_to_generate,
            speed_priority=speed_priority,
            force_retrain=force_retrain
        )

        # Save synthetic data for debugging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        synthetic_filename = f"optimized_{timestamp}_{file.filename}"
        synthetic_filepath = os.path.join(DATA_DIR, synthetic_filename)
        synthetic_data.to_csv(synthetic_filepath, index=False)
        
        # Log the generation
        log = UsageLog(user_id=current_user.id, dataset_name=synthetic_filename)
        db.add(log)
        db.commit()

        # Return CSV with performance stats in headers
        output = io.StringIO()
        synthetic_data.to_csv(output, index=False)
        output.seek(0)

        headers = {
            "Content-Disposition": f"attachment; filename=optimized_{file.filename}",
            "X-Model-Type": generation_stats.get('model_type', 'gaussian_copula'),
            "X-Generation-Time": str(round(generation_stats.get('generation_time', 0), 2)),
            "X-Cached": str(generation_stats.get('cached', False)),
            "X-Data-Type": generation_stats.get('data_type', 'tabular'),
            "Access-Control-Expose-Headers": "X-Model-Type,X-Generation-Time,X-Cached,X-Data-Type"
        }

        return StreamingResponse(
            io.StringIO(output.getvalue()), 
            media_type="text/csv", 
            headers=headers
        )
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        print(f"Optimized generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@router.post("/generate/ultra-fast")
async def generate_ultra_fast(
    file: UploadFile = Depends(validate_csv_file), 
    num_rows: Optional[int] = Form(100),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Ultra-fast generation using only Gaussian Copula (1-5 seconds)
    Best for: Quick prototyping, small datasets, when speed is critical
    """
    try:
        rows_to_generate = num_rows or 100
        if rows_to_generate <= 0:
            raise HTTPException(status_code=400, detail="Number of rows must be positive")

        contents = await file.read()
        data = await validate_csv_content(contents)
        
        # Force Gaussian Copula for maximum speed
        from sdv.single_table import GaussianCopulaSynthesizer
        from sdv.metadata import SingleTableMetadata
        
        # Preprocess quickly
        processed_data = model_manager.preprocess_data_fast(data)
        
        # Simple metadata
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(processed_data)
        
        # Ultra-fast generation
        start_time = datetime.now()
        synthesizer = GaussianCopulaSynthesizer(metadata)
        synthesizer.fit(processed_data)
        synthetic_data = synthesizer.sample(rows_to_generate)
        generation_time = (datetime.now() - start_time).total_seconds()
        
        # Log the generation
        log = UsageLog(user_id=current_user.id, dataset_name=f"ultra_fast_{file.filename}")
        db.add(log)
        db.commit()

        # Return CSV
        output = io.StringIO()
        synthetic_data.to_csv(output, index=False)
        output.seek(0)

        headers = {
            "Content-Disposition": f"attachment; filename=ultra_fast_{file.filename}",
            "X-Model-Type": "gaussian_copula",
            "X-Generation-Time": str(round(generation_time, 2)),
            "X-Cached": "false",
            "X-Data-Type": "tabular",
            "X-Speed-Mode": "ultra-fast",
            "Access-Control-Expose-Headers": "X-Model-Type,X-Generation-Time,X-Cached,X-Data-Type,X-Speed-Mode"
        }

        return StreamingResponse(
            io.StringIO(output.getvalue()), 
            media_type="text/csv", 
            headers=headers
        )
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        print(f"Ultra-fast generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@router.get("/generate/model-cache/stats")
async def get_cache_stats(current_user = Depends(get_current_user)):
    """Get statistics about cached models"""
    try:
        stats = model_manager.get_cache_stats()
        return {
            "cache_stats": stats,
            "performance_benefits": {
                "cached_models_available": stats["cached_models"],
                "estimated_time_saved": f"{stats['cached_models'] * 30}+ seconds",
                "cache_size": f"{stats['total_cache_size_mb']:.1f} MB"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting cache stats: {str(e)}")

@router.delete("/generate/model-cache/clear")
async def clear_model_cache(
    older_than_days: int = Query(30, description="Clear models older than X days"),
    current_user = Depends(get_current_user)
):
    """Clear old cached models to free up space"""
    try:
        removed_count = model_manager.clear_cache(older_than_days)
        return {
            "message": f"Cleared {removed_count} old cached models",
            "older_than_days": older_than_days
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")

@router.get("/generate/performance-comparison")
async def get_performance_comparison():
    """Get comparison of different generation modes"""
    return {
        "generation_modes": {
            "ultra_fast": {
                "endpoint": "/generate/ultra-fast",
                "model": "Gaussian Copula",
                "time": "1-5 seconds",
                "quality": "Good",
                "best_for": "Quick prototyping, small datasets",
                "features": ["Statistical modeling", "Preserves correlations", "No neural networks"]
            },
            "optimized": {
                "endpoint": "/generate/optimized", 
                "model": "Auto-selected (Gaussian Copula/TVAE/CTGAN)",
                "time": "5-30 seconds (cached: <5s)",
                "quality": "Very Good to Excellent",
                "best_for": "Production use, automatic optimization",
                "features": ["Model caching", "Intelligent selection", "Apple MPS acceleration"]
            },
            "fast": {
                "endpoint": "/generate/fast",
                "model": "CTGAN (reduced epochs)",
                "time": "10-60 seconds",
                "quality": "Good",
                "best_for": "Balanced speed/quality",
                "features": ["Neural networks", "Reduced training time"]
            },
            "enterprise": {
                "endpoint": "/generate",
                "model": "Full CTGAN + constraints + evaluation",
                "time": "60-300+ seconds",
                "quality": "Excellent",
                "best_for": "Enterprise compliance, research",
                "features": ["Full constraints", "Quality evaluation", "Privacy analysis"]
            }
        },
        "recommendations": {
            "development": "Use ultra-fast for quick iterations",
            "production": "Use optimized for best speed/quality balance",
            "compliance": "Use enterprise for full validation",
            "caching": "Optimized mode caches models for instant reuse"
        },
        "performance_tips": {
            "model_caching": "Repeated schemas generate instantly from cache",
            "data_size": "Large datasets (>100k rows) are automatically sampled for training",
            "apple_silicon": "Neural models use MPS acceleration on M1/M2 Macs",
            "polars": "Large datasets use Polars for faster preprocessing"
        }
    }

@router.post("/generate/benchmark")
async def benchmark_generation_modes(
    file: UploadFile = Depends(validate_csv_file),
    num_rows: int = Form(100),
    current_user = Depends(get_current_user)
):
    """
    Benchmark different generation modes on your dataset
    Returns timing and quality comparisons
    """
    try:
        contents = await file.read()
        data = await validate_csv_content(contents)
        
        # Limit to small sample for benchmarking
        if len(data) > 1000:
            data = data.sample(n=1000, random_state=42)
        
        benchmark_results = {}
        
        # Test ultra-fast mode
        try:
            start_time = datetime.now()
            synthetic_data, stats = model_manager.generate_synthetic_data(
                data, num_rows, speed_priority=True, force_retrain=True
            )
            ultra_fast_time = (datetime.now() - start_time).total_seconds()
            
            benchmark_results["ultra_fast"] = {
                "time_seconds": ultra_fast_time,
                "model_used": stats.get('model_type'),
                "rows_generated": len(synthetic_data),
                "status": "success"
            }
        except Exception as e:
            benchmark_results["ultra_fast"] = {"status": "failed", "error": str(e)}
        
        # Test optimized mode
        try:
            start_time = datetime.now()
            synthetic_data, stats = model_manager.generate_synthetic_data(
                data, num_rows, speed_priority=False, force_retrain=True
            )
            optimized_time = (datetime.now() - start_time).total_seconds()
            
            benchmark_results["optimized"] = {
                "time_seconds": optimized_time,
                "model_used": stats.get('model_type'),
                "rows_generated": len(synthetic_data),
                "status": "success"
            }
        except Exception as e:
            benchmark_results["optimized"] = {"status": "failed", "error": str(e)}
        
        return {
            "benchmark_results": benchmark_results,
            "dataset_info": {
                "rows": len(data),
                "columns": len(data.columns),
                "data_type": model_manager.detect_data_type(data)
            },
            "recommendations": {
                "fastest": min(benchmark_results.keys(), key=lambda x: benchmark_results[x].get('time_seconds', float('inf'))),
                "suggested_mode": "ultra_fast" if len(data) < 10000 else "optimized"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")