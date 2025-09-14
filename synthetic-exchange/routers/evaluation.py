"""
Evaluation API Endpoints for Synthetic Data Quality Assessment
"""
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import pandas as pd
import json
import io
import os
from datetime import datetime

from evaluation_manager import QualityEvaluator
from models import EvaluationRequest, EvaluationResponse
from routers.auth import get_current_user
from database import SessionLocal, UsageLog

router = APIRouter()
quality_evaluator = QualityEvaluator()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_synthetic_data(
    real_data_file: UploadFile = File(..., description="Original real dataset"),
    synthetic_data_file: UploadFile = File(..., description="Generated synthetic dataset"),
    target_column: Optional[str] = Form(None, description="Target column for ML evaluation"),
    include_privacy_analysis: bool = Form(True),
    include_utility_analysis: bool = Form(True),
    include_statistical_analysis: bool = Form(True),
    current_user = Depends(get_current_user),
    db = Depends(get_db)
):
    """
    Comprehensive evaluation of synthetic data quality including:
    - Statistical similarity (KS tests, correlation analysis)
    - Utility preservation (ML model performance comparison)
    - Privacy protection (nearest neighbor, membership inference)
    """
    try:
        # Read real data
        real_contents = await real_data_file.read()
        if not real_contents:
            raise HTTPException(status_code=400, detail="Real data file is empty")
        
        real_df = pd.read_csv(io.StringIO(real_contents.decode('utf-8')))
        if real_df.empty:
            raise HTTPException(status_code=400, detail="Real data file contains no data")
        
        # Read synthetic data
        synthetic_contents = await synthetic_data_file.read()
        if not synthetic_contents:
            raise HTTPException(status_code=400, detail="Synthetic data file is empty")
        
        synthetic_df = pd.read_csv(io.StringIO(synthetic_contents.decode('utf-8')))
        if synthetic_df.empty:
            raise HTTPException(status_code=400, detail="Synthetic data file contains no data")
        
        # Validate that datasets have compatible structure
        if len(real_df.columns) == 0 or len(synthetic_df.columns) == 0:
            raise HTTPException(status_code=400, detail="Datasets must have at least one column")
        
        # Perform comprehensive evaluation
        evaluation_report = quality_evaluator.evaluate_synthetic_data(
            real_data=real_df,
            synthetic_data=synthetic_df,
            target_column=target_column
        )
        
        # Generate recommendations based on scores
        recommendations = _generate_recommendations(evaluation_report)
        
        # Log the evaluation
        log = UsageLog(
            user_id=current_user.id, 
            dataset_name=f"evaluation_{real_data_file.filename}_{synthetic_data_file.filename}"
        )
        db.add(log)
        db.commit()
        
        # Prepare response
        overall_scores = evaluation_report.get('overall_scores', {})
        
        response = EvaluationResponse(
            overall_quality_score=overall_scores.get('overall_quality_score', 0),
            grade=overall_scores.get('grade', 'F'),
            statistical_similarity_score=overall_scores.get('statistical_similarity_score', 0),
            utility_preservation_score=overall_scores.get('utility_preservation_score', 0),
            privacy_protection_score=overall_scores.get('privacy_protection_score', 0),
            detailed_report=evaluation_report,
            recommendations=recommendations
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during evaluation: {str(e)}")

@router.post("/evaluate/quick")
async def quick_evaluation(
    real_data_file: UploadFile = File(...),
    synthetic_data_file: UploadFile = File(...),
    current_user = Depends(get_current_user)
):
    """
    Quick evaluation focusing on basic statistical similarity
    """
    try:
        # Read data files
        real_contents = await real_data_file.read()
        synthetic_contents = await synthetic_data_file.read()
        
        real_df = pd.read_csv(io.StringIO(real_contents.decode('utf-8')))
        synthetic_df = pd.read_csv(io.StringIO(synthetic_contents.decode('utf-8')))
        
        # Quick statistical analysis only
        ks_results = quality_evaluator.statistical_evaluator.kolmogorov_smirnov_test(real_df, synthetic_df)
        correlation_results = quality_evaluator.statistical_evaluator.correlation_comparison(real_df, synthetic_df)
        
        # Calculate quick score
        if ks_results:
            ks_score = sum([1 for result in ks_results.values() if result.get('similar', False)]) / len(ks_results) * 100
        else:
            ks_score = 50
        
        if 'correlation_similarity' in correlation_results:
            corr_score = max(0, correlation_results['correlation_similarity'] * 100)
        else:
            corr_score = 50
        
        quick_score = (ks_score + corr_score) / 2
        
        return {
            "quick_quality_score": round(quick_score, 2),
            "grade": quality_evaluator._get_quality_grade(quick_score),
            "ks_test_results": ks_results,
            "correlation_analysis": correlation_results,
            "message": "Quick evaluation completed. Use full evaluation for comprehensive analysis."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during quick evaluation: {str(e)}")

@router.get("/evaluate/metrics")
async def get_evaluation_metrics():
    """
    Get information about available evaluation metrics
    """
    return {
        "statistical_similarity": {
            "kolmogorov_smirnov_test": {
                "description": "Tests if real and synthetic data come from the same distribution",
                "interpretation": "p-value > 0.05 indicates similar distributions"
            },
            "correlation_analysis": {
                "description": "Compares correlation matrices between real and synthetic data",
                "interpretation": "Higher correlation similarity indicates better preservation of relationships"
            },
            "distribution_comparison": {
                "description": "Compares statistical properties (mean, std, median) of each column",
                "interpretation": "Lower differences indicate better statistical fidelity"
            }
        },
        "utility_preservation": {
            "ml_model_comparison": {
                "description": "Trains ML models on real vs synthetic data and compares performance",
                "interpretation": "Higher utility score means synthetic data preserves predictive patterns"
            }
        },
        "privacy_protection": {
            "nearest_neighbor_distance": {
                "description": "Measures how close synthetic records are to real records",
                "interpretation": "Higher distances indicate better privacy protection"
            },
            "membership_inference_test": {
                "description": "Tests if an attacker can determine if a record was in training data",
                "interpretation": "Lower accuracy indicates better privacy protection"
            }
        },
        "scoring": {
            "overall_score": "Weighted average: 40% statistical + 40% utility + 20% privacy",
            "grades": {
                "A+": "90-100: Excellent quality",
                "A": "85-89: Very good quality", 
                "B": "70-84: Good quality",
                "C": "50-69: Acceptable quality",
                "F": "0-49: Poor quality"
            }
        }
    }

@router.post("/evaluate/batch")
async def batch_evaluation(
    files: List[UploadFile] = File(..., description="Multiple synthetic datasets to evaluate against first file as real data"),
    target_column: Optional[str] = Form(None),
    current_user = Depends(get_current_user)
):
    """
    Evaluate multiple synthetic datasets against the same real dataset
    """
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 files: 1 real dataset + 1+ synthetic datasets")
    
    try:
        # First file is the real dataset
        real_contents = await files[0].read()
        real_df = pd.read_csv(io.StringIO(real_contents.decode('utf-8')))
        
        results = []
        
        # Evaluate each synthetic dataset
        for i, synthetic_file in enumerate(files[1:], 1):
            synthetic_contents = await synthetic_file.read()
            synthetic_df = pd.read_csv(io.StringIO(synthetic_contents.decode('utf-8')))
            
            evaluation_report = quality_evaluator.evaluate_synthetic_data(
                real_data=real_df,
                synthetic_data=synthetic_df,
                target_column=target_column
            )
            
            overall_scores = evaluation_report.get('overall_scores', {})
            
            results.append({
                "dataset_name": synthetic_file.filename,
                "overall_quality_score": overall_scores.get('overall_quality_score', 0),
                "grade": overall_scores.get('grade', 'F'),
                "statistical_similarity_score": overall_scores.get('statistical_similarity_score', 0),
                "utility_preservation_score": overall_scores.get('utility_preservation_score', 0),
                "privacy_protection_score": overall_scores.get('privacy_protection_score', 0)
            })
        
        # Sort by overall quality score
        results.sort(key=lambda x: x['overall_quality_score'], reverse=True)
        
        return {
            "real_dataset": files[0].filename,
            "evaluated_datasets": len(results),
            "results": results,
            "best_dataset": results[0] if results else None,
            "summary": {
                "average_quality_score": sum(r['overall_quality_score'] for r in results) / len(results) if results else 0,
                "datasets_above_70": len([r for r in results if r['overall_quality_score'] >= 70]),
                "datasets_above_80": len([r for r in results if r['overall_quality_score'] >= 80])
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during batch evaluation: {str(e)}")

def _generate_recommendations(evaluation_report: Dict[str, Any]) -> List[str]:
    """Generate actionable recommendations based on evaluation results"""
    recommendations = []
    
    overall_scores = evaluation_report.get('overall_scores', {})
    statistical_score = overall_scores.get('statistical_similarity_score', 0)
    utility_score = overall_scores.get('utility_preservation_score', 0)
    privacy_score = overall_scores.get('privacy_protection_score', 0)
    
    # Statistical similarity recommendations
    if statistical_score < 70:
        recommendations.append("Consider increasing training epochs or adjusting model hyperparameters to improve statistical similarity")
        recommendations.append("Check if your dataset has sufficient size and diversity for training")
    
    # Utility preservation recommendations
    if utility_score < 60:
        recommendations.append("The synthetic data may not preserve important patterns. Consider using domain-specific constraints")
        recommendations.append("Try different synthetic data generation models (e.g., CTGAN, TVAE, CopulaGAN)")
    
    # Privacy protection recommendations
    if privacy_score < 50:
        recommendations.append("Consider adding more noise or using differential privacy techniques")
        recommendations.append("Increase the size of your training dataset to improve privacy protection")
    
    # Overall recommendations
    overall_score = overall_scores.get('overall_quality_score', 0)
    if overall_score >= 80:
        recommendations.append("Excellent quality! This synthetic data is suitable for most use cases")
    elif overall_score >= 70:
        recommendations.append("Good quality synthetic data. Consider minor improvements for critical applications")
    elif overall_score >= 50:
        recommendations.append("Acceptable quality but improvements recommended before production use")
    else:
        recommendations.append("Quality is below acceptable threshold. Significant improvements needed")
    
    # Specific technical recommendations
    statistical_similarity = evaluation_report.get('statistical_similarity', {})
    if 'correlation_analysis' in statistical_similarity:
        corr_sim = statistical_similarity['correlation_analysis'].get('correlation_similarity', 0)
        if corr_sim < 0.7:
            recommendations.append("Correlation patterns are not well preserved. Consider using constraint-based generation")
    
    return recommendations