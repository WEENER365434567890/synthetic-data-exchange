"""
Fast Generation API - Optimized for Speed
"""
from fastapi import APIRouter, File, UploadFile, Depends, HTTPException, Form, Body
from fastapi.responses import StreamingResponse
import pandas as pd
import io
import os
import json
from datetime import datetime
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from dependencies import validate_csv_file, validate_csv_content
from models import GenerateRequest
from typing import Optional, Union
from sqlalchemy.orm import Session
from database import SessionLocal, UsageLog
from routers.auth import get_current_user

router = APIRouter()

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/generate/fast")
async def generate_fast(
    file: UploadFile = Depends(validate_csv_file), 
    num_rows: Optional[int] = Form(100),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Fast generation with minimal processing - optimized for speed
    Skips: schema validation, constraints, evaluation
    """
    try:
        # Validate num_rows
        rows_to_generate = num_rows or 100
        if rows_to_generate <= 0:
            raise HTTPException(status_code=400, detail="Number of rows must be positive")
        if rows_to_generate > 10000:
            raise HTTPException(status_code=400, detail="Number of rows cannot exceed 10,000")

        contents = await file.read()
        
        # Basic CSV validation only
        data = await validate_csv_content(contents)

        # Simple metadata detection
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=data)

        # Fast generation with minimal epochs
        try:
            synthesizer = CTGANSynthesizer(metadata, epochs=50, verbose=False)
            synthesizer.fit(data)
        except Exception as e:
            print(f"CTGAN error: {e}")
            # Fallback to basic synthesizer
            synthesizer = CTGANSynthesizer(metadata)
            synthesizer.fit(data)

        synthetic_data = synthesizer.sample(num_rows=rows_to_generate)

        # Log the generation (async to not block response)
        log = UsageLog(user_id=current_user.id, dataset_name=f"fast_gen_{file.filename}")
        db.add(log)
        db.commit()

        # Return CSV immediately
        output = io.StringIO()
        synthetic_data.to_csv(output, index=False)
        output.seek(0)

        headers = {
            "Content-Disposition": f"attachment; filename=fast_synthetic_{file.filename}",
            "X-Model-Type": "ctgan_fast",
            "X-Generation-Time": str(round((datetime.now() - start_time).total_seconds(), 2)),
            "X-Cached": "false",
            "X-Data-Type": "tabular",
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
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate/balanced")
async def generate_balanced(
    file: UploadFile = Depends(validate_csv_file), 
    num_rows: Optional[int] = Form(100),
    use_schema: bool = Form(False),
    schema_template: Optional[str] = Form(None),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Balanced generation - good quality with reasonable speed
    Includes: basic schema support, no evaluation
    """
    try:
        rows_to_generate = num_rows or 100
        if rows_to_generate <= 0:
            raise HTTPException(status_code=400, detail="Number of rows must be positive")
        if rows_to_generate > 10000:
            raise HTTPException(status_code=400, detail="Number of rows cannot exceed 10,000")

        contents = await file.read()
        data = await validate_csv_content(contents)

        # Optional schema handling (lightweight)
        if use_schema and schema_template:
            from schema_manager import SchemaManager
            schema_manager = SchemaManager()
            schema = schema_manager.get_template(schema_template)
            if schema:
                metadata = schema_manager.apply_schema_to_metadata(schema)
            else:
                metadata = SingleTableMetadata()
                metadata.detect_from_dataframe(data=data)
        else:
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(data=data)

        # Optimized generation settings
        try:
            synthesizer = CTGANSynthesizer(metadata, epochs=100, verbose=False)
            synthesizer.fit(data)
        except Exception as e:
            print(f"CTGAN error: {e}")
            # Fallback to basic synthesizer
            synthesizer = CTGANSynthesizer(metadata)
            synthesizer.fit(data)

        synthetic_data = synthesizer.sample(num_rows=rows_to_generate)

        # Log the generation
        log = UsageLog(user_id=current_user.id, dataset_name=f"balanced_gen_{file.filename}")
        db.add(log)
        db.commit()

        # Return CSV
        output = io.StringIO()
        synthetic_data.to_csv(output, index=False)
        output.seek(0)

        return StreamingResponse(
            io.StringIO(output.getvalue()), 
            media_type="text/csv", 
            headers={"Content-Disposition": f"attachment; filename=balanced_synthetic_{file.filename}"}
        )
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/generate/performance-guide")
async def get_performance_guide():
    """
    Guide for choosing the right generation mode
    """
    return {
        "generation_modes": {
            "fast": {
                "endpoint": "/generate/fast",
                "time": "10-30 seconds",
                "quality": "Good",
                "features": ["Basic generation", "Minimal validation"],
                "use_case": "Quick prototyping, testing"
            },
            "balanced": {
                "endpoint": "/generate/balanced", 
                "time": "30-60 seconds",
                "quality": "Very Good",
                "features": ["Schema support", "Optimized settings"],
                "use_case": "Production use, good quality needed"
            },
            "enterprise": {
                "endpoint": "/generate",
                "time": "60-120+ seconds",
                "quality": "Excellent",
                "features": ["Full constraints", "Quality evaluation", "Privacy analysis"],
                "use_case": "Enterprise, compliance, research"
            }
        },
        "performance_tips": {
            "dataset_size": "Smaller datasets (< 1000 rows) generate faster",
            "columns": "Fewer columns = faster generation",
            "epochs": "Reduced epochs for speed vs quality tradeoff",
            "evaluation": "Skip evaluation for faster results",
            "constraints": "Simple constraints vs complex business rules"
        },
        "optimization_settings": {
            "fast_mode": {
                "epochs": 50,
                "batch_size": "min(500, dataset_size)",
                "features_disabled": ["constraints", "evaluation", "complex_validation"]
            },
            "balanced_mode": {
                "epochs": 100,
                "batch_size": "optimized",
                "features_enabled": ["basic_schema", "validation"]
            },
            "enterprise_mode": {
                "epochs": 300,
                "batch_size": "default",
                "features_enabled": ["all"]
            }
        }
    }