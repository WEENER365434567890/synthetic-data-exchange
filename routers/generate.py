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
from models import GenerateRequest, EvaluationResponse
from schema_manager import SchemaManager, DatasetSchema
from constraints_manager import ConstraintsManager
from evaluation_manager import QualityEvaluator
from typing import Optional, Union
from sqlalchemy.orm import Session
from database import SessionLocal, UsageLog
from routers.auth import get_current_user

router = APIRouter()

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize managers
schema_manager = SchemaManager()
constraints_manager = ConstraintsManager()
quality_evaluator = QualityEvaluator()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/generate")
async def generate(
    file: UploadFile = Depends(validate_csv_file), 
    num_rows: Optional[int] = Form(None),
    use_schema: bool = Form(False),
    schema_template: Optional[str] = Form(None),
    custom_schema: Optional[str] = Form(None),
    apply_constraints: bool = Form(True),
    evaluate_quality: bool = Form(False),
    target_column: Optional[str] = Form(None),
    request_body: Optional[GenerateRequest] = Body(None),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Track start time for performance measurement
        start_time = datetime.now()
        
        # Determine parameters from form data or JSON body
        if request_body is not None:
            rows_to_generate = request_body.num_rows
            use_schema = request_body.use_schema
            schema_template = request_body.schema_template
            custom_schema = request_body.custom_schema
            apply_constraints = request_body.apply_constraints
            evaluate_quality = request_body.evaluate_quality
            target_column = request_body.target_column
        else:
            rows_to_generate = num_rows or 100
        
        # Validate num_rows
        if rows_to_generate <= 0:
            raise HTTPException(status_code=400, detail="Number of rows must be positive")
        if rows_to_generate > 10000:
            raise HTTPException(status_code=400, detail="Number of rows cannot exceed 10,000")

        contents = await file.read()
        
        # Validate CSV content using our new validation function
        data = await validate_csv_content(contents)

        # Save uploaded file with timestamp to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        uploaded_filename = f"{timestamp}_{file.filename}"
        uploaded_filepath = os.path.join(DATA_DIR, uploaded_filename)
        with open(uploaded_filepath, "wb") as f:
            f.write(contents)

        # Schema and constraints handling
        schema = None
        constraints = []
        
        if use_schema:
            if schema_template:
                # Use predefined template
                schema = schema_manager.get_template(schema_template)
                if not schema:
                    raise HTTPException(status_code=400, detail=f"Schema template '{schema_template}' not found")
            elif custom_schema:
                # Use custom schema
                try:
                    schema_dict = json.loads(custom_schema) if isinstance(custom_schema, str) else custom_schema
                    schema = DatasetSchema(**schema_dict)
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Invalid custom schema: {str(e)}")
            else:
                # Infer schema from data
                schema = schema_manager.infer_schema(data)
            
            # Validate data against schema
            validation_results = schema_manager.validate_data_against_schema(data, schema)
            if validation_results["column_errors"] or validation_results["constraint_errors"]:
                print(f"Schema validation warnings: {validation_results}")
            
            # Build constraints from schema
            if apply_constraints:
                constraints = constraints_manager.build_constraints_from_schema(schema)
        
        # Generate synthetic data with enhanced metadata
        if schema:
            metadata = schema_manager.apply_schema_to_metadata(schema)
        else:
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(data=data)

        # Apply constraints if any
        synthesizer = CTGANSynthesizer(metadata)
        
        if constraints and apply_constraints:
            # Get SDV-compatible constraints
            sdv_constraints = constraints_manager.get_sdv_constraints(constraints)
            if sdv_constraints:
                try:
                    # Note: Constraint application depends on SDV version
                    synthesizer.fit(data)
                except Exception as e:
                    print(f"Warning: Could not apply constraints: {e}")
                    synthesizer.fit(data)
            else:
                synthesizer.fit(data)
        else:
            synthesizer.fit(data)

        synthetic_data = synthesizer.sample(num_rows=rows_to_generate)
        
        # Apply constraint corrections if needed
        if constraints and apply_constraints:
            synthetic_data = constraints_manager.apply_constraints_to_data(synthetic_data, constraints)

        # Save synthetic data for debugging
        synthetic_filename = f"synthetic_{uploaded_filename}"
        synthetic_filepath = os.path.join(DATA_DIR, synthetic_filename)
        synthetic_data.to_csv(synthetic_filepath, index=False)
        
        # Log the generation
        log = UsageLog(user_id=current_user.id, dataset_name=synthetic_filename)
        db.add(log)
        db.commit()

        # Always return CSV for consistency, but add evaluation info to headers
        evaluation_result = None
        if evaluate_quality:
            try:
                evaluation_report = quality_evaluator.evaluate_synthetic_data(
                    real_data=data,
                    synthetic_data=synthetic_data,
                    target_column=target_column
                )
                evaluation_result = {
                    "overall_quality_score": evaluation_report.get('overall_scores', {}).get('overall_quality_score', 0),
                    "grade": evaluation_report.get('overall_scores', {}).get('grade', 'F'),
                    "summary": "Quality evaluation completed."
                }
            except Exception as e:
                evaluation_result = {"error": f"Evaluation failed: {str(e)}"}

        # Always return CSV file for consistency
        output = io.StringIO()
        synthetic_data.to_csv(output, index=False)
        output.seek(0)

        # Calculate actual generation time
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Prepare headers with metadata
        headers = {
            "Content-Disposition": f"attachment; filename=synthetic_{file.filename}",
            "X-Model-Type": "enterprise_ctgan",
            "X-Generation-Time": str(round(total_time, 2)),
            "X-Cached": "false",
            "X-Data-Type": "tabular",
            "X-Rows-Generated": str(len(synthetic_data)),
            "X-Constraints-Applied": str(len(constraints) if constraints else 0),
            "Access-Control-Expose-Headers": "X-Model-Type,X-Generation-Time,X-Cached,X-Data-Type,X-Rows-Generated,X-Constraints-Applied"
        }
        
        # Add evaluation info to headers if available
        if evaluation_result:
            headers["X-Quality-Score"] = str(evaluation_result.get("overall_quality_score", 0))
            headers["X-Quality-Grade"] = evaluation_result.get("grade", "F")

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
