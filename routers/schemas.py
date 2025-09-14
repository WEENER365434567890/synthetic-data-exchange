"""
Schema Management API Endpoints
"""
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import pandas as pd
import json
import io

from schema_manager import SchemaManager, DatasetSchema
from models import SchemaTemplate
from routers.auth import get_current_user

router = APIRouter()
schema_manager = SchemaManager()

@router.get("/schemas/templates", response_model=List[SchemaTemplate])
async def list_schema_templates():
    """List all available schema templates"""
    templates = []
    for domain, schema in schema_manager.templates.items():
        template = SchemaTemplate(
            name=schema.name,
            domain=schema.domain,
            description=schema.description or f"Standard schema for {domain} domain",
            columns=[col.dict() for col in schema.columns]
        )
        templates.append(template)
    
    return templates

@router.get("/schemas/templates/{domain}")
async def get_schema_template(domain: str):
    """Get a specific schema template by domain"""
    template = schema_manager.get_template(domain)
    if not template:
        raise HTTPException(status_code=404, detail=f"Schema template for domain '{domain}' not found")
    
    return template.dict()

@router.post("/schemas/infer")
async def infer_schema_from_data(
    file: UploadFile = File(...),
    current_user = Depends(get_current_user)
):
    """Infer schema from uploaded CSV data"""
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        # Read CSV
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file contains no data")
        
        # Infer schema
        inferred_schema = schema_manager.infer_schema(df)
        
        return {
            "schema": inferred_schema.dict(),
            "data_preview": df.head().to_dict('records'),
            "data_info": {
                "rows": len(df),
                "columns": len(df.columns),
                "column_types": df.dtypes.to_dict()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error inferring schema: {str(e)}")

@router.post("/schemas/validate")
async def validate_data_against_schema(
    file: UploadFile = File(...),
    schema_json: str = None,
    template_domain: str = None,
    current_user = Depends(get_current_user)
):
    """Validate uploaded data against a schema"""
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        # Read CSV
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file contains no data")
        
        # Get schema
        if template_domain:
            schema = schema_manager.get_template(template_domain)
            if not schema:
                raise HTTPException(status_code=404, detail=f"Template '{template_domain}' not found")
        elif schema_json:
            schema_dict = json.loads(schema_json)
            schema = DatasetSchema(**schema_dict)
        else:
            raise HTTPException(status_code=400, detail="Either template_domain or schema_json must be provided")
        
        # Validate
        validation_results = schema_manager.validate_data_against_schema(df, schema)
        
        return {
            "valid": len(validation_results["column_errors"]) == 0 and len(validation_results["constraint_errors"]) == 0,
            "validation_results": validation_results,
            "schema_used": schema.dict(),
            "data_summary": {
                "rows": len(df),
                "columns": len(df.columns)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error validating data: {str(e)}")

@router.post("/schemas/create")
async def create_custom_schema(
    schema_data: Dict[str, Any],
    current_user = Depends(get_current_user)
):
    """Create a custom schema"""
    try:
        schema = DatasetSchema(**schema_data)
        
        # Save schema (in a real implementation, you'd save to database)
        schema_id = f"custom_{current_user.id}_{schema.name.lower().replace(' ', '_')}"
        
        return {
            "schema_id": schema_id,
            "schema": schema.dict(),
            "message": "Custom schema created successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating schema: {str(e)}")

@router.get("/schemas/domains")
async def list_supported_domains():
    """List all supported domains with their descriptions"""
    domains = {
        "mining": {
            "name": "Mining Operations",
            "description": "Mining production data including tonnage, energy usage, ore grades",
            "typical_columns": ["mine_id", "date", "tonnage", "energy_used", "ore_grade", "equipment_hours"],
            "constraints": ["Energy proportional to tonnage", "Positive values for production metrics"]
        },
        "healthcare": {
            "name": "Healthcare Records",
            "description": "Patient health data with medical measurements and diagnoses",
            "typical_columns": ["patient_id", "age", "gender", "blood_pressure", "diagnosis"],
            "constraints": ["Unique patient IDs", "Systolic > Diastolic BP", "Age consistency"]
        },
        "energy": {
            "name": "Energy Grid Operations",
            "description": "Power generation and grid monitoring data",
            "typical_columns": ["station_id", "timestamp", "power_output", "max_capacity", "efficiency"],
            "constraints": ["Output ≤ Capacity", "Efficiency within realistic bounds"]
        }
    }
    
    return domains

@router.get("/schemas/examples/{domain}")
async def get_domain_example_data(domain: str):
    """Get example data for a specific domain"""
    examples = {
        "mining": {
            "sample_data": [
                {"mine_id": "MINE001", "date": "2023-01-15", "tonnage": 1250.5, "energy_used": 3400.2, "ore_grade": 2.1},
                {"mine_id": "MINE001", "date": "2023-01-16", "tonnage": 1180.3, "energy_used": 3200.8, "ore_grade": 2.3},
                {"mine_id": "MINE002", "date": "2023-01-15", "tonnage": 980.2, "energy_used": 2800.5, "ore_grade": 2.8}
            ],
            "description": "Sample mining production data showing daily operations"
        },
        "healthcare": {
            "sample_data": [
                {"patient_id": "P001", "age": 45, "gender": "Male", "blood_pressure_systolic": 120, "blood_pressure_diastolic": 80, "diagnosis": "Healthy"},
                {"patient_id": "P002", "age": 62, "gender": "Female", "blood_pressure_systolic": 140, "blood_pressure_diastolic": 90, "diagnosis": "Hypertension"},
                {"patient_id": "P003", "age": 38, "gender": "Male", "blood_pressure_systolic": 110, "blood_pressure_diastolic": 70, "diagnosis": "Healthy"}
            ],
            "description": "Sample patient health records with basic vital signs"
        },
        "energy": {
            "sample_data": [
                {"station_id": "STATION_A", "timestamp": "2023-01-15 12:00:00", "power_output": 850, "max_capacity": 1000, "efficiency": 85},
                {"station_id": "STATION_A", "timestamp": "2023-01-15 13:00:00", "power_output": 920, "max_capacity": 1000, "efficiency": 87},
                {"station_id": "STATION_B", "timestamp": "2023-01-15 12:00:00", "power_output": 650, "max_capacity": 800, "efficiency": 82}
            ],
            "description": "Sample energy grid monitoring data with power generation metrics"
        }
    }
    
    if domain not in examples:
        raise HTTPException(status_code=404, detail=f"No example data available for domain '{domain}'")
    
    return examples[domain]