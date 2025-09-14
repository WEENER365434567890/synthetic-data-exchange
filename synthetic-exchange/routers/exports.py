"""
Export API Endpoints for Multiple Formats
"""
from fastapi import APIRouter, HTTPException, Depends, Query, Form, UploadFile, File
from fastapi.responses import StreamingResponse
from typing import Optional
import pandas as pd
import io
from datetime import datetime

from export_manager import ExportManager
from evaluation_manager import QualityEvaluator
from routers.auth import get_current_user
from dependencies import validate_csv_content

router = APIRouter()
export_manager = ExportManager()
quality_evaluator = QualityEvaluator()

@router.get("/export/formats")
async def get_supported_formats():
    """Get list of supported export formats"""
    return {
        "data_formats": ["csv", "xlsx", "sql", "json"],
        "report_formats": ["pdf", "json"],
        "description": {
            "csv": "Comma-separated values (universal compatibility)",
            "xlsx": "Excel format with multiple sheets and metadata",
            "sql": "SQL INSERT statements for database import",
            "json": "JSON format with metadata",
            "pdf": "Professional PDF reports (evaluation only)"
        }
    }

@router.post("/export/data/{format}")
async def export_synthetic_data(
    format: str,
    file: UploadFile = File(..., description="CSV file containing synthetic data"),
    table_name: Optional[str] = Form("synthetic_data", description="Table name for SQL export"),
    current_user = Depends(get_current_user)
):
    """
    Export synthetic data in specified format
    Supports: csv, xlsx, sql, json
    """
    try:
        # Validate format
        if format.lower() not in ["csv", "xlsx", "sql", "json"]:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
        
        # Read and validate CSV data
        contents = await file.read()
        data = await validate_csv_content(contents)
        
        # Export in requested format
        if format.lower() == "sql":
            exported_data = export_manager.export_data(data, format, table_name=table_name)
        else:
            exported_data = export_manager.export_data(data, format)
        
        # Set appropriate content type and filename
        content_types = {
            "csv": "text/csv",
            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "sql": "text/sql",
            "json": "application/json"
        }
        
        file_extensions = {
            "csv": "csv",
            "xlsx": "xlsx", 
            "sql": "sql",
            "json": "json"
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"synthetic_data_{timestamp}.{file_extensions[format.lower()]}"
        
        return StreamingResponse(
            io.BytesIO(exported_data),
            media_type=content_types[format.lower()],
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@router.post("/export/evaluation-report")
async def export_evaluation_report(
    real_data_file: UploadFile = File(..., description="Original real dataset"),
    synthetic_data_file: UploadFile = File(..., description="Generated synthetic dataset"),
    format: str = Form("pdf", description="Export format: pdf or json"),
    target_column: Optional[str] = Form(None, description="Target column for ML evaluation"),
    current_user = Depends(get_current_user)
):
    """
    Generate and export comprehensive evaluation report
    Supports: pdf (recommended), json
    """
    try:
        # Validate format
        if format.lower() not in ["pdf", "json"]:
            raise HTTPException(status_code=400, detail=f"Unsupported report format: {format}")
        
        # Read datasets
        real_contents = await real_data_file.read()
        synthetic_contents = await synthetic_data_file.read()
        
        real_data = await validate_csv_content(real_contents)
        synthetic_data = await validate_csv_content(synthetic_contents)
        
        # Generate comprehensive evaluation
        evaluation_report = quality_evaluator.evaluate_synthetic_data(
            real_data=real_data,
            synthetic_data=synthetic_data,
            target_column=target_column
        )
        
        # Export report in requested format
        exported_report = export_manager.export_evaluation_report(evaluation_report, format)
        
        # Set appropriate response
        content_types = {
            "pdf": "application/pdf",
            "json": "application/json"
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_report_{timestamp}.{format.lower()}"
        
        return StreamingResponse(
            io.BytesIO(exported_report),
            media_type=content_types[format.lower()],
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Report export failed: {str(e)}")

@router.post("/export/quick-excel")
async def quick_excel_export(
    file: UploadFile = File(..., description="CSV file to convert to Excel"),
    current_user = Depends(get_current_user)
):
    """
    Quick conversion from CSV to Excel with enhanced formatting
    """
    try:
        contents = await file.read()
        data = await validate_csv_content(contents)
        
        # Export as Excel
        excel_data = export_manager.export_data(data, "xlsx")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"synthetic_data_{timestamp}.xlsx"
        
        return StreamingResponse(
            io.BytesIO(excel_data),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Excel export failed: {str(e)}")

@router.post("/export/sql-dump")
async def sql_dump_export(
    file: UploadFile = File(..., description="CSV file to convert to SQL"),
    table_name: str = Form("synthetic_data", description="Database table name"),
    current_user = Depends(get_current_user)
):
    """
    Convert CSV data to SQL INSERT statements for database import
    """
    try:
        contents = await file.read()
        data = await validate_csv_content(contents)
        
        # Export as SQL
        sql_data = export_manager.export_data(data, "sql", table_name=table_name)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{table_name}_{timestamp}.sql"
        
        return StreamingResponse(
            io.BytesIO(sql_data),
            media_type="text/sql",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"SQL export failed: {str(e)}")

@router.get("/export/examples")
async def get_export_examples():
    """
    Get examples of how to use different export formats
    """
    return {
        "excel_use_cases": [
            "Business users who need to analyze data in Excel",
            "Creating charts and pivot tables from synthetic data",
            "Sharing data with non-technical stakeholders"
        ],
        "sql_use_cases": [
            "Importing synthetic data into test databases",
            "Populating development environments",
            "Creating database fixtures for testing"
        ],
        "pdf_reports_use_cases": [
            "Sharing quality assessments with management",
            "Compliance documentation for audits",
            "Technical reports for data science teams"
        ],
        "api_examples": {
            "excel_export": "curl -X POST '/export/data/xlsx' -F 'file=@data.csv'",
            "sql_export": "curl -X POST '/export/data/sql' -F 'file=@data.csv' -F 'table_name=my_table'",
            "pdf_report": "curl -X POST '/export/evaluation-report' -F 'real_data_file=@real.csv' -F 'synthetic_data_file=@synthetic.csv' -F 'format=pdf'"
        }
    }