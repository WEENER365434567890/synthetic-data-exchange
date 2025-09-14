from fastapi import APIRouter, File, UploadFile, Depends, HTTPException, Form, Query
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Optional
import os
import pandas as pd
import io
from datetime import datetime
from models import Dataset, DatasetUpload
from dependencies import validate_csv_file
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

@router.post("/datasets", response_model=Dataset)
async def upload_dataset(
    file: UploadFile = Depends(validate_csv_file), 
    description: str = Form(""),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    try:
        contents = await file.read()
        
        # Validate file is not empty
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        # Save file with timestamp to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(DATA_DIR, filename)
        
        with open(filepath, "wb") as f:
            f.write(contents)
        
        # Get file size
        file_size = len(contents)
        
        # Log the upload
        log = UsageLog(user_id=current_user.id, dataset_name=filename)
        db.add(log)
        db.commit()
        
        return Dataset(
            name=filename, 
            description=description or f"Dataset uploaded by {current_user.email}",
            size=file_size,
            created_at=datetime.now(),
            uploaded_by=current_user.email
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error uploading dataset: {str(e)}")

@router.get("/datasets", response_model=List[Dataset])
async def list_datasets(
    search: Optional[str] = Query(None, description="Search datasets by name"),
    limit: Optional[int] = Query(50, description="Maximum number of datasets to return")
):
    try:
        datasets = []
        for filename in os.listdir(DATA_DIR):
            if filename.endswith(".csv"):
                # Skip temporary files
                if filename.startswith("tmp_"):
                    continue
                    
                # Apply search filter
                if search and search.lower() not in filename.lower():
                    continue
                
                filepath = os.path.join(DATA_DIR, filename)
                file_size = os.path.getsize(filepath)
                file_mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                
                # Try to extract uploader from filename pattern
                uploaded_by = "Unknown"
                if "_" in filename:
                    # If it has timestamp prefix, it was uploaded via the new system
                    uploaded_by = "Registered User"
                
                datasets.append(Dataset(
                    name=filename, 
                    description=f"Dataset: {filename}",
                    size=file_size,
                    created_at=file_mtime,
                    uploaded_by=uploaded_by
                ))
        
        # Sort by creation date (newest first) and limit results
        datasets.sort(key=lambda x: x.created_at or datetime.min, reverse=True)
        return datasets[:limit]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing datasets: {str(e)}")

@router.get("/datasets/{dataset_name}/download")
async def download_dataset(dataset_name: str):
    try:
        filepath = os.path.join(DATA_DIR, dataset_name)
        
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        if not dataset_name.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files can be downloaded")
        
        # Read file and return as streaming response
        with open(filepath, "rb") as f:
            content = f.read()
        
        return StreamingResponse(
            io.BytesIO(content),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={dataset_name}"}
        )
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error downloading dataset: {str(e)}")
