from fastapi import File, UploadFile, HTTPException
import pandas as pd
import io

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {'.csv'}

def check_file_size(file: UploadFile = File(...)):
    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File size exceeds the limit of 10MB")
    return file

def validate_csv_file(file: UploadFile = File(...)):
    # Check file extension
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    # Check file size
    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File size exceeds the limit of 10MB")
    
    return file

async def validate_csv_content(contents: bytes):
    """Validate CSV content and return DataFrame"""
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    
    try:
        # Try to decode and parse CSV
        csv_string = contents.decode('utf-8')
        data = pd.read_csv(io.StringIO(csv_string))
        
        if data.empty:
            raise HTTPException(status_code=400, detail="CSV file contains no data")
        
        if len(data.columns) < 2:
            raise HTTPException(status_code=400, detail="CSV must have at least 2 columns")
            
        if len(data) < 5:
            raise HTTPException(status_code=400, detail="CSV must have at least 5 rows for meaningful synthetic data generation")
            
        return data
        
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File encoding not supported. Please use UTF-8 encoded CSV")
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty")
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"Error parsing CSV file: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")
