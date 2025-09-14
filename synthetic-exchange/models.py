from pydantic import BaseModel, validator
from typing import List, Optional, Dict, Any
from datetime import datetime

class GenerateRequest(BaseModel):
    num_rows: int = 100
    use_schema: bool = False
    schema_template: Optional[str] = None  # mining, healthcare, energy
    custom_schema: Optional[Dict[str, Any]] = None
    apply_constraints: bool = True
    evaluate_quality: bool = False
    target_column: Optional[str] = None  # For ML evaluation
    
    @validator('num_rows')
    def validate_num_rows(cls, v):
        if v <= 0:
            raise ValueError('Number of rows must be positive')
        if v > 10000:
            raise ValueError('Number of rows cannot exceed 10,000')
        return v

class Dataset(BaseModel):
    name: str
    description: str
    size: Optional[int] = None
    created_at: Optional[datetime] = None
    uploaded_by: Optional[str] = None

class DatasetUpload(BaseModel):
    description: str = ""
    
class UserCreate(BaseModel):
    email: str
    password: str
    
    @validator('email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 6:
            raise ValueError('Password must be at least 6 characters long')
        return v

class Token(BaseModel):
    access_token: str
    token_type: str

class EvaluationRequest(BaseModel):
    target_column: Optional[str] = None
    include_privacy_analysis: bool = True
    include_utility_analysis: bool = True
    include_statistical_analysis: bool = True

class EvaluationResponse(BaseModel):
    overall_quality_score: float
    grade: str
    statistical_similarity_score: float
    utility_preservation_score: float
    privacy_protection_score: float
    detailed_report: Dict[str, Any]
    recommendations: List[str] = []

class SchemaTemplate(BaseModel):
    name: str
    domain: str
    description: str
    columns: List[Dict[str, Any]]
    sample_data_url: Optional[str] = None
