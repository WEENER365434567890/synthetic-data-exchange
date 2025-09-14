"""
Schema and Metadata Management for Synthetic Data Generation
"""
import json
import pandas as pd
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, validator
from enum import Enum
from datetime import datetime
import numpy as np

class ColumnType(str, Enum):
    CATEGORICAL = "categorical"
    CONTINUOUS = "continuous"
    DATETIME = "datetime"
    GEOSPATIAL = "geospatial"
    ID = "id"
    BOOLEAN = "boolean"

class ConstraintType(str, Enum):
    RANGE = "range"
    POSITIVE = "positive"
    UNIQUE = "unique"
    INCREASING = "increasing"
    PROPORTIONAL = "proportional"
    CUSTOM = "custom"

class ColumnSchema(BaseModel):
    name: str
    type: ColumnType
    description: Optional[str] = None
    nullable: bool = True
    constraints: List[Dict[str, Any]] = []
    
    # Type-specific properties
    categories: Optional[List[str]] = None  # For categorical
    min_value: Optional[float] = None       # For continuous
    max_value: Optional[float] = None       # For continuous
    date_format: Optional[str] = None       # For datetime
    coordinate_system: Optional[str] = None # For geospatial

class DatasetSchema(BaseModel):
    name: str
    description: Optional[str] = None
    domain: Optional[str] = None  # mining, healthcare, energy, etc.
    columns: List[ColumnSchema]
    global_constraints: List[Dict[str, Any]] = []
    
    @validator('columns')
    def validate_columns(cls, v):
        if len(v) == 0:
            raise ValueError("Schema must have at least one column")
        return v

class SchemaManager:
    """Manages dataset schemas and applies them to synthetic data generation"""
    
    def __init__(self):
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, DatasetSchema]:
        """Load predefined schema templates for different domains"""
        templates = {}
        
        # Mining Operations Template
        mining_schema = DatasetSchema(
            name="Mining Operations",
            description="Standard schema for mining production data",
            domain="mining",
            columns=[
                ColumnSchema(
                    name="mine_id",
                    type=ColumnType.CATEGORICAL,
                    description="Unique identifier for mine location",
                    nullable=False,
                    categories=["MINE001", "MINE002", "MINE003", "MINE004"]
                ),
                ColumnSchema(
                    name="date",
                    type=ColumnType.DATETIME,
                    description="Date of operation",
                    nullable=False,
                    date_format="%Y-%m-%d",
                    constraints=[{"type": "increasing", "description": "Dates should be in chronological order"}]
                ),
                ColumnSchema(
                    name="tonnage",
                    type=ColumnType.CONTINUOUS,
                    description="Amount of material processed (tons)",
                    nullable=False,
                    min_value=0,
                    max_value=2000,
                    constraints=[{"type": "positive", "description": "Tonnage must be positive"}]
                ),
                ColumnSchema(
                    name="energy_used",
                    type=ColumnType.CONTINUOUS,
                    description="Energy consumption (kWh)",
                    nullable=False,
                    min_value=0,
                    max_value=5000,
                    constraints=[
                        {"type": "positive", "description": "Energy must be positive"},
                        {"type": "proportional", "target": "tonnage", "ratio_range": [2.5, 3.5], 
                         "description": "Energy should be proportional to tonnage"}
                    ]
                ),
                ColumnSchema(
                    name="ore_grade",
                    type=ColumnType.CONTINUOUS,
                    description="Quality of ore extracted (%)",
                    nullable=False,
                    min_value=0.5,
                    max_value=5.0
                ),
                ColumnSchema(
                    name="equipment_hours",
                    type=ColumnType.CONTINUOUS,
                    description="Equipment operation time (hours)",
                    nullable=False,
                    min_value=8,
                    max_value=24
                ),
                ColumnSchema(
                    name="maintenance_cost",
                    type=ColumnType.CONTINUOUS,
                    description="Daily maintenance expenses ($)",
                    nullable=False,
                    min_value=500,
                    max_value=2000
                )
            ],
            global_constraints=[
                {
                    "type": "business_rule",
                    "rule": "higher_tonnage_higher_energy",
                    "description": "Higher tonnage should correlate with higher energy usage"
                }
            ]
        )
        templates["mining"] = mining_schema
        
        # Healthcare Template
        healthcare_schema = DatasetSchema(
            name="Patient Health Records",
            description="Standard schema for patient health data",
            domain="healthcare",
            columns=[
                ColumnSchema(
                    name="patient_id",
                    type=ColumnType.ID,
                    description="Unique patient identifier",
                    nullable=False,
                    constraints=[{"type": "unique", "description": "Patient IDs must be unique"}]
                ),
                ColumnSchema(
                    name="age",
                    type=ColumnType.CONTINUOUS,
                    description="Patient age in years",
                    nullable=False,
                    min_value=0,
                    max_value=120
                ),
                ColumnSchema(
                    name="gender",
                    type=ColumnType.CATEGORICAL,
                    description="Patient gender",
                    nullable=False,
                    categories=["Male", "Female", "Other"]
                ),
                ColumnSchema(
                    name="blood_pressure_systolic",
                    type=ColumnType.CONTINUOUS,
                    description="Systolic blood pressure (mmHg)",
                    nullable=True,
                    min_value=80,
                    max_value=200
                ),
                ColumnSchema(
                    name="blood_pressure_diastolic",
                    type=ColumnType.CONTINUOUS,
                    description="Diastolic blood pressure (mmHg)",
                    nullable=True,
                    min_value=50,
                    max_value=120
                ),
                ColumnSchema(
                    name="diagnosis",
                    type=ColumnType.CATEGORICAL,
                    description="Primary diagnosis",
                    nullable=True,
                    categories=["Healthy", "Hypertension", "Diabetes", "Heart Disease", "Other"]
                )
            ],
            global_constraints=[
                {
                    "type": "medical_rule",
                    "rule": "blood_pressure_relationship",
                    "description": "Systolic BP should be higher than diastolic BP"
                }
            ]
        )
        templates["healthcare"] = healthcare_schema
        
        # Energy Grid Template
        energy_schema = DatasetSchema(
            name="Energy Grid Operations",
            description="Standard schema for energy grid monitoring",
            domain="energy",
            columns=[
                ColumnSchema(
                    name="station_id",
                    type=ColumnType.CATEGORICAL,
                    description="Power station identifier",
                    nullable=False,
                    categories=["STATION_A", "STATION_B", "STATION_C"]
                ),
                ColumnSchema(
                    name="timestamp",
                    type=ColumnType.DATETIME,
                    description="Measurement timestamp",
                    nullable=False,
                    date_format="%Y-%m-%d %H:%M:%S"
                ),
                ColumnSchema(
                    name="power_output",
                    type=ColumnType.CONTINUOUS,
                    description="Power generation (MW)",
                    nullable=False,
                    min_value=0,
                    max_value=1000
                ),
                ColumnSchema(
                    name="max_capacity",
                    type=ColumnType.CONTINUOUS,
                    description="Maximum generation capacity (MW)",
                    nullable=False,
                    min_value=500,
                    max_value=1000
                ),
                ColumnSchema(
                    name="efficiency",
                    type=ColumnType.CONTINUOUS,
                    description="Generation efficiency (%)",
                    nullable=False,
                    min_value=70,
                    max_value=95
                )
            ],
            global_constraints=[
                {
                    "type": "capacity_constraint",
                    "rule": "output_within_capacity",
                    "description": "Power output must not exceed max capacity"
                }
            ]
        )
        templates["energy"] = energy_schema
        
        return templates
    
    def get_template(self, domain: str) -> Optional[DatasetSchema]:
        """Get a predefined schema template for a domain"""
        return self.templates.get(domain)
    
    def list_templates(self) -> List[str]:
        """List available schema templates"""
        return list(self.templates.keys())
    
    def infer_schema(self, df: pd.DataFrame) -> DatasetSchema:
        """Automatically infer schema from a pandas DataFrame"""
        columns = []
        
        for col_name in df.columns:
            col_data = df[col_name]
            
            # Infer column type
            if col_data.dtype == 'object':
                if col_data.nunique() / len(col_data) < 0.1:  # Low cardinality
                    col_type = ColumnType.CATEGORICAL
                    categories = col_data.unique().tolist()
                else:
                    col_type = ColumnType.ID if 'id' in col_name.lower() else ColumnType.CATEGORICAL
                    categories = None
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                col_type = ColumnType.DATETIME
                categories = None
            elif pd.api.types.is_bool_dtype(col_data):
                col_type = ColumnType.BOOLEAN
                categories = None
            else:
                col_type = ColumnType.CONTINUOUS
                categories = None
            
            # Create column schema
            column_schema = ColumnSchema(
                name=col_name,
                type=col_type,
                nullable=col_data.isnull().any(),
                categories=categories
            )
            
            # Add range constraints for continuous variables
            if col_type == ColumnType.CONTINUOUS:
                column_schema.min_value = float(col_data.min())
                column_schema.max_value = float(col_data.max())
            
            columns.append(column_schema)
        
        return DatasetSchema(
            name="Inferred Schema",
            description=f"Auto-inferred schema from dataset with {len(df)} rows and {len(df.columns)} columns",
            columns=columns
        )
    
    def validate_data_against_schema(self, df: pd.DataFrame, schema: DatasetSchema) -> Dict[str, List[str]]:
        """Validate a DataFrame against a schema and return validation errors"""
        errors = {"column_errors": [], "constraint_errors": []}
        
        # Check if all schema columns exist
        schema_cols = {col.name for col in schema.columns}
        data_cols = set(df.columns)
        
        missing_cols = schema_cols - data_cols
        extra_cols = data_cols - schema_cols
        
        if missing_cols:
            errors["column_errors"].append(f"Missing columns: {missing_cols}")
        if extra_cols:
            errors["column_errors"].append(f"Extra columns: {extra_cols}")
        
        # Validate each column
        for col_schema in schema.columns:
            if col_schema.name not in df.columns:
                continue
                
            col_data = df[col_schema.name]
            
            # Check nullability
            if not col_schema.nullable and col_data.isnull().any():
                errors["constraint_errors"].append(f"{col_schema.name}: Contains null values but schema requires non-null")
            
            # Check type-specific constraints
            if col_schema.type == ColumnType.CONTINUOUS:
                if col_schema.min_value is not None and col_data.min() < col_schema.min_value:
                    errors["constraint_errors"].append(f"{col_schema.name}: Values below minimum {col_schema.min_value}")
                if col_schema.max_value is not None and col_data.max() > col_schema.max_value:
                    errors["constraint_errors"].append(f"{col_schema.name}: Values above maximum {col_schema.max_value}")
            
            elif col_schema.type == ColumnType.CATEGORICAL and col_schema.categories:
                invalid_values = set(col_data.dropna().unique()) - set(col_schema.categories)
                if invalid_values:
                    errors["constraint_errors"].append(f"{col_schema.name}: Invalid categories {invalid_values}")
        
        return errors
    
    def apply_schema_to_metadata(self, schema: DatasetSchema) -> Dict[str, Any]:
        """Convert schema to SDV metadata format"""
        from sdv.metadata import SingleTableMetadata
        
        metadata = SingleTableMetadata()
        
        # Build column specifications
        for col in schema.columns:
            if col.type == ColumnType.CATEGORICAL:
                metadata.add_column(col.name, sdtype='categorical')
            elif col.type == ColumnType.CONTINUOUS:
                metadata.add_column(col.name, sdtype='numerical')
            elif col.type == ColumnType.DATETIME:
                metadata.add_column(col.name, sdtype='datetime', datetime_format=col.date_format or '%Y-%m-%d')
            elif col.type == ColumnType.BOOLEAN:
                metadata.add_column(col.name, sdtype='boolean')
            elif col.type == ColumnType.ID:
                metadata.add_column(col.name, sdtype='id')
        
        return metadata
    
    def save_schema(self, schema: DatasetSchema, filepath: str):
        """Save schema to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(schema.dict(), f, indent=2, default=str)
    
    def load_schema(self, filepath: str) -> DatasetSchema:
        """Load schema from JSON file"""
        with open(filepath, 'r') as f:
            schema_dict = json.load(f)
        return DatasetSchema(**schema_dict)