"""
Advanced Data Types API - High-Value Synthetic Data Generation
Focus: Time-Series, Geospatial, Healthcare - What Companies Pay For
"""
from fastapi import APIRouter, HTTPException, Depends, Form, Query
from fastapi.responses import StreamingResponse
from typing import Optional, List
import pandas as pd
import io
from datetime import datetime, timedelta

from advanced_data_types import AdvancedDataTypeGenerator, DataTypeConfig
from export_manager import ExportManager
from routers.auth import get_current_user
from sqlalchemy.orm import Session
from database import SessionLocal, UsageLog

router = APIRouter()
generator = AdvancedDataTypeGenerator()
export_manager = ExportManager()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/advanced-types/supported")
async def get_supported_types():
    """Get all supported advanced data types and industry verticals"""
    return {
        "data_types": generator.get_supported_types(),
        "industry_verticals": generator.get_industry_verticals(),
        "high_value_types": {
            "timeseries": {
                "description": "Time-based data for IoT, energy, and monitoring",
                "industries": ["energy", "mining", "healthcare", "manufacturing"],
                "commercial_value": "Very High - Real-time analytics, predictive maintenance"
            },
            "geospatial": {
                "description": "Location-based data for mapping and logistics",
                "industries": ["mining", "energy", "logistics", "retail"],
                "commercial_value": "High - Asset tracking, site planning, route optimization"
            },
            "healthcare": {
                "description": "Medical records and patient monitoring data",
                "industries": ["clinical", "research", "pharmaceutical"],
                "commercial_value": "Extremely High - HIPAA-compliant research, AI training"
            }
        }
    }

@router.post("/advanced-types/timeseries/{industry}")
async def generate_timeseries_data(
    industry: str,
    num_rows: int = Form(1000, description="Number of time points to generate"),
    days_back: int = Form(30, description="How many days back to start from"),
    frequency: str = Form("H", description="Time frequency: H=hourly, D=daily, 15T=15min"),
    export_format: str = Form("csv", description="Export format: csv, xlsx, sql"),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Generate high-value time-series data for specific industries
    
    Industries:
    - energy: Power grid monitoring, consumption patterns
    - mining: Equipment sensors, production metrics
    - healthcare: Patient monitoring, vital signs
    """
    try:
        if industry not in ["energy", "mining", "healthcare"]:
            raise HTTPException(status_code=400, detail=f"Unsupported industry: {industry}")
        
        # Configure generation
        config = DataTypeConfig(
            data_type="timeseries",
            num_rows=num_rows,
            start_date=datetime.now() - timedelta(days=days_back),
            frequency=frequency,
            industry_vertical=industry
        )
        
        # Generate data
        data = generator.generate_timeseries_data(config)
        
        # Export in requested format
        exported_data = export_manager.export_data(data, export_format)
        
        # Log usage
        log = UsageLog(user_id=current_user.id, dataset_name=f"timeseries_{industry}_{num_rows}rows")
        db.add(log)
        db.commit()
        
        # Return file
        content_types = {
            "csv": "text/csv",
            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "sql": "text/sql"
        }
        
        extensions = {"csv": "csv", "xlsx": "xlsx", "sql": "sql"}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"timeseries_{industry}_{timestamp}.{extensions[export_format]}"
        
        return StreamingResponse(
            io.BytesIO(exported_data),
            media_type=content_types[export_format],
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@router.post("/advanced-types/geospatial/{industry}")
async def generate_geospatial_data(
    industry: str,
    num_locations: int = Form(500, description="Number of locations to generate"),
    min_lat: float = Form(-26.0, description="Minimum latitude"),
    max_lat: float = Form(-23.0, description="Maximum latitude"),
    min_lon: float = Form(115.0, description="Minimum longitude"),
    max_lon: float = Form(120.0, description="Maximum longitude"),
    export_format: str = Form("csv", description="Export format: csv, xlsx, sql"),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Generate high-value geospatial data for specific industries
    
    Industries:
    - mining: Mine site locations, ore deposits, infrastructure
    - energy: Power plants, grid infrastructure, renewable sites
    - logistics: Distribution centers, routes, service areas
    
    Default coordinates: Pilbara mining region, Australia
    """
    try:
        if industry not in ["mining", "energy", "logistics"]:
            raise HTTPException(status_code=400, detail=f"Unsupported industry: {industry}")
        
        # Configure generation
        config = DataTypeConfig(
            data_type="geospatial",
            num_rows=num_locations,
            location_bounds=(min_lat, max_lat, min_lon, max_lon),
            industry_vertical=industry
        )
        
        # Generate data
        data = generator.generate_geospatial_data(config)
        
        # Export in requested format
        exported_data = export_manager.export_data(data, export_format)
        
        # Log usage
        log = UsageLog(user_id=current_user.id, dataset_name=f"geospatial_{industry}_{num_locations}locations")
        db.add(log)
        db.commit()
        
        # Return file
        content_types = {
            "csv": "text/csv",
            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "sql": "text/sql"
        }
        
        extensions = {"csv": "csv", "xlsx": "xlsx", "sql": "sql"}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"geospatial_{industry}_{timestamp}.{extensions[export_format]}"
        
        return StreamingResponse(
            io.BytesIO(exported_data),
            media_type=content_types[export_format],
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@router.post("/advanced-types/healthcare")
async def generate_healthcare_data(
    num_patients: int = Form(1000, description="Number of patient records to generate"),
    include_timeseries: bool = Form(False, description="Include patient monitoring time-series"),
    monitoring_hours: int = Form(24, description="Hours of monitoring data per patient"),
    export_format: str = Form("csv", description="Export format: csv, xlsx, sql"),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Generate HIPAA-compliant synthetic healthcare data
    
    Extremely high commercial value for:
    - Clinical research without patient privacy risks
    - Healthcare AI/ML model training
    - EHR system testing and development
    - Pharmaceutical research datasets
    """
    try:
        # Generate patient demographics and health records
        config = DataTypeConfig(
            data_type="healthcare",
            num_rows=num_patients,
            industry_vertical="healthcare"
        )
        
        patient_data = generator.generate_healthcare_data(config)
        
        if include_timeseries:
            # Generate monitoring data for each patient
            monitoring_config = DataTypeConfig(
                data_type="timeseries",
                num_rows=monitoring_hours * 12,  # 5-minute intervals
                start_date=datetime.now() - timedelta(hours=monitoring_hours),
                frequency="5T",
                industry_vertical="healthcare"
            )
            
            monitoring_data = generator.generate_timeseries_data(monitoring_config)
            
            # Export both datasets
            patient_export = export_manager.export_data(patient_data, export_format)
            monitoring_export = export_manager.export_data(monitoring_data, export_format)
            
            # For now, return patient data (could be enhanced to return both)
            exported_data = patient_export
            dataset_type = "healthcare_with_monitoring"
        else:
            exported_data = export_manager.export_data(patient_data, export_format)
            dataset_type = "healthcare_demographics"
        
        # Log usage
        log = UsageLog(user_id=current_user.id, dataset_name=f"{dataset_type}_{num_patients}patients")
        db.add(log)
        db.commit()
        
        # Return file
        content_types = {
            "csv": "text/csv",
            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "sql": "text/sql"
        }
        
        extensions = {"csv": "csv", "xlsx": "xlsx", "sql": "sql"}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"healthcare_data_{timestamp}.{extensions[export_format]}"
        
        return StreamingResponse(
            io.BytesIO(exported_data),
            media_type=content_types[export_format],
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Healthcare data generation failed: {str(e)}")

@router.get("/advanced-types/examples")
async def get_generation_examples():
    """Get examples of advanced data type generation"""
    return {
        "timeseries_examples": {
            "energy_grid": {
                "description": "Power grid monitoring with demand patterns",
                "sample_columns": ["timestamp", "station_id", "power_demand_mw", "frequency_hz", "voltage_kv"],
                "use_cases": ["Grid optimization", "Demand forecasting", "Outage prediction"],
                "api_call": "POST /advanced-types/timeseries/energy"
            },
            "mining_sensors": {
                "description": "Mining equipment sensor data",
                "sample_columns": ["timestamp", "equipment_id", "tonnage_processed", "energy_consumption", "vibration_level"],
                "use_cases": ["Predictive maintenance", "Production optimization", "Safety monitoring"],
                "api_call": "POST /advanced-types/timeseries/mining"
            },
            "patient_monitoring": {
                "description": "Continuous patient vital signs",
                "sample_columns": ["timestamp", "patient_id", "heart_rate_bpm", "blood_pressure", "oxygen_saturation"],
                "use_cases": ["Clinical research", "Algorithm training", "System testing"],
                "api_call": "POST /advanced-types/timeseries/healthcare"
            }
        },
        "geospatial_examples": {
            "mining_sites": {
                "description": "Mine locations with operational data",
                "sample_columns": ["site_id", "latitude", "longitude", "ore_type", "production_capacity"],
                "use_cases": ["Site planning", "Logistics optimization", "Environmental impact"],
                "api_call": "POST /advanced-types/geospatial/mining"
            },
            "energy_infrastructure": {
                "description": "Power generation facility locations",
                "sample_columns": ["facility_id", "latitude", "longitude", "facility_type", "capacity_mw"],
                "use_cases": ["Grid planning", "Renewable energy siting", "Infrastructure analysis"],
                "api_call": "POST /advanced-types/geospatial/energy"
            }
        },
        "healthcare_examples": {
            "ehr_records": {
                "description": "Synthetic electronic health records",
                "sample_columns": ["patient_id", "age", "gender", "diagnosis", "medications", "vitals"],
                "use_cases": ["Clinical research", "AI training", "System testing", "Compliance testing"],
                "api_call": "POST /advanced-types/healthcare"
            }
        }
    }

@router.get("/advanced-types/pricing-info")
async def get_pricing_information():
    """Information about commercial value of different data types"""
    return {
        "commercial_value_ranking": {
            "1_healthcare": {
                "value": "Extremely High",
                "reason": "HIPAA compliance, clinical research, AI training",
                "market_size": "$50B+ healthcare AI market",
                "customer_willingness_to_pay": "Very High"
            },
            "2_timeseries": {
                "value": "Very High", 
                "reason": "IoT analytics, predictive maintenance, real-time monitoring",
                "market_size": "$30B+ IoT analytics market",
                "customer_willingness_to_pay": "High"
            },
            "3_geospatial": {
                "value": "High",
                "reason": "Asset tracking, site planning, logistics optimization",
                "market_size": "$15B+ location analytics market", 
                "customer_willingness_to_pay": "Medium-High"
            }
        },
        "pricing_strategy": {
            "freemium": "Basic templates free, advanced types premium",
            "usage_based": "Price per 1000 rows generated",
            "enterprise": "Unlimited generation + custom schemas",
            "api_access": "Per API call pricing for developers"
        }
    }