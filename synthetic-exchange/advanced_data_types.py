"""
Advanced Data Type Generators for High-Value Synthetic Data
Focus: Time-Series, Geospatial, Healthcare - What Companies Pay For
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

# Try to import specialized libraries
try:
    from sdv.timeseries import PARSynthesizer
    TIMESERIES_AVAILABLE = True
except ImportError:
    TIMESERIES_AVAILABLE = False
    logging.warning("SDV timeseries not available")

try:
    import geopandas as gpd
    from shapely.geometry import Point
    GEOSPATIAL_AVAILABLE = True
except ImportError:
    GEOSPATIAL_AVAILABLE = False
    logging.warning("Geopandas not available for geospatial data")

logger = logging.getLogger(__name__)

@dataclass
class DataTypeConfig:
    """Configuration for advanced data type generation"""
    data_type: str
    num_rows: int
    start_date: Optional[datetime] = None
    frequency: Optional[str] = None  # For time-series: 'H', 'D', 'M'
    location_bounds: Optional[Tuple[float, float, float, float]] = None  # For geospatial: (min_lat, max_lat, min_lon, max_lon)
    industry_vertical: Optional[str] = None  # healthcare, mining, energy
    
class AdvancedDataTypeGenerator:
    """Generates high-value synthetic data types that companies pay for"""
    
    def __init__(self):
        self.supported_types = ['timeseries', 'geospatial', 'healthcare', 'industrial_iot', 'financial']
        
    def generate_timeseries_data(self, config: DataTypeConfig) -> pd.DataFrame:
        """
        Generate realistic time-series data for various industries
        High value for: Energy consumption, sensor data, production metrics
        """
        
        if config.industry_vertical == 'energy':
            return self._generate_energy_timeseries(config)
        elif config.industry_vertical == 'mining':
            return self._generate_mining_timeseries(config)
        elif config.industry_vertical == 'healthcare':
            return self._generate_healthcare_timeseries(config)
        else:
            return self._generate_generic_timeseries(config)
    
    def _generate_energy_timeseries(self, config: DataTypeConfig) -> pd.DataFrame:
        """Energy grid monitoring data - high commercial value"""
        
        start_date = config.start_date or datetime.now() - timedelta(days=30)
        frequency = config.frequency or 'H'  # Hourly by default
        
        # Generate time index
        time_index = pd.date_range(start=start_date, periods=config.num_rows, freq=frequency)
        
        # Base load patterns (realistic energy consumption)
        base_load = 1000 + 500 * np.sin(2 * np.pi * np.arange(config.num_rows) / 24)  # Daily cycle
        seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * np.arange(config.num_rows) / (24 * 365))  # Yearly cycle
        
        # Add realistic noise and spikes
        noise = np.random.normal(0, 50, config.num_rows)
        spikes = np.random.choice([0, 200, -150], config.num_rows, p=[0.95, 0.03, 0.02])
        
        power_demand = base_load * seasonal_factor + noise + spikes
        power_demand = np.maximum(power_demand, 100)  # Minimum load
        
        # Generate related metrics
        data = pd.DataFrame({
            'timestamp': time_index,
            'station_id': np.random.choice(['GRID_A', 'GRID_B', 'GRID_C'], config.num_rows),
            'power_demand_mw': power_demand,
            'power_generation_mw': power_demand * np.random.uniform(1.05, 1.15, config.num_rows),  # Slight oversupply
            'frequency_hz': 50.0 + np.random.normal(0, 0.1, config.num_rows),  # Grid frequency
            'voltage_kv': 400 + np.random.normal(0, 5, config.num_rows),
            'temperature_c': 20 + 15 * np.sin(2 * np.pi * np.arange(config.num_rows) / (24 * 365)) + np.random.normal(0, 3, config.num_rows),
            'efficiency_percent': np.random.uniform(85, 95, config.num_rows),
            'carbon_intensity_g_kwh': np.random.uniform(200, 400, config.num_rows)
        })
        
        return data
    
    def _generate_mining_timeseries(self, config: DataTypeConfig) -> pd.DataFrame:
        """Mining operations sensor data - high commercial value"""
        
        start_date = config.start_date or datetime.now() - timedelta(days=7)
        frequency = config.frequency or '15T'  # 15-minute intervals
        
        time_index = pd.date_range(start=start_date, periods=config.num_rows, freq=frequency)
        
        # Realistic mining production patterns
        shift_pattern = np.tile([1, 1, 1, 0.3, 0.3, 0.3, 1, 1], config.num_rows // 8 + 1)[:config.num_rows]  # 3 shifts
        base_production = 100 * shift_pattern
        
        # Equipment efficiency degradation over time
        equipment_age = np.linspace(0, 1, config.num_rows)
        efficiency_factor = 1 - 0.1 * equipment_age + 0.05 * np.random.random(config.num_rows)
        
        tonnage = base_production * efficiency_factor + np.random.normal(0, 10, config.num_rows)
        tonnage = np.maximum(tonnage, 0)
        
        data = pd.DataFrame({
            'timestamp': time_index,
            'mine_id': np.random.choice(['MINE_NORTH', 'MINE_SOUTH', 'MINE_EAST'], config.num_rows),
            'equipment_id': np.random.choice(['TRUCK_01', 'TRUCK_02', 'EXCAVATOR_01', 'DRILL_01'], config.num_rows),
            'tonnage_processed': tonnage,
            'energy_consumption_kwh': tonnage * np.random.uniform(2.5, 3.5, config.num_rows),  # Energy proportional to tonnage
            'ore_grade_percent': np.random.uniform(1.5, 4.0, config.num_rows),
            'equipment_temperature_c': np.random.uniform(60, 90, config.num_rows),
            'vibration_level': np.random.uniform(0.1, 2.0, config.num_rows),
            'maintenance_alert': np.random.choice([0, 1], config.num_rows, p=[0.95, 0.05]),
            'safety_incidents': np.random.choice([0, 1], config.num_rows, p=[0.999, 0.001]),
            'weather_condition': np.random.choice(['clear', 'rain', 'wind', 'storm'], config.num_rows, p=[0.7, 0.2, 0.08, 0.02])
        })
        
        return data
    
    def _generate_healthcare_timeseries(self, config: DataTypeConfig) -> pd.DataFrame:
        """Patient monitoring time-series - high commercial value for healthtech"""
        
        start_date = config.start_date or datetime.now() - timedelta(hours=24)
        frequency = config.frequency or '5T'  # 5-minute intervals
        
        time_index = pd.date_range(start=start_date, periods=config.num_rows, freq=frequency)
        
        # Realistic vital signs patterns
        base_hr = 70 + np.random.normal(0, 10, config.num_rows)  # Heart rate
        activity_factor = 1 + 0.3 * np.random.choice([0, 1], config.num_rows, p=[0.8, 0.2])  # Activity spikes
        heart_rate = base_hr * activity_factor + np.random.normal(0, 5, config.num_rows)
        heart_rate = np.clip(heart_rate, 50, 180)
        
        # Blood pressure correlated with heart rate
        systolic_bp = 120 + 0.3 * (heart_rate - 70) + np.random.normal(0, 8, config.num_rows)
        diastolic_bp = systolic_bp * 0.67 + np.random.normal(0, 5, config.num_rows)
        
        data = pd.DataFrame({
            'timestamp': time_index,
            'patient_id': np.random.choice([f'P{i:04d}' for i in range(1, 101)], config.num_rows),
            'device_id': np.random.choice(['MON_001', 'MON_002', 'MON_003'], config.num_rows),
            'heart_rate_bpm': heart_rate,
            'systolic_bp_mmhg': systolic_bp,
            'diastolic_bp_mmhg': diastolic_bp,
            'oxygen_saturation_percent': np.random.uniform(95, 100, config.num_rows),
            'respiratory_rate_bpm': np.random.uniform(12, 20, config.num_rows),
            'body_temperature_c': 36.5 + np.random.normal(0, 0.5, config.num_rows),
            'activity_level': np.random.choice(['rest', 'light', 'moderate', 'high'], config.num_rows, p=[0.6, 0.25, 0.1, 0.05]),
            'alert_triggered': np.random.choice([0, 1], config.num_rows, p=[0.98, 0.02]),
            'medication_administered': np.random.choice([0, 1], config.num_rows, p=[0.95, 0.05])
        })
        
        return data
    
    def generate_geospatial_data(self, config: DataTypeConfig) -> pd.DataFrame:
        """
        Generate realistic geospatial data for various industries
        High value for: Mine locations, service areas, asset tracking
        """
        
        if config.industry_vertical == 'mining':
            return self._generate_mining_geospatial(config)
        elif config.industry_vertical == 'energy':
            return self._generate_energy_geospatial(config)
        elif config.industry_vertical == 'logistics':
            return self._generate_logistics_geospatial(config)
        else:
            return self._generate_generic_geospatial(config)
    
    def _generate_mining_geospatial(self, config: DataTypeConfig) -> pd.DataFrame:
        """Mining site locations and operations - high commercial value"""
        
        # Default to Australian mining region if no bounds specified
        bounds = config.location_bounds or (-26.0, -23.0, 115.0, 120.0)  # Pilbara region
        min_lat, max_lat, min_lon, max_lon = bounds
        
        # Generate realistic mine site clusters
        num_clusters = max(1, config.num_rows // 20)  # Cluster mines realistically
        cluster_centers = [
            (np.random.uniform(min_lat, max_lat), np.random.uniform(min_lon, max_lon))
            for _ in range(num_clusters)
        ]
        
        data = []
        for i in range(config.num_rows):
            # Assign to nearest cluster with some spread
            cluster = cluster_centers[i % len(cluster_centers)]
            lat = cluster[0] + np.random.normal(0, 0.1)  # ~10km spread
            lon = cluster[1] + np.random.normal(0, 0.1)
            
            # Ensure within bounds
            lat = np.clip(lat, min_lat, max_lat)
            lon = np.clip(lon, min_lon, max_lon)
            
            data.append({
                'site_id': f'MINE_{i+1:03d}',
                'latitude': lat,
                'longitude': lon,
                'elevation_m': np.random.uniform(200, 800),
                'site_type': np.random.choice(['open_pit', 'underground', 'processing'], p=[0.6, 0.3, 0.1]),
                'production_capacity_tpd': np.random.uniform(10000, 100000),  # Tonnes per day
                'ore_type': np.random.choice(['iron_ore', 'gold', 'copper', 'coal'], p=[0.4, 0.2, 0.2, 0.2]),
                'operational_status': np.random.choice(['active', 'maintenance', 'planned'], p=[0.8, 0.15, 0.05]),
                'environmental_zone': np.random.choice(['arid', 'semi_arid', 'temperate'], p=[0.6, 0.3, 0.1]),
                'nearest_town_km': np.random.uniform(5, 200),
                'rail_access': np.random.choice([True, False], p=[0.7, 0.3]),
                'water_source': np.random.choice(['bore', 'river', 'recycled'], p=[0.6, 0.2, 0.2])
            })
        
        return pd.DataFrame(data)
    
    def _generate_energy_geospatial(self, config: DataTypeConfig) -> pd.DataFrame:
        """Energy infrastructure locations - high commercial value"""
        
        # Default to renewable energy regions
        bounds = config.location_bounds or (32.0, 42.0, -125.0, -114.0)  # California
        min_lat, max_lat, min_lon, max_lon = bounds
        
        data = []
        for i in range(config.num_rows):
            lat = np.random.uniform(min_lat, max_lat)
            lon = np.random.uniform(min_lon, max_lon)
            
            facility_type = np.random.choice(['solar', 'wind', 'hydro', 'grid_station'], p=[0.4, 0.3, 0.1, 0.2])
            
            # Capacity based on type
            if facility_type == 'solar':
                capacity = np.random.uniform(1, 500)  # MW
            elif facility_type == 'wind':
                capacity = np.random.uniform(10, 300)
            elif facility_type == 'hydro':
                capacity = np.random.uniform(50, 2000)
            else:  # grid_station
                capacity = np.random.uniform(100, 1000)
            
            data.append({
                'facility_id': f'PWR_{i+1:03d}',
                'latitude': lat,
                'longitude': lon,
                'facility_type': facility_type,
                'capacity_mw': capacity,
                'commissioning_year': np.random.randint(2000, 2024),
                'grid_connection': np.random.choice(['transmission', 'distribution'], p=[0.3, 0.7]),
                'environmental_impact_score': np.random.uniform(1, 10),
                'land_use_hectares': capacity * np.random.uniform(0.5, 2.0),  # Rough correlation
                'annual_generation_gwh': capacity * np.random.uniform(2000, 4000) / 1000,  # Capacity factor
                'carbon_offset_tonnes_year': capacity * np.random.uniform(1000, 3000),
                'maintenance_access': np.random.choice(['road', 'helicopter', 'boat'], p=[0.8, 0.15, 0.05])
            })
        
        return pd.DataFrame(data)
    
    def generate_healthcare_data(self, config: DataTypeConfig) -> pd.DataFrame:
        """
        Generate comprehensive healthcare datasets
        High value for: EHR systems, clinical research, healthtech
        """
        
        return self._generate_synthetic_ehr(config)
    
    def _generate_synthetic_ehr(self, config: DataTypeConfig) -> pd.DataFrame:
        """Electronic Health Records - extremely high commercial value"""
        
        # Generate realistic patient demographics
        ages = np.random.gamma(2, 20, config.num_rows)  # Realistic age distribution
        ages = np.clip(ages, 0, 100).astype(int)
        
        data = []
        for i in range(config.num_rows):
            age = ages[i]
            gender = np.random.choice(['M', 'F'], p=[0.49, 0.51])
            
            # Age-correlated health metrics
            base_systolic = 110 + age * 0.5 + np.random.normal(0, 10)
            base_diastolic = 70 + age * 0.3 + np.random.normal(0, 8)
            
            # BMI with realistic distribution
            bmi = np.random.normal(26, 5)
            bmi = np.clip(bmi, 15, 50)
            
            # Chronic conditions based on age
            diabetes_risk = min(0.3, age * 0.003)
            hypertension_risk = min(0.5, age * 0.005)
            
            has_diabetes = np.random.random() < diabetes_risk
            has_hypertension = np.random.random() < hypertension_risk
            
            # Adjust vitals based on conditions
            if has_diabetes:
                glucose = np.random.uniform(140, 250)
            else:
                glucose = np.random.uniform(70, 120)
            
            if has_hypertension:
                base_systolic += 20
                base_diastolic += 10
            
            data.append({
                'patient_id': f'PT_{i+1:06d}',
                'age': age,
                'gender': gender,
                'bmi': round(bmi, 1),
                'systolic_bp': max(80, int(base_systolic)),
                'diastolic_bp': max(50, int(base_diastolic)),
                'heart_rate': int(np.random.normal(72, 12)),
                'glucose_mg_dl': int(glucose),
                'cholesterol_mg_dl': int(np.random.normal(200, 40)),
                'hemoglobin_g_dl': round(np.random.normal(14, 2), 1),
                'smoking_status': np.random.choice(['never', 'former', 'current'], p=[0.6, 0.25, 0.15]),
                'alcohol_use': np.random.choice(['none', 'moderate', 'heavy'], p=[0.4, 0.5, 0.1]),
                'exercise_frequency': np.random.choice(['none', 'weekly', 'daily'], p=[0.3, 0.5, 0.2]),
                'primary_diagnosis': self._get_age_appropriate_diagnosis(age, has_diabetes, has_hypertension),
                'medication_count': np.random.poisson(age // 20),
                'hospital_visits_year': np.random.poisson(max(1, age // 30)),
                'insurance_type': np.random.choice(['private', 'medicare', 'medicaid', 'uninsured'], p=[0.5, 0.3, 0.15, 0.05]),
                'emergency_contact': np.random.choice(['spouse', 'child', 'parent', 'sibling'], p=[0.4, 0.3, 0.2, 0.1])
            })
        
        return pd.DataFrame(data)
    
    def _get_age_appropriate_diagnosis(self, age: int, has_diabetes: bool, has_hypertension: bool) -> str:
        """Generate realistic primary diagnosis based on age and conditions"""
        
        if has_diabetes and has_hypertension:
            return 'diabetes_with_hypertension'
        elif has_diabetes:
            return 'type_2_diabetes'
        elif has_hypertension:
            return 'essential_hypertension'
        
        if age < 18:
            diagnoses = ['healthy', 'asthma', 'adhd', 'allergies']
            weights = [0.7, 0.15, 0.1, 0.05]
        elif age < 40:
            diagnoses = ['healthy', 'anxiety', 'depression', 'back_pain', 'migraine']
            weights = [0.5, 0.2, 0.15, 0.1, 0.05]
        elif age < 65:
            diagnoses = ['healthy', 'hypertension', 'arthritis', 'depression', 'back_pain', 'diabetes']
            weights = [0.3, 0.25, 0.15, 0.1, 0.1, 0.1]
        else:
            diagnoses = ['hypertension', 'arthritis', 'diabetes', 'heart_disease', 'osteoporosis']
            weights = [0.3, 0.25, 0.2, 0.15, 0.1]
        
        return np.random.choice(diagnoses, p=weights)
    
    def _generate_generic_timeseries(self, config: DataTypeConfig) -> pd.DataFrame:
        """Generic time-series for any industry"""
        
        start_date = config.start_date or datetime.now() - timedelta(days=30)
        frequency = config.frequency or 'H'
        
        time_index = pd.date_range(start=start_date, periods=config.num_rows, freq=frequency)
        
        # Generic sensor-like data
        trend = np.linspace(0, 1, config.num_rows)
        seasonal = np.sin(2 * np.pi * np.arange(config.num_rows) / 24)
        noise = np.random.normal(0, 0.1, config.num_rows)
        
        value = 100 + 20 * trend + 10 * seasonal + noise
        
        return pd.DataFrame({
            'timestamp': time_index,
            'sensor_id': np.random.choice(['SENSOR_A', 'SENSOR_B', 'SENSOR_C'], config.num_rows),
            'value': value,
            'quality': np.random.choice(['good', 'fair', 'poor'], config.num_rows, p=[0.8, 0.15, 0.05]),
            'status': np.random.choice(['online', 'offline'], config.num_rows, p=[0.95, 0.05])
        })
    
    def _generate_generic_geospatial(self, config: DataTypeConfig) -> pd.DataFrame:
        """Generic geospatial data"""
        
        bounds = config.location_bounds or (-90, 90, -180, 180)  # World bounds
        min_lat, max_lat, min_lon, max_lon = bounds
        
        data = []
        for i in range(config.num_rows):
            data.append({
                'location_id': f'LOC_{i+1:04d}',
                'latitude': np.random.uniform(min_lat, max_lat),
                'longitude': np.random.uniform(min_lon, max_lon),
                'elevation': np.random.uniform(0, 1000),
                'category': np.random.choice(['urban', 'rural', 'industrial'], p=[0.5, 0.3, 0.2])
            })
        
        return pd.DataFrame(data)
    
    def get_supported_types(self) -> List[str]:
        """Get list of supported advanced data types"""
        return self.supported_types.copy()
    
    def get_industry_verticals(self) -> Dict[str, List[str]]:
        """Get supported industry verticals for each data type"""
        return {
            'timeseries': ['energy', 'mining', 'healthcare', 'manufacturing', 'iot'],
            'geospatial': ['mining', 'energy', 'logistics', 'retail', 'agriculture'],
            'healthcare': ['clinical', 'research', 'pharmaceutical', 'insurance'],
            'industrial_iot': ['manufacturing', 'mining', 'energy', 'transportation'],
            'financial': ['banking', 'insurance', 'trading', 'fintech']
        }