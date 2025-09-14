"""
Constraints and Business Logic Manager for Synthetic Data Generation
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from abc import ABC, abstractmethod
try:
    from sdv.constraints import create_custom_constraint
except ImportError:
    # Fallback for newer SDV versions
    def create_custom_constraint(*args, **kwargs):
        return None
from schema_manager import DatasetSchema, ColumnSchema, ConstraintType
import logging

logger = logging.getLogger(__name__)

class BaseConstraint(ABC):
    """Base class for all constraints"""
    
    def __init__(self, name: str, description: str, columns: List[str]):
        self.name = name
        self.description = description
        self.columns = columns
    
    @abstractmethod
    def validate(self, data: pd.DataFrame) -> bool:
        """Validate if data satisfies the constraint"""
        pass
    
    @abstractmethod
    def to_sdv_constraint(self):
        """Convert to SDV constraint format"""
        pass

class PositiveConstraint(BaseConstraint):
    """Ensures values are positive"""
    
    def __init__(self, column: str):
        super().__init__(f"positive_{column}", f"{column} must be positive", [column])
        self.column = column
    
    def validate(self, data: pd.DataFrame) -> bool:
        return (data[self.column] >= 0).all()
    
    def to_sdv_constraint(self):
        # Simplified constraint for compatibility
        return {
            'constraint_type': 'positive',
            'column': self.column,
            'description': self.description
        }

class RangeConstraint(BaseConstraint):
    """Ensures values are within a specified range"""
    
    def __init__(self, column: str, min_val: float, max_val: float):
        super().__init__(f"range_{column}", f"{column} must be between {min_val} and {max_val}", [column])
        self.column = column
        self.min_val = min_val
        self.max_val = max_val
    
    def validate(self, data: pd.DataFrame) -> bool:
        return ((data[self.column] >= self.min_val) & (data[self.column] <= self.max_val)).all()
    
    def to_sdv_constraint(self):
        return {
            'constraint_type': 'range',
            'column': self.column,
            'min_value': self.min_val,
            'max_value': self.max_val,
            'description': self.description
        }

class ProportionalConstraint(BaseConstraint):
    """Ensures one column is proportional to another"""
    
    def __init__(self, column1: str, column2: str, ratio_range: tuple = (0.5, 2.0)):
        super().__init__(
            f"proportional_{column1}_{column2}", 
            f"{column1} should be proportional to {column2}", 
            [column1, column2]
        )
        self.column1 = column1
        self.column2 = column2
        self.min_ratio, self.max_ratio = ratio_range
    
    def validate(self, data: pd.DataFrame) -> bool:
        # Avoid division by zero
        mask = data[self.column2] > 0
        if not mask.any():
            return False
        
        ratios = data.loc[mask, self.column1] / data.loc[mask, self.column2]
        return ((ratios >= self.min_ratio) & (ratios <= self.max_ratio)).all()
    
    def to_sdv_constraint(self):
        return {
            'constraint_type': 'proportional',
            'column1': self.column1,
            'column2': self.column2,
            'min_ratio': self.min_ratio,
            'max_ratio': self.max_ratio,
            'description': self.description
        }

class IncreasingConstraint(BaseConstraint):
    """Ensures values in a column are increasing (for time series)"""
    
    def __init__(self, column: str):
        super().__init__(f"increasing_{column}", f"{column} values should be increasing", [column])
        self.column = column
    
    def validate(self, data: pd.DataFrame) -> bool:
        return data[self.column].is_monotonic_increasing
    
    def to_sdv_constraint(self):
        return {
            'constraint_type': 'increasing',
            'column': self.column,
            'description': self.description
        }

class UniqueConstraint(BaseConstraint):
    """Ensures values in a column are unique"""
    
    def __init__(self, column: str):
        super().__init__(f"unique_{column}", f"{column} values must be unique", [column])
        self.column = column
    
    def validate(self, data: pd.DataFrame) -> bool:
        return not data[self.column].duplicated().any()
    
    def to_sdv_constraint(self):
        return {
            'constraint_type': 'unique',
            'column': self.column,
            'description': self.description
        }

class DomainConstraint(BaseConstraint):
    """Domain-specific business logic constraints"""
    
    def __init__(self, name: str, description: str, columns: List[str], validation_fn: Callable, correction_fn: Callable):
        super().__init__(name, description, columns)
        self.validation_fn = validation_fn
        self.correction_fn = correction_fn
    
    def validate(self, data: pd.DataFrame) -> bool:
        return self.validation_fn(data)
    
    def to_sdv_constraint(self):
        return {
            'constraint_type': 'domain_specific',
            'name': self.name,
            'columns': self.columns,
            'description': self.description
        }

class ConstraintsManager:
    """Manages and applies constraints for synthetic data generation"""
    
    def __init__(self):
        self.constraints = []
        self.domain_rules = self._load_domain_rules()
    
    def _load_domain_rules(self) -> Dict[str, List[Callable]]:
        """Load predefined domain-specific rules"""
        rules = {}
        
        # Mining domain rules
        def mining_energy_tonnage_rule(data):
            """Energy should be proportional to tonnage in mining operations"""
            if 'energy_used' in data.columns and 'tonnage' in data.columns:
                mask = data['tonnage'] > 0
                if mask.any():
                    ratios = data.loc[mask, 'energy_used'] / data.loc[mask, 'tonnage']
                    return (ratios >= 2.0) & (ratios <= 4.0)  # Reasonable energy/tonnage ratio
            return pd.Series([True] * len(data))
        
        def mining_correction(data):
            """Correct energy values to maintain proportionality with tonnage"""
            if 'energy_used' in data.columns and 'tonnage' in data.columns:
                mask = data['tonnage'] > 0
                data.loc[mask, 'energy_used'] = data.loc[mask, 'tonnage'] * np.random.uniform(2.5, 3.5, mask.sum())
            return data
        
        rules['mining'] = [
            DomainConstraint(
                "mining_energy_tonnage",
                "Energy consumption should be proportional to tonnage processed",
                ['energy_used', 'tonnage'],
                mining_energy_tonnage_rule,
                mining_correction
            )
        ]
        
        # Healthcare domain rules
        def healthcare_bp_rule(data):
            """Systolic BP should be higher than diastolic BP"""
            if 'blood_pressure_systolic' in data.columns and 'blood_pressure_diastolic' in data.columns:
                return data['blood_pressure_systolic'] > data['blood_pressure_diastolic']
            return pd.Series([True] * len(data))
        
        def healthcare_bp_correction(data):
            """Correct blood pressure values to maintain systolic > diastolic"""
            if 'blood_pressure_systolic' in data.columns and 'blood_pressure_diastolic' in data.columns:
                # Ensure systolic is always higher than diastolic
                mask = data['blood_pressure_systolic'] <= data['blood_pressure_diastolic']
                if mask.any():
                    data.loc[mask, 'blood_pressure_systolic'] = data.loc[mask, 'blood_pressure_diastolic'] + np.random.uniform(10, 30, mask.sum())
            return data
        
        rules['healthcare'] = [
            DomainConstraint(
                "healthcare_blood_pressure",
                "Systolic blood pressure must be higher than diastolic",
                ['blood_pressure_systolic', 'blood_pressure_diastolic'],
                healthcare_bp_rule,
                healthcare_bp_correction
            )
        ]
        
        # Energy domain rules
        def energy_capacity_rule(data):
            """Power output should not exceed max capacity"""
            if 'power_output' in data.columns and 'max_capacity' in data.columns:
                return data['power_output'] <= data['max_capacity']
            return pd.Series([True] * len(data))
        
        def energy_capacity_correction(data):
            """Correct power output to not exceed capacity"""
            if 'power_output' in data.columns and 'max_capacity' in data.columns:
                mask = data['power_output'] > data['max_capacity']
                data.loc[mask, 'power_output'] = data.loc[mask, 'max_capacity'] * np.random.uniform(0.7, 0.95, mask.sum())
            return data
        
        rules['energy'] = [
            DomainConstraint(
                "energy_capacity_constraint",
                "Power output must not exceed maximum capacity",
                ['power_output', 'max_capacity'],
                energy_capacity_rule,
                energy_capacity_correction
            )
        ]
        
        return rules
    
    def add_constraint(self, constraint: BaseConstraint):
        """Add a constraint to the manager"""
        self.constraints.append(constraint)
    
    def build_constraints_from_schema(self, schema: DatasetSchema) -> List[BaseConstraint]:
        """Build constraints from a dataset schema"""
        constraints = []
        
        # Add column-level constraints
        for col in schema.columns:
            # Add range constraints for continuous variables
            if col.type.value == "continuous" and col.min_value is not None and col.max_value is not None:
                constraints.append(RangeConstraint(col.name, col.min_value, col.max_value))
            
            # Add unique constraints for ID columns
            if col.type.value == "id":
                constraints.append(UniqueConstraint(col.name))
            
            # Process column-specific constraints
            for constraint_def in col.constraints:
                constraint_type = constraint_def.get("type")
                
                if constraint_type == "positive":
                    constraints.append(PositiveConstraint(col.name))
                elif constraint_type == "range":
                    min_val = constraint_def.get("min", col.min_value)
                    max_val = constraint_def.get("max", col.max_value)
                    if min_val is not None and max_val is not None:
                        constraints.append(RangeConstraint(col.name, min_val, max_val))
                elif constraint_type == "unique":
                    constraints.append(UniqueConstraint(col.name))
                elif constraint_type == "increasing":
                    constraints.append(IncreasingConstraint(col.name))
                elif constraint_type == "proportional":
                    target_col = constraint_def.get("target")
                    ratio_range = constraint_def.get("ratio_range", (0.5, 2.0))
                    if target_col:
                        constraints.append(ProportionalConstraint(col.name, target_col, ratio_range))
        
        # Add domain-specific constraints
        if schema.domain and schema.domain in self.domain_rules:
            constraints.extend(self.domain_rules[schema.domain])
        
        return constraints
    
    def validate_data(self, data: pd.DataFrame, constraints: List[BaseConstraint] = None) -> Dict[str, Any]:
        """Validate data against constraints"""
        if constraints is None:
            constraints = self.constraints
        
        results = {
            "valid": True,
            "violations": [],
            "constraint_results": {}
        }
        
        for constraint in constraints:
            try:
                is_valid = constraint.validate(data)
                results["constraint_results"][constraint.name] = {
                    "valid": is_valid,
                    "description": constraint.description
                }
                
                if not is_valid:
                    results["valid"] = False
                    results["violations"].append({
                        "constraint": constraint.name,
                        "description": constraint.description,
                        "columns": constraint.columns
                    })
            except Exception as e:
                logger.error(f"Error validating constraint {constraint.name}: {e}")
                results["constraint_results"][constraint.name] = {
                    "valid": False,
                    "error": str(e),
                    "description": constraint.description
                }
        
        return results
    
    def get_sdv_constraints(self, constraints: List[BaseConstraint] = None) -> List:
        """Convert constraints to SDV format for use in synthetic data generation"""
        if constraints is None:
            constraints = self.constraints
        
        sdv_constraints = []
        for constraint in constraints:
            try:
                sdv_constraint = constraint.to_sdv_constraint()
                if sdv_constraint:
                    sdv_constraints.append(sdv_constraint)
            except Exception as e:
                logger.error(f"Error converting constraint {constraint.name} to SDV format: {e}")
        
        return sdv_constraints
    
    def apply_constraints_to_data(self, data: pd.DataFrame, constraints: List[BaseConstraint] = None) -> pd.DataFrame:
        """Apply constraint corrections to data"""
        if constraints is None:
            constraints = self.constraints
        
        corrected_data = data.copy()
        
        for constraint in constraints:
            try:
                if hasattr(constraint, 'correction_fn'):
                    corrected_data = constraint.correction_fn(corrected_data)
            except Exception as e:
                logger.error(f"Error applying constraint correction {constraint.name}: {e}")
        
        return corrected_data