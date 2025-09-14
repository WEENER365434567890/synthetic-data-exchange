"""
Export Manager for Multiple Data Formats
"""
import pandas as pd
import io
import json
from typing import Dict, Any, Optional
from datetime import datetime
import logging

# PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.lineplots import LinePlot
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logging.warning("ReportLab not available - PDF export disabled")

# Visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import base64
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logging.warning("Matplotlib/Seaborn not available - charts disabled")

logger = logging.getLogger(__name__)

class ExportManager:
    """Handles multiple export formats for synthetic data and reports"""
    
    def __init__(self):
        self.supported_formats = ['csv', 'xlsx', 'sql', 'json']
        if PDF_AVAILABLE:
            self.supported_formats.append('pdf')
    
    def export_data(self, data: pd.DataFrame, format_type: str, **kwargs) -> bytes:
        """Export synthetic data in specified format"""
        
        if format_type.lower() == 'csv':
            return self._export_csv(data, **kwargs)
        elif format_type.lower() == 'xlsx':
            return self._export_excel(data, **kwargs)
        elif format_type.lower() == 'sql':
            return self._export_sql(data, **kwargs)
        elif format_type.lower() == 'json':
            return self._export_json(data, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _export_csv(self, data: pd.DataFrame, **kwargs) -> bytes:
        """Export as CSV"""
        output = io.StringIO()
        data.to_csv(output, index=False, **kwargs)
        return output.getvalue().encode('utf-8')
    
    def _export_excel(self, data: pd.DataFrame, **kwargs) -> bytes:
        """Export as Excel (.xlsx)"""
        output = io.BytesIO()
        
        # Create Excel writer with multiple sheets if needed
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Main data sheet
            data.to_excel(writer, sheet_name='Synthetic Data', index=False)
            
            # Add metadata sheet
            metadata = pd.DataFrame({
                'Property': ['Generated On', 'Rows', 'Columns', 'Data Types'],
                'Value': [
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    len(data),
                    len(data.columns),
                    ', '.join(data.dtypes.astype(str).unique())
                ]
            })
            metadata.to_excel(writer, sheet_name='Metadata', index=False)
            
            # Add column info sheet
            column_info = pd.DataFrame({
                'Column': data.columns,
                'Data Type': data.dtypes.astype(str),
                'Non-Null Count': data.count(),
                'Unique Values': [data[col].nunique() for col in data.columns],
                'Sample Value': [str(data[col].iloc[0]) if len(data) > 0 else '' for col in data.columns]
            })
            column_info.to_excel(writer, sheet_name='Column Info', index=False)
        
        output.seek(0)
        return output.getvalue()
    
    def _export_sql(self, data: pd.DataFrame, table_name: str = 'synthetic_data', **kwargs) -> bytes:
        """Export as SQL INSERT statements"""
        
        # Generate CREATE TABLE statement
        sql_types = {
            'int64': 'INTEGER',
            'float64': 'REAL',
            'object': 'TEXT',
            'bool': 'BOOLEAN',
            'datetime64[ns]': 'TIMESTAMP'
        }
        
        create_table = f"-- Synthetic Data Export\n-- Generated on {datetime.now()}\n\n"
        create_table += f"CREATE TABLE {table_name} (\n"
        
        columns = []
        for col, dtype in data.dtypes.items():
            sql_type = sql_types.get(str(dtype), 'TEXT')
            columns.append(f"    {col} {sql_type}")
        
        create_table += ",\n".join(columns)
        create_table += "\n);\n\n"
        
        # Generate INSERT statements
        insert_statements = []
        for _, row in data.iterrows():
            values = []
            for val in row:
                if pd.isna(val):
                    values.append('NULL')
                elif isinstance(val, str):
                    # Escape single quotes
                    escaped_val = val.replace("'", "''")
                    values.append(f"'{escaped_val}'")
                elif isinstance(val, (int, float)):
                    values.append(str(val))
                else:
                    values.append(f"'{str(val)}'")
            
            insert_stmt = f"INSERT INTO {table_name} VALUES ({', '.join(values)});"
            insert_statements.append(insert_stmt)
        
        # Combine all statements
        full_sql = create_table + "\n".join(insert_statements)
        return full_sql.encode('utf-8')
    
    def _export_json(self, data: pd.DataFrame, **kwargs) -> bytes:
        """Export as JSON"""
        # Convert to JSON with metadata
        export_data = {
            'metadata': {
                'generated_on': datetime.now().isoformat(),
                'rows': len(data),
                'columns': len(data.columns),
                'column_types': data.dtypes.astype(str).to_dict()
            },
            'data': data.to_dict('records')
        }
        
        return json.dumps(export_data, indent=2, default=str).encode('utf-8')
    
    def export_evaluation_report(self, evaluation_data: Dict[str, Any], format_type: str = 'pdf') -> bytes:
        """Export evaluation report as PDF"""
        
        if format_type.lower() == 'pdf':
            if not PDF_AVAILABLE:
                raise ValueError("PDF export not available - reportlab not installed")
            return self._export_evaluation_pdf(evaluation_data)
        elif format_type.lower() == 'json':
            return json.dumps(evaluation_data, indent=2, default=str).encode('utf-8')
        else:
            raise ValueError(f"Unsupported report format: {format_type}")
    
    def _export_evaluation_pdf(self, evaluation_data: Dict[str, Any]) -> bytes:
        """Generate PDF evaluation report"""
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.darkblue
        )
        story.append(Paragraph("Synthetic Data Quality Evaluation Report", title_style))
        story.append(Spacer(1, 20))
        
        # Metadata
        metadata = evaluation_data.get('metadata', {})
        story.append(Paragraph("Report Information", styles['Heading2']))
        
        metadata_data = [
            ['Property', 'Value'],
            ['Generated On', metadata.get('evaluation_timestamp', 'Unknown')],
            ['Real Data Shape', str(metadata.get('real_data_shape', 'Unknown'))],
            ['Synthetic Data Shape', str(metadata.get('synthetic_data_shape', 'Unknown'))],
            ['Target Column', metadata.get('target_column', 'None')]
        ]
        
        metadata_table = Table(metadata_data)
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(metadata_table)
        story.append(Spacer(1, 20))
        
        # Overall Scores
        overall_scores = evaluation_data.get('overall_scores', {})
        story.append(Paragraph("Overall Quality Assessment", styles['Heading2']))
        
        # Quality grade with color
        grade = overall_scores.get('grade', 'F')
        overall_score = overall_scores.get('overall_quality_score', 0)
        
        grade_color = colors.green if grade.startswith('A') else colors.blue if grade.startswith('B') else colors.orange if grade.startswith('C') else colors.red
        
        grade_style = ParagraphStyle(
            'GradeStyle',
            parent=styles['Normal'],
            fontSize=36,
            textColor=grade_color,
            alignment=1  # Center
        )
        
        story.append(Paragraph(f"Quality Grade: {grade}", grade_style))
        story.append(Paragraph(f"Overall Score: {overall_score:.1f}/100", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Detailed scores
        scores_data = [
            ['Metric', 'Score', 'Grade'],
            ['Statistical Similarity', f"{overall_scores.get('statistical_similarity_score', 0):.1f}/100", self._score_to_grade(overall_scores.get('statistical_similarity_score', 0))],
            ['Utility Preservation', f"{overall_scores.get('utility_preservation_score', 0):.1f}/100", self._score_to_grade(overall_scores.get('utility_preservation_score', 0))],
            ['Privacy Protection', f"{overall_scores.get('privacy_protection_score', 0):.1f}/100", self._score_to_grade(overall_scores.get('privacy_protection_score', 0))]
        ]
        
        scores_table = Table(scores_data)
        scores_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(scores_table)
        story.append(Spacer(1, 30))
        
        # Statistical Similarity Details
        statistical_similarity = evaluation_data.get('statistical_similarity', {})
        if statistical_similarity:
            story.append(Paragraph("Statistical Similarity Analysis", styles['Heading2']))
            
            # KS Test Results
            ks_results = statistical_similarity.get('kolmogorov_smirnov', {})
            if ks_results:
                story.append(Paragraph("Kolmogorov-Smirnov Test Results", styles['Heading3']))
                ks_data = [['Column', 'KS Statistic', 'P-Value', 'Similar']]
                for col, result in ks_results.items():
                    ks_data.append([
                        col,
                        f"{result.get('ks_statistic', 0):.4f}",
                        f"{result.get('p_value', 0):.4f}",
                        "✓" if result.get('similar', False) else "✗"
                    ])
                
                ks_table = Table(ks_data)
                ks_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(ks_table)
                story.append(Spacer(1, 20))
        
        # Privacy Protection Details
        privacy_protection = evaluation_data.get('privacy_protection', {})
        if privacy_protection:
            story.append(Paragraph("Privacy Protection Analysis", styles['Heading2']))
            
            nn_analysis = privacy_protection.get('nearest_neighbor_analysis', {})
            if nn_analysis and 'error' not in nn_analysis:
                story.append(Paragraph("Nearest Neighbor Analysis", styles['Heading3']))
                nn_data = [
                    ['Metric', 'Value'],
                    ['Mean Distance', f"{nn_analysis.get('mean_nearest_neighbor_distance', 0):.4f}"],
                    ['Min Distance', f"{nn_analysis.get('min_distance', 0):.4f}"],
                    ['Risky Records', f"{nn_analysis.get('risky_records_count', 0)} ({nn_analysis.get('risky_records_percentage', 0):.1f}%)"],
                    ['Privacy Score', f"{nn_analysis.get('privacy_score', 0):.1f}/100"]
                ]
                
                nn_table = Table(nn_data)
                nn_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(nn_table)
                story.append(Spacer(1, 20))
        
        # Recommendations
        story.append(Paragraph("Recommendations", styles['Heading2']))
        recommendations = [
            "• Use this synthetic data for development and testing environments",
            "• Consider additional privacy measures for production use",
            "• Validate synthetic data quality with domain experts",
            "• Monitor data drift over time with regular evaluations"
        ]
        
        for rec in recommendations:
            story.append(Paragraph(rec, styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # Footer
        story.append(Paragraph("Generated by Synthetic Data Exchange", styles['Normal']))
        story.append(Paragraph(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    
    def _score_to_grade(self, score: float) -> str:
        """Convert numerical score to letter grade"""
        if score >= 90:
            return 'A+'
        elif score >= 85:
            return 'A'
        elif score >= 80:
            return 'A-'
        elif score >= 75:
            return 'B+'
        elif score >= 70:
            return 'B'
        elif score >= 65:
            return 'B-'
        elif score >= 60:
            return 'C+'
        elif score >= 55:
            return 'C'
        elif score >= 50:
            return 'C-'
        else:
            return 'F'
    
    def get_supported_formats(self) -> list:
        """Get list of supported export formats"""
        return self.supported_formats.copy()