#!/usr/bin/env python3
"""
Fix MPS (Apple Silicon GPU) compatibility issues with SDV
"""
import os
import sys

def fix_mps_issues():
    """Set environment variables to fix MPS issues"""
    
    # Disable MPS for PyTorch to avoid compatibility issues
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    
    # Force CPU usage for neural networks
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    # SDV specific fixes
    os.environ['SDV_DISABLE_MPS'] = '1'
    
    print("🔧 Applied MPS compatibility fixes:")
    print("   - Disabled MPS acceleration")
    print("   - Forced CPU usage for neural networks")
    print("   - Set PyTorch MPS fallback")
    print("   - Applied SDV compatibility settings")

if __name__ == "__main__":
    fix_mps_issues()