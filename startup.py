#!/usr/bin/env python3
"""
Startup script to launch the Streamlit app with proper environment setup
This helps avoid torch/pytorch compatibility issues with Streamlit
"""

import os
import sys
import subprocess

def setup_environment():
    """Set up environment variables to avoid torch issues"""
    # Set environment variables to avoid torch issues
    os.environ['TORCH_HOME'] = os.path.join(os.getcwd(), 'torch_cache')
    os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'hf_cache')
    
    # Create cache directories
    os.makedirs(os.environ['TORCH_HOME'], exist_ok=True)
    os.makedirs(os.environ['HF_HOME'], exist_ok=True)
    
    print("✓ Environment variables set")

def install_requirements():
    """Install required packages"""
    required_packages = [
        "sentence-transformers",
        "langchain-huggingface",
        "torch",
        "transformers"
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package} already installed")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    """Main startup function"""
    print("🚀 Starting Document Research Assistant...")
    
    # Setup environment
    setup_environment()
    
    # Check and install requirements
    try:
        install_requirements()
    except Exception as e:
        print(f"⚠️ Warning: Could not install some packages: {e}")
        print("Please manually install: pip install sentence-transformers langchain-huggingface")
    
    # Launch Streamlit
    print("🌟 Launching Streamlit app...")
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"❌ Error launching Streamlit: {e}")
        print("Try running manually: streamlit run streamlit_app.py")

if __name__ == "__main__":
    main()