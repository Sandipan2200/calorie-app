#!/usr/bin/env python3
"""
Setup script for Advanced Food Detection System
This script helps install dependencies and set up the environment
"""

import subprocess
import sys
import os
import json
from datetime import datetime

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("❌ Error: Python 3.7 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_package(package):
    """Install a single package using pip"""
    try:
        print(f"📦 Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def install_dependencies():
    """Install all required dependencies"""
    required_packages = [
        "tensorflow>=2.8.0",
        "opencv-python>=4.5.0",
        "pillow>=8.0.0",
        "matplotlib>=3.3.0",
        "requests>=2.25.0",
        "numpy>=1.19.0"
    ]
    
    print("\n🚀 Installing required packages...")
    print("="*50)
    
    failed_packages = []
    
    for package in required_packages:
        if not install_package(package):
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\n❌ Failed to install: {', '.join(failed_packages)}")
        print("💡 Try installing them manually:")
        for package in failed_packages:
            print(f"   pip install {package}")
        return False
    
    print("\n✅ All packages installed successfully!")
    return True

def create_sample_database():
    """Create a sample JSON database file"""
    database_file = "food_nutrition_database.json"
    
    if os.path.exists(database_file):
        print(f"✅ Database file already exists: {database_file}")
        return True
    
    print(f"📝 Creating sample database: {database_file}")
    
    sample_data = {
        "foods": {
            "apple": {"calories": 52, "protein": 0.3, "carbs": 14, "fat": 0.2, "source": "default"},
            "banana": {"calories": 89, "protein": 1.1, "carbs": 23, "fat": 0.3, "source": "default"},
            "pizza": {"calories": 266, "protein": 11, "carbs": 33, "fat": 10, "source": "default"},
            "burger": {"calories": 295, "protein": 17, "carbs": 28, "fat": 14, "source": "default"},
            "pasta": {"calories": 220, "protein": 8, "carbs": 44, "fat": 1.5, "source": "default"},
            "rice": {"calories": 130, "protein": 2.7, "carbs": 28, "fat": 0.3, "source": "default"},
            "chicken": {"calories": 239, "protein": 27, "carbs": 0, "fat": 14, "source": "default"},
            "salad": {"calories": 33, "protein": 3, "carbs": 6, "fat": 0.3, "source": "default"}
        },
        "learning_data": {
            "corrections": [],
            "total_corrections": 0,
            "last_updated": datetime.now().isoformat()
        },
        "statistics": {
            "total_predictions": 0,
            "correct_predictions": 0,
            "accuracy_rate": 0.0
        }
    }
    
    try:
        with open(database_file, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Created database with {len(sample_data['foods'])} default foods")
        return True
    except Exception as e:
        print(f"❌ Failed to create database: {e}")
        return False

def check_camera_availability():
    """Check if camera is available"""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Camera is available")
            cap.release()
            return True
        else:
            print("⚠️ Camera not available (optional feature)")
            return False
    except ImportError:
        print("⚠️ OpenCV not available yet - camera check skipped")
        return False
    except Exception as e:
        print(f"⚠️ Camera check failed: {e}")
        return False

def test_tensorflow():
    """Test TensorFlow installation"""
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
        
        # Test if GPU is available (optional)
        if tf.config.list_physical_devices('GPU'):
            print("🚀 GPU acceleration available!")
        else:
            print("💻 Using CPU (GPU not available)")
        
        return True
    except ImportError:
        print("❌ TensorFlow not available")
        return False
    except Exception as e:
        print(f"⚠️ TensorFlow test failed: {e}")
        return False

def create_sample_script():
    """Create a sample usage script"""
    sample_script = """#!/usr/bin/env python3
# Sample usage of Advanced Food Detection System

from food_detection_system import AdvancedFoodDetector

def main():
    # Initialize the detector
    detector = AdvancedFoodDetector()
    
    # Example 1: Detect from image file
    # result = detector.detect_food("path/to/your/image.jpg")
    
    # Example 2: Detect from URL
    # result = detector.detect_food("https://example.com/food_image.jpg")
    
    # Example 3: Use camera
    # detector.detect_from_camera()
    
    print("🍽️ Food Detection System is ready!")
    print("Run the main script to start: python food_detection_system.py")

if __name__ == "__main__":
    main()
"""
    
    try:
        with open("sample_usage.py", "w") as f:
            f.write(sample_script)
        print("✅ Created sample_usage.py")
        return True
    except Exception as e:
        print(f"❌ Failed to create sample script: {e}")
        return False

def main():
    """Main setup function"""
    print("🍽️ ADVANCED FOOD DETECTION SYSTEM - SETUP")
    print("="*55)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed due to dependency issues")
        return False
    
    print("\n🔧 Setting up environment...")
    print("="*30)
    
    # Create database
    create_sample_database()
    
    # Test TensorFlow
    test_tensorflow()
    
    # Check camera
    check_camera_availability()
    
    # Create sample script
    create_sample_script()
    
    print("\n✅ SETUP COMPLETE!")
    print("="*20)
    print("🚀 Ready to use the Advanced Food Detection System!")
    print("\n📋 Quick Start:")
    print("   1. Run: python food_detection_system.py")
    print("   2. Choose option 2 to test with a URL")
    print("   3. Try this sample URL: https://example.com/pizza.jpg")
    print("   4. Provide feedback to help the system learn!")
    
    print("\n📁 Files created:")
    print("   • food_nutrition_database.json (food database)")
    print("   • sample_usage.py (usage examples)")
    
    print("\n💡 Tips:")
    print("   • The system learns from your corrections")
    print("   • Use clear, well-lit food images for best results")
    print("   • The GUI feedback system requires tkinter (usually pre-installed)")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
    
    print("\n🎉 Setup successful! You can now run the food detection system.")