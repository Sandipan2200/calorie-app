#!/usr/bin/env python3
"""
Enhanced Setup Script for Advanced Food Detection System
Automatically installs dependencies and configures the system
"""

import subprocess
import sys
import os
import json
import platform
from datetime import datetime
import requests

def print_banner():
    """Print setup banner"""
    print("🍽️" + "="*60 + "🍽️")
    print("     ENHANCED FOOD DETECTION SYSTEM SETUP")
    print("🍽️" + "="*60 + "🍽️")
    print("🤖 Multi-Model AI Detection")
    print("🌐 Web Nutrition Data Scraping")
    print("📚 Advanced Learning System")
    print("📊 Real-time Analytics")
    print("="*64)

def check_system_requirements():
    """Check system requirements"""
    print("\n🔍 Checking system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 8):
        print(f"❌ Python 3.8+ required. Current: {python_version.major}.{python_version.minor}")
        return False
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check platform
    system_info = platform.system()
    print(f"✅ Platform: {system_info} {platform.release()}")
    
    # Check available memory (approximate)
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb < 4:
            print(f"⚠️ Warning: Only {memory_gb:.1f}GB RAM available. 4GB+ recommended.")
        else:
            print(f"✅ Memory: {memory_gb:.1f}GB available")
    except ImportError:
        print("ℹ️ Memory check skipped (psutil not available)")
    
    return True

def install_packages():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    
    # Core packages for enhanced system
    packages = [
        "tensorflow>=2.10.0",
        "opencv-python>=4.6.0", 
        "pillow>=9.0.0",
        "numpy>=1.21.0",
        "requests>=2.28.0",
        "beautifulsoup4>=4.11.0",
        "scikit-learn>=1.1.0",
        "matplotlib>=3.5.0",
        "psutil>=5.9.0",  # For system monitoring
        "lxml>=4.9.0",    # For better HTML parsing
    ]
    
    failed_packages = []
    
    for package in packages:
        try:
            print(f"📦 Installing {package.split('>=')[0]}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package, "--upgrade"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"✅ {package.split('>=')[0]} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\n⚠️ Failed to install: {', '.join(failed_packages)}")
        print("💡 Try installing manually:")
        for package in failed_packages:
            print(f"   pip install {package}")
        return False
    
    print("✅ All packages installed successfully!")
    return True

def test_tensorflow_setup():
    """Test TensorFlow installation and GPU availability"""
    print("\n🧠 Testing AI model setup...")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} loaded successfully")
        
        # Check GPU availability
        gpu_devices = tf.config.list_physical_devices('GPU')
        if gpu_devices:
            print(f"🚀 GPU acceleration available! Found {len(gpu_devices)} GPU(s)")
            for i, gpu in enumerate(gpu_devices):
                print(f"   GPU {i}: {gpu.name}")
        else:
            print("💻 Using CPU (no GPU detected)")
        
        # Test model loading
        print("🔄 Testing model loading...")
        from tensorflow.keras.applications import ResNet50
        model = ResNet50(weights='imagenet', include_top=True)
        print("✅ ResNet50 model loaded successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ TensorFlow import failed: {e}")
        return False
    except Exception as e:
        print(f"⚠️ TensorFlow test failed: {e}")
        return True  # Continue setup even if test fails

def create_enhanced_database():
    """Create enhanced nutrition database"""
    print("\n🗃️ Creating enhanced nutrition database...")
    
    db_file = "enhanced_food_nutrition_database.json"
    
    if os.path.exists(db_file):
        print(f"✅ Database already exists: {db_file}")
        return True
    
    # Comprehensive food database
    enhanced_data = {
        "foods": {
            # Fruits
            "apple": {"calories": 52, "protein": 0.3, "carbs": 14, "fat": 0.2, "fiber": 2.4, "sugar": 10.4, "sodium": 1, "source": "default"},
            "banana": {"calories": 89, "protein": 1.1, "carbs": 23, "fat": 0.3, "fiber": 2.6, "sugar": 12.2, "sodium": 1, "source": "default"},
            "orange": {"calories": 47, "protein": 0.9, "carbs": 12, "fat": 0.1, "fiber": 2.4, "sugar": 9.4, "sodium": 0, "source": "default"},
            "strawberry": {"calories": 32, "protein": 0.7, "carbs": 8, "fat": 0.3, "fiber": 2.0, "sugar": 4.9, "sodium": 1, "source": "default"},
            "grapes": {"calories": 62, "protein": 0.6, "carbs": 16, "fat": 0.2, "fiber": 0.9, "sugar": 16, "sodium": 2, "source": "default"},
            
            # Vegetables  
            "broccoli": {"calories": 34, "protein": 2.8, "carbs": 7, "fat": 0.4, "fiber": 2.6, "sugar": 1.5, "sodium": 33, "source": "default"},
            "carrot": {"calories": 41, "protein": 0.9, "carbs": 10, "fat": 0.2, "fiber": 2.8, "sugar": 4.7, "sodium": 69, "source": "default"},
            "tomato": {"calories": 18, "protein": 0.9, "carbs": 3.9, "fat": 0.2, "fiber": 1.2, "sugar": 2.6, "sodium": 5, "source": "default"},
            "potato": {"calories": 77, "protein": 2, "carbs": 17, "fat": 0.1, "fiber": 2.2, "sugar": 0.8, "sodium": 6, "source": "default"},
            "spinach": {"calories": 23, "protein": 2.9, "carbs": 3.6, "fat": 0.4, "fiber": 2.2, "sugar": 0.4, "sodium": 79, "source": "default"},
            
            # Proteins
            "chicken": {"calories": 239, "protein": 27, "carbs": 0, "fat": 14, "fiber": 0, "sugar": 0, "sodium": 82, "source": "default"},
            "beef": {"calories": 250, "protein": 26, "carbs": 0, "fat": 15, "fiber": 0, "sugar": 0, "sodium": 72, "source": "default"},
            "fish": {"calories": 206, "protein": 22, "carbs": 0, "fat": 12, "fiber": 0, "sugar": 0, "sodium": 59, "source": "default"},
            "egg": {"calories": 155, "protein": 13, "carbs": 1.1, "fat": 11, "fiber": 0, "sugar": 1.1, "sodium": 124, "source": "default"},
            "tofu": {"calories": 76, "protein": 8, "carbs": 1.9, "fat": 4.8, "fiber": 0.3, "sugar": 0.6, "sodium": 7, "source": "default"},
            
            # Grains & Starches
            "rice": {"calories": 130, "protein": 2.7, "carbs": 28, "fat": 0.3, "fiber": 0.4, "sugar": 0.1, "sodium": 5, "source": "default"},
            "pasta": {"calories": 220, "protein": 8, "carbs": 44, "fat": 1.5, "fiber": 2.5, "sugar": 2.7, "sodium": 6, "source": "default"},
            "bread": {"calories": 265, "protein": 9, "carbs": 49, "fat": 3.2, "fiber": 2.7, "sugar": 5.7, "sodium": 681, "source": "default"},
            "quinoa": {"calories": 120, "protein": 4.4, "carbs": 22, "fat": 1.9, "fiber": 2.8, "sugar": 0.9, "sodium": 7, "source": "default"},
            "oats": {"calories": 68, "protein": 2.4, "carbs": 12, "fat": 1.4, "fiber": 1.7, "sugar": 0.3, "sodium": 49, "source": "default"},
            
            # Fast Food
            "pizza": {"calories": 266, "protein": 11, "carbs": 33, "fat": 10, "fiber": 2.3, "sugar": 3.6, "sodium": 598, "source": "default"},
            "burger": {"calories": 295, "protein": 17, "carbs": 28, "fat": 14, "fiber": 2.1, "sugar": 4.0, "sodium": 396, "source": "default"},
            "hot_dog": {"calories": 290, "protein": 10, "carbs": 4, "fat": 26, "fiber": 0.1, "sugar": 1.2, "sodium": 1090, "source": "default"},
            "french_fries": {"calories": 365, "protein": 4, "carbs": 63, "fat": 17, "fiber": 3.8, "sugar": 0.3, "sodium": 246, "source": "default"},
            "sandwich": {"calories": 250, "protein": 12, "carbs": 30, "fat": 8, "fiber": 2.0, "sugar": 3.5, "sodium": 450, "source": "default"},
            
            # Desserts
            "ice_cream": {"calories": 207, "protein": 3.5, "carbs": 24, "fat": 11, "fiber": 0.7, "sugar": 21, "sodium": 80, "source": "default"},
            "cake": {"calories": 257, "protein": 4, "carbs": 46, "fat": 7, "fiber": 1.2, "sugar": 35, "sodium": 242, "source": "default"},
            "cookie": {"calories": 502, "protein": 5.9, "carbs": 64, "fat": 24, "fiber": 2.4, "sugar": 40, "sodium": 386, "source": "default"},
            "donut": {"calories": 452, "protein": 5, "carbs": 51, "fat": 25, "fiber": 1.7, "sugar": 23, "sodium": 375, "source": "default"},
            "chocolate": {"calories": 546, "protein": 4.9, "carbs": 61, "fat": 31, "fiber": 7, "sugar": 48, "sodium": 24, "source": "default"},
            
            # Beverages
            "milk": {"calories": 42, "protein": 3.4, "carbs": 5, "fat": 1, "fiber": 0, "sugar": 5, "sodium": 44, "source": "default"},
            "coffee": {"calories": 2, "protein": 0.3, "carbs": 0, "fat": 0, "fiber": 0, "sugar": 0, "sodium": 5, "source": "default"},
            "tea": {"calories": 1, "protein": 0, "carbs": 0.3, "fat": 0, "fiber": 0, "sugar": 0, "sodium": 3, "source": "default"},
            "soda": {"calories": 41, "protein": 0, "carbs": 10.6, "fat": 0, "fiber": 0, "sugar": 10.6, "sodium": 6, "source": "default"},
            
            # Dairy
            "cheese": {"calories": 402, "protein": 25, "carbs": 1.3, "fat": 33, "fiber": 0, "sugar": 0.5, "sodium": 653, "source": "default"},
            "yogurt": {"calories": 59, "protein": 10, "carbs": 3.6, "fat": 0.4, "fiber": 0, "sugar": 3.2, "sodium": 36, "source": "default"},
            "butter": {"calories": 717, "protein": 0.9, "carbs": 0.1, "fat": 81, "fiber": 0, "sugar": 0.1, "sodium": 643, "source": "default"},
            
            # Nuts & Seeds
            "almonds": {"calories": 579, "protein": 21, "carbs": 22, "fat": 50, "fiber": 12, "sugar": 4.4, "sodium": 1, "source": "default"},
            "walnuts": {"calories": 654, "protein": 15, "carbs": 14, "fat": 65, "fiber": 6.7, "sugar": 2.6, "sodium": 2, "source": "default"},
            "peanuts": {"calories": 567, "protein": 26, "carbs": 16, "fat": 49, "fiber": 8.5, "sugar": 4.7, "sodium": 18, "source": "default"}
        },
        "learning_data": {
            "corrections": [],
            "user_confirmations": [],
            "confidence_feedback": [],
            "total_corrections": 0,
            "accuracy_history": [],
            "last_updated": datetime.now().isoformat()
        },
        "statistics": {
            "total_predictions": 0,
            "correct_predictions": 0,
            "accuracy_rate": 0.0,
            "high_confidence_accuracy": 0.0,
            "low_confidence_accuracy": 0.0,
            "model_performance": {
                "resnet50": {"predictions": 0, "accuracy": 0.0},
                "efficientnet": {"predictions": 0, "accuracy": 0.0},
                "inception": {"predictions": 0, "accuracy": 0.0}
            }
        },
        "web_scraping": {
            "total_searches": 0,
            "successful_searches": 0,
            "cache_hits": 0,
            "last_scraping_update": datetime.now().isoformat(),
            "sources_used": {
                "google_search": 0,
                "usda_database": 0,
                "openfoodfacts": 0
            }
        },
        "system_config": {
            "version": "2.0.0",
            "created_date": datetime.now().isoformat(),
            "features": {
                "multi_model_detection": True,
                "web_nutrition_scraping": True,
                "advanced_feedback": True,
                "real_time_analytics": True,
                "camera_detection": True
            }
        }
    }
    
    try:
        with open(db_file, 'w', encoding='utf-8') as f:
            json.dump(enhanced_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Enhanced database created with {len(enhanced_data['foods'])} foods")
        print(f"📊 Features enabled: Multi-model AI, Web scraping, Advanced analytics")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create database: {e}")
        return False

def test_web_connectivity():
    """Test internet connectivity for web scraping"""
    print("\n🌐 Testing web connectivity...")
    
    test_urls = [
        "https://www.google.com",
        "https://world.openfoodfacts.org",
        "https://httpbin.org/json"
    ]
    
    working_connections = 0
    
    for url in test_urls:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ Reachable: {url}")
                working_connections += 1
            else:
                print(f"⚠️ {url} returned status {response.status_code}")
        except Exception as e:
            print(f"❌ Failed to reach {url}: {e}")

    # Summary
    if working_connections == len(test_urls):
        print(f"\n✅ All {len(test_urls)} test URLs are reachable.")
        return True
    elif working_connections > 0:
        print(f"\n⚠️ Partial connectivity: {working_connections}/{len(test_urls)} reachable.")
        return True
    else:
        print(f"\n❌ No test URLs reachable. Web scraping features will be limited.")
        return False


def main():
    print_banner()

    ok = check_system_requirements()
    if not ok:
        print("\n❌ System requirements not met. Aborting setup.")
        return

    installed = install_packages()
    if not installed:
        print("\n⚠️ Some packages failed to install. You can retry manually.")

    tf_ok = test_tensorflow_setup()
    if not tf_ok:
        print("\n⚠️ TensorFlow test failed or missing. The app may still run with CPU or if you install TensorFlow later.")

    db_ok = create_enhanced_database()
    web_ok = test_web_connectivity()

    print("\n🎉 Setup summary:")
    print(f"  - Packages installed: {'yes' if installed else 'partial/no'}")
    print(f"  - TensorFlow ok: {'yes' if tf_ok else 'no'}")
    print(f"  - Database created: {'yes' if db_ok else 'no'}")
    print(f"  - Web connectivity: {'yes' if web_ok else 'no'}")
    print("\n✅ Setup finished. You can now run the detector scripts.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nSetup interrupted by user.")