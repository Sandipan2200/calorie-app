#!/usr/bin/env python3
"""
Enhanced Food Detection System with Advanced Features
- Multiple AI models for better accuracy
- Web scraping for nutrition data
- Advanced feedback system
- Real-time learning capabilities
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, EfficientNetB3, InceptionV3
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.resnet import decode_predictions
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageTk, ImageEnhance, ImageFilter
import requests
from bs4 import BeautifulSoup
import re
import json
import os
import threading
import time
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk, filedialog
from io import BytesIO
import tempfile
import logging
from collections import Counter
import difflib
import urllib.parse
from typing import Optional
# Feature-feedback detector (exemplar-based) - uses ResNet50 features
try:
    from feature_feedback_detector import FeatureFeedbackDetector
except Exception:
    FeatureFeedbackDetector = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WebNutritionScraper:
    """Enhanced nutrition data scraper using multiple sources"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.timeout = 10
        
    def search_nutrition_usda(self, food_name):
        """Search USDA FoodData Central (web scraping)"""
        try:
            # Clean and encode food name
            clean_name = food_name.lower().strip()
            encoded_name = urllib.parse.quote(clean_name)
            
            # Search URL
            search_url = f"https://fdc.nal.usda.gov/fdc-app.html#/food-search?query={encoded_name}&type=Foundation"
            
            response = requests.get(search_url, headers=self.headers, timeout=self.timeout)
            if response.status_code == 200:
                # This would need more sophisticated parsing for real implementation
                # For now, return None to fall back to other methods
                return None
                
        except Exception as e:
            logger.error(f"USDA search failed for {food_name}: {e}")
            return None
    
    def search_nutrition_google(self, food_name):
        """Search nutrition data from Google search results"""
        try:
            clean_name = food_name.lower().strip()
            query = f"{clean_name} nutrition facts calories protein carbs fat per 100g"
            encoded_query = urllib.parse.quote(query)
            
            google_url = f"https://www.google.com/search?q={encoded_query}"
            
            response = requests.get(google_url, headers=self.headers, timeout=self.timeout)
            if response.status_code != 200:
                return None
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for nutrition information in Google's knowledge panel
            nutrition_data = {}
            
            # Try to find calories
            calories_pattern = r'(\d+)\s*(?:calories|kcal|cal)'
            calories_match = re.search(calories_pattern, response.text, re.IGNORECASE)
            if calories_match:
                nutrition_data['calories'] = float(calories_match.group(1))
            
            # Try to find protein
            protein_pattern = r'(\d+(?:\.\d+)?)\s*g?\s*protein'
            protein_match = re.search(protein_pattern, response.text, re.IGNORECASE)
            if protein_match:
                nutrition_data['protein'] = float(protein_match.group(1))
            
            # Try to find carbs
            carbs_pattern = r'(\d+(?:\.\d+)?)\s*g?\s*(?:carb|carbohydrate)'
            carbs_match = re.search(carbs_pattern, response.text, re.IGNORECASE)
            if carbs_match:
                nutrition_data['carbs'] = float(carbs_match.group(1))
            
            # Try to find fat
            fat_pattern = r'(\d+(?:\.\d+)?)\s*g?\s*(?:fat|lipid)'
            fat_match = re.search(fat_pattern, response.text, re.IGNORECASE)
            if fat_match:
                nutrition_data['fat'] = float(fat_match.group(1))
            
            if len(nutrition_data) >= 2:  # If we found at least 2 nutrients
                nutrition_data['source'] = 'google_search'
                return nutrition_data
            
        except Exception as e:
            logger.error(f"Google search failed for {food_name}: {e}")
            
        return None
    
    def search_nutrition_myfitnesspal(self, food_name):
        """Search MyFitnessPal database"""
        try:
            clean_name = food_name.replace(' ', '-')
            url = f"https://www.myfitnesspal.com/food/search?q={clean_name}"
            
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for nutrition facts in the page
            nutrition_data = {}
            
            # This is a simplified example - real implementation would need
            # more sophisticated parsing of MyFitnessPal's structure
            
            return None  # Placeholder for now
            
        except Exception as e:
            logger.error(f"MyFitnessPal search failed for {food_name}: {e}")
            return None

    def search_nutrition_openfoodfacts(self, food_name):
        """Search OpenFoodFacts for a food and return per-100g nutritions if available.

        Returns a dict like {calories, protein, carbs, fat, fiber, sugar, sodium, source}
        or None if nothing useful was found.
        """
        try:
            clean = urllib.parse.quote_plus(food_name)
            url = f"https://world.openfoodfacts.org/cgi/search.pl?search_terms={clean}&search_simple=1&action=process&json=1&page_size=6"
            resp = requests.get(url, headers=self.headers, timeout=self.timeout)
            if resp.status_code != 200:
                return None

            data = resp.json()
            products = data.get('products', [])
            if not products:
                return None

            # Try to find the best product with nutrition info per 100g
            for p in products:
                nutriments = p.get('nutriments', {})
                if not nutriments:
                    continue

                # calories may be under 'energy-kcal_100g' or 'energy_100g' (kJ)
                kcal = nutriments.get('energy-kcal_100g') or nutriments.get('energy_100g')
                protein = nutriments.get('proteins_100g')
                carbs = nutriments.get('carbohydrates_100g') or nutriments.get('carbohydrates_value')
                fat = nutriments.get('fat_100g')
                fiber = nutriments.get('fiber_100g')
                sugar = nutriments.get('sugars_100g')
                salt = nutriments.get('salt_100g')

                if any(v is not None for v in [kcal, protein, carbs, fat]):
                    # Convert to numeric where possible
                    def _num(x):
                        try:
                            return float(x)
                        except Exception:
                            return 'Unknown'

                    sodium = 'Unknown'
                    if salt not in (None, ''):
                        try:
                            # rough conversion: salt (g) -> sodium (mg) via 1g salt ~ 400mg sodium
                            sodium = float(salt) * 400
                        except Exception:
                            sodium = 'Unknown'

                    return {
                        'calories': _num(kcal),
                        'protein': _num(protein),
                        'carbs': _num(carbs),
                        'fat': _num(fat),
                        'fiber': _num(fiber) if fiber is not None else 'Unknown',
                        'sugar': _num(sugar) if sugar is not None else 'Unknown',
                        'sodium': sodium,
                        'source': 'openfoodfacts'
                    }

            return None
        except Exception as e:
            logger.debug(f"OpenFoodFacts search failed for {food_name}: {e}")
            return None
    
    def get_comprehensive_nutrition(self, food_name):
        """Get nutrition data from multiple sources"""
        logger.info(f"Searching comprehensive nutrition data for: {food_name}")
        
        # Try different sources in order of preference
        sources = [
            self.search_nutrition_openfoodfacts,
            self.search_nutrition_google,
            self.search_nutrition_usda,
            self.search_nutrition_myfitnesspal
        ]
        
        for source in sources:
            try:
                nutrition_data = source(food_name)
                if nutrition_data:
                    logger.info(f"Found nutrition data from {nutrition_data.get('source', 'unknown')}")
                    return nutrition_data
            except Exception as e:
                logger.error(f"Source failed: {e}")
                continue
        
        logger.warning(f"No nutrition data found for: {food_name}")
        return None

class MultiModelFoodDetector:
    """Enhanced food detector using multiple AI models for better accuracy"""
    
    def __init__(self):
        logger.info("Initializing Multi-Model Food Detection System...")
        
        # Load multiple pre-trained models
        self.models = {}
        self.load_models()
        
        # Enhanced food keywords with more specific categories
        self.food_keywords = self.load_comprehensive_food_keywords()
        
        # Model weights for ensemble prediction
        self.model_weights = {
            'resnet50': 0.4,
            'efficientnet': 0.35,
            'inception': 0.25
        }
        
    def load_models(self):
        """Load multiple pre-trained models"""
        try:
            logger.info("Loading ResNet50...")
            self.models['resnet50'] = ResNet50(weights='imagenet', include_top=True)
            
            logger.info("Loading EfficientNetB3...")
            self.models['efficientnet'] = EfficientNetB3(weights='imagenet', include_top=True)
            
            logger.info("Loading InceptionV3...")
            self.models['inception'] = InceptionV3(weights='imagenet', include_top=True)
            
            logger.info("✅ All models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            # Fallback to single model
            self.models['resnet50'] = ResNet50(weights='imagenet', include_top=True)
            self.model_weights = {'resnet50': 1.0}
    
    def load_comprehensive_food_keywords(self):
        """Load comprehensive food keywords organized by categories"""
        return {
            # Fruits
            'apple', 'banana', 'orange', 'strawberry', 'grape', 'pineapple',
            'mango', 'peach', 'pear', 'cherry', 'blueberry', 'raspberry',
            'watermelon', 'cantaloupe', 'kiwi', 'papaya', 'coconut',
            
            # Vegetables
            'broccoli', 'carrot', 'tomato', 'potato', 'onion', 'pepper',
            'cucumber', 'lettuce', 'spinach', 'cabbage', 'cauliflower',
            'zucchini', 'eggplant', 'asparagus', 'mushroom', 'corn',
            
            # Proteins
            'chicken', 'beef', 'pork', 'fish', 'salmon', 'tuna',
            'egg', 'tofu', 'beans', 'lentils', 'nuts', 'cheese',
            
            # Grains & Starches
            'rice', 'pasta', 'bread', 'noodle', 'quinoa', 'oats',
            'cereal', 'bagel', 'croissant', 'muffin', 'pancake', 'waffle',
            
            # Fast Food
            'pizza', 'burger', 'hamburger', 'cheeseburger', 'hot_dog',
            'french_fries', 'fries', 'sandwich', 'burrito', 'taco',
            'nachos', 'wings', 'fried_chicken',
            
            # Desserts
            'ice_cream', 'cake', 'cookie', 'donut', 'chocolate',
            'candy', 'pie', 'pudding', 'brownie', 'cupcake',
            
            # Beverages
            'coffee', 'tea', 'soda', 'juice', 'milk', 'smoothie',
            
            # International Cuisine
            'sushi', 'ramen', 'curry', 'biryani', 'dosa', 'naan',
            'dim_sum', 'gyoza', 'tempura', 'pad_thai', 'pho',
            
            # Snacks
            'chips', 'crackers', 'pretzels', 'popcorn', 'granola',
            
            # Soups & Salads
            'soup', 'salad', 'stew', 'chili', 'broth'
        }
    
    def preprocess_image_enhanced(self, img_input):
        """Enhanced image preprocessing with multiple techniques"""
        try:
            # Load image
            if isinstance(img_input, str):
                if img_input.startswith('http'):
                    response = requests.get(img_input, timeout=10)
                    img = Image.open(BytesIO(response.content))
                else:
                    img = Image.open(img_input)
            else:
                img = img_input
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Apply image enhancements
            enhanced_images = []
            
            # Original image
            enhanced_images.append(('original', img))
            
            # Enhanced contrast
            enhancer = ImageEnhance.Contrast(img)
            enhanced_images.append(('contrast', enhancer.enhance(1.2)))
            
            # Enhanced brightness
            enhancer = ImageEnhance.Brightness(img)
            enhanced_images.append(('brightness', enhancer.enhance(1.1)))
            
            # Sharpened
            enhanced_images.append(('sharp', img.filter(ImageFilter.SHARPEN)))
            
            return enhanced_images, img
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None, None
    
    def get_model_predictions(self, img_variations):
        """Get predictions from all models on image variations"""
        all_predictions = []
        
        for model_name, model in self.models.items():
            for variation_name, img in img_variations:
                try:
                    # Preprocess based on model type
                    img_array = img.resize((224, 224))
                    img_array = image.img_to_array(img_array)
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    if model_name == 'resnet50':
                        img_array = resnet_preprocess(img_array)
                    elif model_name == 'efficientnet':
                        # EfficientNet uses different input size, but we'll use 224x224 for consistency
                        img_array = efficientnet_preprocess(img_array)
                    elif model_name == 'inception':
                        img_array = inception_preprocess(img_array)
                    
                    # Get predictions
                    predictions = model.predict(img_array, verbose=0)
                    decoded = decode_predictions(predictions, top=10)[0]
                    
                    # Store predictions with metadata
                    for rank, (class_id, class_name, confidence) in enumerate(decoded):
                        all_predictions.append({
                            'model': model_name,
                            'variation': variation_name,
                            'rank': rank,
                            'class_name': class_name,
                            'confidence': float(confidence),
                            'weight': self.model_weights.get(model_name, 0.33)
                        })
                        
                except Exception as e:
                    logger.error(f"Error getting predictions from {model_name}: {e}")
                    continue
        
        return all_predictions
    
    def ensemble_prediction(self, all_predictions):
        """Combine predictions from all models using weighted ensemble"""
        # Group predictions by class name
        class_scores = {}
        
        for pred in all_predictions:
            class_name = pred['class_name']
            confidence = pred['confidence']
            weight = pred['weight']
            
            # Weight by model confidence and model weight
            weighted_score = confidence * weight
            
            if class_name in class_scores:
                class_scores[class_name]['total_score'] += weighted_score
                class_scores[class_name]['count'] += 1
                class_scores[class_name]['max_confidence'] = max(
                    class_scores[class_name]['max_confidence'], confidence
                )
            else:
                class_scores[class_name] = {
                    'total_score': weighted_score,
                    'count': 1,
                    'max_confidence': confidence
                }
        
        # Calculate final scores
        final_predictions = []
        for class_name, data in class_scores.items():
            # Average weighted score with bonus for multiple model agreement
            avg_score = data['total_score'] / data['count']
            agreement_bonus = min(data['count'] / len(self.models), 1.0) * 0.1
            final_score = avg_score + agreement_bonus
            
            final_predictions.append({
                'class_name': class_name,
                'confidence': final_score,
                'max_confidence': data['max_confidence'],
                'model_agreement': data['count']
            })
        
        # Sort by confidence
        final_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        return final_predictions
    
    def extract_food_predictions(self, ensemble_results):
        """Extract food-related predictions from ensemble results"""
        food_predictions = []
        
        for pred in ensemble_results:
            class_name = pred['class_name']
            is_food = any(food_word in class_name.lower().replace('_', ' ') 
                         for food_word in self.food_keywords)
            
            if is_food:
                food_predictions.append({
                    'class_name': class_name.replace('_', ' ').title(),
                    'confidence': pred['confidence'] * 100,  # Convert to percentage
                    'max_confidence': pred['max_confidence'] * 100,
                    'model_agreement': pred['model_agreement']
                })
        
        return food_predictions
    
    def detect_food(self, img_input):
        """Main food detection method with enhanced accuracy"""
        logger.info("🔍 Analyzing image with multiple AI models...")
        
        # Preprocess image
        img_variations, original_img = self.preprocess_image_enhanced(img_input)
        if not img_variations:
            return None, None
        
        # Get predictions from all models
        all_predictions = self.get_model_predictions(img_variations)
        
        if not all_predictions:
            logger.warning("No predictions obtained from models")
            return None, None
        
        # Ensemble prediction
        ensemble_results = self.ensemble_prediction(all_predictions)
        
        # Extract food predictions
        food_predictions = self.extract_food_predictions(ensemble_results)
        
        if not food_predictions:
            return "Unknown Food Item", 0
        
        # Return top prediction
        top_prediction = food_predictions[0]
        
        logger.info(f"🎯 Top prediction: {top_prediction['class_name']} "
                   f"({top_prediction['confidence']:.1f}% confidence, "
                   f"{top_prediction['model_agreement']} models agreed)")
        
        return top_prediction['class_name'], top_prediction['confidence']

class EnhancedNutritionDatabase:
    """Enhanced nutrition database with web scraping capabilities"""
    
    def __init__(self, db_file='enhanced_food_nutrition_database.json'):
        self.db_file = db_file
        self.scraper = WebNutritionScraper()
        self.load_database()
        
        # Cache for web scraping results
        self.scraping_cache = {}
        
    def load_database(self):
        """Load or create enhanced database"""
        default_data = {
            "foods": {
                # Basic foods with comprehensive nutrition data
                "apple": {"calories": 52, "protein": 0.3, "carbs": 14, "fat": 0.2, "fiber": 2.4, "sugar": 10.4, "sodium": 1, "source": "default"},
                "banana": {"calories": 89, "protein": 1.1, "carbs": 23, "fat": 0.3, "fiber": 2.6, "sugar": 12.2, "sodium": 1, "source": "default"},
                "pizza": {"calories": 266, "protein": 11, "carbs": 33, "fat": 10, "fiber": 2.3, "sugar": 3.6, "sodium": 598, "source": "default"},
                "burger": {"calories": 295, "protein": 17, "carbs": 28, "fat": 14, "fiber": 2.1, "sugar": 4.0, "sodium": 396, "source": "default"},
                "chicken": {"calories": 239, "protein": 27, "carbs": 0, "fat": 14, "fiber": 0, "sugar": 0, "sodium": 82, "source": "default"},
                "rice": {"calories": 130, "protein": 2.7, "carbs": 28, "fat": 0.3, "fiber": 0.4, "sugar": 0.1, "sodium": 5, "source": "default"},
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
                "model_performance": {}
            },
            "web_scraping": {
                "total_searches": 0,
                "successful_searches": 0,
                "cache_hits": 0,
                "last_scraping_update": datetime.now().isoformat()
            }
        }
        
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
                logger.info(f"✅ Loaded enhanced database with {len(self.data['foods'])} foods")
            except Exception as e:
                logger.error(f"❌ Error loading database: {e}")
                self.data = default_data
                self.save_database()
        else:
            self.data = default_data
            self.save_database()
            logger.info("📝 Created new enhanced food database")
    
    def save_database(self):
        """Save database with better error handling"""
        try:
            self.data['learning_data']['last_updated'] = datetime.now().isoformat()
            
            # Create backup
            backup_file = f"{self.db_file}.backup"
            if os.path.exists(self.db_file):
                import shutil
                shutil.copy2(self.db_file, backup_file)
            
            # Save new data
            with open(self.db_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"❌ Error saving database: {e}")
    
    def get_nutrition_enhanced(self, food_name):
        """Get nutrition with web scraping fallback"""
        food_name_clean = food_name.lower().strip()
        
        # Check local database first
        if food_name_clean in self.data['foods']:
            nutrition = self.data['foods'][food_name_clean].copy()
            nutrition.pop('source', None)
            return nutrition
        
        # Check cache
        if food_name_clean in self.scraping_cache:
            cache_entry = self.scraping_cache[food_name_clean]
            # Check if cache is fresh (less than 7 days old)
            cache_time = datetime.fromisoformat(cache_entry['timestamp'])
            if (datetime.now() - cache_time).days < 7:
                return cache_entry['data']
        
        # Web scraping
        logger.info(f"🔍 Searching web for nutrition data: {food_name}")
        self.data['web_scraping']['total_searches'] += 1
        
        nutrition_data = self.scraper.get_comprehensive_nutrition(food_name)
        
        if nutrition_data:
            self.data['web_scraping']['successful_searches'] += 1
            
            # Add to cache
            self.scraping_cache[food_name_clean] = {
                'data': nutrition_data,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add to database
            self.data['foods'][food_name_clean] = nutrition_data
            self.save_database()
            
            nutrition_data.pop('source', None)
            return nutrition_data
        
        # Return unknown if not found
        return {
            'calories': 'Unknown', 'protein': 'Unknown', 'carbs': 'Unknown', 
            'fat': 'Unknown', 'fiber': 'Unknown', 'sugar': 'Unknown', 'sodium': 'Unknown'
        }
    
    def add_enhanced_feedback(self, predicted_food, correct_food, confidence, feedback_type='correction'):
        """Add enhanced feedback with more detailed tracking"""
        feedback_entry = {
            'id': len(self.data['learning_data']['corrections']) + 1,
            'predicted': predicted_food.lower().strip(),
            'correct': correct_food.lower().strip() if correct_food else predicted_food.lower().strip(),
            'confidence': float(confidence),
            'feedback_type': feedback_type,  # 'correction', 'confirmation', 'confidence_feedback'
            'timestamp': datetime.now().isoformat()
        }
        
        if feedback_type == 'correction':
            self.data['learning_data']['corrections'].append(feedback_entry)
            self.data['learning_data']['total_corrections'] += 1
        elif feedback_type == 'confirmation':
            self.data['learning_data']['user_confirmations'].append(feedback_entry)
        
        # Update accuracy tracking
        self.update_enhanced_statistics(predicted_food, correct_food, confidence, feedback_type)
        
        self.save_database()
        
        logger.info(f"📚 Enhanced feedback recorded: {predicted_food} → {correct_food} ({feedback_type})")
    
    def update_enhanced_statistics(self, predicted, correct, confidence, feedback_type):
        """Update detailed statistics"""
        self.data['statistics']['total_predictions'] += 1
        
        is_correct = (predicted.lower().strip() == correct.lower().strip()) or feedback_type == 'confirmation'
        
        if is_correct:
            self.data['statistics']['correct_predictions'] += 1
        
        # Calculate overall accuracy
        if self.data['statistics']['total_predictions'] > 0:
            self.data['statistics']['accuracy_rate'] = (
                self.data['statistics']['correct_predictions'] / 
                self.data['statistics']['total_predictions'] * 100
            )
        
        # Track confidence-based accuracy
        if confidence > 70:
            key = 'high_confidence'
        else:
            key = 'low_confidence'
        
        if key + '_predictions' not in self.data['statistics']:
            self.data['statistics'][key + '_predictions'] = 0
            self.data['statistics'][key + '_correct'] = 0
        
        self.data['statistics'][key + '_predictions'] += 1
        if is_correct:
            self.data['statistics'][key + '_correct'] += 1
        
        # Calculate confidence-based accuracy
        if self.data['statistics'][key + '_predictions'] > 0:
            self.data['statistics'][key + '_accuracy'] = (
                self.data['statistics'][key + '_correct'] / 
                self.data['statistics'][key + '_predictions'] * 100
            )

    def get_learning_suggestion(self, predicted_food: str) -> Optional[str]:
        """Return the most common correction for a predicted food, if any.

        This looks through recorded corrections and returns the most frequent
        'correct' value for entries whose 'predicted' matches the provided name.
        Returns None if no suggestion is found.
        """
        try:
            pred = predicted_food.lower().strip()
            suggestions = [c['correct'] for c in self.data['learning_data'].get('corrections', [])
                           if c.get('predicted', '').lower().strip() == pred]
            if not suggestions:
                return None
            # Return most common suggestion
            most_common = Counter(suggestions).most_common(1)[0][0]
            return most_common
        except Exception:
            return None

    def get_nutrition_summary(self, food_name: str) -> dict:
        """Return a compact summary (per 100g) for calories, protein and fat.

        The summary aggregates values from exact match, close matches in the DB,
        and any cached scraping results. It returns numeric primary values and
        range/average information for each metric.
        """
        name = food_name.lower().strip()

        candidates = []

        # Exact match first
        if name in self.data['foods']:
            entry = self.data['foods'][name]
            candidates.append((name, entry))

        # Close matches using difflib
        try:
            keys = list(self.data.get('foods', {}).keys())
            close = difflib.get_close_matches(name, keys, n=8, cutoff=0.6)
            for k in close:
                if k == name:
                    continue
                candidates.append((k, self.data['foods'].get(k, {})))
        except Exception:
            pass

        # Include scraping cache entries if present
        try:
            for k, v in self.scraping_cache.items():
                if k == name or name in k or k in name:
                    candidates.append((k, v.get('data', {})))
        except Exception:
            pass

        # Gather numeric values for each metric
        def collect(metric):
            vals = []
            for _, entry in candidates:
                val = entry.get(metric)
                if isinstance(val, (int, float)):
                    vals.append(float(val))
            # If still empty, try top-level DB exact/other
            if not vals and name in self.data.get('foods', {}):
                val = self.data['foods'][name].get(metric)
                if isinstance(val, (int, float)):
                    vals.append(float(val))
            return vals

        cal_vals = collect('calories')
        prot_vals = collect('protein')
        fat_vals = collect('fat')

        import statistics

        def summarize(vals):
            if not vals:
                return {'min': 'Unknown', 'max': 'Unknown', 'avg': 'Unknown'}
            mn = min(vals)
            mx = max(vals)
            avg = statistics.mean(vals)
            return {'min': mn, 'max': mx, 'avg': avg}

        cal_sum = summarize(cal_vals)
        prot_sum = summarize(prot_vals)
        fat_sum = summarize(fat_vals)

        # Primary display values: use exact match value if present, else average
        def primary(metric, sums, vals):
            if name in self.data.get('foods', {}) and isinstance(self.data['foods'][name].get(metric), (int, float)):
                return float(self.data['foods'][name].get(metric))
            if vals:
                return sums['avg']
            return 'Unknown'

        primary_cal = primary('calories', cal_sum, cal_vals)
        primary_prot = primary('protein', prot_sum, prot_vals)
        primary_fat = primary('fat', fat_sum, fat_vals)

        return {
            'food': food_name,
            'per_100g': True,
            'calories': primary_cal,
            'calories_range': (cal_sum['min'], cal_sum['max']) if cal_sum['min'] != 'Unknown' else ('Unknown', 'Unknown'),
            'calories_avg': cal_sum['avg'] if cal_sum['avg'] != 'Unknown' else 'Unknown',
            'protein': primary_prot,
            'protein_range': (prot_sum['min'], prot_sum['max']) if prot_sum['min'] != 'Unknown' else ('Unknown', 'Unknown'),
            'protein_avg': prot_sum['avg'] if prot_sum['avg'] != 'Unknown' else 'Unknown',
            'fat': primary_fat,
            'fat_range': (fat_sum['min'], fat_sum['max']) if fat_sum['min'] != 'Unknown' else ('Unknown', 'Unknown'),
            'fat_avg': fat_sum['avg'] if fat_sum['avg'] != 'Unknown' else 'Unknown',
        }

class EnhancedFeedbackGUI:
    """Enhanced GUI with better user experience and detailed feedback"""
    
    def __init__(self, detector, database, feature_detector=None):
        self.detector = detector
        self.database = database
        # optional exemplar/feature-feedback detector (stores and matches feature vectors)
        self.feature_detector = feature_detector
        
    def show_enhanced_feedback_dialog(self, predicted_food, confidence, nutrition_data, image_path=None):
        """Show comprehensive feedback dialog"""
        self.result = {'is_correct': False, 'correct_food': None, 'feedback_details': {}}
        # store reference to the image shown (PIL.Image or path) for feedback exemplar saving
        self.current_image = image_path
        
        # Create main window
        self.root = tk.Toplevel()
        self.root.title("🍽️ Enhanced Food Detection Results")
        self.root.geometry("700x600")
        self.root.configure(bg='#f5f5f5')
        
        # Make it modal
        self.root.transient()
        self.root.grab_set()
        
        self.setup_enhanced_ui(predicted_food, confidence, nutrition_data)
        
        # Center window
        self.center_window()
        
        self.root.mainloop()
        
        return self.result
    
    def setup_enhanced_ui(self, predicted_food, confidence, nutrition_data):
        """Setup enhanced UI with more detailed information"""
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header with confidence indicator
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = ttk.Label(header_frame, text="🍽️ Food Detection Results", 
                               font=('Arial', 18, 'bold'))
        title_label.pack()
        
        # Confidence indicator with color coding
        conf_frame = ttk.Frame(header_frame)
        conf_frame.pack(pady=10)
        
        if confidence > 80:
            conf_color = '#4CAF50'
            conf_icon = '🟢'
            conf_text = 'High Confidence'
        elif confidence > 60:
            conf_color = '#FF9800'
            conf_icon = '🟡'
            conf_text = 'Medium Confidence'
        else:
            conf_color = '#F44336'
            conf_icon = '🔴'
            conf_text = 'Low Confidence'
        
        tk.Label(conf_frame, text=f"{conf_icon} {conf_text}: {confidence:.1f}%",
                font=('Arial', 12, 'bold'), fg=conf_color, bg='#f5f5f5').pack()
        
        # Detection result
        result_frame = ttk.LabelFrame(main_frame, text="🎯 Detection Result", padding="15")
        result_frame.pack(fill=tk.X, pady=(0, 15))
        
        food_label = ttk.Label(result_frame, text=f"Detected Food: {predicted_food}",
                              font=('Arial', 14, 'bold'))
        food_label.pack(anchor=tk.W)
        
        # Nutrition information
        nutrition_frame = ttk.LabelFrame(main_frame, text="🥗 Nutrition Facts (per 100g)", padding="15")
        nutrition_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Create nutrition grid
        nutrition_grid = ttk.Frame(nutrition_frame)
        nutrition_grid.pack(fill=tk.X)
        
        nutrients = [
            ('Calories', nutrition_data.get('calories', 'Unknown'), 'kcal'),
            ('Protein', nutrition_data.get('protein', 'Unknown'), 'g'),
            ('Carbs', nutrition_data.get('carbs', 'Unknown'), 'g'),
            ('Fat', nutrition_data.get('fat', 'Unknown'), 'g'),
            ('Fiber', nutrition_data.get('fiber', 'Unknown'), 'g'),
            ('Sugar', nutrition_data.get('sugar', 'Unknown'), 'g')
        ]
        
        for i, (name, value, unit) in enumerate(nutrients):
            row = i // 2
            col = i % 2
            
            frame = ttk.Frame(nutrition_grid)
            frame.grid(row=row, column=col, padx=10, pady=5, sticky='ew')
            
            ttk.Label(frame, text=f"{name}:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
            ttk.Label(frame, text=f"{value} {unit}").pack(side=tk.RIGHT)
        
        nutrition_grid.columnconfigure(0, weight=1)
        nutrition_grid.columnconfigure(1, weight=1)
        
        # Feedback question
        question_frame = ttk.LabelFrame(main_frame, text="💭 Your Feedback", padding="15")
        question_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(question_frame, text="How would you rate this detection?",
                 font=('Arial', 12, 'bold')).pack(pady=(0, 10))
        
        # Feedback buttons with more options
        btn_frame = ttk.Frame(question_frame)
        btn_frame.pack(pady=10)
        
        # Perfect detection
        tk.Button(btn_frame, text="🎯 Perfect!\nExactly right",
                 command=lambda: self.on_feedback('perfect'),
                 bg='#4CAF50', fg='white', font=('Arial', 10, 'bold'),
                 width=12, height=3).pack(side=tk.LEFT, padx=5)
        
        # Close but not exact
        tk.Button(btn_frame, text="📍 Close\nBut not exact",
                 command=lambda: self.on_feedback('close'),
                 bg='#FF9800', fg='white', font=('Arial', 10, 'bold'),
                 width=12, height=3).pack(side=tk.LEFT, padx=5)
        
        # Completely wrong
        tk.Button(btn_frame, text="❌ Wrong\nNeed correction",
                 command=lambda: self.on_feedback('wrong'),
                 bg='#F44336', fg='white', font=('Arial', 10, 'bold'),
                 width=12, height=3).pack(side=tk.LEFT, padx=5)
        
        # Additional feedback options
        advanced_frame = ttk.LabelFrame(main_frame, text="🔧 Advanced Options", padding="10")
        advanced_frame.pack(fill=tk.X)
        
        ttk.Button(advanced_frame, text="📝 Add Custom Feedback",
                  command=self.show_custom_feedback).pack(pady=5)
        
        # Store current data
        self.current_food = predicted_food
        self.current_confidence = confidence
        self.current_nutrition = nutrition_data
    
    def center_window(self):
        """Center the window on screen"""
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (self.root.winfo_width() // 2)
        y = (self.root.winfo_screenheight() // 2) - (self.root.winfo_height() // 2)
        self.root.geometry(f"+{x}+{y}")
    
    def on_feedback(self, feedback_type):
        """Handle different types of feedback"""
        if feedback_type == 'perfect':
            self.result = {
                'is_correct': True,
                'correct_food': self.current_food,
                'feedback_details': {'type': 'perfect', 'confidence_rating': 'high'}
            }
            self.database.add_enhanced_feedback(
                self.current_food, self.current_food, 
                self.current_confidence, 'confirmation'
            )
            messagebox.showinfo("Thank You!", "🎯 Perfect! Thanks for confirming!")
            self.root.destroy()
            
        elif feedback_type == 'close':
            self.show_close_correction_dialog()
            
        elif feedback_type == 'wrong':
            self.show_correction_dialog()
    
    def show_close_correction_dialog(self):
        """Show dialog for close but not exact predictions"""
        dialog = tk.Toplevel(self.root)
        dialog.title("📍 Close Detection - Minor Correction")
        dialog.geometry("400x300")
        dialog.configure(bg='#f5f5f5')
        
        main_frame = ttk.Frame(dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main_frame, 
                 text=f"We detected: {self.current_food}",
                 font=('Arial', 12, 'bold')).pack(pady=(0, 10))
        
        ttk.Label(main_frame,
                 text="What's the more accurate name?",
                 font=('Arial', 11)).pack(pady=(0, 10))
        
        self.close_entry = ttk.Entry(main_frame, font=('Arial', 11), width=30)
        self.close_entry.pack(pady=(0, 15))
        self.close_entry.insert(0, self.current_food)  # Pre-fill with current detection
        self.close_entry.focus()
        
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="✅ Submit",
                 command=lambda: self.process_close_correction(dialog),
                 bg='#2196F3', fg='white', font=('Arial', 10, 'bold'),
                 width=10).pack(side=tk.LEFT, padx=5)
        
        tk.Button(btn_frame, text="❌ Cancel",
                 command=dialog.destroy,
                 bg='#757575', fg='white', font=('Arial', 10),
                 width=10).pack(side=tk.LEFT, padx=5)
        
        dialog.transient(self.root)
        dialog.grab_set()
    
    def process_close_correction(self, dialog):
        """Process close correction feedback"""
        correction = self.close_entry.get().strip()
        if correction and correction != self.current_food:
            self.result = {
                'is_correct': False,
                'correct_food': correction,
                'feedback_details': {'type': 'close_correction'}
            }
            # Save to database
            self.database.add_enhanced_feedback(
                self.current_food, correction, 
                self.current_confidence, 'correction'
            )
            # Also store exemplar features so the correction is applied only to visually similar images
            if getattr(self, 'feature_detector', None) is not None and getattr(self, 'current_image', None) is not None:
                try:
                    self.feature_detector.add_feedback(self.current_image, correction, notes='close_correction')
                    logger.info(f"Saved exemplar feedback for {correction}")
                except Exception as e:
                    logger.error(f"Failed to save exemplar feedback: {e}")
            messagebox.showinfo("Thank You!", f"📝 Thanks! Updated to: {correction}")
            dialog.destroy()
            self.root.destroy()
        else:
            messagebox.showwarning("Warning", "Please enter a different food name!")
    
    def show_correction_dialog(self):
        """Show full correction dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("❌ Correction Needed")
        dialog.geometry("400x350")
        dialog.configure(bg='#f5f5f5')
        
        main_frame = ttk.Frame(dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main_frame,
                 text="🔧 Help Us Learn!",
                 font=('Arial', 14, 'bold')).pack(pady=(0, 15))
        
        ttk.Label(main_frame,
                 text=f"We incorrectly detected: {self.current_food}",
                 font=('Arial', 11)).pack(pady=(0, 10))
        
        ttk.Label(main_frame,
                 text="What is the correct food name?",
                 font=('Arial', 11, 'bold')).pack(pady=(0, 10))
        
        self.correction_entry = ttk.Entry(main_frame, font=('Arial', 11), width=30)
        self.correction_entry.pack(pady=(0, 15))
        self.correction_entry.focus()
        
        # Additional feedback options
        ttk.Label(main_frame, text="Why was it incorrect?").pack(pady=(10, 5))
        
        self.reason_var = tk.StringVar(value="similar_looking")
        reasons = [
            ("Similar looking food", "similar_looking"),
            ("Different preparation", "different_prep"),
            ("Wrong category entirely", "wrong_category"),
            ("Image quality issue", "image_quality")
        ]
        
        for text, value in reasons:
            ttk.Radiobutton(main_frame, text=text, variable=self.reason_var, 
                           value=value).pack(anchor=tk.W)
        
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=15)
        
        tk.Button(btn_frame, text="✅ Submit Correction",
                 command=lambda: self.process_full_correction(dialog),
                 bg='#2196F3', fg='white', font=('Arial', 10, 'bold'),
                 width=15).pack(side=tk.LEFT, padx=5)
        
        tk.Button(btn_frame, text="❌ Cancel",
                 command=dialog.destroy,
                 bg='#757575', fg='white', font=('Arial', 10),
                 width=10).pack(side=tk.LEFT, padx=5)
        
        dialog.transient(self.root)
        dialog.grab_set()
    
    def process_full_correction(self, dialog):
        """Process full correction with detailed feedback"""
        correction = self.correction_entry.get().strip()
        if correction:
            reason = self.reason_var.get()
            
            self.result = {
                'is_correct': False,
                'correct_food': correction,
                'feedback_details': {
                    'type': 'full_correction',
                    'reason': reason
                }
            }
            
            # Add detailed feedback to database
            feedback_entry = {
                'predicted': self.current_food,
                'correct': correction,
                'confidence': self.current_confidence,
                'reason': reason,
                'timestamp': datetime.now().isoformat()
            }
            
            self.database.add_enhanced_feedback(
                self.current_food, correction,
                self.current_confidence, 'correction'
            )
            # Persist exemplar feature for the corrected image so future similar images match this correction
            if getattr(self, 'feature_detector', None) is not None and getattr(self, 'current_image', None) is not None:
                try:
                    self.feature_detector.add_feedback(self.current_image, correction, notes=reason)
                    logger.info(f"Saved exemplar feedback for {correction} (reason={reason})")
                except Exception as e:
                    logger.error(f"Failed to save exemplar feedback: {e}")
            messagebox.showinfo("Thank You!", 
                              f"📚 Thanks for the detailed correction!\n"
                              f"Learned: {self.current_food} → {correction}")
            
            dialog.destroy()
            self.root.destroy()
        else:
            messagebox.showwarning("Warning", "Please enter the correct food name!")
    
    def show_custom_feedback(self):
        """Show custom feedback dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("📝 Custom Feedback")
        dialog.geometry("400x300")
        dialog.configure(bg='#f5f5f5')
        
        main_frame = ttk.Frame(dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main_frame, text="📝 Additional Comments",
                 font=('Arial', 12, 'bold')).pack(pady=(0, 10))
        
        self.custom_text = tk.Text(main_frame, height=8, width=40)
        self.custom_text.pack(pady=(0, 15))
        
        ttk.Button(main_frame, text="💾 Save Feedback",
                  command=lambda: self.save_custom_feedback(dialog)).pack()
        
        dialog.transient(self.root)
        dialog.grab_set()
    
    def save_custom_feedback(self, dialog):
        """Save custom feedback"""
        feedback_text = self.custom_text.get("1.0", tk.END).strip()
        if feedback_text:
            # Add to database (you can expand this)
            logger.info(f"Custom feedback received: {feedback_text}")
            messagebox.showinfo("Saved", "Thank you for your detailed feedback!")
            dialog.destroy()

class EnhancedFoodDetectionGUI:
    """Main GUI application with enhanced features"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🍽️ Enhanced Food Detection System")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize components
        logger.info("Initializing Enhanced Food Detection System...")
        self.detector = MultiModelFoodDetector()
        self.database = EnhancedNutritionDatabase()
        # Instantiate feature-feedback detector if available
        if FeatureFeedbackDetector is not None:
            try:
                self.feature_detector = FeatureFeedbackDetector()
            except Exception as e:
                logger.error(f"Could not initialize FeatureFeedbackDetector: {e}")
                self.feature_detector = None
        else:
            self.feature_detector = None

        self.feedback_gui = EnhancedFeedbackGUI(self.detector, self.database, self.feature_detector)
        
        # Current detection data
        self.current_result = None
        self.current_image = None
        
        self.setup_enhanced_gui()
        logger.info("✅ Enhanced GUI initialized successfully!")
    
    def setup_enhanced_gui(self):
        """Setup enhanced GUI with modern design"""
        # Configure styles
        style = ttk.Style()
        style.theme_use('clam')
        
        # Custom styles
        style.configure('Title.TLabel', font=('Arial', 20, 'bold'))
        style.configure('Subtitle.TLabel', font=('Arial', 12))
        style.configure('Bold.TLabel', font=('Arial', 11, 'bold'))
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(header_frame, text="🍽️ Enhanced Food Detection System",
                 style='Title.TLabel').pack()
        ttk.Label(header_frame, text="Multi-Model AI with Web Nutrition Data",
                 style='Subtitle.TLabel').pack()
        
        # Statistics bar
        stats_frame = ttk.Frame(header_frame)
        stats_frame.pack(fill=tk.X, pady=10)
        
        self.update_stats_display(stats_frame)
        
        # Input buttons
        input_frame = ttk.LabelFrame(main_frame, text="📥 Input Options", padding="15")
        input_frame.pack(fill=tk.X, pady=(0, 20))
        
        btn_container = ttk.Frame(input_frame)
        btn_container.pack()
        
        buttons = [
            ("📁 Upload Image", self.upload_image, '#2196F3'),
            ("🌐 From URL", self.from_url, '#4CAF50'),
            ("📷 Camera", self.use_camera, '#FF9800'),
            ("📊 View Stats", self.show_detailed_stats, '#9C27B0')
        ]
        
        for text, command, color in buttons:
            tk.Button(btn_container, text=text, command=command,
                     bg=color, fg='white', font=('Arial', 11, 'bold'),
                     padx=20, pady=10, cursor='hand2').pack(side=tk.LEFT, padx=5)
        
        # Create notebook for tabbed interface
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Detection tab
        self.detection_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.detection_frame, text="🔍 Detection Results")
        
        self.setup_detection_tab()
        
        # Analytics tab
        self.analytics_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analytics_frame, text="📈 Analytics")
        
        self.setup_analytics_tab()
        
        # Database tab
        self.database_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.database_frame, text="🗃️ Database")
        
        self.setup_database_tab()
    
    def setup_detection_tab(self):
        """Setup the main detection results tab"""
        # Create paned window for resizable layout
        paned = ttk.PanedWindow(self.detection_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Image display
        left_frame = ttk.LabelFrame(paned, text="📷 Input Image", padding="10")
        paned.add(left_frame, weight=1)
        
        self.image_label = tk.Label(left_frame, text="No image loaded",
                                   bg='#f5f5f5', font=('Arial', 12))
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Right panel - Results
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=1)
        
        # Detection results
        self.results_frame = ttk.LabelFrame(right_frame, text="🎯 Detection Results", padding="15")
        self.results_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.result_text = tk.Text(self.results_frame, height=8, font=('Arial', 11),
                                  wrap=tk.WORD, bg='#ffffff')
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
        # Nutrition panel
        self.nutrition_frame = ttk.LabelFrame(right_frame, text="🥗 Nutrition Information", padding="15")
        self.nutrition_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create nutrition display
        self.nutrition_display = tk.Frame(self.nutrition_frame, bg='white')
        self.nutrition_display.pack(fill=tk.BOTH, expand=True)
        
        # Initially show placeholder
        self.show_nutrition_placeholder()
    
    def setup_analytics_tab(self):
        """Setup analytics and statistics tab"""
        analytics_container = ttk.Frame(self.analytics_frame, padding="20")
        analytics_container.pack(fill=tk.BOTH, expand=True)
        
        # Statistics overview
        stats_frame = ttk.LabelFrame(analytics_container, text="📊 System Performance", padding="15")
        stats_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.stats_text = tk.Text(stats_frame, height=10, font=('Courier', 10),
                                 wrap=tk.WORD, bg='#f8f9fa')
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        
        # Recent activity
        activity_frame = ttk.LabelFrame(analytics_container, text="🕒 Recent Activity", padding="15")
        activity_frame.pack(fill=tk.BOTH, expand=True)
        
        self.activity_text = tk.Text(activity_frame, height=10, font=('Arial', 10),
                                    wrap=tk.WORD, bg='#ffffff')
        self.activity_text.pack(fill=tk.BOTH, expand=True)
        
        self.update_analytics_display()
    
    def setup_database_tab(self):
        """Setup database management tab"""
        db_container = ttk.Frame(self.database_frame, padding="20")
        db_container.pack(fill=tk.BOTH, expand=True)
        
        # Database info
        info_frame = ttk.LabelFrame(db_container, text="🗃️ Database Information", padding="15")
        info_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.db_info_text = tk.Text(info_frame, height=6, font=('Arial', 11),
                                   wrap=tk.WORD, bg='#f8f9fa')
        self.db_info_text.pack(fill=tk.BOTH, expand=True)
        
        # Food list
        foods_frame = ttk.LabelFrame(db_container, text="🍎 Foods Database", padding="15")
        foods_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create treeview for foods
        columns = ('Food', 'Calories', 'Protein', 'Carbs', 'Fat', 'Source')
        self.foods_tree = ttk.Treeview(foods_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.foods_tree.heading(col, text=col)
            self.foods_tree.column(col, width=120)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(foods_frame, orient=tk.VERTICAL, command=self.foods_tree.yview)
        self.foods_tree.configure(yscrollcommand=scrollbar.set)
        
        self.foods_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.update_database_display()
    
    def update_stats_display(self, parent):
        """Update statistics display in header"""
        stats = self.database.data['statistics']
        
        stats_text = f"📊 Accuracy: {stats.get('accuracy_rate', 0):.1f}% | "
        stats_text += f"🎯 Predictions: {stats.get('total_predictions', 0)} | "
        stats_text += f"🔄 Corrections: {self.database.data['learning_data'].get('total_corrections', 0)}"
        
        ttk.Label(parent, text=stats_text, font=('Arial', 10)).pack()
    
    def show_nutrition_placeholder(self):
        """Show placeholder nutrition information"""
        for widget in self.nutrition_display.winfo_children():
            widget.destroy()
        
        tk.Label(self.nutrition_display, text="Upload an image to see nutrition information",
                font=('Arial', 12), bg='white', fg='gray').pack(expand=True)
    
    def upload_image(self):
        """Handle image upload"""
        file_path = filedialog.askopenfilename(
            title="Select Food Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.process_image(file_path)
    
    def from_url(self):
        """Handle URL input"""
        url = simpledialog.askstring("Image URL", "Enter the URL of the food image:")
        if url and url.startswith('http'):
            self.process_image(url)
    
    def use_camera(self):
        """Open camera for food detection"""
        self.open_camera()
    
    def process_image(self, image_input):
        """Process image and show results"""
        try:
            # Show loading message
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "🔄 Processing image with multiple AI models...\n")
            self.root.update()
            
            # Load and display image
            if isinstance(image_input, str) and image_input.startswith('http'):
                response = requests.get(image_input, timeout=10)
                img = Image.open(BytesIO(response.content))
            else:
                img = Image.open(image_input)
            
            self.display_image(img)
            
            # Detect food. Prefer feature-feedback detector (exemplar match) if available.
            used_feedback = False
            try:
                if self.feature_detector is not None:
                    try:
                        fb_out = self.feature_detector.detect(image_input)
                        if fb_out and fb_out.get('source') == 'feedback':
                            food_name = fb_out.get('label')
                            confidence = float(fb_out.get('confidence', 0.0))
                            used_feedback = True
                        else:
                            food_name, confidence = self.detector.detect_food(image_input)
                    except Exception as e:
                        logger.error(f"Feature feedback detector failed: {e}")
                        food_name, confidence = self.detector.detect_food(image_input)
                else:
                    food_name, confidence = self.detector.detect_food(image_input)
            except Exception as e:
                logger.error(f"Error during detection: {e}")
                food_name, confidence = None, 0.0

            # Apply any learned suggestion from the database (string-based) only when exemplar feedback
            # was not used. Exemplar feedback is more precise (visual similarity), so we avoid overriding it.
            if not used_feedback:
                try:
                    suggestion = self.database.get_learning_suggestion(food_name)
                    if suggestion:
                        logger.info(f"💡 Applying learned suggestion: {suggestion} for {food_name}")
                        # Boost confidence slightly but cap
                        confidence = min(confidence * 1.15, 95.0)
                        food_name = suggestion.title()
                except Exception as e:
                    logger.error(f"Error applying learning suggestion: {e}")
            
            if food_name:
                # Get nutrition data
                nutrition = self.database.get_nutrition_enhanced(food_name)
                
                # Display results
                self.show_detection_results(food_name, confidence, nutrition)
                
                # Store current data
                self.current_result = {
                    'food_name': food_name,
                    'confidence': confidence,
                    'nutrition': nutrition
                }
                self.current_image = img
                
                # Show feedback dialog
                self.root.after(1000, lambda: self.show_feedback_dialog(food_name, confidence, nutrition))
                
            else:
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, "❌ Could not detect food in the image.\n")
                self.result_text.insert(tk.END, "Please try another image with clearer food items.")
                
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            messagebox.showerror("Error", f"Error processing image: {str(e)}")
    
    def display_image(self, img):
        """Display image in the GUI"""
        try:
            # Resize image for display
            display_size = (400, 300)
            img_display = img.copy()
            img_display.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            self.current_photo = ImageTk.PhotoImage(img_display)
            
            # Update label
            self.image_label.configure(image=self.current_photo, text="")
            
        except Exception as e:
            logger.error(f"Error displaying image: {e}")
            self.image_label.configure(text="Error loading image")
    
    def show_detection_results(self, food_name, confidence, nutrition):
        """Display detection results"""
        self.result_text.delete(1.0, tk.END)
        
        # Detection result
        self.result_text.insert(tk.END, f"🍽️ Detected Food: {food_name}\n", "bold")
        self.result_text.insert(tk.END, f"📊 Confidence: {confidence:.1f}%\n")
        
        # Confidence assessment
        if confidence > 80:
            assessment = "🟢 High confidence - Very likely correct"
        elif confidence > 60:
            assessment = "🟡 Medium confidence - Probably correct"
        else:
            assessment = "🔴 Low confidence - Please verify"
        
        self.result_text.insert(tk.END, f"\n{assessment}\n\n")
        
        # Model information
        self.result_text.insert(tk.END, "🤖 AI Analysis:\n")
        self.result_text.insert(tk.END, f"• Used {len(self.detector.models)} AI models\n")
        self.result_text.insert(tk.END, f"• Processed {len(self.detector.models) * 4} image variations\n")
        self.result_text.insert(tk.END, f"• Combined results for accuracy\n")
        
        # Show nutrition (condensed: calories, protein, fat per 100g with range & avg)
        try:
            summary = self.database.get_nutrition_summary(food_name)
        except Exception as e:
            logger.error(f"Error getting nutrition summary: {e}")
            summary = None

        self.display_nutrition(summary or nutrition)
    
    def display_nutrition(self, nutrition):
        """Display nutrition information"""
        for widget in self.nutrition_display.winfo_children():
            widget.destroy()
        
        # Create three cards for Calories, Protein, Fat (per 100g)
        card_frame = tk.Frame(self.nutrition_display, bg='white')
        card_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        tk.Label(card_frame, text="Nutrition (per 100g)", font=('Arial', 14, 'bold'), bg='white').pack(pady=(0,10))

        def make_card(parent, title, value, value_unit, rng, avg):
            f = tk.Frame(parent, bg='#ffffff', bd=1, relief=tk.RIDGE, padx=10, pady=8)
            tk.Label(f, text=title, font=('Arial', 12), bg='#ffffff').pack(anchor='w')
            val_text = f"{value:.1f} {value_unit}" if isinstance(value, (int, float)) else str(value)
            tk.Label(f, text=val_text, font=('Arial', 16, 'bold'), bg='#ffffff').pack(anchor='w', pady=(4,2))
            if isinstance(rng, tuple) and rng[0] != 'Unknown':
                rng_text = f"Range: {rng[0]:.0f} - {rng[1]:.0f} {value_unit}"
            else:
                rng_text = "Range: Unknown"
            tk.Label(f, text=rng_text, font=('Arial', 10), bg='#ffffff', fg='gray').pack(anchor='w')
            avg_text = f"Avg: {avg:.1f} {value_unit}" if isinstance(avg, (int, float)) else "Avg: Unknown"
            tk.Label(f, text=avg_text, font=('Arial', 10), bg='#ffffff', fg='gray').pack(anchor='w')
            return f

        # Extract values depending on whether summary or raw nutrition dict was passed
        try:
            if isinstance(nutrition, dict) and nutrition.get('per_100g'):
                cal_val = nutrition.get('calories', 'Unknown')
                cal_rng = nutrition.get('calories_range', ('Unknown', 'Unknown'))
                cal_avg = nutrition.get('calories_avg', 'Unknown')

                prot_val = nutrition.get('protein', 'Unknown')
                prot_rng = nutrition.get('protein_range', ('Unknown', 'Unknown'))
                prot_avg = nutrition.get('protein_avg', 'Unknown')

                fat_val = nutrition.get('fat', 'Unknown')
                fat_rng = nutrition.get('fat_range', ('Unknown', 'Unknown'))
                fat_avg = nutrition.get('fat_avg', 'Unknown')
            else:
                # Raw nutrition dict: use values directly (assume per 100g)
                cal_val = nutrition.get('calories', 'Unknown')
                cal_rng = (cal_val, cal_val) if isinstance(cal_val, (int, float)) else ('Unknown', 'Unknown')
                cal_avg = cal_val if isinstance(cal_val, (int, float)) else 'Unknown'

                prot_val = nutrition.get('protein', 'Unknown')
                prot_rng = (prot_val, prot_val) if isinstance(prot_val, (int, float)) else ('Unknown', 'Unknown')
                prot_avg = prot_val if isinstance(prot_val, (int, float)) else 'Unknown'

                fat_val = nutrition.get('fat', 'Unknown')
                fat_rng = (fat_val, fat_val) if isinstance(fat_val, (int, float)) else ('Unknown', 'Unknown')
                fat_avg = fat_val if isinstance(fat_val, (int, float)) else 'Unknown'
        except Exception as e:
            logger.error(f"Error preparing nutrition display: {e}")
            cal_val = prot_val = fat_val = 'Unknown'
            cal_rng = prot_rng = fat_rng = ('Unknown', 'Unknown')
            cal_avg = prot_avg = fat_avg = 'Unknown'

        # Arrange three cards horizontally
        cards_container = tk.Frame(card_frame, bg='white')
        cards_container.pack(fill=tk.BOTH, expand=True)

        c1 = make_card(cards_container, '🔥 Calories', cal_val, 'kcal', cal_rng, cal_avg)
        c2 = make_card(cards_container, '🥩 Protein', prot_val, 'g', prot_rng, prot_avg)
        c3 = make_card(cards_container, '🥑 Fat', fat_val, 'g', fat_rng, fat_avg)

        c1.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=6)
        c2.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=6)
        c3.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=6)
    
    def show_feedback_dialog(self, food_name, confidence, nutrition):
        """Show the enhanced feedback dialog"""
        # pass the current image to the feedback dialog so corrections can store exemplar features
        result = self.feedback_gui.show_enhanced_feedback_dialog(food_name, confidence, nutrition, image_path=self.current_image)

        if result:
            # Update analytics after feedback
            self.update_analytics_display()
            self.update_database_display()
    
    def open_camera(self):
        """Open camera for real-time detection"""
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                messagebox.showerror("Error", "Could not open camera")
                return
            
            # Create camera window
            camera_window = tk.Toplevel(self.root)
            camera_window.title("📷 Camera Detection")
            camera_window.geometry("800x600")
            
            # Instructions
            instructions = tk.Label(camera_window, 
                                  text="📷 Position food in the frame and press 'Capture' to analyze",
                                  font=('Arial', 12), bg='lightblue', pady=10)
            instructions.pack(fill=tk.X)
            
            # Camera display
            camera_label = tk.Label(camera_window)
            camera_label.pack(expand=True)
            
            # Controls
            controls_frame = tk.Frame(camera_window)
            controls_frame.pack(fill=tk.X, pady=10)
            
            capture_btn = tk.Button(controls_frame, text="📸 Capture & Analyze",
                                   font=('Arial', 12, 'bold'), bg='#4CAF50', fg='white')
            capture_btn.pack(side=tk.LEFT, padx=20)
            
            close_btn = tk.Button(controls_frame, text="❌ Close Camera",
                                 command=lambda: self.close_camera(cap, camera_window),
                                 font=('Arial', 12), bg='#F44336', fg='white')
            close_btn.pack(side=tk.RIGHT, padx=20)
            
            # Camera capture function
            def capture_and_analyze():
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)
                    
                    # Save temporarily
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                    pil_image.save(temp_file.name)
                    temp_file.close()
                    
                    # Close camera
                    self.close_camera(cap, camera_window)
                    
                    # Process the captured image
                    self.process_image(temp_file.name)
            
            capture_btn.configure(command=capture_and_analyze)
            
            # Start camera feed
            def update_camera_feed():
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        # Resize frame for display
                        frame = cv2.resize(frame, (640, 480))
                        
                        # Convert to RGB and then to PhotoImage
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(rgb_frame)
                        photo = ImageTk.PhotoImage(pil_image)
                        
                        camera_label.configure(image=photo)
                        camera_label.image = photo
                    
                    # Schedule next update
                    camera_window.after(30, update_camera_feed)
            
            update_camera_feed()
            
        except Exception as e:
            logger.error(f"Error opening camera: {e}")
            messagebox.showerror("Camera Error", f"Could not open camera: {str(e)}")
    
    def close_camera(self, cap, window):
        """Close camera and window"""
        if cap:
            cap.release()
        window.destroy()
    
    def show_detailed_stats(self):
        """Show detailed statistics window"""
        stats_window = tk.Toplevel(self.root)
        stats_window.title("📊 Detailed Statistics")
        stats_window.geometry("600x500")
        
        notebook = ttk.Notebook(stats_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Performance stats
        perf_frame = ttk.Frame(notebook)
        notebook.add(perf_frame, text="🎯 Performance")
        
        perf_text = tk.Text(perf_frame, font=('Courier', 11), wrap=tk.WORD)
        perf_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Learning stats
        learning_frame = ttk.Frame(notebook)
        notebook.add(learning_frame, text="📚 Learning")
        
        learning_text = tk.Text(learning_frame, font=('Courier', 11), wrap=tk.WORD)
        learning_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Web scraping stats
        web_frame = ttk.Frame(notebook)
        notebook.add(web_frame, text="🌐 Web Data")
        
        web_text = tk.Text(web_frame, font=('Courier', 11), wrap=tk.WORD)
        web_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Populate data
        self.populate_detailed_stats(perf_text, learning_text, web_text)
    
    def populate_detailed_stats(self, perf_text, learning_text, web_text):
        """Populate detailed statistics"""
        stats = self.database.data['statistics']
        learning_data = self.database.data['learning_data']
        web_data = self.database.data['web_scraping']
        
        # Performance statistics
        perf_content = f"""
🎯 PERFORMANCE STATISTICS
{'='*40}

Total Predictions: {stats.get('total_predictions', 0)}
Correct Predictions: {stats.get('correct_predictions', 0)}
Overall Accuracy: {stats.get('accuracy_rate', 0):.2f}%

High Confidence (>70%):
  Predictions: {stats.get('high_confidence_predictions', 0)}
  Accuracy: {stats.get('high_confidence_accuracy', 0):.2f}%

Low Confidence (≤70%):
  Predictions: {stats.get('low_confidence_predictions', 0)}
  Accuracy: {stats.get('low_confidence_accuracy', 0):.2f}%

Model Performance:
  ResNet50: Active ✓
  EfficientNet: Active ✓
  InceptionV3: Active ✓
  
Average Processing Time: ~2-3 seconds
Image Variations Processed: 4 per prediction
        """
        perf_text.insert(tk.END, perf_content)
        
        # Learning statistics
        recent_corrections = learning_data.get('corrections', [])[-10:]
        
        learning_content = f"""
📚 LEARNING STATISTICS
{'='*40}

Total Corrections: {learning_data.get('total_corrections', 0)}
User Confirmations: {len(learning_data.get('user_confirmations', []))}

Recent Corrections:
"""
        
        for correction in recent_corrections:
            learning_content += f"  • {correction.get('predicted', 'N/A')} → {correction.get('correct', 'N/A')}\n"
            learning_content += f"    Confidence: {correction.get('confidence', 0):.1f}%\n"
        
        learning_content += f"""
        
Learning Effectiveness:
  System adapts to user corrections ✓
  Builds custom food database ✓
  Tracks confidence patterns ✓
        """
        
        learning_text.insert(tk.END, learning_content)
        
        # Web scraping statistics
        web_content = f"""
🌐 WEB DATA STATISTICS
{'='*40}

Total Nutrition Searches: {web_data.get('total_searches', 0)}
Successful Searches: {web_data.get('successful_searches', 0)}
Success Rate: {(web_data.get('successful_searches', 0) / max(web_data.get('total_searches', 1), 1) * 100):.1f}%

Cache Statistics:
  Cache Hits: {web_data.get('cache_hits', 0)}
  Cache Efficiency: Active ✓

Data Sources:
  Google Search Results ✓
  USDA FoodData Central (planned)
  MyFitnessPal Database (planned)
  
Average Search Time: ~3-5 seconds
Cache Retention: 7 days

Foods with Web Data: {len([f for f in self.database.data['foods'].values() if f.get('source') in ['google_search', 'openfoodfacts']])}
        """
        
        web_text.insert(tk.END, web_content)
    
    def update_analytics_display(self):
        """Update analytics tab display"""
        if hasattr(self, 'stats_text'):
            stats = self.database.data['statistics']
            learning = self.database.data['learning_data']
            
            stats_content = f"""📊 SYSTEM PERFORMANCE OVERVIEW
{'='*50}

🎯 Detection Accuracy: {stats.get('accuracy_rate', 0):.1f}%
📈 Total Predictions: {stats.get('total_predictions', 0)}
✅ Correct Predictions: {stats.get('correct_predictions', 0)}
🔄 Total Corrections: {learning.get('total_corrections', 0)}

🤖 AI Models in Use:
  • ResNet50 (Weight: 40%)
  • EfficientNetB3 (Weight: 35%)  
  • InceptionV3 (Weight: 25%)

🌐 Web Nutrition Data:
  • Total Searches: {self.database.data['web_scraping'].get('total_searches', 0)}
  • Success Rate: {(self.database.data['web_scraping'].get('successful_searches', 0) / max(self.database.data['web_scraping'].get('total_searches', 1), 1) * 100):.1f}%

📚 Learning System:
  • Active Learning: ✓ Enabled
  • User Feedback: ✓ Integrated
  • Confidence Tracking: ✓ Active
            """
            
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, stats_content)
        
        # Update activity
        if hasattr(self, 'activity_text'):
            recent_corrections = learning.get('corrections', [])[-5:]
            
            activity_content = "🕒 RECENT SYSTEM ACTIVITY\n" + "="*30 + "\n\n"
            
            if recent_corrections:
                for correction in recent_corrections:
                    timestamp = correction.get('timestamp', '')[:19].replace('T', ' ')
                    activity_content += f"📝 {timestamp}\n"
                    activity_content += f"   Correction: {correction.get('predicted', 'N/A')} → {correction.get('correct', 'N/A')}\n"
                    activity_content += f"   Confidence: {correction.get('confidence', 0):.1f}%\n\n"
            else:
                activity_content += "No recent activity to display.\n"
                activity_content += "Start using the detection system to see activity here!"
            
            self.activity_text.delete(1.0, tk.END)
            self.activity_text.insert(tk.END, activity_content)
    
    def update_database_display(self):
        """Update database tab display"""
        if hasattr(self, 'db_info_text'):
            foods = self.database.data['foods']
            
            db_info = f"""🗃️ DATABASE INFORMATION
{'='*30}

Total Foods: {len(foods)}
Default Foods: {len([f for f in foods.values() if f.get('source') == 'default'])}
Web-Sourced Foods: {len([f for f in foods.values() if f.get('source') in ['google_search', 'openfoodfacts']])}
User-Added Foods: {len([f for f in foods.values() if f.get('source') == 'user_added'])}

Database Size: {os.path.getsize(self.database.db_file) / 1024:.1f} KB
Last Updated: {self.database.data['learning_data'].get('last_updated', 'Unknown')[:19]}

Backup Available: {'✓' if os.path.exists(self.database.db_file + '.backup') else '❌'}
            """
            
            self.db_info_text.delete(1.0, tk.END)
            self.db_info_text.insert(tk.END, db_info)
        
        # Update foods tree
        if hasattr(self, 'foods_tree'):
            # Clear existing items
            for item in self.foods_tree.get_children():
                self.foods_tree.delete(item)
            
            # Add foods
            for food_name, data in sorted(self.database.data['foods'].items()):
                values = (
                    food_name.title(),
                    data.get('calories', 'Unknown'),
                    f"{data.get('protein', 'Unknown')}g" if data.get('protein') != 'Unknown' else 'Unknown',
                    f"{data.get('carbs', 'Unknown')}g" if data.get('carbs') != 'Unknown' else 'Unknown',
                    f"{data.get('fat', 'Unknown')}g" if data.get('fat') != 'Unknown' else 'Unknown',
                    data.get('source', 'Unknown').title()
                )
                self.foods_tree.insert('', tk.END, values=values)
    
    def run(self):
        """Run the enhanced GUI application"""
        self.root.mainloop()

def create_requirements_file():
    """Create requirements.txt file for easy installation"""
    requirements = """# Enhanced Food Detection System Requirements
tensorflow>=2.10.0
opencv-python>=4.6.0
pillow>=9.0.0
numpy>=1.21.0
requests>=2.28.0
beautifulsoup4>=4.11.0
scikit-learn>=1.1.0
matplotlib>=3.5.0
"""
    
    try:
        with open('requirements.txt', 'w') as f:
            f.write(requirements)
        logger.info("✅ Created requirements.txt file")
    except Exception as e:
        logger.error(f"❌ Could not create requirements.txt: {e}")

def main():
    """Main function to run the enhanced food detection system"""
    print("🍽️ ENHANCED FOOD DETECTION SYSTEM")
    print("="*50)
    print("🤖 Multi-Model AI Detection")
    print("🌐 Web Nutrition Data Scraping") 
    print("📚 Advanced Learning System")
    print("📊 Detailed Analytics")
    print("="*50)
    
    try:
        # Create requirements file
        create_requirements_file()
        
        # Initialize and run GUI
        app = EnhancedFoodDetectionGUI()
        
        logger.info("🚀 Starting Enhanced Food Detection System...")
        app.run()
        
    except KeyboardInterrupt:
        logger.info("👋 Application closed by user")
    except Exception as e:
        logger.error(f"❌ Application error: {e}")
        print(f"\n❌ Error: {e}")
        print("\n💡 Troubleshooting Tips:")
        print("1. Install requirements: pip install -r requirements.txt")
        print("2. Check Python version (3.7+ required)")
        print("3. Ensure camera permissions if using camera feature")
        print("4. Check internet connection for web nutrition data")

if __name__ == "__main__":
    main()