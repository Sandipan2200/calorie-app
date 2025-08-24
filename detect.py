import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps, ImageTk
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import os
import json
import pickle
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk, filedialog
try:
    import tkinter as tk
    from tkinter import messagebox, simpledialog, ttk, filedialog

    def setup_ui(self):
        # Configure style
        style = ttk.Style()
        style.configure('TLabel', font=('Arial', 11))
        style.configure('TLabelframe.Label', font=('Arial', 12, 'bold'))
        
        # Main container with white background
        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel for image
        self.image_frame = ttk.LabelFrame(self.main_frame, text="📷 Detected Image", padding="10")
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Image label with gray background to show boundaries
        self.image_label = ttk.Label(self.image_frame, background='#f0f0f0')
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    print("GUI not available, using console interface")
import threading
import difflib
import tempfile

class JSONFoodDatabase:
    """JSON-based food database with learning capabilities"""
    def __init__(self, db_file='food_nutrition_database.json'):
        self.db_file = db_file
        self.load_database()
    
    def load_database(self):
        """Load or create the food database"""
        default_data = {
            "foods": {
                "apple": {"calories": 52, "protein": 0.3, "carbs": 14, "fat": 0.2, "source": "default"},
                "banana": {"calories": 89, "protein": 1.1, "carbs": 23, "fat": 0.3, "source": "default"},
                "pizza": {"calories": 266, "protein": 11, "carbs": 33, "fat": 10, "source": "default"},
                "burger": {"calories": 295, "protein": 17, "carbs": 28, "fat": 14, "source": "default"},
                "hamburger": {"calories": 295, "protein": 17, "carbs": 28, "fat": 14, "source": "default"},
                "cheeseburger": {"calories": 313, "protein": 18, "carbs": 28, "fat": 16, "source": "default"},
                "hot dog": {"calories": 290, "protein": 10, "carbs": 4, "fat": 26, "source": "default"},
                "french fries": {"calories": 365, "protein": 4, "carbs": 63, "fat": 17, "source": "default"},
                "ice cream": {"calories": 207, "protein": 3.5, "carbs": 24, "fat": 11, "source": "default"},
                "donut": {"calories": 452, "protein": 5, "carbs": 51, "fat": 25, "source": "default"},
                "sandwich": {"calories": 250, "protein": 12, "carbs": 30, "fat": 8, "source": "default"},
                "pasta": {"calories": 220, "protein": 8, "carbs": 44, "fat": 1.5, "source": "default"},
                "rice": {"calories": 130, "protein": 2.7, "carbs": 28, "fat": 0.3, "source": "default"},
                "chicken": {"calories": 239, "protein": 27, "carbs": 0, "fat": 14, "source": "default"},
                "salad": {"calories": 33, "protein": 3, "carbs": 6, "fat": 0.3, "source": "default"},
                "bread": {"calories": 265, "protein": 9, "carbs": 49, "fat": 3.2, "source": "default"},
                "egg": {"calories": 155, "protein": 13, "carbs": 1.1, "fat": 11, "source": "default"},
                "milk": {"calories": 42, "protein": 3.4, "carbs": 5, "fat": 1, "source": "default"},
                "cheese": {"calories": 402, "protein": 25, "carbs": 1.3, "fat": 33, "source": "default"},
                "yogurt": {"calories": 59, "protein": 10, "carbs": 3.6, "fat": 0.4, "source": "default"}
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
        
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
                print(f"✅ Loaded database with {len(self.data['foods'])} foods")
            except Exception as e:
                print(f"❌ Error loading database: {e}")
                self.data = default_data
                self.save_database()
        else:
            self.data = default_data
            self.save_database()
            print("📝 Created new food database")
    
    def save_database(self):
        """Save the database to JSON file"""
        try:
            self.data['learning_data']['last_updated'] = datetime.now().isoformat()
            with open(self.db_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Error saving database: {e}")
    
    def get_nutrition(self, food_name):
        """Get nutrition information for a food item"""
        food_name = food_name.lower().strip()
        
        # Try exact match first
        if food_name in self.data['foods']:
            nutrition = self.data['foods'][food_name].copy()
            nutrition.pop('source', None)  # Remove source from returned data
            return nutrition
        
        # Try partial matching
        for food, nutrition_data in self.data['foods'].items():
            if food in food_name or food_name in food:
                nutrition = nutrition_data.copy()
                nutrition.pop('source', None)
                return nutrition
        
        # Return unknown values if not found
        return {'calories': 'Unknown', 'protein': 'Unknown', 'carbs': 'Unknown', 'fat': 'Unknown'}

    def fetch_nutrition_online(self, food_name):
        """Try to fetch nutrition per 100g for a food using OpenFoodFacts as a fallback.

        Returns a dict with calories, protein, carbs, fat or None if not found.
        This function is best-effort and will not raise on network errors.
        """
        try:
            search_url = "https://world.openfoodfacts.org/cgi/search.pl"
            params = {
                'search_terms': food_name,
                'search_simple': 1,
                'action': 'process',
                'json': 1,
                'page_size': 10
            }
            resp = requests.get(search_url, params=params, timeout=10)
            if not resp.ok:
                return None

            data = resp.json()
            products = data.get('products', [])

            for p in products:
                nutriments = p.get('nutriments', {}) or {}

                # try common keys
                calories = nutriments.get('energy-kcal_100g') or nutriments.get('energy_100g')
                # some entries use kj; convert if needed
                if calories is None and 'energy-kj_100g' in nutriments:
                    try:
                        calories = float(nutriments.get('energy-kj_100g')) / 4.184
                    except Exception:
                        calories = None

                protein = nutriments.get('proteins_100g') or nutriments.get('protein_100g')
                carbs = nutriments.get('carbohydrates_100g') or nutriments.get('carbs_100g')
                fat = nutriments.get('fat_100g')

                # Accept if any meaningful value exists
                if any(v is not None for v in (calories, protein, carbs, fat)):
                    def safe_num(v):
                        try:
                            return float(v)
                        except Exception:
                            return v

                    return {
                        'calories': safe_num(calories) if calories is not None else 'Unknown',
                        'protein': safe_num(protein) if protein is not None else 'Unknown',
                        'carbs': safe_num(carbs) if carbs is not None else 'Unknown',
                        'fat': safe_num(fat) if fat is not None else 'Unknown',
                        'source': 'openfoodfacts'
                    }

            return None
        except Exception:
            return None

    def _background_fetch_and_update(self, target_key, search_name):
        """Background worker: fetch nutrition for search_name and update target_key in DB if found."""
        try:
            fetched = self.fetch_nutrition_online(search_name)
            if not fetched:
                print(f"🔎 Background: no online data found for '{search_name}'")
                return

            existing = self.data['foods'].get(target_key)
            if not existing:
                print(f"🔎 Background: target '{target_key}' no longer exists")
                return

            updated = False
            for k in ('calories', 'protein', 'carbs', 'fat'):
                if str(existing.get(k, 'Unknown')).lower() == 'unknown' and fetched.get(k) is not None:
                    existing[k] = fetched.get(k)
                    updated = True

            if updated:
                existing['source'] = fetched.get('source', existing.get('source', 'user_added'))
                self.save_database()
                print(f"🔄 Background: updated '{target_key}' with online nutrition from '{search_name}'.")
        except Exception as e:
            print(f"🔎 Background fetch error for '{search_name}': {e}")

    def find_similar_food(self, food_name, cutoff=0.8):
        """Return the best similar existing food key or None if no close match."""
        try:
            keys = list(self.data.get('foods', {}).keys())
            matches = difflib.get_close_matches(food_name, keys, n=1, cutoff=cutoff)
            return matches[0] if matches else None
        except Exception:
            return None
    
    def add_food(self, food_name, nutrition_data=None):
        """Add or update a food in the database with online verification.

        Behavior:
        - Normalize name (lowercase, strip).
        - If exact name exists: optionally update missing/unknown nutrition from online.
        - If a close (fuzzy) match exists: treat as same food and update that entry instead of creating duplicate.
        - Otherwise, try to fetch nutrition online and save a new entry.

        Returns:
          (added: bool, target_name: str) -> added is True if a new entry was created, False otherwise.
        """
        food_name = food_name.lower().strip()

        # If exact exists, consider updating missing fields
        if food_name in self.data['foods']:
            existing = self.data['foods'][food_name]
            need_lookup = any(str(existing.get(k, 'Unknown')).lower() == 'unknown' for k in ('calories', 'protein', 'carbs', 'fat'))
            if need_lookup:
                # Start background update to avoid blocking UI
                threading.Thread(target=self._background_fetch_and_update, args=(food_name, food_name), daemon=True).start()
                print(f"🔄 Scheduled background nutrition lookup for existing food '{food_name}'.")
            else:
                print(f"ℹ️ Food '{food_name}' already exists and has nutrition — not adding.")
            return (False, food_name)

        # Check for close/fuzzy match to avoid near-duplicates
        similar = self.find_similar_food(food_name, cutoff=0.82)
        if similar:
            print(f"⚠️ Found similar existing food '{similar}' for '{food_name}'. Will not create a duplicate.")
            # Schedule background update to try to fill missing nutrition for the similar entry
            threading.Thread(target=self._background_fetch_and_update, args=(similar, food_name), daemon=True).start()
            print(f"🔄 Scheduled background nutrition lookup to update similar food '{similar}' using '{food_name}'.")
            return (False, similar)

        # Prepare nutrition entry
        if nutrition_data is None:
            nutrition_data = {'calories': 'Unknown', 'protein': 'Unknown', 'carbs': 'Unknown', 'fat': 'Unknown'}
            source_label = 'user_added'
        else:
            source_label = nutrition_data.get('source', 'user_added')

        # If any field is Unknown, try fetching online once and prefer online numeric values
        need_lookup = any(str(nutrition_data.get(k, 'Unknown')).lower() == 'unknown' for k in ('calories', 'protein', 'carbs', 'fat'))
        if need_lookup:
            # Save placeholder entry first, then fetch online in background to fill in details
            nutrition_data['source'] = source_label
            self.data['foods'][food_name] = nutrition_data
            self.save_database()
            threading.Thread(target=self._background_fetch_and_update, args=(food_name, food_name), daemon=True).start()
            print(f"🔎 Saved '{food_name}' with placeholder nutrition and scheduled background lookup.")
            print(f"➕ Added new food: {food_name}")
            return (True, food_name)

        # Finalize and save (no lookup needed)
        nutrition_data['source'] = source_label
        self.data['foods'][food_name] = nutrition_data
        self.save_database()
        print(f"➕ Added new food: {food_name}")
        return (True, food_name)
    
    def add_correction(self, predicted_food, correct_food, confidence):
        """Add a correction to the learning data.

        Returns a dict with whether a new food was added and the target food key.
        """
        # Clean up names
        predicted_clean = str(predicted_food).lower().strip()
        correct_clean = str(correct_food).lower().strip()

        correction = {
            'id': len(self.data['learning_data']['corrections']) + 1,
            'predicted': predicted_clean,
            'correct': correct_clean,
            'confidence': float(confidence) if confidence is not None else None,
            'timestamp': datetime.now().isoformat()
        }

        # Add correction record
        self.data['learning_data']['corrections'].append(correction)
        self.data['learning_data']['total_corrections'] += 1

        # Ensure corrected food exists: add placeholder and schedule background lookup if needed
        added = False
        target = correct_clean
        if correct_clean not in self.data['foods']:
            added, target = self.add_food(correct_clean)
        else:
            existing = self.data['foods'].get(correct_clean, {})
            need_lookup = any(str(existing.get(k, 'Unknown')).lower() == 'unknown' for k in ('calories', 'protein', 'carbs', 'fat'))
            if need_lookup:
                threading.Thread(target=self._background_fetch_and_update, args=(correct_clean, correct_clean), daemon=True).start()

        # Save changes
        self.save_database()

        # Log the learning
        print(f"\n{'='*50}")
        print(f"📚 Learning Update:")
        print(f"Previous detection: {predicted_food}")
        print(f"Corrected to: {correct_food}")
        try:
            print(f"Confidence: {float(confidence):.1f}%")
        except Exception:
            print(f"Confidence: {confidence}")
        print(f"Total corrections: {self.data['learning_data']['total_corrections']}")
        print(f"{'='*50}\n")

        return {'added': added, 'target': target}
    
    def update_statistics(self, is_correct):
        """Update prediction statistics"""
        self.data['statistics']['total_predictions'] += 1
        if is_correct:
            self.data['statistics']['correct_predictions'] += 1
        
        if self.data['statistics']['total_predictions'] > 0:
            self.data['statistics']['accuracy_rate'] = (
                self.data['statistics']['correct_predictions'] / 
                self.data['statistics']['total_predictions'] * 100
            )
        
        self.save_database()
    
    def get_learning_suggestions(self, predicted_food):
        """Get suggestions based on previous corrections"""
        suggestions = []
        predicted_lower = predicted_food.lower()
        
        for correction in self.data['learning_data']['corrections']:
            if correction['predicted'].lower() == predicted_lower:
                suggestions.append(correction['correct'])
        
        # Return most common correction
        if suggestions:
            from collections import Counter
            most_common = Counter(suggestions).most_common(1)[0][0]
            return most_common
        
        return None

class SmartFeedbackGUI:
    """Enhanced GUI for user feedback"""
    def __init__(self, detector):
        self.detector = detector
        self.root = tk.Tk()
        self.root.title("🍽️ Food Detection System")
        self.root.geometry("1200x800")
        self.root.configure(bg='white')
        self.setup_ui()
        
    def show_feedback_dialog(self, predicted_food, confidence, image_path=None):
        """Show enhanced feedback dialog with buttons"""
        self.root = tk.Tk()
        self.root.title("🍽️ Food Detection Result")
        self.root.geometry("500x400")
        self.root.configure(bg='#f0f0f0')
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="🍽️ Food Detection Result", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Result display
        result_frame = ttk.LabelFrame(main_frame, text="Detection Result", padding="15")
        result_frame.pack(fill=tk.X, pady=(0, 20))
        
        food_label = ttk.Label(result_frame, text=f"🎯 Detected Food: {predicted_food}", 
                              font=('Arial', 12, 'bold'))
        food_label.pack(anchor=tk.W)
        
        confidence_label = ttk.Label(result_frame, text=f"📊 Confidence: {confidence:.1f}%", 
                                   font=('Arial', 10))
        confidence_label.pack(anchor=tk.W, pady=(5, 0))
        
        # Confidence color indicator
        if confidence > 80:
            status_text = "✅ High Confidence"
            status_color = "green"
        elif confidence > 50:
            status_text = "⚠️ Medium Confidence"  
            status_color = "orange"
        else:
            status_text = "❓ Low Confidence"
            status_color = "red"
        
        status_label = ttk.Label(result_frame, text=status_text, foreground=status_color)
        status_label.pack(anchor=tk.W, pady=(5, 0))
        
        # Question
        question_label = ttk.Label(main_frame, text="Is this detection correct?", 
                                  font=('Arial', 12, 'bold'))
        question_label.pack(pady=(0, 15))
        
        # Buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(pady=(0, 20))
        
        # Yes button (green)
        yes_btn = tk.Button(buttons_frame, text="✅ Yes, it's correct!", 
                           command=self.on_correct,
                           bg='#4CAF50', fg='white', font=('Arial', 11, 'bold'),
                           padx=20, pady=10, cursor='hand2')
        yes_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # No button (red)  
        no_btn = tk.Button(buttons_frame, text="❌ No, it's wrong", 
                          command=self.on_incorrect,
                          bg='#f44336', fg='white', font=('Arial', 11, 'bold'),
                          padx=20, pady=10, cursor='hand2')
        no_btn.pack(side=tk.LEFT)
        
        # Tips section
        tips_frame = ttk.LabelFrame(main_frame, text="💡 Tips", padding="10")
        tips_frame.pack(fill=tk.X)
        
        tips_text = ("• Your feedback helps the AI learn and improve\n"
                    "• Higher confidence usually means better accuracy\n" 
                    "• Corrections are saved for future predictions")
        
        tips_label = ttk.Label(tips_frame, text=tips_text, font=('Arial', 9))
        tips_label.pack(anchor=tk.W)
        
        # Center the window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (self.root.winfo_width() // 2)
        y = (self.root.winfo_screenheight() // 2) - (self.root.winfo_height() // 2)
        self.root.geometry(f"+{x}+{y}")
        
        # Make window modal
        self.root.transient()
        self.root.grab_set()
        self.root.focus_force()
        
        self.root.mainloop()
        
        return self.is_correct, self.correct_food
    
    def on_correct(self):
        """Handle correct prediction"""
        self.is_correct = True
        self.correct_food = None
        self.root.destroy()
    
    def on_incorrect(self):
        """Handle incorrect prediction - show correction dialog"""
        self.root.destroy()
        self.show_correction_dialog()
    
    def show_correction_dialog(self):
        """Show dialog to get correct food name"""
        self.root = tk.Tk()
        self.root.title("🔧 Correction Required")
        self.root.geometry("400x250")
        self.root.configure(bg='#f0f0f0')
        
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="🔧 Help Me Learn!", 
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=(0, 15))
        
        # Instruction
        instruction_label = ttk.Label(main_frame, 
                                     text="What is the correct name of the food?",
                                     font=('Arial', 11))
        instruction_label.pack(pady=(0, 10))
        
        # Entry field
        entry_frame = ttk.Frame(main_frame)
        entry_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.food_entry = ttk.Entry(entry_frame, font=('Arial', 11))
        self.food_entry.pack(fill=tk.X)
        self.food_entry.focus()
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack()
        
        submit_btn = tk.Button(button_frame, text="✅ Submit Correction",
                              command=self.on_correction_submit,
                              bg='#2196F3', fg='white', font=('Arial', 10, 'bold'),
                              padx=15, pady=8, cursor='hand2')
        submit_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        cancel_btn = tk.Button(button_frame, text="❌ Cancel",
                              command=self.on_correction_cancel,
                              bg='#757575', fg='white', font=('Arial', 10),
                              padx=15, pady=8, cursor='hand2')
        cancel_btn.pack(side=tk.LEFT)
        
        # Bind Enter key
        self.food_entry.bind('<Return>', lambda e: self.on_correction_submit())
        
        # Center the window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (self.root.winfo_width() // 2)
        y = (self.root.winfo_screenheight() // 2) - (self.root.winfo_height() // 2)
        self.root.geometry(f"+{x}+{y}")
        
        self.root.transient()
        self.root.grab_set()
        
        self.root.mainloop()
    
    def on_correction_submit(self):
        """Handle correction submission"""
        correction = self.food_entry.get().strip()
        if correction:
            self.correct_food = correction
            self.is_correct = False
            self.root.destroy()
        else:
            messagebox.showwarning("Warning", "Please enter a food name!")
    
    def on_correction_cancel(self):
        """Handle correction cancellation"""
        self.correct_food = None
        self.is_correct = False
        self.root.destroy()

class AdvancedFoodDetector:
    def __init__(self):
        """Initialize the advanced food detection system"""
        print("🚀 Initializing Advanced Food Detection System...")
        
        # Initialize components
        self.database = JSONFoodDatabase()
        self.gui_feedback = None  # Will be initialized later
        
        # Load pre-trained model
        print("🧠 Loading ResNet50 model...")
        self.model = ResNet50(weights='imagenet', include_top=True)
        
        # Enhanced food keywords
        self.food_keywords = {
            'pizza', 'burger', 'hamburger', 'cheeseburger', 'hot_dog', 'hotdog',
            'french_fries', 'fries', 'ice_cream', 'icecream', 'donut', 'doughnut',
            'sandwich', 'pasta', 'spaghetti', 'noodle', 'rice', 'chicken', 'fish',
            'salad', 'bread', 'egg', 'milk', 'cheese', 'yogurt', 'apple', 'banana',
            'orange', 'strawberry', 'broccoli', 'carrot', 'tomato', 'potato',
            'cake', 'cookie', 'chocolate', 'coffee', 'tea', 'soda', 'burrito',
            'taco', 'sushi', 'ramen', 'soup', 'steak', 'bacon', 'cereal',
            'pancake', 'waffle', 'muffin', 'croissant', 'bagel', 'pretzel'
        }
        
        print("✅ System initialized successfully!")
    
    def preprocess_image(self, img_input):
        """Enhanced image preprocessing"""
        try:
            if isinstance(img_input, str):
                if img_input.startswith('http'):
                    response = requests.get(img_input)
                    img = Image.open(BytesIO(response.content))
                else:
                    img = Image.open(img_input)
            else:
                img = img_input
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            original_img = img.copy()
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            return img_array, original_img
        
        except Exception as e:
            print(f"❌ Error preprocessing image: {str(e)}")
            return None, None
    
    def extract_food_predictions(self, predictions, top_n=10):
        """Extract and rank food-related predictions"""
        decoded = decode_predictions(predictions, top=top_n)[0]
        
        food_predictions = []
        
        for i, (class_id, class_name, confidence) in enumerate(decoded):
            clean_name = class_name.replace('_', ' ').title()
            confidence_pct = float(confidence * 100)
            
            is_food = any(food_word in class_name.lower() for food_word in self.food_keywords)
            
            if is_food:
                food_predictions.append({
                    'rank': i + 1,
                    'class_name': clean_name,
                    'confidence': confidence_pct
                })
        
        return food_predictions
    
    def detect_food(self, img_input, show_image=True, get_feedback=True):
        """Main food detection method"""
        print("🔍 Analyzing image...")
        
        processed_img, original_img = self.preprocess_image(img_input)
        if processed_img is None:
            return None
        
        # Make prediction
        predictions = self.model.predict(processed_img, verbose=0)
        food_predictions = self.extract_food_predictions(predictions)
        
        if food_predictions:
            # Check for learned suggestions first
            top_food = food_predictions[0]
            predicted_name = top_food['class_name']
            confidence = top_food['confidence']
            
            # Check learned corrections from database
            suggestion = self.database.get_learning_suggestions(predicted_name)
            if suggestion:
                print(f"💡 Found learned suggestion: {suggestion}")
                predicted_name = suggestion.title()
                confidence = min(confidence * 1.2, 95)  # Boost confidence slightly
                
            # Check recent corrections in database
            corrections = self.database.data['learning_data']['corrections']
            if corrections:
                # Look for recent corrections of this prediction
                recent_corrections = [c for c in corrections 
                                   if c['predicted'].lower() == predicted_name.lower()]
                if recent_corrections:
                    # Use most recent correction
                    latest = sorted(recent_corrections, 
                                 key=lambda x: x['timestamp'])[-1]
                    predicted_name = latest['correct'].title()
                    print(f"📚 Using learned correction: {predicted_name}")
        else:
            predicted_name = "Unknown Food Item"
            confidence = 0
        
        # Get nutrition information
        nutrition = self.database.get_nutrition(predicted_name)

        # If nutrition contains Unknown values, try an immediate online lookup
        try:
            needs_lookup = any(str(nutrition.get(k, 'Unknown')).lower() == 'unknown' for k in ('calories', 'protein', 'carbs', 'fat'))
        except Exception:
            needs_lookup = False

        if needs_lookup:
            print(f"🔎 Nutrition incomplete for '{predicted_name}'. Attempting online lookup...")
            fetched = self.database.fetch_nutrition_online(predicted_name)
            if fetched:
                key = predicted_name.lower().strip()
                # Ensure an entry exists in DB
                existing = self.database.data['foods'].get(key)
                if existing is None:
                    # create a minimal entry and then update
                    self.database.data['foods'][key] = {
                        'calories': 'Unknown', 'protein': 'Unknown', 'carbs': 'Unknown', 'fat': 'Unknown',
                        'source': fetched.get('source', 'openfoodfacts')
                    }
                    existing = self.database.data['foods'][key]

                updated = False
                for k in ('calories', 'protein', 'carbs', 'fat'):
                    if (str(existing.get(k, 'Unknown')).lower() == 'unknown' or existing.get(k) is None) and fetched.get(k) is not None:
                        existing[k] = fetched.get(k)
                        nutrition[k] = fetched.get(k)
                        updated = True
                    else:
                        # ensure nutrition dict reflects DB value if present
                        nutrition[k] = existing.get(k, nutrition.get(k))

                if updated:
                    existing['source'] = fetched.get('source', existing.get('source', 'openfoodfacts'))
                    self.database.save_database()
                    print(f"✅ Fetched nutrition saved for '{key}' from online source.")
            else:
                print(f"🔎 No online nutrition found for '{predicted_name}'.")
        
        # Display results
        self.display_results(original_img, predicted_name, confidence, nutrition, show_image)
        
        return {
            'predicted_food': predicted_name,
            'confidence': confidence,
            'nutrition': nutrition
        }
    
    def display_results(self, image_obj, food_name, confidence, nutrition, show_image=True):
        """Display results in the tkinter window"""
        if hasattr(self, 'gui'):
            self.gui.update_results(image_obj, food_name, confidence, nutrition)
        
        # Console output for backup
        print(f"\n{'='*60}")
        print("🍽️ FOOD DETECTION RESULTS")
        print(f"{'='*60}")
        print(f"🎯 Food: {food_name}")
        print(f"📊 Confidence: {confidence:.1f}%")
        if nutrition['calories'] != 'Unknown':
            print(f"\n🥗 Nutrition (per 100g):")
            print(f"   Calories: {nutrition['calories']}")
            print(f"   Protein: {nutrition['protein']}g")
            print(f"   Carbs: {nutrition['carbs']}g")
            print(f"   Fat: {nutrition['fat']}g")
        print(f"{'='*60}")

    def detect_from_camera(self):
        """Camera detection with enhanced feedback"""
        print("📷 Opening camera... Press 'c' to capture, 'q' to quit")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip the frame horizontally for a mirror effect (what user sees)
            frame = cv2.flip(frame, 1)

            # Compute a centered square capture region (relative to frame size)
            h, w = frame.shape[:2]
            box_size = int(min(w, h) * 0.6)  # 60% of smaller dimension
            cx, cy = w // 2, h // 2
            x1, y1 = max(0, cx - box_size // 2), max(0, cy - box_size // 2)
            x2, y2 = min(w, cx + box_size // 2), min(h, cy + box_size // 2)

            # Draw the capture rectangle and instructions
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 255), 2)
            cv2.putText(overlay, "Press 'c' to capture, 'q' to quit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('🍽️ Food Detection Camera', overlay)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                # Crop to the capture rectangle so captured image matches preview
                crop_bgr = frame[y1:y2, x1:x2]
                if crop_bgr.size == 0:
                    # fallback to full frame if crop failed
                    crop_bgr = frame

                # Convert BGR to RGB and create PIL image
                rgb_frame = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)

                # Save to a temporary file so the app can load it as an uploaded image
                tmpf = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                pil_image.save(tmpf.name, format='PNG')
                tmpf.close()

                print(f"\n📸 Captured image saved to {tmpf.name}; closing camera and uploading to app...")
                cap.release()
                cv2.destroyAllWindows()

                # Call detection using path (keeps behaviour consistent with uploads)
                self.detect_food(tmpf.name)
                return

            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
class FoodDetectorGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🍽️ Food Detection System")
        self.root.geometry("1200x800")
        self.root.configure(bg='white')  # Set white background
        self.setup_ui()
        
    def setup_ui(self):
        # Main container
        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel for image
        self.image_frame = ttk.LabelFrame(self.main_frame, text="� Detected Image", padding="10")
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Right panel for results and feedback
        right_panel = ttk.Frame(self.main_frame)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Results section
        self.results_frame = ttk.LabelFrame(right_panel, text="🔍 Detection Results", padding="10")
        self.results_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Food name and confidence
        self.food_label = ttk.Label(self.results_frame, 
                                  font=('Arial', 14, 'bold'),
                                  foreground='#333333',  # Dark gray text
                                  background='#ffffff')  # White background
        self.food_label.pack(anchor=tk.W, pady=5, padx=5, fill=tk.X)
        
        self.confidence_label = ttk.Label(self.results_frame,
                                        font=('Arial', 12),
                                        foreground='#666666',  # Medium gray text
                                        background='#ffffff')  # White background
        self.confidence_label.pack(anchor=tk.W, pady=5, padx=5, fill=tk.X)
        
        # Nutrition section with custom style
        self.nutrition_frame = ttk.LabelFrame(right_panel, text="🥗 Nutrition Facts (per 100g)", padding="15")
        self.nutrition_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Style for nutrition labels
        nutrition_style = {'font': ('Arial', 12),
                          'foreground': '#333333',
                          'background': '#f8f9fa',
                          'padding': 8}
        
        # Create nutrition labels with better styling
        self.calories_label = tk.Label(self.nutrition_frame, **nutrition_style)
        self.calories_label.pack(anchor=tk.W, pady=2, fill=tk.X)
        
        self.protein_label = tk.Label(self.nutrition_frame, **nutrition_style)
        self.protein_label.pack(anchor=tk.W, pady=2, fill=tk.X)
        
        self.carbs_label = tk.Label(self.nutrition_frame, **nutrition_style)
        self.carbs_label.pack(anchor=tk.W, pady=2, fill=tk.X)
        
        self.fat_label = tk.Label(self.nutrition_frame, **nutrition_style)
        self.fat_label.pack(anchor=tk.W, pady=2, fill=tk.X)
        
        # Feedback section
        feedback_frame = ttk.LabelFrame(right_panel, text="💭 Your Feedback", padding="10")
        feedback_frame.pack(fill=tk.X)
        
        ttk.Label(feedback_frame, 
                 text="Was this detection correct?",
                 font=('Arial', 12, 'bold')).pack(pady=(0, 10))
        
        # Feedback buttons
        btn_frame = ttk.Frame(feedback_frame)
        btn_frame.pack()
        
        self.yes_btn = tk.Button(btn_frame,
                               text="✅ YES, CORRECT!",
                               command=self.on_correct,
                               bg='#4CAF50', fg='white',
                               font=('Arial', 12, 'bold'),
                               width=20, height=2)
        self.yes_btn.pack(side=tk.LEFT, padx=5)
        
        self.no_btn = tk.Button(btn_frame,
                              text="❌ NO, WRONG",
                              command=self.on_wrong,
                              bg='#f44336', fg='white',
                              font=('Arial', 12, 'bold'),
                              width=20, height=2)
        self.no_btn.pack(side=tk.LEFT, padx=5)
        
        # Input section
        input_frame = ttk.LabelFrame(right_panel, text="📥 Load Image", padding="10")
        input_frame.pack(fill=tk.X, pady=20)
        
        btn_container = ttk.Frame(input_frame)
        btn_container.pack()
        
        ttk.Button(btn_container,
                  text="📁 Load Image File",
                  command=self.load_image_file).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_container,
                  text="🌐 Load from URL",
                  command=self.load_image_url).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_container,
                  text="📷 Use Camera",
                  command=self.use_camera).pack(side=tk.LEFT, padx=5)
    
    def update_results(self, image_obj, food_name, confidence, nutrition):
        # Update image
        if image_obj:
            try:
                # Convert PIL image to PhotoImage with error handling
                # Resize image while maintaining aspect ratio
                width = 500
                ratio = float(width) / image_obj.size[0]
                height = int(image_obj.size[1] * ratio)
                
                image = image_obj.resize((width, height), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                
                self.image_label.configure(image=photo)
                self.image_label.image = photo  # Keep a reference
            except Exception as e:
                print(f"Error displaying image: {e}")
                self.image_label.configure(text="Error loading image", foreground='red')
        
        # Update detection results
        self.food_label.configure(text=f"�️ Detected Food: {food_name}")
        self.confidence_label.configure(text=f"📊 Confidence: {confidence:.1f}%")
        
        # Update nutrition info
        self.calories_label.configure(text=f"🔥 Calories: {nutrition['calories']}")
        self.protein_label.configure(text=f"🥩 Protein: {nutrition['protein']}g")
        self.carbs_label.configure(text=f"🍚 Carbs: {nutrition['carbs']}g")
        self.fat_label.configure(text=f"� Fat: {nutrition['fat']}g")
        
        # Store current results for feedback
        self.current_food = food_name
        self.current_confidence = confidence
    
    def on_correct(self):
        self.detector.database.update_statistics(True)
        messagebox.showinfo("Thank You!", "✅ Thanks for confirming! This helps improve the system!")
    
    def on_wrong(self):
        correction = simpledialog.askstring(
            "Correction Needed",
            "What is the correct name of this food?",
            parent=self.root
        )
        if correction:
            self.detector.database.add_correction(
                self.current_food,
                correction,
                self.current_confidence
            )
            self.detector.database.update_statistics(False)
            messagebox.showinfo(
                "Thank You!",
                f"📚 Thanks for the correction! I've learned that this is {correction}!"
            )
    
    def load_image_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Food Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")]
        )
        if file_path:
            self.detector.detect_food(file_path)
    
    def load_image_url(self):
        url = simpledialog.askstring("Image URL", "Enter the URL of the food image:")
        if url:
            self.detector.detect_food(url)
    
    def use_camera(self):
        self.detector.detect_from_camera()
    
    def show(self):
        self.root.deiconify()  # Make window visible
        self.root.mainloop()
    
    def set_detector(self, detector):
        self.detector = detector
    
    def detect_from_camera(self):
        """Camera detection with enhanced feedback"""
        print("📷 Opening camera... Press 'c' to capture, 'q' to quit")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Compute centered capture rectangle
            h, w = frame.shape[:2]
            box_size = int(min(w, h) * 0.6)
            cx, cy = w // 2, h // 2
            x1, y1 = max(0, cx - box_size // 2), max(0, cy - box_size // 2)
            x2, y2 = min(w, cx + box_size // 2), min(h, cy + box_size // 2)

            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 255), 2)
            cv2.putText(overlay, "Press 'c' to capture, 'q' to quit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('🍽️ Food Detection Camera', overlay)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                crop_bgr = frame[y1:y2, x1:x2]
                if crop_bgr.size == 0:
                    crop_bgr = frame

                rgb_frame = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)

                tmpf = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                pil_image.save(tmpf.name, format='PNG')
                tmpf.close()

                print(f"\n📸 Captured image saved to {tmpf.name}; closing camera and uploading to app...")
                cap.release()
                cv2.destroyAllWindows()

                # Use the associated detector to process the saved image
                if hasattr(self, 'detector') and self.detector is not None:
                    self.detector.detect_food(tmpf.name, show_image=True, get_feedback=True)
                else:
                    # Fallback: attempt to call detect_food on self if present
                    try:
                        self.detect_food(tmpf.name, show_image=True, get_feedback=True)
                    except Exception:
                        print("⚠️ No detector attached to GUI; cannot process captured image.")
                return

            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

class MainWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🍽️ Food Detection System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        # Initialize detector and store current detection
        self.detector = AdvancedFoodDetector()
        self.current_result = None
        self.current_image = None
        self.setup_ui()
        # Allow detector to update this MainWindow when running camera captures
        try:
            self.detector.gui = self
        except Exception:
            pass
        
    def setup_ui(self):
        # Set window size
        self.root.geometry("1200x800")
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, 
                               text="🍽️ FOOD DETECTION SYSTEM",
                               font=('Arial', 24, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Buttons Frame
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=20)
        
        # File Analysis Button
        file_btn = tk.Button(btn_frame, 
                            text="📁 Analyze Image File",
                            command=self.analyze_file,
                            bg='#2196F3', fg='white',
                            font=('Arial', 12, 'bold'),
                            padx=20, pady=10)
        file_btn.pack(side=tk.LEFT, padx=10)
        
        # URL Analysis Button
        url_btn = tk.Button(btn_frame,
                           text="🌐 Analyze from URL",
                           command=self.analyze_url,
                           bg='#2196F3', fg='white',
                           font=('Arial', 12, 'bold'),
                           padx=20, pady=10)
        url_btn.pack(side=tk.LEFT, padx=10)
        
        # Camera Button
        camera_btn = tk.Button(btn_frame,
                              text="📷 Use Camera",
                              command=self.use_camera,
                              bg='#2196F3', fg='white',
                              font=('Arial', 12, 'bold'),
                              padx=20, pady=10)
        camera_btn.pack(side=tk.LEFT, padx=10)
        
        # Create horizontal paned window for side-by-side layout
        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True, pady=20)
        
        # Left side - Image display
        self.image_frame = ttk.LabelFrame(paned_window, text="📷 Input Image", padding="15")
        paned_window.add(self.image_frame, weight=1)
        
        self.image_label = ttk.Label(self.image_frame, text="No image loaded")
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Right side - Results
        right_frame = ttk.Frame(paned_window)
        paned_window.add(right_frame, weight=1)
        
        # Results Frame
        self.results_frame = ttk.LabelFrame(right_frame, text="🔍 Detection Results", padding="15")
        self.results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Initial message
        self.result_label = ttk.Label(self.results_frame,
                                    text="Select an option above to start",
                                    font=('Arial', 12))
        self.result_label.pack(pady=20)
        
        # Feedback Frame (initially hidden)
        self.feedback_frame = ttk.Frame(main_frame)
        
        # Feedback Label
        feedback_label = ttk.Label(self.feedback_frame,
                                 text="Was the detection correct?",
                                 font=('Arial', 14, 'bold'))
        feedback_label.pack(pady=(0, 10))
        
        # Feedback Buttons
        fb_btn_frame = ttk.Frame(self.feedback_frame)
        fb_btn_frame.pack()
        
        yes_btn = tk.Button(fb_btn_frame,
                           text="✅ YES, CORRECT!",
                           command=self.on_correct_feedback,
                           bg='#4CAF50', fg='white',
                           font=('Arial', 12, 'bold'),
                           padx=20, pady=10)
        yes_btn.pack(side=tk.LEFT, padx=10)
        
        no_btn = tk.Button(fb_btn_frame,
                          text="❌ NO, WRONG",
                          command=self.on_wrong_feedback,
                          bg='#f44336', fg='white',
                          font=('Arial', 12, 'bold'),
                          padx=20, pady=10)
        no_btn.pack(side=tk.LEFT, padx=10)
        
    def analyze_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")]
        )
        if file_path:
            self.process_image(file_path)
    
    def analyze_url(self):
        url = simpledialog.askstring("Image URL", "Enter the URL of the image:")
        if url:
            self.process_image(url)
    
    def use_camera(self):
        self.detector.detect_from_camera()
    
    def process_image(self, input_source):
        """Process image and show results"""
        try:
            # Load and store the input image
            if isinstance(input_source, str):
                if input_source.startswith('http'):
                    response = requests.get(input_source)
                    input_image = Image.open(BytesIO(response.content))
                else:
                    input_image = Image.open(input_source)
            else:
                input_image = input_source

            if input_image.mode != 'RGB':
                input_image = input_image.convert('RGB')

            # Process the image
            print("🔄 Processing image...")
            result = self.detector.detect_food(input_source, show_image=True, get_feedback=False)
            
            if result:
                # Store current detection info
                self.current_result = result
                self.current_image = input_image
                self.current_confidence = result['confidence']
                self.current_food = result['predicted_food']
                
                # Show results
                print(f"✨ Detected: {result['predicted_food']} ({result['confidence']:.1f}%)")
                self.show_results(result, input_image)
            else:
                print("❌ Could not process image")
                messagebox.showerror("Error", "Could not process the image. Please try another one.")
                
        except Exception as e:
            print(f"❌ Error processing image: {e}")
            messagebox.showerror("Error", f"Error processing image: {str(e)}")
            self.current_result = None
            self.current_image = None
    
    def show_results(self, result, input_image=None):
        # Update input image display
        if input_image:
            try:
                # Resize image while maintaining aspect ratio
                display_width = 400
                ratio = float(display_width) / input_image.size[0]
                display_height = int(input_image.size[1] * ratio)
                resized_img = input_image.resize((display_width, display_height), Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage
                self.current_photo = ImageTk.PhotoImage(resized_img)
                self.image_label.configure(image=self.current_photo)
                self.image_label.image = self.current_photo
            except Exception as e:
                print(f"Error displaying input image: {e}")
                self.image_label.configure(text="Error loading image")
        
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        # Create scrollable frame for results
        canvas = tk.Canvas(self.results_frame)
        scrollbar = ttk.Scrollbar(self.results_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Show new results
        food_label = ttk.Label(scrollable_frame,
                             text=f"🍽️ Detected Food: {result['predicted_food']}",
                             font=('Arial', 14, 'bold'))
        food_label.pack(pady=(10, 5), padx=10)
        
        # Confidence with color indicator
        confidence = result['confidence']
        if confidence > 80:
            conf_color = '#4CAF50'  # Green
            conf_icon = '✅'
        elif confidence > 50:
            conf_color = '#FFA726'  # Orange
            conf_icon = '⚠️'
        else:
            conf_color = '#F44336'  # Red
            conf_icon = '❓'
        
        conf_label = ttk.Label(scrollable_frame,
                             text=f"{conf_icon} Confidence: {confidence:.1f}%",
                             font=('Arial', 12))
        conf_label.pack(pady=5, padx=10)
        
        # Nutrition info
        nutr_frame = ttk.LabelFrame(scrollable_frame, text="🥗 Nutrition Facts", padding="10")
        nutr_frame.pack(pady=10, fill=tk.X, padx=20)
        
        nutrition = result['nutrition']
        
        # Styled nutrition labels
        nutrition_style = {'font': ('Arial', 11),
                         'anchor': 'w',
                         'padding': 5}
        
        ttk.Label(nutr_frame,
                 text=f"🔥 Calories: {nutrition['calories']}",
                 **nutrition_style).pack(fill=tk.X)
        
        ttk.Label(nutr_frame,
                 text=f"🥩 Protein: {nutrition['protein']}g",
                 **nutrition_style).pack(fill=tk.X)
        
        ttk.Label(nutr_frame,
                 text=f"🍚 Carbs: {nutrition['carbs']}g",
                 **nutrition_style).pack(fill=tk.X)
        
        ttk.Label(nutr_frame,
                 text=f"🥑 Fat: {nutrition['fat']}g",
                 **nutrition_style).pack(fill=tk.X)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Add feedback buttons
        feedback_frame = ttk.LabelFrame(scrollable_frame, text="💭 Your Feedback", padding="10")
        feedback_frame.pack(pady=10, fill=tk.X, padx=20)
        
        ttk.Label(feedback_frame,
                text="Was this detection correct?",
                font=('Arial', 12, 'bold')).pack(pady=10)
        
        # Feedback buttons frame
        btn_frame = ttk.Frame(feedback_frame)
        btn_frame.pack(pady=10)
        
        # Yes button (green)
        tk.Button(btn_frame,
                text="✅ YES, CORRECT!",
                command=self.on_correct_feedback,
                bg='#4CAF50',
                fg='white',
                font=('Arial', 12, 'bold'),
                width=15, height=1,
                cursor='hand2').pack(side=tk.LEFT, padx=5)
        
        # No button (red)
        tk.Button(btn_frame,
                text="❌ NO, WRONG",
                command=self.on_wrong_feedback,
                bg='#f44336',
                fg='white',
                font=('Arial', 12, 'bold'),
                width=15, height=1,
                cursor='hand2').pack(side=tk.LEFT, padx=5)
    
    def on_correct_feedback(self):
        if hasattr(self, 'current_result'):
            # Update statistics and save to database
            self.detector.database.update_statistics(True)
            # Add the correct prediction to learning data
            self.detector.database.add_correction(
                self.current_result['predicted_food'],
                self.current_result['predicted_food'],  # Same food name as it was correct
                self.current_result['confidence']
            )
            # Save database
            self.detector.database.save_database()
            
            messagebox.showinfo("Thank You!", "✅ Thanks for confirming! This helps improve the system!")
            
            # Clear results and feedback
            for widget in self.results_frame.winfo_children():
                widget.destroy()
            self.result_label = ttk.Label(self.results_frame,
                                        text="✅ Feedback recorded. Select an option above to analyze another image.",
                                        font=('Arial', 12))
            self.result_label.pack(pady=20)
            
            # Clear image
            self.image_label.configure(image='', text="No image loaded")
            self.current_photo = None
            
            # Print learning status
            print(f"📚 Learning: Confirmed {self.current_result['predicted_food']} with {self.current_result['confidence']:.1f}% confidence")
    
    def on_wrong_feedback(self):
        if hasattr(self, 'current_result'):
            correction = simpledialog.askstring(
                "Correction Needed",
                "What is the correct name of this food?",
                parent=self.root
            )
            if correction:
                # Update database with correction
                self.detector.database.add_correction(
                    self.current_result['predicted_food'],
                    correction.lower().strip(),
                    self.current_result['confidence']
                )
                # Update statistics
                self.detector.database.update_statistics(False)
                # Save database
                self.detector.database.save_database()
                
                messagebox.showinfo(
                    "Thank You!",
                    f"📚 Thanks! I've learned that this is {correction}!"
                )
                
                # Clear results
                for widget in self.results_frame.winfo_children():
                    widget.destroy()
                self.result_label = ttk.Label(self.results_frame,
                                            text="✅ Correction recorded. Select an option above to analyze another image.",
                                            font=('Arial', 12))
                self.result_label.pack(pady=20)
                
                # Clear image
                self.image_label.configure(image='', text="No image loaded")
                self.current_photo = None
                
                # Print learning status
                print(f"📚 Learning: Corrected {self.current_result['predicted_food']} to {correction}")
                print(f"🔄 Updating database with new information...")
    
    def run(self):
        self.root.mainloop()

    def update_results(self, image_obj, food_name, confidence, nutrition):
        """Bridge for AdvancedFoodDetector.display_results -> MainWindow.show_results

        This method is intentionally small: it builds the expected result dict and
        forwards to the existing show_results method so camera captures appear in UI.
        """
        result = {
            'predicted_food': food_name,
            'confidence': confidence,
            'nutrition': nutrition
        }
        try:
            self.show_results(result, image_obj)
        except Exception as e:
            print(f"⚠️ Error updating MainWindow results: {e}")

def main():
    """Enhanced main function with GUI"""
    print("🚀 Starting Food Detection System...")
    detector = AdvancedFoodDetector()
    detector.gui_feedback = SmartFeedbackGUI(detector)
    detector.gui_feedback.run()
    
    try:
        detector = AdvancedFoodDetector()
        
        while True:
            print(f"\n🔍 Choose an option:")
            print("1. 📁 Analyze image file")
            print("2. 🌐 Analyze image from URL")
            print("3. 📷 Use camera")
            print("4. 📊 View statistics")
            print("5. 🗃️ View database info")
            print("6. ❌ Exit")
            
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == '1':
                img_path = input("📁 Enter image file path: ").strip().strip('"')
                if img_path and os.path.exists(img_path):
                    detector.detect_food(img_path, get_feedback=True)
                else:
                    print("❌ File not found!")
            
            elif choice == '2':
                img_url = input("🌐 Enter image URL: ").strip()
                if img_url.startswith('http'):
                    detector.detect_food(img_url, get_feedback=True)
                else:
                    print("❌ Invalid URL!")
            
            elif choice == '3':
                detector.detect_from_camera()
            
            elif choice == '4':
                stats = detector.database.data['statistics']
                learning = detector.database.data['learning_data']
                print(f"\n📊 SYSTEM STATISTICS:")
                print(f"   Total Predictions: {stats['total_predictions']}")
                print(f"   Correct Predictions: {stats['correct_predictions']}")
                print(f"   Accuracy Rate: {stats['accuracy_rate']:.1f}%")
                print(f"   Total Corrections: {learning['total_corrections']}")
                print(f"   Last Updated: {learning['last_updated']}")
            
            elif choice == '5':
                foods_count = len(detector.database.data['foods'])
                user_added = sum(1 for food in detector.database.data['foods'].values() 
                               if food.get('source') == 'user_added')
                print(f"\n🗃️ DATABASE INFO:")
                print(f"   Total Foods: {foods_count}")
                print(f"   Default Foods: {foods_count - user_added}")
                print(f"   User Added Foods: {user_added}")
                print(f"   Database File: {detector.database.db_file}")
            
            elif choice == '6':
                print("👋 Thank you for using the Advanced Food Detection System!")
                break
            
            else:
                print("❌ Invalid choice. Please try again.")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Program interrupted by user.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    print("📦 Required packages: tensorflow opencv-python pillow matplotlib requests numpy")
    print("💻 Install with: pip install tensorflow opencv-python pillow matplotlib requests numpy")
    print("⚠️  Note: tkinter is built into Python\n")
    
    app = MainWindow()
    app.run()