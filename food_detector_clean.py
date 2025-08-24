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

class JSONFoodDatabase:
    """JSON-based food database with learning capabilities"""
    def __init__(self, db_file='food_nutrition_database.json'):
        self.db_file = db_file
        self.load_database()
    
    def load_database(self):
        """Load or create the food database"""
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
                print(f"✅ Loaded database with {len(self.data['foods'])} foods")
            except Exception as e:
                print(f"❌ Error loading database: {e}")
                self.create_default_database()
        else:
            self.create_default_database()
            print("📝 Created new food database")
    
    def create_default_database(self):
        """Create a new database with default values"""
        self.data = {
            "foods": {
                "apple": {"calories": 52, "protein": 0.3, "carbs": 14, "fat": 0.2},
                "banana": {"calories": 89, "protein": 1.1, "carbs": 23, "fat": 0.3},
                "pizza": {"calories": 266, "protein": 11, "carbs": 33, "fat": 10},
                "burger": {"calories": 295, "protein": 17, "carbs": 28, "fat": 14}
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
        self.save_database()
    
    def save_database(self):
        try:
            self.data['learning_data']['last_updated'] = datetime.now().isoformat()
            with open(self.db_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Error saving database: {e}")
    
    def get_nutrition(self, food_name):
        """Get nutrition information for a food item"""
        food_name = food_name.lower().strip()
        if food_name in self.data['foods']:
            return self.data['foods'][food_name]
        return {'calories': 'Unknown', 'protein': 'Unknown', 'carbs': 'Unknown', 'fat': 'Unknown'}
    
    def add_correction(self, predicted_food, correct_food, confidence):
        correction = {
            'predicted': predicted_food,
            'correct': correct_food.lower().strip(),
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }
        self.data['learning_data']['corrections'].append(correction)
        self.data['learning_data']['total_corrections'] += 1
        self.save_database()
        print(f"📚 Learning recorded: {predicted_food} → {correct_food}")
    
    def update_statistics(self, is_correct):
        self.data['statistics']['total_predictions'] += 1
        if is_correct:
            self.data['statistics']['correct_predictions'] += 1
        self.save_database()

class FoodDetector:
    def __init__(self):
        print("🚀 Initializing Food Detection System...")
        self.database = JSONFoodDatabase()
        print("🧠 Loading ResNet50 model...")
        self.model = ResNet50(weights='imagenet')
        print("✅ System initialized successfully!")
    
    def detect_food(self, img_input, show_image=True):
        print("🔍 Analyzing image...")
        
        # Process image
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
        
        # Prepare for model
        img_array = img.resize((224, 224))
        img_array = image.img_to_array(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)
        decoded = decode_predictions(predictions)[0]
        
        # Get best food prediction
        for _, name, confidence in decoded:
            if name in self.database.data['foods']:
                food_name = name
                confidence = float(confidence * 100)
                nutrition = self.database.get_nutrition(food_name)
                
                # Update GUI if available
                if hasattr(self, 'gui'):
                    self.gui.update_results(img, food_name, confidence, nutrition)
                
                return food_name, confidence, nutrition
        
        return "Unknown Food", 0, {'calories': 'Unknown', 'protein': 'Unknown', 'carbs': 'Unknown', 'fat': 'Unknown'}

class FoodDetectorGUI:
    def __init__(self, detector):
        self.detector = detector
        self.root = tk.Tk()
        self.root.title("🍽️ Food Detection System")
        self.root.geometry("1200x800")
        self.root.configure(bg='white')
        self.setup_ui()
    
    def setup_ui(self):
        # Main container with white background
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel for image
        image_frame = ttk.LabelFrame(main_frame, text="📷 Detected Image", padding="10")
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Image label with gray background
        self.image_label = tk.Label(image_frame, bg='#f0f0f0', text="No image loaded")
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Right panel
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Input section at top
        input_frame = ttk.LabelFrame(right_panel, text="📥 Load Image", padding="10")
        input_frame.pack(fill=tk.X, pady=(0, 20))
        
        btn_frame = ttk.Frame(input_frame)
        btn_frame.pack(pady=10)
        
        # Load buttons with consistent styling
        button_style = {'width': 15, 'padding': 5}
        
        ttk.Button(btn_frame,
                  text="📁 Load Image File",
                  command=self.load_image_file,
                  **button_style).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame,
                  text="🌐 Load from URL",
                  command=self.load_image_url,
                  **button_style).pack(side=tk.LEFT, padx=5)
        
        # Results section
        self.results_frame = ttk.LabelFrame(right_panel, text="🔍 Detection Results", padding="10")
        self.results_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Food name and confidence
        self.food_label = tk.Label(self.results_frame,
                               text="No food detected yet",
                               font=('Arial', 14, 'bold'),
                               fg='#333333',  # Dark gray text
                               bg='white')    # White background
        self.food_label.pack(fill=tk.X, pady=5, padx=10)
        
        self.confidence_label = tk.Label(self.results_frame,
                                    text="",
                                    font=('Arial', 12),
                                    fg='#666666',  # Medium gray text
                                    bg='white')    # White background
        self.confidence_label.pack(fill=tk.X, pady=5, padx=10)
        
        # Nutrition section
        self.nutrition_frame = ttk.LabelFrame(right_panel, text="🥗 Nutrition Facts", padding="10")
        self.nutrition_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Style for nutrition labels
        nutrition_style = {'font': ('Arial', 12),
                         'fg': '#333333',
                         'bg': '#f8f9fa',
                         'anchor': 'w'}
        
        self.calories_label = tk.Label(self.nutrition_frame, text="Calories: -", **nutrition_style)
        self.calories_label.pack(fill=tk.X, pady=2, padx=10)
        
        self.protein_label = tk.Label(self.nutrition_frame, text="Protein: -", **nutrition_style)
        self.protein_label.pack(fill=tk.X, pady=2, padx=10)
        
        self.carbs_label = tk.Label(self.nutrition_frame, text="Carbs: -", **nutrition_style)
        self.carbs_label.pack(fill=tk.X, pady=2, padx=10)
        
        self.fat_label = tk.Label(self.nutrition_frame, text="Fat: -", **nutrition_style)
        self.fat_label.pack(fill=tk.X, pady=2, padx=10)
        
        # Feedback section
        feedback_frame = ttk.LabelFrame(right_panel, text="💭 Your Feedback", padding="10")
        feedback_frame.pack(fill=tk.X)
        
        ttk.Label(feedback_frame,
                text="Was this detection correct?",
                font=('Arial', 12, 'bold')).pack(pady=10)
        
        # Feedback buttons
        btn_frame = ttk.Frame(feedback_frame)
        btn_frame.pack(pady=10)
        
        # Yes button (green)
        tk.Button(btn_frame,
                text="✅ YES, CORRECT!",
                command=self.on_correct,
                bg='#4CAF50',
                fg='white',
                font=('Arial', 12, 'bold'),
                width=20, height=2).pack(side=tk.LEFT, padx=5)
        
        # No button (red)
        tk.Button(btn_frame,
                text="❌ NO, WRONG",
                command=self.on_wrong,
                bg='#f44336',
                fg='white',
                font=('Arial', 12, 'bold'),
                width=20, height=2).pack(side=tk.LEFT, padx=5)
    
    def update_results(self, image_obj, food_name, confidence, nutrition):
        # Update image
        if image_obj:
            try:
                # Resize image while maintaining aspect ratio
                width = 500
                ratio = float(width) / image_obj.size[0]
                height = int(image_obj.size[1] * ratio)
                image = image_obj.resize((width, height), Image.Resampling.LANCZOS)
                
                # Keep a reference to prevent garbage collection
                self.photo = ImageTk.PhotoImage(image)
                self.image_label.configure(image=self.photo)
                
            except Exception as e:
                print(f"Error displaying image: {e}")
                self.image_label.configure(text="Error loading image")
        
        # Update detection results
        self.food_label.configure(text=f"🍽️ Detected: {food_name}")
        self.confidence_label.configure(text=f"📊 Confidence: {confidence:.1f}%")
        
        # Update nutrition info
        self.calories_label.configure(text=f"🔥 Calories: {nutrition['calories']}")
        self.protein_label.configure(text=f"🥩 Protein: {nutrition['protein']}g")
        self.carbs_label.configure(text=f"🍚 Carbs: {nutrition['carbs']}g")
        self.fat_label.configure(text=f"🥑 Fat: {nutrition['fat']}g")
        
        # Store current food info for feedback
        self.current_food = food_name
        self.current_confidence = confidence
    
    def on_correct(self):
        if hasattr(self, 'current_food'):
            self.detector.database.update_statistics(True)
            messagebox.showinfo("Thank You!", "✅ Thanks for confirming! This helps improve the system!")
    
    def on_wrong(self):
        if hasattr(self, 'current_food'):
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
                    f"📚 Thanks! I've learned that this is {correction}!"
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
    
    def run(self):
        self.root.mainloop()

def main():
    print("🚀 Starting Food Detection System...")
    detector = FoodDetector()
    gui = FoodDetectorGUI(detector)
    detector.gui = gui
    gui.run()

if __name__ == "__main__":
    main()
