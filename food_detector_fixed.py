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

class FoodDetectorGUI:
    def __init__(self, detector):
        self.detector = detector
        self.root = tk.Tk()
        self.root.title("🍽️ Food Detection System")
        self.root.geometry("1200x800")
        self.root.configure(bg='white')
        self.setup_ui()
    
    def setup_ui(self):
        # Main container
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
        
        ttk.Button(btn_frame, text="📁 Load Image File",
                  command=self.load_image_file).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="🌐 Load from URL",
                  command=self.load_image_url).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="📷 Use Camera",
                  command=self.use_camera).pack(side=tk.LEFT, padx=5)
        
        # Results section
        self.results_frame = ttk.LabelFrame(right_panel, text="🔍 Detection Results", padding="10")
        self.results_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Food name and confidence
        self.food_label = tk.Label(self.results_frame,
                               text="No food detected yet",
                               font=('Arial', 14, 'bold'),
                               fg='#333333',
                               bg='white')
        self.food_label.pack(fill=tk.X, pady=5, padx=10)
        
        self.confidence_label = tk.Label(self.results_frame,
                                    text="",
                                    font=('Arial', 12),
                                    fg='#666666',
                                    bg='white')
        self.confidence_label.pack(fill=tk.X, pady=5, padx=10)
        
        # Nutrition section
        self.nutrition_frame = ttk.LabelFrame(right_panel, text="🥗 Nutrition Facts (per 100g)", padding="10")
        self.nutrition_frame.pack(fill=tk.X, pady=(0, 20))
        
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
        
        # Buttons frame
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
        if image_obj:
            try:
                # Resize image while maintaining aspect ratio
                width = 500
                ratio = float(width) / image_obj.size[0]
                height = int(image_obj.size[1] * ratio)
                image = image_obj.resize((width, height), Image.Resampling.LANCZOS)
                
                # Keep a reference to the photo to prevent garbage collection
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
            self.detector.detect_food(file_path, show_image=False)
    
    def load_image_url(self):
        url = simpledialog.askstring("Image URL", "Enter the URL of the food image:")
        if url:
            self.detector.detect_food(url, show_image=False)
    
    def use_camera(self):
        self.detector.detect_from_camera()
    
    def run(self):
        self.root.mainloop()

def main():
    """Enhanced main function with GUI"""
    print("🚀 Starting Food Detection System...")
    detector = AdvancedFoodDetector()
    gui = FoodDetectorGUI(detector)
    detector.gui = gui
    gui.run()

if __name__ == "__main__":
    main()
