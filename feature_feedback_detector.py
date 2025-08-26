"""Feature-feedback food detector

This module uses a pretrained ResNet50 to extract features (global average pooled)
and stores user corrections as feature vectors in a JSON-backed feedback store.

Prediction flow:
 - Extract features for the input image.
 - Search feedback store for a similar feature vector (cosine similarity).
   - If a close match exists (similarity >= threshold) return the corrected label.
   - Otherwise, fall back to the ResNet50 classifier (ImageNet decode_predictions).

Feedback persistence:
 - Stored in `feedback_store.json` alongside timestamps and optional notes.

Notes on Indian + world foods:
 - This file contains an extended `FOOD_KEYWORDS` set with many Indian and world
   food names to help filter ImageNet labels. For richer nutrition lookups use
   OpenFoodFacts, Spoonacular or Edamam APIs (recommendations in README/help).

Usage (quick):
    python feature_feedback_detector.py --image path/to/image.jpg
    python feature_feedback_detector.py --add-feedback path/to/image.jpg --label "dosa"

"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
from PIL import Image

try:
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.applications.resnet import preprocess_input, decode_predictions
    from tensorflow.keras.preprocessing import image as keras_image
except Exception as e:
    raise ImportError(
        "tensorflow and keras are required. Install with: pip install tensorflow"
    ) from e

FEEDBACK_FILE = os.path.join(os.path.dirname(__file__), 'feedback_store.json')

# A modest set of food keywords including common Indian dishes and world foods.
FOOD_KEYWORDS = {
    # common
    'pizza', 'burger', 'sandwich', 'pasta', 'noodle', 'spaghetti', 'rice', 'sushi',
    'taco', 'burrito', 'ramen', 'soup', 'salad', 'steak', 'chicken', 'fish', 'egg',
    'bread', 'cake', 'cookie', 'donut', 'ice cream', 'fries', 'french fries',
    # Indian
    'biryani', 'dosa', 'idli', 'samosa', 'vada', 'paratha', 'chapati', 'naan',
    'pakora', 'chutney', 'masala', 'dal', 'sambar', 'rasam', 'gulab jamun',
    'paneer', 'tikka', 'kebab',
    # other
    'pancake', 'waffle', 'curry', 'omelette', 'omelet', 'omelette', 'bento',
}


def _ensure_feedback_file():
    if not os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, 'w', encoding='utf-8') as f:
            json.dump({'entries': []}, f, indent=2)


class FeedbackStore:
    """Stores feedback entries (feature vectors + corrected labels) in JSON.

    Each entry:
      {
        'id': int,
        'label': str,
        'feature': [float...],
        'timestamp': isoformat,
        'notes': optional
      }
    """

    def __init__(self, path: str = FEEDBACK_FILE):
        self.path = path
        _ensure_feedback_file()
        self._load()

    def _load(self):
        with open(self.path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.entries: List[Dict[str, Any]] = data.get('entries', [])
        # Build feature matrix for fast search
        self._refresh_matrix()

    def _refresh_matrix(self):
        if not self.entries:
            self._features = None
            self._labels = []
            return
        self._features = np.array([np.array(e['feature'], dtype=np.float32) for e in self.entries])
        # normalize rows
        norms = np.linalg.norm(self._features, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._features = self._features / norms
        self._labels = [e['label'] for e in self.entries]

    def add(self, feature: np.ndarray, label: str, notes: Optional[str] = None) -> Dict[str, Any]:
        feature_list = feature.astype(float).tolist()
        entry = {
            'id': (self.entries[-1]['id'] + 1) if self.entries else 1,
            'label': label,
            'feature': feature_list,
            'timestamp': datetime.now().isoformat(),
            'notes': notes,
        }
        self.entries.append(entry)
        self._save()
        self._refresh_matrix()
        return entry

    def _save(self):
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump({'entries': self.entries}, f, indent=2)

    def search_similar(self, feature: np.ndarray, top_k: int = 1) -> List[Tuple[float, Dict[str, Any]]]:
        """Return list of (similarity, entry) sorted by descending similarity."""
        if self._features is None:
            return []
        # normalize input
        f = feature.astype(np.float32)
        fn = np.linalg.norm(f)
        if fn == 0:
            return []
        f = f / fn
        sims = np.dot(self._features, f)
        # clamp
        sims = np.clip(sims, -1.0, 1.0)
        # get top_k indices
        idx = np.argsort(-sims)[:top_k]
        results = [(float(sims[i]), self.entries[i]) for i in idx]
        return results


class FeatureFeedbackDetector:
    """Detector combining pretrained classifier + feature-feedback lookup."""

    def __init__(self, feedback_store: Optional[FeedbackStore] = None):
        # classifier (top) for labels
        self.classifier = ResNet50(weights='imagenet', include_top=True)
        # feature extractor (average pooled conv features)
        self.feature_extractor = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        self.feedback = feedback_store or FeedbackStore()

    def _load_image(self, img_input) -> Tuple[np.ndarray, Image.Image]:
        """Load an image from path/URL/PIL and return model array and original PIL image."""
        if isinstance(img_input, str) and img_input.startswith('http'):
            # remote URL input - keep simple by using requests here
            try:
                import requests
                from io import BytesIO
                resp = requests.get(img_input, timeout=10)
                img = Image.open(BytesIO(resp.content)).convert('RGB')
            except Exception as e:
                raise RuntimeError(f"Failed to load image from URL: {e}")
        elif isinstance(img_input, str):
            img = Image.open(img_input).convert('RGB')
        elif isinstance(img_input, Image.Image):
            img = img_input.convert('RGB')
        else:
            raise TypeError('img_input must be a file path, URL or PIL.Image')

        img_resized = img.resize((224, 224))
        x = keras_image.img_to_array(img_resized)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return x, img

    def _extract_feature(self, img_array: np.ndarray) -> np.ndarray:
        feat = self.feature_extractor.predict(img_array, verbose=0)
        return feat.reshape(-1)

    def _classify(self, img_array: np.ndarray, top: int = 5) -> List[Dict[str, Any]]:
        preds = self.classifier.predict(img_array, verbose=0)
        decoded = decode_predictions(preds, top=top)[0]
        results = []
        for _, name, prob in decoded:
            results.append({'name': name.replace('_', ' '), 'probability': float(prob)})
        return results

    def detect(self, img_input, similarity_threshold: float = 0.85) -> Dict[str, Any]:
        """Detects food and applies feature-feedback lookup.

        Returns a dict with keys: `label`, `confidence`, `source`, `model_predictions`.
        """
        img_array, pil_img = self._load_image(img_input)
        feature = self._extract_feature(img_array)

        # search feedback
        matches = self.feedback.search_similar(feature, top_k=1)
        if matches:
            sim, entry = matches[0]
            if sim >= similarity_threshold:
                # return corrected label from feedback
                return {
                    'label': entry['label'],
                    'confidence': sim * 100.0,
                    'source': 'feedback',
                    'matched_entry': entry,
                    'model_predictions': self._classify(img_array, top=5),
                }

        # fallback to model
        model_preds = self._classify(img_array, top=5)
        # try to choose a food-like label by matching keywords
        chosen = model_preds[0]
        for p in model_preds:
            lname = p['name'].lower()
            if any(k in lname for k in FOOD_KEYWORDS):
                chosen = p
                break

        return {
            'label': chosen['name'],
            'confidence': chosen['probability'] * 100.0,
            'source': 'model',
            'model_predictions': model_preds,
        }

    def add_feedback(self, img_input, corrected_label: str, notes: Optional[str] = None) -> Dict[str, Any]:
        """Add a feedback entry by extracting features from the corrected image."""
        img_array, _ = self._load_image(img_input)
        feature = self._extract_feature(img_array)
        entry = self.feedback.add(feature, corrected_label, notes=notes)
        return entry


def _cli():
    import argparse

    parser = argparse.ArgumentParser(description='Feature-feedback food detector (example)')
    parser.add_argument('--image', help='Image path or URL to classify')
    parser.add_argument('--add-feedback', help='Add feedback from image path or URL')
    parser.add_argument('--label', help='Label to associate when adding feedback')
    parser.add_argument('--threshold', type=float, default=0.85, help='Similarity threshold (0-1)')
    args = parser.parse_args()

    det = FeatureFeedbackDetector()

    if args.add_feedback:
        if not args.label:
            print('Provide --label when adding feedback')
            return
        entry = det.add_feedback(args.add_feedback, args.label)
        print('Feedback saved:')
        print(json.dumps(entry, indent=2))
        return

    if args.image:
        out = det.detect(args.image, similarity_threshold=args.threshold)
        print('\nDETECTION RESULT:\n')
        print(f"Label: {out['label']}")
        print(f"Confidence: {out['confidence']:.2f}%")
        print(f"Source: {out['source']}")
        if out.get('model_predictions'):
            print('\nTop model predictions:')
            for p in out['model_predictions']:
                print(f"  - {p['name']}: {p['probability']*100:.2f}%")
        return

    parser.print_help()


if __name__ == '__main__':
    _cli()
