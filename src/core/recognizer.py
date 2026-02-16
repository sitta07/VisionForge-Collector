"""
Medicine Recognizer using existing database system
Compatible with hospital_pills.db and hospital_boxes.db
"""

import os
import sqlite3
import cv2
import numpy as np
from pathlib import Path


class MedicineRecognizer:
    """
    Medicine recognition using database similarity matching
    """
    
    def __init__(self, base_dir=None):
        """
        Initialize recognizer
        
        Args:
            base_dir: Base directory path (default: auto-detect)
        """
        if base_dir is None:
            # Auto-detect base directory
            current_file = os.path.abspath(__file__)
            self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
        else:
            self.base_dir = base_dir
        
        self.current_mode = "pills"
        self.db_path = None
        self.conn = None
        self.cursor = None
        
        # Load database for default mode
        self._load_database()
    
    def _load_database(self):
        """Load database for current mode"""
        # Close previous connection if exists
        if self.conn:
            self.conn.close()
        
        # Determine database path based on mode
        db_name = f"hospital_{self.current_mode}.db"
        
        # Try multiple possible locations
        possible_paths = [
            os.path.join(self.base_dir, "data", self.current_mode, db_name),
            os.path.join(self.base_dir, "databases", db_name),
            os.path.join(self.base_dir, db_name),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                self.db_path = path
                break
        
        if not self.db_path:
            print(f"⚠️ Database not found for {self.current_mode} mode")
            print(f"   Searched in:")
            for p in possible_paths:
                print(f"   - {p}")
            return False
        
        try:
            # Connect to database
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            
            # Get table info
            self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = self.cursor.fetchall()
            
            print(f"✅ Database loaded: {os.path.basename(self.db_path)}")
            print(f"   Path: {self.db_path}")
            print(f"   Tables: {[t[0] for t in tables]}")
            
            # Count entries
            try:
                self.cursor.execute("SELECT COUNT(DISTINCT name) FROM drugs")
                count = self.cursor.fetchone()[0]
                print(f"   Entries: {count} medicine types")
            except Exception as e:
                print(f"   Could not count entries: {e}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load database: {e}")
            return False
    
    def set_mode(self, mode):
        """
        Change recognition mode
        
        Args:
            mode: "pills" or "boxes"
        """
        if mode != self.current_mode:
            self.current_mode = mode
            self._load_database()
    
    def recognize(self, image):
        """
        Recognize medicine from image
        
        Args:
            image: OpenCV image (BGR)
        
        Returns:
            List of recognition results [{"name": str, "confidence": float}, ...]
        """
        if not self.conn:
            return []
        
        try:
            # Extract features from query image
            query_features = self._extract_features(image)
            
            # Get all reference images from database
            matches = []
            
            # Try different table structures
            try:
                # Structure 1: drugs table with embedding column
                self.cursor.execute("""
                    SELECT name, embedding 
                    FROM drugs
                    WHERE embedding IS NOT NULL
                """)
                rows = self.cursor.fetchall()
                
                for name, features in rows:
                    if features:
                        # Compare features
                        ref_features = np.frombuffer(features, dtype=np.float32)
                        similarity = self._calculate_similarity(query_features, ref_features)
                        
                        # Check if this name already in matches
                        existing = next((m for m in matches if m["name"] == name), None)
                        if existing:
                            # Keep higher confidence
                            if similarity * 100 > existing["confidence"]:
                                existing["confidence"] = similarity * 100
                        else:
                            matches.append({
                                "name": name,
                                "confidence": similarity * 100
                            })
            
            except Exception as e1:
                print(f"❌ Database query failed: {e1}")
                return []
            
            # Sort by confidence (highest first)
            matches.sort(key=lambda x: x["confidence"], reverse=True)
            
            # Return top 3 matches
            return matches[:3] if matches else []
            
        except Exception as e:
            print(f"❌ Recognition error: {e}")
            return []
    
    def _extract_features(self, image):
        """
        Extract simple features from image
        
        Args:
            image: OpenCV image (BGR)
        
        Returns:
            Feature vector (numpy array)
        """
        # Resize to standard size
        img_resized = cv2.resize(image, (128, 128))
        
        # Extract color histogram
        features = []
        for channel in range(3):
            hist = cv2.calcHist([img_resized], [channel], None, [32], [0, 256])
            features.extend(hist.flatten())
        
        # Extract texture features (gradient)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        grad_hist_x = np.histogram(grad_x.flatten(), bins=16, range=(-256, 256))[0]
        grad_hist_y = np.histogram(grad_y.flatten(), bins=16, range=(-256, 256))[0]
        
        features.extend(grad_hist_x)
        features.extend(grad_hist_y)
        
        # Normalize
        features = np.array(features, dtype=np.float32)
        features = features / (np.linalg.norm(features) + 1e-6)
        
        return features
    
    def _calculate_similarity(self, feat1, feat2):
        """
        Calculate similarity between two feature vectors
        
        Args:
            feat1, feat2: Feature vectors
        
        Returns:
            Similarity score (0-1)
        """
        # Ensure same length
        min_len = min(len(feat1), len(feat2))
        feat1 = feat1[:min_len]
        feat2 = feat2[:min_len]
        
        # Cosine similarity
        dot_product = np.dot(feat1, feat2)
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        # Convert to 0-1 range (from -1 to 1)
        similarity = (similarity + 1) / 2
        
        return similarity
    
    def _decode_image(self, img_data):
        """
        Decode image from database blob
        
        Args:
            img_data: Binary image data
        
        Returns:
            OpenCV image or None
        """
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(img_data, np.uint8)
            # Decode image
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print(f"⚠️ Failed to decode image: {e}")
            return None
    
    def get_medicine_info(self, name):
        """
        Get detailed information about a medicine
        
        Args:
            name: Medicine name
        
        Returns:
            Dictionary with medicine information or None
        """
        if not self.conn:
            return None
        
        try:
            self.cursor.execute("""
                SELECT * FROM drugs WHERE name = ?
            """, (name,))
            
            row = self.cursor.fetchone()
            if row:
                columns = [desc[0] for desc in self.cursor.description]
                return dict(zip(columns, row))
            
        except Exception as e:
            print(f"⚠️ Could not get medicine info: {e}")
        
        return None
    
    def __del__(self):
        """Close database connection on cleanup"""
        if self.conn:
            self.conn.close()


# Test code
if __name__ == "__main__":
    recognizer = MedicineRecognizer()
    
    print("\n--- Testing Recognizer ---")
    
    # Create dummy test image
    test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    results = recognizer.recognize(test_img)
    
    if results:
        print("\nRecognition results:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['name']} ({result['confidence']:.1f}%)")
    else:
        print("\nNo results found")