import sqlite3
import time
import numpy as np
import os

class DatabaseManager:
    def __init__(self, db_path):
        """
        db_path: รับ Path มาจาก config.py (ตามโหมดที่เลือก)
        """
        self.db_path = db_path
        self.ensure_directory()
        self.connect_db()

    def ensure_directory(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def connect_db(self):
        # check_same_thread=False เพื่อให้ GUI Thread เรียกใช้ได้ไม่พัง
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        
        # 🚀 PERFORMANCE TUNING: WAL Mode = เร็ว + ปลอดภัย (Production Grade)
        self.conn.execute("PRAGMA journal_mode=WAL")  
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.create_tables()

    def create_tables(self):
        # สร้าง Table เดียวแต่ใช้ได้ครอบจักรวาล
        query = """
        CREATE TABLE IF NOT EXISTS drugs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            vector BLOB,          -- เก็บ Vector เป็น Binary
            img_path TEXT,
            timestamp REAL
        );
        -- สร้าง Index ที่ชื่อยา เพื่อให้การ Group/Count เร็วจัดๆ
        CREATE INDEX IF NOT EXISTS idx_drug_name ON drugs(name);
        """
        self.conn.executescript(query)
        self.conn.commit()

    def add_entry(self, name, vector_np, img_path):
        """เพิ่มข้อมูลยาใหม่"""
        try:
            # แปลง Numpy -> Bytes (BLOB)
            if vector_np is not None and len(vector_np) > 0:
                vector_blob = vector_np.astype(np.float32).tobytes()
            else:
                vector_blob = None

            with self.conn: # Auto-commit transaction
                self.conn.execute(
                    "INSERT INTO drugs (name, vector, img_path, timestamp) VALUES (?, ?, ?, ?)",
                    (name, vector_blob, img_path, time.time())
                )
            return True
        except Exception as e:
            print(f"❌ DB Insert Error: {e}")
            return False

    def delete_class(self, class_name):
        """ลบยาทั้ง Class (เช่น ลบ Paracetamol ทั้งหมด)"""
        try:
            with self.conn:
                cursor = self.conn.execute("DELETE FROM drugs WHERE name = ?", (class_name,))
                return cursor.rowcount
        except Exception as e:
            print(f"❌ DB Delete Error: {e}")
            return 0

    def get_stats(self):
        """ดึงสถิติสำหรับหน้า Analytics (Count by Name)"""
        try:
            cursor = self.conn.cursor()
            # ใช้ SQL Group By ซึ่งเร็วกว่า Python Loop ล้านเท่า
            cursor.execute("SELECT name, COUNT(*) FROM drugs GROUP BY name ORDER BY COUNT(*) DESC")
            return cursor.fetchall() # Returns [(name, count), ...]
        except Exception as e:
            print(f"❌ Stats Error: {e}")
            return []

    def get_all_vectors(self):
        """โหลด Vector ขึ้น RAM เพื่อทำ Live Search"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT name, vector FROM drugs WHERE vector IS NOT NULL")
            rows = cursor.fetchall()
            
            data = []
            for name, blob in rows:
                if blob:
                    vec = np.frombuffer(blob, dtype=np.float32)
                    data.append({'name': name, 'vector': vec})
            return data
        except Exception as e:
            print(f"❌ Load Vector Error: {e}")
            return []
    
    def close(self):
        if self.conn:
            self.conn.close()