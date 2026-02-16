import os
import sqlite3
import numpy as np
import faiss


class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

        self._create_table()

        # FAISS
        self.index = None
        self.id_map = []

        self.load_faiss_index()

    # ==============================
    # DB SETUP
    # ==============================

    def _create_table(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS drugs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                embedding BLOB,
                image_path TEXT
            )
        """)
        self.conn.commit()

        # 🔥 ตรวจ schema ปัจจุบัน
        self.cursor.execute("PRAGMA table_info(drugs)")
        columns = [row[1] for row in self.cursor.fetchall()]

        if "embedding" not in columns:
            print("⚠️ Adding missing embedding column...")
            self.cursor.execute("ALTER TABLE drugs ADD COLUMN embedding BLOB")

        if "image_path" not in columns:
            print("⚠️ Adding missing image_path column...")
            self.cursor.execute("ALTER TABLE drugs ADD COLUMN image_path TEXT")

        self.conn.commit()


    # ==============================
    # ADD ENTRY
    # ==============================

    def add_entry(self, name, vector, image_path):
        if vector is not None:
            vector = vector.astype("float32")
            emb_blob = vector.tobytes()
        else:
            emb_blob = None

        self.cursor.execute(
            "INSERT INTO drugs (name, embedding, image_path) VALUES (?, ?, ?)",
            (name, emb_blob, image_path)
        )
        self.conn.commit()

        # rebuild FAISS index
        self.load_faiss_index()

    # ==============================
    # DELETE CLASS
    # ==============================

    def delete_class(self, name):
        self.cursor.execute("DELETE FROM drugs WHERE name = ?", (name,))
        self.conn.commit()

        self.load_faiss_index()

    # ==============================
    # STATS
    # ==============================

    def get_stats(self):
        self.cursor.execute("""
            SELECT name, COUNT(*) 
            FROM drugs 
            GROUP BY name
        """)
        return self.cursor.fetchall()

    # ==============================
    # LOAD FAISS INDEX
    # ==============================

    def load_faiss_index(self):
        self.cursor.execute("SELECT name, embedding FROM drugs WHERE embedding IS NOT NULL")
        rows = self.cursor.fetchall()

        vectors = []
        self.id_map = []

        for name, emb_blob in rows:
            vec = np.frombuffer(emb_blob, dtype=np.float32)
            if vec.size > 0:
                vectors.append(vec)
                self.id_map.append(name)

        if len(vectors) == 0:
            self.index = None
            return

        vectors = np.vstack(vectors).astype("float32")

        dim = vectors.shape[1]

        # Cosine similarity = Inner Product + normalize
        self.index = faiss.IndexFlatIP(dim)

        # Normalize for cosine similarity
        faiss.normalize_L2(vectors)

        self.index.add(vectors)

        print(f"✅ FAISS index loaded with {len(vectors)} vectors")

    # ==============================
    # SEARCH (FAST)
    # ==============================

    def search(self, query_vector, threshold=0.75):
        if self.index is None:
            return None, 0.0

        vec = np.array([query_vector]).astype("float32")
        faiss.normalize_L2(vec)

        scores, indices = self.index.search(vec, 1)

        score = float(scores[0][0])
        idx = int(indices[0][0])

        if idx < 0:
            return None, score

        if score >= threshold:
            return self.id_map[idx], score

        return None, score
    # ใน class DatabaseManager

    def get_all_vectors(self):
        """
        Return all entries with embeddings as list of dicts:
        [{'name': name, 'vector': np.array([...], dtype=float32)}, ...]
        """
        self.cursor.execute("SELECT name, embedding FROM drugs WHERE embedding IS NOT NULL")
        rows = self.cursor.fetchall()
        data = []
        for name, emb in rows:
            vec = np.frombuffer(emb, dtype=np.float32)  # assuming embedding stored as BLOB
            data.append({"name": name, "vector": vec})
        return data


    # ==============================
    # CLOSE
    # ==============================

    def close(self):
        self.conn.close()
