import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 9 Kolom Wajib dari Panitia
TARGET_COLUMNS = [
    "Risk ID", "Risk Description", "Project Stage", "Project Category",
    "Risk Owner", "Mitigating Action", "Likelihood (1-10)", 
    "Impact (1-10)", "Risk Priority (low, med, high)"
]

def test_semantic_mapping_from_file(file_path):
    print(f"Membaca data dari: {file_path}")
    
    # 1. Mengekstrak langsung dari file (Mencoba Excel dulu, jika gagal coba CSV)
    try:
        df = pd.read_excel(file_path)
    except Exception:
        df = pd.read_csv(file_path)
        
    # Mengambil nama kolom, mengabaikan kolom kosong bawaan pandas seperti 'Unnamed: 0'
    extracted_columns = [str(c) for c in df.columns if 'Unnamed' not in str(c) and pd.notna(c)]
    
    print("\n[DAFTAR KOLOM HASIL EKSTRAKSI]")
    print(extracted_columns)
    print("-" * 50)
    
    print("\nMemuat model NLP (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Menghitung vektor semantik...\n")
    target_embeddings = model.encode(TARGET_COLUMNS)
    input_embeddings = model.encode(extracted_columns)
    
    sim_matrix = cosine_similarity(input_embeddings, target_embeddings)
    
    print("=== HASIL PEMETAAN DINAMIS (INPUT 4) ===")
    for i, col in enumerate(extracted_columns):
        best_idx = np.argmax(sim_matrix[i])
        best_score = sim_matrix[i][best_idx]
        best_target = TARGET_COLUMNS[best_idx]
        
        # Threshold: Jika kemiripannya di atas 55% (0.55), kita anggap cocok
        if best_score >= 0.55:
            print(f"✅ '{col}' \n   --> DIPETAKAN KE: '{best_target}' (Skor: {best_score:.2f})\n")
        else:
            print(f"❌ '{col}' \n   --> DIABAIKAN (Skor tertinggi hannya {best_score:.2f} dengan '{best_target}')\n")

if __name__ == "__main__":
    # Ganti string di bawah ini dengan nama file/path file Input 4 Anda jika berbeda
    FILE_PATH = "inputs/4. Moorgate Crossrail Register (Input).xlsx" 
    
    test_semantic_mapping_from_file(FILE_PATH)