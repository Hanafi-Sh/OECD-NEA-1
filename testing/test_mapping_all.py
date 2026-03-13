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

def load_columns_from_file(file_path):
    """
    Membaca file Excel secara dinamis. Otomatis mencari Sheet dan Baris Header 
    yang paling tepat berdasarkan kata kunci manajemen risiko.
    """
    # Kata kunci umum untuk mendeteksi baris yang merupakan Header Tabel Risiko
    keywords = ['risk', 'description', 'impact', 'likelihood', 'probability', 
                'owner', 'category', 'stage', 'mitigat', 'action', 'severity', 'name', 'id']
    
    # Membaca seluruh sheet tanpa asumsi header (header=None)
    try:
        sheets = pd.read_excel(file_path, sheet_name=None, header=None)
    except Exception:
        # Fallback jika ternyata filenya CSV
        df = pd.read_csv(file_path, header=None)
        sheets = {"Sheet1": df}

    best_sheet = None
    best_header_row = 0
    max_score = -1
    extracted_columns = []

    # Memindai setiap sheet
    for sheet_name, df in sheets.items():
        # Cek 15 baris pertama saja untuk mencari Header
        for idx, row in df.head(15).iterrows():
            # Ubah isi baris jadi string huruf kecil untuk pencarian
            row_values_lower = [str(val).lower() for val in row.values if pd.notna(val)]
            
            score = 0
            # 1. Tambah skor jika ada kata kunci (Risk, Impact, dll) di baris ini
            for val in row_values_lower:
                if any(kw in val for kw in keywords):
                    score += 1
            
            # 2. Tambah skor kepadatan (Header biasanya punya banyak kolom yang terisi, bukan cuma 1-2 sel)
            score += len(row_values_lower) * 0.1 

            # Simpan baris dengan skor tertinggi sebagai Header Pemenang
            if score > max_score:
                max_score = score
                best_sheet = sheet_name
                best_header_row = idx
                
                # Simpan nama kolom aslinya (bukan yang lowercase)
                extracted_columns = [str(val) for val in row.values if pd.notna(val)]

    print(f"  [Auto-Detect] Sheet Terpilih: '{best_sheet}' | Header di Baris ke-{best_header_row}")
    
    # Membersihkan nama kolom dari Unnamed atau spasi berlebih
    clean_columns = [c.strip() for c in extracted_columns if 'Unnamed' not in c and c.strip() != '']
    return clean_columns

# def load_columns_from_file(file_path):
#     """
#     Membaca file Excel dan mengekstrak nama kolomnya secara dinamis.
#     """
#     # Penanganan khusus untuk Input 1 karena headernya tersembunyi di bawah
#     if "1." in file_path:
#         df = pd.read_excel(file_path, header=2)
#         # df = pd.read_excel(file_path, sheet_name="Risk Register", skiprows=11)
#     else:
#         df = pd.read_excel(file_path)
        
#     # Mengambil nama kolom, membuang kolom kosong/Unnamed bawaan pandas
#     cols = [str(c) for c in df.columns if 'Unnamed' not in str(c) and pd.notna(c)]
#     return cols

def test_multiple_files(file_paths):
    print("Memuat model NLP (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    target_embeddings = model.encode(TARGET_COLUMNS)
    
    for file_path in file_paths:
        print(f"\n{'='*60}")
        print(f"MEMPROSES FILE: {file_path}")
        print(f"{'='*60}")
        try:
            input_cols = load_columns_from_file(f"inputs/{file_path}")
            print(f"Kolom yang berhasil diekstrak: {len(input_cols)} kolom")
            
            input_embeddings = model.encode(input_cols)
            sim_matrix = cosine_similarity(input_embeddings, target_embeddings)
            
            print("\n[HASIL PEMETAAN]")
            for i, col in enumerate(input_cols):
                best_idx = np.argmax(sim_matrix[i])
                best_score = sim_matrix[i][best_idx]
                best_target = TARGET_COLUMNS[best_idx]
                
                # Threshold: Jika kemiripannya di atas 55% (0.55)
                if best_score >= 0.55:
                    print(f"✅ '{col}' \n   --> {best_target} (Skor: {best_score:.2f})")
                else:
                    print(f"❌ '{col}' \n   --> DIABAIKAN (Skor tertinggi: {best_target} - {best_score:.2f})")
                    
        except Exception as e:
            print(f"Gagal memproses {file_path}. Pastikan file ada di folder yang benar.")
            print(f"Error: {e}")

if __name__ == "__main__":
    # Pastikan ketiga file ini berada di satu folder dengan script ini
    files_to_test = [
        "1. IVC DOE R2 (Input).xlsx",
        "2. City of York Council (Input).xlsx",
        "3. Digital Security IT Sample Register (Input).xlsx"
    ]
    
    test_multiple_files(files_to_test)