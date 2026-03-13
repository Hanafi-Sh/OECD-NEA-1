import pandas as pd
import json
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# KONFIGURASI API DEEPSEEK
# ==========================================
API_KEY = os.getenv("DEEPSEEK_API_KEY")
client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com"
)

TARGET_COLUMNS = [
    "Risk ID", "Risk Description", "Project Stage", "Project Category",
    "Risk Owner", "Mitigating Action", "Likelihood (1-10)", 
    "Impact (1-10)", "Risk Priority (low, med, high)"
]

def load_columns_from_file_dynamically(file_path):
    """
    Membuka file Excel, mencari sheet dan baris header yang tepat, 
    lalu mengekstrak nama kolomnya.
    """
    print(f"\n{'='*60}")
    print(f"MEMPROSES FILE: {file_path}")
    print(f"{'='*60}")
    
    keywords = ['risk', 'description', 'impact', 'likelihood', 'probability', 
                'owner', 'category', 'stage', 'mitigat', 'action', 'severity', 'name', 'id']
    
    try:
        # Baca semua sheet tanpa header
        sheets = pd.read_excel(file_path, sheet_name=None, header=None)
    except Exception:
        # Fallback jika CSV
        try:
            df = pd.read_csv(file_path, header=None)
            sheets = {"Sheet1": df}
        except Exception as e:
            print(f"Gagal membuka file {file_path}: {e}")
            return []

    best_sheet = None
    best_header_row = 0
    max_score = -1
    extracted_columns = []

    for sheet_name, df in sheets.items():
        for idx, row in df.head(15).iterrows():
            row_values_lower = [str(val).lower() for val in row.values if pd.notna(val)]
            
            score = 0
            for val in row_values_lower:
                if any(kw in val for kw in keywords):
                    score += 1
            
            score += len(row_values_lower) * 0.1 

            if score > max_score:
                max_score = score
                best_sheet = sheet_name
                best_header_row = idx
                extracted_columns = [str(val) for val in row.values if pd.notna(val)]

    print(f"  [Auto-Detect] Sheet: '{best_sheet}' | Header Row: {best_header_row}")
    
    clean_columns = [c.strip() for c in extracted_columns if 'Unnamed' not in c and c.strip() != '']
    return clean_columns

def map_columns_with_deepseek(input_columns, file_name):
    """
    Mengirimkan daftar kolom yang diekstrak ke DeepSeek untuk dipetakan.
    """
    if not input_columns:
        print("  ❌ Tidak ada kolom yang bisa diproses.")
        return None
        
    print("  [API] Mengirim data ke DeepSeek untuk dipetakan...")
    
    system_prompt = """
    Kamu adalah seorang Data Scientist dan ahli Manajemen Risiko.
    Tugasmu adalah memetakan daftar nama kolom mentah ke dalam 9 Kolom Wajib standar.
    
    ATURAN MUTLAK:
    1. Cocokkan makna, bukan hanya huruf. (Contoh: 'SEV' atau 'Severity' = 'Impact (1-10)', 'FRQ' atau 'Probability' = 'Likelihood (1-10)', 'Strategy' = 'Mitigating Action').
    2. 'RPN' = 'Risk Priority (low, med, high)'.
    3. Jika kolom mentah sama sekali tidak relevan (seperti 'Date Added', '+ -', 'TYP', 'Revision Date'), petakan nilainya menjadi "IGNORE".
    4. HANYA KEMBALIKAN FORMAT JSON MURNI. Tanpa awalan ```json, tanpa penjelasan apapun.
    """

    user_prompt = f"""
    9 Kolom Wajib (Target): {TARGET_COLUMNS}
    Kolom Mentah (Input): {input_columns}
    """

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        
        output_text = response.choices[0].message.content.strip()
        
        if output_text.startswith("```json"):
            output_text = output_text[7:]
        if output_text.endswith("```"):
            output_text = output_text[:-3]
            
        mapping_result = json.loads(output_text.strip())
        
        print("  ✅ Hasil Pemetaan dari DeepSeek:")
        for raw_col, target_col in mapping_result.items():
            if target_col == "IGNORE":
                print(f"      ❌ '{raw_col}' --> DIABAIKAN")
            else:
                print(f"      ✅ '{raw_col}' --> {target_col}")
                
        return mapping_result

    except Exception as e:
        print(f"  ❌ Gagal memanggil API DeepSeek: {e}")
        return None

if __name__ == "__main__":
    # Daftar file asli yang ada di folder Anda
    files_to_process = [
        "1. IVC DOE R2 (Input).xlsx",
        "2. City of York Council (Input).xlsx",
        "3. Digital Security IT Sample Register (Input).xlsx",
        "4. Moorgate Crossrail Register (Input).xlsx"
    ]
    
    if API_KEY == "MASUKKAN_API_KEY_DEEPSEEK_ANDA_DISINI":
        print("PENTING: Silakan masukkan API Key DeepSeek Anda terlebih dahulu!")
    else:
        for file_path in files_to_process:
            # 1. Buka file dan ekstrak kolom secara dinamis
            extracted_cols = load_columns_from_file_dynamically(f"inputs/{file_path}")
            
            # 2. Petakan menggunakan AI
            if extracted_cols:
                map_columns_with_deepseek(extracted_cols, file_path)