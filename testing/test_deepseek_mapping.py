import json
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# ==========================================
# KONFIGURASI API DEEPSEEK
# ==========================================
# Ganti dengan API Key dari akun DeepSeek Anda
API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Inisialisasi client menggunakan base_url khusus DeepSeek
client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com"
)

TARGET_COLUMNS = [
    "Risk ID", "Risk Description", "Project Stage", "Project Category",
    "Risk Owner", "Mitigating Action", "Likelihood (1-10)", 
    "Impact (1-10)", "Risk Priority (low, med, high)"
]

def map_columns_with_deepseek(input_columns, file_name):
    print(f"\nMeminta DeepSeek untuk memetakan kolom dari: {file_name}...")
    
    # PROMPT ENGINEERING: Instruksi tegas untuk DeepSeek
    system_prompt = """
    Kamu adalah seorang Data Scientist dan ahli Manajemen Risiko.
    Tugasmu adalah memetakan daftar nama kolom mentah ke dalam 9 Kolom Wajib standar.
    
    ATURAN MUTLAK:
    1. Cocokkan makna, bukan hanya huruf. (Contoh: 'SEV' atau 'Severity' = 'Impact (1-10)', 'FRQ' atau 'Probability' = 'Likelihood (1-10)', 'Strategy' = 'Mitigating Action').
    2. Jika kolom mentah sama sekali tidak relevan (seperti 'Date Added', '+ -', 'TYP', 'Revision Date'), petakan nilainya menjadi "IGNORE".
    3. HANYA KEMBALIKAN FORMAT JSON MURNI. Tanpa awalan ```json, tanpa penjelasan apapun.
    """

    user_prompt = f"""
    9 Kolom Wajib (Target): {TARGET_COLUMNS}
    Kolom Mentah (Input): {input_columns}
    """

    try:
        # Memanggil DeepSeek API
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            # Memaksa model untuk lebih deterministik (tidak berhalusinasi)
            temperature=0.0
        )
        
        # Mengekstrak teks dari respons DeepSeek
        output_text = response.choices[0].message.content.strip()
        
        # Membersihkan markdown jika DeepSeek masih membandel menambahkannya
        if output_text.startswith("```json"):
            output_text = output_text[7:]
        if output_text.endswith("```"):
            output_text = output_text[:-3]
            
        # Parse ke bentuk Dictionary Python
        mapping_result = json.loads(output_text.strip())
        
        print("✅ Hasil Pemetaan (JSON):")
        for raw_col, target_col in mapping_result.items():
            if target_col == "IGNORE":
                print(f"  ❌ '{raw_col}' --> DIABAIKAN")
            else:
                print(f"  ✅ '{raw_col}' --> {target_col}")
                
        return mapping_result

    except Exception as e:
        print(f"Gagal memanggil API DeepSeek: {e}")
        return None

if __name__ == "__main__":
    # Data dari Input 1 yang sangat teknis/engineering-heavy
    input_1_cols = [
        'Revision Date', 'RBS Level 1', 'RBS Level 2', 'Risk Name', 'TRL', 'TPL', '+ -', 'TYP', 
        'SEV', 'FRQ', 'RPN', 'Description (with assumptions)', 'Strategy', 
        'Secondary Risks Resulting From Risk Response', 'Recommendations & Action Items'
    ]
    
    # Data dari Input 3 yang memiliki sinonim beda ('Probability', 'Severity')
    input_3_cols = [
        'Date Added', 'Number', 'Risk Description', 'Project Stage', 
        'Risk Category', 'Probability', 'Severity', 'Score', 'Risk Owner', 'Action Plan'
    ]

    if API_KEY != "MASUKKAN_API_KEY_DEEPSEEK_ANDA_DISINI":
        map_columns_with_deepseek(input_1_cols, "Input 1 (IVC DOE)")
        print("-" * 50)
        map_columns_with_deepseek(input_3_cols, "Input 3 (Digital Security)")
    else:
        print("Silakan masukkan API Key DeepSeek Anda terlebih dahulu.")