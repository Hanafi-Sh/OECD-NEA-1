import pandas as pd
import os
import warnings

# Hanya menyaring peringatan spesifik dari library openpyxl yang berkaitan 
# dengan ekstensi file yang tidak didukung, agar tidak mengaburkan output.
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl.worksheet._reader")

def ratakan_header_excel(file_path, jumlah_baris_header=1):
    """
    Mendeteksi sheet dan baris header secara otomatis, lalu meratakan header 
    yang bertingkat/di-merge menjadi DataFrame datar (1D).
    """
    xls = pd.ExcelFile(file_path)
    best_df_raw = None
    best_sheet = None
    best_header_row_idx = -1
    highest_score = -1
    
    keywords = ['risk', 'description', 'impact', 'likelihood', 'probability', 'owner', 'action', 'category', 'status', 'mitigation', 'severity', 'id', 'ref']
    
    # 1. Fase Deteksi: Cari sheet dan baris header terbaik
    for sheet_name in xls.sheet_names:
        df_tmp = pd.read_excel(xls, sheet_name=sheet_name, header=None).head(50)
        if df_tmp.empty:
            continue
            
        for idx, row in df_tmp.iterrows():
            score = 0
            for val in row.values:
                if pd.notna(val) and isinstance(val, str):
                    cell_lower = val.lower()
                    for kw in keywords:
                        if kw in cell_lower:
                            score += 1
            
            if score > highest_score:
                highest_score = score
                best_sheet = sheet_name
                best_header_row_idx = idx

    if highest_score < 2:
        print(f"Warning: Tidak menemukan tabel risiko yang meyakinkan di {os.path.basename(file_path)}")
        return None

    # 2. Baca data dari sheet dan baris yang terpilih
    # Kita ambil dari baris header terpilih sampai jumlah_baris_header di bawahnya
    df_full = pd.read_excel(xls, sheet_name=best_sheet, header=None)
    
    # Pisahkan zona header dan zona data
    df_header = df_full.iloc[best_header_row_idx : best_header_row_idx + jumlah_baris_header].copy()
    df_data = df_full.iloc[best_header_row_idx + jumlah_baris_header:].copy()
    
    # Opt-in ke perilaku masa depan untuk menghindari FutureWarning downcasting
    pd.set_option('future.no_silent_downcasting', True)
    
    # 3. Retas masalah "Merged Cells" (Logika Manual Forward Fill)
    header_vals = df_header.values.tolist()
    for row_idx in range(len(header_vals)):
        last_val = None
        for col_idx in range(len(header_vals[row_idx])):
            val = header_vals[row_idx][col_idx]
            if pd.isna(val) or str(val).strip() == '':
                header_vals[row_idx][col_idx] = last_val
            else:
                last_val = val
    
    df_header = pd.DataFrame(header_vals)
    
    # 4. Rangkai baris header menjadi satu string deskriptif
    header_baru = []
    for col in df_header.columns:
        komponen = [str(val).strip() for val in df_header[col].values if pd.notna(val) and str(val).strip() != '']
        
        komponen_bersih = []
        for k in komponen:
            if not komponen_bersih or komponen_bersih[-1] != k:
                komponen_bersih.append(k)
        
        nama_kolom = "_".join(komponen_bersih)
        if not nama_kolom:
            nama_kolom = f"Kolom_{col}"
            
        header_baru.append(nama_kolom)
        
    # 5. Pasang header baru ke zona data
    df_data.columns = header_baru
    
    # 6. Pembersihan Akhir
    df_data = df_data.dropna(how='all', axis=0).dropna(how='all', axis=1)
    df_data = df_data.reset_index(drop=True)
    
    return df_data

def format_df_to_llm_text(df):
    """
    Mengonversi DataFrame menjadi list of string teks Key-Value
    yang sangat mudah dipahami oleh LLM.
    Nilai kosong (NaN) akan diabaikan agar tidak membingungkan LLM.
    """
    llm_texts = []
    for idx, row in df.iterrows():
        baris_teks = [f"--- [Item Risiko ke-{idx+1}] ---"]
        for col_name, val in row.items():
            if pd.notna(val) and str(val).strip() != '':
                baris_teks.append(f"{col_name}: {str(val).strip()}")
        llm_texts.append("\n".join(baris_teks))
    return llm_texts

# Path absolut ke folder 'inputs' terlepas dijalankan di mana
try:
    DIR_INI = os.path.dirname(os.path.abspath(__file__))
except NameError:
    DIR_INI = os.getcwd()

# Sesuaikan penggabungan path berdasarkan lokasi DIR_INI Anda saat ini
if os.path.basename(DIR_INI) == 'try':
    INPUT_DIR = os.path.abspath(os.path.join(DIR_INI, "..", "..", "inputs"))
else:
    INPUT_DIR = os.path.abspath(os.path.join(DIR_INI, "inputs"))

if os.path.exists(INPUT_DIR):
    excel_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.xlsx')]
    for f in sorted(excel_files):
        print(f"\n==============================================")
        print(f"--- Processing: {f} ---")
        full_path = os.path.join(INPUT_DIR, f)
        try:
            df_result = ratakan_header_excel(full_path)
            if df_result is not None:
                formatted_texts = format_df_to_llm_text(df_result)
                print(f"Berhasil mengekstrak {len(formatted_texts)} item risiko.")
                if len(formatted_texts) > 0:
                    print("Contoh Konversi Item Pertama untuk LLM:\n")
                    print(formatted_texts[0])
        except Exception as e:
            print(f"Gagal memproses {f}: {e}")
else:
    print(f"Error: Folder tidak ditemukan di {INPUT_DIR}")