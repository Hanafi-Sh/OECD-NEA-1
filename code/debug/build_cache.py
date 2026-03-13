import os
import pandas as pd
from extract_excel import extract_excel_data, format_df_to_llm_text
from extract_pdf import extract_pdf_data

DIR_INI = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.abspath(os.path.join(DIR_INI, "..", "..", "inputs"))
OUTPUT_DIR = os.path.abspath(os.path.join(DIR_INI, "..", "..", "outputs"))
CACHE_DIR = os.path.abspath(os.path.join(DIR_INI, "..", "..", "debug_cache"))

os.makedirs(CACHE_DIR, exist_ok=True)

def cache_inputs():
    print("\n--- CACHING INPUTS ---")
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.xlsx') or f.endswith('.pdf')])
    for fname in files:
        full_path = os.path.join(INPUT_DIR, fname)
        print(f"Baking {fname} into cache...")
        if fname.endswith('.xlsx'):
            df = extract_excel_data(full_path)
        elif fname.endswith('.pdf'):
            df = extract_pdf_data(fname)
            
        print(f"  Result Shape: {df.shape if df is not None else 'FAILED'}")

def cache_outputs():
    print("\n--- CACHING TARGET OUTPUTS ---")
    files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.xlsx') and 'Final' in f and 'Moorgate' not in f and 'Corporate' not in f])
    for fname in files:
        full_path = os.path.join(OUTPUT_DIR, fname)
        print(f"Baking {fname} into cache...")
        try:
            df = pd.read_excel(full_path)
            
            # Save CSV
            cache_csv = os.path.join(CACHE_DIR, f"{fname.replace('.xlsx', '.csv')}")
            df.to_csv(cache_csv, index=False)
            
            # Save LLM Text Format
            cache_txt = os.path.join(CACHE_DIR, f"{fname.replace('.xlsx', '_format.txt')}")
            with open(cache_txt, "w") as f:
                f.write("\n\n".join(format_df_to_llm_text(df)))
                
            print(f"  Result Shape: {df.shape}")
        except Exception as e:
            print(f"  Failed caching output {fname}: {e}")

if __name__ == "__main__":
    cache_inputs()
    cache_outputs()
    print(f"\nSuccessfully baked all inputs and outputs into DataFrames and LLM formatted text in '{CACHE_DIR}'!")
