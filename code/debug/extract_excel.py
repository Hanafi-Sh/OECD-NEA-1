import os
import pandas as pd
import warnings

# Filter out specific openpyxl warnings related to unsupported data validation extensions.
# This ensures a clean terminal output, as these warnings do not affect data extraction.
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl.worksheet._reader")

def extract_excel_data(filepath, jumlah_baris_header=1):
    """
    Extracts tabular data from complex Excel files automatically.
    
    This function performs a "Smart Target Detection":
    1. Iterates through all sheets in the Excel file.
    2. Uses a 'Structural Density + Keyword Scoring' algorithm to find the exact row
       where the true table header begins (ignoring titles, covers, and empty rows).
    3. Handles merged/multi-line header cells by dynamically forward-filling missing values.
    4. Cleans the data by dropping empty rows, duplicate rows, and formula ghost rows.
    
    Args:
        filepath (str): The absolute path to the Excel file.
        jumlah_baris_header (int): Number of rows expected to form the header (default is 1).
                                   For complex multi-tier headers, this can be increased, 
                                   but 1 is generally sufficient to capture the table boundary.
                                   
    Returns:
        pd.DataFrame: A flattened, cleaned pandas DataFrame ready for LLM consumption.
                      Returns None if no valid table is found or an error occurs.
    """
    cache_dir = os.path.abspath(os.path.join(os.path.dirname(filepath), "..", "..", "debug_cache"))
    cache_csv = os.path.join(cache_dir, f"{os.path.basename(filepath)}.csv")
    if os.path.exists(cache_csv):
        print(f"  [CACHE HIT] Loading Excel '{os.path.basename(filepath)}' directly from '{cache_dir}'...")
        return pd.read_csv(cache_csv)
        
    try:
        xls = pd.ExcelFile(filepath)
    except Exception as e:
        print(f"Error opening Excel file {filepath}: {e}")
        return None

    best_sheet = None
    best_header_row_idx = -1
    highest_score = -1
    
    # Common keywords typically found in the header of a Risk Register
    keywords = ['risk', 'description', 'impact', 'likelihood', 'probability', 'owner', 'action', 'category', 'status', 'mitigation', 'severity', 'id', 'ref']
    
    # ==========================================
    # PHASE 1: Detection (Scoring Algorithm)
    # ==========================================
    # We iterate through all sheets and inspect the first 50 rows to find the most probable table header.
    for sheet_name in xls.sheet_names:
        df_tmp = pd.read_excel(xls, sheet_name=sheet_name, header=None).head(50)
        if df_tmp.empty:
            continue
            
        for idx, row in df_tmp.iterrows():
            # A. Fill-Rate / Density Check
            # Calculate how many cells in this row actually contain valid text
            non_null_cells = [val for val in row.values if pd.notna(val) and str(val).strip() != '']
            fill_count = len(non_null_cells)
            
            # If the row is almost empty (e.g., project title), it cannot be a column header.
            if fill_count < 3:
                continue
                
            # B. Keyword Matching
            # Headers usually contain specific words. We heavily reward the row if these are found.
            keyword_matches = 0
            string_cells = [val for val in non_null_cells if isinstance(val, str)]
            for val in string_cells:
                # To prevent false positives from long paragraphs inside the table data, 
                # we only consider cells with short strings (typical of column names).
                if len(val) > 60:
                    continue
                
                cell_lower = val.lower()
                for kw in keywords:
                    if kw in cell_lower:
                        keyword_matches += 1
                        
            # Base score is row density. Massive bonus applied for each matched keyword (+50).
            score = fill_count + (keyword_matches * 50)  
            
            # C. Type Consistency Bonus & Depth Penalty
            # Headers are typically entirely string-based.
            if len(string_cells) == fill_count:
                score += 10
            # Apply a minor penalty for depth (row index) so that in case of a tie, 
            # the topmost candidate row wins.
            score -= idx  
            
            # Update the best candidate if the current score is higher.
            if score > highest_score:
                highest_score = score
                best_sheet = sheet_name
                best_header_row_idx = idx

    # If the highest score is very low, it indicates no valid table was found.
    if highest_score < 10:
        print(f"Warning: No valid structured table found in {os.path.basename(filepath)}")
        return None

    # ==========================================
    # PHASE 2: Data Extraction & Header Assembly
    # ==========================================
    # Re-read the winning sheet without a predefined header so we can manually process it.
    df_full = pd.read_excel(xls, sheet_name=best_sheet, header=None)
    
    # Segregate the dataframe into the 'Header Zone' and 'Data Zone'
    df_header = df_full.iloc[best_header_row_idx : best_header_row_idx + jumlah_baris_header].copy()
    df_data = df_full.iloc[best_header_row_idx + jumlah_baris_header:].copy()
    
    # Opt-in to the future pandas behavior to avoid specific deprecation warnings
    pd.set_option('future.no_silent_downcasting', True)
    
    # Hack/Fix for "Merged Cells" in Excel Headers
    # In Excel, merged cells often result in a value in the first column and NaNs in subsequent columns.
    # Here, we manually perform a forward-fill horizontally to restore the context.
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
    
    # Concatenate multi-tier headers into a single descriptive string per column
    # Example: "Baseline Risk" + "Description" -> "Baseline Risk_Description"
    header_baru = []
    for col in df_header.columns:
        komponen = [str(val).strip() for val in df_header[col].values if pd.notna(val) and str(val).strip() != '']
        
        # Remove consecutive duplicate labels resulting from vertical merging
        komponen_bersih = []
        for k in komponen:
            if not komponen_bersih or komponen_bersih[-1] != k:
                komponen_bersih.append(k)
        
        nama_kolom = "_".join(komponen_bersih)
        if not nama_kolom:
            nama_kolom = f"Kolom_{col}"
            
        header_baru.append(nama_kolom)
        
    # Apply the flattened headers to the Data Zone
    df_data.columns = header_baru
    
    # ==========================================
    # PHASE 3: Data Cleaning
    # ==========================================
    # Drop rows and columns that are 100% empty (NaN)
    df_data = df_data.dropna(how='all', axis=0).dropna(how='all', axis=1)
    
    # Filter out empty "ghost" rows (artifacts generated from dragging empty Excel formulas).
    # We consider a row to be valid only if it contains at least 2 cells with substantive text.
    valid_rows = []
    for idx, row in df_data.iterrows():
        text_cols = 0
        for val in row.values:
            if pd.notna(val):
                val_str = str(val).strip()
                # Consider it valid text if length > 1 and it's not a mathematical null indicator
                if len(val_str) > 1 and val_str not in ['0.0', 'NaN', 'None']:
                    text_cols += 1
        valid_rows.append(text_cols >= 2)
        
    df_data = df_data[valid_rows]
    
    # Drop duplicate rows (perfect matches) to clean up repetitive noise
    df_data = df_data.drop_duplicates()
    
    df_data = df_data.reset_index(drop=True)
    
    # Save cache
    cache_dir = os.path.abspath(os.path.join(os.path.dirname(filepath), "..", "..", "debug_cache"))
    os.makedirs(cache_dir, exist_ok=True)
    
    filename = os.path.basename(filepath)
    cache_csv = os.path.join(cache_dir, f"{filename}.csv")
    cache_txt = os.path.join(cache_dir, f"{filename}_format.txt")
    
    df_data.to_csv(cache_csv, index=False)
    with open(cache_txt, "w") as f:
        f.write("\n\n".join(format_df_to_llm_text(df_data)))
        
    return df_data

def format_df_to_llm_text(df):
    """
    Converts the processed pandas DataFrame into a list of semantic Key-Value texts.
    
    This format is highly optimized for LLM consumption. It represents each row as a
    vertical list of its column-value pairs, preventing the LLM from losing context 
    which often happens with wide or sparse tabular data (e.g., Markdown tables).
    
    Args:
        df (pd.DataFrame): The cleaned DataFrame.
        
    Returns:
        list of str: A list where each element is a formatted text block representing a row.
    """
    if df is None or df.empty:
        return []
        
    llm_texts = []
    for idx, row in df.iterrows():
        baris_teks = [f"--- [Risk Item {idx+1}] ---"]
        for col_name, val in row.items():
            if pd.notna(val) and str(val).strip() != '':
                baris_teks.append(f"{col_name}: {str(val).strip()}")
            else:
                # Explicitly flag empty columns. Knowing a column exists but is empty 
                # provides critical contextual clues to the LLM.
                baris_teks.append(f"{col_name}: [NO DATA PPROVIDED]")
        llm_texts.append("\n".join(baris_teks))
    return llm_texts

if __name__ == "__main__":
    # Standardize the path to locate the 'inputs' directory regardless of 
    # the terminal's Current Working Directory (CWD).
    DIR_INI = os.path.dirname(os.path.abspath(__file__))
    INPUT_DIR = os.path.abspath(os.path.join(DIR_INI, "..", "..", "inputs"))
    
    if os.path.exists(INPUT_DIR):
        excel_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.xlsx')]
        for fname in sorted(excel_files):
            print(f"\n--- Extracting: {fname} ---")
            full_path = os.path.join(INPUT_DIR, fname)
            
            # Execute the smart extraction
            df = extract_excel_data(full_path)
            if df is not None:
                print(f"Shape: {df.shape}")
                
                # Test the LLM text conversion module
                formatted_texts = format_df_to_llm_text(df)
                if len(formatted_texts) > 0:
                    print("Sample of First Item sent to LLM Context:")
                    print(formatted_texts[0])
    else:
        print(f"Error: Input directory not found at {INPUT_DIR}")
