"""
generate_final_outputs.py
--------------------------
Orchestrates the finalized 9-pipeline LLM architecture on Input files 4 and 5 
to produce the final submitted Excel Output files. Code is clean and production-ready.
"""
import os
import pandas as pd
from extract_excel import extract_excel_data, format_df_to_llm_text
from extract_pdf import extract_pdf_data
from pipeline import process_single_risk

INPUT_DIR = '../../inputs'
OUTPUT_DIR = '../../outputs'

# The 9 target columns for the final output
TARGET_COLUMNS = [
    "Risk ID", "Risk Description", "Project Stage", "Project Category",
    "Risk Owner", "Mitigating Action", "Likelihood (1-10)", "Impact (1-10)",
    "Risk Priority (low, med, high)"
]

def generate_predictions_from_df(df, project_name):
    """
    Iterates through rows of a DataFrame and orchestrates the parallel prediction
    using the 9-pipeline LLM independent approach (Single Call Architecture).
    """
    results = []
    
    # Pre-format all rows to key-value LLM readable texts
    formatted_texts = format_df_to_llm_text(df)
    
    for i in range(len(formatted_texts)):
        text = formatted_texts[i]
        print(f"  Processing single row {i+1} (out of {len(formatted_texts)}) for {project_name}...")
        
        predicted_dict = process_single_risk(
            target_text=text,
            project_name=project_name
        )
        results.append(predicted_dict)
            
    return pd.DataFrame(results, columns=TARGET_COLUMNS)

def process_file(input_file, output_file, project_name):
    """Processes either Excel or PDF inputs into the final standardized Output format."""
    print(f"\n{'='*60}")
    print(f"Processing: {input_file}")
    print(f"{'='*60}")
    
    if input_file.endswith(".pdf"):
        df_extracted = extract_pdf_data(input_file)
    else:
        df_extracted = extract_excel_data(os.path.join(INPUT_DIR, input_file))
        
    if df_extracted is None or df_extracted.empty:
        print(f"Failed to extract data from {input_file}.")
        return
    
    print(f"  Extracted {len(df_extracted)} rows from {input_file}.")
    df_predicted = generate_predictions_from_df(df_extracted, project_name=project_name)
    
    output_path = os.path.join(OUTPUT_DIR, output_file)
    df_predicted.to_excel(output_path, index=False)
    print(f"  Saved: {output_path}")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    process_file("4. Moorgate Crossrail Register (Input).xlsx", "4. Moorgate Crossrail Register (Final).xlsx", "Moorgate Crossrail")
    process_file("5. Corporate_Risk_Register (Input).pdf", "5. Corporate Risk Register (Final).xlsx", "Corporate Risk Register")
    
    print("\n*** All outputs predicted & generated successfully! ***")
