import os
import json
import pandas as pd
from threading import Lock

# Import our robust excel extractor to get the raw text representation
from extract_excel import extract_excel_data, format_df_to_llm_text

FILE_PAIRS = [
    {
        "input": "1. IVC DOE R2 (Input).xlsx",
        "output": "1. IVC DOE (Final).xlsx",
        # Map our 9 target columns to the specific column names in this output file
        "mapping": {
            "Risk ID": "Risk ID",
            "Risk Description": "Risk Description",
            "Project Stage": "Project Stage",
            "Project Category": "Project Category",
            "Risk Owner": "Risk Owner",
            "Mitigating Action": "Mitigating Action",
            "Likelihood (1-10)": "Likelihood (1-10) (pre-mitigation)",
            "Impact (1-10)": "Impact (1-10) (pre-mitigation)",
            "Risk Priority (low, med, high)": "Risk Priority (pre-mitigation)"
        }
    },
    {
        "input": "2. City of York Council (Input).xlsx",
        "output": "2. City of York Council (Final).xlsx",
        "mapping": {
            "Risk ID": "Risk ID",
            "Risk Description": "Risk Description",
            "Project Stage": "Project Stage",
            "Project Category": "Risk Category",
            "Risk Owner": "Risk Owner",
            "Mitigating Action": "Mitigation",
            "Likelihood (1-10)": "Likelihood (1-10)",
            "Impact (1-10)": "Impact (1-10)",
            "Risk Priority (low, med, high)": "Risk Priority (low, med, high)"
        }
    },
    {
        "input": "3. Digital Security IT Sample Register (Input).xlsx",
        "output": "3. Digital Security IT Sample Register (Final).xlsx",
        "mapping": {
            "Risk ID": "Number",
            "Risk Description": "Risk Description",
            "Project Stage": "Project Stage",
            "Project Category": "Project Category",
            "Risk Owner": "Risk Owner",
            "Mitigating Action": "Mitigating Action",
            "Likelihood (1-10)": "Likelihood (1-10)",
            "Impact (1-10)": "Impact (1-10)",
            "Risk Priority (low, med, high)": "Risk Priority (low, med, high)"
        }
    }
]

# Global cache variables
_CACHE_EXAMPLES_BY_COL = None
_CACHE_LOCK = Lock()

def _load_all_data():
    """Reads all 3 input/output pairs and returns a dictionary of examples grouped by our 9 standard columns."""
    global _CACHE_EXAMPLES_BY_COL
    
    with _CACHE_LOCK:
        if _CACHE_EXAMPLES_BY_COL is not None:
            return _CACHE_EXAMPLES_BY_COL
            
        current_dir = os.path.dirname(os.path.abspath(__file__))
        input_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "inputs"))
        output_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "outputs"))
        
        # Initialize dictionary to hold list of {input_text, expected_output} for each standard column
        all_columns = FILE_PAIRS[0]["mapping"].keys()
        _CACHE_EXAMPLES_BY_COL = {col: [] for col in all_columns}
        
        for pair in FILE_PAIRS:
            in_path = os.path.join(input_dir, pair['input'])
            out_path = os.path.join(output_dir, pair['output'])
            
            # Extract inputs using our smart logic
            df_in = extract_excel_data(in_path)
            if df_in is None:
                print(f"Failed to process input {in_path}")
                continue
                
            input_texts = format_df_to_llm_text(df_in)
            
            # Read expected outputs
            try:
                df_out = pd.read_excel(out_path)
            except Exception as e:
                print(f"Failed to read output file {out_path}: {e}")
                continue
                
            min_len = min(len(input_texts), len(df_out))
            if len(input_texts) != len(df_out):
                print(f"Warning: Discrepancy for {pair['input']}. Input lines: {len(input_texts)}, Output lines: {len(df_out)}. Using lowest common denominator.")
                
            for i in range(min_len):
                in_text = input_texts[i]
                
                # Retrieve expected values for each of our 9 columns
                for standard_col, actual_col in pair["mapping"].items():
                    try:
                        out_val = df_out.iloc[i][actual_col]
                        if pd.isna(out_val):
                            out_val = ""
                        else:
                            out_val = str(out_val).strip()
                            
                            # Remove trailing '.0' for numeric types interpreted as floats by pandas
                            if standard_col in ["Likelihood (1-10)", "Impact (1-10)"] and out_val.endswith('.0'):
                                out_val = out_val[:-2]
                    except KeyError:
                        out_val = ""
                        
                    _CACHE_EXAMPLES_BY_COL[standard_col].append({
                        "input_text": in_text,
                        "expected_output": out_val
                    })
                    
        return _CACHE_EXAMPLES_BY_COL


def get_few_shots_for_column(col_name):
    """
    Returns a unified JSON string containing all examples (Input 1, 2, 3) tailored
    to extract `col_name`. Utilizing JSON ensures safety from special character breaks.
    """
    examples_by_col = _load_all_data()
    
    if col_name not in examples_by_col:
        return "[]" # Return empty JSON array if column isn't mapped
        
    examples_list = examples_by_col[col_name]
    
    # We dump to JSON string. Ensure_ascii preserves readable characters.
    # Indent=None saves physical tokens so the prompt is dense.
    return json.dumps(examples_list, ensure_ascii=False)


if __name__ == "__main__":
    # Test
    print("Pre-loading data...")
    shots = get_few_shots_for_column("Risk Priority (low, med, high)")
    print(f"Type: {type(shots)}")
    print(f"Total length of JSON string: {len(shots)} characters.")
    
    # Let's peek at the first 2 items
    import json
    parsed = json.loads(shots)
    print(f"Total examples loaded: {len(parsed)}")
    print(f"Sample Example 1 Output: {parsed[0]['expected_output']}")
    print(f"Sample Length of Example 1 Input: {len(parsed[0]['input_text'])}")
