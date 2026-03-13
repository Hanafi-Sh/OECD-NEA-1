import os
import pandas as pd

INPUT_DIR = '../../inputs'
OUTPUT_DIR = '../../outputs'

def explore_excel(filename):
    print(f"\n--- Exploring Excel: {filename} ---")
    filepath = os.path.join(INPUT_DIR, filename)
    try:
        # Load without defining a header first to see raw structure
        df = pd.read_excel(filepath, header=None)
        print(f"Shape: {df.shape}")
        print("First 10 rows:")
        print(df.head(10).to_string())
    except Exception as e:
        print(f"Error loading {filename}: {e}")

def explore_pdf(filename):
    print(f"\n--- Exploring PDF: {filename} ---")
    filepath = os.path.join(INPUT_DIR, filename)
    try:
        import PyPDF2
        with open(filepath, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            print(f"Number of pages: {len(reader.pages)}")
            print("First 1000 characters of Page 1:")
            page_text = reader.pages[0].extract_text()
            print(page_text[:1000] if page_text else "No text extracted.")
    except ImportError:
        print("PyPDF2 not installed. Cannot read PDF.")
    except Exception as e:
        print(f"Error loading {filename}: {e}")

if __name__ == "__main__":
    # Explore Input Excel files
    excel_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.xlsx')]
    for f in sorted(excel_files):
        explore_excel(f)
        
    # Explore Input PDF files
    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.pdf')]
    for f in sorted(pdf_files):
        explore_pdf(f)
