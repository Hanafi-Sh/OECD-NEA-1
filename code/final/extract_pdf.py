import os
import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
try:
    import fitz # PyMuPDF
except ImportError:
    fitz = None

INPUT_DIR = '../../inputs'

load_dotenv('../../.env')
api_key = os.getenv("DEEPSEEK_API_KEY")

if api_key:
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
else:
    client = None

def raw_pdf_to_json(page_text):
    """
    Sends raw vertically-extracted PDF text to the LLM (DeepSeek) to be parsed 
    into a structured JSON Array representing tabular rows.
    """
    if not client:
        print("API Key not found. Cannot parse PDF to DataFrame.")
        return []

    system_prompt = """You are a highly precise data parsing assistant.
The following is vertically extracted text from a Corporate Risk Register PDF.
Groups of values reading down the page represent columns.
Look closely at how Reference, Risk and effects, Mitigation, etc., map to the risk.
Output exactly and ONLY a valid JSON array of objects representing these risks.
Keep all the original text content intact for each field.
Do not wrap your response in markdown fences (like ```json), just output the raw JSON array.
Keys to extract for each object:
- Reference
- Risk and effects
- Mitigation
- Risk Owner
- Actions being taken to managing risk
- Comments and progress of actions
"""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"--- PAGE TEXT ---\n{page_text}"}
            ],
            temperature=0.0
        )
        res = response.choices[0].message.content.strip()
        
        # Clean up markdown formatting if the LLM output includes code fences
        if res.startswith("```json"):
            res = res[7:]
        if res.startswith("```"):
            res = res[3:]
        if res.endswith("```"):
            res = res[:-3]
            
        return json.loads(res.strip())
    except Exception as e:
        print(f"Error calling LLM or parsing JSON for PDF page: {e}")
        return []

def extract_pdf_data(filename):
    """
    Extracts underlying structural data from a PDF file using PyMuPDF and DeepSeek,
    returning a clean Pandas DataFrame representing the contained risks.
    """
    if fitz is None:
        raise ImportError("PyMuPDF (fitz) is not installed. Please install it with 'pip install PyMuPDF'")
        
    filepath = os.path.join(INPUT_DIR, filename)
    doc = fitz.open(filepath)
    
    all_risks = []
    
    print(f"Parsing PDF {filename} to DataFrame using LLM...")
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if not text or len(text.strip()) < 100:
            continue
            
        # Invoke the LLM to format this raw page text into a structured JSON array
        print(f"  Parsing Page {page_num + 1}/{len(doc)}...")
        page_risks = raw_pdf_to_json(text)
        
        if page_risks:
            all_risks.extend(page_risks)
            
        if os.environ.get("PDF_SINGLE_PAGE") == "1":
            break
            
    if not all_risks:
        return pd.DataFrame()
        
    df = pd.DataFrame(all_risks)
    
    return df

if __name__ == "__main__":
    df = extract_pdf_data("5. Corporate_Risk_Register (Input).pdf")
    print(f"Extracted DataFrame with shape: {df.shape}")
    if not df.empty:
        print(df.head())
