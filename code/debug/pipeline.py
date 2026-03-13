import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import concurrent.futures

# Load environment variables
load_dotenv('../../.env')
api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise ValueError("API Key not found. Please set DEEPSEEK_API_KEY in .env")

# Initialize OpenAI client with DeepSeek base URL
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

from few_shot_builder import get_few_shots_for_column

def call_llm(system_prompt, user_content):
    """Helper function to call the LLM API."""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=2048
        )
        ans = response.choices[0].message.content
        if ans is None:
            return ""
        return ans.strip()
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return ""

def _build_system_prompt(column_name, task_instruction):
    """
    Constructs a highly cacheable system prompt.
    The massive FEW_SHOT_JSON is placed at the very beginning to ensure 
    DeepSeek's Prefix Cache hits consistently across all rows processed by this pipeline.
    """
    few_shots_json = get_few_shots_for_column(column_name)
    
    prompt = f"""=== FEW-SHOT EXAMPLES FOR '{column_name}' ===
The following is a JSON array of examples mapping raw risk inputs to the exact desired output for the '{column_name}' column.
{few_shots_json}
=== END FEW-SHOT EXAMPLES ===

You are an expert risk manager and data extractor.
INSTRUCTION:
{task_instruction}
"""
    return prompt

def pipeline_risk_id(text):
    instruction = "Extract or formulate a short unique 'Risk ID' (e.g., R1, ICT-001, 3). If one exists in the text, extract it verbatim. If none exists, formulate a generic short ID. Output ONLY the ID string and nothing else."
    prompt = _build_system_prompt("Risk ID", instruction)
    return call_llm(prompt, text)

def pipeline_risk_description(text):
    instruction = "Concisely summarize the 'Risk Description' in 1-2 sentences. Describe the risk itself. If a description is explicitly present in the input, extract and clean it. Output ONLY the description string."
    prompt = _build_system_prompt("Risk Description", instruction)
    return call_llm(prompt, text)

def pipeline_project_stage(text):
    instruction = "Infer the 'Project Stage' from the risk text. Examples: Pre-construction, Construction, Planning, Operational, IT Setup, Unknown. Output ONLY the short stage name without punctuation."
    prompt = _build_system_prompt("Project Stage", instruction)
    return call_llm(prompt, text)

def pipeline_project_category(text):
    instruction = "Infer the 'Project Category' or 'Risk Category' from the risk text. Examples: Planning, Legislation, Financial, Technical, Operational. Output ONLY the short category name."
    prompt = _build_system_prompt("Project Category", instruction)
    return call_llm(prompt, text)

def pipeline_risk_owner(text):
    instruction = "Extract or infer the 'Risk Owner' responsible for this risk (e.g., Project Manager, IT Manager). If none is mentioned, output 'Unknown'. Output ONLY the role name."
    prompt = _build_system_prompt("Risk Owner", instruction)
    return call_llm(prompt, text)

def pipeline_mitigating_action(text):
    instruction = "Concisely summarize the 'Mitigating Action' or 'Action Plan' in 1-2 sentences. If explicitly stated, extract and clean it. Output ONLY the action string."
    prompt = _build_system_prompt("Mitigating Action", instruction)
    return call_llm(prompt, text)

def pipeline_likelihood(text):
    instruction = "Infer the 'Likelihood' of the risk occurring on a scale from 1 (lowest) to 10 (highest). Output ONLY the whole number between 1 and 10 without decimals."
    prompt = _build_system_prompt("Likelihood (1-10)", instruction)
    return call_llm(prompt, text)

def pipeline_impact(text):
    instruction = "Infer the 'Impact' (or Severity) of the risk on a scale from 1 (lowest) to 10 (highest). Output ONLY the whole number between 1 and 10 without decimals."
    prompt = _build_system_prompt("Impact (1-10)", instruction)
    return call_llm(prompt, text)

def pipeline_risk_priority(text):
    instruction = "Categorize the 'Risk Priority' strictly as one of three values: 'Low', 'Med', or 'High'. Output ONLY the single word (Low, Med, High) and nothing else."
    prompt = _build_system_prompt("Risk Priority (low, med, high)", instruction)
    return call_llm(prompt, text)

def process_single_risk(target_text, project_name=""):
    """
    Runs all 9 pipelines in parallel for a single pre-formatted text string.
    
    Args:
        target_text: Formatted LLM-friendly string of the risk extracted via extract_excel.py
        project_name: Overall project context.
    """
    # Wrap user payload to clearly distinguish it from system instructions
    user_payload = ""
    if project_name:
        user_payload += f"--- PROJECT CONTEXT ---\nProject Name: {project_name}\n\n"
        
    user_payload += f"--- TARGET RISK ---\n{target_text}\n"
    
    pipelines = {
        "Risk ID": pipeline_risk_id,
        "Risk Description": pipeline_risk_description,
        "Project Stage": pipeline_project_stage,
        "Project Category": pipeline_project_category,
        "Risk Owner": pipeline_risk_owner,
        "Mitigating Action": pipeline_mitigating_action,
        "Likelihood (1-10)": pipeline_likelihood,
        "Impact (1-10)": pipeline_impact,
        "Risk Priority (low, med, high)": pipeline_risk_priority
    }
    
    results = {}
    # Use ThreadPoolExecutor to run all 9 API calls in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=9) as executor:
        future_to_col = {executor.submit(func, user_payload): col_name for col_name, func in pipelines.items()}
        for future in concurrent.futures.as_completed(future_to_col):
            col_name = future_to_col[future]
            try:
                result = future.result()
                results[col_name] = result
            except Exception as exc:
                print(f"Pipeline {col_name} generated an exception: {exc}")
                results[col_name] = ""
                
    return results

def process_risks_batch(target_texts, project_name=""):
    """
    Processes a batch of texts for non-mitigating functions (grouped),
    and per-text for mitigating functions.
    Returns a list of dicts.
    """
    batch_size = len(target_texts)
    user_payload_batch = ""
    if project_name:
        user_payload_batch += f"--- PROJECT CONTEXT ---\nProject Name: {project_name}\n\n"
        
    for i, text in enumerate(target_texts):
        user_payload_batch += f"--- RISK {i+1} ---\n{text}\n\n"

    # We will accumulate results across the batch
    final_results = [{} for _ in range(batch_size)]

    # We reuse the functions from process_single_risk by extracting their instructions
    # and appending the batch requirement.
    def batch_worker(col_name, original_func):
        # We need to extract the instruction from the original prompt builder
        # But a simpler way is to just call a newly formed prompt.
        batch_instr = f"\n\nIMPORTANT LIMITATION: You will receive {batch_size} risk items in the input. You MUST map your answer to each risk. Reply EXACTLY with a JSON array containing {batch_size} string/number elements in the same sequence. Output ONLY a valid JSON array (`[\"Ans 1\", \"Ans 2\", ...]`) with no formatting blocks."
        
        # We can extract the original instruction by mapping col names to instructions
        instructions = {
            "Risk ID": "Extract or formulate a short unique 'Risk ID' (e.g., R1, ICT-001, 3). If one exists in the text, extract it verbatim.",
            "Risk Description": "Concisely summarize the 'Risk Description' in 1-2 sentences. Describe the risk itself. If a description is explicitly present in the input, extract and clean it.",
            "Project Stage": "Infer the 'Project Stage' from the risk text. Examples: Pre-construction, Construction, Planning, Operational, IT Setup, Unknown. Output ONLY the short stage name without punctuation.",
            "Project Category": "Infer the 'Project Category' or 'Risk Category' from the risk text. Examples: Planning, Legislation, Financial, Technical, Operational. Output ONLY the short category name.",
            "Risk Owner": "Extract or infer the 'Risk Owner' responsible for this risk (e.g., Project Manager, IT Manager). If none is mentioned, output 'Unknown'.",
            "Likelihood (1-10)": "Infer the 'Likelihood' of the risk occurring on a scale from 1 (lowest) to 10 (highest). Output ONLY the whole number between 1 and 10 without decimals.",
            "Impact (1-10)": "Infer the 'Impact' (or Severity) of the risk on a scale from 1 (lowest) to 10 (highest). Output ONLY the whole number between 1 and 10 without decimals.",
            "Risk Priority (low, med, high)": "Categorize the 'Risk Priority' strictly as one of three values: 'Low', 'Med', or 'High'."
        }
        
        prompt = _build_system_prompt(col_name, instructions[col_name] + batch_instr)
        ans = call_llm(prompt, user_payload_batch)
        
        import json
        if "```json" in ans:
            ans = ans.split("```json")[-1].split("```")[0]
        try:
            arr = json.loads(ans.strip())
            if isinstance(arr, list):
                # pad or cut
                while len(arr) < batch_size:
                    arr.append("")
                return col_name, arr[:batch_size]
        except Exception as e:
            print(f"Failed JSON decode for batched {col_name}: {e}")
            
        return col_name, [""] * batch_size

    def single_worker(row_idx, text):
        return row_idx, pipeline_mitigating_action(text)

    # Dispatch tasks
    simple_cols = ["Risk ID", "Risk Description", "Project Stage", "Project Category", 
                   "Risk Owner", "Likelihood (1-10)", "Impact (1-10)", "Risk Priority (low, med, high)"]
                   
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(10, batch_size + len(simple_cols))) as executor:
        batch_futures = [executor.submit(batch_worker, col, None) for col in simple_cols]
        
        # Mitigating Action is processed row by row purely 
        mitig_payloads = []
        for text in target_texts:
            payload = ""
            if project_name:
                payload += f"--- PROJECT CONTEXT ---\nProject Name: {project_name}\n\n"
            payload += f"--- TARGET RISK ---\n{text}\n"
            mitig_payloads.append(payload)
            
        mitig_futures = [executor.submit(single_worker, i, p) for i, p in enumerate(mitig_payloads)]
        
        for future in concurrent.futures.as_completed(batch_futures):
            col_name, arr = future.result()
            for i in range(batch_size):
                final_results[i][col_name] = arr[i]
                
        for future in concurrent.futures.as_completed(mitig_futures):
            idx, ans = future.result()
            final_results[idx]["Mitigating Action"] = ans
            
    return final_results

if __name__ == "__main__":
    mock_risk_text = (
        "Col 1: R45\n"
        "Details: Delay in getting the required steel shipment because of bad weather.\n"
        "Response: The Project Manager needs to contact alternative suppliers.\n"
        "Freq: 8\n"
        "Severity: 9.5"
    )
    print("Testing context-aware pipeline...")
    res = process_single_risk(mock_risk_text, project_name="Construction of Moorgate")
    for k, v in res.items():
        print(f"{k}: {v}")

