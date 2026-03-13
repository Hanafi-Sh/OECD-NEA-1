"""
pipeline.py
-----------
Core LLM Pipeline for extracting and categorizing risk register attributes.
This final version uses independent parallel calls (1 row per API call) 
with DeepSeek Chat V3 to ensure maximum accuracy and stability.
"""

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
    Runs all 9 analysis pipelines in parallel for a single pre-formatted text string.
    
    This function utilizes `concurrent.futures.ThreadPoolExecutor` to execute
    9 separate API calls simultaneously, significantly reducing processing time.
    
    Args:
        target_text (str): Formatted LLM-friendly string of a single risk item.
        project_name (str): Overall project context to aid LLM reasoning.
        
    Returns:
        dict: A mapping of the 9 target columns to their extracted/predicted values.
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



