from pipeline import call_llm, _build_system_prompt, pipeline_mitigating_action
import concurrent.futures
import json

PIPELINE_INSTRUCTIONS = {
    "Risk ID": "Extract or formulate a short unique 'Risk ID' (e.g., R1, ICT-001, 3). If one exists in the text, extract it verbatim.",
    "Risk Description": "Concisely summarize the 'Risk Description' in 1-2 sentences. Describe the risk itself.",
    "Project Stage": "Infer the 'Project Stage' from the risk text. Examples: Pre-construction, Construction, Planning, Operational, IT Setup, Unknown.",
    "Project Category": "Infer the 'Project Category' or 'Risk Category' from the risk text. Examples: Planning, Legislation, Financial, Technical, Operational.",
    "Risk Owner": "Extract or infer the 'Risk Owner' responsible for this risk (e.g., Project Manager, IT Manager). If none is mentioned, output 'Unknown'.",
    "Likelihood (1-10)": "Infer the 'Likelihood' of the risk occurring on a scale from 1 (lowest) to 10 (highest). Output the whole number 1-10.",
    "Impact (1-10)": "Infer the 'Impact' (or Severity) of the risk on a scale from 1 (lowest) to 10 (highest). Output the whole number 1-10.",
    "Risk Priority (low, med, high)": "Categorize the 'Risk Priority' strictly as one of three values: 'Low', 'Med', or 'High'."
}

def process_batch_risks(target_texts, project_name=""):
    """
    Processes a batch of texts for non-mitigating functions (grouped),
    and per-text for mitigating functions.
    Returns a list of dicts.
    """
    batch_size = len(target_texts)
    user_payload = ""
    if project_name:
        user_payload += f"--- PROJECT CONTEXT ---\nProject Name: {project_name}\n\n"
        
    for i, text in enumerate(target_texts):
        user_payload += f"--- RISK {i+1} ---\n{text}\n\n"

    # We will accumulate results across the batch
    # final_results[i] will be the dict for risk i
    final_results = [{} for _ in range(batch_size)]

    def batch_worker(col_name, instr):
        batch_instr = instr + f"\n\nIMPORTANT: You are receiving a batch of {batch_size} risk items. You MUST map your answer to each risk. Reply EXACTLY with a JSON array containing {batch_size} string/number elements in the same sequence. Output ONLY valid JSON array with no formatting blocks."
        prompt = _build_system_prompt(col_name, batch_instr)
        ans = call_llm(prompt, user_payload)
        
        # Parse JSON
        if "```json" in ans:
            ans = ans.split("```json")[-1].split("```")[0]
        try:
            arr = json.loads(ans.strip())
            if isinstance(arr, list):
                # padd if short
                while len(arr) < batch_size:
                    arr.append("")
                return col_name, arr[:batch_size]
        except Exception as e:
            print(f"Failed JSON decode for {col_name}: {ans}")
            
        return col_name, [""] * batch_size

    def single_worker(row_idx, text):
        res = pipeline_mitigating_action(text)
        return row_idx, res

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(10, batch_size + len(PIPELINE_INSTRUCTIONS))) as executor:
        # Submit batched jobs
        batch_futures = [executor.submit(batch_worker, col, instr) for col, instr in PIPELINE_INSTRUCTIONS.items()]
        # Submit single jobs for Mitigation
        mitig_futures = [executor.submit(single_worker, i, t) for i, t in enumerate(target_texts)]
        
        for future in concurrent.futures.as_completed(batch_futures):
            col_name, arr = future.result()
            for i in range(batch_size):
                final_results[i][col_name] = arr[i]
                
        for future in concurrent.futures.as_completed(mitig_futures):
            idx, ans = future.result()
            final_results[idx]["Mitigating Action"] = ans
            
    return final_results

