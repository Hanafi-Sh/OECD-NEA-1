import pandas as pd
import numpy as np

def standardize_columns(df: pd.DataFrame, source_type: str) -> pd.DataFrame:
    """
    Standardizes the column names to match the final output requirements.
    The required columns are: Risk ID, Risk Description, Project Stage, 
    Project Category, Risk Owner, Mitigating Action, Likelihood (1-10), 
    Impact (1-10), Risk Priority (low, med, high).
    """
    if source_type == "input_4":
        # Rename 'Risk Category' to 'Project Category' to match the ground truth
        rename_map = {
            'Risk Category': 'Project Category'
        }
        df = df.rename(columns=rename_map)
        
    elif source_type == "input_5":
        # We will define this later when we extract the PDF
        pass
        
    return df

def convert_scales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts qualitative text scales (e.g., 'Possible', 'Serious') 
    into quantitative scales (1-10) as required by the output format.
    """
    # Define mapping for Likelihood based on qualitative terms found in Input 4
    likelihood_map = {
        'Rare': 2,
        'Unlikely': 4,
        'Possible': 6,
        'Likely': 8,
        'Almost Certain': 10
    }
    
    # Define mapping for Impact based on qualitative terms found in Input 4
    impact_map = {
        'Minor': 3,
        'Serious': 7,
        'Major': 9
    }
    
    # Apply mapping only to non-null values, converting them to numeric (float/int)
    if 'Likelihood (1-10)' in df.columns:
        # Strip whitespaces and title-case the text to ensure exact matching
        df['Likelihood (1-10)'] = df['Likelihood (1-10)'].astype(str).str.strip().str.title()
        df['Likelihood (1-10)'] = df['Likelihood (1-10)'].map(likelihood_map).fillna(df['Likelihood (1-10)'])
        
    if 'Impact (1-10)' in df.columns:
        df['Impact (1-10)'] = df['Impact (1-10)'].astype(str).str.strip().str.title()
        df['Impact (1-10)'] = df['Impact (1-10)'].map(impact_map).fillna(df['Impact (1-10)'])
        
    return df

def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops columns that are completely empty or not required by the final output,
    such as the empty 'Date Added' column in Input 4.
    """
    # Drop 'Date Added' if it exists
    if 'Date Added' in df.columns:
        df = df.drop(columns=['Date Added'])
        
    return df