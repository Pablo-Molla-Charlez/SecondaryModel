#!/usr/bin/env python3
import os
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def convert_meta_label(df, target_mode):
    """
    Applies the conversion logic based on 'pred' and 'lab' columns.
    
    Logic for 'tp' (True Positives):
    - pred=1, lab=1 -> meta_label=1
    - pred=1, lab=0 -> meta_label=0
    - pred=0 -> meta_label=False (object type)
    
    Logic for 'fp' (False Positives):
    - pred=1, lab=0 -> meta_label=1
    - pred=1, lab=1 -> meta_label=0
    - pred=0 -> meta_label=False (object type)
    """
    if 'pred' not in df.columns or 'lab' not in df.columns:
        raise ValueError("Missing 'pred' or 'lab' columns in CSV.")
        
    def _logic(row):
        pred = row['pred']
        lab = row['lab']
        
        # ┏━━━━━━━━━━ Apply conversion logic ━━━━━━━━━━┓
        if pred == 1:
            if target_mode == 'tp':
                return 1 if lab == 1 else 0
            else: # fp
                return 1 if lab == 0 else 0
        else:
            return False

    df['meta_label'] = df.apply(_logic, axis=1)
    return df

def main():
    # ┏━━━━━━━━━━ Parse arguments ━━━━━━━━━━┓
    parser = argparse.ArgumentParser(description="Convert meta-labels in CSV datasets between TP and FP modes.")
    parser.add_argument("--source_dir", type=str, help="Path to the source folder containing CSVs (e.g., .../30m_fp)")
    parser.add_argument("--mode", type=str, choices=['tp', 'fp'], help="Target mode (tp or fp). If not provided, inferred from source_dir.")
    
    args = parser.parse_args()
    
    # ┏━━━━━━━━━━ Validate source directory ━━━━━━━━━━┓
    src_path = Path(args.source_dir).resolve()
    if not src_path.exists() or not src_path.is_dir():
        print(f"Error: Source directory {src_path} does not exist.")
        return

    # ┏━━━━━━━━━━ Infer target mode if not provided ━━━━━━━━━━┓
    if args.mode:
        target_mode = args.mode
    else:
        if src_path.name.endswith('_fp'):
            target_mode = 'tp'
        elif src_path.name.endswith('_tp'):
            target_mode = 'fp'
        else:
            print("Error: Could not infer target mode from folder name. Please use --mode.")
            return

    # ┏━━━━━━━━━━ Determine target directory ━━━━━━━━━━┓
    base_dir = src_path.parent
    src_suffix = '_fp' if target_mode == 'tp' else '_tp'
    target_suffix = '_tp' if target_mode == 'tp' else '_fp'
    target_name = src_path.name.replace(src_suffix, target_suffix)
    if target_name == src_path.name: # If no suffix match, just append it
        target_name = f"{src_path.name}_{target_mode}"
    target_path = base_dir / target_name
    target_path.mkdir(parents=True, exist_ok=True)
    print(f"Converting: {src_path.name} -> {target_path.name} (Target Mode: {target_mode.upper()})")
    
    # ┏━━━━━━━━━━ Get CSV files ━━━━━━━━━━┓
    csv_files = list(src_path.glob("*.csv"))
    if not csv_files:
        print("No CSV files found in source directory.")
        return

    # ┏━━━━━━━━━━ Process CSV files ━━━━━━━━━━┓
    for csv_file in tqdm(csv_files, desc="Processing CSVs"):
        df = pd.read_csv(csv_file)
        df = convert_meta_label(df, target_mode)
        
        # ┏━━━━━━━━━━ Save to target dir ━━━━━━━━━━┓
        df.to_csv(target_path / csv_file.name, index=False)

    print(f"\nSuccess! Converted {len(csv_files)} files to {target_path}")

if __name__ == "__main__":
    main()
