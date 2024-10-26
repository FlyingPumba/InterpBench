import re
import shutil

import wandb
import sys
import os

def format_metric(value):
    if abs(value - 100) < 1e-6:  # Check if the value is very close to 100
        return "100"
    else:
        return f"{value:.3f}"

def update_table(table, case_metrics):
    for case_id, metrics in case_metrics.items():
        if case_id == "ioi":
            case_string = "IOI"
        elif case_id == "ioi_next_token":
            case_string = "IOI Next token"
        else:
            case_string = case_id
        
        row_pattern = rf'{case_string}\s+&\s+(\w+)\s+&\s+(\w+)\s+&\s+.*?&\s+.*?&\s+.*?&\s+(.*?)\s+&\s+(.*?)\\\\'
        row_match = re.search(row_pattern, table)
        
        if row_match:
            old_row = row_match.group()
            tracr_bench = row_match.group(1)
            type_val = row_match.group(2)
            description = row_match.group(3)
            code_link = row_match.group(4)
            new_row = f"{case_string} & {tracr_bench} & {type_val} & {metrics['acc']} & {metrics['iia']} & {metrics['siia']} & {description} & {code_link}\\\\"
            table = table.replace(old_row, new_row)
    return table

if __name__ == "__main__":

    appendix_file_path = "/home/ivan/latex/mech-interp-benchmark-paper-neurips-2024-pro-overleaf/appendix.tex"
    wandb_project_name = "iit-eval-seed-92-max-len-4K"

    api = wandb.Api()
    runs = api.runs(wandb_project_name)
    
    # Dictionary to store metrics for each case
    case_metrics = {}

    for run in runs:
        if run.state != "finished":
            continue

        # extract case_id from run name
        if not run.name.startswith("case-"):
            raise ValueError(f"Run name {run.name} does not start with 'case-'")
        case_id = run.name.split("-")[1]

        acc = run.summary["val/accuracy"]
        iia = run.summary["val/IIA"]
        siia = run.summary["val/strict_accuracy"]

        case_metrics[case_id] = {
            "acc": format_metric(acc),
            "iia": format_metric(iia),
            "siia": format_metric(siia)
        }

    # Read the appendix file
    with open(appendix_file_path, 'r') as file:
        content = file.read()

    # Find and update the first table (TracrBench?)
    table_pattern1 = r'\\begin{longtable}.*?Case & TracrBench\? & Type & Acc & IIA & SIIA & Description & Code.*?\\end{longtable}'
    table_match1 = re.search(table_pattern1, content, re.DOTALL)

    if table_match1:
        table1 = table_match1.group()
        updated_table1 = update_table(table1, case_metrics)
        content = content.replace(table1, updated_table1)
        print("First table (TracrBench?) updated successfully!")
    else:
        print("First table (TracrBench?) not found in the appendix file.")

    # Find and update the second table (Tracr?)
    table_pattern2 = r'\\begin{longtable}.*?Case & Tracr\? & Type & Acc & IIA & SIIA & Description & Code.*?\\end{longtable}'
    table_match2 = re.search(table_pattern2, content, re.DOTALL)

    if table_match2:
        table2 = table_match2.group()
        updated_table2 = update_table(table2, case_metrics)
        content = content.replace(table2, updated_table2)
        print("Second table (Tracr?) updated successfully!")
    else:
        print("Second table (Tracr?) not found in the appendix file.")

    # Write the updated content back to the file
    with open(appendix_file_path, 'w') as file:
        file.write(content)

    print("All tables updated successfully!")
