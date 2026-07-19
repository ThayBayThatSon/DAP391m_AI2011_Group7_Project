import os
import subprocess

def run_script(script_name):
    print(f"===========================================================")
    print(f"Running {script_name}...")
    print(f"===========================================================")
    result = subprocess.run(["python", script_name], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error in {script_name}:\n{result.stderr}")
        exit(1)
    print(f"Finished {script_name}")
    print(result.stdout)

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    scripts = [
        "train_combined_panel_models.py",
        "run_lightgbm_ablation.py",
        "leakage_audit.py",
        "generate_paper_figures.py",
    ]
    for script in scripts:
        run_script(script)
    print("All scripts executed successfully.")
