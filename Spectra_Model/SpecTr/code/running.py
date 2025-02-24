import sys
import subprocess

python_exe = r'c:\Users\maria\OneDrive\Documents\New_Masters_Plan\New_Models\Spectr\.venv\Scripts\python.exe'
script_path = r'C:\Users\maria\OneDrive\Documents\New_Masters_Plan\New_Models\Spectr\SpecTr\code\train_main.py'
args = [
    '-r', '/kaggle/input/odsi-db/dataset',
    '-dd', r'C:\Users\maria\OneDrive\Documents\New_Masters_Plan\New_Models\Spectr\SpecTr\four_folds_dental.json',
    '-sn', '51',
    '-cut', '50',
    '-e', '20',
    '-b', '8',
    '-c', '36',
    '--dataset', 'dental'
]


# Construct the command as a list of strings
command = [python_exe, script_path] + args

# Join the command parts into a single command string
command_str = ' '.join(command)

# Execute the command using subprocess
result = subprocess.run(command_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Print the result
print(result.stdout.decode('utf-8'))
print(result.stderr.decode('utf-8'))
