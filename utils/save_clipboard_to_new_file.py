import pyperclip
import re
from datetime import datetime
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import sys 

# Retrieve text from clipboard
text = pyperclip.paste()

# Identify if it's Python code
is_python = re.search(r'self\.', text) or re.search(r'try:\s*.*\n.*except', text, re.DOTALL)

# Identify if it's a large document
is_large_doc = len(text) > 1000

# Identify if it's short text
is_short_text = len(text) < 1000

# Determine the type, prioritizing Python code, then large doc, then short text
if is_python:
    file_type = 'py'
elif is_large_doc:
    file_type = 'md'
elif is_short_text:
    file_type = 'txt'
else:
    file_type = 'txt'  # Default

# If markdown, add headers for standalone single lines
if file_type == 'md':
    lines = text.split('\n')
    for i in range(len(lines)):
        line = lines[i].strip()
        if not line or re.match(r'^\s*#', line):
            continue
        if not re.match(r'^\s*\w+', line):
            lines[i] = '#' + lines[i]
    text = '\n'.join(lines)

# Extract the first non-import line for the filename
lines = text.split('\n')
for line in lines:
    if not re.match(r'^\s*(from\s+.*\s+import|import)\b', line):
        first_line = line.strip()
        break
else:
    first_line = "untitled"

# Create timestamp
timestamp = datetime.now().strftime("%m_%d_%y")

# Generate filename
filename = f"{first_line}_{timestamp}.{file_type}"
# Sanitize filename
filename = re.sub(r'[<>:"/\\|?*\n]+', '_', filename)

# Prompt user for directory selection
root = tk.Tk()
root.withdraw()  # Hides the root window
dir_path = filedialog.askdirectory()
if not dir_path:
    dir_path = '.'  # Use current directory if user cancels

# Construct the full file path
full_path = os.path.join(dir_path, filename)
print(text)
print("Save to file: " + full_path)
# Save the file and handle exceptions
try:
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(text)
    # Show confirmation popup
    messagebox.showinfo("Success", f"File saved to:\n{full_path}")
except Exception as e:
    messagebox.showerror("Error", f"Failed to save file:\n{e}")
    

def close_terminal(): 
    if sys.platform == 'win32': os.system('taskkill /F /IM cmd.exe')

close_terminal()
