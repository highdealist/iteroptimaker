import tkinter as tk
from tkinter import messagebox, ttk, simpledialog, Listbox
import os
import subprocess

def run_script():
    # Get the selected script from the listbox
    selection = script_listbox.get(script_listbox.curselection())
    script_name = next(name for name, label in script_options if label == selection)
    
    # Ask for input if needed
    input_var = simpledialog.askstring("Input", "Please enter the argument for the script (if any):")
    
    # Prepare the command to run the script
    command = ["python", os.path.join("c:", "agent_tools", script_name)]
    if input_var:
        command.append(input_var)  # Add the argument if provided
    
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Failed to run script: {e}")

root = tk.Tk()
root.title("Agent Tools")
root.geometry("400x500")

# Create main frame
main_frame = ttk.Frame(root, padding="10")
main_frame.pack(fill=tk.BOTH, expand=True)

# Create and pack listbox
script_listbox = tk.Listbox(main_frame, selectmode=tk.SINGLE)
script_listbox.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

script_options = [
    ("get_yt_comments.py", "Get Youtube Comments"),
    ("code review.py", "Code Review"), 
    ("extract_with_newspaper.py", "Extract with Newspaper"),
    ("fetch_latest_arxiv_papers.py", "Fetch Latest Arxiv Papers"),
    ("foia_search.py", "FOIA Search"),
    ("google_search.py", "Google Search"),
    ("print_directory_structure.py", "Print Directory Structure"),
    ("research_summary.py", "Research Summary"),
    ("scrape_current_window_url.py", "Scrape Current Window URL"),
    ("scrape_url_in_clipboard.py", "Scrape URL in Clipboard"),
    ("search_manager.py", "Search Manager")
]

# Populate listbox
for _, label in script_options:
    script_listbox.insert(tk.END, label)

# Create and pack run button
run_button = ttk.Button(main_frame, text="Run Selected Script", command=run_script)
run_button.pack(fill=tk.X)

def main():
    root.mainloop()

if __name__ == "__main__":
    main()