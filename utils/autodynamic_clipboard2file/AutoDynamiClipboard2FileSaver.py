import pyperclip
import re
from datetime import datetime
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import logging

# Configure logging
logging.basicConfig(filename='clipboard_saver.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

class ClipboardSaver(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Clipboard Saver")
        self.geometry("600x400")
        self.resizable(True, True)
        
        self.text = pyperclip.paste()
        self.file_type = self.determine_file_type(self.text)
        self.process_md = self.file_type == 'md'
        self.text = self.process_markdown(self.text, process=self.process_md)
        
        self.filename = self.sanitize_filename(self.extract_filename())
        self.timestamp = datetime.now().strftime("%m_%d_%y")
        
        self.create_widgets()
    
    def create_widgets(self):
        # Frame for filename entry
        frame_filename = ttk.Frame(self)
        frame_filename.pack(pady=10)
        
        ttk.Label(frame_filename, text="Filename:").pack(side=tk.LEFT)
        self.entry_filename = ttk.Entry(frame_filename, width=50)
        self.entry_filename.pack(side=tk.LEFT, padx=10)
        self.entry_filename.insert(0, f"{self.filename}_{self.timestamp}.{self.file_type}")
        
        # Frame for directory selection
        frame_dir = ttk.Frame(self)
        frame_dir.pack(pady=10)
        
        ttk.Label(frame_dir, text="Save to Directory:").pack(side=tk.LEFT)
        self.entry_dir = ttk.Entry(frame_dir, width=50)
        self.entry_dir.pack(side=tk.LEFT, padx=10)
        self.entry_dir.insert(0, os.getcwd())
        
        ttk.Button(frame_dir, text="Browse", command=self.select_directory).pack(side=tk.LEFT, padx=10)
        
        # Frame for markdown processing option
        frame_options = ttk.Frame(self)
        frame_options.pack(pady=10)
        
        self.var_process_md = tk.BooleanVar(value=self.process_md)
        ttk.Checkbutton(frame_options, text="Process Markdown", variable=self.var_process_md,
                        command=self.toggle_process_md).pack(side=tk.LEFT)
        
        # Frame for save and cancel buttons
        frame_buttons = ttk.Frame(self)
        frame_buttons.pack(pady=20)
        
        ttk.Button(frame_buttons, text="Save", command=self.save_file).pack(side=tk.LEFT, padx=10)
        ttk.Button(frame_buttons, text="Cancel", command=self.destroy).pack(side=tk.LEFT, padx=10)
        
        # Frame for status message
        self.status_label = ttk.Label(self, text="", foreground="green")
        self.status_label.pack(pady=5)
    
    def select_directory(self):
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.entry_dir.delete(0, tk.END)
            self.entry_dir.insert(0, dir_path)
    
    def toggle_process_md(self):
        self.process_md = self.var_process_md.get()
        self.text = self.process_markdown(self.text, process=self.process_md)
        self.filename = self.sanitize_filename(self.extract_filename())
        self.entry_filename.delete(0, tk.END)
        self.entry_filename.insert(0, f"{self.filename}_{self.timestamp}.{self.file_type}")
    
    def determine_file_type(self, text):
        is_python = re.search(r'^import\s+|^\s*def\s+|^\s*class\s+', text, re.MULTILINE)
        is_markdown = re.search(r'^#+\s+|^\s*-\s+|^\s*\d+\.\s+', text, re.MULTILINE)
        
        if is_python:
            return 'py'
        elif is_markdown:
            return 'md'
        else:
            return 'txt'
    
    def process_markdown(self, text, process=True):
        if not process:
            return text
        lines = text.split('\n')
        for i in range(len(lines)):
            line = lines[i].strip()
            if not line or re.match(r'^\s*#', line):
                continue
            if not re.match(r'^\s*\w+', line):
                lines[i] = '#' + lines[i]
        return '\n'.join(lines)
    
    def extract_filename(self):
        lines = self.text.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not re.match(r'^import\s+|^\s*def\s+|^\s*class\s+', line):
                return line
        return "untitled"
    
    def sanitize_filename(self, filename):
        return re.sub(r'[<>:"/\\|?*\n]+', '_', filename)
    
    def save_file(self):
        filename = self.entry_filename.get()
        dir_path = self.entry_dir.get()
        full_path = os.path.join(dir_path, filename)
        
        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(self.text)
            self.status_label.config(text=f"File saved to:\n{full_path}", foreground="green")
            logging.info(f"File saved to {full_path}")
            messagebox.showinfo("Success", "File saved successfully!")
        except Exception as e:
            self.status_label.config(text=f"Failed to save file:\n{e}", foreground="red")
            logging.error(f"Failed to save file to {full_path}: {e}")
            messagebox.showerror("Error", f"Failed to save file:\n{e}")

if __name__ == "__main__":
    app = ClipboardSaver()
    app.mainloop()