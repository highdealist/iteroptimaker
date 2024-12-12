"""
Notebook File Format Converter Tool.
Converts between Jupyter notebooks and individual source files.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import nbformat
from nbformat.v4 import new_notebook, new_code_cell
import json
from typing import List, Optional, Dict, Any
from pathlib import Path
from fnmatch import fnmatch
import datetime
import logging

logger = logging.getLogger(__name__)

import time
class NotebookFileConverter:
    """GUI tool for converting between Jupyter notebooks and source files."""

    # Class-level constants for configuration
    WINDOW_SIZE = "1000x800"
    PADDING = "15"
    DEFAULT_THEME = "clam"

    # Color schemes
    DARK_COLORS = {
        'bg': '#2d2d2d',
        'fg': '#ffffff',
        'button': '#404040',
        'entry': '#404040',
        'select': '#505050'
    }

    LIGHT_COLORS = {
        'bg': '#ffffff',
        'fg': '#000000',
        'button': '#e0e0e0',
        'entry': '#ffffff',
        'select': '#f0f0f0'
    }

    def __init__(self):
        """Initialize the converter application."""
        self.setup_logging()
        self.init_window()
        self.init_styles()
        self.create_notebook()
        self.create_tabs()
        self.create_color_toggle()

    def setup_logging(self):
        """Configure logging for the application."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('notebook_converter.log')
            ]
        )

    def init_window(self):
        """Initialize the main window."""
        self.window = tk.Tk()
        self.window.title("Notebook File Converter")
        self.window.geometry(self.WINDOW_SIZE)
        self.window.resizable(True, True)

        # Configure main window grid weights
        self.window.grid_columnconfigure(0, weight=1)
        self.window.grid_rowconfigure(0, weight=1)

    def init_styles(self):
        """Initialize ttk styles and theme."""
        self.style = ttk.Style()
        self.style.theme_use(self.DEFAULT_THEME)
        self.color_mode = 'dark'
        self.random_colors = False
        self.configure_dark_mode()
        self.cursor_color_index = 0
        self.cursor_colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF']
        
        # Start cursor color animation
        self.animate_cursor()

    def create_notebook(self):
        """Create the main notebook widget."""
        self.notebook = ttk.Notebook(self.window)
        self.notebook.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        # Create main container
        main_container = ttk.Frame(self.window, padding="10")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure window grid
        self.window.grid_columnconfigure(0, weight=1)
        self.window.grid_rowconfigure(0, weight=1)
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_container)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    def create_tabs(self):
        """Create and configure the main application tabs."""
        # Create main frames
        self.converter_frame = ttk.Frame(self.notebook, padding=self.PADDING)
        self.directory_frame = ttk.Frame(self.notebook, padding=self.PADDING)

        # Configure frame grid weights
        self.converter_frame.grid_columnconfigure(0, weight=1)
        self.converter_frame.grid_rowconfigure(4, weight=1)

        # Setup tabs
        self.setup_converter_tab()
        # Create frames for each tab
        self.directory_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.directory_frame, text="Directory")
        
        # Initialize UI components
        self.setup_directory_tab()
        
        # Add frames to notebook
        self.notebook.add(self.converter_frame, text="File Converter")
        self.notebook.add(self.directory_frame, text="Directory Structure")

    def create_color_toggle(self):
        """Create the color mode toggle button."""
        self.toggle_frame = ttk.Frame(self.window)
        self.toggle_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

        self.toggle_btn = ttk.Button(
            self.toggle_frame,
            text="Toggle Color Mode",
            command=self.toggle_color_mode
        )
        self.toggle_btn.pack(side=tk.RIGHT)

    def setup_converter_tab(self):
        """Set up the file converter tab."""
        # Mode Selection
        mode_frame = ttk.LabelFrame(
            self.converter_frame,
            text="Conversion Mode",
            padding="10",
            style='Section.TLabelframe'
        )
        mode_frame.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 10))

        self.mode_var = tk.StringVar(value="files_to_notebook")

        # Create radio buttons with tooltips
        files_to_nb = ttk.Radiobutton(
            mode_frame,
            text="Files to Notebook",
            variable=self.mode_var,
            value="files_to_notebook",
            command=self.update_ui
        )
        files_to_nb.grid(row=0, column=0, padx=20)

        nb_to_files = ttk.Radiobutton(
            mode_frame,
            text="Notebook to Files",
            variable=self.mode_var,
            value="notebook_to_files",
            command=self.update_ui
        )
        nb_to_files.grid(row=0, column=1, padx=20)

        # Add tooltips
        self.create_tooltip(
            files_to_nb,
            "Combine multiple source files into a single Jupyter notebook"
        )
        self.create_tooltip(
            nb_to_files,
            "Extract code cells from a Jupyter notebook into separate files"
        )

        # Create frames for different modes
        self.sources_frame = ttk.Frame(self.converter_frame)
        self.notebook_frame = ttk.Frame(self.converter_frame)

        self.setup_sources_frame()
        self.setup_notebook_frame()

        # Show initial frame
        self.update_ui()

    def setup_sources_frame(self):
        # Input Section
        input_frame = ttk.LabelFrame(self.sources_frame, text="Input Sources", padding="10", style='Input.TLabelframe')
        input_frame.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 10))

        # Source directory selection with multiple file/directory support
        ttk.Label(input_frame, text="Input Sources:", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W)
        self.sources_var = tk.StringVar(value="/")
        self.sources_entry = ttk.Entry(input_frame, textvariable=self.sources_var, width=50)
        self.sources_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        browse_frame = ttk.Frame(input_frame)
        browse_frame.grid(row=0, column=2, sticky="w")

        browse_file_btn = ttk.Button(browse_frame, text="Browse Files",
                                   command=self.browse_multiple_files)
        browse_file_btn.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(browse_file_btn, "Select multiple files to include")

        browse_dir_btn = ttk.Button(browse_frame, text="Browse Folders",
                                  command=self.browse_multiple_directories)
        browse_dir_btn.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(browse_dir_btn, "Select multiple folders to include")

        # File extension settings
        ttk.Label(input_frame, text="File Extensions:", style='Header.TLabel').grid(row=1, column=0, sticky=tk.W)
        self.extensions_var = tk.StringVar(value=".py,.txt,.md")
        ext_entry = ttk.Entry(input_frame, textvariable=self.extensions_var, width=50)
        ext_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.create_tooltip(ext_entry, "Comma-separated list of file extensions to include (e.g., .py,.txt,.md)")

        # Include subdirectories option
        self.include_subdirs_var = tk.BooleanVar(value=True)
        subdir_check = ttk.Checkbutton(input_frame, text="Include Subdirectories",
                                      variable=self.include_subdirs_var)
        subdir_check.grid(row=1, column=2, sticky=tk.W)
        self.create_tooltip(subdir_check, "Include files from subdirectories in the conversion")

        # Exclusion Section
        exclude_frame = ttk.LabelFrame(self.sources_frame, text="Exclude Patterns", padding="10", style='Exclude.TLabelframe')
        exclude_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0, 10))

        ttk.Label(exclude_frame, text="Patterns to Exclude:", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W)
        self.exclusions_var = tk.StringVar(value="venv/,__pycache__/,.git/,*.pyc,logs,.env,.vscode*,.aider**,Lib/**")
        self.exclusions_entry = ttk.Entry(exclude_frame, textvariable=self.exclusions_var, width=50)
        self.exclusions_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # Help button for exclusion patterns
        help_btn = ttk.Button(exclude_frame, text="?", width=3,
                            command=self.show_pattern_help)
        help_btn.grid(row=0, column=2, padx=5)
        self.create_tooltip(help_btn, "Click for help with wildcard patterns")

        # Output Section
        output_frame = ttk.LabelFrame(self.sources_frame, text="Output Settings", padding="10", style='Output.TLabelframe')
        output_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(0, 10))

        # Output format selection
        ttk.Label(output_frame, text="Output Format:", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W)
        self.output_format_var = tk.StringVar(value="ipynb")
        format_combo = ttk.Combobox(output_frame, textvariable=self.output_format_var,
                                  values=["ipynb", "py", "html", "md", "pdf"])
        format_combo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.create_tooltip(format_combo, "Select the desired output format")

        # Output directory
        ttk.Label(output_frame, text="Output Directory:", style='Header.TLabel').grid(row=1, column=0, sticky=tk.W)
        self.output_dir_var = tk.StringVar(value="./output")
        self.output_dir_entry = ttk.Entry(output_frame, textvariable=self.output_dir_var, width=50)
        self.output_dir_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        browse_output_btn = ttk.Button(output_frame, text="Browse",
                                     command=lambda: self.browse_path(self.output_dir_var))
        browse_output_btn.grid(row=1, column=2)

        # Output filename
        ttk.Label(output_frame, text="Output Filename:", style='Header.TLabel').grid(row=2, column=0, sticky=tk.W)
        self.output_filename_var = tk.StringVar(value="combined_notebook")
        self.output_filename_entry = ttk.Entry(output_frame, textvariable=self.output_filename_var, width=50)
        self.output_filename_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        # Log Frame with resizable textarea
        log_frame = ttk.LabelFrame(self.sources_frame, text="Log", padding="5")
        log_frame.grid(row=3, column=0, columnspan=3, sticky="nsew", pady=5)
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(0, weight=1)

        self.log_text = tk.Text(log_frame, height=10, width=70)
        self.log_text.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text['yscrollcommand'] = scrollbar.set

        # Run Button
        ttk.Button(self.sources_frame, text="Run Conversion",
                  command=self.run_conversion).grid(row=4, column=0, columnspan=3, pady=10)

    def setup_notebook_frame(self):
        # Notebook file selection
        ttk.Label(self.notebook_frame, text="Input Notebook:", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W)
        self.notebook_path_var = tk.StringVar()
        self.notebook_path_entry = ttk.Entry(self.notebook_frame, textvariable=self.notebook_path_var, width=50)
        self.notebook_path_entry.grid(row=0, column=1, padx=5, pady=5)
        browse_nb_btn = ttk.Button(self.notebook_frame, text="Browse",
                                 command=self.browse_notebook)
        browse_nb_btn.grid(row=0, column=2)

        # Output directory for notebook to files
        ttk.Label(self.notebook_frame, text="Output Directory:", style='Header.TLabel').grid(row=1, column=0, sticky=tk.W)
        self.nb_output_dir_var = tk.StringVar(value="./output")
        self.nb_output_dir_entry = ttk.Entry(self.notebook_frame, textvariable=self.nb_output_dir_var, width=50)
        self.nb_output_dir_entry.grid(row=1, column=1, padx=5, pady=5)
        browse_output_btn = ttk.Button(self.notebook_frame, text="Browse",
                                     command=lambda: self.browse_path(self.nb_output_dir_var))
        browse_output_btn.grid(row=1, column=2)

        # Run Button for notebook to files
        ttk.Button(self.notebook_frame, text="Extract Files",
                  command=self.extract_files).grid(row=2, column=0, columnspan=3, pady=10)

    def browse_notebook(self):
        filename = filedialog.askopenfilename(
            title="Select Notebook",
            filetypes=[("Jupyter Notebooks", "*.ipynb"), ("All Files", "*.*")]
        )
        if filename:
            self.notebook_path_var.set(filename)

    def run_conversion(self):
        try:
            # Create output directory if it doesn't exist
            os.makedirs(self.output_dir_var.get(), exist_ok=True)

            # Get list of files to process
            source_files = self.get_source_files()

            if not source_files:
                self.log_message("No files found matching the specified criteria.")
                return

            # Create new notebook
            nb = new_notebook()

            # Process each file
            for file_path in source_files:
                self.log_message(f"Processing {file_path}...")
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    cell = new_code_cell(content)
                    nb.cells.append(cell)

            # Save notebook
            output_path = os.path.join(self.output_dir_var.get(), self.output_filename_var.get())
            if self.output_format_var.get() == "ipynb":
                with open(output_path + ".ipynb", 'w', encoding='utf-8') as f:
                    nbformat.write(nb, f)
            elif self.output_format_var.get() == "py":
                with open(output_path + ".py", 'w', encoding='utf-8') as f:
                    f.write("\n".join([cell.source for cell in nb.cells]))
            elif self.output_format_var.get() == "html":
                with open(output_path + ".html", 'w', encoding='utf-8') as f:
                    f.write(nbformat.writes(nb, version=4))
            elif self.output_format_var.get() == "md":
                with open(output_path + ".md", 'w', encoding='utf-8') as f:
                    f.write("\n".join([cell.source for cell in nb.cells]))
            elif self.output_format_var.get() == "pdf":
                # Requires additional libraries (e.g., nbconvert)
                pass

            self.log_message(f"Successfully created notebook: {output_path}")

        except Exception as e:
            self.log_message(f"Error: {str(e)}")

    def extract_files(self):
        try:
            notebook_path = self.notebook_path_var.get()
            if not notebook_path:
                messagebox.showerror("Error", "Please select a notebook file")
                return

            output_dir = self.nb_output_dir_var.get()
            os.makedirs(output_dir, exist_ok=True)

            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)

            for i, cell in enumerate(nb.cells):
                if cell.cell_type == 'code':
                    output_file = os.path.join(output_dir, f'cell_{i+1}.py')
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(cell.source)

            messagebox.showinfo("Success", f"Files extracted to {output_dir}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to extract files: {str(e)}")

    def get_source_files(self) -> List[str]:
        source_dir = self.sources_var.get()
        extensions = self.extensions_var.get().split(',')
        exclusions = self.exclusions_var.get().split(',')

        files = []
        for root, dirs, filenames in os.walk(source_dir):
            if not self.include_subdirs_var.get():
                dirs.clear()  # Don't recurse into subdirectories

            # Skip excluded directories
            dirs[:] = [d for d in dirs if not any(fnmatch(d, pat.strip()) for pat in exclusions)]

            for filename in filenames:
                if any(filename.endswith(ext.strip()) for ext in extensions) and \
                   not any(fnmatch(filename, pat.strip()) for pat in exclusions):
                    files.append(os.path.join(root, filename))

        return sorted(files)

    def log_message(self, message: str):
        self.log_text.insert(tk.END, message + '\n')
        self.log_text.see(tk.END)

    def setup_directory_tab(self):
        # Directory path selection
        ttk.Label(self.directory_frame, text="Directory Path:", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W)
        self.dir_path_var = tk.StringVar(value="/")
        self.dir_path_entry = ttk.Entry(self.directory_frame, textvariable=self.dir_path_var, width=50)
        self.dir_path_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        browse_btn = ttk.Button(self.directory_frame, text="Browse",
                              command=lambda: self.browse_path(self.dir_path_var))
        browse_btn.grid(row=0, column=2)

        # Exclusion Section
        exclude_frame = ttk.LabelFrame(self.directory_frame, text="Exclude Patterns", padding="10", style='Exclude.TLabelframe')
        exclude_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(10, 10))

        ttk.Label(exclude_frame, text="Patterns to Exclude:", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W)
        self.dir_exclusions_var = tk.StringVar(value="venv/,__pycache__/,.git/,*.pyc,logs,.env,.vscode*,.aider**,Lib/**")
        self.dir_exclusions_entry = ttk.Entry(exclude_frame, textvariable=self.dir_exclusions_var, width=50)
        self.dir_exclusions_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # Help button for exclusion patterns
        help_btn = ttk.Button(exclude_frame, text="?", width=3,
                            command=self.show_pattern_help)
        help_btn.grid(row=0, column=2, padx=5)
        self.create_tooltip(help_btn, "Click for help with wildcard patterns")

        # Directory structure display
        ttk.Label(self.directory_frame, text="Directory Structure:", style='Header.TLabel').grid(row=2, column=0, sticky=tk.W, pady=(10, 0))

        # Create a frame for the tree and scrollbar
        tree_frame = ttk.Frame(self.directory_frame)
        tree_frame.grid(row=3, column=0, columnspan=3, sticky="nsew", pady=5)
        tree_frame.grid_columnconfigure(0, weight=1)
        tree_frame.grid_rowconfigure(0, weight=1)

        # Create the tree widget
        self.tree = ttk.Treeview(tree_frame, selectmode="browse", height=20)
        self.tree.grid(row=0, column=0, sticky="nsew")

        # Add scrollbar
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scrollbar.set)

        # Configure the tree columns
        self.tree["columns"] = ("size", "type")
        self.tree.column("#0", width=300, minwidth=200)
        self.tree.column("size", width=100, minwidth=100)
        self.tree.column("type", width=100, minwidth=100)

        self.tree.heading("#0", text="Name")
        self.tree.heading("size", text="Size")
        self.tree.heading("type", text="Type")

        # Buttons frame
        button_frame = ttk.Frame(self.directory_frame)
        button_frame.grid(row=4, column=0, columnspan=3, pady=10)

        # Update and Save buttons
        update_btn = ttk.Button(button_frame, text="Update Structure",
                              command=self.update_directory_structure)
        update_btn.pack(side=tk.LEFT, padx=5)

        save_btn = ttk.Button(button_frame, text="Save Structure",
                            command=self.save_directory_structure)
        save_btn.pack(side=tk.LEFT, padx=5)

        # Configure weights for resizing
        self.directory_frame.grid_columnconfigure(1, weight=1)
        self.directory_frame.grid_rowconfigure(3, weight=1)

    def update_directory_structure(self):
        """Update the directory structure display"""
        # Clear the tree
        for item in self.tree.get_children():
            self.tree.delete(item)

        try:
            path = self.dir_path_var.get()
            if not os.path.exists(path):
                messagebox.showerror("Error", f"Path does not exist: {path}")
                return

            # Get exclusion patterns
            exclusion_patterns = [p.strip() for p in self.dir_exclusions_var.get().split(',') if p.strip()]

            def should_exclude(path):
                path = os.path.normpath(path)
                for pattern in exclusion_patterns:
                    if fnmatch(path, pattern) or fnmatch(os.path.basename(path), pattern):
                        return True
                return False

            def insert_node(parent, path, parent_path=""):
                try:
                    if should_exclude(os.path.relpath(path, self.dir_path_var.get())):
                        return

                    name = os.path.basename(path) or path
                    rel_path = os.path.relpath(path, parent_path) if parent_path else path

                    if os.path.isfile(path):
                        size = os.path.getsize(path)
                        size_str = f"{size:,} bytes"
                        self.tree.insert(parent, "end", text=name, values=(size_str, "File"))
                    else:
                        node = self.tree.insert(parent, "end", text=name, values=("", "Directory"))
                        for item in sorted(os.listdir(path)):
                            item_path = os.path.join(path, item)
                            insert_node(node, item_path, path)
                except (PermissionError, OSError) as e:
                    print(f"Error accessing {path}: {e}")

            # Start the recursion
            insert_node("", path)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to update directory structure: {str(e)}")

    def save_directory_structure(self):
        """Save the directory structure to a file"""
        try:
            # Ask for save location
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                title="Save Directory Structure"
            )

            if not file_path:
                return

            def get_structure(node, level=0):
                result = []
                item_text = self.tree.item(node)["text"]
                item_values = self.tree.item(node)["values"]

                if node == "":  # Root node
                    result.append(f"{self.dir_path_var.get()}")
                else:
                    prefix = "    " * (level - 1) + "├── " if level > 0 else ""
                    if item_values[1] == "File":
                        result.append(f"{prefix}{item_text} ({item_values[0]})")
                    else:
                        result.append(f"{prefix}{item_text}/")

                for child in self.tree.get_children(node):
                    result.extend(get_structure(child, level + 1))

                return result

            # Get the structure and write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("Directory Structure\n")
                f.write("=================\n\n")
                f.write("\n".join(get_structure("")))

            messagebox.showinfo("Success", f"Directory structure saved to {file_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save directory structure: {str(e)}")

    def browse_path(self, path_var):
        """Browse for a directory path"""
        initial_dir = path_var.get() if os.path.exists(path_var.get()) else os.getcwd()
        directory = filedialog.askdirectory(initialdir=initial_dir)
        if directory:
            path_var.set(directory)

    def create_tooltip(self, widget, text):
        def show_tooltip(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")

            label = ttk.Label(tooltip, text=text, background="#ffffe0", relief='solid', borderwidth=1)
            label.pack()

            def hide_tooltip():
                tooltip.destroy()

            widget.tooltip = tooltip
            widget.bind('<Leave>', lambda e: hide_tooltip())
            tooltip.bind('<Leave>', lambda e: hide_tooltip())

        widget.bind('<Enter>', show_tooltip)

    def browse_multiple_files(self):
        files = filedialog.askopenfilenames(
            title="Select Files",
            filetypes=[
                ("All Files", "*.*"),
                ("Python Files", "*.py"),
                ("Text Files", "*.txt"),
                ("Markdown Files", "*.md")
            ]
        )
        if files:
            current = self.sources_var.get().strip()
            new_paths = ";".join(files)
            self.sources_var.set(new_paths if not current else f"{current};{new_paths}")

    def browse_multiple_directories(self):
        def select_dir():
            dir_path = filedialog.askdirectory(title="Select Directory")
            if dir_path:
                current = self.sources_var.get().strip()
                self.sources_var.set(dir_path if not current else f"{current};{dir_path}")

        # Create a temporary window for multiple selection
        select_window = tk.Toplevel(self.window)
        select_window.title("Select Multiple Directories")
        select_window.geometry("400x300")

        ttk.Button(select_window, text="Add Directory",
                  command=select_dir).pack(pady=10)
        ttk.Button(select_window, text="Done",
                  command=select_window.destroy).pack(pady=5)

    def show_pattern_help(self):
        help_text = """
Wildcard Pattern Examples:
* *.py - Matches all Python files
* test_* - Matches anything starting with 'test_'
* **/*.txt - Matches .txt files in any subdirectory
* data/**/temp - Matches 'temp' in any subfolder of 'data'
* [abc]*.py - Matches Python files starting with a, b, or c

Common Patterns:
* venv/ - Exclude virtual environment
* __pycache__/ - Exclude Python cache
* .git/ - Exclude git directory
* *.pyc - Exclude compiled Python files
* .DS_Store - Exclude macOS system files
        """
        messagebox.showinfo("Pattern Help", help_text)

    def update_ui(self):
        """Update UI based on selected mode"""
        if self.mode_var.get() == "files_to_notebook":
            if hasattr(self, 'notebook_frame'):
                self.notebook_frame.grid_remove()
            if hasattr(self, 'sources_frame'):
                self.sources_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        else:
            if hasattr(self, 'sources_frame'):
                self.sources_frame.grid_remove()
            if hasattr(self, 'notebook_frame'):
                self.notebook_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))

    def configure_dark_mode(self):
        # Dark mode with darker green text
        darker_green = '#00cc00'  # Slightly darker green
        self.style.configure('TFrame', background='#1e1e1e')
        self.style.configure('TLabel', background='#1e1e1e', foreground=darker_green)
        self.style.configure('TEntry', fieldbackground='#2d2d2d', foreground=darker_green)
        self.style.configure('TButton', background='#2d2d2d', foreground=darker_green)
        self.style.map('TButton',
            background=[('active', '#3d3d3d'), ('pressed', '#2d2d2d')],
            foreground=[('active', darker_green), ('pressed', darker_green)])

        # Radio button styling
        self.style.configure('TRadiobutton', background='#1e1e1e', foreground=darker_green)
        self.style.map('TRadiobutton',
            background=[('active', '#3d3d3d'), ('pressed', '#2d2d2d')],
            foreground=[('active', darker_green), ('pressed', darker_green)])

        self.style.configure('TCheckbutton', background='#1e1e1e', foreground=darker_green)
        self.style.map('TCheckbutton',
            background=[('active', '#3d3d3d'), ('pressed', '#2d2d2d')],
            foreground=[('active', darker_green), ('pressed', darker_green)])

        self.style.configure('TCombobox', fieldbackground='#2d2d2d', foreground=darker_green, selectbackground='#3d3d3d')
        self.style.map('TCombobox',
            fieldbackground=[('readonly', '#2d2d2d')],
            selectbackground=[('readonly', '#3d3d3d')],
            background=[('active', '#3d3d3d'), ('pressed', '#2d2d2d')],
            foreground=[('active', darker_green), ('pressed', darker_green)])

        self.style.configure('TScale', background='#1e1e1e')
        self.style.configure('TLabelframe', background='#1e1e1e', foreground=darker_green)
        self.style.configure('TLabelframe.Label', background='#1e1e1e', foreground=darker_green)
        self.style.configure('TMenubutton', background='#2d2d2d', foreground=darker_green)
        self.style.map('TMenubutton',
            background=[('active', '#3d3d3d'), ('pressed', '#2d2d2d')],
            foreground=[('active', darker_green), ('pressed', darker_green)])

        self.style.configure('TNotebook', background='#1e1e1e')
        self.style.configure('TNotebook.Tab', background='#2d2d2d', foreground=darker_green)
        self.style.map('TNotebook.Tab',
            background=[('selected', '#3d3d3d'), ('active', '#4d4d4d')],
            foreground=[('selected', darker_green), ('active', darker_green)])

        self.style.configure('Header.TLabel', font=('Helvetica', 10, 'bold'), background='#1e1e1e', foreground=darker_green)
        self.style.configure('Section.TLabelframe', padding=10, background='#1e1e1e')
        self.style.configure('Input.TLabelframe', background='#2d2d2d')
        self.style.configure('Exclude.TLabelframe', background='#2d2d2d')
        self.style.configure('Output.TLabelframe', background='#2d2d2d')

        # Configure text widget colors
        if hasattr(self, 'log_text'):
            self.log_text.configure(bg='#2d2d2d', fg=darker_green, insertbackground=darker_green)

        # Update window background
        self.window.configure(bg='#1e1e1e')

    def configure_random_colors(self):
        import random
        def random_color():
            return f'#{random.randint(0, 255):02x}{random.randint(0, 255):02x}{random.randint(0, 255):02x}'

        styles = ['TFrame', 'TLabel', 'TEntry', 'TButton', 'TCheckbutton', 'TCombobox',
                 'TScale', 'TLabelframe', 'TLabelframe.Label', 'TMenubutton', 'TNotebook',
                 'TNotebook.Tab', 'Header.TLabel', 'Section.TLabelframe', 'Input.TLabelframe',
                 'Exclude.TLabelframe', 'Output.TLabelframe']

        for style_name in styles:
            bg_color = random_color()
            fg_color = random_color()
            self.style.configure(style_name, background=bg_color, foreground=fg_color)
            if 'Entry' in style_name or 'Combobox' in style_name:
                self.style.configure(style_name, fieldbackground=bg_color)

        # Configure text widget colors
        if hasattr(self, 'log_text'):
            self.log_text.configure(bg=random_color(), fg=random_color(), insertbackground=random_color())

        # Update window background
        self.window.configure(bg=random_color())

    def toggle_color_mode(self):
        if self.color_mode == 'dark':
            self.color_mode = 'random'
            self.configure_random_colors()
        else:
            self.color_mode = 'dark'
        # Configure dark mode initially
            self.configure_dark_mode()

def main():
    app = NotebookFileConverter()
    app.window.mainloop()

if __name__ == "__main__":
    main()

    def animate_cursor(self):
        # Update cursor color for all entry widgets
        entry_widgets = [widget for widget in self.window.winfo_children() if isinstance(widget, ttk.Entry)]
        for entry in entry_widgets:
            entry.configure(insertbackground=self.cursor_colors[self.cursor_color_index])
        
        # Cycle through colors
        self.cursor_color_index = (self.cursor_color_index + 1) % len(self.cursor_colors)
        
        # Schedule next update
        window.after(500, animate_cursor)

    def get_all_files(self, directory: str, exclusions: list) -> list:
        files = []
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if not any(fnmatch(os.path.join(root, filename), pattern) for pattern in exclusions):
                    files.append(os.path.join(root, filename))
        
        return sorted(files)

    def log_message(self, message: str):
        self.log_text.insert(tk.END, message + '\n')
        self.log_text.see(tk.END)

    def setup_directory_tab(self):
        # Directory path selection
        ttk.Label(self.directory_frame, text="Directory Path:", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W)
        self.dir_path_var = tk.StringVar(value="/")
        self.dir_path_entry = ttk.Entry(self.directory_frame, textvariable=self.dir_path_var, width=50)
        self.dir_path_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        browse_btn = ttk.Button(self.directory_frame, text="Browse", 
                              command=lambda: self.browse_path(self.dir_path_var))
        browse_btn.grid(row=0, column=2)
        
        # Exclusion Section
        exclude_frame = ttk.LabelFrame(self.directory_frame, text="Exclude Patterns", padding="10", style='Exclude.TLabelframe')
        exclude_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(10, 10))
        
        ttk.Label(exclude_frame, text="Patterns to Exclude:", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W)
        self.dir_exclusions_var = tk.StringVar(value="venv/,__pycache__/,.git/,*.pyc,logs,.env,.vscode*,.aider**,Lib/**")
        self.dir_exclusions_entry = ttk.Entry(exclude_frame, textvariable=self.dir_exclusions_var, width=50)
        self.dir_exclusions_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        # Help button for exclusion patterns
        help_btn = ttk.Button(exclude_frame, text="?", width=3, 
                            command=self.show_pattern_help)
        help_btn.grid(row=0, column=2, padx=5)
        self.create_tooltip(help_btn, "Click for help with wildcard patterns")

        # Directory structure display
        ttk.Label(self.directory_frame, text="Directory Structure:", style='Header.TLabel').grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        
        # Create a frame for the tree and scrollbar
        tree_frame = ttk.Frame(self.directory_frame)
        tree_frame.grid(row=3, column=0, columnspan=3, sticky="nsew", pady=5)
        tree_frame.grid_columnconfigure(0, weight=1)
        tree_frame.grid_rowconfigure(0, weight=1)
        
        # Create the tree widget
        self.tree = ttk.Treeview(tree_frame, selectmode="browse", height=20)
        self.tree.grid(row=0, column=0, sticky="nsew")
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Configure the tree columns
        self.tree["columns"] = ("size", "type")
        self.tree.column("#0", width=300, minwidth=200)
        self.tree.column("size", width=100, minwidth=100)
        self.tree.column("type", width=100, minwidth=100)
        
        self.tree.heading("#0", text="Name")
        self.tree.heading("size", text="Size")
        self.tree.heading("type", text="Type")
        
        # Buttons frame
        button_frame = ttk.Frame(self.directory_frame)
        button_frame.grid(row=4, column=0, columnspan=3, pady=10)
        
        # Update and Save buttons
        update_btn = ttk.Button(button_frame, text="Update Structure", 
                              command=self.update_directory_structure)
        update_btn.pack(side=tk.LEFT, padx=5)
        
        save_btn = ttk.Button(button_frame, text="Save Structure", 
                            command=self.save_directory_structure)
        save_btn.pack(side=tk.LEFT, padx=5)
        
        # Configure weights for resizing
        self.directory_frame.grid_columnconfigure(1, weight=1)
        self.directory_frame.grid_rowconfigure(3, weight=1)

    def update_directory_structure(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        try:
            path = self.dir_path_var.get()
            if not os.path.exists(path):
                messagebox.showerror("Error", f"Path does not exist: {path}")
                return
            
            exclusion_patterns = [p.strip() for p in self.dir_exclusions_var.get().split(',') if p.strip()]
            
            def should_exclude(path):
                path = os.path.normpath(path)
                for pattern in exclusion_patterns:
                    if fnmatch(path, pattern) or fnmatch(os.path.basename(path), pattern):
                        return True
                return False
            
            def insert_node(parent, path, parent_path=""):
                try:
                    if should_exclude(os.path.relpath(path, self.dir_path_var.get())):
                        return
                    
                    name = os.path.basename(path) or path
                    rel_path = os.path.relpath(path, parent_path) if parent_path else path
                    
                    if os.path.isfile(path):
                        size = os.path.getsize(path)
                        size_str = f"{size:,} bytes"
                        self.tree.insert(parent, "end", text=name, values=(size_str, "File"))
                    else:
                        node = self.tree.insert(parent, "end", text=name, values=("", "Directory"))
                        for item in sorted(os.listdir(path)):
                            item_path = os.path.join(path, item)
                            insert_node(node, item_path, path)
                except (PermissionError, OSError) as e:
                    print(f"Error accessing {path}: {e}")
            
            insert_node("", path)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update directory structure: {str(e)}")

    def save_directory_structure(self):
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                title="Save Directory Structure"
            )
            
            if not file_path:
                return
            
            def get_structure(node, level=0):
                result = []
                item_text = self.tree.item(node)["text"]
                item_values = self.tree.item(node)["values"]
                
                if node == "":
                    result.append(f"{self.dir_path_var.get()}")
                else:
                    prefix = "    " * (level - 1) + "├── " if level > 0 else ""
                    if item_values[1] == "File":
                        result.append(f"{prefix}{item_text} ({item_values[0]})")
                    else:
                        result.append(f"{prefix}{item_text}/")
                
                for child in self.tree.get_children(node):
                    result.extend(get_structure(child, level + 1))
                
                return result
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("Directory Structure\n")
                f.write("=================\n\n")
                f.write("\n".join(get_structure("")))
            
            messagebox.showinfo("Success", f"Directory structure saved to {file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save directory structure: {str(e)}")

    def browse_path(self, path_var):
        initial_dir = path_var.get() if os.path.exists(path_var.get()) else os.getcwd()
        directory = filedialog.askdirectory(initialdir=initial_dir)
        if directory:
            path_var.set(directory)

    def create_tooltip(self, widget, text):
        def show_tooltip(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            
            label = ttk.Label(tooltip, text=text, background="#ffffe0", relief='solid', borderwidth=1)
            label.pack()
            
            def hide_tooltip():
                tooltip.destroy()
            
            widget.tooltip = tooltip
            widget.bind('<Leave>', lambda e: hide_tooltip())
            tooltip.bind('<Leave>', lambda e: hide_tooltip())
        
        widget.bind('<Enter>', show_tooltip)

    def browse_multiple_files(self):
        files = filedialog.askopenfilenames(
            title="Select Files",
            filetypes=[
                ("All Files", "*.*"),
                ("Python Files", "*.py"),
                ("Text Files", "*.txt"),
                ("Markdown Files", "*.md")
            ]
        )
        if files:
            current = self.sources_var.get().strip()
            new_paths = ";".join(files)
            self.sources_var.set(new_paths if not current else f"{current};{new_paths}")

    def browse_multiple_directories(self):
        def select_dir():
            dir_path = filedialog.askdirectory(title="Select Directory")
            if dir_path:
                current = self.sources_var.get().strip()
                self.sources_var.set(dir_path if not current else f"{current};{dir_path}")
                
        select_window = tk.Toplevel(self.window)
        select_window.title("Select Multiple Directories")
        select_window.geometry("400x300")
        
        ttk.Button(select_window, text="Add Directory", 
                  command=select_dir).pack(pady=10)
        ttk.Button(select_window, text="Done", 
                  command=select_window.destroy).pack(pady=5)

    def show_pattern_help(self):
        help_text = """
Wildcard Pattern Examples:
* *.py - Matches all Python files
* test_* - Matches anything starting with 'test_'
* **/*.txt - Matches .txt files in any subdirectory
* data/**/temp - Matches 'temp' in any subfolder of 'data'
* [abc]*.py - Matches Python files starting with a, b, or c

Common Patterns:
* venv/ - Exclude virtual environment
* __pycache__/ - Exclude Python cache
* .git/ - Exclude git directory
* *.pyc - Exclude compiled Python files
* .DS_Store - Exclude macOS system files
        """
        messagebox.showinfo("Pattern Help", help_text)

    def update_ui(self):
        if self.mode_var.get() == "files_to_notebook":
            if hasattr(self, 'notebook_frame'):
                self.notebook_frame.grid_remove()
            if hasattr(self, 'sources_frame'):
                self.sources_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        else:
            if hasattr(self, 'sources_frame'):
                self.sources_frame.grid_remove()
            if hasattr(self, 'notebook_frame'):
                self.notebook_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))

    def configure_dark_mode(self):
        darker_green = '#00cc00'
        self.style.configure('TFrame', background='#1e1e1e')
        self.style.configure('TLabel', background='#1e1e1e', foreground=darker_green)
        self.style.configure('TEntry', fieldbackground='#2d2d2d', foreground=darker_green)
        self.style.configure('TButton', background='#2d2d2d', foreground=darker_green)
        self.style.map('TButton',
            background=[('active', '#3d3d3d'), ('pressed', '#2d2d2d')],
            foreground=[('active', darker_green), ('pressed', darker_green)])
        
        self.style.configure('TRadiobutton', background='#1e1e1e', foreground=darker_green)
        self.style.map('TRadiobutton',
            background=[('active', '#3d3d3d'), ('pressed', '#2d2d2d')],
            foreground=[('active', darker_green), ('pressed', darker_green)])
