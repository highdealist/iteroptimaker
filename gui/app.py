"""Main application window for the ID8R GUI."""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Dict, Any, List, Optional
import logging

from ..models.model_manager import ModelManager
from ..tools.tool_manager import ToolManager
from ..agents.base.agent import BaseAgent
from .base import ModifierTreeView, ModifierGroup

logger = logging.getLogger(__name__)

class App(tk.Tk):
    """Main application window."""

    modifier_groups: Dict[str, Dict[str, str]] = {
        "SONG WRITING": {
            "Brainstorm Concepts": "Generate potential concepts for song lyrics",
            "Narrow Down and Select Concept": "Evaluate and select the best concept"
        },
        "JOKE WRITING": {
            "Comedy Technique": "Analyze and suggest comedy techniques",
            "Punch-up": "Enhance humor and engagement"
        },
        "REASONING": {
            "Analyze for error": "Find potential errors and flaws",
            "Identify Assumptions": "Uncover unstated assumptions",
            "Consider Counterarguments": "Explore alternative perspectives",
            "Logical Fallacies": "Identify and address logical fallacies",
            "Additional Context": "Add comprehensive context",
            "Organize for Clarity": "Improve structure and organization",
            "Clarify Ambiguities": "Address vague statements",
            "Enhance with examples": "Add specific examples and evidence",
            "Mitigate Bias": "Address potential biases",
            "Address Limitations": "Acknowledge and handle limitations",
            "Lossless Conciseness": "Make concise without losing information",
            "Broaden Usefulness": "Address wider range of scenarios",
            "Explore and Elaborate Implications": "Explore consequences",
            "Enhance Applicability": "Make more actionable",
            "Enrich with Interdisciplinary Connections": "Add interdisciplinary insights",
            "Refine certainty/uncertainty": "Convey certainty levels accurately",
            "Is anything missing?": "Identify missing elements",
            "Enhance engagement potential": "Make more engaging",
            "Recap and synthesize": "Create refined final response"
        }
    }

    def __init__(
        self,
        model_manager: ModelManager,
        tool_manager: ToolManager,
        researcher_agent: BaseAgent,
        writer_agent: BaseAgent,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.title("ID8R Creative AI")
        self.geometry("1024x768")

        # Initialize managers and agents
        self.model_manager = model_manager
        self.tool_manager = tool_manager
        self.researcher_agent = researcher_agent
        self.writer_agent = writer_agent

        # Initialize state
        self.chat_log: List[str] = []
        self.context = ""
        self.current_prompt = ""
        self.last_output = ""

        # Setup UI components
        self.setup_ui()

    def setup_ui(self) -> None:
        """Setup the main UI components."""
        self.setup_menu()
        self.setup_main_frame()
        self.setup_sidebar()

    def setup_menu(self) -> None:
        """Setup the application menu."""
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open", command=self.open_file)
        file_menu.add_command(label="Save", command=self.save_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)

        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Undo", command=self.undo)
        edit_menu.add_command(label="Redo", command=self.redo)
        edit_menu.add_separator()
        edit_menu.add_command(label="Cut", command=self.cut)
        edit_menu.add_command(label="Copy", command=self.copy)
        edit_menu.add_command(label="Paste", command=self.paste)

    def setup_main_frame(self) -> None:
        """Setup the main content frame."""
        main_frame = ttk.Frame(self, padding="10")
        main_frame.grid(row=0, column=1, sticky="nsew")

        # Input area
        input_frame = ttk.LabelFrame(main_frame, text="Input", padding="5")
        input_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 5))

        self.input_text = tk.Text(input_frame, height=5, wrap=tk.WORD)
        self.input_text.grid(row=0, column=0, sticky="nsew")
        self.input_text.bind("<Return>", self.run_workflow)

        # Output area
        output_frame = ttk.LabelFrame(main_frame, text="Output", padding="5")
        output_frame.grid(row=1, column=0, sticky="nsew")

        self.output_text = tk.Text(output_frame, height=20, wrap=tk.WORD)
        self.output_text.grid(row=0, column=0, sticky="nsew")

        # Configure weights
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)

    def setup_sidebar(self) -> None:
        """Setup the sidebar with modifiers."""
        sidebar = ttk.Frame(self, padding="10")
        sidebar.grid(row=0, column=0, sticky="ns")

        # Create ModifierTreeView
        self.modifier_tree = ModifierTreeView(sidebar)
        self.modifier_tree.grid(row=0, column=0, sticky="nsew")

        # Convert modifier_groups to ModifierGroup objects
        modifier_groups = {
            name: ModifierGroup(name=name, modifiers=modifiers)
            for name, modifiers in self.modifier_groups.items()
        }
        self.modifier_tree.populate_modifiers(modifier_groups)

    def run_workflow(self, event=None) -> None:
        """Run the creative workflow with selected modifiers."""
        user_input = self.input_text.get("1.0", tk.END).strip()
        if not user_input:
            return

        try:
            # Generate prompt with selected modifiers
            prompt = self.generate_prompt(user_input)

            # Run research phase
            research_result = self.researcher_agent.generate_response(prompt)
            
            # Run writing phase with research context
            final_result = self.writer_agent.generate_response(
                prompt, context=research_result
            )

            # Process and validate agent response
            self.process_agent_response(final_result)

        except Exception as e:
            logger.error(f"Error in workflow: {e}")
            messagebox.showerror("Error", f"An error occurred: {e}")

    def generate_prompt(self, user_input: str) -> str:
        """Generate a prompt with selected modifiers."""
        modifiers = self.modifier_tree.selected_modifiers
        if not modifiers:
            return user_input

        prompt = f"Input: {user_input}\n\nApply the following modifications:\n"
        for group, modifier in modifiers:
            prompt += f"\n- {modifier} ({group})"
        
        return prompt

    def open_file(self) -> None:
        """Open and load a file."""
        try:
            filepath = filedialog.askopenfilename(
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            if not filepath:
                return
                
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            self.input_text.delete('1.0', tk.END)
            self.input_text.insert('1.0', content)
            self.current_file = filepath
        except Exception as e:
            logger.error(f"Error opening file: {e}")
            messagebox.showerror("Error", f"Failed to open file: {e}")

    def save_file(self) -> None:
        """Save current content to a file."""
        try:
            if not hasattr(self, 'current_file'):
                filepath = filedialog.asksaveasfilename(
                    defaultextension=".txt",
                    filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
                )
                if not filepath:
                    return
                self.current_file = filepath
                
            content = self.input_text.get('1.0', tk.END)
            with open(self.current_file, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            messagebox.showerror("Error", f"Failed to save file: {e}")

    def process_agent_response(self, response: str) -> None:
        """Process and validate agent response before updating UI."""
        try:
            # Basic validation
            if not response or not isinstance(response, str):
                raise ValueError("Invalid response from agent")
                
            # Update UI with validated response
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert("1.0", response)
            
        except Exception as e:
            logger.error(f"Error processing agent response: {e}")
            messagebox.showerror("Error", "Failed to process agent response")

    # Edit operations
    def undo(self) -> None:
        """Undo last text change."""
        try:
            self.input_text.edit_undo()
        except tk.TclError:
            pass

    def redo(self) -> None:
        """Redo last undone text change."""
        try:
            self.input_text.edit_redo()
        except tk.TclError:
            pass

    def cut(self) -> None:
        """Cut selected text."""
        try:
            self.clipboard_clear()
            self.clipboard_append(self.input_text.selection_get())
            self.input_text.delete(tk.SEL_FIRST, tk.SEL_LAST)
        except tk.TclError:
            pass

    def copy(self) -> None:
        """Copy selected text."""
        try:
            self.clipboard_clear()
            self.clipboard_append(self.input_text.selection_get())
        except tk.TclError:
            pass

    def paste(self) -> None:
        """Paste text from clipboard."""
        try:
            self.input_text.insert(tk.INSERT, self.clipboard_get())
        except tk.TclError:
            pass
