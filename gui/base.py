"""Base GUI components for the ID8R application."""

import tkinter as tk
from tkinter import ttk
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class ModifierGroup:
    """Represents a group of text modifiers."""
    name: str
    modifiers: Dict[str, str]
    description: Optional[str] = None


class ModifierTreeView(ttk.Treeview):
    """Custom Treeview for displaying and selecting text modifiers."""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.selected_modifiers: List[tuple] = []
        self.bind("<<TreeviewSelect>>", self.on_select)
        
        # Configure columns
        self.configure(columns=("description",))
        self.column("#0", width=150)
        self.column("description", width=300)
        self.heading("#0", text="Modifier")
        self.heading("description", text="Description")

    def on_select(self, event) -> None:
        """Handle modifier selection."""
        self.selected_modifiers = []
        for item_id in self.selection():
            parent_id = self.parent(item_id)
            if parent_id:  # It's a modifier, not a group
                group = self.item(parent_id)["text"]
                modifier = self.item(item_id)["text"]
                self.selected_modifiers.append((group, modifier))

    def populate_modifiers(self, modifier_groups: Dict[str, ModifierGroup]) -> None:
        """Populate the tree with modifier groups."""
        # Clear existing items
        for item in self.get_children():
            self.delete(item)

        # Add groups and their modifiers
        for group_name, group in modifier_groups.items():
            group_id = self.insert("", "end", text=group_name)
            if group.description:
                self.set(group_id, "description", group.description)

            for mod_name, mod_desc in group.modifiers.items():
                mod_id = self.insert(group_id, "end", text=mod_name)
                self.set(mod_id, "description", mod_desc)
