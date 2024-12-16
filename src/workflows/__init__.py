"""Workflow module initialization."""
from .base_workflow import BaseWorkflow, WorkflowState
from .creative_writing import CreativeWritingWorkflow
from .research import ResearchWorkflow
from .workflow_manager import WorkflowManager

__all__ = [
    'BaseWorkflow',
    'WorkflowState',
    'CreativeWritingWorkflow',
    'ResearchWorkflow',
    'WorkflowManager'
]
