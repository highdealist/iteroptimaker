# Research Workflow

The Research Workflow is a modular component of the ID8R framework designed to facilitate automated research tasks. It implements a systematic approach to gathering, analyzing, and synthesizing information on a given topic.

## Overview

The Research Workflow follows a multi-phase process:
1. Query Generation
2. Search Execution
3. Finding Analysis
4. Summary Generation

## Components

### ResearchState

The `ResearchState` class extends the base `WorkflowState` and tracks:
- Generated research queries
- Search results
- Analyzed findings
- Final summary

### ResearchWorkflow

The `ResearchWorkflow` class implements the research process with the following key methods:

#### Initialization
```python
workflow = ResearchWorkflow(
    model_manager=model_manager,
    tool_manager=tool_manager,
    researcher_agent=researcher,
    writer_agent=writer,
    config={
        "depth": "moderate",  # Options: shallow, moderate, deep
        "focus": "general",   # Options: general, technical, business
        "format": "summary"   # Options: summary, detailed, bullet-points
    }
)
```

#### Workflow Phases

1. **Query Generation**
   - Generates focused research queries based on input topic
   - Considers research depth and focus from configuration
   - Returns structured list of queries

2. **Search Execution**
   - Utilizes available search tools
   - Handles search failures gracefully
   - Returns standardized search results

3. **Finding Analysis**
   - Analyzes search results for key insights
   - Extracts structured findings
   - Maintains context and relationships

4. **Summary Generation**
   - Synthesizes findings into coherent summary
   - Adapts to specified output format
   - Highlights key insights and implications

## Usage

```python
from id8r.workflows.research import ResearchWorkflow

# Initialize workflow
workflow = ResearchWorkflow(...)

# Create initial state
state = workflow.initialize(
    input_text="Topic to research",
    context="Additional context"
)

# Execute workflow
result = workflow.execute(state)

# Access results
print(result.summary)  # Final research summary
print(result.findings) # List of key findings
```

## Configuration Options

### Depth
- `shallow`: Quick overview
- `moderate`: Balanced depth (default)
- `deep`: Comprehensive analysis

### Focus
- `general`: Broad overview
- `technical`: Technical details
- `business`: Business implications

### Format
- `summary`: Narrative summary
- `detailed`: In-depth report
- `bullet-points`: Structured points

## Error Handling

The workflow implements comprehensive error handling:
- Missing tool/agent validation
- Search failure recovery
- State validation
- Logging for debugging

## Integration

The Research Workflow integrates with:
- Model Manager for AI operations
- Tool Manager for search capabilities
- Agent system for specialized tasks
- Workflow Manager for orchestration

## Testing

Comprehensive test coverage includes:
- Unit tests for each component
- Integration tests for workflow phases
- Error handling scenarios
- Configuration validation

## Best Practices

1. **Configuration**
   - Set appropriate depth/focus for task
   - Configure error handling strategy
   - Tune search parameters

2. **Usage**
   - Provide clear input topics
   - Include relevant context
   - Handle results appropriately

3. **Integration**
   - Use with other workflows
   - Extend for specific needs
   - Maintain modularity

## Examples

### Basic Research
```python
workflow = ResearchWorkflow(...)
state = workflow.initialize("Artificial Intelligence trends")
result = workflow.execute(state)
print(result.summary)
```

### Technical Deep-Dive
```python
workflow = ResearchWorkflow(
    config={
        "depth": "deep",
        "focus": "technical",
        "format": "detailed"
    }
)
state = workflow.initialize("Quantum Computing algorithms")
result = workflow.execute(state)
```

### Business Analysis
```python
workflow = ResearchWorkflow(
    config={
        "depth": "moderate",
        "focus": "business",
        "format": "bullet-points"
    }
)
state = workflow.initialize("Market trends in AI")
result = workflow.execute(state)
```
