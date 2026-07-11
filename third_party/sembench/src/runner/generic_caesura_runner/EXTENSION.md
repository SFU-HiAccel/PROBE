# How to Create a CAESURA Runner for Your Scenario

This guide shows how to create a CAESURA runner for a new scenario, using the movie example.

## What is CAESURA?

CAESURA processes natural language queries through a 4-phase pipeline:
1. **DiscoveryPhase** - Understands query and identifies relevant data
2. **PlanningPhase** - Creates execution plan 
3. **MappingPhase** - Maps plan to available tools
4. **RunnerPhase** - Executes the plan step-by-step

## CAESURA Tools

CAESURA automatically selects from these tools based on your query:

- **SqlTool** - Traditional database operations (SELECT, JOIN, GROUP BY)
- **VisualQATool** - Single image analysis (answer questions about individual images)
- **ImageSelectTool** - Semantic image retrieval (find images matching descriptions)
- **PlottingTool** - Basic data visualization (scatter, line, bar plots)
- **TextQATool** - Text information extraction (find answers in text)
- **TransformTool** - AI-powered data transformation (generate Python code)

## Runner Architecture

CAESURA runners have two parts:
- **Generic runner** - handles common functionality and CAESURA execution
- **Scenario runner** - implements specific queries and column selection

## Step 1: Create Your Runner Class

Create `src/scenario/[your_scenario]/runner/caesura_runner/caesura_runner.py`:

```python
from runner.generic_caesura_runner.generic_caesura_runner import GenericCaesuraRunner
import pandas as pd

class CaesuraRunner(GenericCaesuraRunner):
    def __init__(self, use_case: str = "your_scenario", model_name: str = "gpt-4o", 
                 concurrent_llm_worker: int = 1, skip_setup: bool = False):
        super().__init__(use_case, model_name, concurrent_llm_worker, skip_setup)
```

## Step 2: Define Expected Results

Override `_get_empty_results_dataframe()` to specify which columns each query should return:

```python
def _get_empty_results_dataframe(self, query_id: int) -> pd.DataFrame:
    """Define expected columns for each query."""
    
    if query_id == 1:
        return pd.DataFrame(columns=['reviewId'])  # Single item queries
    elif query_id == 3:
        return pd.DataFrame(columns=['count'])     # Count queries
    elif query_id == 5:
        return pd.DataFrame(columns=['id', 'reviewId1', 'reviewId2'])  # Pair queries
    # Add more queries...
    else:
        return pd.DataFrame()
```

## Step 3: Implement Query Methods

Create `_execute_q<N>()` methods for each query:

```python
def _execute_q1(self) -> pd.DataFrame:
    """Execute Q1 and return only the columns needed for evaluation."""
    
    # Get the natural language query
    query_text = self.get_query_text(1, "natural_language")
    
    # Run it through CAESURA
    results = self.execute_caesura_query(query_text)
    
    # Return only the columns the evaluator expects
    if not results.empty and 'reviewId' in results.columns:
        return results[['reviewId']].head(5)
    else:
        return self._get_empty_results_dataframe(1)

def _execute_q3(self) -> pd.DataFrame:
    """Execute Q3 and return count result."""
    
    query_text = self.get_query_text(3, "natural_language")
    results = self.execute_caesura_query(query_text)
    
    # Find the count column
    if not results.empty:
        count_cols = [col for col in results.columns if 'count' in col.lower()]
        if count_cols:
            return results[[count_cols[0]]].rename(columns={count_cols[0]: 'count'})
    
    # Return 0 if no results
    return pd.DataFrame({'count': [0]})
```

## Query Types and Expected Columns

Different queries need different column formats:

| Query Type | Expected Columns | Example |
|------------|------------------|---------|
| **Retrieval** (find items) | Single ID column | `['reviewId']` |
| **Count** (count items) | Single number column | `['count']` |  
| **Ratio** (calculate ratio) | Single number column | `['ratio']` |
| **Pairs** (find relationships) | Three columns | `['id', 'item1', 'item2']` |
| **Group by** (count by category) | Two columns | `['category', 'count']` |

## Column Selection Tips

Use smart column detection with fallbacks:

```python
# 1. Look for expected column names first
expected_cols = [col for col in results.columns if 'review' in col.lower()]

# 2. Fall back to first column if needed
if not expected_cols and not results.empty:
    expected_cols = [results.columns[0]]

# 3. Rename to match evaluator expectations
return results[expected_cols].rename(columns={expected_cols[0]: 'reviewId'})
```

## Data Setup (Optional)

If you need special data processing, override `_setup_database_from_files()`:

```python
def _setup_database_from_files(self):
    """Setup database with custom data processing."""
    from caesura.database.database import Database
    
    db = Database()
    
    # Load and process your data
    data_path = str(self.data_path / "your_data.csv")
    df = pd.read_csv(data_path)
    
    # Clean/process data as needed
    df = df.dropna()
    
    # Add to CAESURA database
    db.add_tabular_table("your_table", data_path, "description")
    
    return db
```

## That's It!

Your runner will now:
1. Execute queries through CAESURA's 4-phase pipeline
2. Use CAESURA's tools automatically based on query needs
3. Return the exact columns the evaluator expects
4. Handle errors gracefully with empty results

The key insight: **customize which columns you return, not how CAESURA works**.