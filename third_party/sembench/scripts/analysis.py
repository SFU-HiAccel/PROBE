#!/usr/bin/env python3

import json
import os
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class SystemAnalyzer:
    def __init__(self, base_path: str = "./files", output_dir: str = "./analysis_results", tolerance_levels: List[float] = None):
        self.base_path = Path(base_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Snowflake inclusion setting - set to True to include snowflake in analysis
        # self.include_snowflake = True  # Uncomment this line to include snowflake
        self.include_snowflake = False  # Default: exclude snowflake
        self.include_bigquery = True   # Default: exclude bigquery
        
        # Configurable tolerance levels
        if tolerance_levels:
            self.tolerance_levels = tolerance_levels
        else:
            # Default metric-specific tolerance levels
            self.tolerance_levels = {
                'execution_time': [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0, 1.5, 2.0],
                'money_cost': [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0, 1.5, 2.0],
                'quality': [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.0]
            }
        
        # Define query groupings by operator type
        # Note: mmqa queries use different IDs in the JSON files (Q1, Q2a, etc.)
        self.operator_queries = {
            'SEM_CLASSIFY': {
                'ecomm': ['Q5', 'Q6'],
                'medical': ['Q10', 'Q6', 'Q7'],
                'mmqa': ['Q4'],  # q4 -> Q4 in JSON
                'movie': ['Q8']
            },
            'SEM_FILTER': {
                'ecomm': ['Q1', 'Q10', 'Q11', 'Q13', 'Q14', 'Q2'],
                'medical': ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9'],
                'mmqa': ['Q3a', 'Q3b', 'Q3c', 'Q3d', 'Q3e', 'Q3f', 'Q3g', 'Q6a', 'Q6b', 'Q6c'],  # Updated to match JSON IDs
                'movie': ['Q1', 'Q2', 'Q3', 'Q4'],
                'animals': ['Q1', 'Q10', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9']  # wildlife -> animals
            },
            'SEM_JOIN': {
                'ecomm': ['Q10', 'Q11', 'Q14', 'Q7', 'Q8', 'Q9'],
                'mmqa': ['Q2a', 'Q2b', 'Q7'],  # Updated to match JSON IDs
                'movie': ['Q5', 'Q6', 'Q7']
            },
            'SEM_MAP': {
                'ecomm': ['Q3', 'Q4'],
                'mmqa': ['Q1', 'Q2b', 'Q4', 'Q5']  # Added Q5 from former SEM_AGG
            },
            'SEM_SCORE': {
                'ecomm': ['Q14'],
                'movie': ['Q10', 'Q9']
            }
        }
        
        # Available scenarios and systems (will be populated by scanning)
        self.scenarios = []
        self.systems = []
        
        # Data storage
        self.data = {}  # scenario -> system -> query -> metrics
    
    def set_tolerance_levels(self, tolerance_levels):
        """
        Set custom tolerance levels for analysis.
        
        Args:
            tolerance_levels: Either a list (same for all metrics) or dict with metric-specific levels
                             e.g., [0.0, 0.1, 0.2, 0.5] or {'execution_time': [...], 'money_cost': [...], 'quality': [...]}
        """
        if isinstance(tolerance_levels, list):
            # Use same levels for all metrics
            self.tolerance_levels = {
                'execution_time': tolerance_levels,
                'money_cost': tolerance_levels,
                'quality': tolerance_levels
            }
        else:
            # Use metric-specific levels
            self.tolerance_levels = tolerance_levels
        print(f"Updated tolerance levels to: {self.tolerance_levels}")
    
    @staticmethod
    def create_tolerance_range(start: float = 0.0, stop: float = 1.0, step: float = 0.1) -> List[float]:
        """
        Create a range of tolerance levels.
        
        Args:
            start: Starting tolerance value (default: 0.0)
            stop: Ending tolerance value (default: 1.0)  
            step: Step size (default: 0.1)
            
        Returns:
            List of tolerance values
        """
        return [round(x, 2) for x in np.arange(start, stop + step/2, step)]
    
    def calculate_system_upper_bounds(self) -> Dict[str, int]:
        """
        Calculate the upper bound (maximum possible wins) for each system.
        This is the number of competitive queries each system can participate in.
        Only counts queries where at least 2 systems can compete.
        
        Returns:
            {system: max_possible_wins}
        """
        upper_bounds = defaultdict(int)
        
        for scenario in self.scenarios:
            scenario_systems = self.data[scenario]
            if not scenario_systems:
                continue
                
            # Find all available queries across systems for this scenario
            all_queries = set()
            for system_data in scenario_systems.values():
                all_queries.update(system_data.keys())
            
            # Count competitive queries for each system (where >=2 systems can compete)
            for query in all_queries:
                query_participants = []
                for system in self.systems:
                    if system in scenario_systems and query in scenario_systems[system]:
                        query_data = scenario_systems[system][query]
                        if query_data.get('status') == 'success':
                            query_participants.append(system)
                
                # Only count if at least 2 systems can compete
                if len(query_participants) >= 2:
                    for system in query_participants:
                        upper_bounds[system] += 1
        
        return dict(upper_bounds)
    
    def calculate_operator_upper_bounds(self) -> Dict[str, Dict[str, int]]:
        """Calculate upper bounds for each operator type"""
        operator_upper_bounds = {}
        
        for operator_type, operator_scenarios in self.operator_queries.items():
            operator_upper_bounds[operator_type] = defaultdict(int)
            
            for scenario, queries in operator_scenarios.items():
                if scenario not in self.data:
                    continue
                    
                scenario_systems = self.data[scenario]
                
                for query in queries:
                    query_participants = []
                    for system in self.systems:
                        if system in scenario_systems and query in scenario_systems[system]:
                            query_data = scenario_systems[system][query]
                            if query_data.get('status') == 'success':
                                query_participants.append(system)
                    
                    # Only count if at least 2 systems can compete
                    if len(query_participants) >= 2:
                        for system in query_participants:
                            operator_upper_bounds[operator_type][system] += 1
        
        return {op: dict(bounds) for op, bounds in operator_upper_bounds.items()}
    
    def calculate_scenario_upper_bounds(self) -> Dict[str, Dict[str, int]]:
        """Calculate upper bounds for each scenario"""
        scenario_upper_bounds = {}
        
        for scenario in self.scenarios:
            scenario_upper_bounds[scenario] = defaultdict(int)
            scenario_systems = self.data[scenario]
            if not scenario_systems:
                continue
                
            # Find all available queries for this scenario
            all_queries = set()
            for system_data in scenario_systems.values():
                all_queries.update(system_data.keys())
            
            # Count competitive queries for each system
            for query in all_queries:
                query_participants = []
                for system in self.systems:
                    if system in scenario_systems and query in scenario_systems[system]:
                        query_data = scenario_systems[system][query]
                        if query_data.get('status') == 'success':
                            query_participants.append(system)
                
                # Only count if at least 2 systems can compete
                if len(query_participants) >= 2:
                    for system in query_participants:
                        scenario_upper_bounds[scenario][system] += 1
        
        return {scenario: dict(bounds) for scenario, bounds in scenario_upper_bounds.items()}
    
    def find_convergence_tolerance(self, max_search_tolerances: Dict[str, float] = None, steps: Dict[str, float] = None) -> Dict[str, Dict[str, float]]:
        """
        Find the minimum tolerance level where each system reaches its upper bound for each metric.
        
        Args:
            max_search_tolerances: Dict with max tolerance to search for each metric
            steps: Dict with step size for each metric
            
        Returns:
            {metric: {system: convergence_tolerance}}
        """
        # Set metric-specific search parameters
        if max_search_tolerances is None:
            max_search_tolerances = {
                'execution_time': 3000.0,  # Search up to 300,000% for extreme time ratios
                'money_cost': 7000.0,      # Search up to 700,000% for extreme cost ratios  
                'quality': 2.0             # Search up to 200% for quality (absolute tolerance)
            }
        
        if steps is None:
            steps = {
                'execution_time': 10.0,   # Very coarse steps for extreme ranges
                'money_cost': 20.0,       # Very coarse steps for extreme ranges
                'quality': 0.02           # Fine steps for quality
            }
        
        upper_bounds = self.calculate_system_upper_bounds()
        convergence_tolerances = {
            'execution_time': {},
            'money_cost': {},
            'quality': {}
        }
        
        print("Finding convergence tolerances with metric-specific search ranges...")
        
        # Search for convergence tolerance for each metric and system
        for metric in ['execution_time', 'money_cost', 'quality']:
            max_search = max_search_tolerances[metric]
            step = steps[metric]
            print(f"  Analyzing {metric} (search range: 0-{max_search}, step: {step})...")
            
            for system in self.systems:
                if system not in upper_bounds:
                    continue
                    
                target_wins = upper_bounds[system]
                convergence_tolerance = None
                
                # Linear search with metric-specific parameters
                search_tolerances = [i * step for i in range(int(max_search / step) + 1)]
                
                for test_tolerance in search_tolerances:
                    test_levels = {
                        'execution_time': [test_tolerance],
                        'money_cost': [test_tolerance],
                        'quality': [test_tolerance]
                    }
                    tolerance_results = self.find_winners_with_tolerance(test_levels)
                    actual_wins = tolerance_results[metric][test_tolerance].get(system, 0)
                    
                    if actual_wins >= target_wins:
                        convergence_tolerance = test_tolerance
                        break
                
                if convergence_tolerance is not None:
                    convergence_tolerances[metric][system] = convergence_tolerance
                    print(f"    {system}: {convergence_tolerance:.3f} ({target_wins} wins)")
                else:
                    print(f"    {system}: >={max_search} (not converged)")
        
        return convergence_tolerances
    
    def _save_convergence_results(self, convergence_tolerances: Dict[str, Dict[str, float]]):
        """Save convergence tolerance results to file"""
        output_lines = []
        
        output_lines.append("# Convergence Tolerance Analysis")
        output_lines.append("")
        output_lines.append("Minimum tolerance levels where each system reaches its upper bound:")
        output_lines.append("")
        
        for metric in ['execution_time', 'money_cost', 'quality']:
            metric_name = metric.replace('_', ' ').title()
            output_lines.append(f"## {metric_name}")
            output_lines.append("")
            
            if metric in convergence_tolerances:
                for system, tolerance in sorted(convergence_tolerances[metric].items()):
                    tolerance_pct = tolerance * 100
                    output_lines.append(f"- **{system}**: {tolerance:.3f} ({tolerance_pct:.1f}%)")
                output_lines.append("")
        
        # Save to file
        with open(self.output_dir / "convergence_tolerances.md", 'w') as f:
            f.write("\n".join(output_lines))
        
        print(f"Convergence analysis saved to {self.output_dir}/convergence_tolerances.md")
        
    def scan_available_data(self):
        """Scan the file system to determine available scenarios and systems"""
        self.scenarios = []
        self.systems = set()
        
        for scenario_dir in self.base_path.iterdir():
            if scenario_dir.is_dir():
                # Check for standard metrics directory
                standard_metrics = scenario_dir / "metrics" / "across_system_2.5flash"
                
                if standard_metrics.exists():
                    self.scenarios.append(scenario_dir.name)
                    
                    # Check available systems for this scenario
                    metrics_dir = standard_metrics
                    for system_file in metrics_dir.glob("*.json"):
                        system_name = system_file.stem
                        if system_name == "snowflake" and not self.include_snowflake:
                            pass  # or 'return' / 'continue' depending on context
                        elif system_name == "bigquery" and not self.include_bigquery:
                            pass
                        else:
                            self.systems.add(system_name)

        
        self.systems = sorted(list(self.systems))
        self.scenarios = sorted(self.scenarios)
        
        print(f"Found scenarios: {self.scenarios}")
        print(f"Found systems: {self.systems}")
        
    def load_data(self):
        """Load all metrics data from JSON files"""
        self.data = defaultdict(lambda: defaultdict(dict))
        
        for scenario in self.scenarios:
            # if scenario in ["ecomm", "medical", "mmqa"]:
            #     continue
            
            # Use different metrics directory for mmqa
            metrics_dir = self.base_path / scenario / "metrics" / "across_system_2.5flash"
            
            for system in self.systems:
                json_file = metrics_dir / f"{system}.json"
                if json_file.exists():
                    try:
                        with open(json_file, 'r') as f:
                            system_data = json.load(f)
                            self.data[scenario][system] = system_data
                    except Exception as e:
                        print(f"Error loading {json_file}: {e}")
        
        print(f"Loaded data for {len(self.data)} scenarios")
        
    def extract_metrics(self, query_data: Dict[str, Any]) -> Tuple[float, float, float]:
        """
        Extract the three main metrics: execution_time, money_cost, quality
        For quality, use f1_score, transformed relative_error, or spearman_correlation
        """
        execution_time = query_data.get('execution_time', float('inf'))
        money_cost = query_data.get('money_cost', float('inf'))
        
        # Determine quality metric
        quality = 0.0
        if 'f1_score' in query_data:
            quality = query_data['f1_score']
        elif 'accuracy' in query_data:
            quality = query_data['accuracy']
        elif 'relative_error' in query_data:
            # Transform relative error: 1/(1 + relative_error)
            relative_error = query_data['relative_error']
            quality = 1.0 / (1.0 + relative_error)
        elif 'spearman_correlation' in query_data:
            # Assume correlation is positive (as mentioned by user)
            quality = max(0.0, query_data['spearman_correlation'])
        
        return execution_time, money_cost, quality
    
    def find_winners_by_scenario(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Find winners for each scenario for each metric.
        Returns: {scenario: {metric: {system: count_of_wins}}}
        """
        results = {}
        
        for scenario in self.scenarios:
            results[scenario] = {
                'execution_time': defaultdict(float),
                'money_cost': defaultdict(float),
                'quality': defaultdict(float)
            }
            
            # Get all queries for this scenario
            scenario_systems = self.data[scenario]
            if not scenario_systems:
                continue
                
            # Find all available queries across systems
            all_queries = set()
            for system_data in scenario_systems.values():
                all_queries.update(system_data.keys())
            
            # Compare each query across systems
            for query in all_queries:
                query_metrics = {}  # system -> (execution_time, money_cost, quality)
                
                for system in self.systems:
                    if system in scenario_systems and query in scenario_systems[system]:
                        query_data = scenario_systems[system][query]
                        if query_data.get('status') == 'success':
                            query_metrics[system] = self.extract_metrics(query_data)
                
                if len(query_metrics) < 2:  # Need at least 2 systems to compare
                    continue
                
                # Find winners for each metric (lower is better for time and cost, higher for quality)
                # Execution time (lower is better)
                if query_metrics:
                    min_time = min(metrics[0] for metrics in query_metrics.values())
                    time_winners = [sys for sys, metrics in query_metrics.items() if metrics[0] == min_time]
                    for winner in time_winners:
                        results[scenario]['execution_time'][winner] += 1 / len(time_winners)
                
                # Money cost (lower is better)
                if query_metrics:
                    min_cost = min(metrics[1] for metrics in query_metrics.values())
                    cost_winners = [sys for sys, metrics in query_metrics.items() if metrics[1] == min_cost]
                    for winner in cost_winners:
                        results[scenario]['money_cost'][winner] += 1 / len(cost_winners)
                
                # Quality (higher is better)
                if query_metrics:
                    max_quality = max(metrics[2] for metrics in query_metrics.values())
                    quality_winners = [sys for sys, metrics in query_metrics.items() if metrics[2] == max_quality]
                    for winner in quality_winners:
                        results[scenario]['quality'][winner] += 1 / len(quality_winners)
        
        return results
    
    def find_winners_across_scenarios(self) -> Dict[str, Dict[str, float]]:
        """
        Find winners across all scenarios for each metric.
        Returns: {metric: {system: total_wins}}
        """
        scenario_results = self.find_winners_by_scenario()
        
        cross_scenario_results = {
            'execution_time': defaultdict(float),
            'money_cost': defaultdict(float),
            'quality': defaultdict(float)
        }
        
        for scenario, scenario_data in scenario_results.items():
            for metric in ['execution_time', 'money_cost', 'quality']:
                for system, wins in scenario_data[metric].items():
                    cross_scenario_results[metric][system] += wins
        
        return cross_scenario_results
    
    def calculate_system_statistics_across_scenarios(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Calculate statistics (average, median, sum) for each system across all scenarios
        Returns: {system: {metric: {avg: value, median: value, sum: value}}}
        """
        system_stats = {}
        
        for system in self.systems:
            system_stats[system] = {
                'execution_time': {'values': [], 'avg': 0.0, 'median': 0.0, 'sum': 0.0},
                'money_cost': {'values': [], 'avg': 0.0, 'median': 0.0, 'sum': 0.0},
                'quality': {'values': [], 'avg': 0.0, 'median': 0.0, 'sum': 0.0}
            }
        
        # Collect all values for each system across all scenarios
        for scenario in self.scenarios:
            scenario_systems = self.data[scenario]
            if not scenario_systems:
                continue
                
            for system in self.systems:
                if system in scenario_systems:
                    for query, query_data in scenario_systems[system].items():
                        if query_data.get('status') == 'success':
                            execution_time, money_cost, quality = self.extract_metrics(query_data)
                            
                            # Only add finite values
                            if execution_time != float('inf'):
                                system_stats[system]['execution_time']['values'].append(execution_time)
                            if money_cost != float('inf'):
                                system_stats[system]['money_cost']['values'].append(money_cost)
                            if quality > 0:
                                system_stats[system]['quality']['values'].append(quality)
        
        # Calculate statistics for each system
        for system in self.systems:
            for metric in ['execution_time', 'money_cost', 'quality']:
                values = system_stats[system][metric]['values']
                if values:
                    system_stats[system][metric]['avg'] = np.mean(values)
                    system_stats[system][metric]['median'] = np.median(values)
                    system_stats[system][metric]['sum'] = np.sum(values)
                # Remove the raw values to keep the result clean
                del system_stats[system][metric]['values']
        
        return system_stats
    
    def find_winners_by_operator_type(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Find winners by operator type for each metric.
        Returns: {operator_type: {metric: {system: wins}}}
        """
        results = {}
        
        for operator_type, operator_scenarios in self.operator_queries.items():
            results[operator_type] = {
                'execution_time': defaultdict(float),
                'money_cost': defaultdict(float),
                'quality': defaultdict(float)
            }
            
            for scenario, queries in operator_scenarios.items():
                if scenario not in self.data:
                    continue
                    
                scenario_systems = self.data[scenario]
                
                for query in queries:
                    query_metrics = {}  # system -> (execution_time, money_cost, quality)
                    
                    for system in self.systems:
                        if system in scenario_systems and query in scenario_systems[system]:
                            query_data = scenario_systems[system][query]
                            if query_data.get('status') == 'success':
                                query_metrics[system] = self.extract_metrics(query_data)
                    
                    if len(query_metrics) < 2:
                        continue
                    
                    # Find winners for each metric
                    # Execution time (lower is better)
                    if query_metrics:
                        min_time = min(metrics[0] for metrics in query_metrics.values())
                        time_winners = [sys for sys, metrics in query_metrics.items() if metrics[0] == min_time]
                        for winner in time_winners:
                            results[operator_type]['execution_time'][winner] += 1 / len(time_winners)
                    
                    # Money cost (lower is better)
                    if query_metrics:
                        min_cost = min(metrics[1] for metrics in query_metrics.values())
                        cost_winners = [sys for sys, metrics in query_metrics.items() if metrics[1] == min_cost]
                        for winner in cost_winners:
                            results[operator_type]['money_cost'][winner] += 1 / len(cost_winners)
                    
                    # Quality (higher is better)
                    if query_metrics:
                        max_quality = max(metrics[2] for metrics in query_metrics.values())
                        quality_winners = [sys for sys, metrics in query_metrics.items() if metrics[2] == max_quality]
                        for winner in quality_winners:
                            results[operator_type]['quality'][winner] += 1 / len(quality_winners)
        
        return results
    
    def calculate_system_statistics_by_operator_type(self) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
        """
        Calculate statistics (average, median, sum) for each system by operator type
        Returns: {operator_type: {system: {metric: {avg: value, median: value, sum: value}}}}
        """
        operator_stats = {}
        
        for operator_type, operator_scenarios in self.operator_queries.items():
            operator_stats[operator_type] = {}
            
            for system in self.systems:
                operator_stats[operator_type][system] = {
                    'execution_time': {'values': [], 'avg': 0.0, 'median': 0.0, 'sum': 0.0},
                    'money_cost': {'values': [], 'avg': 0.0, 'median': 0.0, 'sum': 0.0},
                    'quality': {'values': [], 'avg': 0.0, 'median': 0.0, 'sum': 0.0}
                }
            
            # Collect values for each system in this operator type
            for scenario, queries in operator_scenarios.items():
                if scenario not in self.data:
                    continue
                    
                scenario_systems = self.data[scenario]
                
                for query in queries:
                    for system in self.systems:
                        if system in scenario_systems and query in scenario_systems[system]:
                            query_data = scenario_systems[system][query]
                            if query_data.get('status') == 'success':
                                execution_time, money_cost, quality = self.extract_metrics(query_data)
                                
                                # Only add finite values
                                if execution_time != float('inf'):
                                    operator_stats[operator_type][system]['execution_time']['values'].append(execution_time)
                                if money_cost != float('inf'):
                                    operator_stats[operator_type][system]['money_cost']['values'].append(money_cost)
                                if quality > 0:
                                    operator_stats[operator_type][system]['quality']['values'].append(quality)
            
            # Calculate statistics for each system in this operator type
            for system in self.systems:
                for metric in ['execution_time', 'money_cost', 'quality']:
                    values = operator_stats[operator_type][system][metric]['values']
                    if values:
                        operator_stats[operator_type][system][metric]['avg'] = np.mean(values)
                        operator_stats[operator_type][system][metric]['median'] = np.median(values)
                        operator_stats[operator_type][system][metric]['sum'] = np.sum(values)
                    # Remove the raw values to keep the result clean
                    del operator_stats[operator_type][system][metric]['values']
        
        return operator_stats
    
    def calculate_system_statistics_by_scenario(self) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
        """
        Calculate statistics (average, median, sum) for each system by scenario
        Returns: {scenario: {system: {metric: {avg: value, median: value, sum: value}}}}
        """
        scenario_stats = {}
        
        for scenario in self.scenarios:
            scenario_stats[scenario] = {}
            scenario_systems = self.data[scenario]
            if not scenario_systems:
                continue
            
            for system in self.systems:
                scenario_stats[scenario][system] = {
                    'execution_time': {'values': [], 'avg': 0.0, 'median': 0.0, 'sum': 0.0},
                    'money_cost': {'values': [], 'avg': 0.0, 'median': 0.0, 'sum': 0.0},
                    'quality': {'values': [], 'avg': 0.0, 'median': 0.0, 'sum': 0.0}
                }
            
            # Collect values for each system in this scenario
            for system in self.systems:
                if system in scenario_systems:
                    for query, query_data in scenario_systems[system].items():
                        if query_data.get('status') == 'success':
                            execution_time, money_cost, quality = self.extract_metrics(query_data)
                            
                            # Only add finite values
                            if execution_time != float('inf'):
                                scenario_stats[scenario][system]['execution_time']['values'].append(execution_time)
                            if money_cost != float('inf'):
                                scenario_stats[scenario][system]['money_cost']['values'].append(money_cost)
                            if quality > 0:
                                scenario_stats[scenario][system]['quality']['values'].append(quality)
            
            # Calculate statistics for each system in this scenario
            for system in self.systems:
                for metric in ['execution_time', 'money_cost', 'quality']:
                    values = scenario_stats[scenario][system][metric]['values']
                    if values:
                        scenario_stats[scenario][system][metric]['avg'] = np.mean(values)
                        scenario_stats[scenario][system][metric]['median'] = np.median(values)
                        scenario_stats[scenario][system][metric]['sum'] = np.sum(values)
                    # Remove the raw values to keep the result clean
                    del scenario_stats[scenario][system][metric]['values']
        
        return scenario_stats
    
    def calculate_system_coverage_across_scenarios(self) -> Dict[str, Dict[str, int]]:
        """
        Calculate coverage for each system across all scenarios.
        Coverage = number of competitive queries system succeeded / total competitive queries
        Returns: {system: {competitive_queries: count, total_competitive: count}}
        """
        system_coverage = {}
        total_competitive_queries = 0
        
        for system in self.systems:
            system_coverage[system] = {'competitive_queries': 0}
        
        # Count competitive queries and system participation
        for scenario in self.scenarios:
            scenario_systems = self.data[scenario]
            if not scenario_systems:
                continue
                
            # Find all available queries across systems
            all_queries = set()
            for system_data in scenario_systems.values():
                all_queries.update(system_data.keys())
            
            # Check each query for competitiveness
            for query in all_queries:
                query_participants = []
                for system in self.systems:
                    if system in scenario_systems and query in scenario_systems[system]:
                        query_data = scenario_systems[system][query]
                        if query_data.get('status') == 'success':
                            query_participants.append(system)
                
                # Only count competitive queries (≥2 systems succeeded)
                if len(query_participants) >= 2:
                    total_competitive_queries += 1
                    for system in query_participants:
                        system_coverage[system]['competitive_queries'] += 1
        
        # Add total competitive queries count to each system
        for system in self.systems:
            system_coverage[system]['total_competitive'] = total_competitive_queries
        
        return system_coverage
    
    def calculate_system_coverage_by_operator_type(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        """
        Calculate coverage for each system by operator type.
        Returns: {operator_type: {system: {competitive_queries: count, total_competitive: count}}}
        """
        operator_coverage = {}
        
        for operator_type, operator_scenarios in self.operator_queries.items():
            operator_coverage[operator_type] = {}
            total_competitive_queries = 0
            
            for system in self.systems:
                operator_coverage[operator_type][system] = {'competitive_queries': 0}
            
            # Count competitive queries for this operator type
            for scenario, queries in operator_scenarios.items():
                if scenario not in self.data:
                    continue
                    
                scenario_systems = self.data[scenario]
                
                for query in queries:
                    query_participants = []
                    for system in self.systems:
                        if system in scenario_systems and query in scenario_systems[system]:
                            query_data = scenario_systems[system][query]
                            if query_data.get('status') == 'success':
                                query_participants.append(system)
                    
                    # Only count competitive queries
                    if len(query_participants) >= 2:
                        total_competitive_queries += 1
                        for system in query_participants:
                            operator_coverage[operator_type][system]['competitive_queries'] += 1
            
            # Add total competitive queries count to each system for this operator type
            for system in self.systems:
                operator_coverage[operator_type][system]['total_competitive'] = total_competitive_queries
        
        return operator_coverage
    
    def calculate_system_coverage_by_scenario(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        """
        Calculate coverage for each system by scenario.
        Returns: {scenario: {system: {competitive_queries: count, total_competitive: count}}}
        """
        scenario_coverage = {}
        
        for scenario in self.scenarios:
            scenario_coverage[scenario] = {}
            total_competitive_queries = 0
            scenario_systems = self.data[scenario]
            if not scenario_systems:
                continue
            
            for system in self.systems:
                scenario_coverage[scenario][system] = {'competitive_queries': 0}
            
            # Find all available queries across systems
            all_queries = set()
            for system_data in scenario_systems.values():
                all_queries.update(system_data.keys())
            
            # Check each query for competitiveness in this scenario
            for query in all_queries:
                query_participants = []
                for system in self.systems:
                    if system in scenario_systems and query in scenario_systems[system]:
                        query_data = scenario_systems[system][query]
                        if query_data.get('status') == 'success':
                            query_participants.append(system)
                
                # Only count competitive queries
                if len(query_participants) >= 2:
                    total_competitive_queries += 1
                    for system in query_participants:
                        scenario_coverage[scenario][system]['competitive_queries'] += 1
            
            # Add total competitive queries count to each system for this scenario
            for system in self.systems:
                scenario_coverage[scenario][system]['total_competitive'] = total_competitive_queries
        
        return scenario_coverage
    
    def find_winners_with_tolerance(self, tolerance_levels) -> Dict[str, Dict[float, Dict[str, float]]]:
        """
        Find winners across all scenarios with different tolerance levels.

        Args:
            tolerance_levels: Dict with metric-specific tolerance levels or List for all metrics

        Returns:
            {metric: {tolerance: {system: wins}}, 'query_counts': {metric: count}}
        """
        # Handle both dict (metric-specific) and list (same for all) inputs
        if isinstance(tolerance_levels, list):
            metric_tolerances = {
                'execution_time': tolerance_levels,
                'money_cost': tolerance_levels,
                'quality': tolerance_levels
            }
        else:
            metric_tolerances = tolerance_levels

        tolerance_results = {
            'execution_time': {},
            'money_cost': {},
            'quality': {}
        }

        # Track the number of competitive queries for each metric
        query_counts = {
            'execution_time': 0,
            'money_cost': 0,
            'quality': 0
        }

        # Initialize results for each metric and tolerance level
        for metric in ['execution_time', 'money_cost', 'quality']:
            for tolerance in metric_tolerances[metric]:
                tolerance_results[metric][tolerance] = defaultdict(float)
            
        # Go through all scenarios and queries for each metric separately
        for scenario in self.scenarios:
            scenario_systems = self.data[scenario]
            if not scenario_systems:
                continue
                
            # Find all available queries across systems
            all_queries = set()
            for system_data in scenario_systems.values():
                all_queries.update(system_data.keys())
            
            # Compare each query across systems
            for query in all_queries:
                query_metrics = {}  # system -> (execution_time, money_cost, quality)
                
                for system in self.systems:
                    if system in scenario_systems and query in scenario_systems[system]:
                        query_data = scenario_systems[system][query]
                        if query_data.get('status') == 'success':
                            query_metrics[system] = self.extract_metrics(query_data)
                
                if len(query_metrics) < 2:  # Need at least 2 systems to compare
                    continue

                # Process each metric with its specific tolerance levels

                # Execution time (lower is better, use relative tolerance)
                if query_metrics:
                    min_time = min(metrics[0] for metrics in query_metrics.values())
                    # Only count if min_time is not inf (at least one system has valid time)
                    if min_time != float('inf'):
                        query_counts['execution_time'] += 1
                        for tolerance in metric_tolerances['execution_time']:
                            time_winners = []
                            for sys, metrics in query_metrics.items():
                                if self._is_winner_with_tolerance(metrics[0], min_time, tolerance, 'lower', 'relative'):
                                    time_winners.append(sys)

                            for winner in time_winners:
                                tolerance_results['execution_time'][tolerance][winner] += 1

                # Money cost (lower is better, use relative tolerance)
                if query_metrics:
                    min_cost = min(metrics[1] for metrics in query_metrics.values())
                    # Only count if min_cost is not inf (at least one system has valid cost)
                    if min_cost != float('inf'):
                        query_counts['money_cost'] += 1
                        for tolerance in metric_tolerances['money_cost']:
                            cost_winners = []
                            for sys, metrics in query_metrics.items():
                                if self._is_winner_with_tolerance(metrics[1], min_cost, tolerance, 'lower', 'relative'):
                                    cost_winners.append(sys)

                            for winner in cost_winners:
                                tolerance_results['money_cost'][tolerance][winner] += 1

                # Quality (higher is better, use absolute tolerance)
                if query_metrics:
                    max_quality = max(metrics[2] for metrics in query_metrics.values())
                    # Only count if max_quality is > 0 (at least one system has valid quality)
                    if max_quality > 0:
                        query_counts['quality'] += 1
                        for tolerance in metric_tolerances['quality']:
                            quality_winners = []
                            for sys, metrics in query_metrics.items():
                                if self._is_winner_with_tolerance(metrics[2], max_quality, tolerance, 'higher', 'absolute'):
                                    quality_winners.append(sys)

                            for winner in quality_winners:
                                tolerance_results['quality'][tolerance][winner] += 1

        tolerance_results['query_counts'] = query_counts
        return tolerance_results
    
    def _is_winner_with_tolerance(self, value: float, best_value: float, tolerance: float, 
                                 direction: str, tolerance_type: str) -> bool:
        """
        Check if a value is considered a winner given the tolerance.
        
        Args:
            value: The value to check
            best_value: The best (optimal) value
            tolerance: The tolerance level
            direction: 'lower' for metrics where lower is better, 'higher' for metrics where higher is better
            tolerance_type: 'relative' for percentage-based tolerance, 'absolute' for absolute difference
        """
        if tolerance_type == 'relative':
            if direction == 'lower':
                # For lower-is-better metrics: value <= best_value * (1 + tolerance)
                return value <= best_value * (1 + tolerance)
            else:
                # For higher-is-better metrics: value >= best_value * (1 - tolerance)
                return value >= best_value * (1 - tolerance)
        else:  # absolute
            if direction == 'lower':
                # For lower-is-better metrics: value <= best_value + tolerance
                return value <= best_value + tolerance
            else:
                # For higher-is-better metrics: value >= best_value - tolerance
                return value >= best_value - tolerance
    
    def generate_tolerance_analysis(self, include_convergence_analysis: bool = False, auto_tolerance_levels: bool = False, output_suffix: str = ""):
        """Generate tolerance analysis with plots and markdown report"""
        print("Generating tolerance analysis...")
        
        # If convergence analysis is enabled, automatically set tolerance levels
        if include_convergence_analysis or auto_tolerance_levels:
            print("Finding convergence tolerances to set optimal tolerance levels...")
            convergence_tolerances = self.find_convergence_tolerance()
            
            # Create metric-specific auto tolerance levels
            if any(convergence_tolerances.values()):
                auto_tolerance_levels = {}
                
                for metric in ['execution_time', 'money_cost', 'quality']:
                    metric_convergences = list(convergence_tolerances.get(metric, {}).values())
                    
                    if metric_convergences:
                        max_convergence = max(metric_convergences)
                        end_tolerance = max_convergence * 1.2
                        
                        # Different step sizes and intelligent ranges for different metrics
                        if metric == 'quality':
                            step_size = max(0.01, end_tolerance/20)  # Finer steps for quality
                        else:
                            # For time/cost with extreme ranges, use logarithmic-like distribution
                            if end_tolerance > 10:
                                # Use fewer points but include key milestones
                                key_points = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0]
                                auto_levels = [p for p in key_points if p <= end_tolerance]
                                auto_levels.append(end_tolerance)  # Add the exact convergence point
                                auto_tolerance_levels[metric] = sorted(list(set(auto_levels)))
                                print(f"Auto-generated {metric} tolerance levels (logarithmic): {[round(x, 1) for x in auto_tolerance_levels[metric]]}")
                                continue
                            else:
                                step_size = max(0.1, end_tolerance/10)  # Normal steps for moderate ranges
                        
                        auto_levels = self.create_tolerance_range(start=0.0, stop=end_tolerance, step=step_size)
                        
                        # Add specific convergence points
                        for convergence_point in set(metric_convergences):
                            if convergence_point not in auto_levels:
                                auto_levels.append(convergence_point)
                        
                        auto_tolerance_levels[metric] = sorted(list(set(auto_levels)))
                        print(f"Auto-generated {metric} tolerance levels: {[round(x, 3) for x in auto_tolerance_levels[metric]]}")
                    else:
                        # Use default if no convergence found
                        auto_tolerance_levels[metric] = self.tolerance_levels[metric]
                        print(f"Using default {metric} tolerance levels (no convergence found)")
                
                self.tolerance_levels = auto_tolerance_levels
            
            if include_convergence_analysis:
                self._save_convergence_results(convergence_tolerances)
        
        print(f"Using tolerance levels: {self.tolerance_levels}")
        
        # Get tolerance results
        tolerance_results = self.find_winners_with_tolerance(self.tolerance_levels)
        
        # Create plots
        self._create_tolerance_plots(tolerance_results, self.tolerance_levels, output_suffix)
        
        # Generate markdown report
        self._generate_tolerance_markdown(tolerance_results, self.tolerance_levels, output_suffix)
        
        print("Tolerance analysis complete!")
    
    def run_tolerance_analysis(self, convergence_criterion: str = "first_system", max_points_per_metric: Dict[str, int] = None, output_suffix: str = ""):
        """
        Run tolerance analysis with configurable stopping criteria and point density control.
        
        Args:
            convergence_criterion: Either "first_system", "all_systems", or "custom"
            max_points_per_metric: Dict controlling max number of tolerance points per metric
                                  e.g., {'execution_time': 15, 'money_cost': 15, 'quality': 10}
            output_suffix: Suffix to add to output filenames to distinguish different analyses
        """
        print(f"Running tolerance analysis with '{convergence_criterion}' stopping criterion...")
        
        # Set default max points per metric if not provided
        if max_points_per_metric is None:
            max_points_per_metric = {
                'execution_time': 15,  # Good resolution for sensitive low tolerance range
                'money_cost': 15,      # Good resolution for sensitive low tolerance range
                'quality': 10          # Fewer points since quality range is smaller
            }
        
        if convergence_criterion in ["first_system", "all_systems"]:
            # Find convergence tolerances to set optimal ranges
            convergence_tolerances = self.find_convergence_tolerance()
            
            # Set tolerance ranges based on convergence criterion
            if any(convergence_tolerances.values()):
                tolerance_ranges = {}
                
                for metric in ['execution_time', 'money_cost', 'quality']:
                    metric_convergences = list(convergence_tolerances.get(metric, {}).values())
                    
                    if metric_convergences:
                        if convergence_criterion == "first_system":
                            max_tolerance = min(metric_convergences)  # Stop at first convergence
                            # For first_system, use minimal buffer to stop right at convergence point
                            end_tolerance = max_tolerance * 1.01  # Very small buffer (1%)
                        else:  # all_systems
                            max_tolerance = max(metric_convergences)  # Stop when all converge
                            end_tolerance = max_tolerance * 1.1  # Larger buffer for all_systems
                        
                        # Create appropriate range for each metric with controlled point density
                        max_points = max_points_per_metric.get(metric, 15)
                        
                        if metric == 'quality':
                            # For quality, create evenly distributed points
                            step_size = end_tolerance / (max_points - 1) if max_points > 1 else end_tolerance
                            tolerance_ranges[metric] = self.create_tolerance_range(0.0, end_tolerance, step_size)
                        else:
                            # For time/cost metrics, use intelligent distribution
                            if end_tolerance <= 1.0:
                                # Small convergence values: linear distribution
                                step_size = end_tolerance / (max_points - 1) if max_points > 1 else end_tolerance
                                tolerance_ranges[metric] = self.create_tolerance_range(0.0, end_tolerance, step_size)
                            else:
                                # Larger convergence values: hybrid approach with more density at low values
                                # Create a mix of fine-grained low values and coarser high values
                                low_points = max_points // 2  # Half points for low tolerance range
                                high_points = max_points - low_points
                                
                                # Fine-grained points for 0 to 2.0 (sensitive range)
                                low_range_max = min(2.0, end_tolerance * 0.3)
                                low_step = low_range_max / low_points if low_points > 0 else low_range_max
                                low_tolerances = self.create_tolerance_range(0.0, low_range_max, low_step)
                                
                                # Coarser points from low_range_max to end_tolerance
                                if end_tolerance > low_range_max and high_points > 0:
                                    high_step = (end_tolerance - low_range_max) / high_points
                                    high_tolerances = self.create_tolerance_range(low_range_max + high_step, end_tolerance, high_step)
                                    tolerance_ranges[metric] = sorted(list(set(low_tolerances + high_tolerances)))
                                else:
                                    tolerance_ranges[metric] = low_tolerances
                    else:
                        # Use default if no convergence found
                        tolerance_ranges[metric] = self.tolerance_levels[metric]
                
                self.tolerance_levels = tolerance_ranges
                print(f"Auto-set tolerance ranges based on {convergence_criterion} convergence")
                
                # Debug output to show what ranges were set
                for metric in ['execution_time', 'money_cost', 'quality']:
                    if metric in tolerance_ranges:
                        print(f"  {metric}: {[round(x, 3) for x in tolerance_ranges[metric][:10]]}{'...' if len(tolerance_ranges[metric]) > 10 else ''}")
                        if metric in convergence_tolerances:
                            convergences = {sys: round(tol, 3) for sys, tol in convergence_tolerances[metric].items()}
                            print(f"    Convergence points: {convergences}")
                            if convergence_criterion == "first_system":
                                first_convergence = min(convergence_tolerances[metric].values())
                                print(f"    First system convergence: {first_convergence:.3f}")
        else:
            # Use current tolerance levels for custom analysis
            print("Using current tolerance levels for custom analysis")
        
        # Generate tolerance analysis without auto-setting tolerance levels 
        # since we already set them based on convergence criterion
        self.generate_tolerance_analysis(include_convergence_analysis=False, auto_tolerance_levels=False, output_suffix=output_suffix)
    
    
    def _create_tolerance_plots(self, tolerance_results: Dict[str, Dict[float, Dict[str, float]]], 
                               tolerance_levels, output_suffix: str = ""):
        """Create publication-quality plots for ACM double-column format"""
        
        # Calculate upper bounds for each system
        upper_bounds = self.calculate_system_upper_bounds()
        print(f"System upper bounds: {upper_bounds}")
        
        # Find convergence tolerances (where each system reaches its upper bound)
        convergence_tolerances = self.find_convergence_tolerance()
        
        # Publication settings
        plt.rcParams.update({
            'font.size': 10,
            'font.family': 'serif',
            'axes.linewidth': 0.8,
            'axes.labelsize': 10,
            'axes.titlesize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'legend.frameon': False,
            'lines.linewidth': 1.5,
            'lines.markersize': 4,
            'grid.linewidth': 0.5,
            'grid.alpha': 0.3
        })
        
        # Professional color palette (colorblind-friendly)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        # Different dash patterns for upper bound lines to distinguish overlapping lines
        dash_styles = [(3, 1), (6, 2), (1, 1), (3, 1, 1, 1), (5, 2, 1, 2)]

        system_colors = {system: colors[i % len(colors)] for i, system in enumerate(self.systems)}
        system_markers = {system: markers[i % len(markers)] for i, system in enumerate(self.systems)}
        system_dashes = {system: dash_styles[i % len(dash_styles)] for i, system in enumerate(self.systems)}
        
        # Create figure optimized for double-column ACM format
        fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.5))  # Standard ACM double-column width
        metrics = ['execution_time', 'money_cost', 'quality']
        metric_titles = ['Execution Time', 'Monetary Cost', 'Quality']
        
        # Handle both dict (metric-specific) and list (same for all) tolerance levels
        if isinstance(tolerance_levels, list):
            metric_tolerance_levels = {
                'execution_time': tolerance_levels,
                'money_cost': tolerance_levels,
                'quality': tolerance_levels
            }
        else:
            metric_tolerance_levels = tolerance_levels
        
        # Format system names for publication
        system_name_mapping = {
            'bigquery': 'BigQuery',
            'lotus': 'LOTUS', 
            'palimpzest': 'Palimpzest',
            'thalamusdb': 'ThalamusDB',
            'snowflake': 'Snowflake'
        }
        
        for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
            ax = axes[i]
            
            # Get metric-specific tolerance levels and convert to percentages
            metric_tolerances = metric_tolerance_levels[metric]
            tolerance_percentages = [t * 100 for t in metric_tolerances]
            
            # Plot lines for each system with different markers
            for j, system in enumerate(self.systems):
                wins_at_tolerances = []
                for tolerance in metric_tolerances:
                    wins = tolerance_results[metric][tolerance].get(system, 0)
                    wins_at_tolerances.append(wins)
                
                system_label = system_name_mapping.get(system, system.capitalize())
                
                ax.plot(tolerance_percentages, wins_at_tolerances, 
                       marker=system_markers[system], 
                       label=system_label,
                       color=system_colors[system], 
                       linewidth=1.5,
                       markersize=4,
                       markeredgecolor='white',
                       markeredgewidth=0.5)
            
            # Add horizontal lines showing upper bounds for each system
            # Use different dash patterns to make overlapping lines distinguishable
            for system in self.systems:
                if system in upper_bounds:
                    ax.axhline(y=upper_bounds[system],
                              color=system_colors[system],
                              linestyle='--',
                              dashes=system_dashes[system],
                              alpha=0.7,
                              linewidth=1.2)
            
            # Add vertical lines showing convergence tolerances for each system
            # Use different dash patterns to distinguish overlapping convergence lines
            if metric in convergence_tolerances:
                for system in self.systems:
                    if system in convergence_tolerances[metric]:
                        convergence_pct = convergence_tolerances[metric][system] * 100
                        # Use dotted style with custom dash pattern for convergence lines
                        ax.axvline(x=convergence_pct,
                                  color=system_colors[system],
                                  linestyle=(0, system_dashes[system]),
                                  alpha=0.7,
                                  linewidth=1.2)
            
            # Formatting for publication quality
            ax.set_xlabel('Tolerance (%)', fontsize=10)
            if i == 0:  # Only leftmost plot gets y-label
                ax.set_ylabel('Number of Wins', fontsize=10)
            ax.set_title(title, fontsize=11, pad=10)
            ax.grid(True, alpha=0.3, linewidth=0.5)
            
            # Set x-axis to show key tolerance levels dynamically with improved spacing
            min_tolerance_pct = min(tolerance_percentages)
            max_tolerance_pct = max(tolerance_percentages)
            
            # Create adaptive tick marks with better overlap control
            range_pct = max_tolerance_pct - min_tolerance_pct
            
            # Determine optimal number of ticks based on range and figure size
            if range_pct <= 50:
                max_ticks = 8
            elif range_pct <= 200:
                max_ticks = 6
            elif range_pct <= 1000:
                max_ticks = 5
            else:
                max_ticks = 4
            
            # Simplified tick selection: only 3 ticks for execution_time and money_cost
            if metric in ['execution_time', 'money_cost']:
                # For execution time and cost: show only min, max, and a meaningful middle value
                if range_pct > 100:
                    # For large ranges, use: 0, reasonable_middle, max
                    if range_pct <= 500:
                        middle_value = max_tolerance_pct * 0.4  # 40% of max
                    elif range_pct <= 1500:
                        middle_value = max_tolerance_pct * 0.3  # 30% of max
                    else:
                        middle_value = max_tolerance_pct * 0.2  # 20% of max for very large ranges
                    
                    key_tolerances = [min_tolerance_pct, middle_value, max_tolerance_pct]
                else:
                    # For smaller ranges, use 3 evenly spaced points
                    key_tolerances = [min_tolerance_pct, max_tolerance_pct/2, max_tolerance_pct]
            else:
                # For quality metric, keep more granular ticks since range is smaller
                if range_pct <= 100:
                    num_ticks = min(6, len(tolerance_percentages))
                    key_tolerances = np.linspace(min_tolerance_pct, max_tolerance_pct, num_ticks)
                else:
                    key_tolerances = [min_tolerance_pct, max_tolerance_pct/2, max_tolerance_pct]
            
            # Ensure we don't exceed max_ticks and round appropriately
            key_tolerances = key_tolerances[:max_ticks]
            if max(key_tolerances) < 10:
                formatted_ticks = [round(t, 1) for t in key_tolerances]
            elif max(key_tolerances) < 100:
                formatted_ticks = [round(t, 0) for t in key_tolerances]
            else:
                formatted_ticks = [int(round(t, -1)) for t in key_tolerances]  # Round to nearest 10
            
            ax.set_xticks(formatted_ticks)
            ax.set_xlim(min_tolerance_pct - range_pct * 0.02, max_tolerance_pct + range_pct * 0.02)
            
            # Format y-axis to show integers only
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=5))
            
            # Add subtle background
            ax.set_facecolor('#fafafa')
            
            # Remove top and right spines for cleaner look
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        # Single legend for all subplots, positioned at the top
        # Only include main system lines in legend (exclude horizontal bound lines)
        handles, labels = axes[0].get_legend_handles_labels()
        main_handles = handles[:len(self.systems)]  # Only first N handles are main system lines
        main_labels = labels[:len(self.systems)]    # Only first N labels are main system labels
        fig.legend(main_handles, main_labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), 
                  ncol=4, frameon=False, fontsize=9)
        
        # Tight layout with padding for legend
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        
        # Save with high DPI for publication quality
        plot_filename = f"tolerance_analysis_plots{output_suffix}.png"
        plot_path = self.output_dir / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', format='png')
        
        # Also save as PDF for LaTeX inclusion
        pdf_filename = f"tolerance_analysis_plots{output_suffix}.pdf"
        pdf_path = self.output_dir / pdf_filename
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', format='pdf')
        
        plt.close()
        
        # Reset matplotlib parameters to defaults
        plt.rcdefaults()
        
        print(f"Tolerance plots saved to {plot_path} and {pdf_path}")
    
    def _generate_tolerance_markdown(self, tolerance_results: Dict[str, Dict[float, Dict[str, float]]], 
                                    tolerance_levels, output_suffix: str = ""):
        """Generate markdown report for tolerance analysis"""
        md_lines = []
        
        def add_line(text):
            md_lines.append(text)
        
        add_line("# System Performance Analysis with Tolerance")
        add_line("")
        add_line(f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        add_line(f"**Total Scenarios:** {len(self.scenarios)}")
        add_line(f"**Total Systems:** {len(self.systems)}")
        add_line("")
        
        add_line("## Overview")
        add_line("")
        add_line("This analysis examines how system performance rankings change when we allow for tolerance in measurements.")
        add_line("Sometimes systems may have very similar performance (e.g., 1.01s vs 1.04s latency), and strict comparisons")
        add_line("might not reflect practical differences.")
        add_line("")
        
        add_line("## Tolerance Semantics")
        add_line("")
        add_line("### Execution Time & Money Cost (Relative Tolerance)")
        add_line("")
        add_line("Uses **relative percentage tolerance** - tolerance is a fraction of the best value.")
        add_line("")
        add_line("**Formula**: A system wins if `value ≤ best_value × (1 + tolerance)`")
        add_line("")
        add_line("**Examples**:")
        add_line("- **Best execution time**: 2.0 seconds")
        add_line("- **Tolerance 0% (0.0)**: Only systems with exactly 2.0s win")
        add_line("- **Tolerance 10% (0.1)**: Systems with ≤ 2.2s win (2.0 × 1.1)")
        add_line("- **Tolerance 50% (0.5)**: Systems with ≤ 3.0s win (2.0 × 1.5)")
        add_line("- **Tolerance 100% (1.0)**: Systems with ≤ 4.0s win (2.0 × 2.0)")
        add_line("")
        add_line("**Why relative?** Performance differences are often more meaningful as percentages.")
        add_line("A 0.1s difference matters more when the best time is 0.5s vs. 10s.")
        add_line("")
        
        add_line("### Quality Metrics (Absolute Tolerance)")
        add_line("")
        add_line("Uses **absolute distance tolerance** - tolerance is a fixed amount.")
        add_line("")
        add_line("**Formula**: A system wins if `value ≥ best_value - tolerance`")
        add_line("")
        add_line("**Examples**:")
        add_line("- **Best quality score**: 0.85 (f1_score, transformed relative_error, or spearman_correlation)")
        add_line("- **Tolerance 0% (0.0)**: Only systems with exactly 0.85 win")
        add_line("- **Tolerance 10% (0.1)**: Systems with ≥ 0.75 win (0.85 - 0.1)")
        add_line("- **Tolerance 20% (0.2)**: Systems with ≥ 0.65 win (0.85 - 0.2)")
        add_line("- **Tolerance 100% (1.0)**: Systems with ≥ -0.15 win (effectively all systems)")
        add_line("")
        add_line("**Why absolute?** Quality metrics are bounded [0,1], so absolute differences")
        add_line("are more interpretable than relative percentages.")
        add_line("")
        
        add_line("## Tolerance Analysis Results")
        add_line("")
        plot_filename = f"tolerance_analysis_plots{output_suffix}.png"
        add_line(f"![Tolerance Analysis]({plot_filename})")
        add_line("")
        add_line("**Plot Legend:**")
        add_line("- **Solid lines with markers**: Actual wins at each tolerance level")
        add_line("- **Dashed horizontal lines**: Upper bound (maximum possible wins for each system)")
        add_line("- **Dotted vertical lines**: Convergence tolerance (minimum tolerance where system reaches upper bound)")
        add_line("")
        add_line("The horizontal dashed lines represent the theoretical maximum wins each system can achieve,")
        add_line("which equals the total number of queries that system successfully executed. The vertical dotted")
        add_line("lines show the minimum tolerance level where each system reaches its upper bound for that metric.")
        add_line("At infinite tolerance, each system's win count would converge to this upper bound.")
        add_line("")
        
        # Create summary tables for key tolerance levels
        # Use different key tolerances for each metric based on typical ranges
        metric_key_tolerances = {
            'execution_time': [0.0, 0.5, 1.0, 2.0, 5.0, 10.0],
            'money_cost': [0.0, 0.5, 1.0, 2.0, 5.0, 10.0], 
            'quality': [0.0, 0.1, 0.2, 0.5, 1.0]
        }
        
        # Handle both dict and list tolerance levels
        if isinstance(tolerance_levels, list):
            available_tolerances = {
                'execution_time': tolerance_levels,
                'money_cost': tolerance_levels,
                'quality': tolerance_levels
            }
        else:
            available_tolerances = tolerance_levels
        
        # Extract query counts from tolerance_results
        query_counts = tolerance_results.get('query_counts', {})

        # Create tables for each metric separately
        for metric in ['execution_time', 'money_cost', 'quality']:
            metric_name = metric.replace('_', ' ').title()
            add_line(f"### {metric_name} Tolerance Analysis")
            add_line("")

            # Get the total number of competitive queries for this metric
            total_queries = query_counts.get(metric, 0)

            for tolerance in metric_key_tolerances[metric]:
                if tolerance in available_tolerances[metric] and tolerance in tolerance_results[metric]:
                    tolerance_pct = f"{tolerance*100:.0f}%" if tolerance > 0 else "0% (Strict)"
                    add_line(f"**Tolerance Level: {tolerance_pct}**")
                    add_line("")

                    system_wins = tolerance_results[metric][tolerance]
                    if system_wins and total_queries > 0:
                        sorted_systems = sorted(system_wins.items(), key=lambda x: x[1], reverse=True)

                        add_line("| Rank | System | Wins | Win Rate |")
                        add_line("|------|--------|------|----------|")

                        for i, (system, wins) in enumerate(sorted_systems, 1):
                            if wins > 0:
                                win_rate = wins / total_queries * 100 if total_queries > 0 else 0
                                add_line(f"| {i} | **{system}** | {wins:.1f}/{total_queries:.0f} | {win_rate:.1f}% |")

                        add_line("")

            add_line("")
        
        add_line("## Key Insights")
        add_line("")
        add_line("- At **0% tolerance (strict comparison)**, only the exactly best-performing system wins each query")
        add_line("- As **tolerance increases**, more systems are considered winners for queries where performance is similar")
        add_line("- **Win counts should increase** (or stay constant) as tolerance increases - systems don't lose wins, only gain them")
        add_line("- **Convergence point**: At high tolerance levels, all systems become winners for most queries")
        add_line("- **Relative tolerance** for time/cost metrics accounts for proportional differences")
        add_line("- **Absolute tolerance** for quality metrics accounts for bounded 0-1 range")
        add_line("")
        
        add_line("## Notes")
        add_line("")
        add_line("- Win counts are whole numbers - each system gets 1 win per query where it meets the tolerance threshold")
        add_line("- Execution time: Lower is better")
        add_line("- Money cost: Lower is better") 
        add_line("- Quality: Higher is better (f1_score, transformed relative_error, or spearman_correlation)")
        if not self.include_snowflake:
            add_line("- The 'snowflake' system has been excluded from this analysis")
        else:
            add_line("- The 'snowflake' system is included in this analysis")
        add_line("")
        
        # Save markdown file
        md_filename = f"analysis_with_tolerance{output_suffix}.md"
        md_path = self.output_dir / md_filename
        with open(md_path, 'w') as f:
            f.write("\n".join(md_lines))
        
        print(f"Tolerance analysis report saved to {md_path}")
    
    def create_visualization_tables(self):
        """Create and print visualization tables for all analyses"""
        output_lines = []
        
        def log_and_print(text):
            print(text)
            output_lines.append(text)
        
        log_and_print("=" * 80)
        log_and_print("SYSTEM PERFORMANCE ANALYSIS")
        log_and_print("=" * 80)
        
        # 1. Winners by scenario
        log_and_print("\n1. WINNERS BY SCENARIO")
        log_and_print("-" * 40)
        scenario_results = self.find_winners_by_scenario()
        
        for metric in ['execution_time', 'money_cost', 'quality']:
            log_and_print(f"\n{metric.upper().replace('_', ' ')} WINNERS BY SCENARIO:")
            
            for scenario in self.scenarios:
                if scenario in scenario_results:
                    system_wins = scenario_results[scenario][metric]
                    if system_wins:
                        total_queries = sum(system_wins.values())
                        # Sort systems by wins (descending)
                        sorted_systems = sorted(system_wins.items(), key=lambda x: x[1], reverse=True)
                        
                        log_and_print(f"\n  {scenario.upper()}:")
                        for system, wins in sorted_systems:
                            if wins > 0:
                                log_and_print(f"    {system}: {wins:.1f}/{total_queries:.0f} queries")
        
        # 2. Winners across all scenarios
        log_and_print(f"\n\n2. WINNERS ACROSS ALL SCENARIOS")
        log_and_print("-" * 50)
        cross_results = self.find_winners_across_scenarios()
        
        for metric in ['execution_time', 'money_cost', 'quality']:
            log_and_print(f"\n{metric.upper().replace('_', ' ')} WINNERS ACROSS ALL SCENARIOS:")
            system_wins = cross_results[metric]
            if system_wins:
                total_queries = sum(system_wins.values())
                sorted_systems = sorted(system_wins.items(), key=lambda x: x[1], reverse=True)
                
                for system, wins in sorted_systems:
                    if wins > 0:
                        win_rate = wins / total_queries * 100 if total_queries > 0 else 0
                        log_and_print(f"  {system}: {wins:.1f}/{total_queries:.0f} queries ({win_rate:.1f}%)")
        
        # 3. Winners by operator type
        log_and_print(f"\n\n3. WINNERS BY OPERATOR TYPE")
        log_and_print("-" * 50)
        operator_results = self.find_winners_by_operator_type()
        
        for operator_type in sorted(self.operator_queries.keys()):
            if operator_type in operator_results:
                log_and_print(f"\n{operator_type}:")
                
                for metric in ['execution_time', 'money_cost', 'quality']:
                    system_wins = operator_results[operator_type][metric]
                    if system_wins:
                        total_queries = sum(system_wins.values())
                        sorted_systems = sorted(system_wins.items(), key=lambda x: x[1], reverse=True)
                        
                        log_and_print(f"  {metric.replace('_', ' ').title()}:")
                        for system, wins in sorted_systems:
                            if wins > 0:
                                win_rate = wins / total_queries * 100 if total_queries > 0 else 0
                                log_and_print(f"    {system}: {wins:.1f}/{total_queries:.0f} queries ({win_rate:.1f}%)")
        
        # 4. Summary statistics
        log_and_print(f"\n\n4. SUMMARY STATISTICS")
        log_and_print("-" * 30)
        log_and_print(f"Total scenarios analyzed: {len(self.scenarios)}")
        log_and_print(f"Total systems compared: {len(self.systems)}")
        
        # Count total queries
        total_queries = 0
        for scenario in self.data:
            for system in self.data[scenario]:
                total_queries += len(self.data[scenario][system])
        log_and_print(f"Total query results analyzed: {total_queries}")
        
        # Show query breakdown by scenario
        log_and_print(f"\nQuery distribution by scenario:")
        scenario_query_counts = {}
        for scenario in self.scenarios:
            scenario_total = 0
            if scenario in self.data:
                all_queries = set()
                for system_data in self.data[scenario].values():
                    all_queries.update(system_data.keys())
                scenario_total = len(all_queries)
            scenario_query_counts[scenario] = scenario_total
            log_and_print(f"  {scenario}: {scenario_total} queries")
        
        # Individual metric rankings only (no overall average)
        log_and_print(f"\n4. INDIVIDUAL METRIC RANKINGS")
        log_and_print("-" * 40)
        cross_results = self.find_winners_across_scenarios()
        
        for metric in ['execution_time', 'money_cost', 'quality']:
            log_and_print(f"\n{metric.upper().replace('_', ' ')} RANKINGS:")
            system_wins = cross_results[metric]
            if system_wins:
                total_queries = sum(system_wins.values())
                sorted_systems = sorted(system_wins.items(), key=lambda x: x[1], reverse=True)
                
                for i, (system, wins) in enumerate(sorted_systems, 1):
                    if wins > 0:
                        win_rate = wins / total_queries * 100 if total_queries > 0 else 0
                        log_and_print(f"  {i}. {system}: {wins:.1f}/{total_queries:.0f} queries ({win_rate:.1f}%)")
        
        # Save results to files  
        self._save_results(output_lines, scenario_results, cross_results, operator_results)
        self._save_markdown_summary(scenario_results, cross_results, operator_results)
    
    def _save_results(self, output_lines, scenario_results, cross_results, operator_results):
        """Save analysis results to files"""
        # Save main output
        with open(self.output_dir / "analysis_summary.txt", 'w') as f:
            f.write("\n".join(output_lines))
        
        # Save detailed results as JSON
        detailed_results = {
            'scenario_results': dict(scenario_results),
            'cross_scenario_results': dict(cross_results),
            'operator_type_results': dict(operator_results),
            'metadata': {
                'scenarios': self.scenarios,
                'systems': self.systems,
                'total_scenarios': len(self.scenarios),
                'total_systems': len(self.systems)
            }
        }
        
        with open(self.output_dir / "detailed_results.json", 'w') as f:
            json.dump(detailed_results, f, indent=2, default=float)
        
        # Save CSV tables for easy analysis
        self._save_csv_tables(scenario_results, cross_results, operator_results)
        
        print(f"\nResults saved to {self.output_dir}/")
        print(f"  - analysis_summary.txt: Main output (text)")
        print(f"  - analysis_summary.md: Main output (markdown)")
        print(f"  - detailed_results.json: Detailed JSON data")
        print(f"  - scenario_winners.csv: Winners by scenario")
        print(f"  - cross_scenario_winners.csv: Winners across scenarios")
        print(f"  - operator_winners.csv: Winners by operator type")
    
    def _save_csv_tables(self, scenario_results, cross_results, operator_results):
        """Save results as CSV tables"""
        # Scenario winners
        scenario_data = []
        for scenario in self.scenarios:
            if scenario in scenario_results:
                for metric in ['execution_time', 'money_cost', 'quality']:
                    system_wins = scenario_results[scenario][metric]
                    if system_wins:
                        winner = max(system_wins, key=system_wins.get)
                        wins = system_wins[winner]
                        total = sum(system_wins.values())
                        scenario_data.append({
                            'scenario': scenario,
                            'metric': metric,
                            'winner': winner,
                            'wins': wins,
                            'total_queries': total,
                            'win_rate': wins/total*100 if total > 0 else 0
                        })
        
        pd.DataFrame(scenario_data).to_csv(self.output_dir / "scenario_winners.csv", index=False)
        
        # Cross-scenario winners
        cross_data = []
        for metric in ['execution_time', 'money_cost', 'quality']:
            system_wins = cross_results[metric]
            total_queries = sum(system_wins.values())
            for system, wins in system_wins.items():
                cross_data.append({
                    'metric': metric,
                    'system': system,
                    'wins': wins,
                    'win_rate': wins/total_queries*100 if total_queries > 0 else 0
                })
        
        pd.DataFrame(cross_data).to_csv(self.output_dir / "cross_scenario_winners.csv", index=False)
        
        # Operator type winners
        operator_data = []
        for operator_type in operator_results:
            for metric in ['execution_time', 'money_cost', 'quality']:
                system_wins = operator_results[operator_type][metric]
                if system_wins:
                    total_queries = sum(system_wins.values())
                    winner = max(system_wins, key=system_wins.get)
                    winner_score = system_wins[winner]
                    operator_data.append({
                        'operator_type': operator_type,
                        'metric': metric,
                        'winner': winner,
                        'wins': winner_score,
                        'total_queries': total_queries,
                        'win_rate': winner_score/total_queries*100 if total_queries > 0 else 0
                    })
        
        pd.DataFrame(operator_data).to_csv(self.output_dir / "operator_winners.csv", index=False)
    
    def _save_markdown_summary(self, scenario_results, cross_results, operator_results):
        """Save analysis results as markdown file with consistent table format"""
        md_lines = []
        
        def add_line(text):
            md_lines.append(text)
        
        # Calculate statistics and coverage for all groupings
        cross_scenario_stats = self.calculate_system_statistics_across_scenarios()
        operator_stats = self.calculate_system_statistics_by_operator_type()
        scenario_stats = self.calculate_system_statistics_by_scenario()
        cross_scenario_coverage = self.calculate_system_coverage_across_scenarios()
        operator_coverage = self.calculate_system_coverage_by_operator_type()
        scenario_coverage = self.calculate_system_coverage_by_scenario()
        
        add_line("# System Performance Analysis")
        add_line("")
        add_line(f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        add_line(f"**Total Scenarios:** {len(self.scenarios)}")
        add_line(f"**Total Systems:** {len(self.systems)}")
        
        # Count total queries and by scenario
        total_queries = 0
        scenario_query_counts = {}
        
        for scenario in self.scenarios:
            scenario_total = 0
            if scenario in self.data:
                # Get unique queries across all systems for this scenario
                all_queries = set()
                for system_data in self.data[scenario].values():
                    all_queries.update(system_data.keys())
                scenario_total = len(all_queries)
            scenario_query_counts[scenario] = scenario_total
            total_queries += scenario_total
        
        add_line(f"**Total Queries:** {total_queries}")
        add_line("")
        
        # Analysis Summary with bullet points
        add_line("## Analysis Summary")
        add_line("")
        
        # Cross-scenario summary with nested bullet points
        add_line("- **Across All Scenarios**")
        for metric in ['execution_time', 'money_cost', 'quality']:
            metric_name = metric.replace('_', ' ').title()
            system_wins = cross_results[metric]
            if system_wins:
                total_metric_queries = sum(system_wins.values())
                add_line(f"  - **{metric_name}**:")
                
                # Add sub-bullets for top 3 systems at the same level
                sorted_systems = sorted(system_wins.items(), key=lambda x: x[1], reverse=True)[:3]
                for i, (system, wins) in enumerate(sorted_systems):
                    if wins > 0:
                        rate = wins / total_metric_queries * 100 if total_metric_queries > 0 else 0
                        if i == 0:
                            add_line(f"    - 1st: **{system}** ({wins:.1f}/{total_metric_queries:.0f} = {rate:.1f}%)")
                        elif i == 1:
                            add_line(f"    - 2nd: **{system}** ({wins:.1f}/{total_metric_queries:.0f} = {rate:.1f}%)")
                        elif i == 2:
                            add_line(f"    - 3rd: **{system}** ({wins:.1f}/{total_metric_queries:.0f} = {rate:.1f}%)")
        
        add_line("")
        
        # Operator type summary with nested bullet points
        add_line("- **By Operator Type**")
        operator_order = ['SEM_FILTER', 'SEM_JOIN', 'SEM_MAP', 'SEM_SCORE', 'SEM_CLASSIFY']
        for operator_type in operator_order:
            if operator_type in operator_results:
                add_line(f"  - **{operator_type}**:")
                for metric in ['execution_time', 'money_cost', 'quality']:
                    metric_name = metric.replace('_', ' ').title()
                    system_wins = operator_results[operator_type][metric]
                    if system_wins:
                        total_op_queries = sum(system_wins.values())
                        add_line(f"    - **{metric_name}**:")
                        
                        # Add sub-bullets for top 3 systems
                        sorted_systems = sorted(system_wins.items(), key=lambda x: x[1], reverse=True)[:3]
                        for i, (system, wins) in enumerate(sorted_systems):
                            if wins > 0:
                                win_rate = wins / total_op_queries * 100 if total_op_queries > 0 else 0
                                if i == 0:
                                    add_line(f"      - 1st: **{system}** ({wins:.1f}/{total_op_queries:.0f} = {win_rate:.1f}%)")
                                elif i == 1:
                                    add_line(f"      - 2nd: **{system}** ({wins:.1f}/{total_op_queries:.0f} = {win_rate:.1f}%)")
                                elif i == 2:
                                    add_line(f"      - 3rd: **{system}** ({wins:.1f}/{total_op_queries:.0f} = {win_rate:.1f}%)")
        
        add_line("")
        
        add_line("---")
        add_line("")
        add_line("## Notes")
        add_line("")
        add_line("- **Win counts** can be fractional when systems tie on a query (e.g., if 2 systems tie, each gets 0.5 wins)")
        add_line("- **Execution time**: Lower is better (faster systems win)")
        add_line("- **Money cost**: Lower is better (cheaper systems win)")
        add_line("- **Quality**: Higher is better (uses f1_score, transformed relative_error, or spearman_correlation)")
        add_line(f"- **Total comparisons**: {total_queries} queries across {len(self.scenarios)} scenarios")
        add_line("")
        
        
        # 1. Cross-scenario results - consistent table format with statistics
        add_line("## 1. Winners Across All Scenarios")
        add_line("")
        
        for metric in ['execution_time', 'money_cost', 'quality']:
            metric_name = metric.replace('_', ' ').title()
            add_line(f"### {metric_name}")
            add_line("")
            
            system_wins = cross_results[metric]
            if system_wins:
                total_queries = sum(system_wins.values())
                
                # Get all systems with coverage > 0 (including those with 0 wins)
                systems_with_coverage = []
                for system in self.systems:
                    coverage = cross_scenario_coverage.get(system, {})
                    competitive_queries = coverage.get('competitive_queries', 0)
                    if competitive_queries > 0:  # Only include systems with coverage
                        wins = system_wins.get(system, 0.0)  # Get wins, default to 0
                        systems_with_coverage.append((system, wins))
                
                # Sort by wins (descending)
                sorted_systems = sorted(systems_with_coverage, key=lambda x: x[1], reverse=True)
                
                # Extended table with coverage and statistics for this specific metric
                add_line("| Rank | System | Coverage | Wins | Win Rate | Average | Median | Sum |")
                add_line("|------|--------|----------|------|----------|---------|--------|-----|")
                
                for i, (system, wins) in enumerate(sorted_systems, 1):
                    win_rate = wins / total_queries * 100 if total_queries > 0 else 0
                    
                    # Get coverage for this system
                    coverage = cross_scenario_coverage.get(system, {})
                    competitive_queries = coverage.get('competitive_queries', 0)
                    total_competitive = coverage.get('total_competitive', 0)
                    coverage_str = f"{competitive_queries}/{total_competitive}"
                    
                    # Get statistics for this system and metric
                    stats = cross_scenario_stats.get(system, {}).get(metric, {})
                    avg_val = stats.get('avg', 0.0)
                    med_val = stats.get('median', 0.0)
                    sum_val = stats.get('sum', 0.0)
                    
                    # Format values based on metric type
                    if metric == 'quality':
                        avg_str = f"{avg_val:.3f}"
                        med_str = f"{med_val:.3f}"
                        sum_str = f"{sum_val:.2f}"
                    else:  # execution_time, money_cost
                        avg_str = f"{avg_val:.2f}"
                        med_str = f"{med_val:.2f}"
                        sum_str = f"{sum_val:.1f}"
                    
                    add_line(f"| {i} | **{system}** | {coverage_str} | {wins:.1f}/{total_queries:.0f} | {win_rate:.1f}% | {avg_str} | {med_str} | {sum_str} |")
                
                add_line("")
        
        # 2. Operator type results - consistent table format in specific order
        add_line("## 2. Winners by Operator Type")
        add_line("")
        
        operator_order = ['SEM_FILTER', 'SEM_JOIN', 'SEM_MAP', 'SEM_SCORE', 'SEM_CLASSIFY']
        for operator_type in operator_order:
            if operator_type in operator_results:
                add_line(f"### {operator_type}")
                add_line("")
                
                for metric in ['execution_time', 'money_cost', 'quality']:
                    metric_name = metric.replace('_', ' ').title()
                    system_wins = operator_results[operator_type][metric]
                    if system_wins:
                        total_queries = sum(system_wins.values())
                        
                        # Get all systems with coverage > 0 for this operator type (including those with 0 wins)
                        systems_with_coverage = []
                        for system in self.systems:
                            coverage = operator_coverage.get(operator_type, {}).get(system, {})
                            competitive_queries = coverage.get('competitive_queries', 0)
                            if competitive_queries > 0:  # Only include systems with coverage
                                wins = system_wins.get(system, 0.0)  # Get wins, default to 0
                                systems_with_coverage.append((system, wins))
                        
                        # Sort by wins (descending)
                        sorted_systems = sorted(systems_with_coverage, key=lambda x: x[1], reverse=True)
                        
                        add_line(f"**{metric_name}** (Total: {total_queries:.0f} queries):")
                        add_line("")
                        add_line("| Rank | System | Coverage | Wins | Win Rate | Average | Median | Sum |")
                        add_line("|------|--------|----------|------|----------|---------|--------|-----|")
                        
                        for i, (system, wins) in enumerate(sorted_systems, 1):
                            win_rate = wins / total_queries * 100 if total_queries > 0 else 0
                            
                            # Get coverage for this system and operator type
                            coverage = operator_coverage.get(operator_type, {}).get(system, {})
                            competitive_queries = coverage.get('competitive_queries', 0)
                            total_competitive = coverage.get('total_competitive', 0)
                            coverage_str = f"{competitive_queries}/{total_competitive}"
                            
                            # Get statistics for this system, operator type, and metric
                            stats = operator_stats.get(operator_type, {}).get(system, {}).get(metric, {})
                            avg_val = stats.get('avg', 0.0)
                            med_val = stats.get('median', 0.0)
                            sum_val = stats.get('sum', 0.0)
                            
                            # Format values based on metric type
                            if metric == 'quality':
                                avg_str = f"{avg_val:.3f}"
                                med_str = f"{med_val:.3f}"
                                sum_str = f"{sum_val:.2f}"
                            else:  # execution_time, money_cost
                                avg_str = f"{avg_val:.2f}"
                                med_str = f"{med_val:.2f}"
                                sum_str = f"{sum_val:.1f}"
                            
                            add_line(f"| {i} | **{system}** | {coverage_str} | {wins:.1f}/{total_queries:.0f} | {win_rate:.1f}% | {avg_str} | {med_str} | {sum_str} |")
                        
                        add_line("")
                
                add_line("")
        
        # 3. Winners by scenario - consistent table format
        add_line("## 3. Winners by Scenario")
        add_line("")
        
        for scenario in self.scenarios:
            if scenario in scenario_results:
                add_line(f"### {scenario.upper()}")
                add_line("")
                
                for metric in ['execution_time', 'money_cost', 'quality']:
                    metric_name = metric.replace('_', ' ').title()
                    system_wins = scenario_results[scenario][metric]
                    if system_wins:
                        total_queries = sum(system_wins.values())
                        
                        # Get all systems with coverage > 0 for this scenario (including those with 0 wins)
                        systems_with_coverage = []
                        for system in self.systems:
                            coverage = scenario_coverage.get(scenario, {}).get(system, {})
                            competitive_queries = coverage.get('competitive_queries', 0)
                            if competitive_queries > 0:  # Only include systems with coverage
                                wins = system_wins.get(system, 0.0)  # Get wins, default to 0
                                systems_with_coverage.append((system, wins))
                        
                        # Sort by wins (descending)
                        sorted_systems = sorted(systems_with_coverage, key=lambda x: x[1], reverse=True)
                        
                        add_line(f"**{metric_name}** (Total: {total_queries:.0f} queries):")
                        add_line("")
                        add_line("| Rank | System | Coverage | Wins | Win Rate | Average | Median | Sum |")
                        add_line("|------|--------|----------|------|----------|---------|--------|-----|")
                        
                        for i, (system, wins) in enumerate(sorted_systems, 1):
                            win_rate = wins / total_queries * 100 if total_queries > 0 else 0
                            
                            # Get coverage for this system and scenario
                            coverage = scenario_coverage.get(scenario, {}).get(system, {})
                            competitive_queries = coverage.get('competitive_queries', 0)
                            total_competitive = coverage.get('total_competitive', 0)
                            coverage_str = f"{competitive_queries}/{total_competitive}"
                            
                            # Get statistics for this system, scenario, and metric
                            stats = scenario_stats.get(scenario, {}).get(system, {}).get(metric, {})
                            avg_val = stats.get('avg', 0.0)
                            med_val = stats.get('median', 0.0)
                            sum_val = stats.get('sum', 0.0)
                            
                            # Format values based on metric type
                            if metric == 'quality':
                                avg_str = f"{avg_val:.3f}"
                                med_str = f"{med_val:.3f}"
                                sum_str = f"{sum_val:.2f}"
                            else:  # execution_time, money_cost
                                avg_str = f"{avg_val:.2f}"
                                med_str = f"{med_val:.2f}"
                                sum_str = f"{sum_val:.1f}"
                            
                            add_line(f"| {i} | **{system}** | {coverage_str} | {wins:.1f}/{total_queries:.0f} | {win_rate:.1f}% | {avg_str} | {med_str} | {sum_str} |")
                        
                        add_line("")
                
                add_line("")
        
        # Save markdown file
        with open(self.output_dir / "analysis_summary.md", 'w') as f:
            f.write("\n".join(md_lines))

def main():
    """Main analysis function"""
    print("Starting system performance analysis...")
    
    # Initialize analyzer
    analyzer = SystemAnalyzer()
    
    # Step 1: Scan available data
    analyzer.scan_available_data()
    
    # Step 2: Load all data
    analyzer.load_data()
    
    # Step 3: Create visualization tables (original analysis)
    analyzer.create_visualization_tables()
    
    # Control number of points plotted per metric for cleaner visualizations
    max_points_config = {
        'execution_time': 15,  # Good resolution with focus on low tolerance sensitivity
        'money_cost': 15,      # Good resolution with focus on low tolerance sensitivity  
        'quality': 10          # Fewer points since quality has smaller, bounded range
    }
    
    # Step 4: Run three tolerance analyses with different criteria
    
    # Analysis 1: First system convergence
    print("\n" + "="*60)
    print("ANALYSIS 1: FIRST SYSTEM CONVERGENCE")
    print("="*60)
    analyzer.run_tolerance_analysis(
        convergence_criterion="first_system", 
        max_points_per_metric=max_points_config,
        output_suffix="_first_system"
    )
    
    # Analysis 2: All systems convergence
    print("\n" + "="*60)
    print("ANALYSIS 2: ALL SYSTEMS CONVERGENCE")
    print("="*60)
    analyzer.run_tolerance_analysis(
        convergence_criterion="all_systems", 
        max_points_per_metric=max_points_config,
        output_suffix="_all_systems"
    )
    
    # Analysis 3: Custom tolerance ranges
    print("\n" + "="*60)
    print("ANALYSIS 3: CUSTOM TOLERANCE RANGES")
    print("="*60)
    
    # Set custom tolerance ranges as specified: 0-1050 for execution_time, 0-378 for money_cost, 0-100 for quality
    # Use fewer points to prevent overcrowding in plots
    custom_tolerance_levels = {
        'execution_time': SystemAnalyzer.create_tolerance_range(start=0.0, stop=3.0, step=0.2),   
        'money_cost': SystemAnalyzer.create_tolerance_range(start=0.0, stop=3.0, step=0.2),      
        'quality': SystemAnalyzer.create_tolerance_range(start=0.0, stop=1.0, step=0.05)          
    }
    
    analyzer.set_tolerance_levels(custom_tolerance_levels)
    analyzer.run_tolerance_analysis(
        convergence_criterion="custom", 
        max_points_per_metric=max_points_config,
        output_suffix="_custom_ranges"
    )
    
    print("\n" + "="*60)
    print("ALL ANALYSES COMPLETE!")
    print("="*60)
    print("Generated files:")
    print("  First System Analysis:")
    print("    - tolerance_analysis_plots_first_system.png/pdf")
    print("    - analysis_with_tolerance_first_system.md")
    print("  All Systems Analysis:")
    print("    - tolerance_analysis_plots_all_systems.png/pdf")
    print("    - analysis_with_tolerance_all_systems.md")
    print("  Custom Ranges Analysis:")
    print("    - tolerance_analysis_plots_custom_ranges.png/pdf")
    print("    - analysis_with_tolerance_custom_ranges.md")

if __name__ == "__main__":
    main()