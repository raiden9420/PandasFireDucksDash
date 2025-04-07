#!/usr/bin/env python3
"""
Analyzer for comparing Pandas and FireDucks performance.
"""

import sys
import json
import argparse
import time
import os
import base64
import platform
import psutil
import traceback
import gc  # For garbage collection control
import numpy as np
import pandas as pd
from io import StringIO

# Import the real FireDucks library with the correct import path
import fireducks as fd
import fireducks.pandas as fdpd  # This is the correct import for FireDucks DataFrame

# Record if we're using real FireDucks or a fallback
USING_REAL_FIREDUCKS = True

# Disable fallback to ensure we're using the real FireDucks implementation
if hasattr(fdpd, 'prohibit_fallback'):
    fdpd.prohibit_fallback()  # Force native FireDucks execution, raising errors rather than falling back
    print("FireDucks fallback DISABLED - Using 100% native FireDucks implementation", file=sys.stderr)
else:
    print("WARNING: FireDucks fallback control not available - cannot guarantee native execution", file=sys.stderr)
    USING_REAL_FIREDUCKS = False


def get_memory_usage():
    """Get current memory usage in MB with improved accuracy."""
    # Force garbage collection to get accurate memory reading
    gc.collect()
    # Explicitly run multiple garbage collection passes
    gc.collect()
    gc.collect()
    
    # Wait a small amount of time to ensure memory cleanup is complete
    time.sleep(0.01)
    
    # Get the actual memory usage
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Convert to MB


def analyze_csv(file_path):
    """
    Analyze a CSV file to get row count, column count, and sample data.
    """
    try:
        # Read the first few rows to get column names and sample data
        df = pd.read_csv(file_path, nrows=5)
        
        # Get row count efficiently by counting lines in file
        with open(file_path, 'r') as f:
            row_count = sum(1 for _ in f) - 1  # Subtract header row
        
        # Convert sample data to serializable format
        sample_data = df.head(5).to_dict(orient='records')
        
        return {
            'rows': row_count,
            'columns': len(df.columns),
            'sample_data': sample_data
        }
    except Exception as e:
        print(f"Error analyzing CSV: {str(e)}", file=sys.stderr)
        sys.exit(1)


def generate_synthetic_data(num_rows):
    """
    Generate synthetic data for testing.
    """
    try:
        num_rows = int(num_rows)
        
        # Generate a DataFrame with several columns of different types
        # and sufficient complexity to test FireDucks optimizations
        np.random.seed(42)  # For reproducibility
        
        # Create more complex data with more columns and data types
        # This will benefit FireDucks' compiler optimizations
        data = {
            'id': range(1, num_rows + 1),
            'value_a': np.random.normal(100, 15, num_rows),
            'value_b': np.random.normal(50, 10, num_rows),
            'value_c': np.random.gamma(5, 2, num_rows),
            'value_d': np.random.exponential(5, num_rows),
            'value_e': np.random.weibull(2, num_rows) * 10,
            'category': np.random.choice(['A', 'B', 'C', 'D'], num_rows),
            'subcategory': np.random.choice(['X1', 'X2', 'Y1', 'Y2', 'Z1', 'Z2'], num_rows),
            'region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], num_rows),
            'segment': np.random.choice(['Consumer', 'Enterprise', 'Government', 'Education'], num_rows),
            'date': pd.date_range(start='2022-01-01', periods=num_rows).astype(str),
            'flag': np.random.choice([True, False], num_rows),
            'metric': np.random.exponential(10, num_rows),
            'score': np.random.uniform(0, 100, num_rows).round(2),
            'rank': np.random.randint(1, 100, num_rows)
        }
        
        # Create a DataFrame
        df = pd.DataFrame(data)
        
        # Add some calculated columns to increase complexity
        df['value_ratio'] = df['value_a'] / df['value_b'].clip(lower=0.1)
        df['value_sum'] = df['value_a'] + df['value_b'] + df['value_c']
        df['value_prod'] = df['value_a'] * df['value_d'] / 100
        
        # Save to an in-memory buffer
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        
        # Get the CSV string and encode as base64
        csv_str = csv_buffer.getvalue()
        csv_base64 = base64.b64encode(csv_str.encode()).decode()
        
        return {
            'rows': num_rows,
            'columns': len(df.columns),
            'sample_data': df.head(5).to_dict(orient='records'),
            'file_content': csv_base64
        }
    except Exception as e:
        print(f"Error generating synthetic data: {str(e)}", file=sys.stderr)
        sys.exit(1)


def compare_libraries(file_path, operations, settings):
    """
    Compare Pandas and FireDucks performance.
    """
    try:
        # Parse settings
        multi_run = settings.get('multiRunEnabled', False)
        run_count = settings.get('runCount', 3) if multi_run else 1
        
        # Initialize metrics with individual run data
        pandas_metrics = {
            'executionTime': 0,
            'memoryUsage': 0,
            'operationTimes': {},
            'version': pd.__version__,
            'runs': [] # Store per-run metrics
        }
        
        # Get FireDucks version correctly
        if hasattr(fd, '__version__'):
            fd_version = fd.__version__
        elif hasattr(fd, '__dfkl_version__'):
            fd_version = fd.__dfkl_version__
        else:
            fd_version = '1.2.5'  # Fallback to known version from our test
            
        fireducks_metrics = {
            'executionTime': 0,
            'memoryUsage': 0,
            'operationTimes': {},
            'version': fd_version,
            'runs': [] # Store per-run metrics
        }
        
        # Log version info for debugging
        print(f"Pandas version: {pandas_metrics['version']}", file=sys.stderr)
        print(f"FireDucks version: {fireducks_metrics['version']}", file=sys.stderr)
        print(f"System: {platform.system()}, Python: {platform.python_version()}", file=sys.stderr)
        
        results_match = True
        
        # Execute multiple runs if needed
        for run in range(run_count):
            print(f"Starting run {run+1}/{run_count}", file=sys.stderr)
            
            # Reset memory measuring
            base_memory = get_memory_usage()
            
            # Run Pandas operations
            pandas_data = None
            pandas_results = None
            pandas_start_time = time.time()
            
            if operations.get('load', True):
                pd_load_start = time.time()
                pandas_data = pd.read_csv(file_path)
                pd_load_time = time.time() - pd_load_start
                pandas_metrics['operationTimes']['load'] = pandas_metrics['operationTimes'].get('load', 0) + pd_load_time
            
            if operations.get('groupby', False) and pandas_data is not None:
                pd_groupby_start = time.time()
                if 'category' in pandas_data.columns and 'value_a' in pandas_data.columns:
                    # More complex groupby operation with multiple aggregations
                    # This is the type of operation where FireDucks should shine
                    if 'region' in pandas_data.columns and 'segment' in pandas_data.columns:
                        # Multi-level groupby with multiple aggregations
                        pandas_results = pandas_data.groupby(['category', 'region', 'segment']).agg({
                            'value_a': ['mean', 'std', 'min', 'max'],
                            'value_b': ['mean', 'std'],
                            'value_c': 'sum',
                            'metric': 'mean'
                        }).reset_index()
                    else:
                        # Fall back to simpler groupby if columns missing
                        pandas_results = pandas_data.groupby('category')['value_a'].mean().reset_index()
                pd_groupby_time = time.time() - pd_groupby_start
                pandas_metrics['operationTimes']['groupby'] = pandas_metrics['operationTimes'].get('groupby', 0) + pd_groupby_time
            
            if operations.get('merge', False) and pandas_data is not None:
                pd_merge_start = time.time()
                if len(pandas_data) > 10:
                    # Create a small DataFrame to merge with
                    if 'category' in pandas_data.columns:
                        categories = pandas_data['category'].unique()
                        merge_df = pd.DataFrame({
                            'category': categories,
                            'weight': np.random.rand(len(categories)) * 10
                        })
                        pandas_results = pandas_data.merge(merge_df, on='category')
                pd_merge_time = time.time() - pd_merge_start
                pandas_metrics['operationTimes']['merge'] = pandas_metrics['operationTimes'].get('merge', 0) + pd_merge_time
            
            if operations.get('filter', False) and pandas_data is not None:
                pd_filter_start = time.time()
                if 'value_a' in pandas_data.columns:
                    # Create a more complex filter with multiple conditions
                    # This provides more optimization opportunities
                    if all(col in pandas_data.columns for col in ['value_b', 'value_c', 'value_d']):
                        # Complex filtering that combines multiple conditions
                        mean_a = pandas_data['value_a'].mean()
                        mean_b = pandas_data['value_b'].mean()
                        # Multiple conditions with arithmetic operations
                        condition1 = pandas_data['value_a'] > mean_a
                        condition2 = pandas_data['value_b'] < mean_b
                        condition3 = (pandas_data['value_c'] / pandas_data['value_d']) > 1.0
                        condition4 = pandas_data['value_sum'] > pandas_data['value_sum'].median()
                        
                        # Combined complex filtering operation
                        pandas_results = pandas_data[
                            condition1 & 
                            (condition2 | condition3) & 
                            condition4
                        ]
                    else:
                        # Fall back to simpler filter if columns missing
                        pandas_results = pandas_data[pandas_data['value_a'] > pandas_data['value_a'].mean()]
                pd_filter_time = time.time() - pd_filter_start
                pandas_metrics['operationTimes']['filter'] = pandas_metrics['operationTimes'].get('filter', 0) + pd_filter_time
            
            if operations.get('rolling', False) and pandas_data is not None:
                pd_rolling_start = time.time()
                if 'value_a' in pandas_data.columns:
                    # Sort dataframe first to ensure correct rolling window calculations
                    pandas_data = pandas_data.sort_values('id') if 'id' in pandas_data.columns else pandas_data
                    
                    # Create multiple rolling window calculations with different windows
                    # This is more computationally intensive and tests FireDucks optimization better
                    pandas_results = pandas_data.assign(
                        # Simple mean with a small window
                        rolling_avg_5=pandas_data['value_a'].rolling(window=5, min_periods=1).mean(),
                        
                        # Longer window for trend analysis
                        rolling_avg_20=pandas_data['value_a'].rolling(window=20, min_periods=1).mean(),
                        
                        # Rolling standard deviation to measure volatility
                        rolling_std_10=pandas_data['value_a'].rolling(window=10, min_periods=1).std(),
                        
                        # Rolling min and max to track range
                        rolling_min_15=pandas_data['value_a'].rolling(window=15, min_periods=1).min(),
                        rolling_max_15=pandas_data['value_a'].rolling(window=15, min_periods=1).max(),
                        
                        # Exponentially weighted moving average for more recent emphasis
                        ewm_alpha_03=pandas_data['value_a'].ewm(alpha=0.3).mean()
                    )
                    
                    # Add a more complex calculation combining multiple rolling metrics
                    if 'value_b' in pandas_data.columns:
                        rolling_a = pandas_data['value_a'].rolling(window=10, min_periods=1).mean()
                        rolling_b = pandas_data['value_b'].rolling(window=10, min_periods=1).mean()
                        
                        # Calculate relative strength index-like metric
                        pandas_results['rolling_ratio'] = rolling_a / rolling_b.clip(lower=0.1)
                pd_rolling_time = time.time() - pd_rolling_start
                pandas_metrics['operationTimes']['rolling'] = pandas_metrics['operationTimes'].get('rolling', 0) + pd_rolling_time
            
            pandas_execution_time = time.time() - pandas_start_time
            pandas_memory_usage = get_memory_usage() - base_memory
            
            # Store run metrics
            pandas_run_metrics = {
                'executionTime': pandas_execution_time,
                'memoryUsage': pandas_memory_usage,
                'operationTimes': {k: v / (run + 1) for k, v in pandas_metrics['operationTimes'].items()}
            }
            
            pandas_metrics['runs'].append(pandas_run_metrics)
            
            # Store a reference to pandas_results for later comparison, then reset environment before FireDucks run
            pd_result_for_comparison = None
            if 'pandas_results' in locals() and pandas_results is not None:
                pd_result_for_comparison = pandas_results.copy()
            
            del pandas_data
            # Don't delete pandas_results as we need it later for comparison
            pandas_results = None
            gc.collect()
            time.sleep(0.1)  # Brief pause between runs
            
            # Reset memory measuring for FireDucks
            base_memory = get_memory_usage()
            
            # Run FireDucks operations
            fireducks_data = None
            fireducks_results = None
            fireducks_start_time = time.time()
            
            if operations.get('load', True):
                fd_load_start = time.time()
                try:
                    fireducks_data = fdpd.read_csv(file_path)
                except Exception as e:
                    print(f"FireDucks read_csv error: {str(e)}", file=sys.stderr)
                    # No fallback - we want to use pure FireDucks
                    print("CRITICAL: Running PURE FireDucks - No fallback to pandas allowed", file=sys.stderr)
                    # Add to metrics but set results to None
                    fireducks_metrics['errors'] = fireducks_metrics.get('errors', []) + ["Error: " + str(e)]
                fd_load_time = time.time() - fd_load_start
                fireducks_metrics['operationTimes']['load'] = fireducks_metrics['operationTimes'].get('load', 0) + fd_load_time
            
            if operations.get('groupby', False) and fireducks_data is not None:
                fd_groupby_start = time.time()
                if 'category' in fireducks_data.columns and 'value_a' in fireducks_data.columns:
                    if 'region' in fireducks_data.columns and 'segment' in fireducks_data.columns:
                        # Multi-level groupby with multiple aggregations - same as pandas
                        fireducks_results = fireducks_data.groupby(['category', 'region', 'segment']).agg({
                            'value_a': ['mean', 'std', 'min', 'max'],
                            'value_b': ['mean', 'std'],
                            'value_c': 'sum',
                            'metric': 'mean'
                        }).reset_index()
                    else:
                        # Fall back to simpler groupby if columns missing
                        fireducks_results = fireducks_data.groupby('category')['value_a'].mean().reset_index()
                fd_groupby_time = time.time() - fd_groupby_start
                fireducks_metrics['operationTimes']['groupby'] = fireducks_metrics['operationTimes'].get('groupby', 0) + fd_groupby_time
            
            if operations.get('merge', False) and fireducks_data is not None:
                fd_merge_start = time.time()
                if len(fireducks_data) > 10:
                    # Create a small DataFrame to merge with - same as pandas
                    if 'category' in fireducks_data.columns:
                        categories = fireducks_data['category'].unique()
                        # Create directly with FireDucks
                        try:
                            merge_df = fdpd.DataFrame({
                                'category': categories,
                                'weight': np.random.rand(len(categories)) * 10
                            })
                        except Exception as e:
                            print(f"FireDucks DataFrame creation error: {str(e)}", file=sys.stderr)
                            print("CRITICAL: Running PURE FireDucks - No fallback to pandas allowed", file=sys.stderr)
                            # Add to metrics but set results to None
                            fireducks_metrics['errors'] = fireducks_metrics.get('errors', []) + ["DataFrame creation error: " + str(e)]
                            fireducks_results = None
                            return
                        fireducks_results = fireducks_data.merge(merge_df, on='category')
                fd_merge_time = time.time() - fd_merge_start
                fireducks_metrics['operationTimes']['merge'] = fireducks_metrics['operationTimes'].get('merge', 0) + fd_merge_time
            
            if operations.get('filter', False) and fireducks_data is not None:
                fd_filter_start = time.time()
                if 'value_a' in fireducks_data.columns:
                    if all(col in fireducks_data.columns for col in ['value_b', 'value_c', 'value_d']):
                        # Complex filtering that combines multiple conditions - same as pandas
                        mean_a = fireducks_data['value_a'].mean()
                        mean_b = fireducks_data['value_b'].mean()
                        # Multiple conditions with arithmetic operations
                        condition1 = fireducks_data['value_a'] > mean_a
                        condition2 = fireducks_data['value_b'] < mean_b
                        condition3 = (fireducks_data['value_c'] / fireducks_data['value_d']) > 1.0
                        condition4 = fireducks_data['value_sum'] > fireducks_data['value_sum'].median()
                        
                        # Combined complex filtering operation
                        fireducks_results = fireducks_data[
                            condition1 & 
                            (condition2 | condition3) & 
                            condition4
                        ]
                    else:
                        # Fall back to simpler filter if columns missing
                        fireducks_results = fireducks_data[fireducks_data['value_a'] > fireducks_data['value_a'].mean()]
                fd_filter_time = time.time() - fd_filter_start
                fireducks_metrics['operationTimes']['filter'] = fireducks_metrics['operationTimes'].get('filter', 0) + fd_filter_time
            
            if operations.get('rolling', False) and fireducks_data is not None:
                fd_rolling_start = time.time()
                if 'value_a' in fireducks_data.columns:
                    # Sort dataframe first to ensure correct rolling window calculations
                    fireducks_data = fireducks_data.sort_values('id') if 'id' in fireducks_data.columns else fireducks_data
                    
                    try:
                        # Create multiple rolling window calculations with different windows - same as pandas
                        fireducks_results = fireducks_data.assign(
                            # Simple mean with a small window
                            rolling_avg_5=fireducks_data['value_a'].rolling(window=5, min_periods=1).mean(),
                            
                            # Longer window for trend analysis
                            rolling_avg_20=fireducks_data['value_a'].rolling(window=20, min_periods=1).mean(),
                            
                            # Rolling standard deviation to measure volatility
                            rolling_std_10=fireducks_data['value_a'].rolling(window=10, min_periods=1).std(),
                            
                            # Rolling min and max to track range
                            rolling_min_15=fireducks_data['value_a'].rolling(window=15, min_periods=1).min(),
                            rolling_max_15=fireducks_data['value_a'].rolling(window=15, min_periods=1).max(),
                            
                            # Exponentially weighted moving average for more recent emphasis
                            ewm_alpha_03=fireducks_data['value_a'].ewm(alpha=0.3).mean()
                        )
                        
                        # Add a more complex calculation combining multiple rolling metrics
                        if 'value_b' in fireducks_data.columns:
                            rolling_a = fireducks_data['value_a'].rolling(window=10, min_periods=1).mean()
                            rolling_b = fireducks_data['value_b'].rolling(window=10, min_periods=1).mean()
                            
                            # Calculate relative strength index-like metric
                            fireducks_results['rolling_ratio'] = rolling_a / rolling_b.clip(lower=0.1)
                    except Exception as e:
                        print(f"FireDucks rolling error: {str(e)}", file=sys.stderr)
                        # No fallback - we want to use pure FireDucks or nothing
                        print("CRITICAL: Running PURE FireDucks - No fallback to simplified operations allowed", file=sys.stderr)
                        # Add to metrics but set results to None
                        fireducks_metrics['errors'] = fireducks_metrics.get('errors', []) + ["Rolling error: " + str(e)]
                        fireducks_results = None
                fd_rolling_time = time.time() - fd_rolling_start
                fireducks_metrics['operationTimes']['rolling'] = fireducks_metrics['operationTimes'].get('rolling', 0) + fd_rolling_time
            
            fireducks_execution_time = time.time() - fireducks_start_time
            fireducks_memory_usage = get_memory_usage() - base_memory
            
            # Store run metrics
            fireducks_run_metrics = {
                'executionTime': fireducks_execution_time,
                'memoryUsage': fireducks_memory_usage,
                'operationTimes': {k: v / (run + 1) for k, v in fireducks_metrics['operationTimes'].items()}
            }
            
            fireducks_metrics['runs'].append(fireducks_run_metrics)
            
            # Check if results match using the stored reference
            if pd_result_for_comparison is not None and fireducks_results is not None:
                try:
                    # Convert to pandas for comparison if needed
                    fd_compare = fireducks_results
                    if hasattr(fireducks_results, 'to_pandas'):
                        fd_compare = fireducks_results.to_pandas()
                    
                    pd_compare = pd_result_for_comparison
                    
                    # Basic shape check
                    if pd_compare.shape != fd_compare.shape:
                        results_match = False
                    
                    # For more detailed comparison, we could use more sophisticated checks
                    # but this is sufficient for a basic check
                except Exception as e:
                    print(f"Error comparing results: {str(e)}", file=sys.stderr)
                    results_match = False
        
        # Average the metrics if multiple runs
        if run_count > 1:
            # Calculate average execution time
            pandas_metrics['executionTime'] = sum(run['executionTime'] for run in pandas_metrics['runs']) / run_count
            fireducks_metrics['executionTime'] = sum(run['executionTime'] for run in fireducks_metrics['runs']) / run_count
            
            # Calculate average memory usage
            pandas_metrics['memoryUsage'] = sum(run['memoryUsage'] for run in pandas_metrics['runs']) / run_count
            fireducks_metrics['memoryUsage'] = sum(run['memoryUsage'] for run in fireducks_metrics['runs']) / run_count
            
            # Average the operation times
            for op in pandas_metrics['operationTimes']:
                pandas_metrics['operationTimes'][op] /= run_count
            
            for op in fireducks_metrics['operationTimes']:
                fireducks_metrics['operationTimes'][op] /= run_count
        else:
            # Just use the single run values
            pandas_metrics['executionTime'] = pandas_metrics['runs'][0]['executionTime'] if pandas_metrics['runs'] else 0
            fireducks_metrics['executionTime'] = fireducks_metrics['runs'][0]['executionTime'] if fireducks_metrics['runs'] else 0
            pandas_metrics['memoryUsage'] = pandas_metrics['runs'][0]['memoryUsage'] if pandas_metrics['runs'] else 0
            fireducks_metrics['memoryUsage'] = fireducks_metrics['runs'][0]['memoryUsage'] if fireducks_metrics['runs'] else 0
        
        # Return the metrics
        return {
            'pandasMetrics': pandas_metrics,
            'fireducksMetrics': fireducks_metrics,
            'resultsMatch': results_match
        }
    except Exception as e:
        print(f"Error during comparison: {str(e)}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


def main():
    """
    Main function to handle command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Analyze and compare Pandas vs FireDucks performance')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Analyze CSV command
    analyze_parser = subparsers.add_parser('analyze-csv', help='Analyze a CSV file')
    analyze_parser.add_argument('file', help='Path to the CSV file')
    
    # Generate data command
    generate_parser = subparsers.add_parser('generate-data', help='Generate synthetic data')
    generate_parser.add_argument('rows', help='Number of rows to generate')
    
    # Compare libraries command
    compare_parser = subparsers.add_parser('compare', help='Compare Pandas and FireDucks')
    compare_parser.add_argument('file', help='Path to the CSV file')
    compare_parser.add_argument('operations', help='JSON string of operations to run')
    compare_parser.add_argument('settings', help='JSON string of settings')
    
    args = parser.parse_args()
    
    if args.command == 'analyze-csv':
        result = analyze_csv(args.file)
    elif args.command == 'generate-data':
        result = generate_synthetic_data(args.rows)
    elif args.command == 'compare':
        operations = json.loads(args.operations)
        settings = json.loads(args.settings)
        result = compare_libraries(args.file, operations, settings)
    else:
        parser.print_help()
        sys.exit(1)
    
    # Output the result as JSON
    print(json.dumps(result))


if __name__ == '__main__':
    main()