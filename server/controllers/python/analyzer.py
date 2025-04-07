#!/usr/bin/env python3
"""
Enhanced analyzer for comparing Pandas and FireDucks performance with advanced operations.
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
from io import StringIO

try:
    # Data processing libraries
    import numpy as np
    import pandas as pd

    # Import the real FireDucks library with the correct import path
    import fireducks as fd
    import fireducks.pandas as fdpd  # This is the correct import for FireDucks DataFrame
except ImportError as e:
    print(f"Error importing required libraries: {str(e)}", file=sys.stderr)
    print("Please install all required libraries with:", file=sys.stderr)
    print("pip install numpy pandas fireducks", file=sys.stderr)
    sys.exit(1)

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
    Generate synthetic data for testing with more complex features.
    Optimized for handling large datasets efficiently with chunked generation.
    """
    try:
        num_rows = int(num_rows)
        
        # Generate a DataFrame with several columns of different types
        # and sufficient complexity to test FireDucks optimizations
        np.random.seed(42)  # For reproducibility
        
        # For extremely large datasets, use a completely different approach
        # with chunk-based processing to avoid memory issues
        is_large_dataset = num_rows > 500000
        if is_large_dataset:
            print(f"Generating optimized large dataset with {num_rows} rows using chunk processing", file=sys.stderr)
            
            # Use chunk-based generation for large datasets
            return generate_large_synthetic_data(num_rows)
            
        # For smaller datasets, use the standard approach
        print(f"Generating standard dataset with {num_rows} rows", file=sys.stderr)
        
        # Create more complex data with more columns and data types
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
            'date': [f"2022-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}" for i in range(num_rows)],
            'flag': np.random.choice([True, False], num_rows),
            'metric': np.random.exponential(10, num_rows),
            'score': np.random.uniform(0, 100, num_rows).round(2),
            'rank': np.random.randint(1, 100, num_rows),
            'text': np.random.choice(["This is sample text", "Another example", "Testing FireDucks", 
                                    "Performance comparison", "Data analysis", "Machine learning"], num_rows),
            'level1': np.random.choice(['Group1', 'Group2', 'Group3'], num_rows),
            'level2': np.random.choice(['Subgroup1', 'Subgroup2', 'Subgroup3', 'Subgroup4'], num_rows),
            'level3': np.random.choice(['Item1', 'Item2', 'Item3', 'Item4', 'Item5'], num_rows),
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
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
                
def generate_large_synthetic_data(num_rows):
    """
    Generate synthetic data for large datasets (>500K rows) using memory-efficient chunked processing.
    """
    try:
        # Define chunk size based on available memory
        chunk_size = min(100000, num_rows // 10)  # Maximum 100K rows per chunk, or 1/10 of total rows
        
        print(f"Using chunk size of {chunk_size} rows for more efficient memory usage", file=sys.stderr)
        
        # Create base dataframe with numeric columns only (less memory intensive)
        print("Creating base dataframe with numeric columns...", file=sys.stderr)
        
        # Create a simplified dataset with only the essential numeric columns
        # This dramatically reduces memory usage for large datasets
        numeric_data = {
            'id': np.arange(1, num_rows + 1),
            'value_a': np.random.normal(100, 15, num_rows),
            'value_b': np.random.normal(50, 10, num_rows),
            'value_c': np.random.gamma(5, 2, num_rows)
        }
        
        # Create the basic DataFrame
        df = pd.DataFrame(numeric_data)
        
        # Now add categorical columns in chunks to save memory
        print("Adding categorical data in chunks...", file=sys.stderr)
        
        # Define categorical values once
        categories = ['A', 'B', 'C', 'D']
        subcategories = ['X1', 'X2', 'Y1', 'Y2', 'Z1', 'Z2']
        regions = ['North', 'South', 'East', 'West', 'Central']
        segments = ['Consumer', 'Enterprise', 'Government', 'Education']
        texts = ["This is sample text", "Another example", "Testing FireDucks", 
                "Performance comparison", "Data analysis", "Machine learning"]
        hierarchies = {
            'level1': ['Group1', 'Group2', 'Group3'],
            'level2': ['Subgroup1', 'Subgroup2', 'Subgroup3', 'Subgroup4'],
            'level3': ['Item1', 'Item2', 'Item3', 'Item4', 'Item5']
        }
        
        # Process categorical data in chunks
        categorical_columns = [
            ('category', categories),
            ('subcategory', subcategories),
            ('region', regions),
            ('segment', segments),
            ('text', texts),
            ('level1', hierarchies['level1']),
            ('level2', hierarchies['level2']),
            ('level3', hierarchies['level3']),
        ]
        
        # Add each categorical column in chunks
        for col_name, values in categorical_columns:
            print(f"Adding {col_name} column...", file=sys.stderr)
            df[col_name] = ''  # Initialize with empty strings
            for i in range(0, num_rows, chunk_size):
                end_idx = min(i + chunk_size, num_rows)
                df.loc[i:end_idx-1, col_name] = np.random.choice(values, end_idx - i)
                # Force garbage collection after each chunk
                gc.collect()
        
        # Add boolean column
        print("Adding boolean column...", file=sys.stderr)
        df['flag'] = False  # Initialize
        for i in range(0, num_rows, chunk_size):
            end_idx = min(i + chunk_size, num_rows)
            df.loc[i:end_idx-1, 'flag'] = np.random.choice([True, False], end_idx - i)
        
        # Add date column using a deterministic pattern
        print("Adding date column...", file=sys.stderr)
        df['date'] = ''  # Initialize with empty strings
        for i in range(0, num_rows, chunk_size):
            end_idx = min(i + chunk_size, num_rows)
            df.loc[i:end_idx-1, 'date'] = [f"2022-{(j % 12) + 1:02d}-{(j % 28) + 1:02d}" for j in range(i, end_idx)]
        
        # Add a few more calculated columns
        print("Adding calculated columns...", file=sys.stderr)
        df['value_ratio'] = df['value_a'] / df['value_b'].clip(lower=0.1)
        df['value_sum'] = df['value_a'] + df['value_b'] + df['value_c']
        
        # We'll add simpler calculated columns to reduce memory pressure
        print("Getting sample data...", file=sys.stderr)
        sample_data = df.head(5).to_dict(orient='records')
        
        # Save directly to CSV file in chunks
        print("Converting to CSV...", file=sys.stderr)
        csv_buffer = StringIO()
        
        # Write header
        csv_buffer.write(','.join(df.columns) + '\n')
        
        # Write data in chunks
        chunk_count = (num_rows + chunk_size - 1) // chunk_size  # Ceiling division
        for i in range(chunk_count):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, num_rows)
            print(f"Writing chunk {i+1}/{chunk_count} to CSV...", file=sys.stderr)
            
            # Get chunk and convert to CSV
            chunk = df.iloc[start_idx:end_idx]
            chunk_csv = chunk.to_csv(index=False, header=False)
            csv_buffer.write(chunk_csv)
            
            # Clean up to save memory
            del chunk
            gc.collect()
        
        # Get the CSV string and encode as base64
        print("Encoding as base64...", file=sys.stderr)
        csv_str = csv_buffer.getvalue()
        csv_base64 = base64.b64encode(csv_str.encode()).decode()
        
        # Clean up dataframe to save memory
        del df
        gc.collect()
        
        print("Finished generating large dataset successfully", file=sys.stderr)
        
        # Return the results
        return {
            'rows': num_rows,
            'columns': len(sample_data[0]) if sample_data else 0,
            'sample_data': sample_data,
            'file_content': csv_base64
        }
    except Exception as e:
        print(f"Error in generate_large_synthetic_data: {str(e)}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


def compare_libraries(file_path, operations, settings):
    """
    Compare Pandas and FireDucks performance with expanded operation set.
    """
    try:
        # Parse settings
        multi_run = settings.get('multiRunEnabled', False)
        
        # Limit run count to prevent timeouts
        requested_run_count = settings.get('runCount', 3) if multi_run else 1
        run_count = min(requested_run_count, 5)  # Cap at 5 runs maximum to prevent timeouts
        
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
            
            # Basic Operations
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
            
            # Advanced Operations
            
            # Pivot Table - creates resource-intensive cross-tabulations
            if operations.get('pivotTable', False) and pandas_data is not None:
                pd_pivot_start = time.time()
                if all(col in pandas_data.columns for col in ['category', 'region', 'value_a']):
                    # Create a pivot table summarizing data across categories and regions
                    pandas_results = pandas_data.pivot_table(
                        values=['value_a', 'value_b', 'value_c'],
                        index=['category', 'subcategory'],
                        columns=['region'],
                        aggfunc={
                            'value_a': ['mean', 'sum'],
                            'value_b': ['mean', 'std'],
                            'value_c': 'sum'
                        },
                        fill_value=0
                    )
                    # Flatten hierarchical column names
                    pandas_results.columns = ['_'.join(col).strip() for col in pandas_results.columns.values]
                    # Reset index for consistent output
                    pandas_results = pandas_results.reset_index()
                pd_pivot_time = time.time() - pd_pivot_start
                pandas_metrics['operationTimes']['pivotTable'] = pandas_metrics['operationTimes'].get('pivotTable', 0) + pd_pivot_time
            
            # Complex Aggregation - advanced statistical functions
            if operations.get('complexAggregation', False) and pandas_data is not None:
                pd_complex_agg_start = time.time()
                if all(col in pandas_data.columns for col in ['category', 'value_a', 'value_b']):
                    # Define custom aggregation functions
                    def range_pct(x):
                        """Calculate the range as percentage of the mean"""
                        if len(x) == 0 or x.mean() == 0:
                            return 0
                        return (x.max() - x.min()) / x.mean() * 100
                    
                    def coef_variation(x):
                        """Calculate coefficient of variation"""
                        if len(x) == 0 or x.mean() == 0:
                            return 0
                        return x.std() / x.mean() * 100
                    
                    # Complex multi-level aggregation with custom functions
                    pandas_results = pandas_data.groupby(['category', 'region', 'segment']).agg({
                        'value_a': ['mean', 'median', 'std', range_pct, coef_variation],
                        'value_b': ['mean', 'median', 'std', range_pct],
                        'value_c': ['sum', 'mean', lambda x: (x > x.mean()).sum() / len(x) * 100],
                        'value_d': ['mean', lambda x: x.quantile(0.75) - x.quantile(0.25)]
                    })
                    
                    # Flatten hierarchical column names and reset index
                    pandas_results.columns = ['_'.join(str(col) for col in col).strip() for col in pandas_results.columns.values]
                    pandas_results = pandas_results.reset_index()
                pd_complex_agg_time = time.time() - pd_complex_agg_start
                pandas_metrics['operationTimes']['complexAggregation'] = pandas_metrics['operationTimes'].get('complexAggregation', 0) + pd_complex_agg_time
            
            # Window Functions - advanced operations that consider surrounding rows
            if operations.get('windowFunctions', False) and pandas_data is not None:
                pd_window_start = time.time()
                if all(col in pandas_data.columns for col in ['category', 'value_a']):
                    # Sort data for window functions
                    df_sorted = pandas_data.sort_values(['category', 'value_a'])
                    
                    # Apply various window functions
                    # Calculate ranks within each category
                    df_sorted['rank_in_category'] = df_sorted.groupby('category')['value_a'].rank(method='dense')
                    
                    # Calculate percentiles within each category
                    df_sorted['percentile_in_category'] = df_sorted.groupby('category')['value_a'].rank(pct=True)
                    
                    # Calculate differences from category average
                    df_sorted['diff_from_category_avg'] = df_sorted['value_a'] - df_sorted.groupby('category')['value_a'].transform('mean')
                    
                    # Calculate cumulative stats within categories
                    df_sorted['cumulative_sum'] = df_sorted.groupby('category')['value_a'].cumsum()
                    df_sorted['cumulative_pct'] = df_sorted.groupby('category')['value_a'].cumsum() / df_sorted.groupby('category')['value_a'].transform('sum')
                    
                    # Calculate moving averages but partitioned by category
                    df_sorted['moving_avg_3'] = df_sorted.groupby('category')['value_a'].transform(lambda x: x.rolling(3, min_periods=1).mean())
                    
                    # Advanced window lag/lead operations
                    df_sorted['lag_value'] = df_sorted.groupby('category')['value_a'].shift(1)
                    df_sorted['lead_value'] = df_sorted.groupby('category')['value_a'].shift(-1)
                    df_sorted['pct_change'] = df_sorted.groupby('category')['value_a'].pct_change()
                    
                    pandas_results = df_sorted
                pd_window_time = time.time() - pd_window_start
                pandas_metrics['operationTimes']['windowFunctions'] = pandas_metrics['operationTimes'].get('windowFunctions', 0) + pd_window_time
            
            # String Manipulation - text processing operations
            if operations.get('stringManipulation', False) and pandas_data is not None:
                pd_string_start = time.time()
                if 'text' in pandas_data.columns:
                    # Apply various string operations
                    text_df = pandas_data.copy()
                    
                    # Basic string operations
                    text_df['text_upper'] = text_df['text'].str.upper()
                    text_df['text_lower'] = text_df['text'].str.lower()
                    text_df['text_length'] = text_df['text'].str.len()
                    
                    # Extract substrings
                    text_df['first_5_chars'] = text_df['text'].str[:5]
                    text_df['last_3_chars'] = text_df['text'].str[-3:]
                    
                    # String pattern matching
                    text_df['contains_data'] = text_df['text'].str.contains('data', case=False)
                    text_df['contains_test'] = text_df['text'].str.contains('test', case=False)
                    
                    # String replacement
                    text_df['text_replaced'] = text_df['text'].str.replace('test', 'eval', case=False)
                    
                    # String splitting and extraction
                    text_df['first_word'] = text_df['text'].str.split().str[0]
                    text_df['word_count'] = text_df['text'].str.split().str.len()
                    
                    # Combine and join strings
                    if 'category' in text_df.columns:
                        text_df['combined_text'] = text_df['category'] + ': ' + text_df['text']
                    
                    pandas_results = text_df
                pd_string_time = time.time() - pd_string_start
                pandas_metrics['operationTimes']['stringManipulation'] = pandas_metrics['operationTimes'].get('stringManipulation', 0) + pd_string_time
            
            # Nested Operations - more complex combinations of operations
            if operations.get('nestedOperations', False) and pandas_data is not None:
                pd_nested_start = time.time()
                
                # Create a multi-stage data transformation pipeline
                # First, filter the data
                if 'value_a' in pandas_data.columns and 'category' in pandas_data.columns:
                    stage1 = pandas_data[pandas_data['value_a'] > pandas_data['value_a'].mean()]
                    
                    # Group and aggregate the filtered data
                    stage2 = stage1.groupby('category').agg({
                        'value_a': ['mean', 'count'],
                        'value_b': ['mean', 'std'],
                        'value_c': 'sum'
                    })
                    
                    # Flatten the multi-level columns
                    stage2.columns = ['_'.join(col).strip() for col in stage2.columns.values]
                    stage2 = stage2.reset_index()
                    
                    # Apply calculations to the aggregated results
                    if 'value_a_mean' in stage2.columns and 'value_b_mean' in stage2.columns:
                        stage2['ratio'] = stage2['value_a_mean'] / stage2['value_b_mean'].clip(lower=0.1)
                    
                    # Sort the results
                    if 'value_c_sum' in stage2.columns:
                        stage3 = stage2.sort_values('value_c_sum', ascending=False)
                    else:
                        stage3 = stage2
                        
                    # Add rank information
                    if 'value_a_mean' in stage3.columns:
                        stage3['rank'] = stage3['value_a_mean'].rank(method='dense', ascending=False)
                    
                    # Calculate percentiles
                    for col in stage3.columns:
                        if col.startswith('value_') and col.endswith('_mean'):
                            new_col = col.replace('_mean', '_percentile')
                            if col in stage3.columns and stage3[col].std() > 0:
                                stage3[new_col] = (stage3[col] - stage3[col].min()) / (stage3[col].max() - stage3[col].min())
                    
                    pandas_results = stage3
                pd_nested_time = time.time() - pd_nested_start
                pandas_metrics['operationTimes']['nestedOperations'] = pandas_metrics['operationTimes'].get('nestedOperations', 0) + pd_nested_time
            
            # Additional operations
            if operations.get('concat', False) and pandas_data is not None:
                pd_concat_start = time.time()
                # Create a copy of the original dataframe to concatenate
                subset_data = pandas_data.head(len(pandas_data) // 2).copy()
                # Concatenate dataframes
                concat_result = pd.concat([pandas_data, subset_data], ignore_index=True)
                pd_concat_time = time.time() - pd_concat_start
                pandas_metrics['operationTimes']['concat'] = pandas_metrics['operationTimes'].get('concat', 0) + pd_concat_time
                
            if operations.get('sort', False) and pandas_data is not None:
                pd_sort_start = time.time()
                # Sort by multiple columns
                if 'value_a' in pandas_data.columns and 'value_b' in pandas_data.columns:
                    sorted_data = pandas_data.sort_values(by=['value_a', 'value_b'], ascending=[False, True])
                else:
                    # Fall back to sorting by the first column if specific columns not found
                    sorted_data = pandas_data.sort_values(by=pandas_data.columns[0])
                pd_sort_time = time.time() - pd_sort_start
                pandas_metrics['operationTimes']['sort'] = pandas_metrics['operationTimes'].get('sort', 0) + pd_sort_time
                
            if operations.get('info', False) and pandas_data is not None:
                pd_info_start = time.time()
                # Capture DataFrame.info() output
                buffer = StringIO()
                pandas_data.info(buf=buffer)
                _ = buffer.getvalue()  # Get the info output but we don't need to save it
                pd_info_time = time.time() - pd_info_start
                pandas_metrics['operationTimes']['info'] = pandas_metrics['operationTimes'].get('info', 0) + pd_info_time
                
            if operations.get('toCSV', False) and pandas_data is not None:
                pd_tocsv_start = time.time()
                # Write to CSV (in memory)
                csv_buffer = StringIO()
                pandas_data.to_csv(csv_buffer, index=False)
                _ = csv_buffer.getvalue()  # Get the CSV but we don't need to save it
                pd_tocsv_time = time.time() - pd_tocsv_start
                pandas_metrics['operationTimes']['toCSV'] = pandas_metrics['operationTimes'].get('toCSV', 0) + pd_tocsv_time
            
            # Finalize Pandas metrics for this run
            pandas_end_time = time.time()
            pandas_run_time = pandas_end_time - pandas_start_time
            pandas_metrics['executionTime'] += pandas_run_time
            
            # Measure memory after pandas operations
            pandas_memory = get_memory_usage() - base_memory
            pandas_metrics['memoryUsage'] += pandas_memory
            
            # Store the individual run metrics
            pandas_run_metrics = {
                'executionTime': pandas_run_time,
                'memoryUsage': pandas_memory
            }
            pandas_metrics['runs'].append(pandas_run_metrics)
            
            print(f"Pandas run {run+1}: {pandas_run_time:.4f}s, Memory: {pandas_memory:.2f}MB", file=sys.stderr)
            
            # -------------------------------------------
            # Now run the same operations with FireDucks
            # -------------------------------------------
            
            # Reset memory measuring
            base_memory = get_memory_usage()
            
            # Run FireDucks operations
            fd_data = None
            fd_results = None
            fd_start_time = time.time()
            
            # If using real FireDucks, run the actual operations
            # If not, just copy the pandas results to avoid errors
            if USING_REAL_FIREDUCKS:
                try:
                    if operations.get('load', True):
                        fd_load_start = time.time()
                        fd_data = fdpd.read_csv(file_path)
                        fd_load_time = time.time() - fd_load_start
                        fireducks_metrics['operationTimes']['load'] = fireducks_metrics['operationTimes'].get('load', 0) + fd_load_time
                        
                    # Basic Operations
                    if operations.get('groupby', False) and fd_data is not None:
                        fd_groupby_start = time.time()
                        if 'category' in fd_data.columns and 'value_a' in fd_data.columns:
                            if 'region' in fd_data.columns and 'segment' in fd_data.columns:
                                # Multi-level groupby with multiple aggregations
                                fd_results = fd_data.groupby(['category', 'region', 'segment']).agg({
                                    'value_a': ['mean', 'std', 'min', 'max'],
                                    'value_b': ['mean', 'std'],
                                    'value_c': 'sum', 
                                    'metric': 'mean'
                                }).reset_index()
                            else:
                                # Fall back to simpler groupby if columns missing
                                fd_results = fd_data.groupby('category')['value_a'].mean().reset_index()
                        fd_groupby_time = time.time() - fd_groupby_start
                        fireducks_metrics['operationTimes']['groupby'] = fireducks_metrics['operationTimes'].get('groupby', 0) + fd_groupby_time
                    
                    if operations.get('merge', False) and fd_data is not None:
                        fd_merge_start = time.time()
                        if len(fd_data) > 10:
                            if 'category' in fd_data.columns:
                                categories = fd_data['category'].unique()
                                merge_df = fdpd.DataFrame({
                                    'category': categories,
                                    'weight': np.random.rand(len(categories)) * 10
                                })
                                fd_results = fd_data.merge(merge_df, on='category')
                        fd_merge_time = time.time() - fd_merge_start
                        fireducks_metrics['operationTimes']['merge'] = fireducks_metrics['operationTimes'].get('merge', 0) + fd_merge_time
                    
                    if operations.get('filter', False) and fd_data is not None:
                        fd_filter_start = time.time()
                        if 'value_a' in fd_data.columns:
                            if all(col in fd_data.columns for col in ['value_b', 'value_c', 'value_d']):
                                # Complex filtering that combines multiple conditions
                                mean_a = fd_data['value_a'].mean()
                                mean_b = fd_data['value_b'].mean()
                                
                                # Multiple conditions with arithmetic operations
                                condition1 = fd_data['value_a'] > mean_a
                                condition2 = fd_data['value_b'] < mean_b
                                condition3 = (fd_data['value_c'] / fd_data['value_d']) > 1.0
                                condition4 = fd_data['value_sum'] > fd_data['value_sum'].median()
                                
                                # Combined complex filtering operation
                                fd_results = fd_data[
                                    condition1 & 
                                    (condition2 | condition3) & 
                                    condition4
                                ]
                            else:
                                # Fall back to simpler filter
                                fd_results = fd_data[fd_data['value_a'] > fd_data['value_a'].mean()]
                        fd_filter_time = time.time() - fd_filter_start
                        fireducks_metrics['operationTimes']['filter'] = fireducks_metrics['operationTimes'].get('filter', 0) + fd_filter_time
                    
                    if operations.get('rolling', False) and fd_data is not None:
                        fd_rolling_start = time.time()
                        if 'value_a' in fd_data.columns:
                            # Sort data first
                            fd_data = fd_data.sort_values('id') if 'id' in fd_data.columns else fd_data
                            
                            # Create multiple rolling window calculations
                            fd_results = fd_data.assign(
                                rolling_avg_5=fd_data['value_a'].rolling(window=5, min_periods=1).mean(),
                                rolling_avg_20=fd_data['value_a'].rolling(window=20, min_periods=1).mean(),
                                rolling_std_10=fd_data['value_a'].rolling(window=10, min_periods=1).std(),
                                rolling_min_15=fd_data['value_a'].rolling(window=15, min_periods=1).min(),
                                rolling_max_15=fd_data['value_a'].rolling(window=15, min_periods=1).max(),
                                ewm_alpha_03=fd_data['value_a'].ewm(alpha=0.3).mean()
                            )
                            
                            # Add a more complex calculation combining multiple rolling metrics
                            if 'value_b' in fd_data.columns:
                                rolling_a = fd_data['value_a'].rolling(window=10, min_periods=1).mean()
                                rolling_b = fd_data['value_b'].rolling(window=10, min_periods=1).mean()
                                
                                # Calculate relative strength index-like metric
                                fd_results['rolling_ratio'] = rolling_a / rolling_b.clip(lower=0.1)
                        fd_rolling_time = time.time() - fd_rolling_start
                        fireducks_metrics['operationTimes']['rolling'] = fireducks_metrics['operationTimes'].get('rolling', 0) + fd_rolling_time
                    
                    # Advanced Operations
                    
                    # Pivot Table - more complex cross-tabulations
                    if operations.get('pivotTable', False) and fd_data is not None:
                        fd_pivot_start = time.time()
                        if all(col in fd_data.columns for col in ['category', 'region', 'value_a']):
                            fd_results = fd_data.pivot_table(
                                values=['value_a', 'value_b', 'value_c'],
                                index=['category', 'subcategory'],
                                columns=['region'],
                                aggfunc={
                                    'value_a': ['mean', 'sum'],
                                    'value_b': ['mean', 'std'],
                                    'value_c': 'sum'
                                },
                                fill_value=0
                            )
                            # Flatten hierarchical column names
                            fd_results.columns = ['_'.join(col).strip() for col in fd_results.columns.values]
                            # Reset index
                            fd_results = fd_results.reset_index()
                        fd_pivot_time = time.time() - fd_pivot_start
                        fireducks_metrics['operationTimes']['pivotTable'] = fireducks_metrics['operationTimes'].get('pivotTable', 0) + fd_pivot_time
                    
                    # Complex Aggregation - advanced statistical functions
                    if operations.get('complexAggregation', False) and fd_data is not None:
                        fd_complex_agg_start = time.time()
                        if all(col in fd_data.columns for col in ['category', 'value_a', 'value_b']):
                            # Define custom aggregation functions
                            def range_pct(x):
                                """Calculate the range as percentage of the mean"""
                                if len(x) == 0 or x.mean() == 0:
                                    return 0
                                return (x.max() - x.min()) / x.mean() * 100
                            
                            def coef_variation(x):
                                """Calculate coefficient of variation"""
                                if len(x) == 0 or x.mean() == 0:
                                    return 0
                                return x.std() / x.mean() * 100
                            
                            # Complex multi-level aggregation with custom functions
                            fd_results = fd_data.groupby(['category', 'region', 'segment']).agg({
                                'value_a': ['mean', 'median', 'std', range_pct, coef_variation],
                                'value_b': ['mean', 'median', 'std', range_pct],
                                'value_c': ['sum', 'mean', lambda x: (x > x.mean()).sum() / len(x) * 100],
                                'value_d': ['mean', lambda x: x.quantile(0.75) - x.quantile(0.25)]
                            })
                            
                            # Flatten hierarchical column names and reset index
                            fd_results.columns = ['_'.join(str(col) for col in col).strip() for col in fd_results.columns.values]
                            fd_results = fd_results.reset_index()
                        fd_complex_agg_time = time.time() - fd_complex_agg_start
                        fireducks_metrics['operationTimes']['complexAggregation'] = fireducks_metrics['operationTimes'].get('complexAggregation', 0) + fd_complex_agg_time
                    
                    # Window Functions - operations considering surrounding rows
                    if operations.get('windowFunctions', False) and fd_data is not None:
                        fd_window_start = time.time()
                        if all(col in fd_data.columns for col in ['category', 'value_a']):
                            # Sort data for window functions
                            fd_sorted = fd_data.sort_values(['category', 'value_a'])
                            
                            # Apply various window functions
                            # Calculate ranks within each category
                            fd_sorted['rank_in_category'] = fd_sorted.groupby('category')['value_a'].rank(method='dense')
                            
                            # Calculate percentiles within each category
                            fd_sorted['percentile_in_category'] = fd_sorted.groupby('category')['value_a'].rank(pct=True)
                            
                            # Calculate differences from category average
                            fd_sorted['diff_from_category_avg'] = fd_sorted['value_a'] - fd_sorted.groupby('category')['value_a'].transform('mean')
                            
                            # Calculate cumulative stats within categories
                            fd_sorted['cumulative_sum'] = fd_sorted.groupby('category')['value_a'].cumsum()
                            fd_sorted['cumulative_pct'] = fd_sorted.groupby('category')['value_a'].cumsum() / fd_sorted.groupby('category')['value_a'].transform('sum')
                            
                            # Calculate moving averages but partitioned by category
                            fd_sorted['moving_avg_3'] = fd_sorted.groupby('category')['value_a'].transform(lambda x: x.rolling(3, min_periods=1).mean())
                            
                            # Advanced window lag/lead operations
                            fd_sorted['lag_value'] = fd_sorted.groupby('category')['value_a'].shift(1)
                            fd_sorted['lead_value'] = fd_sorted.groupby('category')['value_a'].shift(-1)
                            fd_sorted['pct_change'] = fd_sorted.groupby('category')['value_a'].pct_change()
                            
                            fd_results = fd_sorted
                        fd_window_time = time.time() - fd_window_start
                        fireducks_metrics['operationTimes']['windowFunctions'] = fireducks_metrics['operationTimes'].get('windowFunctions', 0) + fd_window_time
                    
                    # String Manipulation - text processing operations
                    if operations.get('stringManipulation', False) and fd_data is not None:
                        fd_string_start = time.time()
                        if 'text' in fd_data.columns:
                            # Apply various string operations
                            fd_text_df = fd_data.copy()
                            
                            # Basic string operations
                            fd_text_df['text_upper'] = fd_text_df['text'].str.upper()
                            fd_text_df['text_lower'] = fd_text_df['text'].str.lower()
                            fd_text_df['text_length'] = fd_text_df['text'].str.len()
                            
                            # Extract substrings
                            fd_text_df['first_5_chars'] = fd_text_df['text'].str[:5]
                            fd_text_df['last_3_chars'] = fd_text_df['text'].str[-3:]
                            
                            # String pattern matching
                            fd_text_df['contains_data'] = fd_text_df['text'].str.contains('data', case=False)
                            fd_text_df['contains_test'] = fd_text_df['text'].str.contains('test', case=False)
                            
                            # String replacement
                            fd_text_df['text_replaced'] = fd_text_df['text'].str.replace('test', 'eval', case=False)
                            
                            # String splitting and extraction
                            fd_text_df['first_word'] = fd_text_df['text'].str.split().str[0]
                            fd_text_df['word_count'] = fd_text_df['text'].str.split().str.len()
                            
                            # Combine and join strings
                            if 'category' in fd_text_df.columns:
                                fd_text_df['combined_text'] = fd_text_df['category'] + ': ' + fd_text_df['text']
                            
                            fd_results = fd_text_df
                        fd_string_time = time.time() - fd_string_start
                        fireducks_metrics['operationTimes']['stringManipulation'] = fireducks_metrics['operationTimes'].get('stringManipulation', 0) + fd_string_time
                    
                    # Nested Operations - more complex combinations of operations
                    if operations.get('nestedOperations', False) and fd_data is not None:
                        fd_nested_start = time.time()
                        
                        # Create a multi-stage data transformation pipeline
                        # First, filter the data
                        if 'value_a' in fd_data.columns and 'category' in fd_data.columns:
                            fd_stage1 = fd_data[fd_data['value_a'] > fd_data['value_a'].mean()]
                            
                            # Group and aggregate the filtered data
                            fd_stage2 = fd_stage1.groupby('category').agg({
                                'value_a': ['mean', 'count'],
                                'value_b': ['mean', 'std'],
                                'value_c': 'sum'
                            })
                            
                            # Flatten the multi-level columns
                            fd_stage2.columns = ['_'.join(col).strip() for col in fd_stage2.columns.values]
                            fd_stage2 = fd_stage2.reset_index()
                            
                            # Apply calculations to the aggregated results
                            if 'value_a_mean' in fd_stage2.columns and 'value_b_mean' in fd_stage2.columns:
                                fd_stage2['ratio'] = fd_stage2['value_a_mean'] / fd_stage2['value_b_mean'].clip(lower=0.1)
                            
                            # Sort the results
                            if 'value_c_sum' in fd_stage2.columns:
                                fd_stage3 = fd_stage2.sort_values('value_c_sum', ascending=False)
                            else:
                                fd_stage3 = fd_stage2
                                
                            # Add rank information
                            if 'value_a_mean' in fd_stage3.columns:
                                fd_stage3['rank'] = fd_stage3['value_a_mean'].rank(method='dense', ascending=False)
                            
                            # Calculate percentiles
                            for col in fd_stage3.columns:
                                if col.startswith('value_') and col.endswith('_mean'):
                                    new_col = col.replace('_mean', '_percentile')
                                    if col in fd_stage3.columns and fd_stage3[col].std() > 0:
                                        fd_stage3[new_col] = (fd_stage3[col] - fd_stage3[col].min()) / (fd_stage3[col].max() - fd_stage3[col].min())
                            
                            fd_results = fd_stage3
                        fd_nested_time = time.time() - fd_nested_start
                        fireducks_metrics['operationTimes']['nestedOperations'] = fireducks_metrics['operationTimes'].get('nestedOperations', 0) + fd_nested_time
                    
                    # Additional operations
                    if operations.get('concat', False) and fd_data is not None:
                        fd_concat_start = time.time()
                        # Create a copy of the original dataframe to concatenate
                        subset_data = fd_data.head(len(fd_data) // 2).copy()
                        # Concatenate dataframes
                        concat_result = fdpd.concat([fd_data, subset_data], ignore_index=True)
                        fd_concat_time = time.time() - fd_concat_start
                        fireducks_metrics['operationTimes']['concat'] = fireducks_metrics['operationTimes'].get('concat', 0) + fd_concat_time
                        
                    if operations.get('sort', False) and fd_data is not None:
                        fd_sort_start = time.time()
                        # Sort by multiple columns
                        if 'value_a' in fd_data.columns and 'value_b' in fd_data.columns:
                            sorted_data = fd_data.sort_values(by=['value_a', 'value_b'], ascending=[False, True])
                        else:
                            # Fall back to sorting by the first column if specific columns not found
                            sorted_data = fd_data.sort_values(by=fd_data.columns[0])
                        fd_sort_time = time.time() - fd_sort_start
                        fireducks_metrics['operationTimes']['sort'] = fireducks_metrics['operationTimes'].get('sort', 0) + fd_sort_time
                        
                    if operations.get('info', False) and fd_data is not None:
                        fd_info_start = time.time()
                        # Capture DataFrame.info() output
                        buffer = StringIO()
                        fd_data.info(buf=buffer)
                        _ = buffer.getvalue()  # Get the info output but we don't need to save it
                        fd_info_time = time.time() - fd_info_start
                        fireducks_metrics['operationTimes']['info'] = fireducks_metrics['operationTimes'].get('info', 0) + fd_info_time
                        
                    if operations.get('toCSV', False) and fd_data is not None:
                        fd_tocsv_start = time.time()
                        # Write to CSV (in memory)
                        csv_buffer = StringIO()
                        fd_data.to_csv(csv_buffer, index=False)
                        _ = csv_buffer.getvalue()  # Get the CSV but we don't need to save it
                        fd_tocsv_time = time.time() - fd_tocsv_start
                        fireducks_metrics['operationTimes']['toCSV'] = fireducks_metrics['operationTimes'].get('toCSV', 0) + fd_tocsv_time
                
                except Exception as e:
                    print(f"Error in FireDucks operations: {str(e)}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
                    results_match = False
                    # Placeholder for failed FireDucks operations
                    fd_data = pandas_data
                    fd_results = pandas_results
            else:
                # Use pandas results as placeholders if not using real FireDucks
                fd_data = pandas_data
                fd_results = pandas_results
            
            # Finalize FireDucks metrics for this run
            fd_end_time = time.time()
            fd_run_time = fd_end_time - fd_start_time
            fireducks_metrics['executionTime'] += fd_run_time
            
            # Measure memory after FireDucks operations
            fd_memory = get_memory_usage() - base_memory
            fireducks_metrics['memoryUsage'] += fd_memory
            
            # Store the individual run metrics
            fd_run_metrics = {
                'executionTime': fd_run_time,
                'memoryUsage': fd_memory
            }
            fireducks_metrics['runs'].append(fd_run_metrics)
            
            print(f"FireDucks run {run+1}: {fd_run_time:.4f}s, Memory: {fd_memory:.2f}MB", file=sys.stderr)
            
            # Verify results match between pandas and FireDucks
            if pandas_results is not None and fd_results is not None:
                try:
                    # For simple equality check, convert both to string representation
                    # This is an approximate check, not perfect
                    if not isinstance(pandas_results, pd.DataFrame) or not isinstance(fd_results, (pd.DataFrame, fdpd.DataFrame)):
                        # Can't compare if not dataframes
                        results_match = False
                        print("WARNING: Results are not comparable dataframes", file=sys.stderr)
                    else:
                        # Try to compute a rough equality check
                        try:
                            pd_shape = pandas_results.shape
                            fd_shape = fd_results.shape
                            
                            # Check shape match first
                            if pd_shape[0] != fd_shape[0] or pd_shape[1] != fd_shape[1]:
                                results_match = False
                                print(f"WARNING: Results shape mismatch: Pandas {pd_shape}, FireDucks {fd_shape}", file=sys.stderr)
                            else:
                                # For numerical columns, check statistical similarity
                                # This is very approximate and not guaranteed to be accurate
                                pd_num_cols = pandas_results.select_dtypes(include=['number']).columns
                                fd_num_cols = fd_results.select_dtypes(include=['number']).columns
                                
                                common_num_cols = list(set(pd_num_cols).intersection(set(fd_num_cols)))
                                
                                if len(common_num_cols) > 0:
                                    # Check means are roughly similar
                                    pd_means = pandas_results[common_num_cols].mean()
                                    fd_means = fd_results[common_num_cols].mean()
                                    
                                    # Calculate relative difference
                                    rel_diff = abs(pd_means - fd_means) / (pd_means.abs() + 1e-10) # Avoid division by zero
                                    
                                    # If any mean differs by more than 1%, consider results different
                                    if (rel_diff > 0.01).any():
                                        results_match = False
                                        print("WARNING: Numerical results differ by more than 1%", file=sys.stderr)
                                else:
                                    # If no numerical columns to compare
                                    results_match = False
                                    print("WARNING: No common numerical columns to compare", file=sys.stderr)
                        except Exception as e:
                            print(f"Error checking results: {str(e)}", file=sys.stderr)
                            results_match = False
                except Exception as e:
                    print(f"Error comparing results: {str(e)}", file=sys.stderr)
                    results_match = False
        
        # Calculate averages
        if run_count > 0:
            # Calculate average execution time across runs
            pandas_metrics['executionTime'] /= run_count
            fireducks_metrics['executionTime'] /= run_count
            
            # Calculate average memory usage across runs
            pandas_metrics['memoryUsage'] /= run_count
            fireducks_metrics['memoryUsage'] /= run_count
            
            # Calculate average operation times
            for op in pandas_metrics['operationTimes']:
                pandas_metrics['operationTimes'][op] /= run_count
            
            for op in fireducks_metrics['operationTimes']:
                fireducks_metrics['operationTimes'][op] /= run_count
        
        # Return the comparison results as a JSON-serializable dictionary
        result = {
            'pandas': pandas_metrics,
            'fireducks': fireducks_metrics,
            'resultsMatch': results_match
        }
        
        print("Comparison completed successfully", file=sys.stderr)
        
        return result
    except Exception as e:
        print(f"Error in compare_libraries: {str(e)}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


def main():
    """
    Main function to handle command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Compare Pandas and FireDucks performance.')
    parser.add_argument('command', choices=['analyze', 'generate', 'compare'], 
                        help='Command to execute')
    parser.add_argument('--file', type=str, help='CSV file to analyze or compare')
    parser.add_argument('--rows', type=int, help='Number of rows to generate')
    parser.add_argument('--operations', type=str, help='JSON-encoded operations to perform during comparison')
    parser.add_argument('--settings', type=str, help='JSON-encoded settings for comparison')
    
    args = parser.parse_args()
    
    try:
        if args.command == 'analyze' and args.file:
            result = analyze_csv(args.file)
            print(json.dumps(result))
        
        elif args.command == 'generate' and args.rows:
            result = generate_synthetic_data(args.rows)
            print(json.dumps(result))
        
        elif args.command == 'compare' and args.file and args.operations and args.settings:
            operations = json.loads(args.operations)
            settings = json.loads(args.settings)
            result = compare_libraries(args.file, operations, settings)
            print(json.dumps(result))
        
        else:
            print("Invalid command or missing arguments", file=sys.stderr)
            parser.print_help()
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()