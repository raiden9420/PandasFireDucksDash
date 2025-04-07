import fs from 'fs';
import path from 'path';
import { Operations, Comparison } from '@shared/schema';
import { storage } from '../storage';
import { execPython } from '../utils/python';

// Temporary directory for storing files
const TEMP_DIR = path.join(process.cwd(), 'temp');
// Python script directory
const PYTHON_DIR = path.join(process.cwd(), 'server', 'controllers', 'python');

// Initialize directories
if (!fs.existsSync(TEMP_DIR)) {
  fs.mkdirSync(TEMP_DIR, { recursive: true });
}
if (!fs.existsSync(PYTHON_DIR)) {
  fs.mkdirSync(PYTHON_DIR, { recursive: true });
}

// Path to the Python analyzer script
const ANALYZER_SCRIPT = path.join(PYTHON_DIR, 'analyzer.py');

/**
 * Analyze a CSV file to get row count, column count, and sample data
 */
export async function analyzeCSV(filePath: string): Promise<{ rows: number, columns: number, sample_data: any[] }> {
  try {
    // Ensure the script exists
    if (!fs.existsSync(ANALYZER_SCRIPT)) {
      throw new Error(`Python analyzer script not found at ${ANALYZER_SCRIPT}`);
    }
    
    console.log('Using analyzer script: Enhanced FireDucks implementation with advanced operations');
    
    const args = [
      'analyze',
      '--file', filePath
    ];
    
    const result = await execPython(ANALYZER_SCRIPT, args);
    return JSON.parse(result);
  } catch (error) {
    console.error('Error analyzing CSV:', error);
    throw error;
  }
}

/**
 * Generate synthetic data with the specified number of rows
 */
export async function generateSyntheticData(numRows: number): Promise<{ 
  rows: number, 
  columns: number, 
  sample_data: any[],
  filePath: string | null
}> {
  try {
    // Ensure the script exists
    if (!fs.existsSync(ANALYZER_SCRIPT)) {
      throw new Error(`Python analyzer script not found at ${ANALYZER_SCRIPT}`);
    }
    
    console.log('Using analyzer script: Enhanced FireDucks implementation with advanced operations');
    console.log(`Generating synthetic data with ${numRows} rows...`);
    
    const args = [
      'generate',
      '--rows', numRows.toString()
    ];
    
    const result = await execPython(ANALYZER_SCRIPT, args);
    const data = JSON.parse(result);
    let filePath: string | null = null;
    
    // Save the file to disk if it contains file_content
    if (data.file_content) {
      filePath = path.join(TEMP_DIR, `synthetic_${Date.now()}.csv`);
      
      // Decode base64 content
      const buffer = Buffer.from(data.file_content, 'base64');
      fs.writeFileSync(filePath, buffer);
      
      // Remove file_content to avoid sending large data over the wire
      delete data.file_content;
      
      console.log(`Synthetic data saved to ${filePath}`);
    }
    
    // Return with the proper filePath field explicitly set
    return {
      rows: data.rows,
      columns: data.columns,
      sample_data: data.sample_data,
      filePath
    };
  } catch (error) {
    console.error('Error generating synthetic data:', error);
    throw error;
  }
}

/**
 * Run a performance comparison between Pandas and FireDucks
 */
export async function runAnalysis(
  datasetId: number, 
  operations: Operations, 
  settings: { multiRunEnabled: boolean, runCount: number }
): Promise<Comparison> {
  try {
    // Get the dataset
    const dataset = await storage.getDataset(datasetId);
    if (!dataset) {
      throw new Error(`Dataset with ID ${datasetId} not found`);
    }
    
    // Get the file path for the dataset
    if (!dataset.filePath) {
      // For synthetic datasets, we need to generate the CSV file on-the-fly
      const tempFilePath = path.join(TEMP_DIR, `dataset_${datasetId}.csv`);
      
      // If temp file doesn't exist, we need to request the synthetic data to be generated
      if (!fs.existsSync(tempFilePath)) {
        // Here we would theoretically regenerate the dataset - for now we'll throw an error
        throw new Error(`Synthetic dataset ${datasetId} has no associated file`);
      }
      
      // Use the temp file path
      dataset.filePath = tempFilePath;
    }
    
    // Check if the file exists
    if (!fs.existsSync(dataset.filePath)) {
      throw new Error(`File not found for dataset ID ${datasetId}: ${dataset.filePath}`);
    }
    
    // Ensure the script exists
    if (!fs.existsSync(ANALYZER_SCRIPT)) {
      throw new Error(`Python analyzer script not found at ${ANALYZER_SCRIPT}`);
    }
    
    console.log('Using analyzer script: Enhanced FireDucks implementation with advanced operations');
    console.log(`Running comparison for dataset ${datasetId}...`);
    
    // Convert operations to JSON
    const operationsJson = JSON.stringify(operations);
    
    // Convert settings to JSON
    const settingsJson = JSON.stringify(settings);
    
    const args = [
      'compare',
      '--file', dataset.filePath, // Use the dataset's filePath after our validation
      '--operations', operationsJson,
      '--settings', settingsJson
    ];
    
    const result = await execPython(ANALYZER_SCRIPT, args);
    const rawResult = JSON.parse(result);
    
    // Map the raw Python response to the expected Comparison format
    const comparison: Comparison = {
      pandasMetrics: rawResult.pandas,
      fireducksMetrics: rawResult.fireducks,
      resultsMatch: rawResult.resultsMatch,
      datasetId: datasetId
    };
    
    console.log(`Comparison completed for dataset ${datasetId}`);
    
    return comparison;
  } catch (error) {
    console.error('Error running analysis:', error);
    throw error;
  }
}