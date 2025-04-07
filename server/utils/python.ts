import { spawn } from 'child_process';

/**
 * Execute a Python script with the given arguments
 * @param scriptPath Path to the Python script
 * @param args Arguments to pass to the script
 * @returns Promise that resolves with the script's stdout
 */
export function execPython(scriptPath: string, args: string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    // Use the python3 command to ensure we use the correct Python version
    const process = spawn('python3', [scriptPath, ...args]);
    
    let stdout = '';
    let stderr = '';
    
    // Collect stdout
    process.stdout.on('data', (data) => {
      stdout += data.toString();
    });
    
    // Collect stderr
    process.stderr.on('data', (data) => {
      stderr += data.toString();
      console.error(`Python stderr: ${data}`);
    });
    
    // Handle process exit
    process.on('close', (code) => {
      if (code === 0) {
        resolve(stdout.trim());
      } else {
        reject(new Error(`Python process exited with code ${code}: ${stderr}`));
      }
    });
    
    // Handle process error
    process.on('error', (err) => {
      reject(new Error(`Failed to spawn Python process: ${err.message}`));
    });
  });
}
