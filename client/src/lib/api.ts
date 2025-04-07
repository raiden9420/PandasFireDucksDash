import { apiRequest } from '@/lib/queryClient';
import { Dataset, Comparison, GenerateData, RunComparison } from '@shared/schema';

// Get all datasets
export async function getAllDatasets(): Promise<Dataset[]> {
  const response = await apiRequest('GET', '/api/datasets');
  return response.json();
}

// Get dataset by ID
export async function getDataset(id: number): Promise<Dataset> {
  const response = await apiRequest('GET', `/api/datasets/${id}`);
  return response.json();
}

// Upload CSV file
export async function uploadFile(formData: FormData): Promise<Dataset> {
  // We need to use native fetch for FormData
  const response = await fetch('/api/datasets/upload', {
    method: 'POST',
    body: formData,
    credentials: 'include',
  });
  
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || response.statusText);
  }
  
  return response.json();
}

// Generate synthetic data
export async function generateSyntheticData(data: GenerateData): Promise<Dataset> {
  const response = await apiRequest('POST', '/api/datasets/generate', data);
  return response.json();
}

// Run comparison
export async function runComparison(data: RunComparison): Promise<Comparison> {
  const response = await apiRequest('POST', '/api/comparison/run', data);
  return response.json();
}
