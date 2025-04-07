import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Dataset, Operations, Comparison } from '@shared/schema';
import { runComparison } from '@/lib/api';
import { CheckCircle2, AlertTriangle, Clock } from 'lucide-react';

interface LiveComparisonPanelProps {
  dataset: Dataset;
  operations: Operations;
  settings: {
    multiRunEnabled: boolean;
    runCount: number;
  };
  onComplete: (result: Comparison) => void;
}

interface ProgressState {
  stage: 'idle' | 'starting' | 'pandas' | 'fireducks' | 'comparing' | 'complete' | 'error';
  progress: number;
  message: string;
  error?: string;
}

export function LiveComparisonPanel({ 
  dataset, 
  operations, 
  settings,
  onComplete
}: LiveComparisonPanelProps) {
  const [progress, setProgress] = useState<ProgressState>({
    stage: 'idle',
    progress: 0,
    message: 'Ready to run comparison'
  });
  
  const [comparison, setComparison] = useState<Comparison | null>(null);
  
  // Run the comparison when the component mounts
  useEffect(() => {
    const runLiveComparison = async () => {
      // Reset state
      setProgress({
        stage: 'starting',
        progress: 5,
        message: 'Initializing comparison...'
      });
      
      try {
        // Start pandas execution
        setProgress({
          stage: 'pandas',
          progress: 20,
          message: 'Running Pandas operations...'
        });
        
        // Short delay for UI feedback
        await new Promise(resolve => setTimeout(resolve, 500));
        
        // Start fireducks execution
        setProgress({
          stage: 'fireducks',
          progress: 60,
          message: 'Running FireDucks operations...'
        });
        
        // Short delay for UI feedback
        await new Promise(resolve => setTimeout(resolve, 500));
        
        // Start comparison
        setProgress({
          stage: 'comparing',
          progress: 85,
          message: 'Comparing results...'
        });
        
        // Run the actual comparison
        const result = await runComparison({
          datasetId: dataset.id,
          operations,
          settings
        });
        
        // Save the result
        setComparison(result);
        
        // Complete progress
        setProgress({
          stage: 'complete',
          progress: 100,
          message: 'Comparison completed successfully'
        });
        
        // Notify parent
        onComplete(result);
      } catch (error) {
        console.error('Error running live comparison:', error);
        setProgress({
          stage: 'error',
          progress: 100,
          message: 'Error running comparison',
          error: error instanceof Error ? error.message : 'Unknown error occurred'
        });
      }
    };
    
    // Start the comparison immediately
    runLiveComparison();
  }, [dataset, operations, settings, onComplete]);
  
  return (
    <Card className="w-full max-w-4xl mx-auto">
      <CardHeader>
        <CardTitle>Live Comparison</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <div className="text-sm font-medium">
                {progress.stage === 'complete' ? (
                  <span className="flex items-center text-green-600">
                    <CheckCircle2 className="h-4 w-4 mr-1.5" />
                    {progress.message}
                  </span>
                ) : progress.stage === 'error' ? (
                  <span className="flex items-center text-red-600">
                    <AlertTriangle className="h-4 w-4 mr-1.5" />
                    {progress.message}
                  </span>
                ) : (
                  <span className="flex items-center text-blue-600">
                    <Clock className="h-4 w-4 mr-1.5" />
                    {progress.message}
                  </span>
                )}
              </div>
              <div className="text-sm text-gray-500">
                {progress.progress}%
              </div>
            </div>
            <Progress value={progress.progress} className="h-2" />
          </div>
          
          {progress.error && (
            <div className="p-4 bg-red-50 border border-red-200 rounded-md text-red-800 text-sm">
              <p className="font-semibold">Error Details:</p>
              <p className="mt-1">{progress.error}</p>
            </div>
          )}
          
          {progress.stage === 'complete' && comparison && (
            <div className="grid grid-cols-2 gap-6">
              <div className="bg-gray-50 rounded-md p-4">
                <h3 className="text-sm font-medium flex items-center mb-2">
                  <span className="inline-flex items-center justify-center w-5 h-5 bg-[#007AFF] text-white rounded-full text-xs mr-2">P</span>
                  Pandas Results
                </h3>
                <div className="space-y-3 text-sm">
                  <div>
                    <span className="text-gray-500">Execution Time: </span>
                    <span className="font-medium">{(comparison.pandasMetrics.executionTime * 1000).toFixed(2)} ms</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Memory Usage: </span>
                    <span className="font-medium">{comparison.pandasMetrics.memoryUsage.toFixed(2)} MB</span>
                  </div>
                </div>
              </div>
              
              <div className="bg-gray-50 rounded-md p-4">
                <h3 className="text-sm font-medium flex items-center mb-2">
                  <span className="inline-flex items-center justify-center w-5 h-5 bg-[#FF9500] text-white rounded-full text-xs mr-2">F</span>
                  FireDucks Results
                </h3>
                <div className="space-y-3 text-sm">
                  <div>
                    <span className="text-gray-500">Execution Time: </span>
                    <span className="font-medium">{(comparison.fireducksMetrics.executionTime * 1000).toFixed(2)} ms</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Memory Usage: </span>
                    <span className="font-medium">{comparison.fireducksMetrics.memoryUsage.toFixed(2)} MB</span>
                  </div>
                </div>
              </div>
            </div>
          )}
          
          {(progress.stage === 'pandas' || progress.stage === 'fireducks' || progress.stage === 'comparing') && (
            <div className="flex items-center justify-center py-4">
              <div className="animate-pulse flex space-x-4">
                <div className="h-12 w-12 rounded-full bg-blue-200"></div>
                <div className="space-y-2">
                  <div className="h-4 bg-blue-200 rounded w-36"></div>
                  <div className="h-4 bg-blue-200 rounded w-24"></div>
                </div>
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
