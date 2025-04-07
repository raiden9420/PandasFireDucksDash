import { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Sidebar } from '@/components/ui/sidebar';
import { ComparisonPanel } from './comparison-panel';
import { useComparison } from '@/context/comparison-context';
import { getAllDatasets } from '@/lib/api';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { LiveComparisonPanel } from '@/components/ui/live-comparison-panel';
import { Comparison } from '@shared/schema';

export default function Dashboard() {
  const {
    currentDataset,
    setCurrentDataset,
    comparisonResults,
    operations,
    setComparisonResults,
    multiRunEnabled,
    runCount
  } = useComparison();
  
  const [activeTab, setActiveTab] = useState<string>("standard");
  
  // Fetch datasets on mount
  const { data: datasets } = useQuery({
    queryKey: ['/api/datasets'],
    queryFn: getAllDatasets,
  });
  
  // Set first dataset as current if none selected
  useEffect(() => {
    if (!currentDataset && datasets && datasets.length > 0) {
      setCurrentDataset(datasets[0]);
    }
  }, [datasets, currentDataset, setCurrentDataset]);
  
  const pandasMetrics = comparisonResults?.pandasMetrics || null;
  const fireducksMetrics = comparisonResults?.fireducksMetrics || null;
  const sampleData = Array.isArray(currentDataset?.sampleData) ? currentDataset.sampleData : [];
  const resultsMatch = comparisonResults?.resultsMatch || false;
  
  // Handle completion of a live comparison
  const handleLiveComparisonComplete = (comparison: Comparison) => {
    if (comparison) {
      setComparisonResults(comparison);
      setActiveTab("standard");
    }
  };
  
  return (
    <div className="flex h-screen overflow-hidden bg-gray-100">
      <Sidebar />
      
      <main className="flex-1 flex flex-col overflow-hidden">
        <header className="bg-white border-b border-gray-200 py-3 px-6">
          <div className="flex flex-col">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-gray-900">Performance Dashboard</h2>
              {currentDataset && (
                <div className="text-sm text-gray-500">
                  Current dataset: <span className="font-medium">{currentDataset.name}</span> 
                  ({currentDataset.rows.toLocaleString()} rows × {currentDataset.columns} columns)
                </div>
              )}
            </div>
            
            <div className="mt-2 text-xs text-gray-600 bg-gray-50 p-2 rounded border border-gray-200">
              <p className="font-medium mb-1">Understanding the Metrics:</p>
              <ul className="list-disc pl-4 space-y-1">
                <li><strong>Total Execution Time</strong>: Combined time for all operations from start to finish</li>
                <li><strong>Peak Memory Usage</strong>: Maximum memory consumed during benchmark execution</li>
                <li><strong>Per-Operation Times</strong>: Individual timing for each specific operation (shown in charts)</li>
              </ul>
            </div>
          </div>
          
          <div className="mt-4">
            <Tabs value={activeTab} onValueChange={setActiveTab}>
              <TabsList className="grid w-full grid-cols-2 max-w-md mx-auto">
                <TabsTrigger value="standard">Standard Comparison</TabsTrigger>
                <TabsTrigger value="live">Live Comparison</TabsTrigger>
              </TabsList>
            </Tabs>
          </div>
        </header>
        
        {activeTab === "standard" ? (
          <div className="flex-1 flex overflow-hidden">
            <div className="w-1/2 border-r border-gray-200 overflow-y-auto">
              <ComparisonPanel
                title="Pandas"
                metrics={pandasMetrics}
                versionBadge={pandasMetrics?.version || "v1.5.3"}
                versionColor="rgb(0, 122, 255)"
                chartColor="rgb(0, 122, 255)"
                sampleData={sampleData}
                dataset={currentDataset}
              />
            </div>
            
            <div className="w-1/2 overflow-y-auto">
              <ComparisonPanel
                title="FireDucks"
                metrics={fireducksMetrics}
                versionBadge={fireducksMetrics?.version || "v0.4.2"}
                versionColor="rgb(255, 149, 0)"
                chartColor="rgb(255, 149, 0)"
                sampleData={sampleData}
                comparisonMetrics={pandasMetrics}
                resultsMatch={resultsMatch}
                dataset={currentDataset}
              />
            </div>
          </div>
        ) : (
          <div className="flex-1 p-6 overflow-y-auto">
            {currentDataset && (
              <LiveComparisonPanel
                dataset={currentDataset}
                operations={operations}
                settings={{
                  multiRunEnabled,
                  runCount
                }}
                onComplete={handleLiveComparisonComplete}
              />
            )}
          </div>
        )}
      </main>
    </div>
  );
}
