import { useState } from 'react';
import { useComparison } from '@/context/comparison-context';
import { Operations } from '@shared/schema';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Checkbox } from '@/components/ui/checkbox';
import { Separator } from '@/components/ui/separator';
import { uploadFile, generateSyntheticData, runComparison } from '@/lib/api';
import { useToast } from '@/hooks/use-toast';
import { 
  UploadCloud,
  Database, 
  ArrowRightCircle,
  Loader2
} from 'lucide-react';

export function Sidebar() {
  const { toast } = useToast();
  const {
    currentDataset,
    setCurrentDataset,
    operations,
    setOperations,
    multiRunEnabled,
    setMultiRunEnabled,
    runCount,
    setRunCount,
    setComparisonResults,
    isGeneratingData,
    setIsGeneratingData,
    isRunningComparison,
    setIsRunningComparison,
    isUploadingFile,
    setIsUploadingFile
  } = useComparison();
  
  const [numRows, setNumRows] = useState<number>(10000);
  
  // Handle operations change
  const handleOperationChange = (operation: keyof Operations) => {
    setOperations({
      ...operations,
      [operation]: !operations[operation]
    });
  };
  
  // Handle file upload
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    
    // Validate file type
    if (!file.name.endsWith('.csv')) {
      toast({
        title: "Invalid file format",
        description: "Please upload a CSV file",
        variant: "destructive"
      });
      return;
    }
    
    // Validate file size (50MB max)
    if (file.size > 50 * 1024 * 1024) {
      toast({
        title: "File too large",
        description: "Maximum file size is 50MB",
        variant: "destructive"
      });
      return;
    }
    
    setIsUploadingFile(true);
    
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const dataset = await uploadFile(formData);
      setCurrentDataset(dataset);
      
      toast({
        title: "File uploaded successfully",
        description: `Loaded ${dataset.name} with ${dataset.rows.toLocaleString()} rows`
      });
    } catch (error) {
      console.error('Error uploading file:', error);
      toast({
        title: "Upload failed",
        description: error instanceof Error ? error.message : "Failed to upload file",
        variant: "destructive"
      });
    } finally {
      setIsUploadingFile(false);
      
      // Reset file input
      e.target.value = '';
    }
  };
  
  // Handle synthetic data generation
  const handleGenerateData = async () => {
    setIsGeneratingData(true);
    
    try {
      const dataset = await generateSyntheticData({ numRows });
      setCurrentDataset(dataset);
      
      toast({
        title: "Data generated successfully",
        description: `Generated ${dataset.rows.toLocaleString()} rows of synthetic data`
      });
    } catch (error) {
      console.error('Error generating data:', error);
      toast({
        title: "Generation failed",
        description: error instanceof Error ? error.message : "Failed to generate data",
        variant: "destructive"
      });
    } finally {
      setIsGeneratingData(false);
    }
  };
  
  // Handle comparison run
  const handleRunComparison = async () => {
    if (!currentDataset) {
      toast({
        title: "No dataset",
        description: "Please upload a CSV file or generate synthetic data first",
        variant: "destructive"
      });
      return;
    }
    
    setIsRunningComparison(true);
    
    try {
      const result = await runComparison({
        datasetId: currentDataset.id,
        operations,
        settings: {
          multiRunEnabled,
          runCount
        }
      });
      
      setComparisonResults(result);
      
      toast({
        title: "Comparison completed",
        description: "Performance comparison results are ready"
      });
    } catch (error) {
      console.error('Error running comparison:', error);
      toast({
        title: "Comparison failed",
        description: error instanceof Error ? error.message : "Failed to run comparison",
        variant: "destructive"
      });
    } finally {
      setIsRunningComparison(false);
    }
  };
  
  return (
    <aside className="w-72 border-r border-gray-200 bg-white flex flex-col overflow-hidden">
      <div className="p-4 border-b border-gray-200">
        <h1 className="text-xl font-semibold text-gray-900">Performance Dashboard</h1>
        <p className="text-sm text-gray-500 mt-1">Compare Pandas vs FireDucks</p>
      </div>

      <div className="p-4 border-b border-gray-200">
        <h2 className="text-sm font-medium text-gray-700 mb-3">Data Source</h2>
        
        <div className="mb-4">
          <Label className="text-xs mb-1">Upload CSV File</Label>
          <div className="relative border border-gray-300 rounded-md overflow-hidden">
            <Input 
              type="file" 
              id="file-upload" 
              className="sr-only"
              accept=".csv"
              onChange={handleFileUpload}
              disabled={isUploadingFile || isGeneratingData}
            />
            <Label 
              htmlFor="file-upload" 
              className="flex items-center justify-center py-2 px-3 cursor-pointer hover:bg-gray-50 transition-colors"
            >
              {isUploadingFile ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin text-gray-500" />
              ) : (
                <UploadCloud className="h-4 w-4 mr-2 text-gray-500" />
              )}
              <span className="text-sm text-gray-700">
                {isUploadingFile ? "Uploading..." : "Choose file"}
              </span>
            </Label>
          </div>
          <p className="text-xs text-gray-500 mt-1">Max file size: 50MB</p>
        </div>
        
        <Separator className="my-4">
          <span className="bg-white px-2 text-xs text-gray-500">OR</span>
        </Separator>
        
        <div>
          <Label className="text-xs mb-1">Generate Synthetic Data</Label>
          <div className="flex items-center">
            <Input 
              type="number" 
              value={numRows} 
              min={1000}
              max={1000000}
              className="text-sm"
              placeholder="Number of rows"
              onChange={(e) => setNumRows(parseInt(e.target.value) || 10000)}
              disabled={isGeneratingData || isUploadingFile}
            />
            <Button 
              variant="outline" 
              size="sm" 
              className="ml-2 h-9"
              onClick={handleGenerateData}
              disabled={isGeneratingData || isUploadingFile}
            >
              {isGeneratingData ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <Database className="h-4 w-4 mr-2" />
              )}
              {isGeneratingData ? "Generating..." : "Generate"}
            </Button>
          </div>
        </div>
      </div>

      <div className="p-4 border-b border-gray-200 overflow-y-auto">
        <h2 className="text-sm font-medium text-gray-700 mb-3">Operations</h2>
        
        <div className="space-y-2">
          <h3 className="text-xs font-medium text-gray-600 mt-2 mb-1">Basic Operations</h3>
          <div className="flex items-center">
            <Checkbox 
              id="op-load" 
              checked={operations.load}
              onCheckedChange={() => handleOperationChange('load')}
            />
            <Label htmlFor="op-load" className="ml-2 text-sm text-gray-700">Load CSV</Label>
          </div>
          <div className="flex items-center">
            <Checkbox 
              id="op-groupby" 
              checked={operations.groupby}
              onCheckedChange={() => handleOperationChange('groupby')}
            />
            <Label htmlFor="op-groupby" className="ml-2 text-sm text-gray-700">Group By</Label>
          </div>
          <div className="flex items-center">
            <Checkbox 
              id="op-merge" 
              checked={operations.merge}
              onCheckedChange={() => handleOperationChange('merge')}
            />
            <Label htmlFor="op-merge" className="ml-2 text-sm text-gray-700">Merge</Label>
          </div>
          <div className="flex items-center">
            <Checkbox 
              id="op-filter" 
              checked={operations.filter}
              onCheckedChange={() => handleOperationChange('filter')}
            />
            <Label htmlFor="op-filter" className="ml-2 text-sm text-gray-700">Filter</Label>
          </div>
          <div className="flex items-center">
            <Checkbox 
              id="op-rolling" 
              checked={operations.rolling}
              onCheckedChange={() => handleOperationChange('rolling')}
            />
            <Label htmlFor="op-rolling" className="ml-2 text-sm text-gray-700">Rolling Average</Label>
          </div>

          <h3 className="text-xs font-medium text-gray-600 mt-4 mb-1">Advanced Operations</h3>
          <div className="flex items-center">
            <Checkbox 
              id="op-pivotTable" 
              checked={operations.pivotTable}
              onCheckedChange={() => handleOperationChange('pivotTable')}
            />
            <Label htmlFor="op-pivotTable" className="ml-2 text-sm text-gray-700">Pivot Table</Label>
          </div>
          <div className="flex items-center">
            <Checkbox 
              id="op-complexAggregation" 
              checked={operations.complexAggregation}
              onCheckedChange={() => handleOperationChange('complexAggregation')}
            />
            <Label htmlFor="op-complexAggregation" className="ml-2 text-sm text-gray-700">Complex Aggregation</Label>
          </div>
          <div className="flex items-center">
            <Checkbox 
              id="op-windowFunctions" 
              checked={operations.windowFunctions}
              onCheckedChange={() => handleOperationChange('windowFunctions')}
            />
            <Label htmlFor="op-windowFunctions" className="ml-2 text-sm text-gray-700">Window Functions</Label>
          </div>
          <div className="flex items-center">
            <Checkbox 
              id="op-stringManipulation" 
              checked={operations.stringManipulation}
              onCheckedChange={() => handleOperationChange('stringManipulation')}
            />
            <Label htmlFor="op-stringManipulation" className="ml-2 text-sm text-gray-700">String Manipulation</Label>
          </div>
          <div className="flex items-center">
            <Checkbox 
              id="op-nestedOperations" 
              checked={operations.nestedOperations}
              onCheckedChange={() => handleOperationChange('nestedOperations')}
            />
            <Label htmlFor="op-nestedOperations" className="ml-2 text-sm text-gray-700">Nested Operations</Label>
          </div>

          <h3 className="text-xs font-medium text-gray-600 mt-4 mb-1">Additional Operations</h3>
          <div className="flex items-center">
            <Checkbox 
              id="op-concat" 
              checked={operations.concat}
              onCheckedChange={() => handleOperationChange('concat')}
            />
            <Label htmlFor="op-concat" className="ml-2 text-sm text-gray-700">Concat DataFrames</Label>
          </div>
          <div className="flex items-center">
            <Checkbox 
              id="op-sort" 
              checked={operations.sort}
              onCheckedChange={() => handleOperationChange('sort')}
            />
            <Label htmlFor="op-sort" className="ml-2 text-sm text-gray-700">Sort Data</Label>
          </div>
          <div className="flex items-center">
            <Checkbox 
              id="op-info" 
              checked={operations.info}
              onCheckedChange={() => handleOperationChange('info')}
            />
            <Label htmlFor="op-info" className="ml-2 text-sm text-gray-700">Info (DataFrame.info)</Label>
          </div>
          <div className="flex items-center">
            <Checkbox 
              id="op-toCSV" 
              checked={operations.toCSV}
              onCheckedChange={() => handleOperationChange('toCSV')}
            />
            <Label htmlFor="op-toCSV" className="ml-2 text-sm text-gray-700">Export to CSV</Label>
          </div>
        </div>
      </div>

      <div className="p-4 border-b border-gray-200">
        <h2 className="text-sm font-medium text-gray-700 mb-3">Run Settings</h2>
        
        <div className="mb-3">
          <div className="flex items-center justify-between">
            <Label htmlFor="multi-run" className="text-sm text-gray-700">Enable Multi-Run Averaging</Label>
            <Switch 
              id="multi-run"
              checked={multiRunEnabled}
              onCheckedChange={setMultiRunEnabled}
            />
          </div>
          <p className="text-xs text-gray-500 mt-1">Run multiple times and average results</p>
        </div>
        
        <div className={`transition-opacity duration-300 ${multiRunEnabled ? '' : 'opacity-50'}`}>
          <Label className="text-xs mb-1">Number of Runs</Label>
          <Input 
            type="number" 
            value={runCount} 
            min={2}
            max={10}
            className="text-sm"
            onChange={(e) => setRunCount(parseInt(e.target.value) || 3)}
            disabled={!multiRunEnabled}
          />
        </div>
      </div>

      <div className="p-4 mt-auto">
        <Button 
          className="w-full bg-[#007AFF] hover:bg-blue-600"
          onClick={handleRunComparison}
          disabled={isRunningComparison || !currentDataset}
        >
          {isRunningComparison ? (
            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
          ) : (
            <ArrowRightCircle className="h-4 w-4 mr-2" />
          )}
          {isRunningComparison ? "Running..." : "Run Comparison"}
        </Button>
      </div>
    </aside>
  );
}
