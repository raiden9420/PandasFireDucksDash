import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { PerformanceChart } from "@/components/ui/charts";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { 
  ArrowDownIcon, 
  ArrowUpIcon, 
  CheckCircle2Icon 
} from "lucide-react";
import { Dataset, Metrics } from "@shared/schema";

interface ComparisonPanelProps {
  title: string;
  metrics: Metrics | null;
  versionBadge: string;
  versionColor: string;
  chartColor: string;
  sampleData: any[] | null;
  comparisonMetrics?: Metrics | null;
  resultsMatch?: boolean;
  dataset?: Dataset | null;
}

export function ComparisonPanel({
  title,
  metrics,
  versionBadge,
  versionColor,
  chartColor,
  sampleData,
  comparisonMetrics,
  resultsMatch,
  dataset
}: ComparisonPanelProps) {
  if (!metrics) {
    return (
      <div className="w-full p-5">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              {title}
              <Badge style={{ backgroundColor: versionColor }} variant="outline" className="text-white">
                {versionBadge}
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent className="flex items-center justify-center h-72">
            <p className="text-gray-500">Run a comparison to see results</p>
          </CardContent>
        </Card>
      </div>
    );
  }
  
  const formatTime = (seconds: number): string => {
    const ms = seconds * 1000;
    return ms < 1000 ? `${ms.toFixed(2)} ms` : `${(ms / 1000).toFixed(2)} s`;
  };
  
  const formatMemory = (mb: number): string => {
    return mb < 1000 ? `${mb.toFixed(2)} MB` : `${(mb / 1000).toFixed(2)} GB`;
  };
  
  const getPerformanceComparison = (current: number, comparison: number): { percentage: number, isBetter: boolean } => {
    const diff = comparison - current;
    const percentage = (Math.abs(diff) / comparison) * 100;
    return {
      percentage: Math.round(percentage),
      isBetter: diff > 0 // Current is better if difference is positive (comparison is slower)
    };
  };
  
  // Prepare comparison data if comparisonMetrics is provided
  let timeComparison;
  let memoryComparison;
  
  if (comparisonMetrics && metrics) {
    timeComparison = getPerformanceComparison(
      metrics.executionTime,
      comparisonMetrics.executionTime
    );
    
    memoryComparison = getPerformanceComparison(
      metrics.memoryUsage,
      comparisonMetrics.memoryUsage
    );
  }
  
  // Prepare column names from sample data
  const columns = sampleData && sampleData.length > 0 ? Object.keys(sampleData[0]) : [];
  
  return (
    <div className="w-full p-5">
      <Card className="mb-5">
        <CardHeader className="border-b">
          <CardTitle className="flex items-center justify-between">
            {title}
            <Badge style={{ backgroundColor: versionColor }} variant="outline" className="text-white">
              {versionBadge}
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="p-4">
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div className="bg-gray-100 rounded-md p-3">
              <div className="text-xs font-medium text-gray-500 mb-1">Total Execution Time</div>
              <div className="text-xl font-semibold text-gray-900">{formatTime(metrics.executionTime)}</div>
              <div className="text-xs text-gray-500 mt-1">Overall benchmark duration</div>
              {timeComparison && (
                <div className={`text-xs mt-1 flex items-center ${timeComparison.isBetter ? 'text-green-600' : 'text-red-600'}`}>
                  <span className="inline-block w-3 h-3 mr-1">
                    {timeComparison.isBetter ? <ArrowDownIcon size={12} /> : <ArrowUpIcon size={12} />}
                  </span>
                  <span>{timeComparison.percentage}% {timeComparison.isBetter ? 'faster' : 'slower'}</span>
                </div>
              )}
            </div>
            
            <div className="bg-gray-100 rounded-md p-3">
              <div className="text-xs font-medium text-gray-500 mb-1">Peak Memory Usage</div>
              <div className="text-xl font-semibold text-gray-900">{formatMemory(metrics.memoryUsage)}</div>
              <div className="text-xs text-gray-500 mt-1">Maximum memory consumed</div>
              {memoryComparison && (
                <div className={`text-xs mt-1 flex items-center ${memoryComparison.isBetter ? 'text-green-600' : 'text-red-600'}`}>
                  <span className="inline-block w-3 h-3 mr-1">
                    {memoryComparison.isBetter ? <ArrowDownIcon size={12} /> : <ArrowUpIcon size={12} />}
                  </span>
                  <span>{memoryComparison.percentage}% {memoryComparison.isBetter ? 'less' : 'more'} memory</span>
                </div>
              )}
            </div>
          </div>
          
          <div className="mb-2">
            <h3 className="text-sm font-medium mb-1">Operation-Specific Timings</h3>
            <p className="text-xs text-gray-500 mb-3">These show individual operation execution times in milliseconds</p>
          </div>
          
          <PerformanceChart 
            metrics={metrics} 
            color={chartColor} 
            label={title} 
          />
        </CardContent>
      </Card>
      
      <Card>
        <CardHeader className="border-b">
          <div className="flex items-center justify-between">
            <CardTitle>Results</CardTitle>
            {resultsMatch !== undefined && (
              <div className="flex items-center text-xs font-medium text-green-600">
                <CheckCircle2Icon className="h-4 w-4 mr-1" />
                <span>Results match</span>
              </div>
            )}
          </div>
        </CardHeader>
        <CardContent className="p-4 overflow-x-auto">
          {sampleData && sampleData.length > 0 ? (
            <Table>
              <TableHeader>
                <TableRow>
                  {columns.map((column, index) => (
                    <TableHead key={index} className="text-xs bg-gray-100">
                      {column}
                    </TableHead>
                  ))}
                </TableRow>
              </TableHeader>
              <TableBody>
                {sampleData.map((row, rowIndex) => (
                  <TableRow key={rowIndex} className={rowIndex % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                    {columns.map((column, colIndex) => (
                      <TableCell key={colIndex} className="text-sm text-gray-800">
                        {row[column]?.toString()}
                      </TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <p className="text-gray-500 text-center py-4">No data available</p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
