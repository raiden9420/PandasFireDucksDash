import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { PerformanceChart } from "@/components/ui/charts";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import React from "react";

// Import from shared schema
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
  // Derive performance change compared to the other metrics
  const getSpeedupText = () => {
    if (!metrics || !comparisonMetrics || !comparisonMetrics.executionTime || !metrics.executionTime) {
      return null;
    }
    
    const ratio = comparisonMetrics.executionTime / metrics.executionTime;
    const percentage = ((ratio - 1) * 100).toFixed(1);
    
    if (ratio > 1) {
      return <Badge className="bg-green-100 text-green-800 hover:bg-green-100">{percentage}% faster</Badge>;
    } else if (ratio < 1) {
      return <Badge className="bg-orange-100 text-orange-800 hover:bg-orange-100">{Math.abs(Number(percentage))}% slower</Badge>;
    } else {
      return <Badge className="bg-gray-100 text-gray-800 hover:bg-gray-100">Same speed</Badge>;
    }
  };
  
  const getMemoryComparisonText = () => {
    if (!metrics || !comparisonMetrics || !comparisonMetrics.memoryUsage || !metrics.memoryUsage) {
      return null;
    }
    
    const ratio = comparisonMetrics.memoryUsage / metrics.memoryUsage;
    const percentage = ((ratio - 1) * 100).toFixed(1);
    
    if (ratio > 1) {
      return <Badge className="bg-green-100 text-green-800 hover:bg-green-100">{percentage}% less memory</Badge>;
    } else if (ratio < 1) {
      return <Badge className="bg-orange-100 text-orange-800 hover:bg-orange-100">{Math.abs(Number(percentage))}% more memory</Badge>;
    } else {
      return <Badge className="bg-gray-100 text-gray-800 hover:bg-gray-100">Same memory usage</Badge>;
    }
  };
  
  return (
    <div className="p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xl font-bold">{title}</h3>
        <Badge style={{ backgroundColor: versionColor }} className="text-white">
          {versionBadge}
        </Badge>
      </div>
      
      <div className="space-y-6">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-lg">Performance Metrics</CardTitle>
            <CardDescription>Raw execution times for each operation</CardDescription>
          </CardHeader>
          <CardContent>
            {metrics ? (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-gray-50 p-3 rounded-lg border border-gray-200">
                    <div className="text-sm text-gray-500 mb-1">Total Execution Time</div>
                    <div className="text-2xl font-bold">
                      {metrics.executionTime.toFixed(3)}s
                      {comparisonMetrics && getSpeedupText()}
                    </div>
                  </div>
                  <div className="bg-gray-50 p-3 rounded-lg border border-gray-200">
                    <div className="text-sm text-gray-500 mb-1">Peak Memory Usage</div>
                    <div className="text-2xl font-bold">
                      {metrics.memoryUsage.toFixed(1)} MB
                      {comparisonMetrics && getMemoryComparisonText()}
                    </div>
                  </div>
                </div>
                
                <div>
                  <div className="font-medium mb-2">Operation Times</div>
                  <PerformanceChart 
                    metrics={metrics} 
                    color={chartColor}
                    label={title}
                  />
                </div>
                
                {title === "FireDucks" && resultsMatch !== undefined && (
                  <div className="mt-2 p-2 rounded border">
                    <div className="font-medium">Result Validation:</div>
                    {resultsMatch ? (
                      <Badge className="bg-green-100 text-green-800">Results match Pandas output ✓</Badge>
                    ) : (
                      <Badge className="bg-yellow-100 text-yellow-800">Results differ from Pandas output</Badge>
                    )}
                  </div>
                )}
                
                {metrics.runs && metrics.runs.length > 1 && (
                  <div className="mt-4">
                    <div className="font-medium mb-2">Individual Runs</div>
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Run</TableHead>
                          <TableHead>Execution Time</TableHead>
                          <TableHead>Memory Usage</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {metrics.runs.map((run, i) => (
                          <TableRow key={i}>
                            <TableCell>Run {i+1}</TableCell>
                            <TableCell>{run.executionTime.toFixed(3)}s</TableCell>
                            <TableCell>{run.memoryUsage.toFixed(1)} MB</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                )}
              </div>
            ) : (
              <div className="py-8 text-center text-gray-500 italic">
                No comparison results yet. Run a comparison to see performance metrics.
              </div>
            )}
          </CardContent>
        </Card>
        
        {dataset && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-lg">Data Preview</CardTitle>
              <CardDescription>Sample from {dataset.name}</CardDescription>
            </CardHeader>
            <CardContent>
              {sampleData && sampleData.length > 0 ? (
                <div className="overflow-auto max-h-80">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        {Object.keys(sampleData[0]).map((key) => (
                          <TableHead key={key}>{key}</TableHead>
                        ))}
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {sampleData.map((row, i) => (
                        <TableRow key={i}>
                          {Object.values(row).map((value: any, j) => (
                            <TableCell key={j}>{String(value)}</TableCell>
                          ))}
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              ) : (
                <div className="py-8 text-center text-gray-500 italic">
                  No data preview available.
                </div>
              )}
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}