import React, { createContext, useContext, useState, ReactNode } from 'react';
import { Operations, Dataset, Comparison } from '@shared/schema';

interface ComparisonContextProps {
  // Current dataset state
  currentDataset: Dataset | null;
  setCurrentDataset: (dataset: Dataset | null) => void;
  
  // Operations state
  operations: Operations;
  setOperations: (operations: Operations) => void;
  
  // Settings state
  multiRunEnabled: boolean;
  setMultiRunEnabled: (enabled: boolean) => void;
  runCount: number;
  setRunCount: (count: number) => void;
  
  // Results state
  comparisonResults: Comparison | null;
  setComparisonResults: (results: Comparison | null) => void;
  
  // Loading states
  isGeneratingData: boolean;
  setIsGeneratingData: (isLoading: boolean) => void;
  isRunningComparison: boolean;
  setIsRunningComparison: (isLoading: boolean) => void;
  isUploadingFile: boolean;
  setIsUploadingFile: (isLoading: boolean) => void;
}

const ComparisonContext = createContext<ComparisonContextProps | undefined>(undefined);

export function ComparisonProvider({ children }: { children: ReactNode }) {
  // Current dataset state
  const [currentDataset, setCurrentDataset] = useState<Dataset | null>(null);
  
  // Operations state
  const [operations, setOperations] = useState<Operations>({
    load: true,
    groupby: true,
    merge: true,
    filter: true,
    rolling: true,
    pivotTable: false,
    complexAggregation: false,
    windowFunctions: false,
    stringManipulation: false,
    nestedOperations: false,
    concat: false,
    sort: false,
    info: false,
    toCSV: false,
  });
  
  // Settings state
  const [multiRunEnabled, setMultiRunEnabled] = useState<boolean>(false);
  const [runCount, setRunCount] = useState<number>(3);
  
  // Results state
  const [comparisonResults, setComparisonResults] = useState<Comparison | null>(null);
  
  // Loading states
  const [isGeneratingData, setIsGeneratingData] = useState<boolean>(false);
  const [isRunningComparison, setIsRunningComparison] = useState<boolean>(false);
  const [isUploadingFile, setIsUploadingFile] = useState<boolean>(false);
  
  const value = {
    currentDataset,
    setCurrentDataset,
    operations,
    setOperations,
    multiRunEnabled,
    setMultiRunEnabled,
    runCount,
    setRunCount,
    comparisonResults,
    setComparisonResults,
    isGeneratingData,
    setIsGeneratingData,
    isRunningComparison,
    setIsRunningComparison,
    isUploadingFile,
    setIsUploadingFile,
  };
  
  return (
    <ComparisonContext.Provider value={value}>
      {children}
    </ComparisonContext.Provider>
  );
}

export function useComparison() {
  const context = useContext(ComparisonContext);
  if (context === undefined) {
    throw new Error('useComparison must be used within a ComparisonProvider');
  }
  return context;
}
