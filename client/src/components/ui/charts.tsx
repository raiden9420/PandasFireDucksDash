import { useEffect, useRef } from 'react';
import { Metrics } from '@shared/schema';
import { Chart, registerables } from 'chart.js';

Chart.register(...registerables);

interface PerformanceChartProps {
  metrics: Metrics;
  color: string;
  label: string;
}

export function PerformanceChart({ metrics, color, label }: PerformanceChartProps) {
  const chartRef = useRef<HTMLCanvasElement>(null);
  const chartInstance = useRef<Chart | null>(null);
  
  useEffect(() => {
    if (!chartRef.current || !metrics) return;
    
    // Destroy previous chart if it exists
    if (chartInstance.current) {
      chartInstance.current.destroy();
    }
    
    // Get operation times from metrics
    const operationTimes = metrics.operationTimes || {};
    const labels = Object.keys(operationTimes);
    
    // Convert seconds to milliseconds for better readability
    const data = labels.map(op => operationTimes[op] * 1000);
    
    // Create the chart
    const ctx = chartRef.current.getContext('2d');
    if (!ctx) return;
    
    chartInstance.current = new Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            label: `${label} Operation Times (ms)`,
            data,
            backgroundColor: color,
            borderColor: 'rgba(0, 0, 0, 0.1)',
            borderWidth: 1,
            borderRadius: 4,
            barThickness: 'flex',
            maxBarThickness: 40,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: false,
          },
          tooltip: {
            callbacks: {
              label: (context) => `${context.parsed.y.toFixed(2)} ms`,
            },
          },
        },
        scales: {
          y: {
            beginAtZero: true,
            ticks: {
              callback: (value) => `${value} ms`,
            },
            grid: {
              color: 'rgba(0, 0, 0, 0.05)',
            },
          },
          x: {
            grid: {
              display: false,
            },
          },
        },
        animation: {
          duration: 500,
        },
      },
    });
    
    return () => {
      if (chartInstance.current) {
        chartInstance.current.destroy();
      }
    };
  }, [metrics, color, label]);
  
  return (
    <div className="h-64 bg-white border border-gray-200 rounded-md p-3">
      {metrics && Object.keys(metrics.operationTimes || {}).length > 0 ? (
        <canvas ref={chartRef} />
      ) : (
        <div className="h-full flex items-center justify-center text-gray-400">
          <div className="text-center">
            <span className="text-3xl block mb-2">📊</span>
            <p className="text-sm">No operation data available</p>
          </div>
        </div>
      )}
    </div>
  );
}
