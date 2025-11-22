'use client';

import { useState, useEffect, memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import axios from 'axios';

interface DatasetNodeData {
  selectedDataset: string;
  onDatasetChange: (dataset: string) => void;
  apiBaseUrl: string;
}

function DatasetNode({ data }: NodeProps<DatasetNodeData>) {
  const [datasets, setDatasets] = useState<string[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    const fetchDatasets = async () => {
      try {
        const response = await axios.get(`${data.apiBaseUrl}/datasets`);
        setDatasets(response.data.datasets || []);
      } catch (error) {
        console.error('Error fetching datasets:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchDatasets();
  }, [data.apiBaseUrl]);

  return (
    <div className="px-6 py-4 shadow-lg rounded-lg border-2 border-blue-300 bg-white min-w-[280px]">
      <Handle type="source" position={Position.Right} />
      
      <div className="space-y-3">
        <div className="flex items-center gap-2 mb-2">
          <div className="w-3 h-3 rounded-full bg-blue-500"></div>
          <h3 className="font-bold text-gray-800 text-lg">1. Select Dataset</h3>
        </div>
        
        {loading ? (
          <div className="text-sm text-gray-500">Loading datasets...</div>
        ) : (
          <select
            value={data.selectedDataset}
            onChange={(e) => data.onDatasetChange(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm"
          >
            <option value="">-- Select a dataset --</option>
            {datasets.map((dataset) => (
              <option key={dataset} value={dataset}>
                {dataset}
              </option>
            ))}
          </select>
        )}
        
        {data.selectedDataset && (
          <p className="text-xs text-green-600 font-medium">
            ✓ Selected: {data.selectedDataset}
          </p>
        )}
      </div>
    </div>
  );
}

export default memo(DatasetNode);

