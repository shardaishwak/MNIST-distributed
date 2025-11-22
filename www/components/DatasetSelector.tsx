'use client';

import { useState, useEffect } from 'react';
import axios from 'axios';

interface DatasetSelectorProps {
  selectedDataset: string;
  onDatasetChange: (dataset: string) => void;
  apiBaseUrl: string;
}

export default function DatasetSelector({
  selectedDataset,
  onDatasetChange,
  apiBaseUrl,
}: DatasetSelectorProps) {
  const [datasets, setDatasets] = useState<string[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    const fetchDatasets = async () => {
      try {
        const response = await axios.get(`${apiBaseUrl}/datasets`);
        setDatasets(response.data.datasets || []);
      } catch (error) {
        console.error('Error fetching datasets:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchDatasets();
  }, [apiBaseUrl]);

  return (
    <div className="space-y-2">
      <label className="block text-sm font-medium text-gray-700">
        Select Dataset
      </label>
      {loading ? (
        <div className="text-sm text-gray-500">Loading datasets...</div>
      ) : (
        <select
          value={selectedDataset}
          onChange={(e) => onDatasetChange(e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
        >
          <option value="">-- Select a dataset --</option>
          {datasets.map((dataset) => (
            <option key={dataset} value={dataset}>
              {dataset}
            </option>
          ))}
        </select>
      )}
      {selectedDataset && (
        <p className="text-xs text-gray-500 mt-1">
          Selected: <span className="font-medium">{selectedDataset}</span>
        </p>
      )}
    </div>
  );
}

