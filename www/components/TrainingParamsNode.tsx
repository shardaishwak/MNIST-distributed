'use client';

import { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';

interface TrainingParamsNodeData {
  numClients: number;
  epochsPerClient: number;
  onNumClientsChange: (num: number) => void;
  onEpochsPerClientChange: (num: number) => void;
}

function TrainingParamsNode({ data }: NodeProps<TrainingParamsNodeData>) {
  return (
    <div className="px-6 py-4 shadow-lg rounded-lg border-2 border-green-300 bg-white min-w-[280px]">
      <Handle type="target" position={Position.Left} />
      <Handle type="source" position={Position.Right} />
      
      <div className="space-y-4">
        <div className="flex items-center gap-2 mb-3">
          <div className="w-3 h-3 rounded-full bg-green-500"></div>
          <h3 className="font-bold text-gray-800 text-lg">3. Training Parameters</h3>
        </div>
        
        <div className="space-y-3">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Number of Clients
            </label>
            <input
              type="number"
              min="1"
              max="10"
              value={data.numClients}
              onChange={(e) => data.onNumClientsChange(parseInt(e.target.value) || 1)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm bg-white text-gray-900 font-medium focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-green-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Epochs per Client
            </label>
            <input
              type="number"
              min="1"
              max="100"
              value={data.epochsPerClient}
              onChange={(e) => data.onEpochsPerClientChange(parseInt(e.target.value) || 1)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm bg-white text-gray-900 font-medium focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-green-500"
            />
          </div>
        </div>
      </div>
    </div>
  );
}

export default memo(TrainingParamsNode);

