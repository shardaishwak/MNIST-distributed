'use client';

import { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';

interface RunNodeData {
  onRun: () => void;
  isTraining: boolean;
  canRun: boolean;
}

function RunNode({ data }: NodeProps<RunNodeData>) {
  return (
    <div className="px-6 py-4 shadow-lg rounded-lg border-2 border-orange-300 bg-white min-w-[200px]">
      <Handle type="target" position={Position.Left} />
      <Handle type="source" position={Position.Right} />
      
      <div className="space-y-3">
        <div className="flex items-center gap-2 mb-3">
          <div className="w-3 h-3 rounded-full bg-orange-500"></div>
          <h3 className="font-bold text-gray-800 text-lg">4. Run Training</h3>
        </div>
        
        <button
          onClick={data.onRun}
          disabled={!data.canRun || data.isTraining}
          className="w-full px-4 py-3 bg-gradient-to-r from-orange-500 to-orange-600 text-white rounded-md hover:from-orange-600 hover:to-orange-700 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed font-medium text-sm transition-all"
        >
          {data.isTraining ? (
            <span className="flex items-center justify-center gap-2">
              <span className="animate-spin">⏳</span>
              Running...
            </span>
          ) : (
            '▶ Start Training'
          )}
        </button>
        
        {!data.canRun && !data.isTraining && (
          <p className="text-xs text-gray-500 text-center">
            Complete previous steps
          </p>
        )}
      </div>
    </div>
  );
}

export default memo(RunNode);

