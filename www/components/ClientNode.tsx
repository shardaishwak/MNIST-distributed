'use client';

import { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';

interface ClientNodeData {
  clientId: number;
  status: string;
  epoch: number;
  totalEpochs: number;
  connected: boolean;
  client_uuid?: string | null;
  client_address?: string | null;
  crashed?: boolean;
  waiting_replacement?: boolean;
}

function ClientNode({ data }: NodeProps<ClientNodeData>) {
  const { clientId, status, epoch, totalEpochs, client_uuid, client_address, crashed, waiting_replacement } = data;

  // Determine status color
  const getStatusColor = () => {
    if (crashed || status === 'crashed') return 'bg-red-500';
    if (status === 'completed' || status === 'green') return 'bg-green-500';
    if (status === 'training' || status === 'connected') return 'bg-yellow-500';
    if (status === 'disconnected' || status === 'red') return 'bg-red-500';
    return 'bg-gray-400'; // waiting
  };

  const getStatusText = () => {
    if (crashed && waiting_replacement) return 'Crashed - Awaiting Replacement';
    if (crashed) return 'Crashed';
    if (status === 'completed') return 'Completed';
    if (status === 'training' || status === 'green') return 'Training';
    if (status === 'connected' || status === 'yellow') return 'Connected';
    if (status === 'disconnected' || status === 'red') return 'Disconnected';
    return 'Waiting';
  };

  const progress = totalEpochs > 0 ? (epoch / totalEpochs) * 100 : 0;

  return (
    <div className="px-4 py-3 shadow-lg rounded-lg border-2 border-gray-200 bg-white min-w-[200px]">
      <Handle type="target" position={Position.Left} />
      
      <div className="flex items-center justify-between mb-2">
        <h3 className="font-bold text-gray-800">Client {clientId}</h3>
        <div className="flex items-center gap-2">
          <div
            className={`w-3 h-3 rounded-full ${getStatusColor()} animate-pulse`}
            title={getStatusText()}
          />
          <span className="text-xs text-gray-600">{getStatusText()}</span>
        </div>
      </div>

      {/* Client UUID and Address */}
      {client_uuid && (
        <div className="mb-2 pb-2 border-b border-gray-200">
          <div className="text-xs text-gray-500">
            ID: <span className="font-mono font-semibold text-indigo-600">{client_uuid}</span>
          </div>
          {client_address && (
            <div className="text-xs text-gray-500 mt-1">
              From: <span className="font-mono text-gray-700">{client_address}</span>
            </div>
          )}
        </div>
      )}

      {/* Crash/Replacement Warning */}
      {crashed && waiting_replacement && (
        <div className="mb-2 pb-2 border-b border-red-200 bg-red-50 p-2 rounded">
          <div className="text-xs text-red-700 font-semibold">
            ⚠ Client Crashed
          </div>
          <div className="text-xs text-red-600 mt-1">
            Waiting for replacement client...
          </div>
        </div>
      )}

      <div className="space-y-2">
        <div className="text-sm text-gray-600">
          Epoch: <span className="font-semibold">{epoch}</span> / {totalEpochs}
        </div>
        
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div
            className={`h-2 rounded-full transition-all duration-300 ${
              progress === 100 ? 'bg-green-500' : 'bg-indigo-500'
            }`}
            style={{ width: `${progress}%` }}
          />
        </div>
        
        <div className="text-xs text-gray-500 text-center">
          {progress.toFixed(1)}% Complete
        </div>
      </div>

      <Handle type="source" position={Position.Right} />
    </div>
  );
}

export default memo(ClientNode);

