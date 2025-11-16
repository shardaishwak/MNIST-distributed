'use client';

interface TrainingControlsProps {
  numClients: number;
  epochsPerClient: number;
  onNumClientsChange: (num: number) => void;
  onEpochsPerClientChange: (num: number) => void;
  onStartTraining: () => void;
  isTraining: boolean;
}

export default function TrainingControls({
  numClients,
  epochsPerClient,
  onNumClientsChange,
  onEpochsPerClientChange,
  onStartTraining,
  isTraining,
}: TrainingControlsProps) {
  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <label className="block text-sm font-medium text-gray-700">
          Number of Clients
        </label>
        <input
          type="number"
          min="1"
          max="10"
          value={numClients}
          onChange={(e) => onNumClientsChange(parseInt(e.target.value) || 1)}
          disabled={isTraining}
          className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 disabled:bg-gray-100"
        />
      </div>

      <div className="space-y-2">
        <label className="block text-sm font-medium text-gray-700">
          Epochs per Client
        </label>
        <input
          type="number"
          min="1"
          max="100"
          value={epochsPerClient}
          onChange={(e) => onEpochsPerClientChange(parseInt(e.target.value) || 1)}
          disabled={isTraining}
          className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 disabled:bg-gray-100"
        />
      </div>

      <button
        onClick={onStartTraining}
        disabled={isTraining}
        className="w-full px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed font-medium"
      >
        {isTraining ? 'Training...' : 'Start Training'}
      </button>

      {isTraining && (
        <div className="p-3 bg-blue-50 border border-blue-200 rounded-md">
          <p className="text-sm text-blue-800">
            Training in progress. Monitor client nodes for status updates.
          </p>
        </div>
      )}
    </div>
  );
}

