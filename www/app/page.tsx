'use client';

import { useState, useEffect, useCallback, useMemo } from 'react';
import ReactFlow, {
  Node,
  Edge,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
} from 'reactflow';
import 'reactflow/dist/style.css';
import axios from 'axios';
import DatasetNode from '@/components/DatasetNode';
import ModelConfigNode from '@/components/ModelConfigNode';
import TrainingParamsNode from '@/components/TrainingParamsNode';
import RunNode from '@/components/RunNode';
import ClientNode from '@/components/ClientNode';
import ResultsNode from '@/components/ResultsNode';

const API_BASE_URL = 'http://localhost:5001/api';

const nodeTypes = {
  dataset: DatasetNode,
  modelConfig: ModelConfigNode,
  trainingParams: TrainingParamsNode,
  run: RunNode,
  client: ClientNode,
  results: ResultsNode,
};

export default function Home() {
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [modelConfig, setModelConfig] = useState<string>('');
  const [numClients, setNumClients] = useState<number>(3);
  const [epochsPerClient, setEpochsPerClient] = useState<number>(5);
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [clientNodesCreated, setClientNodesCreated] = useState<boolean>(false);
  const [sessionId, setSessionId] = useState<string>(''); // eslint-disable-line @typescript-eslint/no-unused-vars

  const handleStartTraining = useCallback(async () => {
    if (!selectedDataset || !modelConfig) {
      alert('Please select a dataset and provide model configuration');
      return;
    }

    try {
      setIsTraining(true);
      setClientNodesCreated(false);
      
      const response = await axios.post(`${API_BASE_URL}/training/start`, {
        dataset: selectedDataset,
        model_config: modelConfig,
        num_clients: numClients,
        epochs_per_client: epochsPerClient,
      });
      
      setSessionId(response.data.session_id || '');
      console.log('Training started:', response.data);
    } catch (error) {
      console.error('Error starting training:', error);
      const errorMessage = axios.isAxiosError(error) 
        ? error.response?.data?.error || 'Failed to start training'
        : 'Failed to start training';
      alert(errorMessage);
      setIsTraining(false);
    }
  }, [selectedDataset, modelConfig, numClients, epochsPerClient]);

  // Initial flow nodes
  const initialNodes: Node[] = useMemo(() => [
    {
      id: 'dataset',
      type: 'dataset',
      position: { x: 100, y: 200 },
      data: {
        selectedDataset,
        onDatasetChange: setSelectedDataset,
        apiBaseUrl: API_BASE_URL,
      },
    },
    {
      id: 'modelConfig',
      type: 'modelConfig',
      position: { x: 450, y: 200 },
      data: {
        modelConfig,
        onModelConfigChange: setModelConfig,
        apiBaseUrl: API_BASE_URL,
      },
    },
    {
      id: 'trainingParams',
      type: 'trainingParams',
      position: { x: 800, y: 200 },
      data: {
        numClients,
        epochsPerClient,
        onNumClientsChange: setNumClients,
        onEpochsPerClientChange: setEpochsPerClient,
      },
    },
    {
      id: 'run',
      type: 'run',
      position: { x: 1150, y: 200 },
      data: {
        onRun: handleStartTraining,
        isTraining,
        canRun: selectedDataset !== '' && modelConfig !== '',
      },
    },
  ], [selectedDataset, modelConfig, numClients, epochsPerClient, isTraining, handleStartTraining]);

  const initialEdges: Edge[] = useMemo(() => [
    { id: 'e1', source: 'dataset', target: 'modelConfig', type: 'smoothstep', animated: false },
    { id: 'e2', source: 'modelConfig', target: 'trainingParams', type: 'smoothstep', animated: false },
    { id: 'e3', source: 'trainingParams', target: 'run', type: 'smoothstep', animated: false },
  ], []);

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  // Update nodes when data changes (preserve client nodes)
  useEffect(() => {
    setNodes((nds) => {
      const clientNodes = nds.filter((node) => node.id.startsWith('client-'));
      const mainNodes = nds.filter((node) => !node.id.startsWith('client-'));
      
      const updatedMainNodes = mainNodes.map((node) => {
        if (node.id === 'dataset') {
          return { ...node, data: { ...node.data, selectedDataset, onDatasetChange: setSelectedDataset } };
        }
        if (node.id === 'modelConfig') {
          return { ...node, data: { ...node.data, modelConfig, onModelConfigChange: setModelConfig } };
        }
        if (node.id === 'trainingParams') {
          return {
            ...node,
            data: {
              ...node.data,
              numClients,
              epochsPerClient,
              onNumClientsChange: setNumClients,
              onEpochsPerClientChange: setEpochsPerClient,
            },
          };
        }
        if (node.id === 'run') {
          return {
            ...node,
            data: {
              ...node.data,
              onRun: handleStartTraining,
              isTraining,
              canRun: selectedDataset !== '' && modelConfig !== '',
            },
          };
        }
        return node;
      });
      
      return [...updatedMainNodes, ...clientNodes];
    });
  }, [selectedDataset, modelConfig, numClients, epochsPerClient, isTraining, handleStartTraining, setNodes]);

  // Poll for training status
  useEffect(() => {
    if (!isTraining) return;

    const interval = setInterval(async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/training/status`);
        
        // Update or create client nodes based on status
        if (response.data.clients) {
          const clientIds = Object.keys(response.data.clients).map(Number).sort();
          
          // Create client nodes if they don't exist
          if (!clientNodesCreated) {
            const newClientNodes: Node[] = clientIds.map((clientId, index) => {
              const clientData = response.data.clients[clientId];
              const x = 1150 + ((index % 3) * 350);
              const y = 400 + Math.floor(index / 3) * 250;
              
              return {
                id: `client-${clientId}`,
                type: 'client',
                position: { x, y },
                data: {
                  clientId,
                  status: clientData.status,
                  epoch: clientData.epoch,
                  totalEpochs: clientData.total_epochs,
                  connected: clientData.connected,
                },
              };
            });
            
            const newEdges: Edge[] = clientIds.map((clientId) => ({
              id: `run-client-${clientId}`,
              source: 'run',
              target: `client-${clientId}`,
              type: 'smoothstep',
              animated: true,
            }));
            
            setNodes((nds) => [...nds, ...newClientNodes]);
            setEdges((eds) => [...eds, ...newEdges]);
            setClientNodesCreated(true);
          } else {
            // Update existing client nodes
            setNodes((nds) =>
              nds.map((node) => {
                if (node.id.startsWith('client-')) {
                  const clientId = parseInt(node.id.split('-')[1]);
                      const clientData = response.data.clients[clientId];
                      if (clientData) {
                        return {
                          ...node,
                          data: {
                            ...node.data,
                            status: clientData.status,
                            epoch: clientData.epoch,
                            totalEpochs: clientData.total_epochs,
                            connected: clientData.connected,
                          },
                        };
                      }
                    }
                    return node;
                  })
            );
          }
        }
        
        if (response.data.status === 'completed') {
          setIsTraining(false);
          // Replace client nodes with results node
          if (response.data.session_id) {
            const resultsNode: Node = {
              id: 'results',
              type: 'results',
              position: { x: 1150, y: 400 },
              data: {
                sessionId: response.data.session_id,
                apiBaseUrl: API_BASE_URL,
              },
            };

            // Remove client nodes and add results node
            setNodes((nds) => [
              ...nds.filter((n) => !n.id.startsWith('client-')),
              resultsNode,
            ]);

            // Update edges to connect run to results
            setEdges((eds) => [
              ...eds.filter((e) => !e.source.startsWith('run') || !e.target.startsWith('client-')),
              {
                id: 'run-results',
                source: 'run',
                target: 'results',
                type: 'smoothstep',
                animated: false,
                style: { stroke: '#10b981', strokeWidth: 3 },
              },
            ]);
          }
        } else if (response.data.status === 'error') {
          setIsTraining(false);
        }
      } catch (error) {
        console.error('Error fetching training status:', error);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [isTraining, clientNodesCreated, setNodes, setEdges]);

  return (
    <div className="flex h-screen w-full flex-col bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200 px-6 py-4">
        <h1 className="text-2xl font-bold text-gray-900">Federated Learning Dashboard</h1>
        <p className="text-sm text-gray-600 mt-1">Configure and monitor your federated learning training</p>
      </header>

      {/* Main Content - React Flow */}
      <main className="flex-1 relative">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          nodeTypes={nodeTypes}
          fitView
          className="bg-gray-50"
          minZoom={0.2}
          maxZoom={2}
        >
          <Background />
          <Controls />
          <MiniMap />
        </ReactFlow>
      </main>
    </div>
  );
}
