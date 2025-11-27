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
import HandwritingNode from '@/components/HandwritingNode';

const API_BASE_URL = 'http://localhost:5001/api';

const nodeTypes = {
  dataset: DatasetNode,
  modelConfig: ModelConfigNode,
  trainingParams: TrainingParamsNode,
  run: RunNode,
  client: ClientNode,
  results: ResultsNode,
  handwriting: HandwritingNode,
};

export default function Home() {
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [modelConfig, setModelConfig] = useState<string>('');
  const [numClients, setNumClients] = useState<number>(3);
  const [epochsPerClient, setEpochsPerClient] = useState<number>(5);
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [clientNodesCreated, setClientNodesCreated] = useState<boolean>(false);
  const [sessionId, setSessionId] = useState<string>(''); // eslint-disable-line @typescript-eslint/no-unused-vars
  const [serverStatus, setServerStatus] = useState<string>('idle'); // idle, running, completed, error
  const [apiConnected, setApiConnected] = useState<boolean>(false); // API server connectivity

  // Initial flow nodes with better spacing
  const initialNodes: Node[] = useMemo(() => [
    {
      id: 'dataset',
      type: 'dataset',
      position: { x: 50, y: 200 },
      data: {
        selectedDataset,
        onDatasetChange: setSelectedDataset,
        apiBaseUrl: API_BASE_URL,
      },
    },
    {
      id: 'modelConfig',
      type: 'modelConfig',
      position: { x: 400, y: 93 },
      data: {
        modelConfig,
        onModelConfigChange: setModelConfig,
        apiBaseUrl: API_BASE_URL,
      },
    },
    {
      id: 'trainingParams',
      type: 'trainingParams',
      position: { x: 850, y: 144 },
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
      position: { x: 1200, y: 180  },
      data: {
        onRun: () => {}, // Placeholder, will be updated in useEffect
        isTraining,
        canRun: selectedDataset !== '' && modelConfig !== '',
      },
    },
  ], [selectedDataset, modelConfig, numClients, epochsPerClient, isTraining]);

  const initialEdges: Edge[] = useMemo(() => [
    { id: 'e1', source: 'dataset', target: 'modelConfig', type: 'smoothstep', animated: false },
    { id: 'e2', source: 'modelConfig', target: 'trainingParams', type: 'smoothstep', animated: false },
    { id: 'e3', source: 'trainingParams', target: 'run', type: 'smoothstep', animated: false },
  ], []);

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  // Training start handler
  const handleStartTraining = useCallback(async () => {
    if (!selectedDataset || !modelConfig) {
      alert('Please select a dataset and provide model configuration');
      return;
    }

    try {
      // Clear previous training nodes (clients, results, handwriting)
      setNodes((nds) => nds.filter((n) => 
        !n.id.startsWith('client-') && 
        n.id !== 'results' && 
        n.id !== 'handwriting'
      ));
      
      // Clear edges related to clients, results, and handwriting
      setEdges((eds) => eds.filter((e) => 
        !e.source.startsWith('client-') && 
        !e.target.startsWith('client-') &&
        e.source !== 'results' &&
        e.target !== 'results' &&
        e.source !== 'handwriting' &&
        e.target !== 'handwriting' &&
        e.id !== 'results-handwriting'
      ));
      
      setIsTraining(true);
      setClientNodesCreated(false);
      setServerStatus('running');
      
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
      setServerStatus('error');
    }
  }, [selectedDataset, modelConfig, numClients, epochsPerClient, setNodes, setEdges]);

  // Update nodes when data changes (preserve client, results, and handwriting nodes)
  useEffect(() => {
    setNodes((nds) => {
      const clientNodes = nds.filter((node) => node.id.startsWith('client-'));
      const resultsNode = nds.find((node) => node.id === 'results');
      const handwritingNode = nds.find((node) => node.id === 'handwriting');
      const mainNodes = nds.filter((node) => 
        !node.id.startsWith('client-') && 
        node.id !== 'results' && 
        node.id !== 'handwriting'
      );
      
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
      
      // Combine all nodes: main nodes, client nodes, and results/handwriting if they exist
      const allNodes = [...updatedMainNodes, ...clientNodes];
      if (resultsNode) allNodes.push(resultsNode);
      if (handwritingNode) allNodes.push(handwritingNode);
      
      return allNodes;
    });
  }, [selectedDataset, modelConfig, numClients, epochsPerClient, isTraining, handleStartTraining, setNodes]);

  // Poll for API server connectivity
  useEffect(() => {
    const checkApiHealth = async () => {
      try {
        await axios.get(`${API_BASE_URL}/health`, { timeout: 3000 });
        setApiConnected(true);
      } catch (error) {
        console.error('API server not reachable:', error);
        setApiConnected(false);
      }
    };

    // Check immediately on mount
    checkApiHealth();

    // Then check every 3 seconds
    const healthInterval = setInterval(checkApiHealth, 3000);

    return () => clearInterval(healthInterval);
  }, []);

  // Poll for server status (always, even when not training)
  useEffect(() => {
    const pollStatus = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/training/status`);
        setServerStatus(response.data.status || 'idle');
      } catch (error) {
        console.error('Error polling server status:', error);
      }
    };

    // Poll immediately on mount
    pollStatus();

    // Then poll every 2 seconds
    const statusInterval = setInterval(pollStatus, 2000);

    return () => clearInterval(statusInterval);
  }, []);

  // Poll for training status
  useEffect(() => {
    if (!isTraining) return;

    const interval = setInterval(async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/training/status`);
        
        // Update server status
        setServerStatus(response.data.status || 'idle');
        
        // Update or create client nodes based on status
        if (response.data.clients) {
          const clientIds = Object.keys(response.data.clients).map(Number).sort();
          
          // Create client nodes if they don't exist
          if (!clientNodesCreated) {
            const newClientNodes: Node[] = clientIds.map((clientId, index) => {
              const clientData = response.data.clients[clientId];
              // Position clients vertically to the right of Run node
              const x = 1500; // To the right of Run node
              const y = (index * 280); // Vertical stack with 220px spacing
              
              return {
                id: `client-${clientId}`,
                type: 'client',
                position: { x, y: y - 100},
                data: {
                  clientId,
                  status: clientData.status,
                  epoch: clientData.epoch,
                  totalEpochs: clientData.total_epochs,
                  connected: clientData.connected,
                  client_uuid: clientData.client_uuid,
                  client_address: clientData.client_address,
                  crashed: clientData.crashed,
                  waiting_replacement: clientData.waiting_replacement,
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
                            client_uuid: clientData.client_uuid,
                            client_address: clientData.client_address,
                            crashed: clientData.crashed,
                            waiting_replacement: clientData.waiting_replacement,
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
          // Keep client nodes, add results and handwriting nodes
          if (response.data.session_id) {
            // Calculate middle Y position based on number of clients
            const clientNodes = response.data.clients ? Object.keys(response.data.clients) : [];
            const numClients = clientNodes.length;
            const middleY = ((numClients - 1) * 280) / 2; // Middle of vertical client stack

            const resultsNode: Node = {
              id: 'results',
              type: 'results',
              position: { x: 1800, y: middleY - 400 }, // To the right of clients, adjust for node height
              data: {
                sessionId: response.data.session_id,
                apiBaseUrl: API_BASE_URL,
              },
            };

            const handwritingNode: Node = {
              id: 'handwriting',
              type: 'handwriting',
              position: { x: 2500, y: middleY - 300 }, // To the right of results
              data: {
                sessionId: response.data.session_id,
                apiBaseUrl: API_BASE_URL,
              },
            };

            // Keep client nodes but mark them as completed, add results + handwriting nodes
            setNodes((nds) => {
              const updatedNodes = nds.map((node) => {
                if (node.id.startsWith('client-')) {
                  const clientId = parseInt(node.id.split('-')[1]);
                  const clientData = response.data.clients[clientId];
                  return {
                    ...node,
                    data: {
                      ...node.data,
                      status: 'completed',
                      epoch: clientData?.total_epochs || node.data.totalEpochs,
                      totalEpochs: clientData?.total_epochs || node.data.totalEpochs,
                    },
                  };
                }
                return node;
              });
              
              // Check if results and handwriting nodes already exist
              const hasResults = updatedNodes.some((n) => n.id === 'results');
              const hasHandwriting = updatedNodes.some((n) => n.id === 'handwriting');
              
              if (!hasResults) {
                updatedNodes.push(resultsNode);
              }
              if (!hasHandwriting) {
                updatedNodes.push(handwritingNode);
              }
              
              return updatedNodes;
            });

            // Update edges: keep client edges, add edges from all clients to results, and results to handwriting
            setEdges((eds) => {
              const newEdges = [...eds];
              
              // Add edges from each client to results (if not already exists)
              clientNodes.forEach((clientId) => {
                const edgeId = `client-${clientId}-results`;
                if (!newEdges.some((e) => e.id === edgeId)) {
                  newEdges.push({
                    id: edgeId,
                    source: `client-${clientId}`,
                    target: 'results',
                    type: 'smoothstep',
                    animated: false,
                    style: { stroke: '#10b981', strokeWidth: 2 },
                  });
                }
              });
              
              // Add edge from results to handwriting (if not already exists)
              if (!newEdges.some((e) => e.id === 'results-handwriting')) {
                newEdges.push({
                  id: 'results-handwriting',
                  source: 'results',
                  target: 'handwriting',
                  type: 'smoothstep',
                  animated: false,
                  style: { stroke: '#9333ea', strokeWidth: 2 },
                });
              }
              
              return newEdges;
            });
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

  // Helper function to get server status badge style
  const getServerStatusStyle = () => {
    switch (serverStatus) {
      case 'running':
        return { bg: 'bg-green-100', text: 'text-green-800', dot: 'bg-green-500', label: 'Server Active' };
      case 'completed':
        return { bg: 'bg-blue-100', text: 'text-blue-800', dot: 'bg-blue-500', label: 'Completed' };
      case 'error':
        return { bg: 'bg-red-100', text: 'text-red-800', dot: 'bg-red-500', label: 'Error' };
      default: // idle
        return { bg: 'bg-gray-100', text: 'text-gray-800', dot: 'bg-gray-400', label: 'Server Idle' };
    }
  };

  const statusStyle = getServerStatusStyle();

  return (
    <div className="flex h-screen w-full flex-col bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Federated Learning Dashboard</h1>
            <p className="text-sm text-gray-600 mt-1">Configure and monitor your federated learning training</p>
          </div>
          
          {/* Status Indicators */}
          <div className="flex items-center gap-3">
            {/* API Server Connectivity */}
            <div className={`flex items-center gap-2 px-4 py-2 rounded-full ${
              apiConnected 
                ? 'bg-green-100' 
                : 'bg-red-100'
            }`}>
              <div className={`w-3 h-3 rounded-full ${
                apiConnected 
                  ? 'bg-green-500 animate-pulse' 
                  : 'bg-red-500'
              }`}></div>
              <span className={`text-sm font-semibold ${
                apiConnected 
                  ? 'text-green-800' 
                  : 'text-red-800'
              }`}>
                {apiConnected ? 'API Connected' : 'API Disconnected'}
              </span>
            </div>

            {/* Training Server Status */}
            <div className={`flex items-center gap-2 px-4 py-2 rounded-full ${statusStyle.bg}`}>
              <div className={`w-3 h-3 rounded-full ${statusStyle.dot} ${serverStatus === 'running' ? 'animate-pulse' : ''}`}></div>
              <span className={`text-sm font-semibold ${statusStyle.text}`}>{statusStyle.label}</span>
            </div>
          </div>
        </div>
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
