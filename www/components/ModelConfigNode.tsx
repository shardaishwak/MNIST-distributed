'use client';

import { useState, useEffect, memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import Editor from '@monaco-editor/react';
import axios from 'axios';

interface ModelConfigNodeData {
  modelConfig: string;
  onModelConfigChange: (config: string) => void;
  apiBaseUrl: string;
}

const defaultModelConfig = `Loading...`;

function ModelConfigNode({ data }: NodeProps<ModelConfigNodeData>) {
  const [config, setConfig] = useState<string>(data.modelConfig || defaultModelConfig);
  const [validating, setValidating] = useState<boolean>(false);
  const [validationResult, setValidationResult] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(true);

  // Load default model configuration from backend
  useEffect(() => {
    const loadDefaultModel = async () => {
      if (data.modelConfig) {
        setConfig(data.modelConfig);
        setLoading(false);
        return;
      }
      
      try {
        const response = await axios.get(`${data.apiBaseUrl}/default-model`);
        const defaultConfig = JSON.stringify(response.data, null, 2);
        setConfig(defaultConfig);
        data.onModelConfigChange(defaultConfig);
      } catch (error) {
        console.error('Error loading default model:', error);
        // Fallback to a simple config if backend fails
        setConfig(JSON.stringify({
          model_json: '{}',
          optimizer: { type: 'Adam', learning_rate: 0.0005 },
          loss: 'categorical_crossentropy',
          metrics: ['accuracy']
        }, null, 2));
      } finally {
        setLoading(false);
      }
    };

    loadDefaultModel();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data.apiBaseUrl]);

  const handleValidate = async () => {
    setValidating(true);
    setValidationResult('');
    try {
      const response = await axios.post(`${data.apiBaseUrl}/model-config`, {
        model_config: config,
      });
      setValidationResult(`✓ Valid! Input: ${JSON.stringify(response.data.input_shape)}`);
      data.onModelConfigChange(config);
    } catch (error) {
      const errorMessage = axios.isAxiosError(error)
        ? error.response?.data?.error || 'Validation failed'
        : 'Validation failed';
      setValidationResult(`✗ ${errorMessage}`);
    } finally {
      setValidating(false);
    }
  };

  return (
    <div className="px-6 py-4 shadow-lg rounded-lg border-2 border-purple-300 bg-white min-w-[400px]">
      <Handle type="target" position={Position.Left} />
      <Handle type="source" position={Position.Right} />
      
      <div className="space-y-3">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-purple-500"></div>
            <h3 className="font-bold text-gray-800 text-lg">2. Configure Model</h3>
          </div>
          <button
            onClick={handleValidate}
            disabled={validating || loading}
            className="px-3 py-1 text-xs bg-purple-600 text-white rounded hover:bg-purple-700 disabled:opacity-50"
          >
            {validating ? 'Validating...' : loading ? 'Loading...' : 'Validate'}
          </button>
        </div>
        
        <div className="border border-gray-300 rounded-md overflow-hidden" style={{ height: '250px' }}>
          {loading ? (
            <div className="flex items-center justify-center h-full bg-gray-50">
              <p className="text-sm text-gray-500">Loading default model...</p>
            </div>
          ) : (
            <Editor
              height="250px"
              defaultLanguage="json"
              value={config}
              onChange={(value) => {
                setConfig(value || '');
                data.onModelConfigChange(value || '');
              }}
              theme="vs-light"
              options={{
                minimap: { enabled: false },
                scrollBeyondLastLine: false,
                fontSize: 11,
                wordWrap: 'on',
              }}
            />
          )}
        </div>
        
        {validationResult && (
          <p className={`text-xs ${validationResult.startsWith('✓') ? 'text-green-600' : 'text-red-600'}`}>
            {validationResult}
          </p>
        )}
      </div>
    </div>
  );
}

export default memo(ModelConfigNode);

