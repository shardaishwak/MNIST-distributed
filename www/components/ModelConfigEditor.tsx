'use client';

import { useState } from 'react';
import Editor from '@monaco-editor/react';
import axios from 'axios';

interface ModelConfigEditorProps {
  modelConfig: string;
  onModelConfigChange: (config: string) => void;
  apiBaseUrl: string;
}

const defaultModelConfig = JSON.stringify({
  model_json: JSON.stringify({
    className: 'Sequential',
    config: {
      layers: [
        {
          className: 'InputLayer',
          config: { batch_input_shape: [null, 28, 28, 1], dtype: 'float32', name: 'input_1' },
        },
        {
          className: 'Conv2D',
          config: {
            filters: 32,
            kernel_size: [3, 3],
            activation: 'relu',
            name: 'conv2d_1',
          },
        },
        {
          className: 'MaxPooling2D',
          config: { pool_size: [2, 2], name: 'max_pooling2d_1' },
        },
        {
          className: 'Conv2D',
          config: {
            filters: 64,
            kernel_size: [3, 3],
            activation: 'relu',
            name: 'conv2d_2',
          },
        },
        {
          className: 'MaxPooling2D',
          config: { pool_size: [2, 2], name: 'max_pooling2d_2' },
        },
        {
          className: 'Conv2D',
          config: {
            filters: 64,
            kernel_size: [3, 3],
            activation: 'relu',
            name: 'conv2d_3',
          },
        },
        {
          className: 'Flatten',
          config: { name: 'flatten_1' },
        },
        {
          className: 'Dense',
          config: { units: 64, activation: 'relu', name: 'dense_1' },
        },
        {
          className: 'Dropout',
          config: { rate: 0.5, name: 'dropout_1' },
        },
        {
          className: 'Dense',
          config: { units: 10, activation: 'softmax', name: 'dense_2' },
        },
      ],
    },
  }),
  optimizer: {
    type: 'Adam',
    learning_rate: 0.0005,
  },
  loss: 'categorical_crossentropy',
  metrics: ['accuracy'],
}, null, 2);

export default function ModelConfigEditor({
  modelConfig,
  onModelConfigChange,
  apiBaseUrl,
}: ModelConfigEditorProps) {
  const [config, setConfig] = useState<string>(modelConfig || defaultModelConfig);
  const [validating, setValidating] = useState<boolean>(false);
  const [validationResult, setValidationResult] = useState<string>('');

  const handleValidate = async () => {
    setValidating(true);
    setValidationResult('');
    try {
      const response = await axios.post(`${apiBaseUrl}/model-config`, {
        model_config: config,
      });
      setValidationResult(`✓ Valid! Input shape: ${JSON.stringify(response.data.input_shape)}, Output shape: ${JSON.stringify(response.data.output_shape)}`);
      onModelConfigChange(config);
    } catch (error: any) {
      setValidationResult(`✗ Error: ${error.response?.data?.error || error.message}`);
    } finally {
      setValidating(false);
    }
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label className="block text-sm font-medium text-gray-700">
          Model Configuration (JSON)
        </label>
        <button
          onClick={handleValidate}
          disabled={validating}
          className="px-3 py-1 text-xs bg-indigo-600 text-white rounded hover:bg-indigo-700 disabled:opacity-50"
        >
          {validating ? 'Validating...' : 'Validate'}
        </button>
      </div>
      <div className="border border-gray-300 rounded-md overflow-hidden" style={{ height: '300px' }}>
        <Editor
          height="300px"
          defaultLanguage="json"
          value={config}
          onChange={(value) => {
            setConfig(value || '');
            onModelConfigChange(value || '');
          }}
          theme="vs-light"
          options={{
            minimap: { enabled: false },
            scrollBeyondLastLine: false,
            fontSize: 12,
            wordWrap: 'on',
          }}
        />
      </div>
      {validationResult && (
        <p className={`text-xs ${validationResult.startsWith('✓') ? 'text-green-600' : 'text-red-600'}`}>
          {validationResult}
        </p>
      )}
    </div>
  );
}

