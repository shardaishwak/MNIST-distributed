/* eslint-disable @typescript-eslint/no-explicit-any */
'use client';

import { useState, useEffect, memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import axios from 'axios';

interface ResultsNodeData {
  sessionId: string;
  apiBaseUrl: string;
}

function ResultsNode({ data }: NodeProps<ResultsNodeData>) {
  const [metrics, setMetrics] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [downloading, setDownloading] = useState<boolean>(false);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const response = await axios.get(`${data.apiBaseUrl}/results/${data.sessionId}/metrics`);
        setMetrics(response.data);
      } catch (error) {
        console.error('Error fetching metrics:', error);
      } finally {
        setLoading(false);
      }
    };

    if (data.sessionId) {
      fetchMetrics();
    }
  }, [data.sessionId, data.apiBaseUrl]);

  const handleDownloadModel = async () => {
    setDownloading(true);
    try {
      const response = await axios.get(`${data.apiBaseUrl}/results/${data.sessionId}/model`, {
        responseType: 'blob',
      });
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `model_${data.sessionId}.keras`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Error downloading model:', error);
      alert('Failed to download model');
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="px-6 py-4 shadow-xl rounded-lg border-2 border-green-400 bg-white min-w-[500px] max-w-[600px]">
      <Handle type="target" position={Position.Left} />
      
      <div className="space-y-4">
        <div className="flex items-center gap-2 mb-3">
          <div className="w-4 h-4 rounded-full bg-green-500 animate-pulse"></div>
          <h3 className="font-bold text-gray-800 text-xl">Training Complete!</h3>
        </div>

        {loading ? (
          <div className="text-sm text-gray-500">Loading results...</div>
        ) : metrics ? (
          <div className="space-y-4">
            {/* Metrics */}
            <div className="grid grid-cols-2 gap-3 bg-gray-50 p-4 rounded-lg">
              <div>
                <p className="text-xs text-gray-500 uppercase">Accuracy</p>
                <p className="text-2xl font-bold text-green-600">{(metrics.accuracy * 100).toFixed(2)}%</p>
              </div>
              <div>
                <p className="text-xs text-gray-500 uppercase">Loss</p>
                <p className="text-2xl font-bold text-gray-700">{metrics.loss.toFixed(4)}</p>
              </div>
              <div>
                <p className="text-xs text-gray-500 uppercase">Clients</p>
                <p className="text-lg font-semibold text-gray-700">{metrics.num_clients}</p>
              </div>
              <div>
                <p className="text-xs text-gray-500 uppercase">Epochs</p>
                <p className="text-lg font-semibold text-gray-700">{metrics.epochs_per_client}</p>
              </div>
            </div>

            {/* Confusion Matrix */}
            <div className="border border-gray-200 rounded-lg overflow-hidden">
              <img 
                src={`${data.apiBaseUrl}/results/${data.sessionId}/confusion_matrix`} 
                alt="Confusion Matrix"
                className="w-full h-auto"
                onError={(e) => {
                  console.error('Failed to load confusion matrix');
                  e.currentTarget.style.display = 'none';
                }}
              />
            </div>

            {/* Download Button */}
            <button
              onClick={handleDownloadModel}
              disabled={downloading}
              className="w-full px-4 py-3 bg-gradient-to-r from-green-500 to-green-600 text-white rounded-md hover:from-green-600 hover:to-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed font-medium text-sm transition-all flex items-center justify-center gap-2"
            >
              {downloading ? (
                <>
                  <span className="animate-spin"></span>
                  Downloading...
                </>
              ) : (
                <>
                  <span></span>
                  Download Trained Model
                </>
              )}
            </button>

            <p className="text-xs text-gray-500 text-center">
              Session: {data.sessionId}
            </p>
          </div>
        ) : (
          <div className="text-sm text-red-500">Failed to load metrics</div>
        )}
      </div>
    </div>
  );
}

export default memo(ResultsNode);

