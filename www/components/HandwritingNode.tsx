'use client';

import { useState, useRef, useEffect, memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import axios from 'axios';

interface HandwritingNodeData {
  sessionId: string;
  apiBaseUrl: string;
}

function HandwritingNode({ data }: NodeProps<HandwritingNodeData>) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [probabilities, setProbabilities] = useState<number[]>([]);
  const [isPredicting, setIsPredicting] = useState(false);
  const [lastX, setLastX] = useState(0);
  const [lastY, setLastY] = useState(0);

  const CANVAS_SIZE = 280;
  const BRUSH_RADIUS = 10;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Initialize with white background
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  }, []);

  const startDrawing = (e: React.MouseEvent<HTMLCanvasElement>) => {
    e.stopPropagation(); // Prevent ReactFlow from capturing this event
    e.preventDefault();
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    setIsDrawing(true);
    setLastX(x);
    setLastY(y);
    
    drawPoint(x, y);
  };

  const draw = (e: React.MouseEvent<HTMLCanvasElement>) => {
    e.stopPropagation(); // Prevent ReactFlow from capturing this event
    e.preventDefault();
    
    if (!isDrawing) return;
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Draw line
    ctx.strokeStyle = 'black';
    ctx.lineWidth = BRUSH_RADIUS * 2;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();
    
    // Draw circle at end point
    ctx.fillStyle = 'black';
    ctx.beginPath();
    ctx.arc(x, y, BRUSH_RADIUS, 0, Math.PI * 2);
    ctx.fill();
    
    setLastX(x);
    setLastY(y);
  };

  const drawPoint = (x: number, y: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    ctx.fillStyle = 'black';
    ctx.beginPath();
    ctx.arc(x, y, BRUSH_RADIUS, 0, Math.PI * 2);
    ctx.fill();
  };

  const stopDrawing = (e: React.MouseEvent<HTMLCanvasElement>) => {
    e.stopPropagation(); // Prevent ReactFlow from capturing this event
    setIsDrawing(false);
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    
    setPrediction(null);
    setProbabilities([]);
  };

  const predictDigit = async () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    setIsPredicting(true);
    try {
      // Convert canvas to blob
      const blob = await new Promise<Blob>((resolve) => {
        canvas.toBlob((blob) => {
          resolve(blob!);
        }, 'image/png');
      });
      
      // Send to API
      const formData = new FormData();
      formData.append('image', blob, 'canvas.png');
      formData.append('session_id', data.sessionId);
      
      const response = await axios.post(
        `${data.apiBaseUrl}/predict/digit`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );
      
      setPrediction(response.data.prediction);
      setProbabilities(response.data.probabilities);
    } catch (error) {
      console.error('Error predicting digit:', error);
      alert('Failed to predict digit. Make sure the model is loaded.');
    } finally {
      setIsPredicting(false);
    }
  };

  const getTopProbabilities = () => {
    if (probabilities.length === 0) return [];
    
    const indexed = probabilities.map((prob, idx) => ({ digit: idx, prob }));
    indexed.sort((a, b) => b.prob - a.prob);
    return indexed.slice(0, 3);
  };

  return (
    <div className="px-6 py-4 shadow-xl rounded-lg border-2 border-purple-400 bg-white min-w-[400px]">
      <Handle type="target" position={Position.Top} />
      
      <div className="space-y-4">
        <div className="flex items-center gap-2 mb-3">
          <div className="w-4 h-4 rounded-full bg-purple-500"></div>
          <h3 className="font-bold text-gray-800 text-xl">Handwriting Recognition</h3>
        </div>

        {/* Interactive area - prevent node dragging using ReactFlow's nodrag class */}
        <div className="nodrag">
          <p className="text-sm text-gray-600 mb-4">
            Draw a digit (0-9) below and click Predict
          </p>

          {/* Canvas */}
          <div className="border-2 border-gray-300 rounded-lg overflow-hidden bg-white cursor-crosshair mb-4">
            <canvas
              ref={canvasRef}
              width={CANVAS_SIZE}
              height={CANVAS_SIZE}
              onMouseDown={startDrawing}
              onMouseMove={draw}
              onMouseUp={stopDrawing}
              onMouseLeave={stopDrawing}
              className="block"
            />
          </div>

          {/* Controls */}
          <div className="flex gap-2">
          <button
            onClick={predictDigit}
            disabled={isPredicting}
            className="flex-1 px-4 py-2 bg-gradient-to-r from-purple-500 to-purple-600 text-white rounded-md hover:from-purple-600 hover:to-purple-700 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed font-medium text-sm transition-all"
          >
            {isPredicting ? (
              <span className="flex items-center justify-center gap-2">
                <span className="animate-spin"></span>
                Predicting...
              </span>
            ) : (
              'Predict'
            )}
          </button>
          
          <button
            onClick={clearCanvas}
            className="px-4 py-2 bg-gray-200 text-gray-700 rounded-md hover:bg-gray-300 focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 font-medium text-sm transition-all"
          >
            Clear
          </button>
        </div>

        {/* Prediction Results */}
        {prediction !== null && (
          <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-4 rounded-lg border border-purple-200">
            <div className="text-center mb-3">
              <p className="text-sm text-gray-600 mb-1">Prediction:</p>
              <p className="text-5xl font-bold text-purple-600">{prediction}</p>
            </div>
            
            <div className="space-y-1">
              <p className="text-xs text-gray-600 font-semibold mb-2">Top 3 Probabilities:</p>
              {getTopProbabilities().map((item) => (
                <div key={item.digit} className="flex justify-between items-center text-sm">
                  <span className="font-mono font-semibold text-gray-700">Digit {item.digit}:</span>
                  <span className="font-mono text-purple-600">{(item.prob * 100).toFixed(2)}%</span>
                </div>
              ))}
            </div>
          </div>
        )}
        </div>
        {/* End interactive area */}

        <p className="text-xs text-gray-500 text-center">
          Using model from session: {data.sessionId}
        </p>
      </div>
    </div>
  );
}

export default memo(HandwritingNode);

