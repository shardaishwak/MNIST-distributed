# Federated Learning Dashboard

A modern web interface for managing and visualizing federated learning training sessions using Next.js and React Flow.

## Features

- **Dataset Selection**: Choose from available datasets in the `datasets` folder
- **Model Configuration**: Define your neural network model using JSON configuration
- **Client Management**: Configure number of clients and training epochs
- **Real-time Visualization**: Monitor client status and training progress with React Flow
- **Status Indicators**: Green/Yellow/Red status lights for each client node

## Getting Started

### Prerequisites

- Node.js 18+ and npm
- Python 3.10+ with required packages (see main project requirements.txt)

### Installation

1. Install frontend dependencies:
```bash
cd www
npm install
```

2. Install backend dependencies (from project root):
```bash
pip install -r requirements.txt
```

### Running the Application

1. Start the Flask API server (from project root):
```bash
python api_server.py
```
The API will run on `http://localhost:5001`

2. Start the Next.js frontend (from www directory):
```bash
cd www
npm run dev
```
The frontend will run on `http://localhost:3000`

3. Start client instances (from project root):
```bash
# In separate terminals, run:
python distributed_client.py localhost 8888
# Repeat for each client you want to connect
```

## Usage

1. **Select Dataset**: Choose a dataset from the dropdown (datasets should be in `.npz` format in the `datasets` folder)

2. **Configure Model**: 
   - Edit the JSON model configuration in the Monaco editor
   - Click "Validate" to verify the configuration
   - The model should follow Keras Sequential model format

3. **Set Training Parameters**:
   - Choose the number of clients
   - Set epochs per client

4. **Start Training**: Click "Start Training" and watch the React Flow visualization update in real-time

5. **Monitor Progress**: 
   - Each client node shows:
     - Status indicator (Green = Training/Completed, Yellow = Connected, Red = Disconnected)
     - Current epoch / Total epochs
     - Progress bar

## Dataset Format

Datasets should be stored in the `datasets` folder as `.npz` files with the following keys:
- `x_train`: Training features
- `y_train`: Training labels
- `x_test`: Test features (optional)
- `y_test`: Test labels (optional)

## Model Configuration Format

The model configuration should be a JSON object with:
```json
{
  "model_json": "<Keras model JSON string>",
  "optimizer": {
    "type": "Adam",
    "learning_rate": 0.0005
  },
  "loss": "categorical_crossentropy",
  "metrics": ["accuracy"]
}
```

## Architecture

- **Frontend**: Next.js 16 with React Flow for visualization
- **Backend**: Flask API server that manages training sessions
- **Training**: Distributed server/client architecture using sockets
