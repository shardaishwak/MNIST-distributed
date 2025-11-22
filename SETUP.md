# Federated Learning Dashboard - Setup Guide

## Quick Start

### 1. Backend Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Start the Flask API server
python api_server.py
```

The API server will run on `http://localhost:5001`

### 2. Frontend Setup

```bash
# Navigate to www directory
cd www

# Install dependencies
npm install

# Start the development server
npm run dev
```

The frontend will run on `http://localhost:3000`

### 3. Client Setup

In separate terminal windows, start client instances:

```bash
# Client 1
python distributed_client.py localhost 8888

# Client 2 (in another terminal)
python distributed_client.py localhost 8888

# Client 3 (in another terminal)
python distributed_client.py localhost 8888
```

## Dataset Preparation

1. Place your datasets in the `datasets` folder as `.npz` files
2. Each dataset should contain:
   - `x_train`: Training features (numpy array)
   - `y_train`: Training labels (numpy array)
   - `x_test`: Test features (optional)
   - `y_test`: Test labels (optional)

Example: `datasets/mnist.npz`

## Usage Flow

1. **Open the dashboard** at `http://localhost:3000`
2. **Select a dataset** from the dropdown
3. **Configure your model** in the JSON editor (or use the default)
4. **Set parameters**:
   - Number of clients (should match the number of client processes you'll start)
   - Epochs per client
5. **Start client processes** (see step 3 above)
6. **Click "Start Training"** in the dashboard
7. **Monitor progress** in the React Flow visualization

## Features

- Real-time status updates for each client
- Visual progress bars showing epoch completion
- Color-coded status indicators:
  - 🟢 Green: Training/Completed
  - 🟡 Yellow: Connected
  - 🔴 Red: Disconnected
  - ⚪ Gray: Waiting

## Troubleshooting

- **No datasets showing**: Make sure datasets are in the `datasets` folder as `.npz` files
- **Clients not connecting**: Ensure clients are started before clicking "Start Training"
- **API errors**: Check that the Flask server is running on port 5001
- **CORS errors**: The Flask server has CORS enabled, but ensure both servers are running

## Architecture

- **Frontend**: Next.js 16 with React Flow for node visualization
- **Backend API**: Flask REST API for managing training sessions
- **Training Engine**: Distributed server/client architecture using sockets
- **Communication**: 
  - Frontend ↔ Backend: HTTP REST API
  - Server ↔ Clients: Socket connections (port 8888)

