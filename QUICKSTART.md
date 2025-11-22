# Quick Start Guide

## Running the Federated Learning Dashboard

### 1. Start the API Server

```bash
python api_server.py
```

Server will run on `http://localhost:5001`

### 2. Start the Frontend

```bash
cd www
npm run dev
```

Frontend will run on `http://localhost:3000`

### 3. Open the Dashboard

Go to `http://localhost:3000` in your browser

### 4. Configure Training

1. **Select Dataset**: Choose from available datasets (e.g., `mnist`)
2. **Configure Model**: The default model will load automatically. Click "Validate" to verify it.
3. **Set Parameters**:
   - Number of clients (e.g., `1` for testing, `2-3` for real federated learning)
   - Epochs per client (e.g., `3-5`)
4. **Click "Start Training"**

### 5. Start Client(s)

In separate terminal(s), run:

```bash
# Terminal 1 - Client 1
python distributed_client.py localhost 8888

# Terminal 2 - Client 2 (if num_clients > 1)
python distributed_client.py localhost 8888

# Terminal 3 - Client 3 (if num_clients > 2)
python distributed_client.py localhost 8888
```

**New Features**: 
- **Clients can be started BEFORE or AFTER the server!** If the server isn't running yet, clients will continuously retry connection every 5 seconds
- **Continuous Mode**: Clients stay running after training completes and automatically reconnect for the next session
- Start all clients once, then run multiple training sessions without restarting clients
- You can also start the server first (old behavior still works)

### 6. Monitor Progress

Watch the React Flow diagram update with:
- Client nodes appearing dynamically
- Status indicators (Green = Training/Completed, Yellow = Connected, Gray = Waiting)
- **Real-time epoch progress bars** showing training progress as each client completes epochs

### 7. View Results

When training completes:
- **Client nodes merge into a Results node** showing:
  - Confusion matrix visualization
  - Model accuracy and loss
  - Training statistics (clients, epochs, dataset)
  - **Download button** for the trained model

All outputs are saved in the `outputs/` folder organized by session ID (timestamp).

### Outputs

Each training session creates a folder in `outputs/YYYYMMDD_HHMMSS/` containing:
- `model.keras` - The trained aggregated model
- `confusion_matrix.png` - Model performance visualization
- `metrics.json` - Training metrics (accuracy, loss, etc.)
- `client1_accuracy.png`, `client1_loss.png`, etc. - Per-client training metrics

## Continuous Mode (Multiple Training Sessions)

Clients now run in **continuous mode** by default - they stay alive after training completes and automatically reconnect for the next session!

### Workflow for Multiple Sessions:

```bash
# Terminal 1, 2, 3 - Start clients ONCE
python distributed_client.py localhost 8888
```

Then in the dashboard, you can:
1. Click "Start Training" → Clients connect and train → Session 1 completes
2. Clients automatically return to listening mode
3. Change parameters in dashboard (epochs, model config, etc.)
4. Click "Start Training" again → Same clients reconnect → Session 2 starts
5. Repeat as many times as needed!

**Benefits:**
- No need to restart client scripts between sessions
- Experiment with different parameters quickly
- Ideal for iterative model development
- Clients gracefully handle server disconnections

**Note:** Press Ctrl+C on any client to stop it. The client shows a session counter (SESSION 1, SESSION 2, etc.) to track how many training rounds it has participated in.

## Common Issues

### "Server listening on 0.0.0.0:8888 (expecting X clients)" but nothing happens
- **Solution**: Start the client processes! The server waits for clients to connect.
- **Note**: Clients will automatically retry connection if server isn't ready yet.

### Client shows "Connection failed: ConnectionRefusedError"
- **Solution**: This is normal! The client will keep retrying every 5 seconds until the server starts.
- Optionally adjust retry interval: `python distributed_client.py localhost 8888 10` (10 second intervals)

### Client connects but training doesn't start
- **Solution**: Make sure you started the correct number of clients matching your configuration.

### Port already in use
- **Solution**: Stop any running instances of `api_server.py` or clients before restarting.

### Epochs don't match what I set
- **Solution**: Restart all clients after changing the configuration. Old client instances may have cached the previous settings.

## Architecture

```
Browser (localhost:3000)
    ↓ HTTP REST API
Flask API Server (localhost:5001)
    ↓ Socket (port 8888)
Python Training Clients (distributed_client.py)
```

The frontend communicates with the Flask API server via REST, which manages a training server on port 8888 that coordinates with the Python clients.

