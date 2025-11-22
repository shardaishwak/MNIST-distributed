### Distributed Machine Learning training on MNIST dataset

## Authors:
- 
- 
-


## Setup

0. Setup virtual environment
```
pip -m venv venv
```

1. Install packages
```
pip install -r requirements.txt
```

NOTE: You may need to remove `tensorflow-macos` if you are working on macOS.

2. In the `distributed_server.py`, change the number of clients.

3. Run the clients (can be started BEFORE or AFTER the server):
```
python distributed_client.py <IP_ADDRESS> [PORT] [RETRY_INTERVAL]
```

- `IP_ADDRESS`: Server IP address (use `localhost` or `127.0.0.1` for local)
- `PORT`: Server port (default: 8888)
- `RETRY_INTERVAL`: Seconds between connection attempts (default: 5)

Example:
```
python distributed_client.py localhost 8888 5
```

**New Features**: 
- Clients will automatically retry connection every 5 seconds if the server isn't available yet!
- **Continuous mode**: Clients stay running after training and reconnect for the next session automatically

4. Run the server (by default running on PORT `8888`)
```
python distributed_server.py
```

For local simulation, consider 2 clients and 1 server. Open 3 terminals and start clients first, then the server:
```
# Terminal 1 - Client 1
python distributed_client.py localhost 8888

# Terminal 2 - Client 2
python distributed_client.py localhost 8888

# Terminal 3 - Server (run multiple times for multiple training sessions!)
python distributed_server.py
```

## Continuous Mode

Clients run in continuous mode by default. After completing a training session:
1. Clients automatically close their connection
2. Wait a few seconds
3. Start listening for the next server connection
4. You can run `python distributed_server.py` again for a new session

This means you can:
- Start clients once and leave them running
- Run multiple training sessions without restarting clients
- Experiment with different parameters quickly

**Example multi-session workflow:**
```bash
# Start clients once (they keep running)
python distributed_client.py localhost 8888  # Terminal 1
python distributed_client.py localhost 8888  # Terminal 2

# Run server for session 1
python distributed_server.py  # Terminal 3
# ... training completes, clients return to listening mode ...

# Run server again for session 2 (same clients reconnect!)
python distributed_server.py  # Terminal 3 again
# ... clients automatically reconnect and train again ...
```