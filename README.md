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

2. In the `distribute_server.py`, change the number of clients.

3. Run the server (by default running on PORT `8888`)
```
python distribute_server.py
```

4. Run the clients:
```
python distribute_client.py <IP_ADDRESS> 8888
```

- If the client is running on the same host: `IP_ADDRESS = 127.0.0.1`
- If the client is running on a different network, it would be the IP of the machine the server is running on.


For local simulation, consider 2 clients and 1 server. Open 3 terminals: one for server and two for client.