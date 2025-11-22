#!/usr/bin/env python3
"""
Flask API server for federated learning management
"""
import os
import json
import threading
import time
import socket
import pickle
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from typing import Dict, List, Optional
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import to_categorical

# Use non-interactive backend for matplotlib (avoid GUI issues in threads)
import matplotlib
matplotlib.use('Agg')

from distributed_server import NetworkDistributedServer, send_bytes, recv_bytes

app = Flask(__name__)
CORS(app)

# Global state
training_state: Dict = {
    'status': 'idle',  # idle, running, completed, error
    'clients': {},
    'current_round': 0,
    'total_rounds': 0,
    'server': None,
    'thread': None,
    'session_id': None,
}

DATASETS_DIR = 'datasets'
OUTPUTS_DIR = 'outputs'

def ensure_datasets_dir():
    """Ensure datasets directory exists"""
    os.makedirs(DATASETS_DIR, exist_ok=True)

def ensure_outputs_dir():
    """Ensure outputs directory exists"""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

def list_datasets() -> List[str]:
    """List all available datasets in the datasets folder"""
    ensure_datasets_dir()
    datasets = []
    for file in os.listdir(DATASETS_DIR):
        if file.endswith('.npz'):
            datasets.append(file.replace('.npz', ''))
    return datasets

def load_dataset(dataset_name: str):
    """Load a dataset from the datasets folder"""
    dataset_path = os.path.join(DATASETS_DIR, f'{dataset_name}.npz')
    if not os.path.exists(dataset_path):
        # Try loading from root directory as fallback
        dataset_path = f'{dataset_name}.npz'
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset {dataset_name} not found")
    
    with np.load(dataset_path) as data:
        if 'x_train' in data and 'y_train' in data:
            return data['x_train'], data['y_train'], data.get('x_test'), data.get('y_test')
        elif 'X_train' in data and 'y_train' in data:
            return data['X_train'], data['y_train'], data.get('X_test'), data.get('y_test')
        else:
            raise ValueError(f"Dataset {dataset_name} has unexpected format")

def compile_model_from_config(model_config: dict, input_shape: tuple, num_classes: int):
    """Compile a model from JSON configuration"""
    model_json = model_config.get('model_json')
    if not model_json:
        raise ValueError("model_json is required in model_config")
    
    # If model_json is a dict, convert it to JSON string
    if isinstance(model_json, dict):
        model_json = json.dumps(model_json)
    elif not isinstance(model_json, str):
        raise ValueError("model_json must be a string or dict")
    
    model = model_from_json(model_json)
    
    # Get optimizer config
    optimizer_config = model_config.get('optimizer', {'type': 'Adam', 'learning_rate': 0.0005})
    opt_type = optimizer_config.get('type', 'Adam')
    lr = optimizer_config.get('learning_rate', 0.0005)
    
    if opt_type == 'Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    elif opt_type == 'SGD':
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=optimizer_config.get('momentum', 0.9))
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    loss = model_config.get('loss', 'categorical_crossentropy')
    metrics = model_config.get('metrics', ['accuracy'])
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

class APIDistributedServer(NetworkDistributedServer):
    """Extended server class that works with API"""
    
    def __init__(self, dataset_name: str, model_config: dict, num_clients: int, 
                 epochs_per_client: int, host='0.0.0.0', port=8888, 
                 public_holdout=8000, server_finetune_epochs=2):
        self.dataset_name = dataset_name
        self.model_config = model_config
        self.epochs_per_client = epochs_per_client
        self.client_status = {i+1: {
            'status': 'waiting', 
            'epoch': 0, 
            'total_epochs': epochs_per_client, 
            'last_ping': time.time(), 
            'connected': False,
            'client_uuid': None,  # Will be set when client connects
            'client_address': None  # Will be set when client connects
        } for i in range(num_clients)}
        # Initialize parent without loading data (we'll do it ourselves)
        self.host = host
        self.port = port
        self.num_clients = num_clients
        self.public_holdout = public_holdout
        self.server_finetune_epochs = server_finetune_epochs
        self.model_weights = []
        self.client_histories = []
        self.client_sizes = []
        self.lock = threading.Lock()
        # Don't call load_mnist_data - we'll use load_dataset_data instead
    
    def load_dataset_data(self):
        """Load dataset from datasets folder"""
        x_train, y_train, x_test, y_test = load_dataset(self.dataset_name)
        
        # Normalize if needed (assuming images)
        if x_train.dtype != np.float32:
            x_train = x_train.astype('float32') / 255.0
        if x_test is not None and x_test.dtype != np.float32:
            x_test = x_test.astype('float32') / 255.0
        
        # Reshape if needed (assuming 2D images)
        if len(x_train.shape) == 3:
            x_train = x_train.reshape(-1, *x_train.shape[1:], 1) if len(x_train.shape) == 3 else x_train
        if x_test is not None and len(x_test.shape) == 3:
            x_test = x_test.reshape(-1, *x_test.shape[1:], 1) if len(x_test.shape) == 3 else x_test
        
        # Determine number of classes
        num_classes = len(np.unique(y_train))
        y_train_oh = to_categorical(y_train, num_classes)
        y_test_oh = to_categorical(y_test, num_classes) if x_test is not None else None
        
        ph = min(self.public_holdout, len(x_train)//5)
        self.x_public = x_train[:ph] if ph > 0 else np.array([])
        self.y_public = y_train_oh[:ph] if ph > 0 else np.array([])
        self.x_train = x_train[ph:]
        self.y_train = y_train_oh[ph:]
        self.x_test = x_test if x_test is not None else np.array([])
        self.y_test = y_test_oh if y_test_oh is not None else np.array([])
        self.num_classes = num_classes
    
    def create_base_model(self):
        """Create model from configuration"""
        # Determine input shape from data
        input_shape = self.x_train.shape[1:]
        return compile_model_from_config(self.model_config, input_shape, self.num_classes)
    
    def handle_client(self, client_sock: socket.socket, client_id: int, data_split, client_address=None):
        """Override to update status and handle progress updates"""
        # Generate unique ID for this client connection
        client_uuid = str(uuid.uuid4())[:8]  # Use first 8 chars for readability
        
        self.client_status[client_id]['connected'] = True
        self.client_status[client_id]['status'] = 'connected'
        self.client_status[client_id]['last_ping'] = time.time()
        self.client_status[client_id]['client_uuid'] = client_uuid
        self.client_status[client_id]['client_address'] = client_address
        
        print(f"Client {client_id} connected: UUID={client_uuid}, Address={client_address}")
        
        client_sock.settimeout(None)
        x_split, y_split = data_split
        package = {
            'x_train': x_split, 
            'y_train': y_split, 
            'client_id': client_id, 
            'model_config': self.get_model_config(),
            'epochs': self.epochs_per_client
        }
        
        send_bytes(client_sock, pickle.dumps(package, protocol=pickle.HIGHEST_PROTOCOL))
        
        self.client_status[client_id]['status'] = 'training'
        
        # Receive messages - could be progress updates or final weights
        while True:
            payload = recv_bytes(client_sock)
            data = pickle.loads(payload)
            
            # Check if this is a progress update
            if isinstance(data, dict) and data.get('type') == 'progress':
                # Update epoch progress
                epoch = data.get('epoch', 0)
                self.client_status[client_id]['epoch'] = epoch
                self.client_status[client_id]['last_ping'] = time.time()
                print(f"Client {client_id} progress: epoch {epoch}/{self.epochs_per_client}")
                continue
            
            # Otherwise, this is the trained weights
            trained_weights = data
            break
        
        # Receive history
        history_payload = recv_bytes(client_sock)
        history_dict = pickle.loads(history_payload)
        
        with self.lock:
            self.model_weights.append(trained_weights)
            self.client_histories.append((client_id, history_dict))
            self.client_sizes.append(len(x_split))
        
        self.client_status[client_id]['status'] = 'completed'
        self.client_status[client_id]['epoch'] = self.epochs_per_client
        client_sock.close()
    
    def start_server_async(self, session_id: str):
        """Start server in a separate thread"""
        def run():
            try:
                self.load_dataset_data()
                # Now start the server
                srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                srv.bind((self.host, self.port))
                srv.listen(self.num_clients)
                print(f"Server listening on {self.host}:{self.port} (expecting {self.num_clients} clients)")

                splits = self.split_data()
                threads = []
                cid = 0
                while cid < self.num_clients:
                    client_sock, client_address = srv.accept()
                    cid += 1
                    # Format address as "IP:port"
                    addr_str = f"{client_address[0]}:{client_address[1]}" if client_address else "unknown"
                    t = threading.Thread(target=self.handle_client, args=(client_sock, cid, splits[cid - 1], addr_str), daemon=True)
                    t.start()
                    threads.append(t)

                for t in threads: 
                    t.join()
                if len(self.model_weights) != self.num_clients:
                    raise RuntimeError(f"Expected {self.num_clients} models, got {len(self.model_weights)}")

                aggregated = self.federated_averaging()
                self.evaluate_and_visualize_to_session(aggregated, session_id)
                training_state['status'] = 'completed'
            except Exception as e:
                training_state['status'] = 'error'
                training_state['error'] = str(e)
                import traceback
                traceback.print_exc()
        
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread
    
    def evaluate_and_visualize_to_session(self, aggregated_weights, session_id: str):
        """Save evaluation results to session folder"""
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        
        # Create session output directory
        session_dir = os.path.join(OUTPUTS_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        model = self.create_base_model()
        model.set_weights(aggregated_weights)

        if self.server_finetune_epochs and len(self.x_public) > 0:
            model.fit(self.x_public, self.y_public, epochs=self.server_finetune_epochs, batch_size=128, verbose=0, validation_split=0.0)

        loss, acc = model.evaluate(self.x_test, self.y_test, verbose=0)
        
        # Save model
        model_path = os.path.join(session_dir, 'model.keras')
        model.save(model_path)
        print(f"Aggregated model -> loss: {loss:.4f}  acc: {acc:.4f}  (saved to {model_path})")

        # Save confusion matrix
        y_true = np.argmax(self.y_test, axis=1)
        y_pred = np.argmax(model.predict(self.x_test, verbose=0), axis=1)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
        plt.figure(figsize=(8, 8))
        disp.plot(values_format='d', cmap='Blues', colorbar=True)
        plt.title(f'Confusion Matrix - Accuracy: {acc:.2%}')
        plt.tight_layout()
        confusion_path = os.path.join(session_dir, 'confusion_matrix.png')
        plt.savefig(confusion_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Save metrics
        metrics = {
            'loss': float(loss),
            'accuracy': float(acc),
            'num_clients': self.num_clients,
            'epochs_per_client': self.epochs_per_client,
            'dataset': self.dataset_name,
        }
        metrics_path = os.path.join(session_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Save client training plots
        for cid, hist in sorted(self.client_histories, key=lambda x: x[0]):
            self._plot_client_history_to_session(hist, cid, session_dir)
    
    def _plot_client_history_to_session(self, hist: dict, client_id: int, session_dir: str):
        """Save client training history plots to session folder"""
        import matplotlib.pyplot as plt
        
        plt.figure()
        plt.plot(hist['accuracy'], label='train_acc', marker='o')
        if 'val_accuracy' in hist and hist['val_accuracy']:
            plt.plot(hist['val_accuracy'], label='val_acc', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'Client {client_id} Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(session_dir, f'client{client_id}_accuracy.png'), dpi=150)
        plt.close()

        plt.figure()
        plt.plot(hist['loss'], label='train_loss', marker='o')
        if 'val_loss' in hist and hist['val_loss']:
            plt.plot(hist['val_loss'], label='val_loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Client {client_id} Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(session_dir, f'client{client_id}_loss.png'), dpi=150)
        plt.close()

@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    """Get list of available datasets"""
    try:
        datasets = list_datasets()
        return jsonify({'datasets': datasets})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/default-model', methods=['GET'])
def get_default_model():
    """Get a default model configuration"""
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
        
        # Create a default model
        model = Sequential([
            tf.keras.Input(shape=(28, 28, 1)),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(10, activation='softmax')
        ])
        
        # Return the model JSON in the expected format
        return jsonify({
            'model_json': model.to_json(),
            'optimizer': {
                'type': 'Adam',
                'learning_rate': 0.0005
            },
            'loss': 'categorical_crossentropy',
            'metrics': ['accuracy']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-config', methods=['POST'])
def get_model_config():
    """Get model configuration from JSON"""
    try:
        data = request.json
        model_config = data.get('model_config')
        if not model_config:
            return jsonify({'error': 'model_config is required'}), 400
        
        # If it's a string, parse it
        if isinstance(model_config, str):
            try:
                model_config = json.loads(model_config)
            except json.JSONDecodeError as e:
                return jsonify({'error': f'Invalid JSON in model_config: {str(e)}'}), 400
        
        # Ensure model_config is a dict
        if not isinstance(model_config, dict):
            return jsonify({'error': f'model_config must be a dict or JSON string. Got: {type(model_config)}'}), 400
        
        # Get model_json - it might be a string or a dict
        model_json = model_config.get('model_json')
        if not model_json:
            return jsonify({'error': 'model_json is required in model_config'}), 400
        
        # Handle different formats of model_json
        if isinstance(model_json, dict):
            # If it's a dict, convert to JSON string
            model_json_str = json.dumps(model_json)
        elif isinstance(model_json, str):
            # If it's a string, check if it's a JSON string that needs parsing
            try:
                parsed = json.loads(model_json)
                # If parsing succeeds, it was a JSON string
                if isinstance(parsed, dict):
                    # It's a JSON string of a dict, convert back to JSON string for model_from_json
                    model_json_str = json.dumps(parsed)
                else:
                    # It's a JSON string but not a dict, use as-is (might be malformed)
                    model_json_str = model_json
            except json.JSONDecodeError:
                # If it's not valid JSON, assume it's already in the format Keras expects
                model_json_str = model_json
        else:
            return jsonify({'error': f'model_json must be a string or dict. Got: {type(model_json)}'}), 400
        
        # Parse and validate the JSON
        try:
            # Parse the JSON string to a dict
            model_config = json.loads(model_json_str)
            if not isinstance(model_config, dict):
                return jsonify({'error': 'model_json must be a JSON object (dict)'}), 400
            # Check if it has the expected Keras model structure
            if 'className' not in model_config and 'config' not in model_config:
                return jsonify({'error': 'model_json does not appear to be a valid Keras model JSON'}), 400
        except json.JSONDecodeError as e:
            return jsonify({'error': f'Invalid JSON format: {str(e)}'}), 400
        
        # Create a temporary model from the config
        # In Keras 3.x, we should use model_from_config with the parsed dict
        try:
            # Try using tf.keras.models.model_from_config first (Keras 3.x compatible)
            from tensorflow.keras.models import model_from_config
            model = model_from_config(model_config)
        except Exception as e1:
            # Fallback to model_from_json with string (Keras 2.x compatible)
            try:
                model = tf.keras.models.model_from_json(model_json_str)
            except Exception as e2:
                import traceback
                error_details = traceback.format_exc()
                return jsonify({
                    'error': f'Failed to create model. Tried model_from_config: {str(e1)}, model_from_json: {str(e2)}',
                    'details': error_details,
                    'json_preview': model_json_str[:500] if len(model_json_str) > 500 else model_json_str
                }), 400
        
        # Verify model is actually a model object
        if not isinstance(model, tf.keras.Model):
            return jsonify({
                'error': f'Model creation returned unexpected type: {type(model)}. Expected tf.keras.Model'
            }), 400
        
        if not hasattr(model, 'get_weights'):
            return jsonify({
                'error': f'Model object missing get_weights method. Type: {type(model)}'
            }), 400
        
        initial_weights = model.get_weights()
        
        return jsonify({
            'model_json': model.to_json(),
            'initial_weights_shape': [w.shape for w in initial_weights],
            'input_shape': model.input_shape,
            'output_shape': model.output_shape
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/training/start', methods=['POST'])
def start_training():
    """Start federated learning training"""
    global training_state
    
    if training_state['status'] == 'running':
        return jsonify({'error': 'Training already in progress'}), 400
    
    try:
        data = request.json
        dataset_name = data.get('dataset')
        model_config = data.get('model_config')
        num_clients = int(data.get('num_clients', 2))
        epochs_per_client = int(data.get('epochs_per_client', 5))
        
        if isinstance(model_config, str):
            model_config = json.loads(model_config)
        
        # Generate session ID
        import datetime
        session_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Reset state
        training_state = {
            'status': 'running',
            'clients': {},
            'current_round': 0,
            'total_rounds': 1,
            'server': None,
            'thread': None,
            'session_id': session_id,
        }
        
        # Create and start server
        server = APIDistributedServer(
            dataset_name=dataset_name,
            model_config=model_config,
            num_clients=num_clients,
            epochs_per_client=epochs_per_client,
            host='0.0.0.0',
            port=8888
        )
        
        training_state['server'] = server
        training_state['thread'] = server.start_server_async(session_id)
        
        # Initialize client status
        for i in range(1, num_clients + 1):
            training_state['clients'][i] = {
                'status': 'waiting',
                'epoch': 0,
                'total_epochs': epochs_per_client,
                'last_ping': time.time(),
                'connected': False,
                'client_uuid': None,
                'client_address': None
            }
        
        return jsonify({'message': 'Training started', 'num_clients': num_clients, 'session_id': session_id})
    except Exception as e:
        training_state['status'] = 'error'
        return jsonify({'error': str(e)}), 500

@app.route('/api/training/status', methods=['GET'])
def get_training_status():
    """Get current training status"""
    global training_state
    
    # Create a serializable response (exclude non-JSON-serializable objects)
    response = {
        'status': training_state.get('status', 'idle'),
        'clients': training_state.get('clients', {}),
        'current_round': training_state.get('current_round', 0),
        'total_rounds': training_state.get('total_rounds', 0),
        'session_id': training_state.get('session_id'),
    }
    
    if training_state.get('server'):
        # Update client status from server
        server = training_state['server']
        for client_id, status in server.client_status.items():
            # Check if client is still connected (ping check)
            time_since_ping = time.time() - status['last_ping']
            if status['connected']:
                if time_since_ping > 60:  # 60 seconds timeout
                    status['status'] = 'disconnected'
                elif status['status'] == 'connected':
                    status['status'] = 'yellow'  # Connected but not training yet
                elif status['status'] == 'training':
                    status['status'] = 'green'  # Training
                elif status['status'] == 'completed':
                    status['status'] = 'green'  # Completed
            else:
                status['status'] = 'waiting'
        
        response['clients'] = server.client_status
    
    # Include error if present
    if 'error' in training_state:
        response['error'] = training_state['error']
    
    return jsonify(response)

@app.route('/api/training/stop', methods=['POST'])
def stop_training():
    """Stop training"""
    global training_state
    training_state['status'] = 'idle'
    return jsonify({'message': 'Training stopped'})

@app.route('/api/results/<session_id>/confusion_matrix', methods=['GET'])
def get_confusion_matrix(session_id):
    """Get confusion matrix image for a session"""
    try:
        image_path = os.path.join(OUTPUTS_DIR, session_id, 'confusion_matrix.png')
        if os.path.exists(image_path):
            from flask import send_file
            return send_file(image_path, mimetype='image/png')
        else:
            return jsonify({'error': 'Confusion matrix not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/results/<session_id>/model', methods=['GET'])
def download_model(session_id):
    """Download trained model for a session"""
    try:
        model_path = os.path.join(OUTPUTS_DIR, session_id, 'model.keras')
        if os.path.exists(model_path):
            from flask import send_file
            return send_file(model_path, as_attachment=True, download_name=f'model_{session_id}.keras')
        else:
            return jsonify({'error': 'Model not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/results/<session_id>/metrics', methods=['GET'])
def get_metrics(session_id):
    """Get training metrics for a session"""
    try:
        metrics_path = os.path.join(OUTPUTS_DIR, session_id, 'metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            return jsonify(metrics)
        else:
            return jsonify({'error': 'Metrics not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    ensure_datasets_dir()
    ensure_outputs_dir()
    app.run(host='0.0.0.0', port=5001, debug=True)

