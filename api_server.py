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

training_state: Dict = {
    'status': 'idle',
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
    
    if isinstance(model_json, dict):
        model_json = json.dumps(model_json)
    elif not isinstance(model_json, str):
        raise ValueError("model_json must be a string or dict")
    
    model = model_from_json(model_json)
    
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
                 public_holdout=8000, server_finetune_epochs=2, use_balancing=True):
        self.dataset_name = dataset_name
        self.model_config = model_config
        self.epochs_per_client = epochs_per_client
        self.use_balancing = use_balancing
        self.client_status = {i+1: {
            'status': 'waiting', 
            'epoch': 0, 
            'total_epochs': epochs_per_client, 
            'last_ping': time.time(), 
            'connected': False,
            'client_uuid': None, 
            'client_address': None,
            'crashed': False,
            'waiting_replacement': False 
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
        self.client_label_dists = []
        self.client_socks = []
        self.lock = threading.Lock()
        self.clients_connected = threading.Event()
        self.data_splits = []
        self.failed_clients = {}
    
    def load_dataset_data(self):
        """Load dataset from datasets folder"""
        x_train, y_train, x_test, y_test = load_dataset(self.dataset_name)
        
        if x_train.dtype != np.float32:
            x_train = x_train.astype('float32') / 255.0
        if x_test is not None and x_test.dtype != np.float32:
            x_test = x_test.astype('float32') / 255.0
        
        if len(x_train.shape) == 3:
            x_train = x_train.reshape(-1, *x_train.shape[1:], 1) if len(x_train.shape) == 3 else x_train
        if x_test is not None and len(x_test.shape) == 3:
            x_test = x_test.reshape(-1, *x_test.shape[1:], 1) if len(x_test.shape) == 3 else x_test
        
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
        input_shape = self.x_train.shape[1:]
        return compile_model_from_config(self.model_config, input_shape, self.num_classes)
    
    def get_model_config(self):
        """Get model config for sending to clients"""
        base = self.create_base_model()
        return {'model_json': base.to_json(), 'initial_weights': base.get_weights()}
    
    def split_data(self, shuffle_seed: int = 42):
        """Split data among clients (balanced per class)"""
        labels = np.argmax(self.y_train, axis=1)
        rng = np.random.default_rng(shuffle_seed)
        per_class_indices = []
        for c in range(self.num_classes):
            idx = np.where(labels == c)[0]
            rng.shuffle(idx)
            per_class_indices.append(idx)

        client_indices = [[] for _ in range(self.num_clients)]
        for c_idx in per_class_indices:
            parts = np.array_split(c_idx, self.num_clients)
            for i in range(self.num_clients):
                if len(parts[i]) > 0:
                    client_indices[i].extend(parts[i].tolist())

        splits = []
        for i in range(self.num_clients):
            rng.shuffle(client_indices[i])
            idxs = np.array(client_indices[i], dtype=np.int64)
            splits.append((self.x_train[idxs], self.y_train[idxs]))
        return splits
    
    def compute_global_class_weights(self):
        """Compute class weights based on global label distribution across all clients"""
        global_counts = np.zeros(self.num_classes)
        for dist in self.client_label_dists:
            for c in range(self.num_classes):
                global_counts[c] += dist.get(c, 0)
        
        total = global_counts.sum()
        if total == 0:
            return {c: 1.0 for c in range(self.num_classes)}
        
        # Inverse frequency weighting: weight = total / (num_classes * class_count)
        class_weights = {}
        for c in range(self.num_classes):
            if global_counts[c] > 0:
                class_weights[c] = total / (self.num_classes * global_counts[c])
            else:
                class_weights[c] = 1.0
        
        # Normalize so min weight is 1.0
        min_weight = min(class_weights.values())
        for c in range(self.num_classes):
            class_weights[c] /= min_weight
        
        print("\n=== Global Class Weights ===")
        print("Global Label Counts:")
        for c in range(self.num_classes):
            print(f"  Class {c}: {int(global_counts[c])}")
        print("\nClass Weights:")
        for c in range(self.num_classes):
            print(f"  Class {c}: {class_weights[c]:.4f}")
        print("=" * 30 + "\n")
        
        return class_weights
    
    def handle_client(self, client_sock: socket.socket, client_id: int, data_split, client_address=None):
        """Override to update status and handle progress updates and crashes"""
        client_uuid = str(uuid.uuid4())[:8]
        completed_successfully = False
        
        try:
            self.client_status[client_id]['connected'] = True
            self.client_status[client_id]['status'] = 'connected'
            self.client_status[client_id]['last_ping'] = time.time()
            self.client_status[client_id]['client_uuid'] = client_uuid
            self.client_status[client_id]['client_address'] = client_address
            self.client_status[client_id]['crashed'] = False
            self.client_status[client_id]['waiting_replacement'] = False
            
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
            
            print(f"Client {client_id} waiting for all clients to connect...")
            
            # BALANCING: Receive label distribution from client
            if self.use_balancing:
                label_dist = pickle.loads(recv_bytes(client_sock))
                
                with self.lock:
                    self.client_label_dists.append(label_dist)
                    self.client_socks.append(client_sock)
                    num_connected = len(self.client_socks)
                
                print(f"Client {client_id} reported label distribution. Connected: {num_connected}/{self.num_clients}")
                
                # Wait until all clients have connected and reported their distributions
                if num_connected == self.num_clients:
                    print(f"All clients connected! Computing global class weights...")
                    # Compute global class weights
                    global_class_weights = self.compute_global_class_weights()
                    
                    # Send computed weights to all waiting clients
                    with self.lock:
                        for sock in self.client_socks:
                            send_bytes(sock, pickle.dumps(global_class_weights, protocol=pickle.HIGHEST_PROTOCOL))
                    
                    print(f"Global weights sent to all clients. Starting training...")
                    self.clients_connected.set()
                else:
                    # Wait for signal that all clients are connected
                    print(f"Client {client_id} waiting for other clients...")
                    self.clients_connected.wait()
                    print(f"Client {client_id} received signal to proceed")
            else:
                # No balancing, but still wait for all clients to connect
                # I will change this part before we submit, coudlnt get it to work otherwise.
                with self.lock:
                    self.client_socks.append(client_sock)
                    num_connected = len(self.client_socks)
                
                print(f"Client {client_id} connected. Waiting for all clients: {num_connected}/{self.num_clients}")
                
                if num_connected == self.num_clients:
                    # Send dummy weights (all 1.0) to all clients
                    dummy_weights = {c: 1.0 for c in range(self.num_classes)}
                    with self.lock:
                        for sock in self.client_socks:
                            send_bytes(sock, pickle.dumps(dummy_weights, protocol=pickle.HIGHEST_PROTOCOL))
                    
                    print(f"All clients connected. Sent dummy weights. Starting training...")
                    self.clients_connected.set()
                else:
                    # Wait for signal that all clients are connected
                    print(f"Client {client_id} waiting for other clients...")
                    self.clients_connected.wait()
                    print(f"Client {client_id} received signal to proceed")
            
            self.client_status[client_id]['status'] = 'training'
            
            while True:
                payload = recv_bytes(client_sock)
                data = pickle.loads(payload)
                
                if isinstance(data, dict) and data.get('type') == 'crash':
                    print(f"! Client {client_id} (UUID={client_uuid}) sent crash notification")
                    self.client_status[client_id]['crashed'] = True
                    self.client_status[client_id]['status'] = 'crashed'
                    self.client_status[client_id]['waiting_replacement'] = True
                    with self.lock:
                        self.failed_clients[client_id] = data_split
                    client_sock.close()
                    return
                
                if isinstance(data, dict) and data.get('type') == 'progress':
                    epoch = data.get('epoch', 0)
                    self.client_status[client_id]['epoch'] = epoch
                    self.client_status[client_id]['last_ping'] = time.time()
                    print(f"Client {client_id} progress: epoch {epoch}/{self.epochs_per_client}")
                    continue
                
                trained_weights = data
                break
            
            history_payload = recv_bytes(client_sock)
            history_dict = pickle.loads(history_payload)
            
            with self.lock:
                self.model_weights.append(trained_weights)
                self.client_histories.append((client_id, history_dict))
                self.client_sizes.append(len(x_split))
            
            completed_successfully = True
            self.client_status[client_id]['status'] = 'completed'
            self.client_status[client_id]['epoch'] = self.epochs_per_client
            print(f"Client {client_id} (UUID={client_uuid}) completed successfully")
            
            client_sock.close()
            
        except (ConnectionError, ConnectionResetError, BrokenPipeError) as e:
            if not completed_successfully:
                print(f"! Client {client_id} (UUID={client_uuid}) disconnected unexpectedly: {e}")
                self.client_status[client_id]['crashed'] = True
                self.client_status[client_id]['status'] = 'crashed'
                self.client_status[client_id]['waiting_replacement'] = True
                self.client_status[client_id]['connected'] = False
                with self.lock:
                    self.failed_clients[client_id] = data_split
            try:
                client_sock.close()
            except:
                pass
        except Exception as e:
            if not completed_successfully:
                print(f"! Error handling client {client_id} (UUID={client_uuid}): {e}")
                self.client_status[client_id]['crashed'] = True
                self.client_status[client_id]['status'] = 'crashed'
                self.client_status[client_id]['waiting_replacement'] = True
                self.client_status[client_id]['connected'] = False
                with self.lock:
                    self.failed_clients[client_id] = data_split
            try:
                client_sock.close()
            except:
                pass
    
    def federated_averaging(self):
        """Federated averaging with weighted aggregation"""
        sizes = np.array(self.client_sizes, dtype=np.float64)
        sizes = sizes / sizes.sum()

        avg = [np.zeros_like(w, dtype=w.dtype) for w in self.model_weights[0]]
        for w, alpha in zip(self.model_weights, sizes):
            for i in range(len(w)):
                avg[i] += alpha * w[i]
        return avg
    
    def start_server_async(self, session_id: str):
        """Start server in a separate thread"""
        def run():
            try:
                self.load_dataset_data()
                srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                srv.bind((self.host, self.port))
                srv.listen(self.num_clients)
                print(f"Server listening on {self.host}:{self.port} (expecting {self.num_clients} clients)")
                print(f"Balancing enabled: {self.use_balancing}")

                splits = self.split_data()
                self.data_splits = splits 
                threads = {}
                
                cid = 0
                while cid < self.num_clients:
                    client_sock, client_address = srv.accept()
                    cid += 1
                    addr_str = f"{client_address[0]}:{client_address[1]}" if client_address else "unknown"
                    t = threading.Thread(target=self.handle_client, args=(client_sock, cid, splits[cid - 1], addr_str), daemon=True)
                    t.start()
                    threads[cid] = t
                
                srv.close()
                srv = None
                print(f"Accepted {self.num_clients} clients. Server socket closed to prevent extra connections.")
                
                while True:
                    for client_id in list(threads.keys()):
                        threads[client_id].join(timeout=0.1)
                        if not threads[client_id].is_alive():
                            del threads[client_id]
                    
                    with self.lock:
                        failed_client_ids = list(self.failed_clients.keys())
                    
                    if failed_client_ids and srv is None:
                        print(f"Reopening server socket for {len(failed_client_ids)} replacement client(s)...")
                        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        srv.bind((self.host, self.port))
                        srv.listen(len(failed_client_ids))
                        srv.settimeout(1.0)
                    
                    if failed_client_ids and srv is not None:
                        for failed_id in failed_client_ids:
                            print(f"Waiting for replacement client for slot {failed_id}...")
                            try:
                                client_sock, client_address = srv.accept()
                                addr_str = f"{client_address[0]}:{client_address[1]}" if client_address else "unknown"
                                print(f"Replacement client connected for slot {failed_id}")
                                
                                with self.lock:
                                    data_split = self.failed_clients.pop(failed_id)
                                
                                self.client_status[failed_id]['crashed'] = False
                                self.client_status[failed_id]['waiting_replacement'] = False
                                self.client_status[failed_id]['epoch'] = 0
                                
                                t = threading.Thread(target=self.handle_client, args=(client_sock, failed_id, data_split, addr_str), daemon=True)
                                t.start()
                                threads[failed_id] = t
                            except socket.timeout:
                                pass
                        
                        # If no more failed clients, close the replacement socket
                        with self.lock:
                            if len(self.failed_clients) == 0 and srv is not None:
                                srv.close()
                                srv = None
                                print("All replacements handled. Server socket closed again.")
                    
                    if len(self.model_weights) == self.num_clients and len(threads) == 0:
                        break
                    
                    time.sleep(0.5)
                
                print(f"All {self.num_clients} clients completed successfully")
                
                if srv is not None:
                    srv.close()
                    print("Training server socket closed")

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
        
        session_dir = os.path.join(OUTPUTS_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        model = self.create_base_model()
        model.set_weights(aggregated_weights)

        if self.server_finetune_epochs and len(self.x_public) > 0:
            model.fit(self.x_public, self.y_public, epochs=self.server_finetune_epochs, batch_size=128, verbose=0, validation_split=0.0)

        loss, acc = model.evaluate(self.x_test, self.y_test, verbose=0)
        
        model_path = os.path.join(session_dir, 'model.keras')
        model.save(model_path)
        print(f"Aggregated model -> loss: {loss:.4f}  acc: {acc:.4f}  (saved to {model_path})")

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

        metrics = {
            'loss': float(loss),
            'accuracy': float(acc),
            'num_clients': self.num_clients,
            'epochs_per_client': self.epochs_per_client,
            'dataset': self.dataset_name,
            'balancing_enabled': self.use_balancing,
        }
        metrics_path = os.path.join(session_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

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

# ============================================================================
# API Routes
# ============================================================================

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
        
        if isinstance(model_config, str):
            try:
                model_config = json.loads(model_config)
            except json.JSONDecodeError as e:
                return jsonify({'error': f'Invalid JSON in model_config: {str(e)}'}), 400
        
        if not isinstance(model_config, dict):
            return jsonify({'error': f'model_config must be a dict or JSON string. Got: {type(model_config)}'}), 400
        
        model_json = model_config.get('model_json')
        if not model_json:
            return jsonify({'error': 'model_json is required in model_config'}), 400
        
        if isinstance(model_json, dict):
            model_json_str = json.dumps(model_json)
        elif isinstance(model_json, str):
            try:
                parsed = json.loads(model_json)
                if isinstance(parsed, dict):
                    model_json_str = json.dumps(parsed)
                else:
                    model_json_str = model_json
            except json.JSONDecodeError:
                model_json_str = model_json
        else:
            return jsonify({'error': f'model_json must be a string or dict. Got: {type(model_json)}'}), 400
        
        try:
            model_config = json.loads(model_json_str)
            if not isinstance(model_config, dict):
                return jsonify({'error': 'model_json must be a JSON object (dict)'}), 400
            if 'className' not in model_config and 'config' not in model_config:
                return jsonify({'error': 'model_json does not appear to be a valid Keras model JSON'}), 400
        except json.JSONDecodeError as e:
            return jsonify({'error': f'Invalid JSON format: {str(e)}'}), 400
        
        try:
            from tensorflow.keras.models import model_from_config
            model = model_from_config(model_config)
        except Exception as e1:
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
        use_balancing = data.get('use_balancing', True)
        
        if isinstance(model_config, str):
            model_config = json.loads(model_config)
        
        import datetime
        session_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
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
            port=8888,
            use_balancing=use_balancing
        )
        
        training_state['server'] = server
        training_state['thread'] = server.start_server_async(session_id)
        
        for i in range(1, num_clients + 1):
            training_state['clients'][i] = {
                'status': 'waiting',
                'epoch': 0,
                'total_epochs': epochs_per_client,
                'last_ping': time.time(),
                'connected': False,
                'client_uuid': None,
                'client_address': None,
                'crashed': False,
                'waiting_replacement': False
            }
        
        return jsonify({
            'message': 'Training started', 
            'num_clients': num_clients, 
            'session_id': session_id,
            'balancing_enabled': use_balancing
        })
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
        server = training_state['server']
        for client_id, status in server.client_status.items():
            time_since_ping = time.time() - status['last_ping']
            if status['connected']:
                if time_since_ping > 60:
                    status['status'] = 'disconnected'
                elif status['status'] == 'connected':
                    status['status'] = 'yellow'
                elif status['status'] == 'training':
                    status['status'] = 'green'
                elif status['status'] == 'completed':
                    status['status'] = 'green'
            else:
                status['status'] = 'waiting'
        
        response['clients'] = server.client_status
    
    if 'error' in training_state:
        response['error'] = training_state['error']
    
    return jsonify(response)

@app.route('/api/training/stop', methods=['POST'])
def stop_training():
    """Stop training"""
    global training_state
    training_state['status'] = 'idle'
    return jsonify({'message': 'Training stopped'})

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify server is running"""
    return jsonify({
        'status': 'ok',
        'message': 'API server is running',
        'timestamp': time.time()
    })

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

# Handwriting recognition preprocessing functions (from playground.py)
def _center_of_mass(img_arr):
    """Calculate center of mass of image"""
    y_idx, x_idx = np.indices(img_arr.shape)
    m = img_arr.sum() + 1e-8
    cy = (y_idx * img_arr).sum() / m
    cx = (x_idx * img_arr).sum() / m
    return cy, cx

def _shift_to_center(img_arr, target=(14, 14)):
    """Shift image to center based on center of mass"""
    cy, cx = _center_of_mass(img_arr)
    dy = int(round(target[0] - cy))
    dx = int(round(target[1] - cx))
    shifted = np.roll(img_arr, shift=dy, axis=0)
    shifted = np.roll(shifted, shift=dx, axis=1)
    return shifted

def _deskew_28(img_arr):
    """Simple deskew by second-order moments"""
    import math
    y, x = np.indices(img_arr.shape)
    img = img_arr.copy()
    img /= (img.max() + 1e-8)

    m00 = img.sum() + 1e-8
    m10 = (x * img).sum()
    m01 = (y * img).sum()
    x_cent = m10 / m00
    y_cent = m01 / m00

    x = x - x_cent
    y = y - y_cent

    mu11 = (x * y * img).sum() / m00
    mu20 = ((x ** 2) * img).sum() / m00
    mu02 = ((y ** 2) * img).sum() / m00
    
    denom = (mu20 - mu02)
    if abs(denom) < 1e-8:
        return img_arr

    tan2theta = 2 * mu11 / denom
    theta = 0.5 * math.atan(tan2theta)
    t = math.tan(theta)
    h, w = img_arr.shape
    yy, xx = np.indices((h, w)).astype(np.float32)
    yy -= h / 2.0
    xx -= w / 2.0
    yy_prime = yy - t * xx
    yy_prime += h / 2.0
    xx += w / 2.0

    def bilinear(img, yyf, xxf):
        h, w = img.shape
        y0 = np.clip(np.floor(yyf).astype(int), 0, h - 1)
        x0 = np.clip(np.floor(xxf).astype(int), 0, w - 1)
        y1 = np.clip(y0 + 1, 0, h - 1)
        x1 = np.clip(x0 + 1, 0, w - 1)

        wy = yyf - y0
        wx = xxf - x0

        top = (1 - wx) * img[y0, x0] + wx * img[y0, x1]
        bot = (1 - wx) * img[y1, x0] + wx * img[y1, x1]
        return (1 - wy) * top + wy * bot

    deskewed = bilinear(img_arr, yy_prime, xx)
    return deskewed

def preprocess_digit_image(pil_img):
    """Preprocess digit image for MNIST model (from playground.py)"""
    from PIL import Image, ImageOps, ImageFilter
    
    # Convert to grayscale and invert
    img = pil_img.convert("L")
    img = ImageOps.invert(img)

    # Apply filters
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    img = img.filter(ImageFilter.MaxFilter(size=3))

    # Binarize
    arr = np.array(img, dtype=np.uint8)
    th = 128
    bin_mask = (arr >= th)

    if not bin_mask.any():
        x28 = np.zeros((28, 28, 1), dtype=np.float32)
        return x28[np.newaxis, ...]

    # Crop to bounding box
    ys, xs = np.where(bin_mask)
    minx, maxx = xs.min(), xs.max()
    miny, maxy = ys.min(), ys.max()
    img = img.crop((minx, miny, maxx + 1, maxy + 1))
    
    # Resize to fit in 20x20 box, centered in 28x28
    w, h = img.size
    scale = 20 / max(w, h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), color=0)
    left = (28 - new_w) // 2
    top = (28 - new_h) // 2
    canvas.paste(img, (left, top))

    # Normalize and apply transformations
    x = np.array(canvas).astype("float32") / 255.0
    x = _shift_to_center(x)
    x = _deskew_28(x)  # Apply deskewing

    x = x.reshape(1, 28, 28, 1).astype(np.float32)
    return x

@app.route('/api/predict/digit', methods=['POST'])
def predict_digit():
    """Predict a digit from a canvas drawing"""
    try:
        from flask import request
        from PIL import Image
        import io
        
        # Get the image and session_id from the request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        session_id = request.form.get('session_id')
        if not session_id:
            return jsonify({'error': 'No session_id provided'}), 400
        
        # Load the model for this session
        model_path = os.path.join(OUTPUTS_DIR, session_id, 'model.keras')
        if not os.path.exists(model_path):
            return jsonify({'error': 'Model not found for this session'}), 404
        
        model = tf.keras.models.load_model(model_path)
        
        # Load and preprocess the image
        image_file = request.files['image']
        pil_image = Image.open(io.BytesIO(image_file.read()))
        
        # Preprocess the image
        preprocessed = preprocess_digit_image(pil_image)
        
        # Make prediction
        predictions = model.predict(preprocessed, verbose=0)[0]
        predicted_digit = int(np.argmax(predictions))
        
        return jsonify({
            'prediction': predicted_digit,
            'probabilities': predictions.tolist(),
        })
    except Exception as e:
        import traceback
        print(f"Error in predict_digit: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    ensure_datasets_dir()
    ensure_outputs_dir()
    app.run(host='0.0.0.0', port=5001, debug=True)