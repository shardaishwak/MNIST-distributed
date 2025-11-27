#!/usr/bin/env python3
import socket, struct, pickle, numpy as np, threading
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def send_bytes(sock: socket.socket, payload: bytes) -> None:
    sock.sendall(struct.pack("!Q", len(payload)))
    sock.sendall(payload)

def recv_bytes(sock: socket.socket) -> bytes:
    def recvn(n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Socket connection broken during recv")
            buf.extend(chunk)
        return bytes(buf)
    (size,) = struct.unpack("!Q", recvn(8))
    return recvn(size)

class NetworkDistributedServer:
    def __init__(self, host='0.0.0.0', port=8888, num_clients=2, public_holdout=8000, server_finetune_epochs=2, client_epochs=5):
        self.host = host
        self.port = port
        self.num_clients = num_clients
        self.public_holdout = public_holdout
        self.server_finetune_epochs = server_finetune_epochs
        self.client_epochs = client_epochs
        self.model_weights = []
        self.client_histories = []
        self.client_sizes = []
        self.client_label_dists = []
        self.client_socks = []
        self.lock = threading.Lock()
        self.clients_connected = threading.Event()
        self.load_mnist_data()

    def load_mnist_data(self):
        try:
            with np.load('mnist.npz') as data:
                x_train, y_train = data['x_train'], data['y_train']
                x_test, y_test = data['x_test'], data['y_test']
        except FileNotFoundError:
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            np.savez('mnist.npz', x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

        # normalize
        x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        x_test  = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        y_train_oh = to_categorical(y_train, 10)
        y_test_oh  = to_categorical(y_test, 10)

        ph = min(self.public_holdout, len(x_train)//5)
        self.x_public, self.y_public = x_train[:ph], y_train_oh[:ph]

    def create_base_model(self):
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
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def get_model_config(self):
        base = self.create_base_model()
        return {'model_json': base.to_json(), 'initial_weights': base.get_weights()}

    def split_data(self, shuffle_seed: int = 42):
        labels = np.argmax(self.y_train, axis=1)
        rng = np.random.default_rng(shuffle_seed)
        per_class_indices = []
        for c in range(10):
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
        global_counts = np.zeros(10)
        for dist in self.client_label_dists:
            for c in range(10):
                global_counts[c] += dist.get(c, 0)
        
        total = global_counts.sum()
        if total == 0:
            return {c: 1.0 for c in range(10)}
        
        # Inverse frequency weighting: weight = total / (num_classes * class_count)
        class_weights = {}
        for c in range(10):
            if global_counts[c] > 0:
                class_weights[c] = total / (10 * global_counts[c])
            else:
                class_weights[c] = 1.0
        
        # Normalize so min weight is 1.0
        min_weight = min(class_weights.values())
        for c in range(10):
            class_weights[c] /= min_weight
        
        print("\n=== Global Class Weights ===")
        print("Global Label Counts:")
        for c in range(10):
            print(f"  Class {c}: {int(global_counts[c])}")
        print("\nClass Weights:")
        for c in range(10):
            print(f"  Class {c}: {class_weights[c]:.4f}")
        print("=" * 30 + "\n")
        
        return class_weights

    def _plot_client_history(self, hist: dict, client_id: int):
        plt.figure(); plt.plot(hist['accuracy'], label='train_acc')
        if 'val_accuracy' in hist: plt.plot(hist['val_accuracy'], label='val_acc')
        plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title(f'Client {client_id} Accuracy')
        plt.legend(); plt.tight_layout(); plt.savefig(f'client{client_id}_accuracy.png'); plt.close()

        plt.figure(); plt.plot(hist['loss'], label='train_loss')
        if 'val_loss' in hist: plt.plot(hist['val_loss'], label='val_loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title(f'Client {client_id} Loss')
        plt.legend(); plt.tight_layout(); plt.savefig(f'client{client_id}_loss.png'); plt.close()

    def handle_client(self, client_sock: socket.socket, client_id: int, data_split):
        client_sock.settimeout(None)
        x_split, y_split = data_split
        package = {
            'x_train': x_split, 
            'y_train': y_split, 
            'client_id': client_id, 
            'model_config': self.get_model_config(),
            'epochs': self.client_epochs
        }
        send_bytes(client_sock, pickle.dumps(package, protocol=pickle.HIGHEST_PROTOCOL))

        # Receive label distribution from client
        label_dist = pickle.loads(recv_bytes(client_sock))
        
        with self.lock:
            self.client_label_dists.append(label_dist)
            self.client_socks.append(client_sock)
            num_connected = len(self.client_socks)
        
        # Wait until all clients have connected and reported their distributions
        if num_connected == self.num_clients:
            # Compute global class weights
            global_class_weights = self.compute_global_class_weights()
            
            # Send computed weights to all waiting clients
            with self.lock:
                for sock in self.client_socks:
                    send_bytes(sock, pickle.dumps(global_class_weights, protocol=pickle.HIGHEST_PROTOCOL))
            
            self.clients_connected.set()
        else:
            # Wait for signal that all clients are connected
            self.clients_connected.wait()

        weights_payload = recv_bytes(client_sock)
        trained_weights = pickle.loads(weights_payload)

        history_payload = recv_bytes(client_sock)
        history_dict = pickle.loads(history_payload)

        with self.lock:
            self.model_weights.append(trained_weights)
            self.client_histories.append((client_id, history_dict))
            self.client_sizes.append(len(x_split))

        client_sock.close()

    def federated_averaging(self):
        sizes = np.array(self.client_sizes, dtype=np.float64)
        sizes = sizes / sizes.sum()

        avg = [np.zeros_like(w, dtype=w.dtype) for w in self.model_weights[0]]
        for w, alpha in zip(self.model_weights, sizes):
            for i in range(len(w)):
                avg[i] += alpha * w[i]
        return avg

    def evaluate_and_visualize(self, aggregated_weights):
        model = self.create_base_model()
        model.set_weights(aggregated_weights)

        if self.server_finetune_epochs and len(self.x_public) > 0:
            model.fit(self.x_public, self.y_public, epochs=self.server_finetune_epochs, batch_size=128, verbose=0, validation_split=0.0)

        loss, acc = model.evaluate(self.x_test, self.y_test, verbose=0)
        model.save('aggregated_mnist_model.keras')
        print(f"Aggregated model -> loss: {loss:.4f}  acc: {acc:.4f}  (saved aggregated_mnist_model.keras)")

        # Confusion matrix
        y_true = np.argmax(self.y_test, axis=1)
        y_pred = np.argmax(model.predict(self.x_test, verbose=0), axis=1)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
        plt.figure(figsize=(6, 6))
        disp.plot(values_format='d', cmap=None, colorbar=False)
        plt.title('Aggregated Model Confusion Matrix')
        plt.tight_layout(); plt.savefig('confusion_matrix.png'); plt.close()

        for cid, hist in sorted(self.client_histories, key=lambda x: x[0]):
            self._plot_client_history(hist, cid)

    def start_server(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(self.num_clients)
        print(f"Server listening on {self.host}:{self.port} (expecting {self.num_clients} clients)")

        splits = self.split_data()
        threads = []
        cid = 0
        while cid < self.num_clients:
            client_sock, _ = srv.accept()
            cid += 1
            t = threading.Thread(target=self.handle_client, args=(client_sock, cid, splits[cid - 1]), daemon=True)
            t.start()
            threads.append(t)

        srv.close()
        print(f"Accepted {self.num_clients} clients. Server socket closed to prevent extra connections.")

        for t in threads: t.join()
        if len(self.model_weights) != self.num_clients:
            raise RuntimeError(f"Expected {self.num_clients} models, got {len(self.model_weights)}")

        aggregated = self.federated_averaging()
        self.evaluate_and_visualize(aggregated)

def main():
    server = NetworkDistributedServer(
        host='0.0.0.0', 
        port=8888, 
        num_clients=2, 
        public_holdout=8000, 
        server_finetune_epochs=2,
        client_epochs=5
    )
    server.start_server()

if __name__ == "__main__":
    main()
