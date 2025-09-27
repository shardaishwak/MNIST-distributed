#!/usr/bin/env python3
import socket, struct, pickle, time, sys, numpy as np, tensorflow as tf
from tensorflow.keras.models import model_from_json

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

class NetworkDistributedClient:
    def __init__(self, server_host: str, server_port: int = 8888, connection_timeout: int = 30):
        self.server_host = server_host
        self.server_port = server_port
        self.connection_timeout = connection_timeout

    def connect(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self.connection_timeout)
        s.connect((self.server_host, self.server_port))
        s.settimeout(None)
        self.sock = s
        print(f"Connected to {self.server_host}:{self.server_port}")

    def receive_package(self):
        pkg = pickle.loads(recv_bytes(self.sock))
        self.x_train, self.y_train = pkg['x_train'], pkg['y_train']
        self.client_id = pkg['client_id']
        cfg = pkg['model_config']

        labels = np.argmax(self.y_train, axis=1)
        total = len(labels)
        unique, counts = np.unique(labels, return_counts=True)
        dist = dict(zip(unique, counts))
        print(f"\n=== Client {self.client_id} Data Summary ===")
        print(f"Total samples: {total}")
        for c in range(10):
            print(f"  {c}: {dist.get(c, 0)}")
        print("=========================================\n")

        self.model = model_from_json(cfg['model_json'])
        self.model.set_weights(cfg['initial_weights'])
        opt = tf.keras.optimizers.Adam(learning_rate=5e-4)
        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, epochs=5, batch_size=64, validation_split=0.1):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=1e-4),
        ]
        t0 = time.time()
        hist = self.model.fit(
            self.x_train, self.y_train,
            batch_size=batch_size, epochs=epochs,
            validation_split=validation_split,
            verbose=1, callbacks=callbacks
        )
        dt = time.time() - t0
        print(f"Client {self.client_id} finished training in {dt:.1f}s")
        return {
            'accuracy': list(hist.history.get('accuracy', [])),
            'val_accuracy': list(hist.history.get('val_accuracy', [])),
            'loss': list(hist.history.get('loss', [])),
            'val_loss': list(hist.history.get('val_loss', [])),
        }

    def send_weights_and_history(self, history_dict):
        send_bytes(self.sock, pickle.dumps(self.model.get_weights(), protocol=pickle.HIGHEST_PROTOCOL))
        send_bytes(self.sock, pickle.dumps(history_dict, protocol=pickle.HIGHEST_PROTOCOL))

    def run(self, epochs=5, batch_size=64):
        self.connect()
        self.receive_package()
        history = self.train(epochs=epochs, batch_size=batch_size)
        self.send_weights_and_history(history)
        self.sock.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: client.py <server_host> [server_port]")
        sys.exit(1)
    host = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) >= 3 else 8888
    NetworkDistributedClient(host, port).run(epochs=5, batch_size=64)

if __name__ == "__main__":
    main()
