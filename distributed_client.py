#!/usr/bin/env python3
import socket, struct, pickle, time, sys, numpy as np, tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import Callback

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

class EpochProgressCallback(Callback):
    """Callback to send epoch progress updates to server"""
    def __init__(self, sock: socket.socket, client_id: int):
        super().__init__()
        self.sock = sock
        self.client_id = client_id
    
    def on_epoch_end(self, epoch, logs=None):
        """Send progress update after each epoch"""
        try:
            progress_msg = {
                'type': 'progress',
                'client_id': self.client_id,
                'epoch': epoch + 1,  # Convert to 1-indexed
                'logs': logs or {}
            }
            # Use a non-blocking send for progress updates
            try:
                send_bytes(self.sock, pickle.dumps(progress_msg, protocol=pickle.HIGHEST_PROTOCOL))
            except Exception as e:
                print(f"Warning: Could not send progress update: {e}")
        except Exception as e:
            print(f"Error in progress callback: {e}")

class NetworkDistributedClient:
    def __init__(self, server_host: str, server_port: int = 8888, connection_timeout: int = 30, retry_interval: int = 5):
        self.server_host = server_host
        self.server_port = server_port
        self.connection_timeout = connection_timeout
        self.retry_interval = retry_interval

    def connect(self):
        """Keep attempting to connect to the server until successful"""
        if hasattr(self, 'sock') and self.sock:
            try:
                self.sock.close()
            except:
                pass
        
        attempt = 1
        while True:
            try:
                print(f"[Attempt {attempt}] Trying to connect to {self.server_host}:{self.server_port}...")
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(self.connection_timeout)
                s.connect((self.server_host, self.server_port))
                s.settimeout(None)
                self.sock = s
                print(f"Successfully connected to {self.server_host}:{self.server_port}")
                break
            except (ConnectionRefusedError, socket.timeout, OSError) as e:
                print(f"Connection failed: {type(e).__name__}. Retrying in {self.retry_interval} seconds...")
                time.sleep(self.retry_interval)
                attempt += 1
            except KeyboardInterrupt:
                print("\nConnection cancelled by user")
                sys.exit(0)

    def receive_package(self):
        pkg = pickle.loads(recv_bytes(self.sock))
        self.x_train, self.y_train = pkg['x_train'], pkg['y_train']
        self.client_id = pkg['client_id']
        cfg = pkg['model_config']
        self.epochs = pkg.get('epochs', 5)  # Get epochs from server, default to 5

        labels = np.argmax(self.y_train, axis=1)
        total = len(labels)
        unique, counts = np.unique(labels, return_counts=True)
        dist = dict(zip(unique, counts))
        num_classes = len(unique)
        print(f"\n=== Client {self.client_id} Data Summary ===")
        print(f"Total samples: {total}")
        for c in range(num_classes):
            print(f"  {c}: {dist.get(c, 0)}")
        print("=========================================\n")

        self.model = model_from_json(cfg['model_json'])
        self.model.set_weights(cfg['initial_weights'])
        opt = tf.keras.optimizers.Adam(learning_rate=5e-4)
        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, epochs=5, batch_size=64, validation_split=0.1):
        # Add progress callback along with other callbacks
        progress_callback = EpochProgressCallback(self.sock, self.client_id)
        callbacks = [
            progress_callback,
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
    
    def send_crash_notification(self):
        """Notify server that this client is crashing/disconnecting abnormally"""
        try:
            crash_msg = {
                'type': 'crash',
                'client_id': getattr(self, 'client_id', None),
                'message': 'Client crashed or interrupted'
            }
            send_bytes(self.sock, pickle.dumps(crash_msg, protocol=pickle.HIGHEST_PROTOCOL))
            print(f"Crash notification sent to server for client {getattr(self, 'client_id', 'unknown')}")
        except Exception as e:
            print(f"Could not send crash notification: {e}")

    def run_single_session(self, epochs=None, batch_size=64):
        """Run a single training session"""
        try:
            self.connect()
            self.receive_package()
            # Use epochs from server package if not provided
            epochs_to_use = epochs if epochs is not None else getattr(self, 'epochs', 5)
            history = self.train(epochs=epochs_to_use, batch_size=batch_size)
            self.send_weights_and_history(history)
            self.sock.close()
        except KeyboardInterrupt:
            # User interrupted - send crash notification
            print("\n! Client interrupted by user")
            if hasattr(self, 'sock') and self.sock:
                self.send_crash_notification()
                self.sock.close()
            raise
        except Exception as e:
            print(f"\n! Client crashed with error: {e}")
            if hasattr(self, 'sock') and self.sock:
                self.send_crash_notification()
                self.sock.close()
            raise
    
    def run_continuous(self, epochs=None, batch_size=64):
        """Continuously run training sessions - reconnect after each completion"""
        session_count = 0
        print("=" * 60)
        print("CONTINUOUS MODE: Client will reconnect after each session")
        print("=" * 60)
        print()
        
        while True:
            try:
                session_count += 1
                print(f"\n{'='*60}")
                print(f"SESSION {session_count}: Waiting for server...")
                print(f"{'='*60}\n")
                
                self.run_single_session(epochs=epochs, batch_size=batch_size)
                
                print(f"\n{'='*60}")
                print(f"SESSION {session_count} COMPLETED")
                print(f"{'='*60}")
                wait_time = max(self.retry_interval, 10) 
                print(f"Waiting {wait_time} seconds before listening for next session...\n")
                time.sleep(wait_time)
                
            except KeyboardInterrupt:
                print(f"\n\n{'='*60}")
                print(f"Client stopped by user after {session_count} session(s)")
                print(f"{'='*60}")
                sys.exit(0)
            except Exception as e:
                print(f"\nError in session {session_count}: {e}")
                print(f"Retrying in {self.retry_interval} seconds...")
                time.sleep(self.retry_interval)

def main():
    if len(sys.argv) < 2:
        print("Usage: client.py <server_host> [server_port] [retry_interval]")
        print("  server_host: IP address or hostname of the server")
        print("  server_port: Port number (default: 8888)")
        print("  retry_interval: Seconds between connection attempts (default: 5)")
        sys.exit(1)
    host = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) >= 3 else 8888
    retry_interval = int(sys.argv[3]) if len(sys.argv) >= 4 else 5
    print(f"  Federated Learning Client")
    print(f"  Server: {host}:{port}")
    print(f"  Retry interval: {retry_interval}s")
    print(f"  Mode: Continuous (will reconnect after each session)")
    print(f"  Press Ctrl+C to stop\n")
    # Don't hardcode epochs - let the server decide
    NetworkDistributedClient(host, port, retry_interval=retry_interval).run_continuous(epochs=None, batch_size=64)

if __name__ == "__main__":
    main()
