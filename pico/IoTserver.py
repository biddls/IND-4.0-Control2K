from time import sleep
from util import IterDataset as dataset, round_table
from loadModel import getModel
import socket
import json

_STD = 1.638
data = dataset(b"../data/pre_processed2.csv", _STD, 50)

class Server:
    def __init__(self, host, port, data):
        self.HOST = host
        self.PORT = port
        self.server = None
        self.client = None
        self.nickname = []
        self.data = data.__iter__()
        self.model = getModel('../tf/model')
        self.CHUNK_SIZE = 1024

    def broadcast(self, message: str) -> str:
        # Split message into chunks
        chunks = [message[i:i + self.CHUNK_SIZE] for i in range(0, len(message), self.CHUNK_SIZE)]
        print(f"{len(chunks)=}")
        for chunk in chunks:
            self.client.sendall(chunk.encode('utf-8'))
        _message = self.client.recv(1).decode('utf-8')
        if _message != "1":
            return False

        return True

    def handle(self):
        while True:
            try:
                success = False
                while not success:
                    x, _ = self.data.__next__()

                    packet = {
                        "y": round_table(self.model(x).numpy().tolist(), decimals=5)[0],
                        "data": round_table(x.numpy()[0].tolist(), decimals=5)
                    }
                    data_string = json.dumps(packet)  # data serialized
                    success = self.broadcast(data_string)
                    sleep(5)
            except ConnectionError:
                return
            except TimeoutError:
                return

    def receive(self):
        while True:
            self.client = None
            self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server.bind((self.HOST, self.PORT))
            self.server.listen()

            while self.client is None:
                try:
                    self.client, address = self.server.accept()
                except TimeoutError as e:
                    print(e)

            print(f"Connected with {str(address)}")

            self.handle()

    def run(self):
        print("Server running...")
        self.receive()

if __name__ == "__main__":

    # Usage
    HOST = '192.168.0.246'
    PORT = 9090

    server = Server(HOST, PORT, data)
    server.run()
