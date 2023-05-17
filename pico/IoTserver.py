import sys
import socket
import threading

# sys.path is a list of absolute path strings
sys.path.append('../tf/util.py')  # <-- relative path

from tf.util import IterDataset as dataset

_STD = 1.638
data = dataset(b"../data/pre_processed2.csv", _STD, 50)

HOST = '192.168.0.246'
PORT = 9090

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))

server.listen()

clients = []
nickname = []


def broadcast(message):
    for client in clients:
        client.send(message)


def handle(client, address):
    while True:
        try:
            message = client.recv(1024).decode('utf-8')
            print(f"Client {str(address)} says {message}")
            broadcast(message)
        except:
            pass


def receive():
    while True:
        client, address = server.accept()
        print(f"Connected with {str(address)}!")
        clients.append(client)
        broadcast(f"Client {address} connected to the server".encode('utf-8'))
        client.send("Connected to the server".encode('utf-8'))
        handle(client, address)


print("Server running...")

receive()
