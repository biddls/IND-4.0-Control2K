import network
# import socket
from time import sleep
# from picozero import pico_led
import machine


ssid = 'Biddls'
password = 'evilkid1066'


def connect():
    #Connect to WLAN
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    wlan.connect(ssid, password)
    while wlan.isconnected() == False:
        print('Waiting for connection...')
        sleep(1)
    ip = wlan.ifconfig()[0]
    print(f'Connected on {ip}')
    return ip


# def open_socket(ip):
#     # Open a socket
#     HOST = '127.0.0.1'
#     PORT = 9090
#     address = (HOST, PORT)
#     connection = socket.socket()
#     connection.bind(address)
#     connection.listen(1)
#     print(connection)
#     return connection
#
# def webpage(state):
#     #Template HTML
#     html = """<!DOCTYPE html>"""
#     return str(html)
#
# def serve(connection):
#     pico_led.off()
#     while True:
#         client = connection.accept()[0]
#
#         data = client.recv(BUFFER_SIZE)
#         client.close()
#
#         print("received data:", data)
#
try:
    ip = connect()
#     connection = open_socket(ip)
#     serve(connection)
except KeyboardInterrupt:
    machine.reset()
import socket

HOST = '192.168.0.246'
PORT = 9090

BUFFER_SIZE = 1024
MESSAGE = "Hello, World!"

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))
s.send(MESSAGE)
data = s.recv(BUFFER_SIZE)

print("received data:", data)
