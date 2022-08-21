import socket

clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clientsocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
clientsocket.bind(("0.0.0.0", 2022))

# while True:
connection = clientsocket.connect(("192.168.5.106", 8080))
my_bytes = bytearray()
my_bytes.append(123)
my_bytes.append(124)
my_bytes.append(125)
my_bytes.append(126)

clientsocket.send(my_bytes)
data = clientsocket.recv(4)
print(data)
