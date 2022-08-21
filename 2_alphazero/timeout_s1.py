import socket

serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serversocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
serversocket.bind(("0.0.0.0", 8080))
serversocket.listen(10)

while True:
    connection, address = serversocket.accept()
    data = connection.recv(4)
    print(data)
    connection.send(data)
    connection = None
