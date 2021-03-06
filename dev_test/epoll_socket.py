import socket, select
import multiprocessing as mp
import time


def server():
    EOL1 = b"\n\n"
    EOL2 = b"\n\r\n"
    response = b"HTTP/1.0 200 OK\r\nDate: Mon, 1 Jan 1996 01:01:01 GMT\r\n"
    response += b"Content-Type: text/plain\r\nContent-Length: 13\r\n\r\n"
    response += b"Hello, world!"

    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    serversocket.bind(("127.0.0.1", 8080))
    serversocket.listen(1)
    serversocket.setblocking(0)

    epoll = select.epoll()
    epoll.register(serversocket.fileno(), select.EPOLLIN)

    try:
        connections = {}
        requests = {}
        responses = {}
        cnt = 0
        stime = time.time()
        while True:
            if cnt > 100000:
                cnt = 0
                print(f"{time.time() - stime:.2f}")
                stime = time.time()
            events = epoll.poll()
            for fileno, event in events:
                cnt += 1
                if fileno == serversocket.fileno():
                    connection, address = serversocket.accept()
                    connection.setblocking(0)
                    epoll.register(connection.fileno(), select.EPOLLIN)
                    connections[connection.fileno()] = connection
                    requests[connection.fileno()] = b""
                    responses[connection.fileno()] = response
                elif event & select.EPOLLIN:
                    requests[fileno] += connections[fileno].recv(1024)
                    connections[fileno].send(b'1')
                    # if EOL1 in requests[fileno] or EOL2 in requests[fileno]:
                        # epoll.modify(fileno, select.EPOLLOUT)
                        # print("-" * 40 + "\n" + requests[fileno].decode()[:-2])
                # elif event & select.EPOLLOUT:
                #     byteswritten = connections[fileno].send(responses[fileno])
                #     responses[fileno] = responses[fileno][byteswritten:]
                #     if len(responses[fileno]) == 0:
                #         epoll.modify(fileno, 0)
                #         connections[fileno].shutdown(socket.SHUT_RDWR)
                elif event & select.EPOLLHUP:
                    epoll.unregister(fileno)
                    connections[fileno].close()
                    del connections[fileno]
    finally:
        epoll.unregister(serversocket.fileno())
        epoll.close()
        serversocket.close()


def client():
    time.sleep(1)
    HOST = "127.0.0.1"  # The server's hostname or IP address
    PORT = 8080  # The port used by the server

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        while True:
            # print('ss1')
            s.send(b"1")
            data = s.recv(1024)
            # print(len(data))


def main():
    p = mp.Process(target=server, args=())
    p.daemon = True
    p.start()

    for _ in range(100):
        p = mp.Process(target=client, args=())
        p.daemon = True
        p.start()
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
