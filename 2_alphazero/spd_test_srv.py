from multiprocessing.connection import Listener
from io import BytesIO
import numpy as np

address = ('0.0.0.0', 16001)     # family is deduced to be 'AF_INET'
result = np.random.randn(128,15*15)
with Listener(address, authkey=b'secret password123') as listener:
    while True:
        with listener.accept() as conn:
            for _ in range(10):
                command, data = conn.recv()
                if command == 'infer':
                    conn.send(result)
                elif command == "loadonnx":
                    conn.send('ok')
                else:
                    print('invalid cmd:', command, data)
