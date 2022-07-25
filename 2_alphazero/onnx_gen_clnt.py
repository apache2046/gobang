from multiprocessing.connection import Client
import model4
import torch
from io import BytesIO
m = model4.Policy_Value()

with Client(('192.168.5.6', 6000) , authkey=b'secret password123') as conn:
    f = BytesIO()
    torch.save(m.state_dict(), 'g1.pt')
    torch.save(m.state_dict(), f, _use_new_zipfile_serialization=False)
    st = torch.load(BytesIO(f.getvalue()))
    print('S0')
    m.load_state_dict(st)
    print('S1')
    conn.send(f.getvalue())
    conn.send(256)
    onnxbytes = conn.recv()
    print(len(onnxbytes), type(onnxbytes))
    with open('g1.onnx', 'wb') as f:
        f.write(onnxbytes)