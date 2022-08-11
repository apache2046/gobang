# Should run in pytorch 1.2
import torch
from multiprocessing.connection import Listener
#from model4 import Policy_Value
#from model6 import Policy_Value
from model8 import Policy_Value
from io import BytesIO
from datetime import datetime

address = ('0.0.0.0', 6000)     # family is deduced to be 'AF_INET'
model = Policy_Value()

with Listener(address, authkey=b'secret password123', backlog=100) as listener:
    while True:
        with listener.accept() as conn:
            print('connection accepted from', listener.last_accepted)
            print(datetime.now().strftime("%D %H:%M:%S"))
            state_bytes = conn.recv()
            print(type(state_bytes), len(state_bytes))
            batch_size = conn.recv()
            state_dict = torch.load(BytesIO(state_bytes), map_location='cpu')
            model.load_state_dict(state_dict)
            #dummy_input = torch.randn(batch_size, 5, 15, 15).to(torch.float32)
            dummy_input = torch.randn(batch_size, 3, 15, 15).to(torch.float32)
            f = BytesIO()
            torch.onnx.export(
                model,
                dummy_input,
                f,
                export_params=True,  # store the trained parameter weights inside the model file
                # do_constant_folding=False,
                # training=2,
                #operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
                # verbose=True,
                opset_version=9,
                #keep_initializers_as_inputs=False,
                # export_modules_as_functions=True,
                input_names=["input"],
                output_names=["prob", "v"]
            )
            print('done', len(f.getvalue()))
            print('\n')
            conn.send(f.getvalue())
