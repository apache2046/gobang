from multiprocessing.connection import Client
import numpy as np
from mcts5 import MCTS
from game2 import GoBang
import multiprocessing as mp
import time
import pickle
from model4 import Policy_Value
import torch
import ray
import io
import random
from collections import deque
import os
import socket
import traceback
import trtcommon
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
from multiprocessing.connection import Client
from io import BytesIO

print(socket.gethostname(), os.getcwd())
os.environ["RAY_LOG_TO_STDERR"] = "1"
ray.init(address="auto", _node_ip_address="192.168.5.106")
# ray.init(address="auto")
# ray.init(address='ray://192.168.5.7:10001')

# with Client(address, authkey=b'secret password') as conn:
#     conn.send(np.arange(12, dtype=np.int8).reshape(3,4))
#     print(conn.recv())


def executeEpisode(game, epid):
    # mcts = MCTS(game, c_puct=0.5)
    state = game.start_state()
    samples = []
    cnt = 0
    board_record = np.zeros((game.size, game.size), dtype=np.int8)
    stime = time.time()
    tau = 0.8
    while True:
        mcts = MCTS(game, c_puct=5)
        cnt += 1
        if cnt > 2:
            # tau = max(0.05, tau * 0.85)
            tau = max(0.05, tau * 0.9)
        if epid == 0:
            print("GHB", cnt, f"tau:{tau:.2f}, {time.time()-stime: .2f}")
        stime = time.time()
        for i in range(2000):
            yield from mcts.search(state)
        pi = mcts.pi(state, tau)
        samples.append([state, pi, None])
        action = np.random.choice(len(pi), p=pi)
        next_state, isend, reward = game.next_state(state, action)
        y = action // game.size
        x = action % game.size
        board_record[y, x] = cnt
        if isend:
            v = reward
            for j in reversed(range(len(samples))):
                samples[j][2] = v
                v = -v
            return samples, board_record
        else:
            state = next_state


def executeEpisodeEndless(epid, tainer):
    game = GoBang(size=15)
    while True:
        print("executeEpisodeEndless1", epid)
        trajectory, board_record = yield from executeEpisode(game, epid)
        print(
            f"{epid} got trajectory",
            "\n",
            board_record,
            "\n",
            trajectory[-1][2],
            trajectory[-2][2],
            trajectory[-3][2],
            trajectory[-4][2],
        )
        tainer.push_samples.remote(trajectory)


@ray.remote(num_cpus=1)
def simbatch(infer_service, tainer):
    try:
        states = []
        g = []
        print("GHB in simbatch")
        for i in range(128):
            # np.random.seed(100+i)

            item = executeEpisodeEndless(i, tainer)
            g.append(item)
            states.append(next(item))

        while True:
            # print('before remote1')
            prob, v = ray.get(infer_service.infer.remote(inputdata=states))
            # print(prob, v)
            for i in range(len(g)):
                # try:
                states[i] = g[i].send((prob[i], v[i]))
                # except StopIteration:
                #     g[i] = executeEpisodeEndless(i, tainer)
                #     states[i] = next(g[i])
    except Exception:
        print(traceback.format_exc())


@ray.remote(num_cpus=0.1, num_gpus=0.1)
class Infer_srv:
    def __init__(self):
        self.abc = 123
        pass

    def infer(self, inputdata:None):
        try:
            # print("infer", type(inputdata))
            bs = len(inputdata)
            inputdata = np.stack(inputdata).transpose([0, 3, 1, 2]).astype(np.float32).ravel()
            output1 = np.empty((bs, 15*15), np.float32)
            output2 = np.empty((bs, 1), np.float32)

            cuda.memcpy_htod_async(self.bindings[0], inputdata, self.stream)
            # Run inference.
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            # Transfer predictions back from the GPU.
            cuda.memcpy_dtoh_async(output1, int(self.bindings[1]), self.stream)
            cuda.memcpy_dtoh_async(output2, int(self.bindings[2]), self.stream)
            # Synchronize the stream
            self.stream.synchronize()
            
            prob = output1
            v = output2
        except Exception:
            print(traceback.format_exc())
        
        return prob, v

    def load_onnx(self, model_bytes):
        try:
            print("G1 in load_onnx")
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(trtcommon.EXPLICIT_BATCH)
            config = builder.create_builder_config()
            config.set_flag(trt.BuilderFlag.FP16)
            parser = trt.OnnxParser(network, TRT_LOGGER)

            config.max_workspace_size = trtcommon.GiB(1)
            # Load the Onnx model and parse it in order to populate the TensorRT network.
            # with open(model_file, "rb") as model:
            if not parser.parse(model_bytes):
                print("ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
            engine = builder.build_engine(network, config)

            self.bindings = []

            for binding in engine:
                size = (
                    trt.volume(engine.get_binding_shape(binding))
                    * engine.max_batch_size
                )
                dtype = trt.nptype(engine.get_binding_dtype(binding))
                # Allocate host and device buffers
                # host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(size * 4)
                # Append the device buffer to device bindings.
                self.bindings.append(device_mem)

                # Append to the appropriate list.
                if engine.binding_is_input(binding):
                    print("inputmem")
                else:
                    print("outputmem")
                print(engine.get_binding_shape(binding), size * 4)

            self.engin = engine
            self.context = engine.create_execution_context()
            self.stream = cuda.Stream()
            print("G2 in load_onnx")
        except Exception:
            print(traceback.format_exc())



def get_onnx_bytes_from_remote(model):
    with Client(('192.168.5.6', 6000) , authkey=b'secret password123') as conn:
        f = BytesIO()
        # model.load_state_dict(torch.load('models/224.pt'))
        model.eval()
        torch.save(model.state_dict(), f, _use_new_zipfile_serialization=False)
        print('len:', len(f.getvalue()))
        conn.send(f.getvalue())
        conn.send(128)
        onnxbytes = conn.recv()
        return onnxbytes


@ray.remote(num_cpus=0.1, num_gpus=0.2)
class Train_srv:
    def __init__(self):
        print("Train1100")
        pass
    def myinit(self, infer_services:None):
        try:
            print("Train110")
            self.nnet = Policy_Value().to("cuda:0")
            self.opt = torch.optim.AdamW(params=self.nnet.parameters(), lr=1e-4)
            self.infer_services = infer_services
            self.batchsize = 1024
            self.mse_loss = torch.nn.MSELoss()
            self.kl_loss = torch.nn.KLDivLoss()
            self.samples = deque(maxlen=50000)
            self.sn = 0
            self.epoch = 0
            onnxbytes = get_onnx_bytes_from_remote(self.nnet)
            ray.wait([item.load_onnx.remote(onnxbytes) for item in self.infer_services])
            print("GGG after init")
        except Exception:
            print(traceback.format_exc())

    def _train(self):
        print("Train11")
        opt = self.opt
        batch = random.choices(self.samples, k=self.batchsize)
        states = []
        pis = []
        vs = []
        cnt = 4
        for item in batch:
            # print(type(item[0]), item[0])
            # sys.exit()
            s = torch.tensor(item[0], dtype=torch.float32)
            pi = torch.tensor(item[1], dtype=torch.float32).reshape(15, 15)
            v = torch.tensor(item[2], dtype=torch.float32)
            if cnt > 0:
                # print("tsample", s, pi, v)
                cnt -= 1
            r = random.choice([-1, 0, 1, 2])
            torch.rot90(s, r)
            torch.rot90(pi, r)
            if random.random() > 0.5:
                s = s.flip(0)
                pi = pi.flip(0)
            if random.random() > 0.5:
                s = s.flip(1)
                pi = pi.flip(1)
            pi = pi.flatten()

            states.append(s)
            pis.append(pi)
            vs.append(v)
        states = torch.stack(states).permute(0, 3, 1, 2).to("cuda:0")
        pis = torch.stack(pis).to("cuda:0")
        vs = torch.vstack(vs).to("cuda:0")

        self.nnet.train()
        pred_pis, pred_vs = self.nnet(states)
        # pi_loss = -torch.mean(
        #    pis.matmul(torch.log(torch.clip(pred_pis, 1e-9, 1 - 1e-9).transpose(0, 1)))
        # )
        pi_loss = -torch.mean(
            (pis * torch.log(torch.clip(pred_pis, 1e-9, 1 - 1e-9))).sum(1)
        )
        v_loss = self.mse_loss(pred_vs, vs)
        print(f"loss: {pi_loss.tolist():.03f}, {v_loss.tolist():.03f}")
        loss = pi_loss + v_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        print("Train12")

    def push_samples(self, samples):
        try:
            self.samples.extend(samples)
            self.sn += 1
            if self.sn == 400:
                self.sn = 0
                self.train()
        except Exception:
            print(traceback.format_exc())

    def train(self):
        print("Train1")
        for i in range(70):
            self._train()
        print("Train2")
        time.sleep(4)
        torch.save(self.nnet.state_dict(), f"models/{self.epoch}.pt")
        print(f"saved {self.epoch}.pt file...")
        onnxbytes = get_onnx_bytes_from_remote(self.nnet)
        ray.wait([item.load_onnx.remote(onnxbytes) for item in self.infer_services])
        print("Train3")
        self.epoch += 1


def main():
    print("GHB1")
    infer_services = [Infer_srv.remote() for _ in range(2)]
    print("GHB2")
    tainer = Train_srv.remote()
    ray.wait([tainer.myinit.remote(infer_services=infer_services)])
    print("GHB3")
    s = []
    for i in range(52):
        s.append(simbatch.remote(infer_services[i % 2], tainer))
    print("GHB4")
    ray.wait(s)
    print("GHB5")
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
