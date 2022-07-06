import torch
import torch.multiprocessing as mp


def train(model):
    model.weight[0, 0] = 1.11


if __name__ == "__main__":
    mp.set_start_method('spawn')
    num_processes = 1
    model = torch.nn.Linear(2, 3)
    model.weight.requires_grad = False
    # NOTE: this is required for the ``fork`` method to work
    model.share_memory()
    print(model.weight[0, 0])
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=train, args=(model,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    print(model.weight[0, 0])
