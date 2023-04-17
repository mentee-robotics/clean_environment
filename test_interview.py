import torch as th

import torch

import torch
import numpy as np

class BatchQueue:
    def __init__(self, L, B, F):
        self.L = L
        self.B = B
        self.F = F
        self.tensor = torch.zeros((B, L, F))
        self.heads = np.zeros(B)
        self.tails = np.zeros(B)
        self.indices = np.indices(self.tensor.shape)

    def dequeue(self, batch_indices):
        retval = self.tensor[batch_indices, self.heads[batch_indices], :]
        self.heads[batch_indices] = self.heads[batch_indices] + 1 % self.L
        return retval

    def enqueue(self, values, batch_indices):
        T, N, F = values.shape
        tails_on_indices = self.tails[batch_indices] + T
        tails_on_indices_cropped = np.clip(tails_on_indices, 0, self.L)
        items_to_add_per_index_at_end = tails_on_indices_cropped - self.tails[batch_indices]

        mask_to_add_at_end = self.indices[0][batch_indices, :, :] >= np.repeat(np.repeat(self.tails[batch_indices, np.newaxis, np.newaxis], T, axis=1), F, axis=2) &\
                             self.indices[0][batch_indices, :, :] <= np.repeat(np.repeat(tails_on_indices_cropped[:, np.newaxis, np.newaxis], T, axis=1), F, axis=2)
        mask_values_to_add_at_end = np.indices(values.shape)[0] <\
                                    np.repeat(np.repeat(items_to_add_per_index_at_end[np.newaxis, :, np.newaxis], T, axis=0), F, axis=2)
        self.tensor[:, batch_indices, :][mask_to_add_at_end] = values[mask_values_to_add_at_end]

        mask_to_add_at_beginning = self.indices[0][:, batch_indices, :] < items_to_add_per_index_at_end[np.newaxis, :, np.newaxis]
        self.tensor[:, batch_indices, :][mask_to_add_at_beginning] = np.logical_not(mask_values_to_add_at_end)

    def peek(self,location, batch_indices):
        heads_mask = location == 'head'
        heads_at_indices = self.tensor[:, batch_indices, :][self.heads[batch_indices]]
        tails_at_indices = self.tensor[:, batch_indices, :][self.tails[batch_indices]]



# dims = th.Tensor([5, 4, 3])
test_queue = BatchQueue(*[5, 4, 3])
test_queue.enqueue(3 * torch.ones((1, 1, 3)), [0])
test_queue.enqueue(3 * torch.ones((1, 1, 3)), [1])
test_queue.enqueue(3 * torch.ones((1, 1, 3)), [2])
test_queue.enqueue(3 * torch.ones((1, 1, 3)), [3])
test_queue.enqueue(1 * torch.ones((2, 1, 3)), [0])

print('w')

x = test_queue.dequeue([0])
print('w')

print(f"before enqueue the tails are {test_queue.tails}")
L=2
B=3
F=3
batch_indices = th.Tensor([0,1,3])

test_queue.dequeue(th.tensor([0, 1, 2, 3]))
#
# assert batch_indices.shape[0] == B, "batch_indices and B (BATCH_SIZE) mismatch"
#
# new_batch_queue = th.reshape(th.arange(start=0, end=B*L*F, step=1).type(th.float), (B,L,F))
# print(f"New 3D tensor to enqueue:\n {new_batch_queue}\n")
#
# #.resize((B, L, F))
# test_queue.enqueue(new_batch_queue, batch_indices)
# print(f"resulting queue:\n{test_queue.queue}\n")
# new_batch_queue = torch.reshape(10 * torch.arange(start=0, end=B*L*F, step=1).type(torch.float), (B,L,F))
# test_queue.enqueue(new_batch_queue, batch_indices)
# print(f"resulting queue:\n{test_queue.queue}\n")
#
# print(test_queue.dequeue(torch.tensor([0, 1, 2, 3])))
#
# print(f"after enqueue the tails are {test_queue.tails}")
