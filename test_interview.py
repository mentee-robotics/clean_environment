import torch as th

class BatchQueue:
    def __init__(self,dims):
        self.batch_queue = th.zeros(dims[1],dims[0],dims[2])
        self.batch_head = th.zeros(dims[1],dtype=int) - 1

    def dequeue(self, batch_indices):
        removed_instance=self.batch_queue[batch_indices,self.batch_head[batch_indices]]
        self.batch_queue[batch_indices,self.batch_head[batch_indices]] *= th.tensor(0)
        self.batch_head[batch_indices] -= 1
        return removed_instance

    def enqueue(self, values, batch_indices):
        N,T,F = th.tensor(values.shape[1]),th.tensor(values.shape[0]),th.tensor(values.shape[2])
        self.batch_queue[batch_indices] = th.roll(self.batch_queue[batch_indices],int(T),1)
        self.batch_head[batch_indices] += T
        values = th.swapaxes(values,1,0)
        self.batch_queue[batch_indices,0:T] = values


    def peek(self, location, batch_indices):
        result = [0 if x == "Tail" else self.batch_head[batch_indices[i]] for i,x in enumerate(location)]
        output = self.batch_queue[batch_indices,result]
        return output


# dims = th.Tensor([5, 4, 3])
test_queue = BatchQueue([5, 4, 3])
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
