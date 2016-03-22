import cgt
from cgt import nn, utils
import numpy as np, numpy.random as nr
from numpy.linalg import norm
from param_collection import ParamCollection

k_in = 1
size_x = 3
size_mem = 4
size_batch = 4

x = cgt.matrix(fixed_shape=(size_batch, size_x))
prev_h = cgt.matrix(fixed_shape=(size_batch, size_mem))
r_vec = nn.Affine(size_x, 2 * k_in * size_mem)(x)
r_non = cgt.reshape(r_vec, (size_batch, 2 * k_in, size_mem))
r_norm = cgt.norm(r_non, axis=2, keepdims=True)
r = cgt.broadcast('/', r_non, r_norm, "xxx,xx1")
prev_h_3 = cgt.reshape(prev_h, (size_batch, size_mem, 1))
inters = [prev_h_3]

for i in xrange(k_in * 2):
    inter_in = inters[-1]
    r_cur = r[:, i, :]
    r_cur_3_transpose = cgt.reshape(r_cur, (size_batch, 1, size_mem))
    r_cur_3 = cgt.reshape(r_cur, (size_batch, size_mem, 1))
    ref_cur = cgt.batched_matmul(r_cur_3, cgt.batched_matmul(r_cur_3_transpose, inter_in))
    inter_out = inter_in - ref_cur
    inters.append(inter_out)
h = inters[-1]
    
r_nn = nn.Module([x], [h])


params = r_nn.get_parameters()
pc = ParamCollection(params)
pc.set_value_flat(nr.uniform(-.1, .1, size=(pc.get_total_size(),)))
func = cgt.function([x, prev_h], h)


x_in = nr.uniform(-.1, .1, size=(size_batch * size_x)).reshape(size_batch, size_x)
h_in = np.zeros((size_batch, size_mem))
h_in[:, 0] = np.ones(size_batch)
h = func(x_in, h_in)

