"""
A nearly direct translation of Andrej's code
https://github.com/karpathy/char-rnn
"""

from __future__ import division
import cgt
from cgt import nn, utils, profiler
import numpy as np, numpy.random as nr
import os.path as osp
import argparse
from time import time
from StringIO import StringIO
from param_collection import ParamCollection
from IPython import embed
from numpy.linalg import norm

# via https://github.com/karpathy/char-rnn/blob/master/model/GRU.lua
# via http://arxiv.org/pdf/1412.3555v1.pdf
def make_deep_gru(size_input, size_mem, n_layers, size_output, size_batch):
    inputs = [cgt.matrix() for i_layer in xrange(n_layers+1)]
    outputs = []
    for i_layer in xrange(n_layers):
        prev_h = inputs[i_layer+1] # note that inputs[0] is the external input, so we add 1
        x = inputs[0] if i_layer==0 else outputs[i_layer-1]
        size_x = size_input if i_layer==0 else size_mem
        update_gate = cgt.sigmoid(
            nn.Affine(size_x, size_mem,name="i2u")(x)
            + nn.Affine(size_mem, size_mem, name="h2u")(prev_h))
        reset_gate = cgt.sigmoid(
            nn.Affine(size_x, size_mem,name="i2r")(x)
            + nn.Affine(size_mem, size_mem, name="h2r")(prev_h))
        gated_hidden = reset_gate * prev_h
        p2 = nn.Affine(size_mem, size_mem)(gated_hidden)
        p1 = nn.Affine(size_x, size_mem)(x)
        hidden_target = cgt.tanh(p1+p2)
        next_h = (1.0-update_gate)*prev_h + update_gate*hidden_target
        outputs.append(next_h)
    category_activations = nn.Affine(size_mem, size_output,name="pred")(outputs[-1])
    logprobs = nn.logsoftmax(category_activations)
    outputs.append(logprobs)

    return nn.Module(inputs, outputs)

def make_deep_lstm(size_input, size_mem, n_layers, size_output, size_batch):
    inputs = [cgt.matrix(fixed_shape=(size_batch, size_input))]
    for _ in xrange(2*n_layers):
        inputs.append(cgt.matrix(fixed_shape=(size_batch, size_mem)))
    outputs = []
    for i_layer in xrange(n_layers):
        prev_h = inputs[i_layer*2]
        prev_c = inputs[i_layer*2+1]
        if i_layer==0:
            x = inputs[0]
            size_x = size_input
        else:
            x = outputs[(i_layer-1)*2]
            size_x = size_mem
        input_sums = nn.Affine(size_x, 4*size_mem)(x) + nn.Affine(size_x, 4*size_mem)(prev_h)
        sigmoid_chunk = cgt.sigmoid(input_sums[:,0:3*size_mem])
        in_gate = sigmoid_chunk[:,0:size_mem]
        forget_gate = sigmoid_chunk[:,size_mem:2*size_mem]
        out_gate = sigmoid_chunk[:,2*size_mem:3*size_mem]
        in_transform = cgt.tanh(input_sums[:,3*size_mem:4*size_mem])
        next_c = forget_gate*prev_c + in_gate * in_transform
        next_h = out_gate*cgt.tanh(next_c)
        outputs.append(next_c)
        outputs.append(next_h)

    category_activations = nn.Affine(size_mem, size_output)(outputs[-1])
    logprobs = nn.logsoftmax(category_activations)
    outputs.append(logprobs)

    return nn.Module(inputs, outputs)

def make_deep_rrnn_rot_relu(size_input, size_mem, n_layers, size_output, size_batch_in, k_in, k_h):
    inputs = [cgt.matrix() for i_layer in xrange(n_layers+1)]
    outputs = []
    print 'input_size: ', size_input
    for i_layer in xrange(n_layers):
        prev_h = inputs[i_layer+1] # note that inputs[0] is the external input, so we add 1
        x = inputs[0] if i_layer==0 else outputs[i_layer-1]
        size_x = size_input if i_layer==0 else size_mem
        size_batch = prev_h.shape[0]

        xform_h_param = nn.TensorParam((2 * k_h, size_mem), name="rotxform")
        xform_h_non = xform_h_param.weight
        xform_h_non.props["is_rotation"] = True

        xform_h_norm = cgt.norm(xform_h_non, axis=1, keepdims=True)
        xform_h = cgt.broadcast('/', xform_h_non, xform_h_norm, "xx,x1")

        add_in_lin = nn.Affine(size_x, size_mem)(x)
        add_in_relu = nn.rectify(add_in_lin)

        prev_h_scaled = nn.scale_mag(prev_h)


        h_in_added = prev_h_scaled + add_in_relu
        inters_h = [h_in_added]

        colon = slice(None, None, None)

        for i in xrange(2 * k_h):
            inter_in = inters_h[-1]
            r_cur = xform_h[i, :]
            #r_cur = cgt.subtensor(xform_h, [i, colon])
            r_cur_2_transpose = cgt.reshape(r_cur, (size_mem, 1))
            r_cur_2 = cgt.reshape(r_cur, (1, size_mem))
            ref_cur = cgt.dot(cgt.dot(inter_in, r_cur_2_transpose), r_cur_2)
            inter_out = inter_in - 2 * ref_cur
            inters_h.append(inter_out)
        next_h = inters_h[-1]
        outputs.append(next_h)


    category_activations = nn.Affine(size_mem, size_output,name="pred")(outputs[-1])
    logprobs = nn.logsoftmax(category_activations)
    outputs.append(logprobs)

    #print 'len outputs:', len(outputs)
    #print 'len inputs:', len(inputs)

    return nn.Module(inputs, outputs)

def make_deep_rrnn(size_input, size_mem, n_layers, size_output, size_batch_in, k_in, k_h):
    inputs = [cgt.matrix() for i_layer in xrange(n_layers+1)]
    outputs = []
    print 'input_size: ', size_input
    for i_layer in xrange(n_layers):
        prev_h = inputs[i_layer+1] # note that inputs[0] is the external input, so we add 1
        x = inputs[0] if i_layer==0 else outputs[i_layer-1]
        size_x = size_input if i_layer==0 else size_mem
        size_batch = prev_h.shape[0]

        xform_h_param = nn.TensorParam((2 * k_h, size_mem), name="rotxform")
        xform_h_non = xform_h_param.weight
        xform_h_non.props["is_rotation"] = True
        xform_h_norm = cgt.norm(xform_h_non, axis=1, keepdims=True)
        xform_h = cgt.broadcast('/', xform_h_non, xform_h_norm, "xx,x1")

        r_vec = nn.Affine(size_x, 2 * k_in * size_mem)(x)
        r_non = cgt.reshape(r_vec, (size_batch, 2 * k_in, size_mem))
        r_norm = cgt.norm(r_non, axis=2, keepdims=True)
        r = cgt.broadcast('/', r_non, r_norm, "xxx,xx1")
        prev_h_3 = cgt.reshape(prev_h, (size_batch, size_mem, 1))
        inters_in = [prev_h_3]

        colon = slice(None, None, None)

        for i in xrange(2 * k_in):
            inter_in = inters_in[-1]
            r_cur = cgt.subtensor(r, [colon, i, colon])
            r_cur_3_transpose = cgt.reshape(r_cur, (size_batch, 1, size_mem))
            r_cur_3 = cgt.reshape(r_cur, (size_batch, size_mem, 1))
            ref_cur = cgt.batched_matmul(r_cur_3, cgt.batched_matmul(r_cur_3_transpose, inter_in))
            inter_out = inter_in - 2 * ref_cur
            inters_in.append(inter_out)

        h_in_rot = cgt.reshape(inters_in[-1], (size_batch, size_mem))
        inters_h = [h_in_rot]

        for i in xrange(2 * k_h):
            inter_in = inters_h[-1]
            r_cur = cgt.subtensor(xform_h, [i, colon])
            r_cur_2_transpose = cgt.reshape(r_cur, (size_mem, 1))
            r_cur_2 = cgt.reshape(r_cur, (1, size_mem))
            ref_cur = cgt.dot(cgt.dot(inter_in, r_cur_2_transpose), r_cur_2)
            inter_out = inter_in - 2 * ref_cur
            inters_h.append(inter_out)
        next_h = inters_h[-1]
        outputs.append(next_h)


    category_activations = nn.Affine(size_mem, size_output,name="pred")(outputs[-1])
    logprobs = nn.logsoftmax(category_activations)
    outputs.append(logprobs)

    #print 'len outputs:', len(outputs)
    #print 'len inputs:', len(inputs)

    return nn.Module(inputs, outputs)

def flatcat(xs):
    return cgt.concatenate([x.flatten() for x in xs])

def cat_sample(ps):    
    """
    sample from categorical distribution
    ps is a 2D array whose rows are vectors of probabilities
    """
    r = nr.rand(len(ps))
    out = np.zeros(len(ps),dtype='i4')
    cumsums = np.cumsum(ps, axis=1)
    for (irow,csrow) in enumerate(cumsums):
        for (icol, csel) in enumerate(csrow):
            if csel > r[irow]:
                out[irow] = icol
                break
    return out


def rmsprop_update(grad, state):
    state.sqgrad[:] *= state.decay_rate
    state.count *= state.decay_rate

    np.square(grad, out=state.scratch) # scratch=g^2

    state.sqgrad += state.scratch
    state.count += 1

    np.sqrt(state.sqgrad, out=state.scratch) # scratch = sum of squares
    np.divide(state.scratch, np.sqrt(state.count), out=state.scratch) # scratch = rms
    np.divide(grad, state.scratch, out=state.scratch) # scratch = grad/rms
    np.multiply(state.scratch, state.step_size, out=state.scratch)
    state.theta[:] -= state.scratch

def make_loss_and_grad_and_step(arch, size_input, size_output, size_mem, size_batch, n_layers, n_unroll, k_in, k_h):
    # symbolic variables

    x_tnk = cgt.tensor3()
    targ_tnk = cgt.tensor3()
    #make_network = make_deep_lstm if arch=="lstm" else make_deep_gru
    make_network = make_deep_rrnn_rot_relu
    network = make_network(size_input, size_mem, n_layers, size_output, size_batch, k_in, k_h)
    init_hiddens = [cgt.matrix() for _ in xrange(get_num_hiddens(arch, n_layers))]
    # TODO fixed sizes

    cur_hiddens = init_hiddens
    loss = 0
    for t in xrange(n_unroll):
        outputs = network([x_tnk[t]] + cur_hiddens)
        cur_hiddens, prediction_logprobs = outputs[:-1], outputs[-1]
        # loss = loss + nn.categorical_negloglik(prediction_probs, targ_tnk[t]).sum()
        loss = loss - (prediction_logprobs*targ_tnk[t]).sum()
        cur_hiddens = outputs[:-1]

    final_hiddens = cur_hiddens

    loss = loss / (n_unroll * size_batch)

    params = network.get_parameters()
    gradloss = cgt.grad(loss, params)

    flatgrad = flatcat(gradloss)

    with utils.Message("compiling loss+grad"):
        f_loss_and_grad = cgt.function([x_tnk, targ_tnk] + init_hiddens, [loss, flatgrad] + final_hiddens)
    f_loss = cgt.function([x_tnk, targ_tnk] + init_hiddens, loss)

    assert len(init_hiddens) == len(final_hiddens)

    x_nk = cgt.matrix('x')
    outputs = network([x_nk] + init_hiddens)

    f_step = cgt.function([x_nk]+init_hiddens, outputs)

    # print "node count", cgt.count_nodes(flatgrad)
    return network, f_loss, f_loss_and_grad, f_step


class Table(dict):
    "dictionary-like object that exposes its keys as attributes"
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self

def make_rmsprop_state(theta, step_size, decay_rate):
    return Table(theta=theta, sqgrad=np.zeros_like(theta)+1e-6, scratch=np.empty_like(theta), 
        step_size=step_size, decay_rate=decay_rate, count=0)
    
class Loader(object):
    def __init__(self, data_dir, size_batch, n_unroll, split_fractions):
        input_file = osp.join(data_dir,"input.txt")
        preproc_file = osp.join(data_dir, "preproc.npz")
        run_preproc = not osp.exists(preproc_file) or osp.getmtime(input_file) > osp.getmtime(preproc_file)
        if run_preproc:
            text_to_tensor(input_file, preproc_file)
        data_file = np.load(preproc_file)
        self.char2ind = {char:ind for (ind,char) in enumerate(data_file["chars"])}
        data = data_file["inds"]
        data = data[:data.shape[0] - (data.shape[0] % size_batch)].reshape(size_batch, -1).T # inds_tn
        n_batches = (data.shape[0]-1) // n_unroll 
        data = data[:n_batches*n_unroll+1]  # now t-1 is divisble by batch size
        self.n_unroll = n_unroll
        self.data = data

        self.n_train_batches = int(n_batches*split_fractions[0])
        self.n_test_batches = int(n_batches*split_fractions[1])
        self.n_val_batches = n_batches - self.n_train_batches - self.n_test_batches

        print "%i train batches, %i test batches, %i val batches"%(self.n_train_batches, self.n_test_batches, self.n_val_batches)

    @property
    def size_vocab(self):
        return len(self.char2ind)

    def train_batches_iter(self):
        for i in xrange(self.n_train_batches):
            start = i*self.n_unroll
            stop = (i+1)*self.n_unroll
            yield ind2onehot(self.data[start:stop], self.size_vocab), ind2onehot(self.data[start+1:stop+1], self.size_vocab) # XXX


# XXX move elsewhere
def ind2onehot(inds, n_cls):
    inds = np.asarray(inds)
    out = np.zeros(inds.shape+(n_cls,),cgt.floatX)
    out.flat[np.arange(inds.size)*n_cls + inds.ravel()] = 1
    return out


def text_to_tensor(text_file, preproc_file):
    with open(text_file,"r") as fh:
        text = fh.read()
    char2ind = {}
    inds = []
    for char in text:
        ind = char2ind.get(char, -1)
        if ind == -1:
            ind = len(char2ind)
            char2ind[char] = ind
        inds.append(ind)
    np.savez(preproc_file, inds = inds, chars = sorted(char2ind, key = lambda char : char2ind[char]))


def get_num_hiddens(arch, n_layers):
    return {"lstm" : 2 * n_layers, "gru" : n_layers}[arch]

def sample(f_step, init_hiddens, char2ind, n_steps, temp, seed_text = ""):
    vocab_size = len(char2ind)
    ind2char = {ind:char for (char,ind) in char2ind.iteritems()}
    cur_hiddens = init_hiddens
    t = StringIO()
    t.write(seed_text)
    for char in seed_text:
        x_1k = ind2onehot([char2ind[char]], vocab_size)
        net_outputs = f_step(x_1k, cur_hiddens)
        cur_hiddens, logprobs_1k = net_outputs[:-1], net_outputs[-1]

    if len(seed_text)==0:
        logprobs_1k = np.zeros((1,vocab_size))
    

    for _ in xrange(n_steps):        
        logprobs_1k /= temp
        probs_1k = np.exp(logprobs_1k)
        probs_1k /= probs_1k.sum()
        index = cat_sample(probs_1k)[0]
        char = ind2char[index]
        x_1k = ind2onehot([index], vocab_size)
        net_outputs = f_step(x_1k, *cur_hiddens)
        cur_hiddens, logprobs_1k = net_outputs[:-1], net_outputs[-1]
        t.write(char)

    cgt.utils.colorprint(cgt.utils.Color.YELLOW, t.getvalue() + "\n")

def main():

    nr.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="alice")
    parser.add_argument("--size_mem", type=int,default=64)
    parser.add_argument("--size_batch", type=int,default=64)
    parser.add_argument("--n_layers",type=int,default=2)
    parser.add_argument("--n_unroll",type=int,default=16)
    parser.add_argument("--k_in",type=int,default=3)
    parser.add_argument("--k_h",type=int,default=5)
    parser.add_argument("--step_size",type=float,default=.01)
    parser.add_argument("--decay_rate",type=float,default=0.95)
    parser.add_argument("--n_epochs",type=int,default=20)
    parser.add_argument("--arch",choices=["lstm","gru"],default="gru")
    parser.add_argument("--grad_check",action="store_true")
    parser.add_argument("--profile",action="store_true")
    parser.add_argument("--unittest",action="store_true")

    args = parser.parse_args()

    cgt.set_precision("quad" if args.grad_check else "single")

    assert args.n_unroll > 1

    loader = Loader(args.data_dir,args.size_batch, args.n_unroll, (.8,.1,.1))

    network, f_loss, f_loss_and_grad, f_step = make_loss_and_grad_and_step(args.arch, loader.size_vocab, 
        loader.size_vocab, args.size_mem, args.size_batch, args.n_layers, args.n_unroll, args.k_in, args.k_h)

    if args.profile: profiler.start()

    params = network.get_parameters()
    pc = ParamCollection(params)
    pc.set_value_flat(nr.uniform(-0.01, 0.01, size=(pc.get_total_size(),)))

    for i, param in enumerate(pc.params):
        if "is_rotation" in param.props:
            shape = pc.get_shapes()[i]
            num_vec = int(shape[0] / 2)
            size_vec = int(shape[1])
            gauss = nr.normal(size=(num_vec * size_vec))
            gauss = np.reshape(gauss, (num_vec, size_vec))
            gauss_mag = norm(gauss, axis=1, keepdims=True)
            gauss_normed = gauss / gauss_mag
            gauss_perturb = nr.normal(scale=0.01, size=(num_vec * size_vec))
            gauss_perturb = np.reshape(gauss_perturb, (num_vec, size_vec))
            second_vec = gauss_normed + gauss_perturb
            second_vec_mag = norm(second_vec, axis=1, keepdims=True)
            second_vec_normed = second_vec / second_vec_mag
            new_param_value = np.zeros(shape)
            for j in xrange(num_vec):
                new_param_value[2 * j, :] = gauss_normed[j, :]
                new_param_value[2 * j + 1, :] = second_vec_normed[j, :]
            param.op.set_value(new_param_value)
            #print new_param_value



    def initialize_hiddens(n):
        return [np.ones((n, args.size_mem), cgt.floatX) / float(args.size_mem) for _ in xrange(get_num_hiddens(args.arch, args.n_layers))]

    if args.grad_check:
    #if True:
        x,y = loader.train_batches_iter().next()
        prev_hiddens = initialize_hiddens(args.size_batch)
        def f(thnew):
            thold = pc.get_value_flat()
            pc.set_value_flat(thnew)
            loss = f_loss(x,y, *prev_hiddens)
            pc.set_value_flat(thold)
            return loss
        from cgt.numeric_diff import numeric_grad
        print "Beginning grad check"
        g_num = numeric_grad(f, pc.get_value_flat(),eps=1e-10)
        print "Ending grad check"
        result = f_loss_and_grad(x,y,*prev_hiddens)
        g_anal = result[1]
        diff = g_num - g_anal
        abs_diff = np.abs(diff)
        print np.where(abs_diff > 1e-4)
        print diff[np.where(abs_diff > 1e-4)]
        embed()
        assert np.allclose(g_num, g_anal, atol=1e-4)
        print "Gradient check succeeded!"
        return

    optim_state = make_rmsprop_state(theta=pc.get_value_flat(), step_size = args.step_size, 
        decay_rate = args.decay_rate)

    for iepoch in xrange(args.n_epochs):
        losses = []
        tstart = time()
        print "starting epoch",iepoch
        cur_hiddens = initialize_hiddens(args.size_batch)
        for (x,y) in loader.train_batches_iter():
            out = f_loss_and_grad(x,y, *cur_hiddens)
            loss = out[0]
            grad = out[1]
            cur_hiddens = out[2:]
            rmsprop_update(grad, optim_state)
            pc.set_value_flat(optim_state.theta)
            losses.append(loss)
            if args.unittest: return
        print "%.3f s/batch. avg loss = %.3f"%((time()-tstart)/len(losses), np.mean(losses))
        optim_state.step_size *= .98 #pylint: disable=E1101

        sample(f_step, initialize_hiddens(1), char2ind=loader.char2ind, n_steps=300, temp=1.0, seed_text = "")

    if args.profile: profiler.print_stats()

if __name__ == "__main__":
    main()
