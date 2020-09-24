import itertools, torch, sys
import numpy as np
from torchvision.models.resnet import ResNet, BasicBlock

from radbm.loaders.mnist import NoisyMnist
from radbm.search.mbsds import HashingMultiBernoulliSDS
from radbm.search.radius import HammingRadiusSDS
from radbm.metrics.sswr import CounterSSWR
from radbm.metrics.hamming import conditional_hamming_counts, conditional_counts_to_pr_curve
from radbm.utils.time import Chronometer
from radbm.utils.os import StateObj, safe_save

#dataset
N, M = 10000, 1000
rng = np.random.RandomState(0xcafe)
dataset = NoisyMnist(rng=rng).valid().torch()
if torch.cuda.is_available: dataset.cuda()
valid_d = dataset.build_documents(index=range(N), replace=False)
valid_q = dataset.build_queries(index=range(M), replace=False)
dataset.train()
train_d = dataset.build_documents(index=range(N), replace=False)
train_q = dataset.build_queries(index=range(M), replace=False)
relevances = [{i} for i in range(M)]

#model 
#definition code from https://www.kaggle.com/tonysun94/pytorch-1-0-1-on-mnist-acc-99-8
#modified for k class
class View(torch.nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

class MNISTResNet(ResNet):
    def __init__(self, k):
        super(MNISTResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=k) # Based on ResNet18
        # super(MNISTResNet, self).__init__(BasicBlock, [3, 4, 6, 3], num_classes=k) # Based on ResNet34
        # super(MNISTResNet, self).__init__(Bottleneck, [3, 4, 6, 3], num_classes=k) # Based on ResNet50
        self.conv1 = torch.nn.Sequential(
            View(-1, 1, 28, 28),
            torch.nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3,bias=False),
        )

#Experiment: namespace + saving and loading
class Experiment(StateObj):
    def __init__(self, model, dataset, nbits, current_path, least5_hr_path, least5_mb_path):
        self.batch = 0
        self.train_losses = list()
        self.least5_hr_sswrs = 5*[np.inf]
        self.least5_mb_sswrs = 5*[np.inf]
        self.results = {
            'train_pos_counts':[],
            'train_neg_counts':[],
            'train_precisions':[],
            'train_recalls':[],
            'train_2081hr_sswrs':[],
            'train_2081hr_halts':[],
            'train_5001mb_sswrs':[],
            'train_5001mb_halts':[],
            'valid_pos_counts':[],
            'valid_neg_counts':[],
            'valid_precisions':[],
            'valid_recalls':[],
            'valid_2081hr_sswrs':[],
            'valid_2081hr_halts':[],
            'valid_5001mb_sswrs':[],
            'valid_5001mb_halts':[],
            'loss':[],
        }
        self.model = model
        self.dataset = dataset
        self.nbits = nbits
        
        #paths are not meant to be save.
        self.current_path = current_path
        self.least5_hr_path = least5_hr_path
        self.least5_mb_path = least5_mb_path
        
    def get_state(self):
        return {
            'batch': self.batch,
            'results': self.results,
            'train_losses': self.train_losses,
            'least5_hr_sswrs': self.least5_hr_sswrs,
            'least5_mb_sswrs': self.least5_mb_sswrs,
            'model': self.model.get_state(),
            'dataset': self.dataset.get_state(),
            'nbits': self.nbits,
        }
    
    def set_state(self, state):
        self.batch = state['batch']
        self.results = state['results']
        self.train_losses = state['train_losses']
        self.least5_hr_sswrs = state['least5_hr_sswrs']
        self.least5_mb_sswrs = state['least5_mb_sswrs']
        self.model.set_state(state['model'])
        self.dataset.set_state(state['dataset'])
        self.nbits = state['nbits']
        return self
    
#all the above is for evaluation
def batch_call(f, x, batch_size):
    out = list()
    total_batch = int(np.ceil(len(x)/batch_size))
    for i in range(total_batch):
        a, b = i*batch_size, (i+1)*batch_size
        out.append(f(x[a:b]))
    return out

def avg_if_enough(values, minimum_for_avg):
    values = np.array(values)
    notnan = ~np.isnan(values)
    return values[notnan].mean() if notnan.sum() >= minimum_for_avg else np.inf

def eval_sswr(results, prefix, N, relevances, gens, halt, sswrs_timeout):
    sswrs = len(gens)*[float('nan')]
    halts = len(gens)*[float('nan')]
    chrono = Chronometer().start()
    for i, (gen, rel) in enumerate(zip(gens, relevances)):
        if sswrs_timeout < chrono.time(): break
        gen = itertools.islice(gen, 0, halt)
        sswr, ishalt = CounterSSWR(rel, gen, N, allow_halt=True)
        sswrs[i] = sswr
        halts[i] = ishalt
    results[prefix+'_sswrs'].append(sswrs)
    results[prefix+'_halts'].append(halts)
    return avg_if_enough(sswrs, 100)

def evaluation(results, prefix, nbits, elba, queries, documents, relevances, batch_size=100, sswrs_timeout=30):
    N = len(documents)
    with torch.no_grad():
        dlogits = torch.cat(batch_call(elba.fd, documents, batch_size), dim=0)
        qlogits = torch.cat(batch_call(elba.fq, queries, batch_size), dim=0)
        dls_pairs = elba._log_sigmoid_pairs(dlogits)
        qls_pairs = elba._log_sigmoid_pairs(qlogits)
    dbits = 0<dlogits
    qbits = 0<qlogits
    pos_counts, neg_counts = conditional_hamming_counts(dbits, qbits, relevances)
    precisions, recalls = conditional_counts_to_pr_curve(pos_counts, neg_counts)
    
    results[prefix+'_pos_counts'].append(pos_counts.cpu().numpy())
    results[prefix+'_neg_counts'].append(neg_counts.cpu().numpy())
    results[prefix+'_precisions'].append(precisions.cpu().numpy())
    results[prefix+'_recalls'].append(recalls.cpu().numpy())
    
    hr = HammingRadiusSDS(nbits=nbits, radius=2).batch_insert(dbits, range(N))
    hr_gens = hr.batch_itersearch(qbits, yield_empty=True)
    hr_sswr = eval_sswr(results, prefix+'_2081hr', N, relevances, hr_gens, 2081, sswrs_timeout)
    
    mb = HashingMultiBernoulliSDS(1,1).batch_insert(dls_pairs, range(N))
    mb_gens = mb.batch_itersearch(qls_pairs, yield_empty=True)
    mb_sswr = eval_sswr(results, prefix+'_5001mb', N, relevances, mb_gens, 5001, sswrs_timeout)
    return hr_sswr, mb_sswr

#traning loop!
def eval_save(exp, out=sys.stdout):
    avg_loss = np.mean(exp.train_losses) if exp.train_losses else float('nan')
    exp.results['loss'].append(avg_loss)
    exp.train_losses = list()

    model.eval()
    evaluation(exp.results, 'train', exp.nbits, exp.model, train_q, train_d, relevances, sswrs_timeout=30)
    hr_sswr, mb_sswr = evaluation(exp.results, 'valid', exp.nbits, exp.model, valid_q, valid_d, relevances, sswrs_timeout=30)

    msg = 'batch@{} [loss={:.4f}] [hr_sswr={:.4f}] [mb_sswr={:.4f}]\n'.format(exp.batch//500, avg_loss, hr_sswr, mb_sswr)
    out.write(msg); out.flush()
    
    state = exp.model.state_dict()
    state['nbatch'] = exp.batch #used to know when in the training it was saved
    max_hr = max(exp.least5_hr_sswrs)
    if hr_sswr < max_hr:
        index = exp.least5_hr_sswrs.index(max_hr)
        exp.least5_hr_sswrs[index] = hr_sswr
        safe_save(state, exp.least5_hr_path.format(index=index), pickle_protocol=-1)
    max_mb = max(exp.least5_mb_sswrs)
    if mb_sswr < max_mb:
        index = exp.least5_mb_sswrs.index(max_mb)
        exp.least5_mb_sswrs[index] = mb_sswr
        safe_save(state, exp.least5_mb_path.format(index=index), pickle_protocol=-1)
    exp.save(exp.current_path, if_exists='overwrite', pickle_protocol=-1)

def training_loop(exp, training_step, total_batch, out=sys.stdout):
    if exp.batch==0:
        #initialization evaluation
        #might happen multiple times if the script restart before exp.batch is updated
        eval_save(exp, out=out)
    for nbatch in range(exp.batch+1, total_batch):
        exp.model.train()
        loss = training_step(exp.model, nbatch-1)
        exp.train_losses.append(float(loss))
        if nbatch%500==0:
            exp.batch = nbatch
            eval_save(exp, out=out)
