import os, sys, argparse, torch

from radbm.search.elba import MIHash
from radbm.utils.os import safe_save
with open('common.py', 'rb') as f:
    exec(f.read())

# --- (beg) parsing args --- #
parser = argparse.ArgumentParser()
parser.add_argument('name', help='name of the model') 
parser.add_argument('state_dir', help='the path to save states') 
parser.add_argument('--nbits', type=int, default=64, help='the number of bits of the Multi-Bernoulli (default 64)')
parser.add_argument('--share', dest='share', action='store_true',
                    help='to activate parameters sharing between networks (default unshared)')
parser.set_defaults(share=False)
parsed = parser.parse_args()
# --- (end) parsing args --- #

# --- (beg) Log --- #
log_path = os.path.join(parsed.state_dir, 'log','{}.log'.format(parsed.name))
f = open(log_path, 'a')
# --- (end) Log --- #

# --- (beg) building experiment --- #
qnet = MNISTResNet(parsed.nbits)
dnet = qnet if parsed.share else MNISTResNet(parsed.nbits)
HMBS = HashingMultiBernoulliSDS(ntables=1, nlookups=10)
model = MIHash(
    fq=qnet, fd=dnet, struct=HMBS,
    nbits=parsed.nbits, match_prob=1/8, #fake match_prob
)
if torch.cuda.is_available():
    f.write('using cuda\n')
    model.cuda()
else:
    f.write('using cpu\n')
f.flush()

least5_hr_name = '{}_hr{}'.format(parsed.name, '{index}')
least5_mb_name = '{}_mb{}'.format(parsed.name, '{index}')
current_path = os.path.join(parsed.state_dir, 'current', parsed.name)
least5_hr_path = os.path.join(parsed.state_dir, 'hr', least5_hr_name)
least5_mb_path = os.path.join(parsed.state_dir, 'mb', least5_mb_name)
exp = Experiment(model, dataset, parsed.nbits, current_path, least5_hr_path, least5_mb_path)
exp.save(current_path, if_exists='ignore', pickle_protocol=-1)
exp.load(current_path)
# --- (end) building experiment --- #

# --- (beg) training loop --- #
def training_step(model, nbatch):
    batch_size=32
    model.train()
    q, d, r = dataset.batch(batch_size)
    loss = model.step(q, d, r)
    return loss

total_batch=100001
try:
    training_loop(exp, training_step, total_batch, out=f)
except KeyboardInterrupt: pass
finally: f.close()
# --- (end) training loop --- #
