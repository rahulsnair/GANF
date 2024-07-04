#%%
import os
import argparse
import torch
from models.GANF import GANF
import numpy as np
from sklearn.metrics import roc_auc_score
# from data import fetch_dataloaders


parser = argparse.ArgumentParser()
# files
parser.add_argument('--data_dir', type=str, 
                    default='./data/GANF_Pump', help='Location of datasets.')
parser.add_argument('--output_dir', type=str, 
                    default='./checkpoint/model')
parser.add_argument('--name',default='GANF_Pump')
# restore
parser.add_argument('--graph', type=str, default='None')
parser.add_argument('--model', type=str, default='None')
parser.add_argument('--seed', type=int, default=2024, help='Random seed to use.')
# made parameters
parser.add_argument('--n_blocks', type=int, default=1, help='Number of blocks to stack in a model (MADE in MAF; Coupling+BN in RealNVP).')
parser.add_argument('--n_components', type=int, default=1, help='Number of Gaussian clusters for mixture of gaussians models.')
parser.add_argument('--hidden_size', type=int, default=32, help='Hidden layer size for MADE (and each MADE block in an MAF).')
parser.add_argument('--n_hidden', type=int, default=1, help='Number of hidden layers in each MADE.')
parser.add_argument('--batch_norm', type=bool, default=False)
# training params
parser.add_argument('--batch_size', type=int, default=512)

args = parser.parse_known_args()[0]
args.cuda = torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")


print(args)
import random
import pandas as pd
import numpy as np
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
#%%
print("Loading dataset")
from dataset import load_pump

train_loader, _, n_sensor, test_loader = load_pump(args.data_dir, \
                                                                args.batch_size)

#%%
model = GANF(args.n_blocks, 1, args.hidden_size, args.n_hidden, dropout=0.0, batch_norm=args.batch_norm)
model = model.to(device)


model.load_state_dict(torch.load("./checkpoint/model/GANF_pump_seed_2024/GANF_pump_seed_2024_best.pt"))
A = torch.load("./checkpoint/model/GANF_pump_seed_2024/graph_best.pt").to(device)
model.eval()
#%%
loss_test = []
predlist = []
with torch.no_grad():
    for x in test_loader:

        x = x.to(device)
        loss = -model.test(x, A.data).cpu().numpy()
        pred = -model(x, A.data).cpu().numpy()
        loss_test.append(loss)
        predlist.append(pred)
loss_test = np.concatenate(loss_test)
roc_test = roc_auc_score(np.asarray(test_loader.dataset.label.values,dtype=int),loss_test)
print("The ROC score on SWaT dataset is {}".format(roc_test))
outdf = pd.DataFrame(predlist)
outdf.to_csv('checkpoint/model/GANF_Test_anomaly.csv', index=False)
# %%
