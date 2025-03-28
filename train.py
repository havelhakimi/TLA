from transformers import AutoTokenizer, AutoConfig
#from fairseq.data import data_utils
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from optim import ScheduledOptim, Adam
from tqdm import tqdm
import argparse
import os
import datetime
import torch.nn as nn
from eval import evaluate
from model import PLM_Graph
import random
import tarfile
import numpy as np




def seed_torch(seed=1029):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True




class BertDataset(Dataset):
    def __init__(self, max_token=512, device='cpu', pad_idx=0, data_path=None):
        
        super(BertDataset, self).__init__()
        self.device = device
        extraction_path=os.path.join(data_path,'tok.tar.xz')
        with tarfile.open(extraction_path, 'r:*') as tar_ref: 
            tar_ref.extractall(data_path)
            
        tok_path = os.path.join(data_path, 'tok.txt')
        y_path = os.path.join(data_path, 'Y.txt')
        with open(tok_path,'r') as f:
            self.data=[torch.tensor([int(id) for id in line.strip().split()] ,dtype=torch.long) for line in f.readlines() ]
            
        with open(y_path, 'r', encoding='utf-8') as f:
            self.labels = [torch.tensor([int(id) for id in line.strip().split()] ,dtype=torch.long) for line in f.readlines() ]
        
        self.max_token = max_token
        self.pad_idx = pad_idx

    def __getitem__(self, item):
        data = self.data[item][:self.max_token - 2].to(
            self.device)
        labels = self.labels[item].to(self.device)
        return {'data': data, 'label': labels, 'idx': item, }

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        if not isinstance(batch, list):
            return batch['data'], batch['label'], batch['idx']
        label = torch.stack([b['label'] for b in batch], dim=0)
        data = torch.full([len(batch), self.max_token], self.pad_idx, device=label.device, dtype=batch[0]['data'].dtype)
        idx = [b['idx'] for b in batch]
        for i, b in enumerate(batch):
            data[i][:len(b['data'])] = b['data']
        return data, label, idx


class Saver:
    def __init__(self, model, optimizer, scheduler, args):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args

    def __call__(self, score, best_score, name):
        torch.save({'param': self.model.state_dict(),
                    'optim': self.optimizer.state_dict(),
                    'sche': self.scheduler.state_dict() if self.scheduler is not None else None,
                    'score': score, 'args': self.args,
                    'best_score': best_score},
                   name)


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate.')
parser.add_argument('--data', type=str, default='wos', choices=['wos', 'nyt', 'rcv'], help='Dataset.')
parser.add_argument('--batch', type=int, default=12, help='Batch size.')
parser.add_argument('--early-stop', type=int, default=6, help='Epoch before early stop.')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--name', type=str, required=True, help='A name for different runs.')
parser.add_argument('--update', type=int, default=1, help='Gradient accumulate steps')
parser.add_argument('--warmup', default=0, type=int, help='Warmup steps.')
parser.add_argument('--graph', default=0, type=int, help='Whether use graph encoder.')
parser.add_argument('--layer', default=1, type=int, help='Layer of GPTrans.')
parser.add_argument('--mod_type', default='bert-base-uncased', type=str, choices=['bert-base-uncased','roberta-large'], help='Select backbone')
parser.add_argument('--graph_type', type=str, default='GPTrans',choices=['GPTrans','graphormer','GAT', 'GCN'], help='graph type')
parser.add_argument('--edge_dim', default=30, type=int, help='Edge dimension for GPTrans .')
parser.add_argument('--label_refiner', default=1, type=int, help='Label Refiner.')
parser.add_argument('--bce_wt', type=float, default=1, help='bce_wt.')
parser.add_argument('--dot', default=0, type=int, help='Dot prod.')
parser.add_argument('--seed', default=3, type=int, help='Random seed.')
parser.add_argument('--tla', type=int, default=0, help='whether TLA required or not')
parser.add_argument('--tl_pen', type=float, default=1.0, help='weight for TLA loss')
parser.add_argument('--tl_temp', type=float, default=0.07, help='Temperature of TLA loss')
parser.add_argument('--norm', type=int, default=0, help='whether embeddings to be normalized before TLA')
parser.add_argument('--proj', type=int, default=0, help='whether embeddings of text and label to be transformed before TLA')
parser.add_argument('--hsize', default=768, type=int, help='Hidden size after projection')


if __name__ == '__main__':

    args = parser.parse_args()
    device = args.device
    print(args)
    #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    

    #if args.wandb:
        #import wandb
        #wandb.init(config=args, project='htc')
    seed_torch(args.seed)
    mod_name=args.name
    args.name = args.data + '-' + args.name
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    #data_path = os.path.join('data', args.data)
    data_path = os.path.join('../TLA/data', args.data)
    args.data=data_path
    label_dict = torch.load(os.path.join(data_path, 'bert_value_dict.pt'))
    label_dict = {i: tokenizer.decode(v, skip_special_tokens=True) for i, v in label_dict.items()}
    num_class = len(label_dict)

    dataset = BertDataset(device=device, pad_idx=tokenizer.pad_token_id, data_path=data_path)
    config = AutoConfig.from_pretrained(args.mod_type)
    model = PLM_Graph(config, num_labels=num_class,
                                          graph=args.graph,mod_type=args.mod_type,graph_type=args.graph_type,
                                          bce_wt=args.bce_wt,dot=args.dot,
                                          layer=args.layer, data_path=args.data,
                                          tla=args.tla,tl_pen=args.tl_pen,tl_temp=args.tl_temp,norm=args.norm,proj=args.proj,
                                          hsize=args.hsize,label_refiner=args.label_refiner,edge_dim=args.edge_dim
                                          )

    
    #if args.wandb:
        #wandb.watch(model)
    

    
    
    split = torch.load(os.path.join(data_path, 'split.pt'))
    train = Subset(dataset, split['train'])
    dev = Subset(dataset, split['val'])
    if args.warmup > 0:
        optimizer = ScheduledOptim(Adam(model.parameters(),
                                        lr=args.lr), args.lr,
                                   n_warmup_steps=args.warmup)
    else:
        optimizer = Adam(model.parameters(),
                         lr=args.lr)

    train = DataLoader(train, batch_size=args.batch, shuffle=True, collate_fn=dataset.collate_fn, )#sampler=DistributedSampler(train))
    dev = DataLoader(dev, batch_size=args.batch, shuffle=False, collate_fn=dataset.collate_fn, )#sampler=DistributedSampler(dev))
    
    

    
    
    model.to(device)
    save = Saver(model, optimizer, None, args)
    best_score_macro = 0
    best_score_micro = 0
    early_stop_count = 0
    

    os.makedirs(os.path.join(data_path, 'Checkpoints',mod_name), exist_ok=True)

    log_file = open(os.path.join(data_path, 'Checkpoints',mod_name,'log.txt'), 'w')
  

    for epoch in range(1000):
        if early_stop_count >= args.early_stop:
            print("Early stop!")
            break
        model.train()
        i = 0
        loss = 0

        # Train
        pbar = tqdm(train)
        for data, label, idx in pbar:
            padding_mask = data != tokenizer.pad_token_id
            #data=data.to(device),label.to(device),padding_mask.to(device), padding_mask.to(device)
            output = model(data, padding_mask, labels=label, )
            loss /= args.update
            output['loss'].backward()
            loss += output['loss'].item()
            i += 1
            if i % args.update == 0:
                optimizer.step()
                optimizer.zero_grad()
                #if args.wandb:
                #    wandb.log({'train_loss': loss})
                pbar.set_description('loss:{:.4f}'.format(loss))
                i = 0
                loss = 0
                # torch.cuda.empty_cache()
        pbar.close()

        model.eval()
        pbar = tqdm(dev)
        with torch.no_grad():
            truth = []
            pred = []
            for data, label, idx in pbar:
                padding_mask = data != tokenizer.pad_token_id
                #data=data.to(device),label.to(device),padding_mask.to(device), padding_mask.to(device)

                output = model(data, padding_mask, labels=label, )
                for l in label:
                    t = []
                    for i in range(l.size(0)):
                        if l[i].item() == 1:
                            t.append(i)
                    truth.append(t)
                for l in output['logits']:
                    pred.append(torch.sigmoid(l).tolist())

        pbar.close()
        scores = evaluate(pred, truth, label_dict)
        macro_f1 = scores['macro_f1']
        micro_f1 = scores['micro_f1']
        print('macro', macro_f1, 'micro', micro_f1)
        print('macro', macro_f1, 'micro', micro_f1, file=log_file)
        #if args.wandb:
        #    wandb.log({'val_macro': macro_f1, 'val_micro': micro_f1, 'best_macro': best_score_macro,
        #               'best_micro': best_score_micro})
        early_stop_count += 1
        if macro_f1 > best_score_macro:
            best_score_macro = macro_f1
            save(macro_f1, best_score_macro, os.path.join('Checkpoints', mod_name, 'checkpoint_best_macro.pt'))
            early_stop_count = 0

        if micro_f1 > best_score_micro:
            best_score_micro = micro_f1
            save(micro_f1, best_score_micro, os.path.join('Checkpoints', mod_name, 'checkpoint_best_micro.pt'))
            early_stop_count = 0
        # save(macro_f1, best_score, os.path.join('checkpoints', args.name, 'checkpoint_{:d}.pt'.format(epoch)))
        # save(micro_f1, best_score_micro, os.path.join('checkpoints', args.name, 'checkpoint_last.pt'))
    log_file.close()


    
