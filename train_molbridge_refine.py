from collections import Counter
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
import json
import torch
import argparse
import os
import numpy as np
import random
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from modeling_molbridge import MolBridge

os.environ["TOKENIZERS_PARALLELISM"] = "false"


##############################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument('--accum_steps', type=int, default=2)
parser.add_argument('--learning_rate', type=float, default=2e-5)

parser.add_argument("--num_workers", type=int, default=8)

parser.add_argument('--save_path', type=str, default='./output_MolBridge_refine/')
parser.add_argument('--data_dir', type=str, default='/home/user16/HT/MolReGPT4TKDE/')
parser.add_argument('--dataset', type=str, default='pubchem324k', choices=["pubchemstm", "cap2mol_trans_raw", "pubchem324k"])

parser.add_argument('--eval_step', type=int, default=1000)
parser.add_argument('--eval_only', action='store_true')

args = parser.parse_args()

accum_steps = args.accum_steps
batch_size = args.batch_size
learning_rate = args.learning_rate

##############################################################################################################
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

seed = 42
deterministic = False

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
if deterministic:
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

##############################################################################################################
import time
current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
output_dir = os.path.join(args.save_path, f'{current_time}_{args.batch_size}_{args.accum_steps}')
if args.eval_only:
    writer = None
else:
    writer = SummaryWriter(output_dir)

##############################################################################################################
print('PREPARING DATA...')

class AllChem(Dataset):
    def __init__(
        self, 
        data_path = 'filtered_cont_trainset_cem22.json'
        ):
        self.data = json.load(open(data_path))
        self.positives = json.load(open('filtered_positive_set_cem22_augmented.json'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles = self.data[idx]['smiles']
        description = self.data[idx]['description']
        data_type = self.data[idx]['type']
        cands_for_s = []
        cands_for_d = []
        
        if smiles in self.positives.keys():
            tmp = deepcopy(self.positives[smiles])
            tmp = list(set(tmp))
            if description in tmp:
                tmp.remove(description)
            cands_for_s += tmp
        if description in self.positives.keys():
            tmp = deepcopy(self.positives[description])
            tmp = list(set(tmp))
            if smiles in tmp:
                tmp.remove(smiles)
            cands_for_d += tmp

        return {
                'cands_for_s':cands_for_s, 
                'cands_for_d':cands_for_d, 
                'description':description,
                'smiles':smiles,
                'type':data_type
                } 


def collate_fn(batch):
    description, smiles, cands_for_s, cands_for_d, types = [], [], [], [], []
    for b in batch:
        cands_for_s.append(b['cands_for_s'])
        cands_for_d.append(b['cands_for_d']) 
        description.append(b['description'])
        smiles.append(b['smiles'])
        types.append(b['type'])

    B = len(description)
    mask_s = torch.zeros((B, B), dtype=torch.float32)  # text → smiles
    mask_d = torch.zeros((B, B), dtype=torch.float32)  # smiles → text

    for i in range(B):
        for j in range(B):
            if smiles[j] == smiles[i] or smiles[j] in cands_for_s[i]:
                mask_s[i][j] = 1.0
            if description[j] == description[i] or description[j] in cands_for_d[i]:
                mask_d[i][j] = 1.0

    return description, smiles, mask_s, mask_d, types


train_data = AllChem()

train_loader = DataLoader(
    train_data,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=True,
    drop_last=False,
    collate_fn=collate_fn
)

model = MolBridge().to(device)

smiles_tk = AutoTokenizer.from_pretrained(model.smiles_model_path, trust_remote_code=True)
lang_tk = AutoTokenizer.from_pretrained(model.language_model_path)

##############################################################################################################
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
optimizer.zero_grad()
scaler = torch.amp.GradScaler()

##############################################################################################################
@torch.no_grad()
def validate(model, dataloader, epoch):
    total_loss = 0
    current_steps = 0

    for i, batch in enumerate(tqdm(dataloader)):
        description, smiles, loss_mask_for_s, loss_mask_for_d, types = batch
        
        smiles_token = smiles_tk(smiles, return_tensors='pt', padding='longest', truncation=True, max_length=128).to(device)
        description_token = lang_tk(description, return_tensors='pt', padding='longest', truncation=True, max_length=128).to(device)
        loss_mask_for_s = loss_mask_for_s.to(device)
        loss_mask_for_d = loss_mask_for_d.to(device)
        
        outputs = model(smiles_token, description_token, loss_mask_for_s, loss_mask_for_d, types)

        total_loss += outputs[0].item()
        current_steps += 1

    if epoch < 100:
        prefix = 'Loss/val_epoch'
    else:
        prefix = 'Loss/val_steps'
    
    if not args.eval_only:
        writer.add_scalar(prefix, total_loss/current_steps, epoch)
        writer.flush()
    print(f'Loss: {total_loss / current_steps}')
    
    return total_loss / current_steps

##############################################################################################################

accum_loss = 0
current_steps = 0
log_every_n_steps=10
best_score = 99999
for ep in range(50):
    total_loss = 0
    total_steps = 0
    for i, batch in enumerate(tqdm(train_loader)):
        description, smiles, loss_mask_for_s, loss_mask_for_d, types = batch
        # print(smiles)
        # print('---------------')
        # print(description)
        smiles_token = smiles_tk(smiles, return_tensors='pt', padding='longest', truncation=True, max_length=256).to(device)
        description_token = lang_tk(description, return_tensors='pt', padding='longest', truncation=True, max_length=256).to(device)
        loss_mask_for_s = loss_mask_for_s.to(device)
        loss_mask_for_d = loss_mask_for_d.to(device)
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model(smiles_token, description_token, loss_mask_for_s, loss_mask_for_d, types)
        
        loss = outputs[0] / accum_steps
        # print(loss)
        accum_loss += loss.item()
        scaler.scale(loss).backward()

        if ((i + 1) % accum_steps == 0) or (i + 1 == len(train_loader)):
            total_loss += accum_loss
            total_steps += 1
            current_steps += 1

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if current_steps % log_every_n_steps == 0:
                writer.add_scalar("Loss/train", accum_loss, current_steps)
                writer.add_scalar("Loss/contrastive", outputs[-2].item(), current_steps)
                writer.add_scalar("Loss/classification", outputs[-1].item(), current_steps)
                writer.flush()
                
            accum_loss = 0
                    
    writer.flush()
    torch.save(model.state_dict(), output_dir+f'/{ep+1}_model')

    if (ep+1) % 10 == 0:
        print('-----filtering out noisy subsumption relations-----')
        new_data = []
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(train_loader)):
                description, smiles, loss_mask_for_s, loss_mask_for_d, types = batch
                
                smiles_token = smiles_tk(smiles, return_tensors='pt', padding='longest', truncation=True, max_length=256).to(device)
                description_token = lang_tk(description, return_tensors='pt', padding='longest', truncation=True, max_length=256).to(device)
                loss_mask_for_s = loss_mask_for_s.to(device)
                loss_mask_for_d = loss_mask_for_d.to(device)
                
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = model(smiles_token, description_token, loss_mask_for_s, loss_mask_for_d, types, is_train=False)
                
                class_pred = torch.argmax(outputs[-2], dim=-1)
                pair_types = outputs[-1]

                for jj, c in enumerate(class_pred):
                    if c == pair_types[jj] or types[jj] == 'molcap':
                        new_data.append({
                            'smiles': smiles[jj],
                            'description': description[jj],
                            'type': types[jj]
                        })

        new_data_path = f'filtered_cont_trainset_cem2_all_2_{ep+1}.json'
        with open(new_data_path, 'w') as f:
            json.dump(new_data, f, indent=2)
                
        train_data = AllChem(
            data_path=new_data_path
        )

        train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            drop_last=False,
            collate_fn=collate_fn
        )

        model.train()

if not args.eval_only:
    writer.close()

