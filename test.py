from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
import json
import os
import numpy as np
import random
from tqdm import tqdm
from rdkit import Chem
from modeling_molbridge import MoltrieverForEmbed

os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
print('PREPARING DATA...')

def canonicalize_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles 
    return Chem.MolToSmiles(mol, canonical=True)

class Dataset_class(Dataset):
    def __init__(
        self,
        ):
        self.cids = []
        self.descriptions = {}
        self.smiles = {}
        
        #load data
        # data = json.load(open('PubChem324kV2/test.json'))
        data = json.load(open('KV-PLM/test.json')) # scaffold
        # data = json.load(open('KV-PLM/PCDes_all.json'))[12000:]

        for n, line in enumerate(data):
            self.descriptions[line['cid']] = line['caption']
            self.smiles[line['cid']] = canonicalize_smiles(line['smiles'])
            self.cids.append(line['cid'])


    def __len__(self):
        return len(self.cids)

    def __getitem__(self, idx):
        cid = self.cids[idx]
        smiles = self.smiles[cid]
        description = self.descriptions[cid]

        return {
                'cid':cid,
                'description':description,
                'smiles':smiles
                }     

def collate_fn(batch):
    description, smiles, cids = [], [], []
    for b in batch:
        cid = b['cid']
        cids.append(cid)
        description.append( b['description'])
        smiles.append(b['smiles'])

    return description, smiles, cids, 


test_data = Dataset_class(
)

test_loader = DataLoader(
    test_data,
    batch_size=64,
    num_workers=8,
    collate_fn=collate_fn
)
    
##############################################################################################################
model = MoltrieverForEmbed().to(device)
smiles_tk = AutoTokenizer.from_pretrained(model.smiles_model_path, trust_remote_code=True)
lang_tk = AutoTokenizer.from_pretrained(model.language_model_path)

model.load_state_dict(torch.load('/data/user16/molbridge/output_moltriever_refine/2025-05-15-16-20-17_200_2/50_model'), strict=False)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

print(count_parameters(model))

model.eval()
##############################################################################################################
@torch.no_grad()
def embed_smiles(model, dataloader, is_smiles=True):
    all_embeds = []
    all_cids = []
    all_input_text = []
    all_sidekick = []
    for i, batch in enumerate(tqdm(dataloader)):
        if is_smiles:
            sidekick, input_text, cids = batch
            input_tokens = smiles_tk(input_text, return_tensors='pt', padding='longest', truncation=True, max_length=256).to(device)
        else:
            input_text, sidekick, cids = batch
            input_tokens = lang_tk(input_text, return_tensors='pt', padding='longest', truncation=True, max_length=256).to(device)
        
        hidden, logits = model(input_tokens, is_smiles)

        all_embeds.append(hidden)
        all_cids += cids
        all_input_text += input_text
        all_sidekick += sidekick
    
    all_embeds = torch.cat(all_embeds, dim=0)
    
    return all_embeds, logits, cids, all_input_text, all_sidekick


smiles_embeds, _, test_cids, test_smiles, test_descriptions = embed_smiles(model, test_loader, True)
caption_embeds, _, _, _, _ = embed_smiles(model, test_loader, False)

logits_smiles_to_caption = torch.matmul(smiles_embeds.cpu(), caption_embeds.t().cpu())
logits_caption_to_smiles = torch.matmul(caption_embeds.cpu(), smiles_embeds.t().cpu())

def evaluate_retrieval(logits, correct_indices, label_type='caption'):
    r1 = r5 = r10 = r20 = 0
    reciprocal_ranks = []

    # logits: [N, N] similarity scores (query vs. all)
    sorted_indices = torch.argsort(logits, dim=-1, descending=True)  # [N, N]

    for i in range(len(correct_indices)):
        ranking = sorted_indices[i].tolist()
        rank = ranking.index(correct_indices[i]) + 1  # 1-based rank
        reciprocal_ranks.append(1.0 / rank)

        if rank == 1:
            r1 += 1
        if rank <= 5:
            r5 += 1
        if rank <= 10:
            r10 += 1
        if rank <= 20:
            r20 += 1

    n = len(correct_indices)
    print(f"\n{label_type.capitalize()} Retrieval")
    print(f"R@1: {r1/n*100:.2f} | R@5: {r5/n*100:.2f} | R@10: {r10/n*100:.2f} | R@20: {r20/n*100:.2f} | MRR: {np.mean(reciprocal_ranks):.4f}")


correct_indices = list(range(len(test_smiles)))

evaluate_retrieval(logits_smiles_to_caption, correct_indices, label_type='smiles2caption')
evaluate_retrieval(logits_caption_to_smiles, correct_indices, label_type='caption2smiles')