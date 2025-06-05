import torch
from torch import nn

import transformers

def _get_vector_norm(tensor: torch.Tensor) -> torch.Tensor:
    """
    This method is equivalent to tensor.norm(p=2, dim=-1, keepdim=True) and used to make
    model `executorch` exportable. See issue https://github.com/pytorch/executorch/issues/3566
    """
    square_tensor = torch.pow(tensor, 2)
    sum_tensor = torch.sum(square_tensor, dim=-1, keepdim=True)
    normed_tensor = torch.pow(sum_tensor, 0.5)
    return normed_tensor

class MolBridge(nn.Module):
    def __init__(self, 
                 smiles_model_path='ibm-research/MoLFormer-XL-both-10pct', 
                 language_model_path='allenai/scibert_scivocab_uncased', 
                 device='cuda',
                 ):
        super().__init__()

        self.smiles_model_path = smiles_model_path
        self.language_model_path = language_model_path
        self.device = device

        self.smiles_model = transformers.AutoModel.from_pretrained(smiles_model_path, trust_remote_code=True)
        self.language_model = transformers.AutoModel.from_pretrained(language_model_path)

        self.smi_linear = nn.Linear(self.smiles_model.config.hidden_size, 512, bias=False)
        self.lang_linear = nn.Linear(self.language_model.config.hidden_size, 512, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))

        self.dropout = nn.Dropout(0.1)

        self.classifier = nn.Linear(512*2, 3)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, smiles_batch, language_batch, mask_s, mask_d, pair_types, is_train=True):
        smiles_hidden = self.smiles_model(**smiles_batch).last_hidden_state[:, 0, :]
        language_hidden = self.language_model(**language_batch).last_hidden_state[:, 0, :]

        smiles_hidden = self.smi_linear(smiles_hidden)
        language_hidden = self.lang_linear(language_hidden)

        smiles_hidden = self.dropout(smiles_hidden)
        language_hidden = self.dropout(language_hidden)

        smiles_hidden = smiles_hidden / _get_vector_norm(smiles_hidden)
        language_hidden = language_hidden / _get_vector_norm(language_hidden)

        pooled_input_for_classification = torch.cat([smiles_hidden, language_hidden], dim=-1)
        class_logits = self.classifier(pooled_input_for_classification)

        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(language_hidden, smiles_hidden.t()) * logit_scale
        logits_per_smiles = logits_per_text.t()

        type_map = {'molcap': 0, 'molkey': 1, 'subcap': 2}
        if isinstance(pair_types, list):
            pair_types = torch.tensor([type_map[t] for t in pair_types], device=smiles_hidden.device).long()

        if not is_train:
            return logits_per_text, logits_per_smiles, class_logits, pair_types
        
        sub_idx = (pair_types == 2).nonzero(as_tuple=True)[0]  # subcap
        key_idx = (pair_types == 1).nonzero(as_tuple=True)[0]  # molkey

        B = logits_per_text.size(0)
        mask_text_logits = torch.ones((B, B), device=logits_per_text.device)
        mask_smiles_logits = torch.ones((B, B), device=logits_per_text.device)

        if sub_idx.numel() > 0 and key_idx.numel() > 0:
            # text → smiles: key row, sub col
            mask_text_logits[key_idx.unsqueeze(1), sub_idx.unsqueeze(0)] = 0.0
            # smiles → text: sub row, key col
            mask_smiles_logits[sub_idx.unsqueeze(1), key_idx.unsqueeze(0)] = 0.0

        logits_per_text = logits_per_text.masked_fill(mask_text_logits == 0, -1e9)
        logits_per_smiles = logits_per_smiles.masked_fill(mask_smiles_logits == 0, -1e9)
        
        log_probs_text = torch.log_softmax(logits_per_text, dim=1).clamp(min=-30)
        log_probs_smiles = torch.log_softmax(logits_per_smiles, dim=1).clamp(min=-30)

        loss_text = -(mask_d * log_probs_text).sum(dim=1) / mask_d.sum(dim=1).clamp(min=1)
        loss_smiles = -(mask_s * log_probs_smiles).sum(dim=1) / mask_s.sum(dim=1).clamp(min=1)

        cont_loss = (loss_text.mean() + loss_smiles.mean()) / 2
        class_loss = self.loss_fn(class_logits, pair_types)

        loss = cont_loss + class_loss

        return loss, logits_per_text, logits_per_smiles, cont_loss, class_loss

        
class MoltrieverForEmbed(nn.Module):
    def __init__(self, 
                 smiles_model_path='ibm-research/MoLFormer-XL-both-10pct', 
                 language_model_path='allenai/scibert_scivocab_uncased', 
                 device='cuda',
                ):
        super().__init__()

        self.smiles_model_path = smiles_model_path
        self.language_model_path = language_model_path
        self.device = device

        self.smiles_model = transformers.AutoModel.from_pretrained(smiles_model_path, trust_remote_code=True)
        self.language_model = transformers.AutoModel.from_pretrained(language_model_path)

        self.smi_linear = nn.Linear(self.smiles_model.config.hidden_size, 512, bias=False)
        self.lang_linear = nn.Linear(self.language_model.config.hidden_size, 512, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))

        self.dropout = nn.Dropout(0.1)

        self.classifier = nn.Linear(512*2, 3)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, batch, is_smiles=True):
        
        if is_smiles:
            hidden = self.smiles_model(**batch).last_hidden_state[:, 0, :]
            hidden = self.smi_linear(hidden)
        else:
            hidden = self.language_model(**batch).last_hidden_state[:, 0, :]
            hidden = self.lang_linear(hidden)
            
        hidden = hidden / _get_vector_norm(hidden)

        return hidden, self.logit_scale