import torch
from torch import nn

class AttentionPool(nn.Module):
 def __init__(self,dim,hidden=64):
  super().__init__();self.attention=nn.Sequential(nn.Linear(dim,hidden),nn.Tanh(),nn.Linear(hidden,1))
 def forward(self,x,mask):
  scores=self.attention(x).squeeze(-1).masked_fill(~mask,-1e9)
  a=torch.softmax(scores,dim=-1)*mask
  a=a/a.sum(-1,keepdim=True).clamp_min(1e-12)
  return (a.unsqueeze(-1)*x).sum(-2),a

class PatientModel(nn.Module):
 """One logit per patient. Image axis and patch axis remain distinct."""
 def __init__(self,clinical_dim,concept_count,hidden=64,dropout=.2):
  super().__init__();self.visual=nn.Sequential(nn.Linear(512,hidden),nn.ReLU())
  self.patch_pool=AttentionPool(hidden);self.image_pool=AttentionPool(hidden)
  self.concept_value=nn.Linear(512,hidden);self.concept_score=nn.Sequential(nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,1))
  self.clinical=nn.Sequential(nn.Linear(clinical_dim,hidden),nn.ReLU())
  self.classifier=nn.Sequential(nn.Linear(hidden*3,hidden),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden,1))
 def forward(self,patches,image_mask,clinical,evidence,gates,return_details=False):
  v=self.visual(patches)
  patch_mask=image_mask.unsqueeze(-1).expand(patches.shape[:-1])
  images,pa=self.patch_pool(v,patch_mask)
  patient,ia=self.image_pool(images,image_mask)
  cv=self.concept_value(evidence)
  # Multiply by fixed gates, then normalize. All-zero gates produce exact zero.
  ca=torch.softmax(self.concept_score(cv).squeeze(-1),dim=-1)*gates
  ca=ca/ca.sum(-1,keepdim=True).clamp_min(1e-12)
  concept=(ca.unsqueeze(-1)*cv).sum(1)
  clin=self.clinical(clinical)
  z=torch.cat([clin,patient,concept],dim=-1);logit=self.classifier(z).squeeze(-1)
  if return_details:return dict(logit=logit,embedding=z,clinical=clin,image=patient,concept=concept,patch_attention=pa,image_attention=ia,concept_attention=ca)
  return logit

def technical_checks():
 torch.manual_seed(1);torch.set_num_threads(2)
 m=PatientModel(7,5,dropout=0).eval()
 x=torch.randn(2,3,12,512);mask=torch.tensor([[True,True,False],[True,True,True]])
 c=torch.randn(2,7);e=torch.randn(2,5,512);g=torch.rand(2,5)
 base=m(x,mask,c,e,g)
 patch_order=torch.randperm(12);image_order=torch.tensor([2,0,1])
 assert torch.allclose(base,m(x[:,:,patch_order],mask,c,e,g),atol=1e-6)
 assert torch.allclose(base,m(x[:,image_order],mask[:,image_order],c,e,g),atol=1e-6)
 x2=x.clone();x2[0,2]=10000
 assert torch.allclose(base,m(x2,mask,c,e,g),atol=1e-6)
 d=m(x,mask,c,e,torch.zeros_like(g),True)
 assert torch.equal(d['concept'],torch.zeros_like(d['concept']))
 base.sum().backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
 return dict(patch_permutation=True,image_permutation=True,padding_invariance=True,zero_gate_skip=True,finite_gradients=True,scope='synthetic technical validation only; not an experimental result')

if __name__=='__main__':
 import json
 print(json.dumps(technical_checks(),indent=2))
