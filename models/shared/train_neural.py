import os
import json,csv,time,sys,pickle,random,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.metrics import roc_auc_score,confusion_matrix,f1_score
from patient_model import AttentionPool
torch.set_num_threads(2)
B=Path(os.environ['KAML_OUTPUTS_ROOT']);O=B/'coordinate_patch_183_20261105';R=Path(os.environ['KAML_RUN_DIR']);R.mkdir(exist_ok=True,parents=True)
K=Path('external/clinical')
def read(p):return list(csv.DictReader(open(p,encoding='utf-8-sig')))
def write(p,rr):
 p.parent.mkdir(exist_ok=True,parents=True)
 with p.open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rr for k in r)));w.writeheader();w.writerows(rr)
pm={r['patient_id']:r for r in read(O/'audit/fixed_five_fold_manifest.csv')};pids=sorted(pm)
arms={
 '1':dict(name='基础临床',clinical='base',image='none'),
 '2':dict(name='普通MIL',clinical='none',image='mil'),
 '3':dict(name='知识临床',clinical='knowledge',image='none'),
 '4.1':dict(name='知识图像：PLIP概念名称',clinical='none',image='name'),
 '4.2':dict(name='知识图像：名称＋定义＋关系',clinical='none',image='concat'),
 '5':dict(name='基础临床＋普通MIL',clinical='base',image='mil'),
 '6':dict(name='知识临床＋普通MIL',clinical='knowledge',image='mil'),
 '7':dict(name='基础临床＋知识图像（概念名称）',clinical='base',image='name'),
 '8':dict(name='双分支知识增强（概念名称）',clinical='knowledge',image='name'),
 '9.2':dict(name='知识临床＋名称定义关系增强图像',clinical='knowledge',image='concat'),
 '10.1':dict(name='PubMedBERT实体无邻居传播＋知识图像',clinical='no_neighbor',image='name'),
 '10.2':dict(name='随机重连GAT临床＋知识图像',clinical='random_gat',image='name'),
 '14.1':dict(name='知识临床＋基础临床＋知识图像（概念名称）',clinical='both',image='name'),
 '14.2':dict(name='知识临床＋基础临床＋知识图像（名称定义关系）',clinical='both',image='concat'),
 '15':dict(name='基础临床＋知识临床',clinical='both',image='none'),
 '16':dict(name='基础临床＋知识图像（名称定义关系）',clinical='base',image='concat'),
 '17':dict(name='基础临床＋知识临床＋普通MIL',clinical='both',image='mil')}
cfg=dict(seed=20261104,folds='unchanged patient manifest',epochs=70,batch_patients=8,optimizer='AdamW',lr=.001,weight_decay=.0001,hidden=64,dropout=.2,threshold=.5,checkpoint='last epoch; no outer-test model selection',image_encoder='frozen PLIP',patches='H5 coordinates and patch_size, level0; white-pad border; deduplicate exact coordinate rows and image+coordinate duplicates',knowledge_gate='q75/IQR fitted on each outer TRAIN; multiply fixed two-template patch overlap; same rule for both query versions',image_mil='all available coordinate patches -> image attention -> patient attention; one BCE loss per patient per epoch',clinical_provenance='existing fold-paired knowledge assets; original upstream code/cache/folds audited and exact vectors matched; no_neighbor is unsupervised seed entity mean768, random_gat trained using original architecture and same initialization seed per fold',arms=arms,aliases={'9.1':'8'},skipped=['11','12','13'],bootstrap=2000,interval_scope='patient bootstrap conditional on fixed OOF; does not include retraining uncertainty',experiment14='Concatenate train-fitted base onehot and train-standardized same-fold knowledge512, then shared clinical64 projection; image branch unchanged; both inputs used simultaneously',missing_kg='10 patients raw KG vector zero; scaler fitted on available training vectors only; standardized missing remains zero; trailing available mask hard-gates KG-only clinical projection; combined clinical retains base features and masks KG512 input; masks exported; upstream KG models reused, not retrained',code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(R/'training_config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
raw=json.loads((B/'audit/raw_clinical_inputs.json').read_text(encoding='utf-8'));clinical_only='--clinical-only' in sys.argv
assets={} if clinical_only else {p:dict(np.load(O/'patients'/f'{p}.npz')) for p in pids}
class Model(nn.Module):
 def __init__(self,cd,clinical,image):
  super().__init__();self.c_mode=clinical;self.i_mode=image
  self.visual=nn.Sequential(nn.Linear(512,64),nn.ReLU());self.patch=AttentionPool(64);self.image=AttentionPool(64);self.clinical=nn.Sequential(nn.Linear((cd-1 if clinical in ['knowledge','no_neighbor','random_gat','both'] else cd) or 1,64),nn.ReLU());self.cv=nn.Linear(512,64);self.cs=nn.Sequential(nn.Linear(64,64),nn.Tanh(),nn.Linear(64,1));self.head=nn.Sequential(nn.Linear(192,64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,1))
 def forward(self,x,mask,c,e,g,detail=False):
  n=len(c);zero=c.new_zeros((n,64));pa=None;ia=None;ca=None
  if self.i_mode!='none':
   v=x.new_zeros((*mask.shape,64));v[mask]=self.visual(x);im,pa=self.patch(v,mask);vi,ia=self.image(im,mask.any(-1))
  else:vi=zero
  if self.c_mode in ['knowledge','no_neighbor','random_gat']:cl=self.clinical(c[:,:-1])*c[:,-1:]
  elif self.c_mode=='both':cl=self.clinical(torch.cat([c[:,:-513],c[:,-513:-1]*c[:,-1:]],-1))
  else:cl=self.clinical(c) if self.c_mode!='none' else zero
  if self.i_mode in ['name','concat']:
   cv=self.cv(e);ca=torch.softmax(self.cs(cv).squeeze(-1),-1)*g;ca=ca/ca.sum(-1,keepdim=True).clamp_min(1e-12);ki=(ca.unsqueeze(-1)*cv).sum(1)
  else:ki=zero
  z=torch.cat([cl,vi,ki],-1);logit=self.head(z).squeeze(-1)
  return dict(logit=logit,embedding=z,clinical=cl,image=vi,concept=ki,patch_attention=pa,image_attention=ia,concept_attention=ca) if detail else logit
def batch(ids,cl,im,gate):
 if im=='none':x=np.zeros((len(ids),1,1,512),dtype='float32');mask=np.zeros((len(ids),1,1),dtype=bool)
 else:
  ni=max(len(assets[p]['visual']) for p in ids);npmax=max(assets[p]['visual'].shape[1] for p in ids);mask=np.zeros((len(ids),ni,npmax),dtype=bool);chunks=[]
  for i,p in enumerate(ids):
   a=assets[p]['visual'];mm=assets[p]['patch_mask'];chunks.append(a[mm]);mask[i,:a.shape[0],:a.shape[1]]=mm
  x=np.concatenate(chunks,axis=0)
 e=np.stack([assets[p][im+'_evidence'] for p in ids]) if im in ['name','concat'] else np.zeros((len(ids),22,512),dtype='float32')
 g=np.stack([gate[p] for p in ids]) if gate else np.zeros((len(ids),22),dtype='float32')
 return [torch.from_numpy(x),torch.from_numpy(mask),torch.tensor(np.stack([cl[p] for p in ids]),dtype=torch.float32),torch.tensor(e,dtype=torch.float32),torch.tensor(g,dtype=torch.float32)]
def metrics(y,p):
 y=np.array(y,dtype=int);p=np.array(p);pred=p>=.5;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
 def div(a,b):return float(a/b) if b else float('nan')
 return dict(AUC=float(roc_auc_score(y,p)) if len(set(y))==2 else float('nan'),ACC=float(np.mean(pred==y)),Recall=div(tp,tp+fn),Specificity=div(tn,tn+fp),Precision=div(tp,tp+fp),F1=float(f1_score(y,pred,zero_division=0)),NPV=div(tn,tn+fn))
def load_clinical(fold,kind,train,test):
 if kind=='both':
  base,bprep=load_clinical(fold,'base',train,test);kg,kprep=load_clinical(fold,'knowledge',train,test)
  return {p:np.concatenate([base[p],kg[p]]).astype('float32') for p in pids},dict(base=bprep,knowledge=kprep,order=['base_onehot','knowledge_standardized512'],base_dim=len(base[train[0]]),knowledge_dim=512)
 if kind=='none':return {p:np.zeros(1,dtype='float32') for p in pids},None
 if kind=='base':
  enc=OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float32);a=enc.fit_transform([raw['patients'][p] for p in train]);b=enc.transform([raw['patients'][p] for p in test]);return dict(zip(train+test,np.concatenate([a,b]))),enc
 source=O/'clinical'/kind;dd={}
 for split,expected in [('train',train),('test',test)]:
  a=np.load(source/f'fold_{fold}/{split}_patient_embeddings.npz');assert set(a['patient_ids'])==set(expected)
  for p,y,v in zip(a['patient_ids'],a['labels'],a['embeddings']):assert int(y)==int(pm[str(p)]['label']);dd[str(p)]=v
 valid_train=[p for p in train if int(pm[p]['kg_available_mask'])==1];scaler=StandardScaler().fit(np.stack([dd[p] for p in valid_train]));dim=len(dd[valid_train[0]]);return {p:np.r_[scaler.transform(dd[p][None])[0] if int(pm[p]['kg_available_mask']) else np.zeros(dim),int(pm[p]['kg_available_mask'])].astype('float32') for p in pids},scaler
def train_arm(key,arm,fold,tr,te):
 dest=R/key/f'fold_{fold}';dest.mkdir(parents=True,exist_ok=True)
 if (dest/'COMPLETE.json').exists():return
 seed=cfg['seed']+fold;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);cl,prep=load_clinical(fold,arm['clinical'],tr,te);im=arm['image'];gate=None;gm={}
 if im in ['name','concat']:
  s=np.stack([assets[p][im+'_scores'] for p in tr]);q75=np.quantile(s,.75,axis=0);iqr=np.maximum(q75-np.quantile(s,.25,axis=0),.02);gate={p:(np.clip((assets[p][im+'_scores']-q75)/iqr,0,1)*assets[p][im+'_agreement']).astype('float32') for p in pids};gm=dict(q75=q75.tolist(),iqr=iqr.tolist(),train_ids=tr)
 m=Model(len(cl[tr[0]]),arm['clinical'],im);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);rng=np.random.default_rng(seed);history=[];start=time.time()
 for epoch in range(70):
  m.train();order=rng.permutation(tr).tolist();total=0
  for off in range(0,len(order),8):
   ids=order[off:off+8];y=torch.tensor([int(pm[p]['label']) for p in ids],dtype=torch.float32);loss=nn.functional.binary_cross_entropy_with_logits(m(*batch(ids,cl,im,gate)),y);assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step();total+=float(loss.detach())*len(ids)
  history.append(dict(epoch=epoch+1,train_loss=total/len(tr)))
  if (epoch+1)%10==0:print(json.dumps(dict(item=key,fold=fold,epoch=epoch+1,train_loss=history[-1]['train_loss'],seconds=round(time.time()-start))),flush=True)
 write(dest/'training.csv',history);torch.save(dict(state_dict=m.state_dict(),fold=fold,item=key,arm=arm,clinical_dim=len(cl[tr[0]]),config=cfg,train_ids=tr,test_ids=te),dest/'model.pt')
 with (dest/'clinical_preprocessing.pkl').open('wb') as f:pickle.dump(prep,f)
 (dest/'gate_calibration.json').write_text(json.dumps(gm),encoding='utf-8');m.eval();preds=[]
 with torch.inference_mode():
  for split,idsall in [('train',tr),('test',te)]:
   z=[];cv=[];vi=[];ki=[]
   for off in range(0,len(idsall),4):
    ids=idsall[off:off+4];args=batch(ids,cl,im,gate);d=m(*args,detail=True);pr=d['logit'].sigmoid().tolist();z.extend(d['embedding'].numpy());cv.extend(d['clinical'].numpy());vi.extend(d['image'].numpy());ki.extend(d['concept'].numpy())
    for i,p in enumerate(ids):
     preds.append(dict(patient_id=p,fold=fold,split=split,label=int(pm[p]['label']),probability=pr[i]))
     if im!='none':
      ad=dest/'attention'/split;ad.mkdir(parents=True,exist_ok=True);ni=len(assets[p]['visual']);npa=assets[p]['visual'].shape[1]
      np.savez_compressed(ad/f'{p}.npz',patch_attention=d['patch_attention'][i,:ni,:npa].numpy(),image_attention=d['image_attention'][i,:ni].numpy(),concept_attention=d['concept_attention'][i].numpy() if d['concept_attention'] is not None else np.zeros(22),gates=gate[p] if gate else np.zeros(22),image_paths=assets[p]['image_paths'],coordinate_keys=assets[p]['coordinate_keys'])
   np.savez_compressed(dest/f'{split}_patient_embeddings.npz',patient_ids=idsall,labels=[int(pm[p]['label']) for p in idsall],embeddings=np.array(z),clinical=np.array(cv),image=np.array(vi),concept=np.array(ki),clinical_input=np.stack([cl[p] for p in idsall]),kg_available_mask=np.array([int(pm[p]['kg_available_mask']) for p in idsall],dtype=np.uint8),kg_missing_mask=np.array([int(pm[p]['kg_missing_mask']) for p in idsall],dtype=np.uint8))
   with (dest/f'{split}_patient_embeddings.pkl').open('wb') as f:pickle.dump(dict(zip(idsall,z)),f)
 write(dest/'predictions.csv',preds);(dest/'COMPLETE.json').write_text(json.dumps(dict(item=key,fold=fold,seconds=time.time()-start)),encoding='utf-8')
def summarize():
 table=[];foldrows=[];ci=[]
 labels={**{k:a['name'] for k,a in arms.items()},'9.1':'知识临床＋PLIP概念表示增强图像','11':'PLIP','12':'CLIP','13':'医学CLIP（实际MedICaT-ROCO）'}
 for item in ['1','2','3','4.1','4.2','5','6','7','8','9.1','9.2','10.1','10.2','11','12','13','14.1','14.2','15','16','17']:
  canonical='8' if item=='9.1' else item
  if item in ['11','12','13']:table.append(dict(Item=item,Model=labels[item],Status='按用户要求跳过',AUC='',ACC='',Recall='',Specificity='',Precision='',F1='',NPV='',Patients='',Note='不执行'));continue
  files=[R/canonical/f'fold_{k}/predictions.csv' for k in range(1,6)]
  if not all(f.exists() and (f.parent/'COMPLETE.json').exists() for f in files):table.append(dict(Item=item,Model=labels[item],Status='尚未完成',AUC='',ACC='',Recall='',Specificity='',Precision='',F1='',NPV='',Patients='',Note='等待坐标特征或训练'));continue
  rr=[r for f in files for r in read(f) if r['split']=='test'];assert len(rr)==len(pm) and {r['patient_id'] for r in rr}==set(pm)
  assert all(r['fold']==pm[r['patient_id']]['fold'] for r in rr);rr.sort(key=lambda r:r['patient_id']);y=[int(r['label']) for r in rr];p=[float(r['probability']) for r in rr];score=metrics(y,p)
  write(R/canonical/'oof_predictions.csv',rr)
  table.append(dict(Item=item,Model=labels[item],Status='完成（复用第8项）' if item=='9.1' else '完成',**score,Patients=len(rr),Note='与第8项完全同配置，非独立实验' if item=='9.1' else '患者级固定五折OOF，label1恶性'))
  if item!='9.1':
   for k in range(1,6):
    frr=[r for r in rr if int(r['fold'])==k];foldrows.append(dict(Item=item,fold=k,n=len(frr),**metrics([int(r['label']) for r in frr],[float(r['probability']) for r in frr])))
   # Intervals generated only once after all tasks finish, not every intermediate update.
 write(R/'OOF指标汇总.csv',table)
 if foldrows:write(R/'fold_metrics.csv',foldrows)
 return table
if __name__=='__main__':
 selected=['16','17'] if '--items16-17' in sys.argv else (['15'] if '--item15' in sys.argv else (['14.1','14.2'] if '--items14' in sys.argv else (['1','3'] if clinical_only else list(arms))))
 selected=[os.environ['KAML_MODEL_KEY']]
 for k in range(1,6):
  tr=[p for p in pids if int(pm[p]['fold'])!=k];te=[p for p in pids if int(pm[p]['fold'])==k]
  for key in selected:train_arm(key,arms[key],k,tr,te)
 summarize()
 if not clinical_only:(R/'COMPLETE.json').write_text(json.dumps(dict(items=list(arms),folds=5,patients=len(pm),aliases={'9.1':'8'},skipped=['11','12','13'])),encoding='utf-8')
