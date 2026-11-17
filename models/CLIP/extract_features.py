import os
import json,csv,hashlib,time,sys,os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel,CLIPProcessor
B=Path(os.environ['KAML_OUTPUTS_ROOT']);O=B/'coordinate_patch_experiments_20261105';encoder=sys.argv[1] if len(sys.argv)>1 else 'CLIP';torch.set_num_threads(4)
P={'PLIP':os.environ.get('KAML_PATH_04', 'external/plip'),'CLIP':os.environ.get('KAML_PATH_05', 'external/clip_base_patch16'),'provided_medical_CLIP':os.environ.get('KAML_PATH_06', 'external/Med_clip')};dest=O/'features'/encoder;dest.mkdir(parents=True,exist_ok=True)
if encoder!='provided_medical_CLIP':
 m=CLIPModel.from_pretrained(P[encoder],local_files_only=True).eval();pr=CLIPProcessor.from_pretrained(P[encoder],local_files_only=True);batch_size=32
 def encode(tiles):
  a=m.get_image_features(**pr(images=tiles,return_tensors='pt'));return a if isinstance(a,torch.Tensor) else a.pooler_output
else:
 import open_clip
 cfg=json.loads((Path(P[encoder])/'open_clip_config.json').read_text());m,_,pr=open_clip.create_model_and_transforms('ViT-L-14',pretrained=str(Path(P[encoder])/'model.safetensors'),force_image_size=cfg['model_cfg']['vision_cfg']['image_size']);m.eval();batch_size=1
 def encode(tiles):return m.encode_image(torch.stack([pr(t) for t in tiles]))
for p in m.parameters():p.requires_grad_(False)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for a in iter(lambda:f.read(8*1024*1024),b''):h.update(a)
 return h.hexdigest()
source={p.name:sha(p) for p in Path(P[encoder]).iterdir() if p.is_file() and p.suffix in ['.json','.bin','.safetensors']}
(dest/'encoder_source.json').write_text(json.dumps(dict(path=P[encoder],hashes=source,torch=torch.__version__,batch_size=batch_size,actual_name='CLIP-ViT-L-14-448px-MedICaT-ROCO' if encoder=='provided_medical_CLIP' else encoder),indent=2),encoding='utf-8')
rows=list(csv.DictReader(open(O/'audit/experiment_images.csv',encoding='utf-8-sig')));start=time.time();done=0;patches=0
for r in rows:
 target=dest/r['patient_id'];target.mkdir(exist_ok=True);out=target/(r['coordinate_key']+'.npz')
 if out.exists():
  try:
   a=np.load(out);assert a['features'].shape[0]==int(r['n_patches']) and np.isfinite(a['features']).all();done+=1;continue
  except Exception:pass
 a=np.load(O/'coords'/f"{r['coordinate_key']}.npz");coords=a['coords'];s=int(a['patch_size']);zz=[]
 with Image.open(r['image_path']) as original:
  im=original.convert('RGB')
  for off in range(0,len(coords),batch_size):
   tiles=[]
   for x,y in coords[off:off+batch_size]:
    x=int(x);y=int(y);tile=Image.new('RGB',(s,s),'white');tile.paste(im.crop((x,y,min(x+s,im.width),min(y+s,im.height))),(0,0));tiles.append(tile)
   with torch.inference_mode():z=torch.nn.functional.normalize(encode(tiles),dim=-1).cpu().numpy();zz.append(z)
 z=np.concatenate(zz);tmp=out.with_suffix('.partial.npz');np.savez_compressed(tmp,features=z,coords=coords,image_path=r['image_path'],coordinate_key=r['coordinate_key'],patch_size=s);tmp.replace(out);done+=1;patches+=len(coords)
 if done%10==0:
  status=dict(encoder=encoder,images_done=done,total_images=len(rows),patches_this_run=patches,elapsed_seconds=round(time.time()-start),last_patient=r['patient_id']);print(json.dumps(status),flush=True);(dest/'progress.json').write_text(json.dumps(status),encoding='utf-8')
(dest/'COMPLETE.json').write_text(json.dumps(dict(encoder=encoder,images=len(rows),elapsed_seconds=time.time()-start)),encoding='utf-8');print('ENCODER COMPLETE',encoder,flush=True)
