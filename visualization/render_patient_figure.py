"""Render ONE explicitly requested patient's figure directly from features and original images.
No patient/concept/image screening, qualitative ranking, candidate selection, or montage of saved figures.
"""
import argparse,csv
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager,colors
from matplotlib.patches import Rectangle
from threadpoolctl import threadpool_limits
threadpool_limits(2)
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--outputs-root',type=Path,required=True,help='External outputs-layout directory')
p.add_argument('--patient',required=True)
p.add_argument('--query-id',required=True,help='Explicit concept ID; no automatic concept selection')
p.add_argument('--image-index',type=int,required=True,help='1-based index in held-out attention export; no automatic image selection')
p.add_argument('--experiment',default='14.2')
p.add_argument('--cohort-dir',default='coordinate_patch_183_20261105')
p.add_argument('--query-bank',type=Path,help='Override the concept-query NPZ path')
p.add_argument('--query-key',default='name_definition_relation')
p.add_argument('--catalog',type=Path,help='Override the concept catalogue CSV')
p.add_argument('--output',type=Path,required=True)
p.add_argument('--vmin',type=float,default=.18)
p.add_argument('--vmax',type=float,default=.30)
p.add_argument('--dpi',type=int,default=300)
p.add_argument('--title-language',choices=['en','zh'],default='en')
args=p.parse_args();assert args.vmin<args.vmax
plt.rcParams.update({'font.family':'DejaVu Sans','axes.unicode_minus':False})
cnfont=font_manager.FontProperties(family='DejaVu Sans')
if Path('C:/Windows/Fonts/msyh.ttc').exists():cnfont=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
def read(path):return list(csv.DictReader(open(path,encoding='utf-8-sig')))
B=args.outputs_root;R=B/args.cohort_dir;pid=args.patient
pm={r['patient_id']:r for r in read(R/'audit/fixed_five_fold_manifest.csv')}
if pid not in pm:raise ValueError('Patient not in supplied cohort manifest')
catalog=read(args.catalog or B/'features/concept_catalog.csv');qids=[r['query_id'] for r in catalog]
q=np.load(args.query_bank or B/'quick_short_text_pilot/queries.npz')[args.query_key]
assert len(qids)==len(q)
display_labels={'Q0001': 'Papillary pattern', 'Q0002': 'Lobulated pattern', 'Q0003': 'Monomorphic cells', 'Q0005': 'Single-file pattern', 'Q0007': 'Biphasic pattern', 'Q0010': 'Pleomorphic cells', 'Q0011': 'Apocrine metaplasia', 'Q0016': 'Tubular structures', 'Q0017': 'Extracellular mucin', 'Q0019': 'Solid pattern', 'Q0021': 'Intraductal proliferation', 'Q0027': 'Micropapillary pattern', 'Q0028': 'Microcystic change', 'Q0035': 'Mitotic activity', 'Q0038': 'Mixed cell population', 'Q0040': 'Cribriform pattern', 'Q0041': 'Intracanalicular pattern', 'Q0042': 'Comedo necrosis', 'Q0043': 'Fibrosis', 'Q0052': 'Acinar proliferation', 'Q0057': 'Calcification', 'Q0060': 'Mucoid stroma'}
axis_names=[display_labels.get(r['query_id'],r.get('query_text_en') or r['query_id']) for r in catalog]
fold=int(pm[pid]['fold']);a=dict(np.load(R/f'results/{args.experiment}/fold_{fold}/attention/test/{pid}.npz'))
paths=a['image_paths'].tolist();keys=a['coordinate_keys'].tolist();n=len(paths)
if not 1<=args.image_index<=n:raise ValueError(f'--image-index must be1..{n}')
if args.query_id not in qids:raise ValueError('Unknown query ID')
for path in paths:
 if not Path(path).is_file():raise FileNotFoundError('Original histology images are required; feature-only example bundles cannot supply thumbnails: '+str(path))
features=[dict(np.load(R/'features/PLIP'/pid/f'{key}.npz')) for key in keys]
sims=[x['features']@q.T for x in features];matrix=np.stack([np.sort(s,axis=0)[-2:].mean(0) for s in sims])
assert np.isclose(a['image_attention'].sum(),1,atol=1e-5)
k=qids.index(args.query_id);ii=args.image_index-1;label=int(pm[pid]['label']);labelcn=('良性' if label==0 else '恶性') if args.title_language=='zh' else ('Benign' if label==0 else 'Malignant')
concept=axis_names[k];feat=features[ii];coords=feat['coords'];sim=sims[ii][:,k];top=np.argsort(-sim,kind='stable')[:3];path=paths[ii]
assert len(coords)==len(sim)
if 'patch_size' in feat:assert int(feat['patch_size'])==256,'This figure renderer expects the study256px coordinate patches'
c={'matrix':matrix};norm=colors.Normalize(args.vmin,args.vmax,clip=True);cmap=plt.get_cmap('magma');args.output.parent.mkdir(parents=True,exist_ok=True)
figheight=max(11.5,8.0+n*.22);fig=plt.figure(figsize=(18,figheight),facecolor='white');outer=fig.add_gridspec(2,1,height_ratios=[max(4.2,n*.26),3.1],hspace=.70)
# Top: no patient gate/concept attention bars and no explanatory supertitle.
upper=outer[0].subgridspec(1,4,width_ratios=[1.4,12.6,1.6,.32],wspace=.17)
thumbs=fig.add_subplot(upper[0,0]);thumbs.set_axis_off();thumbs.set_title('Histology images',fontsize=12,pad=10)
for i,p in enumerate(paths):
 tx=thumbs.inset_axes([.03,1-(i+.97)/n,.94,.90/n])
 with Image.open(p) as im:tx.imshow(im.convert('RGB').resize((240,180)))
 tx.set_xticks([]);tx.set_yticks([])
 for spine in tx.spines.values():spine.set_visible(i==ii);spine.set_edgecolor('#198D87');spine.set_linewidth(1.6)
ax=fig.add_subplot(upper[0,1]);hm=ax.imshow(c['matrix'],cmap=cmap,norm=norm,aspect='auto',interpolation='nearest');ax.set_yticks(range(n),[f'I{i+1:02d}' for i in range(n)],fontsize=10.5);ax.set_xticks(range(len(qids)),axis_names,rotation=58,ha='right',fontsize=10.5);ax.set_xlabel('Pathology concepts',fontsize=12,labelpad=11);ax.set_ylabel('Image ID',fontsize=12)
ax.get_xticklabels()[k].set_weight('bold');ax.get_xticklabels()[k].set_color('#147D78');ax.get_yticklabels()[ii].set_weight('bold');ax.get_yticklabels()[ii].set_color('#147D78');ax.add_patch(Rectangle((k-.5,ii-.5),1,1,fill=False,ec='#0A7C77',lw=2.1))
aw=fig.add_subplot(upper[0,2]);aw.barh(range(n),a['image_attention'],color=['#198D87' if i==ii else '#287CB5' for i in range(n)],height=.75);aw.set_ylim(n-.5,-.5);aw.set_xlim(0,1);aw.set_yticks([]);aw.set_xticks([0,.5,1]);aw.tick_params(axis='x',labelsize=9);aw.set_xlabel('Weight',fontsize=11);aw.set_title('MIL image\nweight',fontsize=12,pad=8);aw.spines['top'].set_visible(False);aw.spines['right'].set_visible(False)
cbax=fig.add_subplot(upper[0,3]);cb=fig.colorbar(hm,cax=cbax);cb.set_label('PLIP cosine similarity',fontsize=12,labelpad=9);cb.ax.tick_params(labelsize=10)
# Bottom: exact three requested columns. Original image stays unannotated.
lower=outer[1].subgridspec(1,3,width_ratios=[1,1,1.62],wspace=.17)
ao=fig.add_subplot(lower[0,0]);ah=fig.add_subplot(lower[0,1]);patchgrid=lower[0,2].subgridspec(1,3,wspace=.06);aps=[fig.add_subplot(patchgrid[0,j]) for j in range(3)]
with Image.open(path) as source:
 source=source.convert('RGB');rgb=np.asarray(source);w,h=source.size;ao.imshow(rgb);ao.axis('off');ao.set_title(f'Original histology (I{ii+1:02d})',fontsize=13,pad=12)
 total=np.zeros((h,w),dtype=np.float32);count=np.zeros((h,w),dtype=np.float32)
 for (x,y),value in zip(coords,sim):
  x,y=int(x),int(y);x1=min(x+256,w);y1=min(y+256,h);total[y:y1,x:x1]+=value;count[y:y1,x:x1]+=1
 heat=np.ma.array(total/np.maximum(count,1),mask=count==0);ah.imshow(rgb);ah.imshow(heat,cmap=cmap,norm=norm,alpha=.46,interpolation='nearest')
 for x,y in coords:ah.add_patch(Rectangle((int(x),int(y)),256,256,fill=False,ec='white',lw=.35,alpha=.5))
 for rank,j in enumerate(top,1):
  x,y=map(int,coords[j]);ah.add_patch(Rectangle((x,y),256,256,fill=False,ec='#00C5D2',lw=1.65));ah.text(x+9,y+26,str(rank),color='white',fontsize=9,weight='bold',bbox=dict(facecolor='#126D78',edgecolor='none',alpha=.85,pad=1))
  tile=Image.new('RGB',(256,256),'white');tile.paste(source.crop((x,y,min(x+256,w),min(y+256,h))),(0,0));aps[rank-1].imshow(tile,interpolation='nearest');aps[rank-1].set_title(f'Rank {rank}  |  P{j+1:02d}',fontsize=11,pad=9);aps[rank-1].set_xlabel(f'Cosine: {sim[j]:.3f}',fontsize=11,labelpad=7);aps[rank-1].set_xticks([]);aps[rank-1].set_yticks([])
  for spine in aps[rank-1].spines.values():spine.set_linewidth(.6);spine.set_edgecolor('#C3CDD1')
ah.set_xlim(0,w);ah.set_ylim(h,0);ah.axis('off');ah.set_title(f'Concept-patch heatmap\n{concept}',fontsize=12,pad=9)
# The overall third-column heading is outside individual patch axes, without extra method text.
boxpos=aps[1].get_position();fig.text(boxpos.x0+boxpos.width/2,boxpos.y1+.047,'Top-3 retrieved patches',ha='center',va='bottom',fontsize=13)
fig.suptitle(f'{pid}：{labelcn}',fontproperties=cnfont,fontsize=22,weight='bold',y=.975)
fig.subplots_adjust(left=.035,right=.955,top=.935,bottom=.055)
output=args.output;fig.savefig(output,dpi=args.dpi,facecolor='white');plt.close(fig)
with Image.open(output) as im:im.convert('RGB').save(output,dpi=(args.dpi,args.dpi))
print("Saved",str(args.output))
