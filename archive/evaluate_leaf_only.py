"""Evaluate uploaded checkpoint on originals and reviewed leaf-only crops.

Run after make_leaf_only.py. Original uploaded files and earlier output paths
are read only. This script writes only to leaf_only_results/.
"""
from pathlib import Path
import csv
import json
import numpy as np
from PIL import Image, ImageOps
import torch
import timm
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent.parent
CLASSES=['cbb','cbs','cmd','healthy']
OUT=ROOT/'leaf_only_results'
OUT.mkdir(exist_ok=True)
TRANSFORM=transforms.Compose([
 transforms.Resize((256,256)),transforms.CenterCrop(224),
 transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])
])

def main():
 checkpoint=torch.load(ROOT/'upload'/'01-best.pt',map_location='cpu',weights_only=False)
 assert checkpoint['class_names']==CLASSES
 model=timm.create_model('efficientnet_b0',pretrained=False,num_classes=4)
 model.load_state_dict(checkpoint['model_state_dict'],strict=True)
 model.eval()
 metadata=list(csv.DictReader((ROOT/'external_test_leaf_only'/'metadata.csv').open(encoding='utf-8')))
 all_rows=[]
 with torch.inference_mode():
  for item in metadata:
   for variant,path in [('original',ROOT/'source'/item['class_name']/item['original_filename']),
                        ('leaf_only',ROOT/'external_test_leaf_only'/item['crop_filename'])] if item['status']=='included' else [('original',ROOT/'source'/item['class_name']/item['original_filename'])]:
    with Image.open(path) as im:
     inp=TRANSFORM(ImageOps.exif_transpose(im).convert('RGB')).unsqueeze(0)
    probs=torch.softmax(model(inp),dim=1).squeeze(0).numpy()
    pred=int(probs.argmax())
    all_rows.append(dict(variant=variant,source_file=item['original_filename'],filepath=str(path.relative_to(ROOT)),
     true_label=item['class_name'],pred_label=CLASSES[pred],confidence=float(probs[pred]),
     correct=CLASSES[pred]==item['class_name'],**{'prob_'+c:float(probs[i]) for i,c in enumerate(CLASSES)}))
 with (OUT/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
  writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
 summaries={}
 for variant in ('original','leaf_only'):
  rows=[r for r in all_rows if r['variant']==variant]
  y=[r['true_label'] for r in rows];pred=[r['pred_label'] for r in rows]
  cm=confusion_matrix(y,pred,labels=CLASSES)
  summary=dict(n=len(rows),counts={c:y.count(c) for c in CLASSES},accuracy=float(accuracy_score(y,pred)),
   macro_f1=float(f1_score(y,pred,labels=CLASSES,average='macro',zero_division=0)),
   per_class=classification_report(y,pred,labels=CLASSES,output_dict=True,zero_division=0),
   confusion_matrix=cm.tolist())
  summaries[variant]=summary
  fig,ax=plt.subplots(figsize=(6,5));ax.imshow(cm,cmap='Blues')
  ax.set_xticks(range(4),CLASSES);ax.set_yticks(range(4),CLASSES)
  ax.set_xlabel('Predicted');ax.set_ylabel('True');ax.set_title(f'{variant} (n={len(rows)})')
  for i in range(4):
   for j in range(4):ax.text(j,i,str(cm[i,j]),ha='center',va='center')
  fig.tight_layout();fig.savefig(OUT/f'{variant}_confusion_matrix.png',dpi=180);plt.close(fig)
 # Paired 30-case comparison, excluding the same three cases in both variants.
 ids={r['source_file'] for r in all_rows if r['variant']=='leaf_only'}
 for variant in ('original','leaf_only'):
  rows=[r for r in all_rows if r['variant']==variant and r['source_file'] in ids]
  summaries['paired_'+variant]=dict(n=len(rows),accuracy=float(np.mean([r['correct'] for r in rows])),
   macro_f1=float(f1_score([r['true_label'] for r in rows],[r['pred_label'] for r in rows],labels=CLASSES,average='macro',zero_division=0)))
 (OUT/'metrics.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding='utf-8')
 for key,val in summaries.items():print(key,val['n'],f"accuracy={val['accuracy']:.4f}",f"macro_f1={val['macro_f1']:.4f}")
 print('original confusion matrix',summaries['original']['confusion_matrix'])
 print('leaf-only confusion matrix',summaries['leaf_only']['confusion_matrix'])

if __name__=='__main__':main()
