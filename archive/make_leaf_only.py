"""Create a reviewed leaf-focused external evaluation set from source/.

Boxes use fractions of source width/height. One crop per source avoids counting
multiple panels from the same figure as independent evaluation cases.
"""
from pathlib import Path
from PIL import Image, ImageOps
import csv
import hashlib

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'external_test_leaf_only'

# filename: (x0, y0, x1, y1, note); None means excluded.
BOXES = {
 'cbb': {
  'cbb_01.jpg': (.02,.02,.96,.84,'leaf cluster with lesions; omit lower background'),
  'cbb_02.jpg': (.08,.05,.92,.88,'symptomatic leaf cluster'),
  'cbb_03.jpg': None,
  'cbb_04.webp': (.02,.01,.31,.78,'panel A, underside leaf; omit panel label'),
  'cbb_05.jpg': (.51,.01,.96,.46,'panel b, necrotic leaf; omit label and other panels'),
  'cbb_06.jpg': (.68,.03,.98,.44,'panel c, leaf lesions; arrow remains on leaf'),
  'cbb_07.jpg': (.08,.05,.94,.94,'single symptomatic leaf'),
  'cbb_08.webp': (.01,.01,.39,.70,'panel A, spotted leaf; omit caption and other panels'),
 },
 'cbs': {
  'cbs_01.jpg': (.01,.01,.99,.99,'single leaf cluster'),
  'cbs_02.jpg': (.02,.01,.98,.78,'leaf with spots; omit yellow text band'),
  'cbs_03.jpg': (.01,.01,.99,.99,'symptomatic leaf cluster'),
  'cbs_04.jpg': (.02,.01,.98,.99,'leaf cluster with subtle spots'),
  'cbs_06.jpg': None,
  'cbs_07.webp': (.02,.02,.46,.85,'panel A, leaf with spots; omit caption and other panels'),
 },
 'cmd': {
  'cmd_01.jpg': (.02,.01,.98,.98,'symptomatic leaf cluster'),
  'cmd_02.jpg': (.08,.01,.92,.99,'symptomatic leaf cluster'),
  'cmd_03.jpg': (.02,.01,.98,.99,'symptomatic leaf cluster'),
  'cmd_04.jpg': (.25,.08,.81,.83,'central mosaic leaf; omit much of field background'),
  'cmd_05.jpg': (.02,.01,.98,.98,'symptomatic leaf cluster'),
  'cmd_06.jpg': (.16,.05,.84,.93,'symptomatic leaves; minimize hand and soil'),
  'cmd_07.jpg': (.06,.04,.94,.95,'symptomatic leaf cluster'),
  'cmd_08.jpg': (.10,.03,.96,.98,'symptomatic leaf cluster'),
  'cmd_09.jpg': (.09,.04,.90,.91,'symptomatic leaf cluster; omit corner mark'),
  'cmd_10.webp': (.15,.03,.88,.93,'symptomatic leaf cluster'),
  'cmd_11.jpeg': (.03,.03,.96,.98,'symptomatic leaf cluster'),
 },
 'healthy': {
  'healthy_01.jpg': (.16,.12,.93,.91,'foreground leaf cluster'),
  'healthy_04.jpg': (.28,.27,.72,.82,'foreground healthy leaf cluster; omit field'),
  'healthy_05.jpg': None,
  'healthy_06.jpg': (.06,.04,.96,.96,'foreground healthy leaf cluster'),
  'healthy_08.webp': (.05,.05,.94,.96,'foreground healthy leaf cluster'),
  'healthy_09.jpeg': (.15,.02,.86,.95,'healthy leaf cluster'),
  'healthy_10.jpg': (.14,.02,.86,.96,'foreground healthy leaf cluster'),
  'healthy_11_commons.jpg': (.17,.15,.84,.75,'foreground healthy leaf cluster; source label requires verification'),
 },
}

EXCLUDED = {
 'cbb_03.jpg': 'whole plant; no sufficiently detailed single leaf',
 'cbs_06.jpg': 'text and symptom description only; no photograph of leaf',
 'healthy_05.jpg': 'field/whole plants; small overlapping leaves',
}

def run():
 rows=[]
 for cls, specs in BOXES.items():
  target=OUT/'labeled'/cls
  target.mkdir(parents=True,exist_ok=True)
  for filename, spec in specs.items():
   src=ROOT/'source'/cls/filename
   assert src.is_file(),src
   digest=hashlib.sha256(src.read_bytes()).hexdigest()
   if spec is None:
    rows.append(dict(original_filename=filename,crop_filename='',class_name=cls,
     source_url='',source_name='',panel_crop_note=EXCLUDED[filename],status='excluded',
     x0='',y0='',x1='',y1='',source_sha256=digest))
    continue
   with Image.open(src) as raw:
    im=ImageOps.exif_transpose(raw).convert('RGB')
    w,h=im.size
    x0,y0,x1,y1,note=spec
    box=tuple(round(v*s) for v,s in zip((x0,y0,x1,y1),(w,h,w,h)))
    crop=im.crop(box)
    assert min(crop.size)>=85,(filename,crop.size)
    name=src.stem+'_leaf.jpg'
    crop.save(target/name,quality=95,subsampling=0)
   rows.append(dict(original_filename=filename,crop_filename=f'labeled/{cls}/{name}',
    class_name=cls,source_url='',source_name='',panel_crop_note=note,status='included',
    x0=box[0],y0=box[1],x1=box[2],y1=box[3],source_sha256=digest))
 OUT.mkdir(exist_ok=True)
 with (OUT/'metadata.csv').open('w',newline='',encoding='utf-8') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 print('included',sum(r['status']=='included' for r in rows),'excluded',sum(r['status']=='excluded' for r in rows))

if __name__=='__main__':run()
