#!/usr/bin/env python3
"""Export a rectangle scene to GMD, decode it, and validate its geometry."""
from pathlib import Path
import argparse,json,math
from collections import Counter
from xml.sax.saxutils import escape
import xml.etree.ElementTree as ET
import cv2,numpy as np
from PIL import Image
from shapely.ops import unary_union
from merge_v5_16 import polygon
import gd_vw_tracer_v5 as v5

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--protection',type=Path);ap.add_argument('--reference',type=Path,required=True);ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--profile',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--name',default='v5_17_optimized');args=ap.parse_args()
 args.output_dir.mkdir(parents=True,exist_ok=True);objs=json.loads(args.input.read_text());reference=json.loads(args.reference.read_text());cache=np.load(args.cache);colors=cache['colors'];bg=int(cache['background'][0]);size=tuple(map(int,cache['size']));profile=json.loads(args.profile.read_text());factor=800/size[0]
 assert len(objs)<9999,'Exporter uses unique painter-order Z values; split larger scenes.'
 defs=''.join(f'1_{int(c[0])}_2_{int(c[1])}_3_{int(c[2])}_4_-1_6_{940+i}_7_1_8_1_11_0_12_0_13_0_15_0_18_0|' for i,c in enumerate(colors))
 entries=['kS38,'+defs+';kA13,0;kA15,0;kA16,0;kA14,;']
 for z,o in enumerate(objs):
  x,y,w,h,a=o['rr'];props=['1','211','2',f'{150+x*factor:.7f}','3',f'{150+(size[1]-y)*factor:.7f}','6',f'{a:.7f}','128',f'{w*factor/30:.9f}','129',f'{h*factor/30:.9f}','21',str(940+o['color']),'22',str(940+o['color']),'24','3','25',str(z),'20','2','57','1','64','1','67','1','96','1','116','1','121','1'];entries.append(','.join(props)+';')
 xml=f'<?xml version="1.0"?><plist version="1.0" gjver="2.0"><dict><k>kCEK</k><i>4</i><k>k2</k><s>{escape(args.name)}</s><k>k4</k><s>{escape("".join(entries))}</s><k>k5</k><s>Vector overlap optimization. {len(objs)} objects.</s><k>k13</k><t/><k>k21</k><i>2</i><k>k16</k><i>1</i></dict></plist>'
 path=args.output_dir/(args.name+'.gmd');path.write_text(xml)
 nodes=list(ET.parse(path).getroot().find('dict'));kv={nodes[i].text:nodes[i+1].text for i in range(0,len(nodes),2)};decoded=[]
 for e in kv['k4'].split(';'):
  p=e.split(',')
  if p[0]!='1':continue
  q=dict(zip(p[::2],p[1::2]));decoded.append({'color':int(q['21'])-940,'rr':[(float(q['2'])-150)/factor,size[1]-(float(q['3'])-150)/factor,float(q['128'])*30/factor,float(q['129'])*30/factor,float(q['6'])]})
 assert len(decoded)==len(objs)
 corner_error=max(polygon(a['rr']).hausdorff_distance(polygon(b['rr'])) for a,b in zip(objs,decoded));assert corner_error<1e-5
 original_union=unary_union([polygon(o['rr']) for o in reference]);new_union=unary_union([polygon(o['rr']) for o in objs]);missing=original_union.difference(new_union).area;spill=new_union.difference(original_union).area
 from shapely.geometry import Polygon
 def shell(g):
  return unary_union([Polygon(p.exterior) for p in (g.geoms if hasattr(g,'geoms') else [g])])
 outer_difference=shell(original_union).symmetric_difference(shell(new_union)).area
 assert missing<1e-5 and outer_difference<1e-5,(missing,spill,outer_difference)
 # Android memory-safe validation. Quality-mode geometry has already been
 # selected; validation at 512px avoids a large 2048px temporary allocation.
 def render_at(items,width):
  scale=width/size[0]
  scaled=[dict(o,rr=[float(v)*scale if k<4 else float(v) for k,v in enumerate(o['rr'])]) for o in items]
  return v5.render((width,round(size[1]*scale)),colors,scaled,bg,2)
 image=render_at(decoded,512);base=render_at(reference,512)
 a=np.asarray(base,dtype=np.int16);b=np.asarray(image,dtype=np.int16);bgc=colors[bg].astype(np.int16);fg=np.max(np.abs(a-bgc),axis=2)>20;diff=np.abs(a-b);protected=np.zeros(fg.shape,bool)
 for x0,y0,x1,y1 in profile.get('protected_boxes_normalized',[]):protected[round(y0*fg.shape[0]):round(y1*fg.shape[0]),round(x0*fg.shape[1]):round(x1*fg.shape[1])]=True
 if args.protection:
  mask=np.load(args.protection).astype(bool);protected|=np.asarray(Image.fromarray(mask).resize((image.width,image.height),Image.Resampling.NEAREST)).astype(bool)
 report={'outer_silhouette_symmetric_difference':outer_difference,'repaired_enclosed_area':spill,'baseline_roles':dict(Counter(o.get('role','') for o in reference)),'baseline_objects':len(reference),'objects':len(objs),'reduction_percent':round(100*(len(reference)-len(objs))/len(reference),2),'lost_foreground_area_working_pixels':missing,'added_foreground_area_working_pixels':spill,'export_max_corner_error_working_pixels':corner_error,'foreground_rgb_mae_0_to_255':float(diff[fg].mean()) if fg.any() else 0.0,'protected_roi_rgb_mae_0_to_255':float(diff[protected].mean()) if protected.any() else None,'max_gd_z':len(objs)-1,'role_counts':dict(Counter(o.get('role','') for o in objs)),'in_game_verified':False,'limitations':'Android memory-safe validation at 512px. Geometry/export is unchanged. Offline preview is not a GD renderer.','mobile_validation_width':512}
 image.save(args.output_dir/(args.name+'_normal.png'));del a,b,fg,diff,protected
 preview=render_at(decoded,1024);preview.save(args.output_dir/(args.name+'_preview.png'));preview.close();image.close();base.close()
 (args.output_dir/(args.name+'_report.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
