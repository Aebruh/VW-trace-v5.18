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
 ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--protection',type=Path);ap.add_argument('--reference',type=Path,required=True);ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--profile',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--name',default='v5_18_balanced');args=ap.parse_args()
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
 tolerance=profile.get('outer_tolerance',0)*size[0]/profile.get('reference_width',512)
 deep_missing=original_union.difference(new_union).difference(original_union.boundary.buffer(tolerance+1e-7)).area
 excessive_spill=new_union.difference(shell(original_union).buffer(tolerance+1e-7)).area
 assert deep_missing<1e-5 and excessive_spill<1e-5,(deep_missing,excessive_spill)
 source_scale=2048/size[0]
 def render_full(items):
  scaled=[dict(o,rr=[float(v)*source_scale if k<4 else float(v) for k,v in enumerate(o['rr'])]) for o in items]
  return v5.render((2048,round(size[1]*source_scale)),colors,scaled,bg,2)
 image=render_full(decoded);base=render_full(reference);image.save(args.output_dir/(args.name+'_full.png'));image.resize((1024,round(size[1]*1024/size[0])),Image.Resampling.LANCZOS).save(args.output_dir/(args.name+'_preview.png'))
 a=np.asarray(base,dtype=np.int16);b=np.asarray(image,dtype=np.int16);fg=np.linalg.norm(a-colors[bg],axis=2)>20;diff=np.abs(a-b);protected=np.zeros(fg.shape,bool)
 for x0,y0,x1,y1 in profile.get('protected_boxes_normalized',[]):protected[round(y0*fg.shape[0]):round(y1*fg.shape[0]),round(x0*fg.shape[1]):round(x1*fg.shape[1])]=True
 if args.protection:
  mask=np.load(args.protection).astype(bool);protected|=np.asarray(Image.fromarray(mask).resize((image.width,image.height),Image.Resampling.NEAREST)).astype(bool)
 report={'outer_silhouette_symmetric_difference':outer_difference,'outer_tolerance_reference_pixels':profile.get('outer_tolerance',0),'new_deep_internal_gap_area':deep_missing,'excessive_silhouette_spill_area':excessive_spill,'baseline_roles':dict(Counter(o.get('role','') for o in reference)),'baseline_objects':len(reference),'objects':len(objs),'reduction_percent':round(100*(len(reference)-len(objs))/len(reference),2),'lost_foreground_area_working_pixels':missing,'added_foreground_area_working_pixels':spill,'export_max_corner_error_working_pixels':corner_error,'foreground_rgb_mae_0_to_255':float(diff[fg].mean()),'protected_roi_rgb_mae_0_to_255':float(diff[protected].mean()) if protected.any() else None,'max_gd_z':len(objs)-1,'role_counts':dict(Counter(o.get('role','') for o in objs)),'in_game_verified':False,'limitations':'Intentional bounded silhouette and internal-boundary simplification. Offline preview is decoded from GMD but is not the game renderer.'}
 def metrics(width):
  dims=(width,round(size[1]*width/size[0]));aa=np.asarray(base.resize(dims,Image.Resampling.LANCZOS),dtype=np.float32);bb=np.asarray(image.resize(dims,Image.Resampling.LANCZOS),dtype=np.float32)
  mask=(np.max(abs(aa-colors[bg]),2)>20)|(np.max(abs(bb-colors[bg]),2)>20);dd=abs(aa-bb)
  ga=cv2.cvtColor(aa,cv2.COLOR_RGB2GRAY);gb=cv2.cvtColor(bb,cv2.COLOR_RGB2GRAY);mu_a=cv2.GaussianBlur(ga,(7,7),1.5);mu_b=cv2.GaussianBlur(gb,(7,7),1.5);va=cv2.GaussianBlur(ga*ga,(7,7),1.5)-mu_a*mu_a;vb=cv2.GaussianBlur(gb*gb,(7,7),1.5)-mu_b*mu_b;cov=cv2.GaussianBlur(ga*gb,(7,7),1.5)-mu_a*mu_b
  ss=((2*mu_a*mu_b+6.5025)*(2*cov+58.5225))/((mu_a*mu_a+mu_b*mu_b+6.5025)*(va+vb+58.5225))
  return {'canvas_width':width,'foreground_rgb_mae':float(dd[mask].mean()),'foreground_ssim':float(ss[mask].mean()),'changed_foreground_fraction_over_16':float((dd.max(2)[mask]>16).mean())}
 report['viewing_scale_metrics']={label:metrics(width) for label,width in [('normal',512),('medium',1024),('full',2048)]}
 normal=report['viewing_scale_metrics']['normal'];report['passes_normal_threshold']=normal['foreground_rgb_mae']<=profile.get('normal_rgb_mae_limit',1.5) and normal['changed_foreground_fraction_over_16']<=profile.get('normal_changed_fraction_limit',.04)
 image.resize((512,round(size[1]*512/size[0])),Image.Resampling.LANCZOS).save(args.output_dir/(args.name+'_normal.png'))
 report['protected_foreground_rgb_mae']=float(diff[protected&fg].mean()) if np.any(protected&fg) else None
 (args.output_dir/(args.name+'_report.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
