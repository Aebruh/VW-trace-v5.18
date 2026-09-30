#!/usr/bin/env python3
"""Layer-aware rectangle deletion under explicit colour and coverage budgets."""
from pathlib import Path
import argparse,json
import cv2,numpy as np
from shapely.ops import unary_union
from shapely_compat import STRtree
from shapely.geometry import box
from merge_v5_16 import polygon,boxmask
ROOT=Path(__file__).resolve().parent

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--protection',type=Path);ap.add_argument('--reference',type=Path,default=ROOT/'v5_15_final_objects.json');ap.add_argument('--profile',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--cache',type=Path,default=ROOT/'v5_14_export_cache.npz');args=ap.parse_args()
 profile=json.loads(args.profile.read_text());objs=json.loads(args.input.read_text());refobjs=json.loads(args.reference.read_text());c=np.load(args.cache);colors=c['colors'];bg=int(c['background'][0]);size=tuple(map(int,c['size']));scale=profile['evaluation_scale'];unit=float(profile.get('reference_width',512))/size[0]
 objs=[dict(o,rr=[float(v)*unit if k<4 else float(v) for k,v in enumerate(o['rr'])]) for o in objs]
 refobjs=[dict(o,rr=[float(v)*unit if k<4 else float(v) for k,v in enumerate(o['rr'])]) for o in refobjs]
 size=(int(profile.get('reference_width',512)),round(size[1]*unit))
 ps=[polygon(o['rr']) for o in objs];tree=STRtree(ps)
 reference=np.full((size[1]*scale,size[0]*scale),bg,np.int16)
 for o in refobjs:
  z=boxmask(polygon(o['rr']),size,scale)
  if z:sl,m=z;reference[sl][m]=o['color']
 protected=np.zeros(reference.shape,bool)
 for x0,y0,x1,y1 in profile.get('protected_boxes_normalized',[]):protected[round(y0*protected.shape[0]):round(y1*protected.shape[0]),round(x0*protected.shape[1]):round(x1*protected.shape[1])]=True
 if args.protection:protected|=np.load(args.protection)
 distance_cache={}
 removed=set();accepted=[];budget=profile.get('max_total_prune_error',100);used=0;limit=profile.get('max_prune_error',.5)
 for i in sorted(range(len(objs)),key=lambda j:ps[j].area):
  p=ps[i];ids=[int(j) for j in tree.query(p,predicate='intersects') if int(j)!=i and int(j) not in removed]
  if not ids:continue
  if p.difference(unary_union([ps[j] for j in ids])).area>1e-8:continue
  sl,m=boxmask(p,size,scale);y0,y1=sl[0].start,sl[0].stop;x0,x1=sl[1].start,sl[1].stop
  near=sorted(int(j) for j in tree.query(box(x0/scale,y0/scale,x1/scale,y1/scale)) if int(j) not in removed)
  before=np.full(m.shape,bg,np.int16);after=before.copy()
  for j in near:
   pts=np.rint(np.asarray(ps[j].exterior.coords)[:-1]*scale).astype(np.int32)-[x0,y0]
   cv2.fillConvexPoly(before,pts,int(objs[j]['color']))
   if j!=i:cv2.fillConvexPoly(after,pts,int(objs[j]['color']))
  valid=(after!=bg)&(reference[sl]!=bg)&(before!=bg)
  changed=(before!=after)&valid
  if np.any(changed&protected[sl]):continue
  boundary_bad=False
  for ci in np.unique(after[changed]):
   ci=int(ci)
   if ci not in distance_cache:distance_cache[ci]=cv2.distanceTransform((reference!=ci).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
   if np.any(changed & (after==ci) & (distance_cache[ci][sl]>profile.get('max_boundary_shift',1e9)*scale)):
    boundary_bad=True;break
  if boundary_bad:continue
  target=colors[reference[sl]].astype(np.int16)
  old=np.max(np.abs(colors[before].astype(np.int16)-target),axis=2)/255
  new=np.max(np.abs(colors[after].astype(np.int16)-target),axis=2)/255
  extra=float(np.maximum(new-old,0)[valid].sum())/(scale*scale)
  if extra>limit or used+extra>budget:continue
  removed.add(i);used+=extra;accepted.append({'index':i,'extra_error':extra})
 final=[o for i,o in enumerate(objs) if i not in removed];final=[dict(o,rr=[float(v)/unit if k<4 else float(v) for k,v in enumerate(o['rr'])]) for o in final];args.output.write_text(json.dumps(final));args.output.with_suffix('.report.json').write_text(json.dumps({'before':len(objs),'after':len(final),'removed':len(removed),'additional_colour_error_working_pixels':used,'max_per_object_error':limit,'total_budget':budget},indent=2));print('PRUNE',len(objs),'->',len(final),'error',used)
if __name__=='__main__':main()
