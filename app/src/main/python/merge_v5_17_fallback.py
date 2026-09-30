#!/usr/bin/env python3
"""Reusable vector-rectangle merge optimizer with silhouette/coverage invariants.
Raster masks score colour error only; candidates are vector minimum-area rectangles.
"""
from pathlib import Path
import argparse,json,math,heapq
from collections import defaultdict
import cv2,numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely_compat import STRtree
ROOT=Path(__file__).resolve().parent

def polygon(rr):
 x,y,w,h,a=rr;a=math.radians(a);u=np.array([math.cos(a),math.sin(a)])*w/2;v=np.array([-math.sin(a),math.cos(a)])*h/2;c=np.array([x,y]);return Polygon([c-u-v,c+u-v,c+u+v,c-u+v])
def decode(p):
 c=np.asarray(p.exterior.coords)[:4];u=c[1]-c[0];v=c[2]-c[1];mid=c.mean(0)
 return [float(mid[0]),float(mid[1]),float(np.linalg.norm(u)),float(np.linalg.norm(v)),math.degrees(math.atan2(u[1],u[0]))]
def boxmask(p,size,scale):
 pts=np.rint(np.asarray(p.exterior.coords)[:-1]*scale).astype(np.int32)
 lo=np.maximum(pts.min(0)-2,0);hi=np.minimum(pts.max(0)+3,np.asarray(size)*scale)
 x0,y0=lo;x1,y1=hi
 if x1<=x0 or y1<=y0:return None
 m=np.zeros((y1-y0,x1-x0),np.uint8);cv2.fillConvexPoly(m,pts-lo,1)
 return (slice(y0,y1),slice(x0,x1)),m.astype(bool)

def run(args):
 profile=json.loads(args.profile.read_text());objs=json.loads(args.input.read_text());cache=np.load(args.cache);colors=cache['colors'];size=tuple(map(int,cache['size']));bg=int(cache['background'][0]);scale=int(profile['evaluation_scale'])
 unit=float(profile.get('reference_width',512))/size[0]
 objs=[dict(o,rr=[float(v)*unit if k<4 else float(v) for k,v in enumerate(o['rr'])]) for o in objs]
 size=(int(profile.get('reference_width',512)),round(size[1]*unit))
 ps=[polygon(o['rr']) for o in objs];silhouette=unary_union(ps);allowed=prep(silhouette.buffer(1e-8))
 top=np.full((size[1]*scale,size[0]*scale),-1,np.int32)
 for i,p in enumerate(ps):
  bm=boxmask(p,size,scale)
  if bm:sl,m=bm;top[sl][m]=i
 cs=np.array([o['color'] for o in objs]);reference=np.full(top.shape,bg,np.int16);hit=top>=0;reference[hit]=cs[top[hit]]
 if args.reference:
  reference.fill(bg)
  for o in json.loads(args.reference.read_text()):
   pp=polygon([float(v)*unit if k<4 else float(v) for k,v in enumerate(o['rr'])]);bm=boxmask(pp,size,scale)
   if bm:sl,m=bm;reference[sl][m]=o['color']
 protected=np.zeros(top.shape,bool)
 for x0,y0,x1,y1 in profile.get('protected_boxes_normalized',[]):
  protected[round(y0*top.shape[0]):round(y1*top.shape[0]),round(x0*top.shape[1]):round(x1*top.shape[1])]=True
 if args.protection:protected|=np.load(args.protection)
 groups=defaultdict(list)
 for i,o in enumerate(objs):
  if o.get('role')!='vw_undercoat':groups[(0 if profile.get('cross_layer_merges',False) else o['z_rank'],o['color'])].append(i)
 removed=set();merges=[];max_error=profile['max_visible_error_per_merge'];group_fraction=profile['max_group_changed_fraction'];total_bound=0
 for key,ids in sorted(groups.items(),key=lambda item:len(item[1]),reverse=True):
  if len(ids)<2:continue
  ci=key[1];last=max(ids);old_union=unary_union([ps[i] for i in ids]);group_budget=max(profile['min_group_changed_pixels'],old_union.area*scale*scale*group_fraction)
  group_errors=0;heap=[];versions={i:0 for i in ids}
  boundary_distance=cv2.distanceTransform((reference!=ci).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
  def propose(i,j):
   if profile.get('cross_layer_merges',False):i,j=max(i,j),min(i,j)
   if i==j or i in removed or j in removed:return
   a,b=ps[i],ps[j]
   if a.distance(b)>profile['max_pair_distance']:return
   united=a.union(b);p=united.minimum_rotated_rectangle
   if p.area>united.area*profile['max_area_expansion_ratio']:return
   if not allowed.covers(p):return
   sl,m=boxmask(p,size,scale)
   visible=m&(top[sl]<=(i if profile.get('cross_layer_merges',False) else last));wrong=visible&(reference[sl]!=ci)
   if np.any(wrong&protected[sl]):return
   if np.any(wrong & (boundary_distance[sl]>profile.get('max_boundary_shift',1e9)*scale)):return
   delta=np.max(np.abs(colors[reference[sl]].astype(np.int16)-colors[ci].astype(np.int16)),axis=2)/255
   err=float(delta[wrong].sum())/(scale*scale)
   changed=int(wrong.sum())
   if err>max_error or changed>group_budget:return
   heapq.heappush(heap,(err,-united.area,i,j,versions[i],versions[j],decode(p),changed))
  for a,i in enumerate(ids):
   for j in ids[a+1:]:propose(i,j)
  while heap:
   err,area,i,j,vi,vj,rr,changed=heapq.heappop(heap)
   if i in removed or j in removed or versions[i]!=vi or versions[j]!=vj:continue
   if group_errors+changed>group_budget:continue
   ps[i]=polygon(rr);objs[i]=dict(objs[i],rr=rr,role='merged_rect',vector_merge=True);removed.add(j);versions[i]+=1;group_errors+=changed;total_bound+=err
   merges.append({'kept':i,'removed':j,'error':err,'changed_native_pixels_bound':changed})
   for k in ids:
    if k!=i and k not in removed:propose(min(i,k),max(i,k))
  print('group',key,'start',len(ids),'kept',sum(i not in removed for i in ids),'error_bound',group_errors,flush=True)
 final=[o for i,o in enumerate(objs) if i not in removed]
 final=[dict(o,rr=[float(v)/unit if k<4 else float(v) for k,v in enumerate(o['rr'])]) for o in final]
 args.output.write_text(json.dumps(final));args.output.with_suffix('.report.json').write_text(json.dumps({'before':len(objs),'after':len(final),'merges':len(merges),'error_bound_working_pixels':total_bound,'profile':profile},indent=2))
 print('RESULT',len(objs),'->',len(final),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--protection',type=Path);ap.add_argument('--input',type=Path,default=ROOT/'v5_15_final_objects.json');ap.add_argument('--cache',type=Path,default=ROOT/'v5_14_export_cache.npz');ap.add_argument('--reference',type=Path);ap.add_argument('--profile',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);run(ap.parse_args())
