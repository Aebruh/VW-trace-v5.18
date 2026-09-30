#!/usr/bin/env python3
"""Perceptual contour fusion built on v5.17. Vector geometry; no grid packing."""
import argparse,json,math
from pathlib import Path
from collections import Counter
import numpy as np,cv2
from shapely.ops import unary_union
from shapely.geometry import GeometryCollection,box
from shapely_compat import STRtree
from shapely.prepared import prep
from merge_v5_16 import polygon,decode,boxmask

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--profile',type=Path,required=True);ap.add_argument('--protect',type=Path);ap.add_argument('--reference',type=Path);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 cfg=json.loads(args.profile.read_text());objs=json.loads(args.input.read_text());cache=np.load(args.cache);colors=cache['colors'].astype(np.int16);bg=int(cache['background'][0]);native=cache['size'];unit=cfg['reference_width']/native[0];size=(cfg['reference_width'],round(native[1]*unit));s=cfg['evaluation_scale']
 for index,o in enumerate(objs):
  o.setdefault('_v517_source_order',index);o['rr']=[v*unit if k<4 else v for k,v in enumerate(o['rr'])]
 ps=[polygon(o['rr']) for o in objs];sil=unary_union(ps)
 if args.reference:sil=unary_union([polygon([v*unit if k<4 else v for k,v in enumerate(o['rr'])]) for o in json.loads(args.reference.read_text())])
 outer_tolerance=cfg.get('outer_tolerance',0);allowed=prep(sil.buffer(outer_tolerance+1e-8));edge_band=sil.boundary.buffer(outer_tolerance+1e-8);shape=(size[1]*s,size[0]*s)
 def render():
  a=np.full(shape,bg,np.int16)
  for i,o in enumerate(objs):
   if o is None:continue
   b=boxmask(ps[i],size,s)
   if b:sl,m=b;a[sl][m]=o['color']
  return a
 current=render();reference=current.copy()
 if args.reference:
  reference.fill(bg)
  for o in json.loads(args.reference.read_text()):
   q=polygon([v*unit if k<4 else v for k,v in enumerate(o['rr'])]);b=boxmask(q,size,s)
   if b:sl,m=b;reference[sl][m]=o['color']
 protected=np.zeros(shape,np.uint8);context_seeds=np.zeros(shape,np.uint8)
 for ci in np.unique(reference):
  if ci==bg:continue
  mask=(reference==ci).astype(np.uint8);n,lab,stats,_=cv2.connectedComponentsWithStats(mask,8)
  for k in range(1,n):
   area=stats[k,cv2.CC_STAT_AREA]/s**2
   if area<cfg['protect_component_area']:
    protected[lab==k]=1
   if cfg.get('protect_context_min_area',.5)<=area<=cfg.get('protect_context_area',0):
    cm=(lab==k).astype(np.uint8);ring=(cv2.dilate(cm,np.ones((2*s+1,2*s+1),np.uint8))!=0)&(cm==0)&(reference!=bg)
    if ring.any():
     contrast=float(np.percentile(np.max(abs(colors[reference[ring]]-colors[ci]),axis=1),75))
     if contrast>=cfg.get('protect_context_contrast',64):context_seeds[cm!=0]=1
  dist=cv2.distanceTransform(mask,cv2.DIST_L2,5)
  thin=(dist>0)&(dist<=cfg['protect_thin_radius']*s)
  core=cv2.dilate((dist>cfg['protect_thin_radius']*s).astype(np.uint8),np.ones((2*s+1,2*s+1),np.uint8))
  protected[thin&(core==0)]=1
 if cfg.get('protect_context_radius',0)>0:
  radius=round(cfg['protect_context_radius']*s);protected|=cv2.dilate(context_seeds,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))
 auto_protected_count=int(protected.sum());importance=protected.astype(np.float32)*cfg.get('automatic_importance',.65)
 protected[:]=0
 for ci in np.unique(reference):
  if ci==bg:continue
  n,lab,stats,_=cv2.connectedComponentsWithStats((reference==ci).astype(np.uint8),8)
  for k in range(1,n):
   area=stats[k,cv2.CC_STAT_AREA]/s**2
   if cfg.get('salient_min_area',1)<=area<=cfg.get('salient_max_area',12):
    cm=(lab==k).astype(np.uint8);ring=(cv2.dilate(cm,np.ones((2*s+1,2*s+1),np.uint8))!=0)&(cm==0)&(reference!=bg)
    if ring.any() and np.max(abs(colors[reference[ring]]-colors[ci]))>=cfg.get('salient_contrast',96):
     protected[cm!=0]=1;importance[cm!=0]=1
 if args.protect:
  for x0,y0,x1,y1 in json.loads(args.protect.read_text())['boxes']:
   protected[round(y0*shape[0]):round(y1*shape[0]),round(x0*shape[1]):round(x1*shape[1])]=1
   importance[round(y0*shape[0]):round(y1*shape[0]),round(x0*shape[1]):round(x1*shape[1])]=1
 if cfg.get('protection_only',False):
  np.save(args.output.with_suffix('.protection.npy'),protected.astype(bool));return
 protected=protected.astype(bool);distance={int(ci):cv2.distanceTransform((reference!=ci).astype(np.uint8),cv2.DIST_L2,5) for ci in np.unique(reference)}
 accepted=[];seams=[]
 holes=[]
 geoms=list(sil.geoms) if hasattr(sil,'geoms') else [sil]
 from shapely.geometry import Polygon
 for g in geoms:
  for ring in getattr(g,'interiors',[]):
   h=Polygon(ring)
   if h.area<=cfg['seam_max_area']:holes.append(h)
 pools={}
 for h in holes:
  near=[i for i,q in enumerate(ps) if q.distance(h)<.2]
  cs={objs[i]['color'] for i in near}
  if len(cs)==1 and h.area>=cfg.get('seam_min_area',.01):
   pools.setdefault(cs.pop(),[]).append(h)
 shell=unary_union([Polygon(g.exterior) for g in geoms if hasattr(g,'exterior')])
 for ci,hs in pools.items():
  if len(seams)>=cfg.get('max_seam_patches',2):break
  p=unary_union(hs).minimum_rotated_rectangle
  if not shell.buffer(1e-8).covers(p):continue
  if p.difference(sil.union(unary_union(hs))).area>1e-8:continue
  objs.insert(0,{'color':ci,'rr':decode(p),'role':'seam_undercoat','z_rank':-1});ps.insert(0,p)
  seams.append({'area':sum(h.area for h in hs),'color':ci,'holes':len(hs)})
 if seams:sil=unary_union(ps);allowed=prep(sil.buffer(outer_tolerance+1e-8));edge_band=sil.boundary.buffer(outer_tolerance+1e-8);current=render()
 def visible_support():
  cover=GeometryCollection();v=[None]*len(objs)
  for i in range(len(objs)-1,-1,-1):
   if objs[i] is None:continue
   v[i]=ps[i].difference(cover)
   cover=cover.union(ps[i])
  return v
 tree=STRtree(ps)
 def attempt_one(ids,role,anchor="front",parts=1,inset=0,strip=False):
  nonlocal current,tree
  ids=sorted(set(ids));last=ids[-1] if anchor=='front' else ids[0];ci=objs[last]['color']
  supports=[vis[i] for i in ids if not vis[i].is_empty]
  if not supports:return False
  if parts==1:
   candidates=[unary_union(supports).minimum_rotated_rectangle]
  else:
   centers=np.array([[q.centroid.x,q.centroid.y] for q in supports]);_,_,axes=np.linalg.svd(centers-centers.mean(0),full_matrices=False);order=np.argsort(centers@axes[0]);chunks=np.array_split(order,parts)
   if any(len(chunk)==0 for chunk in chunks):return False
   candidates=[unary_union([supports[k] for k in chunk]).minimum_rotated_rectangle for chunk in chunks]
  if strip:
   angles=[];weights=[];thickness=[];points=[]
   for j in ids:
    x,y,w,h,angle=objs[j]['rr'];angles.append(math.radians(angle+(90 if h>w else 0)));weights.append(max(w,h));thickness.append(min(w,h));points.extend(list(ps[j].exterior.coords)[:-1])
   angle=.5*math.atan2(np.dot(weights,np.sin(2*np.array(angles))),np.dot(weights,np.cos(2*np.array(angles))));bucket=cfg.get('angle_bucket_degrees',0)
   if bucket:angle=math.radians(round(math.degrees(angle)/bucket)*bucket)
   u=np.array([math.cos(angle),math.sin(angle)]);v=np.array([-u[1],u[0]]);points=np.array(points);long=points@u;centers=np.array([objs[j]['rr'][:2] for j in ids]);mid=u*(long.min()+long.max())/2+v*np.median(centers@v)
   candidates=[polygon([*mid,long.max()-long.min(),float(np.median(thickness)),math.degrees(angle)])]
  if inset:
   adjusted=[]
   for q in candidates:
    if q.geom_type!='Polygon':return False
    rr=decode(q);axis=2 if rr[2]<rr[3] else 3
    if rr[axis]<=2*inset:return False
    rr[axis]-=2*inset;adjusted.append(polygon(rr))
   candidates=adjusted
  if any(p.geom_type!='Polygon' or p.area<1e-9 or not allowed.covers(p) for p in candidates):return False
  slots=ids[-parts:] if anchor=='front' else ids[:parts]
  replacements=dict(zip(slots,candidates));p=unary_union(candidates)
  old=unary_union([ps[i] for i in ids]);lost=old.difference(p)
  bounds=old.union(p).bounds;x0=max(0,int(bounds[0]*s)-3);y0=max(0,int(bounds[1]*s)-3);x1=min(shape[1],int(bounds[2]*s)+4);y1=min(shape[0],int(bounds[3]*s)+4)
  x0=(x0//s)*s;y0=(y0//s)*s;x1=min(shape[1],((x1+s-1)//s)*s);y1=min(shape[0],((y1+s-1)//s)*s)
  sl=(slice(y0,y1),slice(x0,x1));a=np.full((y1-y0,x1-x0),bg,np.int16)
  near=sorted(set(map(int,tree.query(box(x0/s,y0/s,x1/s,y1/s))))|set(replacements))
  for j in near:
   o=objs[j]
   if o is None or (j in ids and j not in replacements):continue
   q=replacements.get(j,ps[j]);b=q.bounds
   if b[2]*s<x0 or b[0]*s>x1 or b[3]*s<y0 or b[1]*s>y1:continue
   pts=np.rint(np.asarray(q.exterior.coords)[:-1]*s).astype(np.int32)-[x0,y0];cv2.fillConvexPoly(a,pts,int(o['color']))
  valid=(a!=bg)&(reference[sl]!=bg)&(current[sl]!=bg)
  changed=(a!=current[sl])&valid;wrong=(a!=reference[sl])&valid
  if np.any((a!=current[sl])&protected[sl]&(a!=reference[sl])):return False
  for cc in np.unique(a[wrong]):
   if np.any(wrong&(a==cc)&(distance[int(cc)][sl]>(cfg['max_boundary_shift']*(1-importance[sl])+cfg.get('important_boundary_shift',.7)*importance[sl])*s)):return False
  before=np.max(abs(colors[current[sl]]-colors[reference[sl]]),axis=2)/255
  after=np.max(abs(colors[a]-colors[reference[sl]]),axis=2)/255
  extra=float(np.maximum(after-before,0)[valid].sum())/s**2
  def down(labels):
   rgb=colors[labels].astype(np.float32);h,w=labels.shape
   return rgb.reshape(h//s,s,w//s,s,3).mean((1,3))
  ref_small=down(reference[sl]);old_small=down(current[sl]);new_small=down(a)
  delta=np.abs(new_small-ref_small);previous=np.abs(old_small-ref_small)
  weighted=np.maximum(delta.mean(2)-previous.mean(2),0)
  saliency=importance[sl].reshape(a.shape[0]//s,s,a.shape[1]//s,s).max((1,3))
  normal_cost=float((weighted*(1+3*saliency)).sum())/255
  if normal_cost>cfg.get('normal_error_per_candidate',3):return False
  if extra>cfg['max_extra_error'] or float(after[changed].sum())/s**2>cfg['max_candidate_error']:return False
  if lost.area>1e-8:
   rest=[q for j,q in enumerate(ps) if objs[j] is not None and j not in ids and q.intersects(lost)]
   remaining=lost.difference(unary_union(rest)) if rest else lost
   if remaining.difference(edge_band).area>1e-8:return False
  for j in ids:
   if j not in replacements:objs[j]=None
  for j,q in replacements.items():objs[j]=dict(objs[j],rr=decode(q),role=role);ps[j]=q
  current[sl]=a;tree=STRtree(ps)
  accepted.append({'removed':len(ids)-parts,'role':role,'replacement_rectangles':parts,'extra_error':extra,'normal_error':normal_cost,'centerline':strip});return True
 def attempt(ids,role,anchor='front',parts=1):
  for inset in cfg.get('candidate_insets',[0]):
   if attempt_one(ids,role,anchor,parts,inset):return True
  if role=='contour_fusion' and parts==1 and cfg.get('centerline_fit',False):
   if attempt_one(ids,role,anchor,1,0,True):return True
  return False
 for iteration in range(cfg['max_passes']):
  vis=visible_support();start=sum(o is not None for o in objs);print('PASS',iteration,'objects',start,flush=True)
  groups={ci:[i for i,o in enumerate(objs) if o and o['color']==ci] for ci in sorted({o['color'] for o in objs if o})}
  for ci,indices in sorted(groups.items(),key=lambda kv:-len(kv[1])):
   union=unary_union([vis[i] for i in indices if not vis[i].is_empty])
   components=list(union.geoms) if hasattr(union,'geoms') else [union]
   for component in sorted(components,key=lambda q:-q.area):
    ids=[i for i in indices if objs[i] and not vis[i].is_empty and vis[i].intersection(component).area>1e-8]
    if len(ids)>2:
     if attempt(ids,'region_replacement','back') or attempt(ids,'region_replacement','front') or (len(ids)>4 and (attempt(ids,'region_replacement','back',2) or attempt(ids,'region_replacement','front',2))):vis=visible_support()
   for i in indices:
    if objs[i] is None or vis[i].is_empty:continue
    neighbors=sorted([j for j in indices if j!=i and objs[j] and not vis[j].is_empty and vis[i].distance(vis[j])<cfg['region_radius']],key=lambda j:vis[i].distance(vis[j]))[:cfg['max_neighbors']]
    def axis(j):
     _,_,w,h,a=objs[j]['rr'];return (a+(90 if h>w else 0))%180,min(w,h)
    ai,ti=axis(i)
    chain=[j for j in neighbors if min(abs(axis(j)[0]-ai),180-abs(axis(j)[0]-ai))<=cfg.get('chain_angle_degrees',15) and max(ti,axis(j)[1])<=cfg.get('chain_thickness_ratio',4)*max(.01,min(ti,axis(j)[1]))]
    fused=False
    for n in (8,4,2,1):
     ids=[i]+chain[:n]
     if len(ids)>1 and (attempt(ids,'contour_fusion','back') or attempt(ids,'contour_fusion','front')):
      fused=True;break
    if fused:continue
    for n in (12,8,5,3,1):
     ids=[i]+neighbors[:n]
     if len(ids)<2 or any(objs[j] is None for j in ids):continue
     if attempt(ids,'region_replacement' if len(ids)>2 else 'contour_fusion','back') or attempt(ids,'region_replacement' if len(ids)>2 else 'contour_fusion','front') or (len(ids)>4 and (attempt(ids,'region_replacement','back',2) or attempt(ids,'region_replacement','front',2))):
      break
   print('color',ci,'remaining',sum(o is not None for o in objs),flush=True)
   vis=visible_support()
  vis=visible_support()
  for i,o in enumerate(objs):
   if o and vis[i].area<1e-9:objs[i]=None;accepted.append({'removed':1,'role':'hidden_prune','extra_error':0})
  end=sum(o is not None for o in objs)
  args.output.write_text(json.dumps([dict(o,rr=[v/unit if k<4 else v for k,v in enumerate(o['rr'])]) for o in objs if o]))
  if end==start:break
 report={'before':len(reference.nonzero()[0]) if False else len(json.loads(args.input.read_text())),'after':end,'operations':accepted,'seam_patches':seams,'enclosed_small_holes_audited':len(holes),'automatic_protected_pixels':auto_protected_count,'combined_protected_pixels':int(protected.sum()),'roles':dict(Counter(o['role'] for o in objs if o)),'profile':cfg}
 args.output.with_suffix('.operations.json').write_text(json.dumps(report,indent=2));np.save(args.output.with_suffix('.protection.npy'),protected);np.save(args.output.with_suffix('.importance.npy'),importance);print('DONE',end,flush=True)
if __name__=='__main__':main()
