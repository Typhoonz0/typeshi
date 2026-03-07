"""
FPS Multiplayer Client
  pip install pygame PyOpenGL PyOpenGL_accelerate pygltflib numpy websocket-client

  python client.py [--host 127.0.0.1] [--port 7777] [--name YourName]

Controls:
  WASD        Move
  Mouse       Look  (click to capture)
  Space       Jump  (tap = low, hold = high)
  Ctrl        Slide
  LMB         Shoot
  R           Reload
  ESC         Release mouse / quit
"""

import pygame, sys, math, random, os, ctypes, base64, io, time, json, threading
import numpy as np
import argparse

try:
    from OpenGL.GL  import *
    from OpenGL.GLU import *
except ImportError:
    print("Install: pip install pygame PyOpenGL PyOpenGL_accelerate"); sys.exit(1)

try:
    import pygltflib
    _HAS_GLTF = True
except ImportError:
    _HAS_GLTF = False

try:
    import websocket   # websocket-client (pip install websocket-client)
    _HAS_WS = True
except ImportError:
    print("Install: pip install websocket-client"); sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
W, H  = 1280, 720
FPS   = 120

GRAVITY         = 0.0028
JUMP_VEL        = 0.10
JUMP_CUT        = 0.75
JUMP_BUFFER     = 8
COYOTE_FRAMES   = 8
STEP_UP         = 0.30
STEP_DOWN       = 0.30
GROUND_ACCEL    = 0.014
GROUND_FRICTION = 0.78
MAX_GROUND_SPD  = 0.08
AIR_ACCEL       = 0.004
MAX_AIR_WISH    = 0.08
SLIDE_VEL       = 0.22
SLIDE_DUR       = 18
SLIDE_CD        = 36

GUN_MAG        = 12
GUN_FIRE_CD    = 7
GUN_RELOAD_DUR = 90

GUN_POS         = (0.38, -0.19, -0.45)
GUN_SCALE       = 0.08
RECOIL_PITCH    = 8.0
RECOIL_RECOVER  = 0.10
RELOAD_KICK_X   = 0.08
RELOAD_KICK_ROT = 35.0

MOUSE_SENS     = 0.10
BASE_FOV       = 100.0
SPEED_FOV_ADD  = 14.0
FOV_LERP       = 0.2222

MAP_SCALE = 2.0
ARENA_W = 48.0
ARENA_D = 48.0

SKY_TOP  = (0.03, 0.02, 0.10)
SKY_BOT  = (0.08, 0.04, 0.20)
FLOOR_COL= (0.14, 0.12, 0.22)
PLAT_COL = (0.20, 0.18, 0.35)
WALL_COL = (0.10, 0.08, 0.20)
GRID_COL = (0.22, 0.18, 0.38)

P_H = 1.8
P_R = 0.38
EYE = 1.62

# ══════════════════════════════════════════════════════════════════════════════
# MATH
# ══════════════════════════════════════════════════════════════════════════════
def v3(x,y,z):      return [float(x),float(y),float(z)]
def vadd(a,b):      return [a[0]+b[0],a[1]+b[1],a[2]+b[2]]
def clamp(v,lo,hi): return max(lo,min(hi,v))
def lerp(a,b,t):    return a+(b-a)*t

# ══════════════════════════════════════════════════════════════════════════════
# SPATIAL GRID + COLLISION  (client needs it for local prediction)
# ══════════════════════════════════════════════════════════════════════════════
class TriGrid:
    def __init__(self, cell=3.0):
        self.cell=cell; self.buckets={}; self.tri_arr=None
    def build(self, triangles):
        self.buckets.clear()
        if not triangles: self.tri_arr=np.zeros((0,3,3),np.float32); return
        self.tri_arr=np.array(triangles,dtype=np.float32)
        cell=self.cell
        for idx in range(len(self.tri_arr)):
            xs=self.tri_arr[idx,:,0]; zs=self.tri_arr[idx,:,2]
            x0=int(math.floor(xs.min()/cell)); x1=int(math.floor(xs.max()/cell))
            z0=int(math.floor(zs.min()/cell)); z1=int(math.floor(zs.max()/cell))
            for gx in range(x0,x1+1):
                for gz in range(z0,z1+1): self.buckets.setdefault((gx,gz),[]).append(idx)
        self._np={k:np.array(v,dtype=np.int32) for k,v in self.buckets.items()}
    def query_indices(self, px, pz, radius):
        r=int(math.ceil(radius/self.cell))+1; gx=int(math.floor(px/self.cell)); gz=int(math.floor(pz/self.cell))
        parts=[]
        for dx in range(-r,r+1):
            for dz in range(-r,r+1):
                arr=self._np.get((gx+dx,gz+dz))
                if arr is not None: parts.append(arr)
        if not parts: return np.empty(0,dtype=np.int32)
        return np.unique(np.concatenate(parts))

TRI_GRID = TriGrid()

def _point_in_tri_batch(pts,a,b,c):
    v0=c-a;v1=b-a;v2=pts-a; d00=(v0*v0).sum(1);d01=(v0*v1).sum(1);d02=(v0*v2).sum(1);d11=(v1*v1).sum(1);d12=(v1*v2).sum(1)
    denom=d00*d11-d01*d01; safe=np.abs(denom)>1e-12; inv=np.where(safe,1.0/np.where(safe,denom,1.0),0.0)
    u=(d11*d02-d01*d12)*inv; v=(d00*d12-d01*d02)*inv; return (u>=-0.01)&(v>=-0.01)&(u+v<=1.02)

def _closest_pts_tris_batch(p,a,b,c):
    ab=b-a;ac=c-a;ap=p-a; d1=(ab*ap).sum(1);d2=(ac*ap).sum(1); bp=p-b;d3=(ab*bp).sum(1);d4=(ac*bp).sum(1)
    cp_=p-c;d5=(ab*cp_).sum(1);d6=(ac*cp_).sum(1); vc=d1*d4-d3*d2;vb=d5*d2-d1*d6;va=d3*d6-d5*d4
    denom=va+vb+vc; safe=np.abs(denom)>1e-12; v=np.where(safe,vb/np.where(safe,denom,1.0),0.0); w=np.where(safe,vc/np.where(safe,denom,1.0),0.0)
    result=a+v[:,None]*ab+w[:,None]*ac
    m=(d1<=0)&(d2<=0);result[m]=a[m]; m=(d3>=0)&(d4<=d3);result[m]=b[m]; m=(d6>=0)&(d5<=d6);result[m]=c[m]
    m=(vc<=0)&(d1>=0)&(d3<=0);tv=np.where(d1-d3>1e-12,d1/(d1-d3+1e-12),0.0);result[m]=(a+tv[:,None]*ab)[m]
    m=(vb<=0)&(d2>=0)&(d6<=0);tw=np.where(d2-d6>1e-12,d2/(d2-d6+1e-12),0.0);result[m]=(a+tw[:,None]*ac)[m]
    m=(va<=0)&((d4-d3)>=0)&((d5-d6)>=0); denom2=(d4-d3)+(d5-d6); tw2=np.where(np.abs(denom2)>1e-12,(d4-d3)/np.where(np.abs(denom2)>1e-12,denom2,1.0),0.0); result[m]=(b+tw2[:,None]*(c-b))[m]
    return result

def _collide_capsule_fast(prev_pos, new_pos, radius, height, tri_arr, vy=0.0):
    rx,ry,rz=float(new_pos[0]),float(new_pos[1]),float(new_pos[2]); on_ground=False; on_wall=False; hit_ceiling=False
    if len(tri_arr)==0: return on_ground,on_wall,[rx,ry,rz],hit_ceiling
    p0=tri_arr[:,0,:];p1=tri_arr[:,1,:];p2=tri_arr[:,2,:]; e1=p1-p0;e2=p2-p0; n=np.cross(e1,e2); nl=np.linalg.norm(n,axis=1,keepdims=True); valid=(nl[:,0]>1e-9); n[valid]/=nl[valid]; ny=n[:,1]
    if vy<=0:
        floor_mask=valid&(ny>0.4)
        if floor_mask.any():
            fp0=p0[floor_mask];fp1=p1[floor_mask];fp2=p2[floor_mask]; foot=np.array([[rx,ry,rz]],np.float32); cp=_closest_pts_tris_batch(foot,fp0,fp1,fp2)
            dxz=np.sqrt((rx-cp[:,0])**2+(rz-cp[:,2])**2); hit=(dxz<=radius+0.20)&(cp[:,1]<=ry+0.12)&(cp[:,1]>=ry-0.22)
            if hit.any():
                best_y=float(cp[hit,1].max())
                if best_y<=ry+0.12: ry=best_y; on_ground=True
    ceil_mask=valid&(ny<-0.35)
    if ceil_mask.any():
        cp0=p0[ceil_mask];cp1=p1[ceil_mask];cp2=p2[ceil_mask]
        for ct in (1.0,0.85,0.70):
            sy=ry+height*ct; top=np.array([[rx,sy,rz]],np.float32); cc=_closest_pts_tris_batch(top,cp0,cp1,cp2)
            dxz_c=np.sqrt((rx-cc[:,0])**2+(rz-cc[:,2])**2); chit=(dxz_c<=radius+0.15)&(cc[:,1]>=sy-0.08)&(cc[:,1]<=sy+radius+0.15)
            if chit.any():
                lowest=float(cc[chit,1].min()); new_ry=lowest-height*ct-0.02
                if new_ry<ry: ry=new_ry; hit_ceiling=True
    wall_mask=valid&(np.abs(ny)<0.5)
    if wall_mask.any():
        wp0=p0[wall_mask];wp1=p1[wall_mask];wp2=p2[wall_mask]; wn=n[wall_mask]; w_top=np.maximum(np.maximum(wp0[:,1],wp1[:,1]),wp2[:,1]); MIN_PEN=0.06
        for _ in range(6):
            any_pen=False
            for t in (0.35,0.75):
                sy=ry+height*t; pt=np.array([[rx,sy,rz]],np.float32); cp=_closest_pts_tris_batch(pt,wp0,wp1,wp2)
                dx3=rx-cp[:,0];dz3=rz-cp[:,2];dy3=sy-cp[:,1]; dist3=np.sqrt(dx3*dx3+dy3*dy3+dz3*dz3); pen=dist3<radius
                if not pen.any(): continue
                depths=np.where(pen,radius-dist3,-1.0); i=int(np.argmax(depths)); d=float(dist3[i])
                if (radius-d)<MIN_PEN: continue
                face_rise=w_top[i]-ry
                if 0.0<=face_rise<=STEP_UP and vy<=0.001: ry=float(w_top[i]); on_ground=True; any_pen=True; continue
                ox=float(dx3[i]);oz=float(dz3[i]); dxz=math.sqrt(ox*ox+oz*oz)
                if dxz<1e-6:
                    fnx=float(wn[i,0]);fnz=float(wn[i,2]);fnl=math.sqrt(fnx*fnx+fnz*fnz)
                    if fnl>1e-6: rx+=fnx/fnl*(radius-d+0.005); rz+=fnz/fnl*(radius-d+0.005)
                else:
                    ov=radius-d+0.005; rx+=(ox/dxz)*ov; rz+=(oz/dxz)*ov
                on_wall=True; any_pen=True
            if not any_pen: break
    for t in (0.35,0.75):
        sy=ry+height*t; pt=np.array([[rx,sy,rz]],np.float32); cp_all=_closest_pts_tris_batch(pt,p0,p1,p2)
        dist_all=np.sqrt(((pt-cp_all)**2).sum(axis=1)); toward=((pt-cp_all)*n).sum(axis=1)
        deeply_inside=valid&(dist_all<radius*0.45)&(toward<0)
        if deeply_inside.any(): rx=float(prev_pos[0]); rz=float(prev_pos[2]); on_wall=True; break
    return on_ground,on_wall,[rx,ry,rz],hit_ceiling

def _raycast_down(rx,ry,rz,max_dist):
    if TRI_GRID.tri_arr is None or len(TRI_GRID.tri_arr)==0: return None
    idx=TRI_GRID.query_indices(rx,rz,P_R+1.0)
    if len(idx)==0: return None
    tris=TRI_GRID.tri_arr[idx]; p0=tris[:,0,:];p1=tris[:,1,:];p2=tris[:,2,:]
    e1=p1-p0;e2=p2-p0; n=np.cross(e1,e2); nl=np.linalg.norm(n,axis=1,keepdims=True); valid=nl[:,0]>1e-9; n[valid]/=nl[valid]; floor_mask=valid&(n[:,1]>0.3)
    if not floor_mask.any(): return None
    fp0=p0[floor_mask];fp1=p1[floor_mask];fp2=p2[floor_mask];fn=n[floor_mask]; denom=fn[:,1]; valid2=denom>0.01
    if not valid2.any(): return None
    op=fp0[valid2]-np.array([[rx,ry,rz]],np.float32); t_hit=(op*fn[valid2]).sum(axis=1)/denom[valid2]; in_range=(t_hit>=-0.1)&(t_hit<=max_dist)
    if not in_range.any(): return None
    t_vals=t_hit[in_range]; hit_pts=np.stack([np.full(len(t_vals),rx),ry-t_vals,np.full(len(t_vals),rz)],axis=1).astype(np.float32)
    v2idx=np.where(valid2)[0][in_range]; a=fp0[v2idx];b=fp1[v2idx];c=fp2[v2idx]; inside=_point_in_tri_batch(hit_pts,a,b,c)
    if not inside.any(): return None
    return float((ry-t_vals[inside]).max())

def _raycast_hitscan(ox,oy,oz,dx,dy,dz,max_dist=400.0):
    if TRI_GRID.tri_arr is None or len(TRI_GRID.tri_arr)==0: return None
    d_len=math.sqrt(dx*dx+dy*dy+dz*dz)
    if d_len<1e-9: return None
    dx/=d_len; dy/=d_len; dz/=d_len
    steps=[0.0,max_dist*0.25,max_dist*0.5,max_dist*0.75,max_dist]; idx_set=set()
    for s in steps:
        px=ox+dx*s; pz=oz+dz*s
        for i in TRI_GRID.query_indices(px,pz,6.0): idx_set.add(int(i))
    if not idx_set: return None
    idx=np.array(sorted(idx_set),dtype=np.int32); tris=TRI_GRID.tri_arr[idx]
    p0=tris[:,0,:];p1=tris[:,1,:];p2=tris[:,2,:]; e1=p1-p0;e2=p2-p0; n=np.cross(e1,e2); nl=np.linalg.norm(n,axis=1,keepdims=True); valid=nl[:,0]>1e-9; n[valid]/=nl[valid]
    rd=np.array([[dx,dy,dz]],np.float32); denom=(n*rd).sum(axis=1); front=valid&(denom<-1e-4)
    if not front.any(): return None
    ro=np.array([[ox,oy,oz]],np.float32); t_vals=((p0[front]-ro)*n[front]).sum(axis=1)/denom[front]
    in_range=(t_vals>0.05)&(t_vals<=max_dist)
    if not in_range.any(): return None
    t_sub=t_vals[in_range]; hit_pts=ro+t_sub[:,None]*rd
    fp0=p0[front][in_range];fp1=p1[front][in_range];fp2=p2[front][in_range]; inside=_point_in_tri_batch(hit_pts,fp0,fp1,fp2)
    if not inside.any(): return None
    best_t=float(t_sub[inside].min()); hx=ox+dx*best_t; hy=oy+dy*best_t; hz=oz+dz*best_t
    return (hx,hy,hz,best_t)

def collide_capsule_map(prev_pos, new_pos, radius, height, vy):
    if TRI_GRID.tri_arr is None or len(TRI_GRID.tri_arr)==0: return False,False,list(new_pos),False
    idx=TRI_GRID.query_indices(new_pos[0],new_pos[2],radius+1.5)
    if len(idx)==0: return False,False,list(new_pos),False
    return _collide_capsule_fast(prev_pos,new_pos,radius,height,TRI_GRID.tri_arr[idx],vy)

# ══════════════════════════════════════════════════════════════════════════════
# GLTF LOADER  (rendering copy from original)
# ══════════════════════════════════════════════════════════════════════════════
COMP_DTYPE={5120:np.int8,5121:np.uint8,5122:np.int16,5123:np.uint16,5125:np.uint32,5126:np.float32}
TYPE_NCOMP={"SCALAR":1,"VEC2":2,"VEC3":3,"VEC4":4,"MAT2":4,"MAT3":9,"MAT4":16}

class GLTFMap:
    def __init__(self):
        self._cpu=[]; self.triangles=[]; self.aabb_min=None; self.aabb_max=None
        self.parsed=False; self.parse_error=""; self.draw_calls=[]; self.uploaded=False

    def _get_buf(self,gltf,bi,path):
        buf=gltf.buffers[bi]
        if buf.uri is None: return bytes(gltf.binary_blob())
        if buf.uri.startswith("data:"): _,enc=buf.uri.split(",",1); return base64.b64decode(enc)
        with open(os.path.join(os.path.dirname(os.path.abspath(path)),buf.uri),"rb") as f: return f.read()

    def _read_acc(self,gltf,ai,path):
        acc=gltf.accessors[ai]; bvi=getattr(acc,"bufferViewIndex",None) or getattr(acc,"bufferView",None)
        if bvi is None: return np.zeros((acc.count,TYPE_NCOMP[acc.type]),dtype=np.float32)
        bv=gltf.bufferViews[bvi]; raw=self._get_buf(gltf,bv.buffer,path)
        dt=COMP_DTYPE[acc.componentType]; nc=TYPE_NCOMP[acc.type]; bo=bv.byteOffset or 0; ao=acc.byteOffset or 0
        sz=np.dtype(dt).itemsize*nc; st=bv.byteStride
        if st is None or st==sz: return np.frombuffer(raw[bo+ao:bo+ao+acc.count*sz],dtype=dt).reshape(acc.count,nc).astype(np.float32)
        out=np.zeros((acc.count,nc),dtype=np.float32)
        for i in range(acc.count):
            off=bo+ao+i*st; out[i]=np.frombuffer(raw[off:off+sz],dtype=dt).astype(np.float32)
        return out

    @staticmethod
    def _node_mat(node):
        if node.matrix is not None: return np.array(node.matrix,dtype=np.float32).reshape(4,4).T
        M=np.eye(4,dtype=np.float32)
        if node.scale: sx,sy,sz=node.scale; M[0,0]=sx; M[1,1]=sy; M[2,2]=sz
        if node.rotation:
            qx,qy,qz,qw=[float(v) for v in node.rotation]
            R=np.array([[1-2*(qy*qy+qz*qz),2*(qx*qy-qz*qw),2*(qx*qz+qy*qw),0],[2*(qx*qy+qz*qw),1-2*(qx*qx+qz*qz),2*(qy*qz-qx*qw),0],[2*(qx*qz-qy*qw),2*(qy*qz+qx*qw),1-2*(qx*qx+qy*qy),0],[0,0,0,1]],dtype=np.float32); M=R@M
        if node.translation:
            T=np.eye(4,dtype=np.float32); T[0,3],T[1,3],T[2,3]=[float(v) for v in node.translation]; M=T@M
        return M

    def _load_tex(self,gltf,ti,path):
        if not gltf.textures or ti>=len(gltf.textures): return None
        tex=gltf.textures[ti]; si=tex.source
        if si is None or not gltf.images or si>=len(gltf.images): return None
        src=gltf.images[si]; base=os.path.dirname(os.path.abspath(path))
        try:
            if src.uri:
                if src.uri.startswith("data:"): _,enc=src.uri.split(",",1); img=base64.b64decode(enc)
                else:
                    with open(os.path.join(base,src.uri),"rb") as f: img=f.read()
                surf=pygame.image.load(io.BytesIO(img))
            else:
                bvi=getattr(src,"bufferView",None)
                if bvi is None: return None
                bv=gltf.bufferViews[bvi]; raw=self._get_buf(gltf,bv.buffer,path)
                surf=pygame.image.load(io.BytesIO(raw[bv.byteOffset:bv.byteOffset+bv.byteLength]))
            surf=pygame.transform.flip(surf,False,True).convert_alpha()
            data=pygame.image.tostring(surf,"RGBA",False)
            tid=int(glGenTextures(1)); glBindTexture(GL_TEXTURE_2D,tid)
            glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,surf.get_width(),surf.get_height(),0,GL_RGBA,GL_UNSIGNED_BYTE,data)
            glGenerateMipmap(GL_TEXTURE_2D)
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_REPEAT)
            glBindTexture(GL_TEXTURE_2D,0); return tid
        except Exception as e: print(f"[TEX] {e}"); return None

    def parse(self, path, skip_tg=False):
        self.parsed=False; self._cpu=[]; self.triangles=[]
        if not _HAS_GLTF: self.parse_error="no pygltflib"; return False
        if not os.path.exists(path): self.parse_error=f"'{path}' not found"; return False
        try: gltf=pygltflib.GLTF2().load(path)
        except Exception as e: self.parse_error=str(e); return False
        si=gltf.scene if gltf.scene is not None else 0
        roots=(gltf.scenes[si].nodes or []) if gltf.scenes else list(range(len(gltf.nodes)))
        stack=[(ni,np.eye(4,dtype=np.float32)) for ni in roots]; all_pts=[]
        while stack:
            ni,pm=stack.pop(); node=gltf.nodes[ni]; mat=pm@self._node_mat(node)
            for ci in (node.children or []): stack.append((ci,mat))
            if node.mesh is None: continue
            for prim in gltf.meshes[node.mesh].primitives:
                r=self._parse_prim(gltf,prim,mat,path)
                if r:
                    vd,id_,col,pts,ti=r
                    self._cpu.append({"vdata":vd,"idata":id_,"col":col,"tex_idx":ti,"gltf":gltf,"path":path})
                    for tri in pts[id_.reshape(-1,3)]: self.triangles.append(tri.tolist())
                    all_pts.extend(pts.tolist())
        if not all_pts: self.parse_error="No geometry"; return False
        av=np.array(all_pts,np.float32); self.aabb_min=av.min(axis=0).tolist(); self.aabb_max=av.max(axis=0).tolist()
        if not skip_tg: TRI_GRID.build(self.triangles)
        self.parsed=True; return True

    def _parse_prim(self,gltf,prim,wm,path):
        if prim.attributes.POSITION is None: return None
        pl=self._read_acc(gltf,prim.attributes.POSITION,path)
        ones=np.ones((len(pl),1),np.float32); pw=(wm@np.hstack([pl.astype(np.float32),ones]).T).T[:,:3]*MAP_SCALE
        if prim.attributes.NORMAL is not None:
            nl=self._read_acc(gltf,prim.attributes.NORMAL,path)
            try: nm=np.linalg.inv(wm[:3,:3]).T; nw=(nm@nl.T).T
            except np.linalg.LinAlgError: nw=nl
            nw=(nw/np.maximum(np.linalg.norm(nw,axis=1,keepdims=True),1e-9)).astype(np.float32)
        else: nw=np.tile([0.0,1.0,0.0],(len(pw),1)).astype(np.float32)
        uvs=self._read_acc(gltf,prim.attributes.TEXCOORD_0,path) if prim.attributes.TEXCOORD_0 is not None else np.zeros((len(pw),2),dtype=np.float32)
        if prim.indices is not None: idx=self._read_acc(gltf,prim.indices,path).flatten().astype(np.uint32)
        else: idx=np.arange(len(pw),dtype=np.uint32)
        mode=prim.mode if prim.mode is not None else 4
        if mode==5:
            idx2=[]
            for i in range(len(idx)-2):
                if i%2==0: idx2.extend([idx[i],idx[i+1],idx[i+2]])
                else: idx2.extend([idx[i+1],idx[i],idx[i+2]])
            idx=np.array(idx2,dtype=np.uint32)
        elif mode==6:
            idx2=[]
            for i in range(1,len(idx)-1): idx2.extend([idx[0],idx[i],idx[i+1]])
            idx=np.array(idx2,dtype=np.uint32)
        elif mode!=4: return None
        col=(0.55,0.50,0.70); ti=None
        if prim.material is not None and gltf.materials:
            try:
                m=gltf.materials[prim.material]; pbr=m.pbrMetallicRoughness
                if pbr:
                    if pbr.baseColorFactor: bf=pbr.baseColorFactor; col=(float(bf[0]),float(bf[1]),float(bf[2]))
                    if pbr.baseColorTexture and gltf.textures: ti=pbr.baseColorTexture.index
            except: pass
        return np.hstack([pw,nw,uvs]).astype(np.float32),idx,col,pw,ti

    def upload(self):
        if not self.parsed: return False
        self.draw_calls=[]; n=0
        for m in self._cpu:
            try:
                vf=m["vdata"].flatten().astype(np.float32); if_=m["idata"].flatten().astype(np.uint32)
                vbo=int(glGenBuffers(1)); glBindBuffer(GL_ARRAY_BUFFER,vbo); glBufferData(GL_ARRAY_BUFFER,vf.nbytes,vf.tobytes(),GL_STATIC_DRAW); glBindBuffer(GL_ARRAY_BUFFER,0)
                ebo=int(glGenBuffers(1)); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,ebo); glBufferData(GL_ELEMENT_ARRAY_BUFFER,if_.nbytes,if_.tobytes(),GL_STATIC_DRAW); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,0)
                tid=self._load_tex(m["gltf"],m["tex_idx"],m["path"]) if m["tex_idx"] is not None else None
                self.draw_calls.append((vbo,ebo,int(len(if_)),m["col"],tid)); n+=1
            except Exception as e: print(f"[VBO] {e}")
        self.uploaded=n>0; return self.uploaded

    def draw(self):
        if not self.draw_calls: return
        stride=8*4; glEnableClientState(GL_VERTEX_ARRAY); glEnableClientState(GL_NORMAL_ARRAY); glEnableClientState(GL_TEXTURE_COORD_ARRAY)
        for vbo,ebo,count,col,tid in self.draw_calls:
            if tid is not None: glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D,tid); glColor4f(1,1,1,1)
            else: glDisable(GL_TEXTURE_2D); glColor4f(col[0],col[1],col[2],1.0)
            glBindBuffer(GL_ARRAY_BUFFER,vbo); glVertexPointer(3,GL_FLOAT,stride,ctypes.c_void_p(0)); glNormalPointer(GL_FLOAT,stride,ctypes.c_void_p(12)); glTexCoordPointer(2,GL_FLOAT,stride,ctypes.c_void_p(24))
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,ebo); glDrawElements(GL_TRIANGLES,count,GL_UNSIGNED_INT,None)
        glBindTexture(GL_TEXTURE_2D,0); glDisable(GL_TEXTURE_2D); glBindBuffer(GL_ARRAY_BUFFER,0); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,0)
        glDisableClientState(GL_VERTEX_ARRAY); glDisableClientState(GL_NORMAL_ARRAY); glDisableClientState(GL_TEXTURE_COORD_ARRAY)

GLTF_MAP = GLTFMap()

# ══════════════════════════════════════════════════════════════════════════════
# NETWORK  (websocket-client runs in background thread; sends via queue)
# ══════════════════════════════════════════════════════════════════════════════
import queue as _queue

class NetClient:
    def __init__(self, host, port, secure=False):
        if secure:
            self.url = f"wss://{host}"
        else:
            self.url = f"ws://{host}:{port}"
        self.ws        = None
        self.connected = False
        self.our_pid   = None
        self._lock     = threading.Lock()
        self._inbox    = []
        self._outbox   = _queue.Queue(maxsize=256)  # non-blocking send queue
        self._thread   = None
        self._send_thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        def on_open(ws):
            self.connected = True
            print(f"[NET] Connected to {self.url}")
            # start dedicated send thread now that socket is open
            self._send_thread = threading.Thread(target=self._send_loop, daemon=True)
            self._send_thread.start()

        def on_message(ws, msg):
            try:
                data = json.loads(msg)
                with self._lock:
                    self._inbox.append(data)
            except: pass

        def on_close(ws, code, reason):
            self.connected = False
            print(f"[NET] Disconnected ({code})")

        def on_error(ws, err):
            print(f"[NET] Error: {err}")

        self.ws = websocket.WebSocketApp(
            self.url, on_open=on_open, on_message=on_message,
            on_close=on_close, on_error=on_error)
        self.ws.run_forever(reconnect=5)

    def _send_loop(self):
        """Dedicated thread drains the outbox without holding the GIL."""
        while True:
            try:
                data = self._outbox.get_nowait()
                if self.ws and self.connected:
                    try: self.ws.send(data)
                    except: pass
            except _queue.Empty:
                time.sleep(0.001)  # yield without holding GIL

    def poll(self):
        with self._lock:
            msgs = self._inbox
            self._inbox = []   # swap with empty list — main thread gets the old one
        return msgs

    def send(self, obj):
        if not self.connected: return
        data = json.dumps(obj)
        try:
            self._outbox.put_nowait(data)   # never blocks; drops if full (old inputs)
        except _queue.Full:
            pass

# ══════════════════════════════════════════════════════════════════════════════
# OTHER PLAYERS (ghost rendering)
# ══════════════════════════════════════════════════════════════════════════════
class RemotePlayer:
    def __init__(self, pid, name):
        self.pid    = pid
        self.name   = name
        self.pos    = [0.0, 0.0, 0.0]
        self.yaw    = 0.0
        self.pitch  = 0.0
        self.health = 100
        self.alive  = True
        self.sliding= False
        # smooth interpolation targets
        self._tgt_pos = None
        self._tgt_yaw = 0.0

    def update_from_snap(self, snap):
        self._tgt_pos = [snap["x"], snap["y"], snap["z"]]
        self._tgt_yaw = snap["yaw"]
        self.pitch    = snap["pitch"]
        self.health   = snap["health"]
        self.alive    = snap["alive"]
        self.sliding  = snap.get("sliding", False)
        self.name     = snap.get("name", self.name)

    def smooth(self):
        # Time-based lerp: reach ~95% of target in 80ms regardless of framerate
        alpha = 1.0 - math.exp(-120.0 * (1.0/120.0) * 0.10)  # ~0.095 per frame at 120fps
        if self._tgt_pos:
            self.pos[0] = lerp(self.pos[0], self._tgt_pos[0], alpha)
            self.pos[1] = lerp(self.pos[1], self._tgt_pos[1], alpha)
            self.pos[2] = lerp(self.pos[2], self._tgt_pos[2], alpha)
        self.yaw = lerp(self.yaw, self._tgt_yaw, alpha)

    def draw(self):
        if not self.alive: return
        # Draw a simple capsule-ish box for the other player
        cx, cy, cz = self.pos
        yr = math.radians(self.yaw)

        glPushMatrix()
        glTranslatef(cx, cy, cz)
        glRotatef(-self.yaw, 0, 1, 0)

        # Body
        glColor4f(0.3, 0.7, 1.0, 0.9)
        _draw_box_immediate(0, P_H*0.5, 0, P_R*1.8, P_H, P_R*1.8)

        # Head
        glColor4f(0.9, 0.75, 0.6, 0.9)
        _draw_box_immediate(0, P_H + 0.2, 0, 0.3, 0.3, 0.3)

        glPopMatrix()

        # Name tag drawn via Billboard (face camera)
        # (omitted for brevity — add if desired)

def _draw_box_immediate(cx,cy,cz,bw,bh,bd):
    hw,hh,hd=bw/2,bh/2,bd/2
    glPushMatrix(); glTranslatef(cx,cy,cz); glBegin(GL_QUADS)
    for (nx,ny,nz),verts in [
        ((0,0,1),((-hw,-hh,hd),(hw,-hh,hd),(hw,hh,hd),(-hw,hh,hd))),
        ((0,0,-1),((hw,-hh,-hd),(-hw,-hh,-hd),(-hw,hh,-hd),(hw,hh,-hd))),
        ((-1,0,0),((-hw,-hh,-hd),(-hw,-hh,hd),(-hw,hh,hd),(-hw,hh,-hd))),
        ((1,0,0),((hw,-hh,hd),(hw,-hh,-hd),(hw,hh,-hd),(hw,hh,hd))),
        ((0,1,0),((-hw,hh,hd),(hw,hh,hd),(hw,hh,-hd),(-hw,hh,-hd))),
        ((0,-1,0),((-hw,-hh,-hd),(hw,-hh,-hd),(hw,-hh,hd),(-hw,-hh,hd))),
    ]:
        glNormal3f(nx,ny,nz)
        for v in verts: glVertex3f(*v)
    glEnd(); glPopMatrix()

# ══════════════════════════════════════════════════════════════════════════════
# BULLET HOLES
# ══════════════════════════════════════════════════════════════════════════════
MAX_HOLES=64
class BulletHoles:
    def __init__(self): self.holes=[]
    def add(self,pos,normal): self.holes.append((pos,normal)); (len(self.holes)>MAX_HOLES) and self.holes.pop(0)
    def draw(self):
        if not self.holes: return
        glDisable(GL_LIGHTING); glEnable(GL_POLYGON_OFFSET_FILL); glPolygonOffset(-1,-1); glColor4f(0.05,0.03,0.02,0.92); SIZE=0.06
        for (px,py,pz),(nx,ny,nz) in self.holes:
            up=[0,1,0] if abs(ny)<0.9 else [1,0,0]
            tx=ny*up[2]-nz*up[1]; ty=nz*up[0]-nx*up[2]; tz=nx*up[1]-ny*up[0]; tl=math.sqrt(tx*tx+ty*ty+tz*tz)
            if tl<1e-6: continue
            tx/=tl; ty/=tl; tz/=tl; bx=ny*tz-nz*ty; by_=nz*tx-nx*tz; bz=nx*ty-ny*tx
            off=0.003; ox=px+nx*off; oy=py+ny*off; oz=pz+nz*off
            glBegin(GL_QUADS)
            glVertex3f(ox-tx*SIZE-bx*SIZE,oy-ty*SIZE-by_*SIZE,oz-tz*SIZE-bz*SIZE)
            glVertex3f(ox+tx*SIZE-bx*SIZE,oy+ty*SIZE-by_*SIZE,oz+tz*SIZE-bz*SIZE)
            glVertex3f(ox+tx*SIZE+bx*SIZE,oy+ty*SIZE+by_*SIZE,oz+tz*SIZE+bz*SIZE)
            glVertex3f(ox-tx*SIZE+bx*SIZE,oy-ty*SIZE+by_*SIZE,oz-tz*SIZE+bz*SIZE)
            glEnd()
        glDisable(GL_POLYGON_OFFSET_FILL); glEnable(GL_LIGHTING)

BULLET_HOLES=BulletHoles()

# ══════════════════════════════════════════════════════════════════════════════
# MUZZLE FLASH
# ══════════════════════════════════════════════════════════════════════════════
class MuzzleFlash:
    def __init__(self): self.frames=0
    def fire(self): self.frames=3
    def update(self):
        if self.frames>0: self.frames-=1
    def draw(self):
        if self.frames==0: return
        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity(); gluPerspective(75.0,W/H,0.01,10.0)
        glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
        glDisable(GL_LIGHTING); glDisable(GL_DEPTH_TEST); glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA,GL_ONE)
        alpha=self.frames/3.0; glColor4f(1.0,0.85,0.3,alpha*0.7)
        cx=GUN_POS[0]+0.01; cy=GUN_POS[1]+0.04; cz=GUN_POS[2]-0.05; r=0.018
        glBegin(GL_TRIANGLE_FAN); glVertex3f(cx,cy,cz)
        for i in range(9): a=i*math.pi*2/8; rr=r if i%2==0 else r*0.45; glVertex3f(cx+math.cos(a)*rr,cy+math.sin(a)*rr,cz)
        glEnd(); glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA); glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING)
        glMatrixMode(GL_MODELVIEW); glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix()

MUZZLE_FLASH=MuzzleFlash()

# ══════════════════════════════════════════════════════════════════════════════
# GUN STATE
# ══════════════════════════════════════════════════════════════════════════════
class GunState:
    def __init__(self):
        self.ammo=GUN_MAG; self.fire_cd=0; self.reloading=False; self.reload_timer=0
        self._recoil_pitch_tgt=0.0; self._recoil_y_tgt=0.0
        self.recoil_pitch=0.0; self.recoil_offset_y=0.0
        self.reload_offset=0.0; self.reload_rot=0.0
        self._jump_drop_tgt=0.0; self.jump_drop_smooth=0.0
        self._lag_tgt=0.0; self.lag_smooth=0.0

    def try_fire(self, player, net):
        if self.reloading or self.fire_cd>0 or self.ammo<=0: return
        self.ammo-=1; self.fire_cd=GUN_FIRE_CD
        self._recoil_pitch_tgt+=RECOIL_PITCH; self._recoil_y_tgt+=0.06
        player.pitch_target-=RECOIL_PITCH*0.08
        MUZZLE_FLASH.fire()
        ld=player.look_dir(); eye=player.eye_pos()
        # local geometry bullet hole
        result=_raycast_hitscan(eye[0],eye[1],eye[2],ld[0],ld[1],ld[2])
        if result: BULLET_HOLES.add((result[0],result[1],result[2]),(-ld[0],-ld[1],-ld[2]))
        # tell server
        net.send({"type":"shoot","yaw":player.yaw,"pitch":player.pitch})
        if self.ammo==0: self.try_reload()

    def try_reload(self):
        if self.reloading or self.ammo==GUN_MAG: return
        self.reloading=True; self.reload_timer=GUN_RELOAD_DUR

    def update(self, player):
        if self.fire_cd>0: self.fire_cd-=1
        self.recoil_pitch=lerp(self.recoil_pitch,self._recoil_pitch_tgt,0.35)
        self.recoil_offset_y=lerp(self.recoil_offset_y,self._recoil_y_tgt,0.35)
        self._recoil_pitch_tgt=lerp(self._recoil_pitch_tgt,0.0,0.08)
        self._recoil_y_tgt=lerp(self._recoil_y_tgt,0.0,0.08)
        if self.reloading:
            self.reload_timer-=1; t=1.0-self.reload_timer/GUN_RELOAD_DUR; arc=math.sin(t*math.pi)
            self.reload_offset=lerp(self.reload_offset,RELOAD_KICK_X*arc,0.18)
            self.reload_rot=lerp(self.reload_rot,RELOAD_KICK_ROT*arc,0.18)
            if self.reload_timer<=0: self.reloading=False; self.ammo=GUN_MAG
        else:
            self.reload_offset=lerp(self.reload_offset,0.0,0.12)
            self.reload_rot=lerp(self.reload_rot,0.0,0.12)
        self._jump_drop_tgt=-0.04 if not player.on_ground else 0.0
        self.jump_drop_smooth=lerp(self.jump_drop_smooth,self._jump_drop_tgt,0.08)
        spd_h=math.sqrt(player.vel[0]**2+player.vel[2]**2)
        self._lag_tgt=spd_h/MAX_GROUND_SPD; self.lag_smooth=lerp(self.lag_smooth,self._lag_tgt,0.10)
        MUZZLE_FLASH.update()

# ══════════════════════════════════════════════════════════════════════════════
# PLAYER  (client-side prediction — same physics as server)
# ══════════════════════════════════════════════════════════════════════════════
class Player:
    def __init__(self, pos):
        self.pos=[*pos]; self.vel=v3(0,0,0); self._prev=list(pos)
        self.yaw=0.0; self.pitch=0.0; self.yaw_target=0.0; self.pitch_target=0.0
        self.on_ground=False; self.on_wall=False; self.coyote=0
        self.jump_held=False; self.jump_buffer=0; self.grounded_frames=0; self.just_jumped=False
        self.slide_t=0; self.slide_cd=0; self.sliding=False
        self.bob=0.0; self.bob_amp=0.0; self.tilt=0.0; self.land_squish=0.0
        self.fov_current=BASE_FOV; self.eye_y_smooth=None; self._was_on_ground=False
        self.health=100; self.alive=True

    def eye_pos(self):
        slide_off=-0.45 if self.sliding else 0.0; bob_off=math.sin(self.bob)*self.bob_amp
        target_y=self.pos[1]+EYE+slide_off
        if self.eye_y_smooth is None: self.eye_y_smooth=target_y
        else: self.eye_y_smooth=lerp(self.eye_y_smooth,target_y,0.22)
        return [self.pos[0],self.eye_y_smooth+bob_off-self.land_squish,self.pos[2]]

    def fwd(self):
        yr=math.radians(self.yaw); return [math.sin(yr),0.0,-math.cos(yr)]

    def look_dir(self):
        yr=math.radians(self.yaw); pr=math.radians(self.pitch)
        return [math.cos(pr)*math.sin(yr),math.sin(pr),-math.cos(pr)*math.cos(yr)]

    def apply_input(self, inp):
        CAM=0.18; self.yaw_target=(self.yaw_target+inp.get("dy",0))%360
        self.pitch_target=clamp(self.pitch_target+inp.get("dp",0),-89,89)
        dyaw=(self.yaw_target-self.yaw+180)%360-180; self.yaw=(self.yaw+dyaw*CAM)%360
        self.pitch=lerp(self.pitch,self.pitch_target,CAM)
        yr=math.radians(self.yaw_target); fw=[math.sin(yr),0.0,-math.cos(yr)]; rt=[math.cos(yr),0.0,math.sin(yr)]
        wx=wz=0.0
        if inp.get("w"): wx+=fw[0]; wz+=fw[2]
        if inp.get("s"): wx-=fw[0]; wz-=fw[2]
        if inp.get("d"): wx+=rt[0]; wz+=rt[2]
        if inp.get("a"): wx-=rt[0]; wz-=rt[2]
        wlen=math.sqrt(wx*wx+wz*wz)
        if wlen>0: wx/=wlen; wz/=wlen
        if inp.get("jump_press"): self.jump_buffer=JUMP_BUFFER
        if inp.get("slide") and self.on_ground and self.slide_cd==0 and wlen>0 and not self.sliding:
            self.sliding=True; self.slide_t=SLIDE_DUR; self.slide_cd=SLIDE_CD
            self.vel[0]=wx*SLIDE_VEL; self.vel[2]=wz*SLIDE_VEL
        can_jump=self.on_ground or self.coyote>0
        if self.jump_buffer>0 and can_jump:
            self.vel[1]=JUMP_VEL; self.coyote=0; self.jump_held=True; self.jump_buffer=0
            self.grounded_frames=0; self.just_jumped=True; self.sliding=False; self.slide_t=0
        if self.jump_held and not inp.get("jump_held") and self.vel[1]>0: self.vel[1]*=JUMP_CUT; self.jump_held=False
        elif not inp.get("jump_held"): self.jump_held=False
        if self.sliding: self.vel[0]*=0.93; self.vel[2]*=0.93
        elif self.on_ground:
            vxz=[self.vel[0]*GROUND_FRICTION,self.vel[2]*GROUND_FRICTION]
            if wlen>0: vxz=pm_accelerate_xz(vxz,[wx,wz],MAX_GROUND_SPD,GROUND_ACCEL)
            self.vel[0]=vxz[0]; self.vel[2]=vxz[1]
        else:
            if wlen>0:
                vxz=pm_accelerate_xz([self.vel[0],self.vel[2]],[wx,wz],MAX_AIR_WISH,AIR_ACCEL)
                self.vel[0]=vxz[0]; self.vel[2]=vxz[1]
        spd_h=math.sqrt(self.vel[0]**2+self.vel[2]**2)
        bob_t=spd_h*0.018 if self.on_ground and not self.sliding else 0.0
        self.bob_amp=lerp(self.bob_amp,bob_t,0.08)
        if self.on_ground and spd_h>0.04: self.bob+=spd_h*2.5
        else: self.bob*=0.90
        self.tilt=lerp(self.tilt,inp.get("dy",0)*0.55,0.14)
        fov_t=BASE_FOV+SPEED_FOV_ADD*clamp(spd_h/MAX_GROUND_SPD,0,2.0)
        if self.sliding: fov_t=BASE_FOV+SPEED_FOV_ADD*1.8
        self.fov_current=lerp(self.fov_current,fov_t,FOV_LERP)
        self.land_squish=lerp(self.land_squish,0.0,0.22)

    def update(self):
        if not self.alive: return
        if self.slide_t>0: self.slide_t-=1
        else: self.sliding=False
        if self.slide_cd>0: self.slide_cd-=1
        if self.coyote>0: self.coyote-=1
        if self.jump_buffer>0: self.jump_buffer-=1
        self.vel[1]-=GRAVITY; self._prev=list(self.pos)
        new_pos=[self.pos[0]+self.vel[0],self.pos[1]+self.vel[1],self.pos[2]+self.vel[2]]
        prev_g=self.on_ground; self.on_ground=False; self.on_wall=False
        if TRI_GRID.tri_arr is not None and len(TRI_GRID.tri_arr)>0:
            dx=new_pos[0]-self._prev[0]; dz=new_pos[2]-self._prev[2]; dy_=new_pos[1]-self._prev[1]
            dist3d=math.sqrt(dx*dx+dy_*dy_+dz*dz); n_steps=max(1,int(math.ceil(dist3d/(P_R*0.5))))
            cur=list(self._prev); on_g=False; on_w=False
            for s in range(n_steps):
                frac=(s+1)/n_steps; sub=[self._prev[0]+dx*frac,self._prev[1]+dy_*frac,self._prev[2]+dz*frac]
                on_g,on_w,cur,hc=collide_capsule_map(cur,sub,P_R,P_H,self.vel[1])
                if hc: self.vel[1]=min(self.vel[1],0.0); self.jump_held=False; self.jump_buffer=0
                if on_g and self.vel[1]<0: self.vel[1]=0.0
                if on_w and not on_g:
                    if prev_g or on_g:
                        ls=[cur[0],cur[1]+STEP_UP,cur[2]]; ld_=[sub[0],cur[1]+STEP_UP,sub[2]]
                        og2,ow2,cl2,_=collide_capsule_map(ls,ld_,P_R,P_H,0.0)
                        if not ow2:
                            fy=_raycast_down(cl2[0],cl2[1],cl2[2],STEP_UP+0.1)
                            if fy is not None: cl2[1]=fy
                            cur=cl2; on_g=True; on_w=False; continue
                    self.vel[0]=0.0; self.vel[2]=0.0; break
            self.pos=cur
            if on_g: self.on_ground=True
            self.on_wall=on_w
            if self.on_ground: self.grounded_frames=2; self.just_jumped=False
            else:
                if self.just_jumped: self.grounded_frames=0
                elif self.grounded_frames>0:
                    self.grounded_frames-=1
                    if self.grounded_frames==1 and -0.015<self.vel[1]<=0.0:
                        sh=math.sqrt(self.vel[0]**2+self.vel[2]**2)
                        if sh>0.001:
                            fy=_raycast_down(self.pos[0],self.pos[1],self.pos[2],STEP_DOWN)
                            if fy is not None and 0.001<(self.pos[1]-fy)<=STEP_DOWN:
                                self.pos[1]=fy; self.on_ground=True; self.vel[1]=0.0
        else:
            self.pos=new_pos
        if prev_g and not self.on_ground: self.coyote=COYOTE_FRAMES
        self.vel[1]=clamp(self.vel[1],-2.0,JUMP_VEL*1.1)
        if not prev_g and self.on_ground:
            self.land_squish=clamp(-self.vel[1]*3.0,0.0,0.28)
        floor_y=(GLTF_MAP.aabb_min[1] if GLTF_MAP.aabb_min else 0.0)-30
        if self.pos[1]<floor_y: self.pos=list(spawn_pos()); self.vel=v3(0,0,0); self.land_squish=0; self.sliding=False; self.slide_t=0

    def reconcile(self, srv_snap):
        """Only correct position when the server says we're badly wrong.
        For small drift we trust local prediction entirely — no blending,
        which would create the rubber-band / sluggish feel."""
        sx, sy, sz = srv_snap["x"], srv_snap["y"], srv_snap["z"]
        dx = sx - self.pos[0]; dy = sy - self.pos[1]; dz = sz - self.pos[2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        # Only snap on genuinely large divergence (teleport, respawn, etc.)
        # Small drift is normal floating point noise — ignore it entirely.
        if dist > 3.0:
            self.pos = [sx, sy, sz]
            self.vel = [0, 0, 0]
        self.health = srv_snap.get("health", 100)
        self.alive  = srv_snap.get("alive", True)

def pm_accelerate_xz(vel_xz, wish_dir_xz, wish_speed, accel):
    cur=vel_xz[0]*wish_dir_xz[0]+vel_xz[1]*wish_dir_xz[1]; add=wish_speed-cur
    if add<=0: return vel_xz
    gain=min(accel,add); return [vel_xz[0]+gain*wish_dir_xz[0],vel_xz[1]+gain*wish_dir_xz[1]]

# ══════════════════════════════════════════════════════════════════════════════
# SPAWN + GL HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def spawn_pos():
    if GLTF_MAP.aabb_min and GLTF_MAP.aabb_max:
        mn,mx=GLTF_MAP.aabb_min,GLTF_MAP.aabb_max; cx=(mn[0]+mx[0])/2; cz=(mn[2]+mx[2])/2
        cast_y=mx[1]+2.0; hr=mx[1]-mn[1]; offsets=[(0,0),(2,0),(-2,0),(0,2),(0,-2)]
        for ox,oz in offsets:
            fy=_raycast_down(cx+ox,cast_y,cz+oz,hr+4.0)
            if fy is not None: return [cx+ox,fy+P_H*0.6,cz+oz]
        return [cx,mx[1]+6.0,cz]
    return [0.0,4.0,0.0]

def init_gl():
    glEnable(GL_DEPTH_TEST); glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_LIGHTING); glEnable(GL_LIGHT0); glEnable(GL_LIGHT1)
    glEnable(GL_COLOR_MATERIAL); glColorMaterial(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE)
    glLightfv(GL_LIGHT0,GL_POSITION,[10,20,10,1]); glLightfv(GL_LIGHT0,GL_DIFFUSE,[1.0,0.92,0.82,1]); glLightfv(GL_LIGHT0,GL_AMBIENT,[0.22,0.18,0.32,1])
    glLightfv(GL_LIGHT1,GL_POSITION,[-10,6,-10,1]); glLightfv(GL_LIGHT1,GL_DIFFUSE,[0.18,0.14,0.38,1])
    glShadeModel(GL_SMOOTH)

def set_camera(player):
    glMatrixMode(GL_PROJECTION); glLoadIdentity(); gluPerspective(player.fov_current,W/H,0.04,1000)
    glMatrixMode(GL_MODELVIEW); glLoadIdentity(); glRotatef(player.tilt,0,0,1)
    eye=player.eye_pos(); ld=player.look_dir(); ct=vadd(eye,ld)
    gluLookAt(eye[0],eye[1],eye[2],ct[0],ct[1],ct[2],0,1,0)

def draw_sky():
    glDisable(GL_LIGHTING); glDisable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity(); glOrtho(-1,1,-1,1,-1,1)
    glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
    glBegin(GL_QUADS); glColor3f(*SKY_TOP); glVertex2f(-1,1); glVertex2f(1,1); glColor3f(*SKY_BOT); glVertex2f(1,-1); glVertex2f(-1,-1); glEnd()
    glMatrixMode(GL_PROJECTION); glPopMatrix(); glMatrixMode(GL_MODELVIEW); glPopMatrix()
    glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING)

def draw_fallback_arena():
    _draw_box_gl(0,-0.15,0,ARENA_W,0.3,ARENA_D,FLOOR_COL)

def _draw_box_gl(cx,cy,cz,bw,bh,bd,col,alpha=1.0):
    hw,hh,hd=bw/2,bh/2,bd/2; glColor4f(col[0],col[1],col[2],alpha)
    glPushMatrix(); glTranslatef(cx,cy,cz); glBegin(GL_QUADS)
    for (nx,ny,nz),verts in [((0,0,1),((-hw,-hh,hd),(hw,-hh,hd),(hw,hh,hd),(-hw,hh,hd))),((0,0,-1),((hw,-hh,-hd),(-hw,-hh,-hd),(-hw,hh,-hd),(hw,hh,-hd))),((-1,0,0),((-hw,-hh,-hd),(-hw,-hh,hd),(-hw,hh,hd),(-hw,hh,-hd))),((1,0,0),((hw,-hh,hd),(hw,-hh,-hd),(hw,hh,-hd),(hw,hh,hd))),((0,1,0),((-hw,hh,hd),(hw,hh,hd),(hw,hh,-hd),(-hw,hh,-hd))),((0,-1,0),((-hw,-hh,-hd),(hw,-hh,-hd),(hw,-hh,hd),(-hw,-hh,hd)))]:
        glNormal3f(nx,ny,nz)
        for v in verts: glVertex3f(*v)
    glEnd(); glPopMatrix()

def draw_crosshair(player):
    glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity(); glOrtho(0,W,H,0,-1,1)
    glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
    glDisable(GL_DEPTH_TEST); glDisable(GL_LIGHTING)
    cx,cy=W//2,H//2; spd=math.sqrt(player.vel[0]**2+player.vel[2]**2); gap=int(4+spd*18); s=9
    glColor4f(1,1,1,0.90); glLineWidth(1.5); glBegin(GL_LINES)
    glVertex2f(cx-s-gap,cy); glVertex2f(cx-gap,cy); glVertex2f(cx+gap,cy); glVertex2f(cx+s+gap,cy)
    glVertex2f(cx,cy-s-gap); glVertex2f(cx,cy-gap); glVertex2f(cx,cy+gap); glVertex2f(cx,cy+s+gap)
    glEnd(); glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING)
    glMatrixMode(GL_PROJECTION); glPopMatrix(); glMatrixMode(GL_MODELVIEW); glPopMatrix()

_font_tex_cache={}
def _gl_text_tex(font,text,color):
    key=(id(font),text,color)
    if key not in _font_tex_cache:
        surf=font.render(text,True,color); data=pygame.image.tostring(surf,"RGBA",True)
        tid=int(glGenTextures(1)); glBindTexture(GL_TEXTURE_2D,tid)
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,surf.get_width(),surf.get_height(),0,GL_RGBA,GL_UNSIGNED_BYTE,data)
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR); glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR); glBindTexture(GL_TEXTURE_2D,0)
        _font_tex_cache[key]=(tid,surf.get_width(),surf.get_height())
    return _font_tex_cache[key]

def _hud_text(font,text,color,x,y):
    tid,tw,th=_gl_text_tex(font,text,color); by=H-y-th
    glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D,tid); glColor4f(1,1,1,1)
    glBegin(GL_QUADS); glTexCoord2f(0,0); glVertex2f(x,by); glTexCoord2f(1,0); glVertex2f(x+tw,by); glTexCoord2f(1,1); glVertex2f(x+tw,by+th); glTexCoord2f(0,1); glVertex2f(x,by+th); glEnd()
    glDisable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D,0)

def draw_hud(player, font_sm, font_med, gun_state, remote_players, kill_feed, net):
    glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity(); glOrtho(0,W,0,H,-1,1)
    glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
    glDisable(GL_DEPTH_TEST); glDisable(GL_LIGHTING); glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)

    x,y,z=player.pos; spd=math.sqrt(player.vel[0]**2+player.vel[2]**2)
    status="ONLINE" if net.connected else "OFFLINE"
    _hud_text(font_sm,f"XYZ  {x:.1f}  {y:.2f}  {z:.1f}",(200,200,200),14,14)
    _hud_text(font_sm,f"SPD  {spd*100:.0f}",(200,200,200),14,32)
    _hud_text(font_sm,f"NET: {status}  Players: {len(remote_players)+1}",(120,200,120) if net.connected else (200,80,80),14,50)
    _hud_text(font_sm,f"HP: {player.health}",(80,255,80) if player.health>50 else (255,80,80),14,68)

    if not player.alive:
        _hud_text(font_med,"YOU DIED — RESPAWNING",(255,80,80),W//2-180,H//2+10)

    if player.sliding: _hud_text(font_med,"SLIDE",(180,80,255),W//2-42,H-54)

    # Ammo
    if gun_state.reloading: _hud_text(font_med,"RELOADING",(255,200,60),W-200,18)
    else:
        ammo_col=(255,255,255) if gun_state.ammo>3 else (255,80,60)
        _hud_text(font_med,f"{gun_state.ammo}  /  {GUN_MAG}",ammo_col,W-160,18)

    # Kill feed (top right)
    now=time.time()
    kill_feed[:]=[kf for kf in kill_feed if now-kf[2]<6.0]
    for i,(killer,victim,ts) in enumerate(reversed(kill_feed[-5:])):
        alpha=max(0,1.0-(now-ts)/6.0); col=(255,200,60) if alpha>0.5 else (180,140,40)
        _hud_text(font_sm,f"{killer}  killed  {victim}",col,W-360,H-22-i*18)

    # Scoreboard (simple, top right under kill feed)
    all_players=list(remote_players.values())
    for i,rp in enumerate(all_players[:8]):
        _hud_text(font_sm,f"{rp.name}  HP:{rp.health}",(200,220,255),W-280,H-130-i*16)

    _hud_text(font_sm,"WASD move · Space jump · Ctrl slide · LMB shoot · R reload",(90,80,120),14,H-42)

    glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING)
    glMatrixMode(GL_PROJECTION); glPopMatrix(); glMatrixMode(GL_MODELVIEW); glPopMatrix()

# ══════════════════════════════════════════════════════════════════════════════
# GUN MODEL  (viewmodel — same as original)
# ══════════════════════════════════════════════════════════════════════════════
class GunModel:
    def __init__(self): self.draw_calls=[]; self.loaded=False
    def load(self,path="gun.gltf"):
        if not _HAS_GLTF or not os.path.exists(path): return
        tmp=GLTFMap()
        if not tmp.parse(path,skip_tg=True): return
        for m in tmp._cpu:
            try:
                vf=m["vdata"].flatten().astype(np.float32); if_=m["idata"].flatten().astype(np.uint32)
                vbo=int(glGenBuffers(1)); glBindBuffer(GL_ARRAY_BUFFER,vbo); glBufferData(GL_ARRAY_BUFFER,vf.nbytes,vf.tobytes(),GL_STATIC_DRAW); glBindBuffer(GL_ARRAY_BUFFER,0)
                ebo=int(glGenBuffers(1)); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,ebo); glBufferData(GL_ELEMENT_ARRAY_BUFFER,if_.nbytes,if_.tobytes(),GL_STATIC_DRAW); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,0)
                tid=tmp._load_tex(m["gltf"],m["tex_idx"],path) if m["tex_idx"] is not None else None
                self.draw_calls.append((vbo,ebo,int(len(if_)),m["col"],tid))
            except Exception as e: print(f"[GUN] {e}")
        self.loaded=len(self.draw_calls)>0

    def draw(self,player,gs):
        if not self.loaded: return
        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity(); gluPerspective(75.0,W/H,0.01,10.0)
        glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity(); glClear(GL_DEPTH_BUFFER_BIT)
        tx=GUN_POS[0]; ty=GUN_POS[1]+gs.recoil_offset_y-gs.reload_offset; tz=GUN_POS[2]
        glTranslatef(tx,ty,tz); glRotatef(gs.recoil_pitch+gs.reload_rot,1,0,0); glRotatef(72,0,20000,170)
        spd_h=math.sqrt(player.vel[0]**2+player.vel[2]**2); still=max(0.0,1.0-spd_h/(MAX_GROUND_SPD*0.15))
        t=time.time(); by_=math.sin(t*1.1)*0.004*still; bx=math.sin(t*0.55)*0.0016*still
        glTranslatef(gs.lag_smooth*-0.04+bx,gs.lag_smooth*-0.02+by_+gs.jump_drop_smooth,gs.lag_smooth*0.03)
        s=GUN_SCALE/MAP_SCALE; glScalef(s,s,s)
        stride=8*4; glEnable(GL_LIGHTING); glEnableClientState(GL_VERTEX_ARRAY); glEnableClientState(GL_NORMAL_ARRAY); glEnableClientState(GL_TEXTURE_COORD_ARRAY)
        for vbo,ebo,count,col,tid in self.draw_calls:
            if tid is not None: glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D,tid); glColor4f(1,1,1,1)
            else: glDisable(GL_TEXTURE_2D); glColor4f(col[0],col[1],col[2],1.0)
            glBindBuffer(GL_ARRAY_BUFFER,vbo); glVertexPointer(3,GL_FLOAT,stride,ctypes.c_void_p(0)); glNormalPointer(GL_FLOAT,stride,ctypes.c_void_p(12)); glTexCoordPointer(2,GL_FLOAT,stride,ctypes.c_void_p(24))
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,ebo); glDrawElements(GL_TRIANGLES,count,GL_UNSIGNED_INT,None)
        glDisableClientState(GL_VERTEX_ARRAY); glDisableClientState(GL_NORMAL_ARRAY); glDisableClientState(GL_TEXTURE_COORD_ARRAY)
        glBindTexture(GL_TEXTURE_2D,0); glDisable(GL_TEXTURE_2D); glBindBuffer(GL_ARRAY_BUFFER,0); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,0)
        glMatrixMode(GL_MODELVIEW); glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix()

GUN_MODEL=GunModel()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--host",   default="127.0.0.1")
    parser.add_argument("--port",   type=int, default=7777)
    parser.add_argument("--name",   default="Player")
    parser.add_argument("--map",    default="ex.gltf")
    parser.add_argument("--secure", action="store_true", help="Use wss:// (for Render/cloud hosting)")
    args=parser.parse_args()

    pygame.init()
    pygame.display.set_mode((W,H),pygame.OPENGL|pygame.DOUBLEBUF)
    pygame.display.set_caption(f"FPS Client — {args.name}")
    _font_tex_cache.clear(); init_gl()
    font_sm=pygame.font.SysFont("consolas",15); font_med=pygame.font.SysFont("consolas",26,bold=True)

    GLTF_MAP.parse(args.map)
    if GLTF_MAP.parsed: GLTF_MAP.upload()
    GUN_MODEL.load("gun.gltf")

    player    = Player(spawn_pos())
    gun_state = GunState()
    remote_players: dict[str, RemotePlayer] = {}
    kill_feed = []

    net = NetClient(args.host, args.port, secure=args.secure or args.port == 443)
    net.start()

    clock      = pygame.time.Clock()
    mc         = False
    prev_jump  = False
    prev_slide = False
    input_seq  = 0
    name_sent  = False

    rng = random.Random(77)
    amb = [(rng.uniform(-200,200), rng.uniform(10,80), rng.uniform(-200,200)) for _ in range(300)]

    while True:
        clock.tick(120)

        # ── name handshake ────────────────────────────────────────────────────
        if net.connected and not name_sent:
            net.send({"type":"name","name":args.name}); name_sent=True

        # ── network receive ───────────────────────────────────────────────────
        for msg in net.poll():
            mtype = msg.get("type","")
            if mtype == "welcome":
                net.our_pid = str(msg.get("your_pid",""))
                spawn = msg.get("spawn")
                if spawn: player.pos=list(spawn); player.vel=[0,0,0]
                # purge self if we were added to remote_players before welcome arrived
                remote_players.pop(net.our_pid, None)
                print(f"[NET] pid={net.our_pid}")

            elif mtype == "state":
                for snap in msg.get("players",[]):
                    pid = str(snap["pid"])
                    if net.our_pid and pid == net.our_pid:
                        # Own player: only sync health/alive from server
                        player.health = snap.get("health", player.health)
                        player.alive  = snap.get("alive",  player.alive)
                    else:
                        # Remote player: add if new, always update
                        if pid not in remote_players:
                            remote_players[pid] = RemotePlayer(pid, snap.get("name", "?"))
                        remote_players[pid].update_from_snap(snap)
                        # If we later learn this was actually us, remove it
                        if net.our_pid and pid == net.our_pid:
                            del remote_players[pid]
                snap_pids = {str(s["pid"]) for s in msg.get("players", [])}
                for gone in [p for p in list(remote_players) if p not in snap_pids]:
                    del remote_players[gone]
                for ev in msg.get("events", []):
                    if ev.get("type") == "kill":
                        kill_feed.append((ev.get("killer_name","?"), ev.get("victim_name","?"), time.time()))

        # ── pygame events ─────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if mc: pygame.mouse.set_visible(True); pygame.event.set_grab(False); mc=False
                    else: pygame.quit(); sys.exit()
                if event.key == pygame.K_r: gun_state.try_reload()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button==1:
                if not mc: pygame.mouse.set_visible(False); pygame.event.set_grab(True); mc=True
                else: gun_state.try_fire(player, net)

        # ── input ─────────────────────────────────────────────────────────────
        mx, my     = pygame.mouse.get_rel()
        dyaw       =  mx * MOUSE_SENS if mc else 0.0
        dpitch     = -my * MOUSE_SENS if mc else 0.0
        keys       = pygame.key.get_pressed()
        jump_now   = bool(keys[pygame.K_SPACE])
        slide_now  = bool(keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL])

        inp = {
            "w":          bool(keys[pygame.K_w]),
            "s":          bool(keys[pygame.K_s]),
            "a":          bool(keys[pygame.K_a]),
            "d":          bool(keys[pygame.K_d]),
            "jump_press": jump_now and not prev_jump,
            "jump_held":  jump_now,
            "slide":      slide_now and not prev_slide,
            "dy":         dyaw,
            "dp":         dpitch,
        }
        prev_jump  = jump_now
        prev_slide = slide_now

        # ── physics (one tick per frame, always) ──────────────────────────────
        player.apply_input(inp)
        player.update()
        gun_state.update(player)
        for rp in remote_players.values(): rp.smooth()

        # ── send input (non-blocking via queue) ───────────────────────────────
        input_seq += 1
        if net.connected:
            net.send({
                "type":       "input",
                "seq":        input_seq,
                "w":          inp["w"],
                "s":          inp["s"],
                "a":          inp["a"],
                "d":          inp["d"],
                "jump_press": inp["jump_press"],
                "jump_held":  inp["jump_held"],
                "slide":      inp["slide"],
                "yaw":        round(player.yaw,   2),
                "pitch":      round(player.pitch, 2),
                "x":          round(player.pos[0], 3),
                "y":          round(player.pos[1], 3),
                "z":          round(player.pos[2], 3),
            })

        # ── render ────────────────────────────────────────────────────────────
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        draw_sky()
        set_camera(player)

        glDisable(GL_LIGHTING); glPointSize(2); glBegin(GL_POINTS)
        rng2 = random.Random(77)
        for ax,ay,az in amb:
            br = rng2.uniform(0.3,1.0); glColor3f(br,br,min(1,br+0.15)); glVertex3f(ax,ay,az)
        glEnd(); glEnable(GL_LIGHTING)

        if GLTF_MAP.uploaded: GLTF_MAP.draw()
        else: draw_fallback_arena()

        BULLET_HOLES.draw()
        glEnable(GL_LIGHTING)
        for rp in remote_players.values(): rp.draw()
        GUN_MODEL.draw(player, gun_state)
        MUZZLE_FLASH.draw()
        draw_crosshair(player)
        draw_hud(player, font_sm, font_med, gun_state, remote_players, kill_feed, net)

        if not mc:
            glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity(); glOrtho(0,W,0,H,-1,1)
            glMatrixMode(GL_MODELVIEW);  glPushMatrix(); glLoadIdentity()
            glDisable(GL_DEPTH_TEST); glDisable(GL_LIGHTING)
            _hud_text(font_med, "Click window to capture mouse", (255,200,60), W//2-200, H//2)
            glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING)
            glMatrixMode(GL_PROJECTION); glPopMatrix()
            glMatrixMode(GL_MODELVIEW);  glPopMatrix()

        pygame.display.flip()

if __name__=="__main__":
    main()