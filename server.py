"""
FPS Multiplayer Server
  pip install pygltflib numpy

  python server.py [map.gltf] [--port 7777]

  Runs headless physics for all connected players.
  Broadcasts full world state to all clients at ~60Hz.
  Accepts inputs from clients.
"""

import asyncio, json, math, time, sys, os, random, argparse
import numpy as np

# ── Try to import websockets ──────────────────────────────────────────────────
try:
    import websockets
    from websockets.server import serve
except ImportError:
    print("Install: pip install websockets"); sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG  (must match client)
# ══════════════════════════════════════════════════════════════════════════════
TICK_RATE      = 60       # physics ticks per second
BROADCAST_RATE = 20       # state snapshots per second sent to clients
DEFAULT_PORT   = 7777

# Physics constants (copied from game)
GRAVITY        = 0.0028
JUMP_VEL       = 0.10
JUMP_CUT       = 0.75
JUMP_BUFFER    = 8
COYOTE_FRAMES  = 8
STEP_UP        = 0.30
STEP_DOWN      = 0.30
GROUND_ACCEL   = 0.014
GROUND_FRICTION= 0.78
MAX_GROUND_SPD = 0.08
AIR_ACCEL      = 0.004
MAX_AIR_WISH   = 0.08
SLIDE_VEL      = 0.22
SLIDE_DUR      = 18
SLIDE_CD       = 36
MAP_SCALE      = 2.0
P_H = 1.8
P_R = 0.38
EYE = 1.62

# ══════════════════════════════════════════════════════════════════════════════
# MINIMAL MATH
# ══════════════════════════════════════════════════════════════════════════════
def clamp(v, lo, hi): return max(lo, min(hi, v))
def lerp(a, b, t):    return a + (b - a) * t
def vlen(a):          return math.sqrt(a[0]**2 + a[1]**2 + a[2]**2)

def pm_accelerate(vel_xz, wish_dir_xz, wish_speed, accel):
    cur = vel_xz[0]*wish_dir_xz[0] + vel_xz[1]*wish_dir_xz[1]
    add = wish_speed - cur
    if add <= 0: return vel_xz
    gain = min(accel, add)
    return [vel_xz[0] + gain*wish_dir_xz[0], vel_xz[1] + gain*wish_dir_xz[1]]

# ══════════════════════════════════════════════════════════════════════════════
# SPATIAL GRID + COLLISION  (same as game, trimmed)
# ══════════════════════════════════════════════════════════════════════════════
class TriGrid:
    def __init__(self, cell=3.0):
        self.cell = cell; self.buckets = {}; self.tri_arr = None

    def build(self, triangles):
        self.buckets.clear()
        if not triangles:
            self.tri_arr = np.zeros((0,3,3), np.float32); return
        self.tri_arr = np.array(triangles, dtype=np.float32)
        cell = self.cell
        for idx in range(len(self.tri_arr)):
            xs = self.tri_arr[idx,:,0]; zs = self.tri_arr[idx,:,2]
            x0=int(math.floor(xs.min()/cell)); x1=int(math.floor(xs.max()/cell))
            z0=int(math.floor(zs.min()/cell)); z1=int(math.floor(zs.max()/cell))
            for gx in range(x0,x1+1):
                for gz in range(z0,z1+1):
                    self.buckets.setdefault((gx,gz),[]).append(idx)
        self._np = {k: np.array(v, dtype=np.int32) for k,v in self.buckets.items()}

    def query_indices(self, px, pz, radius):
        r = int(math.ceil(radius / self.cell)) + 1
        gx = int(math.floor(px / self.cell)); gz = int(math.floor(pz / self.cell))
        parts = []
        for dx in range(-r, r+1):
            for dz in range(-r, r+1):
                arr = self._np.get((gx+dx, gz+dz))
                if arr is not None: parts.append(arr)
        if not parts: return np.empty(0, dtype=np.int32)
        return np.unique(np.concatenate(parts))

TRI_GRID = TriGrid()

def _point_in_tri_batch(pts, a, b, c):
    v0=c-a; v1=b-a; v2=pts-a
    d00=(v0*v0).sum(1); d01=(v0*v1).sum(1); d02=(v0*v2).sum(1)
    d11=(v1*v1).sum(1); d12=(v1*v2).sum(1)
    denom=d00*d11-d01*d01; safe=np.abs(denom)>1e-12
    inv=np.where(safe,1.0/np.where(safe,denom,1.0),0.0)
    u=(d11*d02-d01*d12)*inv; v=(d00*d12-d01*d02)*inv
    return (u>=-0.01)&(v>=-0.01)&(u+v<=1.02)

def _closest_pts_tris_batch(p,a,b,c):
    ab=b-a; ac=c-a; ap=p-a
    d1=(ab*ap).sum(1); d2=(ac*ap).sum(1)
    bp=p-b; d3=(ab*bp).sum(1); d4=(ac*bp).sum(1)
    cp_=p-c; d5=(ab*cp_).sum(1); d6=(ac*cp_).sum(1)
    vc=d1*d4-d3*d2; vb=d5*d2-d1*d6; va=d3*d6-d5*d4
    denom=va+vb+vc; safe=np.abs(denom)>1e-12
    v=np.where(safe,vb/np.where(safe,denom,1.0),0.0)
    w=np.where(safe,vc/np.where(safe,denom,1.0),0.0)
    result=a+v[:,None]*ab+w[:,None]*ac
    m=(d1<=0)&(d2<=0);          result[m]=a[m]
    m=(d3>=0)&(d4<=d3);         result[m]=b[m]
    m=(d6>=0)&(d5<=d6);         result[m]=c[m]
    m=(vc<=0)&(d1>=0)&(d3<=0); tv=np.where(d1-d3>1e-12,d1/(d1-d3+1e-12),0.0); result[m]=(a+tv[:,None]*ab)[m]
    m=(vb<=0)&(d2>=0)&(d6<=0); tw=np.where(d2-d6>1e-12,d2/(d2-d6+1e-12),0.0); result[m]=(a+tw[:,None]*ac)[m]
    m=(va<=0)&((d4-d3)>=0)&((d5-d6)>=0)
    denom2=(d4-d3)+(d5-d6); tw2=np.where(np.abs(denom2)>1e-12,(d4-d3)/np.where(np.abs(denom2)>1e-12,denom2,1.0),0.0)
    result[m]=(b+tw2[:,None]*(c-b))[m]
    return result

def _collide_capsule(prev_pos, new_pos, radius, height, tri_arr, vy=0.0):
    rx,ry,rz = float(new_pos[0]),float(new_pos[1]),float(new_pos[2])
    on_ground=False; on_wall=False; hit_ceiling=False
    if len(tri_arr)==0: return on_ground,on_wall,[rx,ry,rz],hit_ceiling
    p0=tri_arr[:,0,:]; p1=tri_arr[:,1,:]; p2=tri_arr[:,2,:]
    e1=p1-p0; e2=p2-p0; n=np.cross(e1,e2)
    nl=np.linalg.norm(n,axis=1,keepdims=True); valid=(nl[:,0]>1e-9)
    n[valid]/=nl[valid]; ny=n[:,1]

    if vy<=0:
        floor_mask=valid&(ny>0.4)
        if floor_mask.any():
            fp0=p0[floor_mask]; fp1=p1[floor_mask]; fp2=p2[floor_mask]
            foot=np.array([[rx,ry,rz]],np.float32)
            cp=_closest_pts_tris_batch(foot,fp0,fp1,fp2)
            dxz=np.sqrt((rx-cp[:,0])**2+(rz-cp[:,2])**2)
            hit=(dxz<=radius+0.20)&(cp[:,1]<=ry+0.12)&(cp[:,1]>=ry-0.22)
            if hit.any():
                best_y=float(cp[hit,1].max())
                if best_y<=ry+0.12: ry=best_y; on_ground=True

    ceil_mask=valid&(ny<-0.35)
    if ceil_mask.any():
        cp0=p0[ceil_mask]; cp1=p1[ceil_mask]; cp2=p2[ceil_mask]
        for ct in (1.0,0.85,0.70):
            sy=ry+height*ct
            top=np.array([[rx,sy,rz]],np.float32)
            cc=_closest_pts_tris_batch(top,cp0,cp1,cp2)
            dxz_c=np.sqrt((rx-cc[:,0])**2+(rz-cc[:,2])**2)
            chit=(dxz_c<=radius+0.15)&(cc[:,1]>=sy-0.08)&(cc[:,1]<=sy+radius+0.15)
            if chit.any():
                lowest=float(cc[chit,1].min()); new_ry=lowest-height*ct-0.02
                if new_ry<ry: ry=new_ry; hit_ceiling=True

    wall_mask=valid&(np.abs(ny)<0.5)
    if wall_mask.any():
        wp0=p0[wall_mask]; wp1=p1[wall_mask]; wp2=p2[wall_mask]; wn=n[wall_mask]
        w_top=np.maximum(np.maximum(wp0[:,1],wp1[:,1]),wp2[:,1]); MIN_PEN=0.06
        for _ in range(6):
            any_pen=False
            for t in (0.35,0.75):
                sy=ry+height*t; pt=np.array([[rx,sy,rz]],np.float32)
                cp=_closest_pts_tris_batch(pt,wp0,wp1,wp2)
                dx3=rx-cp[:,0]; dz3=rz-cp[:,2]; dy3=sy-cp[:,1]
                dist3=np.sqrt(dx3*dx3+dy3*dy3+dz3*dz3); pen=dist3<radius
                if not pen.any(): continue
                depths=np.where(pen,radius-dist3,-1.0); i=int(np.argmax(depths)); d=float(dist3[i])
                if (radius-d)<MIN_PEN: continue
                face_rise=w_top[i]-ry
                if 0.0<=face_rise<=STEP_UP and vy<=0.001:
                    ry=float(w_top[i]); on_ground=True; any_pen=True; continue
                ox=float(dx3[i]); oz=float(dz3[i]); dxz=math.sqrt(ox*ox+oz*oz)
                if dxz<1e-6:
                    fnx=float(wn[i,0]); fnz=float(wn[i,2]); fnl=math.sqrt(fnx*fnx+fnz*fnz)
                    if fnl>1e-6: rx+=fnx/fnl*(radius-d+0.005); rz+=fnz/fnl*(radius-d+0.005)
                else:
                    ov=radius-d+0.005; rx+=(ox/dxz)*ov; rz+=(oz/dxz)*ov
                on_wall=True; any_pen=True
            if not any_pen: break

    for t in (0.35,0.75):
        sy=ry+height*t; pt=np.array([[rx,sy,rz]],np.float32)
        cp_all=_closest_pts_tris_batch(pt,p0,p1,p2)
        dist_all=np.sqrt(((pt-cp_all)**2).sum(axis=1))
        toward=((pt-cp_all)*n).sum(axis=1)
        deeply_inside=valid&(dist_all<radius*0.45)&(toward<0)
        if deeply_inside.any(): rx=float(prev_pos[0]); rz=float(prev_pos[2]); on_wall=True; break
    return on_ground,on_wall,[rx,ry,rz],hit_ceiling

def _raycast_down(rx,ry,rz,max_dist):
    if TRI_GRID.tri_arr is None or len(TRI_GRID.tri_arr)==0: return None
    idx=TRI_GRID.query_indices(rx,rz,P_R+1.0)
    if len(idx)==0: return None
    tris=TRI_GRID.tri_arr[idx]
    p0=tris[:,0,:]; p1=tris[:,1,:]; p2=tris[:,2,:]
    e1=p1-p0; e2=p2-p0; n=np.cross(e1,e2); nl=np.linalg.norm(n,axis=1,keepdims=True); valid=nl[:,0]>1e-9
    n[valid]/=nl[valid]; floor_mask=valid&(n[:,1]>0.3)
    if not floor_mask.any(): return None
    fp0=p0[floor_mask]; fp1=p1[floor_mask]; fp2=p2[floor_mask]; fn=n[floor_mask]
    denom=fn[:,1]; valid2=denom>0.01
    if not valid2.any(): return None
    op=fp0[valid2]-np.array([[rx,ry,rz]],np.float32)
    t_hit=(op*fn[valid2]).sum(axis=1)/denom[valid2]
    in_range=(t_hit>=-0.1)&(t_hit<=max_dist)
    if not in_range.any(): return None
    t_vals=t_hit[in_range]
    hit_pts=np.stack([np.full(len(t_vals),rx),ry-t_vals,np.full(len(t_vals),rz)],axis=1).astype(np.float32)
    v2idx=np.where(valid2)[0][in_range]; a=fp0[v2idx]; b=fp1[v2idx]; c=fp2[v2idx]
    inside=_point_in_tri_batch(hit_pts,a,b,c)
    if not inside.any(): return None
    return float((ry-t_vals[inside]).max())

def _raycast_hitscan(ox, oy, oz, dx, dy, dz, max_dist=400.0):
    """Returns distance to first geometry hit along ray, or None."""
    if TRI_GRID.tri_arr is None or len(TRI_GRID.tri_arr) == 0: return None
    d_len = math.sqrt(dx*dx + dy*dy + dz*dz)
    if d_len < 1e-9: return None
    dx /= d_len; dy /= d_len; dz /= d_len
    steps = [0.0, max_dist*0.25, max_dist*0.5, max_dist*0.75, max_dist]
    idx_set = set()
    for s in steps:
        px = ox + dx*s; pz = oz + dz*s
        for i in TRI_GRID.query_indices(px, pz, 6.0): idx_set.add(int(i))
    if not idx_set: return None
    idx = np.array(sorted(idx_set), dtype=np.int32)
    tris = TRI_GRID.tri_arr[idx]
    p0 = tris[:,0,:]; p1 = tris[:,1,:]; p2 = tris[:,2,:]
    e1 = p1-p0; e2 = p2-p0; n = np.cross(e1,e2)
    nl = np.linalg.norm(n, axis=1, keepdims=True); valid = nl[:,0] > 1e-9
    n[valid] /= nl[valid]
    rd = np.array([[dx,dy,dz]], np.float32)
    denom = (n*rd).sum(axis=1); front = valid & (np.abs(denom) > 1e-4)
    if not front.any(): return None
    ro = np.array([[ox,oy,oz]], np.float32)
    t_vals = ((p0[front]-ro)*n[front]).sum(axis=1) / denom[front]
    in_range = (t_vals > 0.05) & (t_vals <= max_dist)
    if not in_range.any(): return None
    t_sub = t_vals[in_range]
    hit_pts = ro + t_sub[:,None]*rd
    fp0 = p0[front][in_range]; fp1 = p1[front][in_range]; fp2 = p2[front][in_range]
    inside = _point_in_tri_batch(hit_pts, fp0, fp1, fp2)
    if not inside.any(): return None
    return float(t_sub[inside].min())

def collide_capsule_map(prev_pos, new_pos, radius, height, vy):
    if TRI_GRID.tri_arr is None or len(TRI_GRID.tri_arr)==0:
        return False,False,list(new_pos),False
    idx=TRI_GRID.query_indices(new_pos[0],new_pos[2],radius+1.5)
    if len(idx)==0: return False,False,list(new_pos),False
    return _collide_capsule(prev_pos,new_pos,radius,height,TRI_GRID.tri_arr[idx],vy)

# ══════════════════════════════════════════════════════════════════════════════
# MAP LOADING  (geometry only, no GL)
# ══════════════════════════════════════════════════════════════════════════════
aabb_min = None
aabb_max = None

def load_map(path):
    global aabb_min, aabb_max
    try:
        import pygltflib, base64, io as _io
    except ImportError:
        print("[SERVER] pygltflib not available — no collision"); return False
    if not os.path.exists(path):
        print(f"[SERVER] Map '{path}' not found — no collision"); return False
    print(f"[SERVER] Loading map '{path}' …")

    COMP = {5120:np.int8,5121:np.uint8,5122:np.int16,5123:np.uint16,5125:np.uint32,5126:np.float32}
    TNCO = {"SCALAR":1,"VEC2":2,"VEC3":3,"VEC4":4,"MAT2":4,"MAT3":9,"MAT4":16}

    def get_buf(gltf, bi):
        buf=gltf.buffers[bi]
        if buf.uri is None: return bytes(gltf.binary_blob())
        if buf.uri.startswith("data:"): _,enc=buf.uri.split(",",1); return base64.b64decode(enc)
        with open(os.path.join(os.path.dirname(os.path.abspath(path)), buf.uri),"rb") as f: return f.read()

    def read_acc(gltf, ai):
        acc=gltf.accessors[ai]; bvi=getattr(acc,"bufferViewIndex",None) or getattr(acc,"bufferView",None)
        if bvi is None: return np.zeros((acc.count,TNCO[acc.type]),dtype=np.float32)
        bv=gltf.bufferViews[bvi]; raw=get_buf(gltf,bv.buffer)
        dt=COMP[acc.componentType]; nc=TNCO[acc.type]; bo=bv.byteOffset or 0; ao=acc.byteOffset or 0
        sz=np.dtype(dt).itemsize*nc
        return np.frombuffer(raw[bo+ao:bo+ao+acc.count*sz],dtype=dt).reshape(acc.count,nc).astype(np.float32)

    def node_mat(node):
        if node.matrix is not None: return np.array(node.matrix,dtype=np.float32).reshape(4,4).T
        M=np.eye(4,dtype=np.float32)
        if node.scale: sx,sy,sz=node.scale; M[0,0]=sx; M[1,1]=sy; M[2,2]=sz
        if node.rotation:
            qx,qy,qz,qw=[float(v) for v in node.rotation]
            R=np.array([[1-2*(qy*qy+qz*qz),2*(qx*qy-qz*qw),2*(qx*qz+qy*qw),0],[2*(qx*qy+qz*qw),1-2*(qx*qx+qz*qz),2*(qy*qz-qx*qw),0],[2*(qx*qz-qy*qw),2*(qy*qz+qx*qw),1-2*(qx*qx+qy*qy),0],[0,0,0,1]],dtype=np.float32); M=R@M
        if node.translation:
            T=np.eye(4,dtype=np.float32); T[0,3],T[1,3],T[2,3]=[float(v) for v in node.translation]; M=T@M
        return M

    gltf=pygltflib.GLTF2().load(path)
    si=gltf.scene if gltf.scene is not None else 0
    roots=(gltf.scenes[si].nodes or []) if gltf.scenes else list(range(len(gltf.nodes)))
    stack=[(ni,np.eye(4,dtype=np.float32)) for ni in roots]
    triangles=[]; all_pts=[]
    while stack:
        ni,pm=stack.pop(); node=gltf.nodes[ni]; mat=pm@node_mat(node)
        for ci in (node.children or []): stack.append((ci,mat))
        if node.mesh is None: continue
        for prim in gltf.meshes[node.mesh].primitives:
            if prim.attributes.POSITION is None: continue
            pts_l=read_acc(gltf,prim.attributes.POSITION)
            ones=np.ones((len(pts_l),1),np.float32)
            pts_w=(mat@np.hstack([pts_l,ones]).T).T[:,:3]*MAP_SCALE
            if prim.indices is not None:
                idx=read_acc(gltf,prim.indices).flatten().astype(np.uint32)
            else:
                idx=np.arange(len(pts_w),dtype=np.uint32)
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
            elif mode!=4: continue
            for tri in pts_w[idx.reshape(-1,3)]: triangles.append(tri.tolist())
            all_pts.extend(pts_w.tolist())
    if not all_pts: print("[SERVER] No geometry"); return False
    av=np.array(all_pts,np.float32)
    aabb_min=av.min(axis=0).tolist(); aabb_max=av.max(axis=0).tolist()
    TRI_GRID.build(triangles)
    print(f"[SERVER] Map loaded — {len(triangles)} triangles")
    return True

# ══════════════════════════════════════════════════════════════════════════════
# SPAWN
# ══════════════════════════════════════════════════════════════════════════════
def spawn_pos():
    if aabb_min and aabb_max:
        mn,mx=aabb_min,aabb_max; cx=(mn[0]+mx[0])/2; cz=(mn[2]+mx[2])/2
        cast_y=mx[1]+2.0; hr=mx[1]-mn[1]
        offsets=[(0,0),(2,0),(-2,0),(0,2),(0,-2),(4,4),(-4,4),(4,-4),(-4,-4)]
        best=None
        for ox,oz in offsets:
            fy=_raycast_down(cx+ox,cast_y,cz+oz,hr+4.0)
            if fy is not None:
                c=[cx+ox,fy+P_H*0.6,cz+oz]
                if best is None or c[1]>best[1]: best=c
        if best: return best
        return [cx,mx[1]+6.0,cz]
    return [0.0,4.0,0.0]

# ══════════════════════════════════════════════════════════════════════════════
# SERVER-SIDE PLAYER  (same physics as original, no rendering)
# ══════════════════════════════════════════════════════════════════════════════
class ServerPlayer:
    def __init__(self, pid, name="Player"):
        self.pid      = pid
        self.name     = name
        self.pos      = list(spawn_pos())
        self.vel      = [0.0, 0.0, 0.0]
        self._prev    = list(self.pos)
        self.yaw      = 0.0
        self.pitch    = 0.0

        self.on_ground      = False
        self.coyote         = 0
        self.jump_held      = False
        self.jump_buffer    = 0
        self.grounded_frames= 0
        self.just_jumped    = False

        self.sliding  = False
        self.slide_t  = 0
        self.slide_cd = 0

        # Latest input from client
        self.inp = {}

        # Kill tracking
        self.health = 100
        self.kills  = 0
        self.deaths = 0
        self.alive  = True
        self.respawn_timer = 0

    def apply_input(self, inp):
        """Called once per tick with the latest input snapshot."""
        self.yaw   = float(inp.get("yaw",   self.yaw))
        self.pitch = float(inp.get("pitch", self.pitch))

        yr = math.radians(self.yaw)
        fw = [math.sin(yr), 0.0, -math.cos(yr)]
        rt = [math.cos(yr), 0.0,  math.sin(yr)]
        wx = wz = 0.0
        if inp.get("w"): wx+=fw[0]; wz+=fw[2]
        if inp.get("s"): wx-=fw[0]; wz-=fw[2]
        if inp.get("d"): wx+=rt[0]; wz+=rt[2]
        if inp.get("a"): wx-=rt[0]; wz-=rt[2]
        wlen=math.sqrt(wx*wx+wz*wz)
        if wlen>0: wx/=wlen; wz/=wlen

        if inp.get("jump_press"):
            self.jump_buffer=JUMP_BUFFER

        if inp.get("slide") and self.on_ground and self.slide_cd==0 and wlen>0 and not self.sliding:
            self.sliding=True; self.slide_t=SLIDE_DUR; self.slide_cd=SLIDE_CD
            self.vel[0]=wx*SLIDE_VEL; self.vel[2]=wz*SLIDE_VEL

        can_jump=self.on_ground or self.coyote>0
        if self.jump_buffer>0 and can_jump:
            self.vel[1]=JUMP_VEL; self.coyote=0; self.jump_held=True
            self.jump_buffer=0; self.grounded_frames=0; self.just_jumped=True; self.sliding=False; self.slide_t=0

        if self.jump_held and not inp.get("jump_held") and self.vel[1]>0:
            self.vel[1]*=JUMP_CUT; self.jump_held=False
        elif not inp.get("jump_held"):
            self.jump_held=False

        if self.sliding:
            self.vel[0]*=0.93; self.vel[2]*=0.93
        elif self.on_ground:
            vxz=[self.vel[0]*GROUND_FRICTION, self.vel[2]*GROUND_FRICTION]
            if wlen>0: vxz=pm_accelerate(vxz,[wx,wz],MAX_GROUND_SPD,GROUND_ACCEL)
            self.vel[0]=vxz[0]; self.vel[2]=vxz[1]
        else:
            if wlen>0:
                vxz=pm_accelerate([self.vel[0],self.vel[2]],[wx,wz],MAX_AIR_WISH,AIR_ACCEL)
                self.vel[0]=vxz[0]; self.vel[2]=vxz[1]

    def update(self):
        if not self.alive:
            self.respawn_timer -= 1
            if self.respawn_timer <= 0:
                self.alive=True; self.health=100
                self.pos=list(spawn_pos()); self.vel=[0,0,0]
            return

        if self.slide_t>0: self.slide_t-=1
        else: self.sliding=False
        if self.slide_cd>0: self.slide_cd-=1
        if self.coyote>0: self.coyote-=1
        if self.jump_buffer>0: self.jump_buffer-=1

        self.vel[1]-=GRAVITY
        self._prev=list(self.pos)
        new_pos=[self.pos[0]+self.vel[0], self.pos[1]+self.vel[1], self.pos[2]+self.vel[2]]
        prev_g=self.on_ground; self.on_ground=False; self.on_wall=False

        if TRI_GRID.tri_arr is not None and len(TRI_GRID.tri_arr)>0:
            dx=new_pos[0]-self._prev[0]; dz=new_pos[2]-self._prev[2]
            dy_=new_pos[1]-self._prev[1]
            dist3d=math.sqrt(dx*dx+dy_*dy_+dz*dz)
            n_steps=max(1,int(math.ceil(dist3d/(P_R*0.5))))
            cur=list(self._prev); on_g=False; on_w=False
            for s in range(n_steps):
                frac=(s+1)/n_steps
                sub=[self._prev[0]+dx*frac, self._prev[1]+dy_*frac, self._prev[2]+dz*frac]
                on_g,on_w,cur,hit_ceil=collide_capsule_map(cur,sub,P_R,P_H,self.vel[1])
                if hit_ceil: self.vel[1]=min(self.vel[1],0.0); self.jump_held=False; self.jump_buffer=0
                if on_g and self.vel[1]<0: self.vel[1]=0.0
                if on_w and not on_g:
                    if prev_g or on_g:
                        ls=[cur[0],cur[1]+STEP_UP,cur[2]]; ld=[sub[0],cur[1]+STEP_UP,sub[2]]
                        og2,ow2,cl2,_=collide_capsule_map(ls,ld,P_R,P_H,0.0)
                        if not ow2:
                            fy=_raycast_down(cl2[0],cl2[1],cl2[2],STEP_UP+0.1)
                            if fy is not None: cl2[1]=fy
                            cur=cl2; on_g=True; on_w=False; continue
                    self.vel[0]=0.0; self.vel[2]=0.0; break
            self.pos=cur
            if on_g: self.on_ground=True
            self.on_wall=on_w

            if self.on_ground:
                self.grounded_frames=2; self.just_jumped=False
            else:
                if self.just_jumped: self.grounded_frames=0
                elif self.grounded_frames>0:
                    self.grounded_frames-=1
                    if self.grounded_frames==1 and -0.015<self.vel[1]<=0.0:
                        spd_h=math.sqrt(self.vel[0]**2+self.vel[2]**2)
                        if spd_h>0.001:
                            fy=_raycast_down(self.pos[0],self.pos[1],self.pos[2],STEP_DOWN)
                            if fy is not None and 0.001<(self.pos[1]-fy)<=STEP_DOWN:
                                self.pos[1]=fy; self.on_ground=True; self.vel[1]=0.0
        else:
            self.pos=new_pos
            if self.pos[1]<0.0 and self.vel[1]<=0: self.pos[1]=0.0; self.vel[1]=0.0; self.on_ground=True

        if prev_g and not self.on_ground: self.coyote=COYOTE_FRAMES
        self.vel[1]=clamp(self.vel[1],-2.0,JUMP_VEL*1.1)

        floor_y=(aabb_min[1] if aabb_min else 0.0)-30
        if self.pos[1]<floor_y:
            self.alive=False; self.deaths+=1; self.respawn_timer=120
            self.health=0

    def take_damage(self, amount):
        if not self.alive: return
        self.health=max(0,self.health-amount)
        if self.health==0:
            self.alive=False; self.deaths+=1; self.respawn_timer=120

    def snapshot(self):
        return {
            "pid":   self.pid,
            "name":  self.name,
            "x":     round(self.pos[0],3),
            "y":     round(self.pos[1],3),
            "z":     round(self.pos[2],3),
            "yaw":   round(self.yaw,2),
            "pitch": round(self.pitch,2),
            "vel_y": round(self.vel[1],3),
            "sliding": self.sliding,
            "on_ground": self.on_ground,
            "health": self.health,
            "alive":  self.alive,
            "kills":  self.kills,
            "deaths": self.deaths,
        }

# ══════════════════════════════════════════════════════════════════════════════
# HIT EVENTS  (bullet hits processed server-side)
# ══════════════════════════════════════════════════════════════════════════════
BULLET_DAMAGE = 25

def process_shoot(shooter: ServerPlayer, players: dict, yaw: float, pitch: float) -> list:
    """Server-side hitscan against other players' capsules, with wall occlusion check."""
    yr=math.radians(yaw); pr=math.radians(pitch)
    dx=math.cos(pr)*math.sin(yr); dy=math.sin(pr); dz=-math.cos(pr)*math.cos(yr)
    eye=[shooter.pos[0], shooter.pos[1]+EYE*0.9, shooter.pos[2]]
    hits=[]
    for pid,p in players.items():
        if pid==shooter.pid or not p.alive: continue
        cx,cy,cz=p.pos; cap_r=P_R*1.1; cap_h=P_H
        ox=eye[0]-cx; oz=eye[2]-cz
        a=dx*dx+dz*dz; b=2*(ox*dx+oz*dz); c_=ox*ox+oz*oz-cap_r*cap_r
        disc=b*b-4*a*c_
        if disc<0 or a<1e-9: continue
        t=(-b-math.sqrt(disc))/(2*a)
        if t<0: t=(-b+math.sqrt(disc))/(2*a)
        if t<0 or t>300: continue
        hy=eye[1]+dy*t
        if cy<=hy<=cy+cap_h:
            # Wall occlusion check — cast ray from eye toward target,
            # reject if geometry is closer than the player capsule
            wall_t = _raycast_hitscan(eye[0], eye[1], eye[2], dx, dy, dz, t + 0.5)
            if wall_t is not None and wall_t < t - 0.3:
                continue  # wall is in the way
            hits.append((t, pid))
    hits.sort(); hit_pids=[]
    for t,pid in hits:
        players[pid].take_damage(BULLET_DAMAGE)
        hit_pids.append(pid)
        if players[pid].health<=0: shooter.kills+=1
        break
    return hit_pids

# ══════════════════════════════════════════════════════════════════════════════
# SERVER STATE
# ══════════════════════════════════════════════════════════════════════════════
players   : dict[str, ServerPlayer] = {}   # ws_id → player
websockets_map : dict[str, object]  = {}   # ws_id → websocket
next_pid  = 1

events_queue = []   # server → all clients (kills, hits, etc)

def next_player_id():
    global next_pid
    pid=next_pid; next_pid+=1; return str(pid)

async def broadcast(msg: dict):
    data=json.dumps(msg)
    dead=[]
    for wsid,ws in list(websockets_map.items()):
        try:
            await ws.send(data)
        except Exception:
            dead.append(wsid)
    for wsid in dead:
        websockets_map.pop(wsid,None)

# ══════════════════════════════════════════════════════════════════════════════
# WEBSOCKET HANDLER
# ══════════════════════════════════════════════════════════════════════════════
async def handler(ws):
    wsid=id(ws); wsid=str(wsid)
    pid=next_player_id()
    name=f"Player{pid}"
    player=ServerPlayer(pid, name)
    players[pid]=player; websockets_map[wsid]=ws
    print(f"[+] {name} connected  (total: {len(players)})")

    # Send welcome
    await ws.send(json.dumps({
        "type":   "welcome",
        "your_pid": pid,
        "map":    os.path.basename(gltf_path),
        "spawn":  player.pos,
        "tick_rate": TICK_RATE,
    }))

    try:
        async for raw in ws:
            try:
                msg=json.loads(raw)
            except json.JSONDecodeError:
                continue
            mtype=msg.get("type","")

            if mtype=="input":
                player.inp=msg
                # Trust client-reported position for rendering/broadcast
                if "x" in msg:
                    player.pos[0] = float(msg["x"])
                    player.pos[1] = float(msg["y"])
                    player.pos[2] = float(msg["z"])

            elif mtype=="shoot":
                hit_pids=process_shoot(player, players,
                                       float(msg.get("yaw",player.yaw)),
                                       float(msg.get("pitch",player.pitch)))
                if hit_pids:
                    events_queue.append({
                        "type":    "hit",
                        "shooter": pid,
                        "targets": hit_pids,
                    })
                    for hpid in hit_pids:
                        hp=players.get(hpid)
                        if hp and not hp.alive:
                            events_queue.append({
                                "type":    "kill",
                                "killer":  pid,
                                "victim":  hpid,
                                "killer_name": player.name,
                                "victim_name": hp.name,
                            })

            elif mtype=="name":
                player.name=str(msg.get("name",""))[:20]

    except Exception:
        pass
    finally:
        print(f"[-] {name} disconnected")
        players.pop(pid, None)
        websockets_map.pop(wsid, None)

# ══════════════════════════════════════════════════════════════════════════════
# PHYSICS + BROADCAST LOOP
# ══════════════════════════════════════════════════════════════════════════════
async def game_loop():
    tick_dt  = 1.0 / TICK_RATE
    bcast_dt = 1.0 / BROADCAST_RATE
    last_bcast = time.time()

    while True:
        t0=time.time()

        # Physics tick — apply_input runs every tick so friction and accel stay balanced
        for p in list(players.values()):
            if p.inp:
                p.apply_input(p.inp)
            p.update()

        # Broadcast snapshot
        if time.time()-last_bcast >= bcast_dt:
            last_bcast=time.time()
            snap={
                "type":    "state",
                "tick":    int(time.time()*1000),
                "players": [p.snapshot() for p in players.values()],
                "events":  list(events_queue),
            }
            events_queue.clear()
            await broadcast(snap)

        elapsed=time.time()-t0
        await asyncio.sleep(max(0, tick_dt-elapsed))

# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
gltf_path = "ex.gltf"

async def main_async(host, port):
    print(f"[SERVER] Starting on ws://{host}:{port}")
    async with serve(handler, host, port):
        await game_loop()

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("map", nargs="?", default="ex.gltf")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args=parser.parse_args()
    gltf_path=args.map
    load_map(gltf_path)
    asyncio.run(main_async(args.host, args.port))