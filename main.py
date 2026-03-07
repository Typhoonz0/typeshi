"""
GLTF FPS Walker  —  Movement Shooter Edition
  pip install pygame PyOpenGL PyOpenGL_accelerate pygltflib numpy

  Place ex.gltf (+ any .bin files) next to this script.
  python gltf_fps.py [yourmap.gltf]
  python gltf_fps.py --debug    (inspect your GLTF)

Controls:
  WASD        Move / Air-strafe (Quake-style)
  Mouse       Look  (click window to capture)
  Space       Jump / double-jump  (tap = low, hold = high)
  Shift       Dash
  Ctrl        Slide
  F           Wireframe
  ESC         Release mouse / quit
"""

import pygame, sys, math, random, os, ctypes, base64, io
import numpy as np

try:
    from OpenGL.GL  import *
    from OpenGL.GLU import *
except ImportError:
    print("Install: pip install pygame PyOpenGL PyOpenGL_accelerate"); sys.exit(1)

try:
    import pygltflib
    _HAS_GLTF = True
except ImportError:
    print("[WARN] pygltflib missing — pip install pygltflib numpy")
    _HAS_GLTF = False

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG  —  tuned for movement-shooter feel
# ══════════════════════════════════════════════════════════════════════════════
W, H  = 1280, 720
FPS   = 120

# --- Physics -----------------------------------------------------------------
# All velocities are in units/frame at 120 FPS.
# GRAVITY 0.004 => ~0.48 u/s^2, terminal ~3.5 u/s after ~1s — feels like Quake
GRAVITY        = 0.004        # per-frame gravity  (was 0.022 — way too fast)
JUMP_VEL       = 0.14         # upward vel on jump  (~16.8 u/s at 120fps)
JUMP_CUT       = 0.50         # vel multiplier on early space release
JUMP_BUFFER    = 8            # frames jump input is remembered before landing
MAX_Y          = 2024.0       # hard ceiling — edit this to suit your map
COYOTE_FRAMES  = 7            # frames you can still jump after walking off edge

# --- Ground movement ---------------------------------------------------------
# GROUND_ACCEL 0.018/frame => reach MAX in ~9 frames — snappy not twitchy
# GROUND_FRICTION 0.78/frame at 120fps: 0.78^6 ~ 0.22 => stops in ~6 frames
GROUND_ACCEL   = 0.018        # vel added per frame toward wish dir
GROUND_FRICTION= 0.78         # per-frame multiplier while grounded
MAX_GROUND_SPD = 0.12         # top ground speed (units/frame)

# --- Air movement ------------------------------------------------------------
AIR_ACCEL      = 0.006        # vel added per frame in air
MAX_AIR_WISH   = 0.12         # air strafe wish speed cap

# --- Abilities ---------------------------------------------------------------
SLIDE_VEL      = 0.22
SLIDE_DUR      = 24
SLIDE_CD       = 36
WALL_JUMP_V    = 0.0
WALL_JUMP_H    = 0.0
DASH_VEL       = 0.16
DASH_DUR       = 10
DASH_CD        = 48

# --- Camera ------------------------------------------------------------------
MOUSE_SENS     = 0.10
BASE_FOV       = 100.0
SPEED_FOV_ADD  = 14.0         # extra FOV at max sprint
FOV_LERP       = 0.10

# --- Arena / Visuals ---------------------------------------------------------
MAP_SCALE = 2.0          # increase to make the GLTF map bigger, e.g. 1.5, 2.0, 3.0
ARENA_W = 48.0
ARENA_D = 48.0
FALLBACK_PLATFORMS = [
    ( 0.0,  0.0, 48.0, 48.0,  0.0),
    (-10.0,-10.0,  8.0,  4.0,  2.0),
    ( 10.0, 10.0,  8.0,  4.0,  2.0),
    (  0.0,  0.0,  6.0,  6.0,  4.5),
    (-14.0, 14.0,  7.0,  4.0,  3.5),
    ( 14.0,-14.0,  7.0,  4.0,  3.5),
    ( -6.0, 10.0,  5.0,  3.0,  6.0),
    (  6.0,-10.0,  5.0,  3.0,  6.0),
    (  0.0,-18.0,  6.0,  4.0,  2.5),
    (  0.0, 18.0,  6.0,  4.0,  2.5),
    ( 18.0,  0.0,  4.0,  6.0,  2.5),
    (-18.0,  0.0,  4.0,  6.0,  2.5),
]

SKY_TOP  = (0.03, 0.02, 0.10)
SKY_BOT  = (0.08, 0.04, 0.20)
FLOOR_COL= (0.14, 0.12, 0.22)
PLAT_COL = (0.20, 0.18, 0.35)
WALL_COL = (0.10, 0.08, 0.20)
GRID_COL = (0.22, 0.18, 0.38)

# ══════════════════════════════════════════════════════════════════════════════
# MATH
# ══════════════════════════════════════════════════════════════════════════════
def v3(x,y,z):      return [float(x),float(y),float(z)]
def vadd(a,b):      return [a[0]+b[0],a[1]+b[1],a[2]+b[2]]
def vsub(a,b):      return [a[0]-b[0],a[1]-b[1],a[2]-b[2]]
def vscale(a,s):    return [a[0]*s,a[1]*s,a[2]*s]
def vlen(a):        return math.sqrt(a[0]**2+a[1]**2+a[2]**2)
def vnorm(a):
    l=vlen(a); return [a[0]/l,a[1]/l,a[2]/l] if l>1e-9 else [0.0,0.0,0.0]
def vdot(a,b):      return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
def clamp(v,lo,hi): return max(lo,min(hi,v))
def lerp(a,b,t):    return a+(b-a)*t

# ══════════════════════════════════════════════════════════════════════════════
# SPATIAL GRID
# ══════════════════════════════════════════════════════════════════════════════
class TriGrid:
    def __init__(self, cell=3.0):
        self.cell = cell
        self.buckets = {}
        self.tri_arr = None

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
        self._np_buckets = {k: np.array(v, dtype=np.int32) for k,v in self.buckets.items()}

    def query_indices(self, px, pz, radius):
        r   = int(math.ceil(radius / self.cell)) + 1
        gx  = int(math.floor(px / self.cell))
        gz  = int(math.floor(pz / self.cell))
        parts = []
        for dx in range(-r, r+1):
            for dz in range(-r, r+1):
                arr = self._np_buckets.get((gx+dx, gz+dz))
                if arr is not None: parts.append(arr)
        if not parts: return np.empty(0, dtype=np.int32)
        return np.unique(np.concatenate(parts))

TRI_GRID = TriGrid()

def _raycast_down(rx, ry, rz, max_dist):
    """Cast a ray straight down from (rx,ry,rz), return floor Y or None."""
    if TRI_GRID.tri_arr is None or len(TRI_GRID.tri_arr) == 0:
        return None
    idx = TRI_GRID.query_indices(rx, rz, P_R + 1.0)
    if len(idx) == 0:
        return None
    tris = TRI_GRID.tri_arr[idx]
    p0 = tris[:,0,:]; p1 = tris[:,1,:]; p2 = tris[:,2,:]
    e1 = p1-p0; e2 = p2-p0
    n = np.cross(e1, e2)
    nl = np.linalg.norm(n, axis=1, keepdims=True)
    valid = nl[:,0] > 1e-9
    n[valid] /= nl[valid]
    # Only upward-facing tris
    floor_mask = valid & (n[:,1] > 0.3)
    if not floor_mask.any():
        return None
    fp0 = p0[floor_mask]; fp1 = p1[floor_mask]; fp2 = p2[floor_mask]
    fn  = n[floor_mask]
    # Ray: origin=(rx,ry,rz), dir=(0,-1,0)
    # Plane intersect: t = dot(n, p0-origin) / dot(n, dir)
    # dir=(0,-1,0) so dot(n,dir) = -ny
    denom = fn[:,1]  # ny component
    valid2 = denom > 0.01  # ray going into upward face
    if not valid2.any():
        return None
    op = fp0[valid2] - np.array([[rx, ry, rz]], np.float32)
    t_hit = (op * fn[valid2]).sum(axis=1) / denom[valid2]
    # t_hit is distance downward — must be positive and within max_dist
    in_range = (t_hit >= -0.1) & (t_hit <= max_dist)
    if not in_range.any():
        return None
    # Check if hit point is inside the triangle
    t_vals = t_hit[in_range]
    hit_pts = np.stack([np.full(len(t_vals), rx),
                        ry - t_vals,
                        np.full(len(t_vals), rz)], axis=1).astype(np.float32)
    v2idx = np.where(valid2)[0][in_range]
    a = fp0[v2idx]; b = fp1[v2idx]; c = fp2[v2idx]
    inside = _point_in_tri_batch(hit_pts, a, b, c)
    if not inside.any():
        return None
    # Return highest floor hit (shallowest step down)
    best_y = float((ry - t_vals[inside]).max())
    return best_y

# ══════════════════════════════════════════════════════════════════════════════
# VECTORISED CAPSULE COLLISION
# ══════════════════════════════════════════════════════════════════════════════
def _collide_capsule_fast(prev_pos, new_pos, radius, height, tri_arr, vy=0.0):
    rx, ry, rz = float(new_pos[0]), float(new_pos[1]), float(new_pos[2])
    on_ground = False; on_wall = False
    if len(tri_arr) == 0: return on_ground, on_wall, [rx, ry, rz]

    p0 = tri_arr[:,0,:]; p1 = tri_arr[:,1,:]; p2 = tri_arr[:,2,:]
    e1 = p1-p0; e2 = p2-p0; n = np.cross(e1, e2)
    nl = np.linalg.norm(n, axis=1, keepdims=True)
    valid = (nl[:,0] > 1e-9)
    n[valid] /= nl[valid]
    ny = n[:,1]

    # ── FLOOR ────────────────────────────────────────────────────────────────
    floor_mask = valid & (ny > 0.4)
    if floor_mask.any() and vy <= 0.01:
        fp0=p0[floor_mask]; fp1=p1[floor_mask]; fp2=p2[floor_mask]
        foot = np.array([[rx, ry, rz]], np.float32)
        cp = _closest_pts_tris_batch(foot, fp0, fp1, fp2)
        dxz = np.sqrt((rx-cp[:,0])**2 + (rz-cp[:,2])**2)
        hit = (dxz <= radius+0.3) & (cp[:,1] <= ry+0.1) & (cp[:,1] >= ry-0.5)
        if hit.any():
            best_y = float(cp[hit,1].max())
            if best_y <= ry + 0.15:
                ry = best_y; on_ground = True

    # ── WALLS — proper 3D sphere vs triangle ────────────────────────────────
    # Use full 3D closest point distance, iterate until settled.
    # Spine = capsule axis, sampled at multiple heights.
    wall_mask = valid & (np.abs(ny) < 0.5)
    if wall_mask.any():
        wp0 = p0[wall_mask]; wp1 = p1[wall_mask]; wp2 = p2[wall_mask]
        wn  = n[wall_mask]

        for _iter in range(12):
            any_pen = False
            for t in (0.1, 0.3, 0.5, 0.7, 0.9):
                sy = ry + height * t
                pt = np.array([[rx, sy, rz]], np.float32)
                cp = _closest_pts_tris_batch(pt, wp0, wp1, wp2)

                # Full 3D distance from spine point to closest point on tri
                dx3 = rx - cp[:,0]; dy3 = sy - cp[:,1]; dz3 = rz - cp[:,2]
                dist3 = np.sqrt(dx3*dx3 + dy3*dy3 + dz3*dz3)

                pen = dist3 < radius
                if not pen.any():
                    continue

                # Deepest first
                depths = np.where(pen, radius - dist3, -1.0)
                i = int(np.argmax(depths))
                d = float(dist3[i])
                ox = float(dx3[i]); oz = float(dz3[i])
                dxz = math.sqrt(ox*ox + oz*oz)

                if dxz < 1e-6:
                    # Pushing straight along face normal XZ
                    fnx = float(wn[i,0]); fnz = float(wn[i,2])
                    fnl = math.sqrt(fnx*fnx + fnz*fnz)
                    if fnl > 1e-6:
                        rx += fnx/fnl * (radius - d + 0.005)
                        rz += fnz/fnl * (radius - d + 0.005)
                else:
                    ov = radius - d + 0.005
                    rx += (ox / dxz) * ov
                    rz += (oz / dxz) * ov

                on_wall = True
                any_pen = True
            if not any_pen:
                break

    return on_ground, on_wall, [rx, ry, rz]

def _point_in_tri_batch(pts, a, b, c):
    """Test if each point pts[i] is inside triangle a[i],b[i],c[i]. Returns bool array."""
    # Barycentric method
    v0 = c - a; v1 = b - a; v2 = pts - a
    dot00 = (v0*v0).sum(axis=1); dot01 = (v0*v1).sum(axis=1)
    dot02 = (v0*v2).sum(axis=1); dot11 = (v1*v1).sum(axis=1)
    dot12 = (v1*v2).sum(axis=1)
    denom = dot00*dot11 - dot01*dot01
    safe  = np.abs(denom) > 1e-12
    inv   = np.where(safe, 1.0/np.where(safe, denom, 1.0), 0.0)
    u = (dot11*dot02 - dot01*dot12) * inv
    v = (dot00*dot12 - dot01*dot02) * inv
    return (u >= -0.01) & (v >= -0.01) & (u + v <= 1.02)

def _closest_pts_tris_batch(p,a,b,c):
    ab=b-a; ac=c-a; ap=p-a
    d1=(ab*ap).sum(axis=1); d2=(ac*ap).sum(axis=1)
    bp=p-b; d3=(ab*bp).sum(axis=1); d4=(ac*bp).sum(axis=1)
    cp_=p-c; d5=(ab*cp_).sum(axis=1); d6=(ac*cp_).sum(axis=1)
    vc=d1*d4-d3*d2; vb=d5*d2-d1*d6; va=d3*d6-d5*d4
    denom=va+vb+vc; safe=np.abs(denom)>1e-12
    v=np.where(safe,vb/np.where(safe,denom,1.0),0.0)
    w=np.where(safe,vc/np.where(safe,denom,1.0),0.0)
    result=a+v[:,None]*ab+w[:,None]*ac
    m=(d1<=0)&(d2<=0);             result[m]=a[m]
    m=(d3>=0)&(d4<=d3);            result[m]=b[m]
    m=(d6>=0)&(d5<=d6);            result[m]=c[m]
    m=(vc<=0)&(d1>=0)&(d3<=0); tv=np.where(d1-d3>1e-12,d1/(d1-d3+1e-12),0.0); result[m]=(a+tv[:,None]*ab)[m]
    m=(vb<=0)&(d2>=0)&(d6<=0); tw=np.where(d2-d6>1e-12,d2/(d2-d6+1e-12),0.0); result[m]=(a+tw[:,None]*ac)[m]
    m=(va<=0)&((d4-d3)>=0)&((d5-d6)>=0)
    denom2=(d4-d3)+(d5-d6); tw2=np.where(np.abs(denom2)>1e-12,(d4-d3)/np.where(np.abs(denom2)>1e-12,denom2,1.0),0.0)
    result[m]=(b+tw2[:,None]*(c-b))[m]
    return result

# ══════════════════════════════════════════════════════════════════════════════
# GLTF LOADER
# ══════════════════════════════════════════════════════════════════════════════
COMP_DTYPE={5120:np.int8,5121:np.uint8,5122:np.int16,5123:np.uint16,5125:np.uint32,5126:np.float32}
TYPE_NCOMP={"SCALAR":1,"VEC2":2,"VEC3":3,"VEC4":4,"MAT2":4,"MAT3":9,"MAT4":16}

class GLTFMap:
    def __init__(self):
        self._cpu_meshes=[]; self.triangles=[]; self.aabb_min=None; self.aabb_max=None
        self.parsed=False; self.parse_error=""; self.draw_calls=[]; self.uploaded=False

    def _get_buffer_bytes(self,gltf,buf_idx,gltf_path):
        buf=gltf.buffers[buf_idx]
        if buf.uri is None:
            blob=gltf.binary_blob()
            if blob is None: raise RuntimeError("GLB binary blob is None")
            return bytes(blob)
        uri=buf.uri
        if uri.startswith("data:"):
            _,enc=uri.split(",",1); return base64.b64decode(enc)
        base_dir=os.path.dirname(os.path.abspath(gltf_path))
        bin_path=os.path.join(base_dir,uri)
        if not os.path.exists(bin_path): raise FileNotFoundError(f"Not found: {bin_path}")
        with open(bin_path,"rb") as f: return f.read()

    def _read_accessor(self,gltf,acc_idx,gltf_path):
        acc=gltf.accessors[acc_idx]
        bv_idx=getattr(acc,'bufferViewIndex',None)
        if bv_idx is None: bv_idx=getattr(acc,'bufferView',None)
        if bv_idx is None: return np.zeros((acc.count,TYPE_NCOMP[acc.type]),dtype=np.float32)
        bv=gltf.bufferViews[bv_idx]; raw=self._get_buffer_bytes(gltf,bv.buffer,gltf_path)
        dtype=COMP_DTYPE[acc.componentType]; n_comp=TYPE_NCOMP[acc.type]; count=acc.count
        bv_off=bv.byteOffset or 0; acc_off=acc.byteOffset or 0
        stride=bv.byteStride; item_sz=np.dtype(dtype).itemsize*n_comp
        if stride is None or stride==item_sz:
            start=bv_off+acc_off; end=start+count*item_sz
            return np.frombuffer(raw[start:end],dtype=dtype).reshape(count,n_comp).astype(np.float32)
        out=np.zeros((count,n_comp),dtype=np.float32)
        for i in range(count):
            off=bv_off+acc_off+i*stride
            out[i]=np.frombuffer(raw[off:off+item_sz],dtype=dtype).astype(np.float32)
        return out

    @staticmethod
    def _node_matrix(node):
        if node.matrix is not None:
            return np.array(node.matrix,dtype=np.float32).reshape(4,4).T
        M=np.eye(4,dtype=np.float32)
        if node.scale is not None:
            sx,sy,sz=node.scale; M[0,0]=sx; M[1,1]=sy; M[2,2]=sz
        if node.rotation is not None:
            qx,qy,qz,qw=[float(v) for v in node.rotation]
            R=np.array([[1-2*(qy*qy+qz*qz),2*(qx*qy-qz*qw),2*(qx*qz+qy*qw),0],
                        [2*(qx*qy+qz*qw),1-2*(qx*qx+qz*qz),2*(qy*qz-qx*qw),0],
                        [2*(qx*qz-qy*qw),2*(qy*qz+qx*qw),1-2*(qx*qx+qy*qy),0],
                        [0,0,0,1]],dtype=np.float32)
            M=R@M
        if node.translation is not None:
            T=np.eye(4,dtype=np.float32); T[0,3],T[1,3],T[2,3]=[float(v) for v in node.translation]
            M=T@M
        return M

    @staticmethod
    def _transform_points(pts,mat):
        ones=np.ones((len(pts),1),dtype=np.float32)
        return (mat@np.hstack([pts.astype(np.float32),ones]).T).T[:,:3]

    def _load_texture(self,gltf,tex_idx,gltf_path):
        if not gltf.textures or tex_idx>=len(gltf.textures): return None
        tex=gltf.textures[tex_idx]; src_idx=tex.source
        if src_idx is None or not gltf.images or src_idx>=len(gltf.images): return None
        src=gltf.images[src_idx]; base=os.path.dirname(os.path.abspath(gltf_path))
        try:
            if src.uri:
                if src.uri.startswith("data:"):
                    _,enc=src.uri.split(",",1); img_bytes=base64.b64decode(enc)
                else:
                    with open(os.path.join(base,src.uri),"rb") as f: img_bytes=f.read()
                surf=pygame.image.load(io.BytesIO(img_bytes))
            else:
                bv_idx=getattr(src,'bufferView',None)
                if bv_idx is None: return None
                bv=gltf.bufferViews[bv_idx]; raw=self._get_buffer_bytes(gltf,bv.buffer,gltf_path)
                surf=pygame.image.load(io.BytesIO(raw[bv.byteOffset:bv.byteOffset+bv.byteLength]))
            surf=pygame.transform.flip(surf,False,True).convert_alpha()
            data=pygame.image.tostring(surf,"RGBA",False)
            tex_id=int(glGenTextures(1)); glBindTexture(GL_TEXTURE_2D,tex_id)
            glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,surf.get_width(),surf.get_height(),0,GL_RGBA,GL_UNSIGNED_BYTE,data)
            glGenerateMipmap(GL_TEXTURE_2D)
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_REPEAT)
            glBindTexture(GL_TEXTURE_2D,0)
            return tex_id
        except Exception as e: print(f"[GLTF] Texture error: {e}"); return None

    def parse(self,path):
        self.parsed=False; self.parse_error=""; self._cpu_meshes=[]; self.triangles=[]
        if not _HAS_GLTF: self.parse_error="pygltflib not installed"; return False
        if not os.path.exists(path): self.parse_error=f"'{path}' not found"; return False
        print(f"[GLTF] Parsing '{path}' …")
        try: gltf=pygltflib.GLTF2().load(path)
        except Exception as e: self.parse_error=f"Load failed: {e}"; return False
        print(f"[GLTF]  scenes:{len(gltf.scenes)} nodes:{len(gltf.nodes)} meshes:{len(gltf.meshes)}")
        scene_idx=gltf.scene if gltf.scene is not None else 0
        root_nodes=(gltf.scenes[scene_idx].nodes or []) if gltf.scenes else list(range(len(gltf.nodes)))
        stack=[(ni,np.eye(4,dtype=np.float32)) for ni in root_nodes]
        all_pts=[]; n_prims=0
        while stack:
            ni,pm=stack.pop(); node=gltf.nodes[ni]; mat=pm@self._node_matrix(node)
            for ci in (node.children or []): stack.append((ci,mat))
            if node.mesh is None: continue
            for prim in gltf.meshes[node.mesh].primitives:
                try:
                    r=self._parse_primitive(gltf,prim,mat,path)
                    if r:
                        vdata,idata,col,pts,tex_idx=r
                        self._cpu_meshes.append({"vdata":vdata,"idata":idata,"col":col,"tex_idx":tex_idx,"gltf":gltf,"path":path})
                        for tri in pts[idata.reshape(-1,3)]: self.triangles.append(tri.tolist())
                        all_pts.extend(pts.tolist()); n_prims+=1
                except Exception as e:
                    import traceback; traceback.print_exc()
        if not all_pts: self.parse_error="No geometry found"; print(f"[GLTF] {self.parse_error}"); return False
        av=np.array(all_pts,dtype=np.float32)
        self.aabb_min=av.min(axis=0).tolist(); self.aabb_max=av.max(axis=0).tolist()
        print(f"[GLTF] OK — {n_prims} prims, {len(self.triangles)} tris")
        TRI_GRID.build(self.triangles)
        self.parsed=True; return True

    def _parse_primitive(self,gltf,prim,world_mat,path):
        attrs=prim.attributes
        if attrs.POSITION is None: return None
        pos_local=self._read_accessor(gltf,attrs.POSITION,path)
        pos_world=self._transform_points(pos_local,world_mat) * MAP_SCALE
        if attrs.NORMAL is not None:
            nor_local=self._read_accessor(gltf,attrs.NORMAL,path)
            try: nm=np.linalg.inv(world_mat[:3,:3]).T; nor_world=(nm@nor_local.T).T
            except np.linalg.LinAlgError: nor_world=nor_local
            nlen=np.linalg.norm(nor_world,axis=1,keepdims=True)+1e-9
            nor_world=(nor_world/nlen).astype(np.float32)
        else:
            nor_world=np.tile([0.0,1.0,0.0],(len(pos_world),1)).astype(np.float32)
        uvs=self._read_accessor(gltf,attrs.TEXCOORD_0,path) if attrs.TEXCOORD_0 is not None else np.zeros((len(pos_world),2),dtype=np.float32)
        if prim.indices is not None:
            indices=self._read_accessor(gltf,prim.indices,path).flatten().astype(np.uint32)
        else:
            indices=np.arange(len(pos_world),dtype=np.uint32)
        mode=prim.mode if prim.mode is not None else 4
        if mode==5:
            idx2=[]
            for i in range(len(indices)-2):
                if i%2==0: idx2.extend([indices[i],indices[i+1],indices[i+2]])
                else: idx2.extend([indices[i+1],indices[i],indices[i+2]])
            indices=np.array(idx2,dtype=np.uint32)
        elif mode==6:
            idx2=[]
            for i in range(1,len(indices)-1): idx2.extend([indices[0],indices[i],indices[i+1]])
            indices=np.array(idx2,dtype=np.uint32)
        elif mode!=4: return None
        col=(0.55,0.50,0.70); tex_idx=None
        if prim.material is not None and gltf.materials:
            try:
                m=gltf.materials[prim.material]; pbr=m.pbrMetallicRoughness
                if pbr:
                    if pbr.baseColorFactor:
                        bf=pbr.baseColorFactor; col=(float(bf[0]),float(bf[1]),float(bf[2]))
                    if pbr.baseColorTexture and gltf.textures: tex_idx=pbr.baseColorTexture.index
            except: pass
        vdata=np.hstack([pos_world,nor_world,uvs]).astype(np.float32)
        return vdata,indices,col,pos_world,tex_idx

    def upload(self):
        if not self.parsed: return False
        self.draw_calls=[]; n_ok=0
        for m in self._cpu_meshes:
            try:
                vflat=m["vdata"].flatten().astype(np.float32); iflat=m["idata"].flatten().astype(np.uint32)
                vbo=int(glGenBuffers(1)); glBindBuffer(GL_ARRAY_BUFFER,vbo)
                glBufferData(GL_ARRAY_BUFFER,vflat.nbytes,vflat.tobytes(),GL_STATIC_DRAW); glBindBuffer(GL_ARRAY_BUFFER,0)
                ebo=int(glGenBuffers(1)); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,ebo)
                glBufferData(GL_ELEMENT_ARRAY_BUFFER,iflat.nbytes,iflat.tobytes(),GL_STATIC_DRAW); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,0)
                tex_id=None
                if m["tex_idx"] is not None: tex_id=self._load_texture(m["gltf"],m["tex_idx"],m["path"])
                self.draw_calls.append((vbo,ebo,int(len(iflat)),m["col"],tex_id)); n_ok+=1
            except Exception as e: print(f"[GLTF] VBO error: {e}")
        self.uploaded=n_ok>0; return self.uploaded

    def draw(self):
        if not self.draw_calls: return
        stride=8*4
        glEnableClientState(GL_VERTEX_ARRAY); glEnableClientState(GL_NORMAL_ARRAY); glEnableClientState(GL_TEXTURE_COORD_ARRAY)
        for vbo,ebo,count,col,tex_id in self.draw_calls:
            if tex_id is not None:
                glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D,tex_id); glColor4f(1,1,1,1)
            else:
                glDisable(GL_TEXTURE_2D); glColor4f(col[0],col[1],col[2],1.0)
            glBindBuffer(GL_ARRAY_BUFFER,vbo)
            glVertexPointer(3,GL_FLOAT,stride,ctypes.c_void_p(0))
            glNormalPointer(GL_FLOAT,stride,ctypes.c_void_p(12))
            glTexCoordPointer(2,GL_FLOAT,stride,ctypes.c_void_p(24))
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,ebo)
            glDrawElements(GL_TRIANGLES,count,GL_UNSIGNED_INT,None)
        glBindTexture(GL_TEXTURE_2D,0); glDisable(GL_TEXTURE_2D)
        glBindBuffer(GL_ARRAY_BUFFER,0); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,0)
        glDisableClientState(GL_VERTEX_ARRAY); glDisableClientState(GL_NORMAL_ARRAY); glDisableClientState(GL_TEXTURE_COORD_ARRAY)

    def collide_capsule(self, prev_pos, new_pos, radius, height, vy):
        if TRI_GRID.tri_arr is None or len(TRI_GRID.tri_arr) == 0:
            return False, False, list(new_pos)
        idx = TRI_GRID.query_indices(new_pos[0], new_pos[2], radius + 1.5)
        if len(idx) == 0: return False, False, list(new_pos)
        subset = TRI_GRID.tri_arr[idx]
        return _collide_capsule_fast(prev_pos, new_pos, radius, height, subset, vy)

GLTF_MAP = GLTFMap()

# ══════════════════════════════════════════════════════════════════════════════
# DEBUG
# ══════════════════════════════════════════════════════════════════════════════
def debug_gltf(path):
    print("="*60+f"\n  GLTF DEBUG: {path}\n"+"="*60)
    if not _HAS_GLTF: print("pygltflib not installed"); return
    if not os.path.exists(path): print(f"File not found: {path}"); return
    gltf=pygltflib.GLTF2().load(path)
    print(f"scenes:{len(gltf.scenes)} nodes:{len(gltf.nodes)} meshes:{len(gltf.meshes)} accessors:{len(gltf.accessors)}")
    print("\n── nodes ──")
    for i,nd in enumerate(gltf.nodes):
        print(f"  [{i}] name={nd.name!r} mesh={nd.mesh} children={nd.children}")
    print("\n── meshes ──")
    for i,m in enumerate(gltf.meshes):
        print(f"  [{i}] {m.name!r}")
        for j,p in enumerate(m.primitives):
            a=p.attributes
            print(f"    prim[{j}]: mode={p.mode} POS={a.POSITION} NRM={a.NORMAL} UV={a.TEXCOORD_0} idx={p.indices} mat={p.material}")
    print("\n── parse attempt ──"); GLTF_MAP.parse(path); print("="*60)

# ══════════════════════════════════════════════════════════════════════════════
# QUAKE-STYLE MOVEMENT FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def pm_accelerate(vel_xz, wish_dir_xz, wish_speed, accel):
    """
    Quake-style acceleration (per-frame, fixed timestep).
    Adds velocity along wish_dir only up to wish_speed projection.
    Keeps extra speed from bunny-hops — no hard cap on total speed.
    """
    cur = vel_xz[0]*wish_dir_xz[0] + vel_xz[1]*wish_dir_xz[1]
    add = wish_speed - cur
    if add <= 0:
        return vel_xz
    gain = min(accel, add)   # accel is already in units/frame
    return [vel_xz[0] + gain*wish_dir_xz[0],
            vel_xz[1] + gain*wish_dir_xz[1]]

def pm_friction(vel_xz, scale):
    """Simple per-frame multiplicative friction. scale e.g. 0.80."""
    return [vel_xz[0]*scale, vel_xz[1]*scale]

# ══════════════════════════════════════════════════════════════════════════════
# PLAYER
# ══════════════════════════════════════════════════════════════════════════════
P_H = 1.8
P_R = 0.38
EYE = 1.62

class Player:
    def __init__(self, pos):
        self.pos        = list(pos)
        self.vel        = v3(0, 0, 0)
        self._prev_pos  = list(pos)
        self.yaw        = 0.0
        self.pitch      = 0.0

        # state
        self.on_ground  = False
        self.on_wall    = False
        self.jumps      = 2          # remaining air jumps
        self.coyote     = 0          # coyote time frames
        self.jump_held  = False      # for variable height
        self.jump_buffer= 0          # buffered jump input frames

        # abilities
        self.slide_t    = 0
        self.slide_cd   = 0
        self.sliding    = False
        self.dash_t     = 0
        self.dash_cd    = 0
        self.dashing    = False
        self.dash_dir   = v3(0, 0, 1)

        # camera feel
        self.bob        = 0.0        # head-bob phase
        self.bob_amp    = 0.0        # current bob amplitude (smoothed)
        self.tilt       = 0.0        # roll tilt on turning
        self.land_squish= 0.0        # camera dip on landing
        self.fov_current= BASE_FOV   # dynamic FOV

        self._was_on_ground = False

    def eye_pos(self):
        slide_off = -0.45 if self.sliding else 0.0
        bob_off   = math.sin(self.bob) * self.bob_amp
        squish    = self.land_squish
        return [self.pos[0],
                self.pos[1] + EYE + bob_off + slide_off - squish,
                self.pos[2]]

    def fwd(self):
        yr = math.radians(self.yaw)
        return [math.sin(yr), 0.0, -math.cos(yr)]

    def rgt(self):
        yr = math.radians(self.yaw)
        return [math.cos(yr), 0.0, math.sin(yr)]

    def look_dir(self):
        yr = math.radians(self.yaw)
        pr = math.radians(self.pitch)
        return [math.cos(pr)*math.sin(yr), math.sin(pr), -math.cos(pr)*math.cos(yr)]

    def apply_input(self, inp):
        # ── Mouse look ──────────────────────────────────────────────────────
        self.yaw   = (self.yaw + inp.get("dy", 0)) % 360
        self.pitch = clamp(self.pitch + inp.get("dp", 0), -89, 89)

        # ── Wish direction ───────────────────────────────────────────────────
        fw = self.fwd()
        rt = self.rgt()
        wx = wy = wz = 0.0
        if inp.get("w"): wx += fw[0]; wz += fw[2]
        if inp.get("s"): wx -= fw[0]; wz -= fw[2]
        if inp.get("d"): wx += rt[0]; wz += rt[2]
        if inp.get("a"): wx -= rt[0]; wz -= rt[2]
        wlen = math.sqrt(wx*wx + wz*wz)
        if wlen > 0: wx /= wlen; wz /= wlen   # normalized wish dir

        # ── Jump buffer ──────────────────────────────────────────────────────
        if inp.get("jump_press"):
            self.jump_buffer = JUMP_BUFFER

        # ── Slide ────────────────────────────────────────────────────────────
        if inp.get("slide") and self.on_ground and self.slide_cd == 0 and wlen > 0:
            self.sliding  = True
            self.slide_t  = SLIDE_DUR
            self.slide_cd = SLIDE_CD
            sv = SLIDE_VEL
            self.vel[0] = wx * sv; self.vel[2] = wz * sv

        # ── Dash ─────────────────────────────────────────────────────────────
        if inp.get("dash") and self.dash_cd == 0:
            self.dashing  = True
            self.dash_t   = DASH_DUR
            self.dash_cd  = DASH_CD
            dw = wx if wlen > 0 else fw[0]
            dz = wz if wlen > 0 else fw[2]
            self.dash_dir = [dw * DASH_VEL, dz * DASH_VEL]
            self.vel[1]   = max(self.vel[1], 0.06)

        # ── Jump — double jump only (2 charges, both usable in air) ────────
        can_jump = (self.on_ground or self.coyote > 0)
        if self.jump_buffer > 0:
            if can_jump and self.jumps > 0:
                self.vel[1]      = JUMP_VEL
                self.coyote      = 0
                self.jumps      -= 1   # costs one charge whether on ground or air
                self.jump_held   = True
                self.jump_buffer = 0
            elif self.on_wall:
                wn = vnorm([-self.vel[0], 0, -self.vel[2]])
                self.vel[0] = wn[0] * WALL_JUMP_H
                self.vel[2] = wn[2] * WALL_JUMP_H
                self.vel[1] = WALL_JUMP_V
                self.jump_held   = True
                self.jump_buffer = 0
            elif self.jumps > 0:
                self.vel[1]      = JUMP_VEL
                self.jumps      -= 1
                self.jump_held   = True
                self.jump_buffer = 0

        # Variable jump height: release space early = shorter jump
        if self.jump_held and not inp.get("jump_held") and self.vel[1] > 0:
            self.vel[1] *= JUMP_CUT
            self.jump_held = False
        elif not inp.get("jump_held"):
            self.jump_held = False

        # ── Apply movement ────────────────────────────────────────────────────
        if self.dashing:
            self.vel[0] = self.dash_dir[0]
            self.vel[2] = self.dash_dir[1]

        elif self.sliding:
            # Sliding: friction but no player control
            self.vel[0] *= 0.93
            self.vel[2] *= 0.93

        elif self.on_ground:
            # Friction first, then accelerate — snappy stops, responsive starts
            vxz = pm_friction([self.vel[0], self.vel[2]], GROUND_FRICTION)
            if wlen > 0:
                vxz = pm_accelerate(vxz, [wx, wz], MAX_GROUND_SPD, GROUND_ACCEL)
            self.vel[0] = vxz[0]
            self.vel[2] = vxz[1]

        else:
            # Air: no friction, strafing adds speed up to wish cap
            if wlen > 0:
                vxz = pm_accelerate(
                    [self.vel[0], self.vel[2]],
                    [wx, wz],
                    MAX_AIR_WISH,
                    AIR_ACCEL
                )
                self.vel[0] = vxz[0]
                self.vel[2] = vxz[1]

        # ── Camera effects ────────────────────────────────────────────────────
        spd_h = math.sqrt(self.vel[0]**2 + self.vel[2]**2)

        # Head bob: smooth amplitude toward target
        bob_target = spd_h * 0.018 if self.on_ground and not self.sliding else 0.0
        self.bob_amp = lerp(self.bob_amp, bob_target, 0.08)
        if self.on_ground and spd_h > 0.04:
            self.bob += spd_h * 2.5
        else:
            self.bob *= 0.90

        # Strafe tilt — feels like weight
        self.tilt = lerp(self.tilt, inp.get("dy", 0) * 0.55, 0.14)

        # Dynamic FOV — higher when fast / dashing
        fov_target = BASE_FOV + SPEED_FOV_ADD * clamp(spd_h / MAX_GROUND_SPD, 0, 2.0)
        if self.dashing: fov_target = BASE_FOV + SPEED_FOV_ADD * 2.5
        self.fov_current = lerp(self.fov_current, fov_target, FOV_LERP)

        # Land squish decay
        self.land_squish = lerp(self.land_squish, 0.0, 0.22)

    def update(self):
        # Countdown timers
        if self.dash_t  > 0: self.dash_t  -= 1
        else:                 self.dashing  = False
        if self.dash_cd > 0: self.dash_cd -= 1
        if self.slide_t > 0: self.slide_t -= 1
        else:                 self.sliding  = False
        if self.slide_cd> 0: self.slide_cd -= 1
        if self.coyote  > 0: self.coyote  -= 1
        if self.jump_buffer > 0: self.jump_buffer -= 1

        # Gravity (not during dash)
        if not self.dashing:
            self.vel[1] -= GRAVITY

        # Integrate
        self._prev_pos = list(self.pos)
        new_pos = vadd(self.pos, self.vel)

        prev_g = self.on_ground
        self.on_ground = False
        self.on_wall   = False

        if GLTF_MAP.parsed:
            # Step size must be smaller than wall_r (0.35 * P_R) to prevent tunnelling
            MAX_STEP = P_R * 0.5   # half radius per step — safely below sphere radius
            dx = new_pos[0] - self._prev_pos[0]
            dz = new_pos[2] - self._prev_pos[2]
            dist = math.sqrt(dx*dx + dz*dz)
            steps = max(1, int(math.ceil(dist / MAX_STEP)))
            cur = list(self._prev_pos)
            on_g = False; on_w = False
            for s in range(steps):
                frac = (s + 1) / steps
                sub_new = [self._prev_pos[0] + dx*frac,
                           self._prev_pos[1] + (new_pos[1]-self._prev_pos[1])*frac,
                           self._prev_pos[2] + dz*frac]
                prev_xz = [cur[0], cur[1], cur[2]]
                on_g, on_w, cur = GLTF_MAP.collide_capsule(
                    prev_xz, sub_new, P_R, P_H, self.vel[1])
                if on_g:
                    if self.vel[1] < 0: self.vel[1] = 0.0
                if on_w:
                    self.vel[0] *= 0.0
                    self.vel[2] *= 0.0
                    break
            self.pos = cur
            if on_g:
                self.on_ground = True
                self.jumps     = 2
            self.on_wall = on_w

            # Step-down: raycast down every frame we are moving horizontally
            # and not grounded. Snaps to floor within STEP_DOWN units.
            # Runs continuously while falling slowly so ledges/stairs work.
            STEP_DOWN = 0.55
            if not self.on_ground and self.vel[1] <= 0.01 and self.vel[1] >= -0.35:
                spd_h = math.sqrt(self.vel[0]**2 + self.vel[2]**2)
                if spd_h > 0.001:
                    floor_y = _raycast_down(self.pos[0], self.pos[1], self.pos[2], STEP_DOWN)
                    if floor_y is not None:
                        self.pos[1] = floor_y
                        self.on_ground = True
                        self.jumps = 2
                        self.vel[1] = 0.0
        else:
            # Fallback arena collision
            self.pos = new_pos
            for cx, cz, pw, pd, yt in FALLBACK_PLATFORMS:
                if (cx-pw/2-P_R < self.pos[0] < cx+pw/2+P_R and
                    cz-pd/2-P_R < self.pos[2] < cz+pd/2+P_R and
                    self.vel[1] <= 0 and
                    self.pos[1] <= yt+0.12 and self.pos[1] >= yt-0.65):
                    self.pos[1]    = yt
                    self.vel[1]    = 0.0
                    self.on_ground = True
                    self.jumps     = 2
            hw = ARENA_W/2-P_R; hd_ = ARENA_D/2-P_R
            if abs(self.pos[0]) > hw: self.vel[0] *= -0.3; self.pos[0] = clamp(self.pos[0], -hw, hw)
            if abs(self.pos[2]) > hd_: self.vel[2] *= -0.3; self.pos[2] = clamp(self.pos[2], -hd_, hd_)

        # Landing squish camera dip
        if not prev_g and self.on_ground:
            impact = clamp(-self.vel[1] * 3.0, 0.0, 0.28)
            self.land_squish = impact

        # Coyote time: started falling this frame
        if prev_g and not self.on_ground:
            self.coyote = COYOTE_FRAMES

        # Clamp upward velocity — prevents fly-hacking / explosion bugs
        # Also hard-cap position so you can never escape the map vertically
        self.vel[1] = clamp(self.vel[1], -2.0, JUMP_VEL * 1.1)
        if self.pos[1] > MAX_Y:
            self.pos[1] = MAX_Y
            self.vel[1] = min(self.vel[1], 0.0)

        # Respawn if fallen far below the map
        floor_y = GLTF_MAP.aabb_min[1] if GLTF_MAP.aabb_min else 0.0
        if self.pos[1] < floor_y - 30:
            self.pos = list(spawn_pos())
            self.vel = v3(0, 0, 0)
            self.land_squish = 0.0
        


        self._was_on_ground = self.on_ground

# ══════════════════════════════════════════════════════════════════════════════
# GL HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def spawn_pos():
    if GLTF_MAP.aabb_min and GLTF_MAP.aabb_max:
        mn, mx = GLTF_MAP.aabb_min, GLTF_MAP.aabb_max
        # spawn well above the floor center so we drop in cleanly
        return [(mn[0]+mx[0])/2, mx[1]+4.0, (mn[2]+mx[2])/2]
    return [0.0, 4.0, 0.0]

def init_gl():
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_LIGHTING); glEnable(GL_LIGHT0); glEnable(GL_LIGHT1)
    glEnable(GL_COLOR_MATERIAL); glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glLightfv(GL_LIGHT0, GL_POSITION, [10, 20, 10, 1])
    glLightfv(GL_LIGHT0, GL_DIFFUSE,  [1.0, 0.92, 0.82, 1])
    glLightfv(GL_LIGHT0, GL_AMBIENT,  [0.22, 0.18, 0.32, 1])
    glLightfv(GL_LIGHT1, GL_POSITION, [-10, 6, -10, 1])
    glLightfv(GL_LIGHT1, GL_DIFFUSE,  [0.18, 0.14, 0.38, 1])
    glShadeModel(GL_SMOOTH)

def set_camera(player):
    glMatrixMode(GL_PROJECTION); glLoadIdentity()
    gluPerspective(player.fov_current, W/H, 0.04, 1000)
    glMatrixMode(GL_MODELVIEW); glLoadIdentity()
    glRotatef(player.tilt, 0, 0, 1)
    eye = player.eye_pos()
    ld  = player.look_dir()
    ct  = vadd(eye, ld)
    gluLookAt(eye[0], eye[1], eye[2], ct[0], ct[1], ct[2], 0, 1, 0)

def draw_sky():
    glDisable(GL_LIGHTING); glDisable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity(); glOrtho(-1,1,-1,1,-1,1)
    glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
    glBegin(GL_QUADS)
    glColor3f(*SKY_TOP); glVertex2f(-1, 1); glVertex2f(1, 1)
    glColor3f(*SKY_BOT); glVertex2f(1, -1); glVertex2f(-1,-1)
    glEnd()
    glMatrixMode(GL_PROJECTION); glPopMatrix()
    glMatrixMode(GL_MODELVIEW);  glPopMatrix()
    glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING)

def draw_box_gl(cx, cy, cz, bw, bh, bd, col, alpha=1.0):
    hw, hh, hd = bw/2, bh/2, bd/2
    glColor4f(col[0], col[1], col[2], alpha)
    glPushMatrix(); glTranslatef(cx, cy, cz); glBegin(GL_QUADS)
    for (nx,ny,nz), verts in [
        ((0,0, 1),((-hw,-hh, hd),(hw,-hh, hd),(hw, hh, hd),(-hw, hh, hd))),
        ((0,0,-1),((hw,-hh,-hd),(-hw,-hh,-hd),(-hw, hh,-hd),(hw, hh,-hd))),
        ((-1,0,0),((-hw,-hh,-hd),(-hw,-hh, hd),(-hw, hh, hd),(-hw, hh,-hd))),
        (( 1,0,0),((hw,-hh, hd),(hw,-hh,-hd),(hw, hh,-hd),(hw, hh, hd))),
        ((0, 1,0),((-hw, hh, hd),(hw, hh, hd),(hw, hh,-hd),(-hw, hh,-hd))),
        ((0,-1,0),((-hw,-hh,-hd),(hw,-hh,-hd),(hw,-hh, hd),(-hw,-hh, hd))),
    ]:
        glNormal3f(nx, ny, nz)
        for v in verts: glVertex3f(*v)
    glEnd(); glPopMatrix()

def draw_fallback_arena():
    draw_box_gl(0, -0.15, 0, ARENA_W, 0.3, ARENA_D, FLOOR_COL)
    glDisable(GL_LIGHTING); glColor4f(*GRID_COL, 0.45); glLineWidth(1); glBegin(GL_LINES)
    for i in range(int(-ARENA_W/2), int(ARENA_W/2)+1, 4):
        glVertex3f(i, 0.01, -ARENA_D/2); glVertex3f(i, 0.01, ARENA_D/2)
    for i in range(int(-ARENA_D/2), int(ARENA_D/2)+1, 4):
        glVertex3f(-ARENA_W/2, 0.01, i); glVertex3f(ARENA_W/2, 0.01, i)
    glEnd(); glEnable(GL_LIGHTING)
    for cx,cz,pw,pd,yt in FALLBACK_PLATFORMS[1:]:
        draw_box_gl(cx, yt-0.18, cz, pw, 0.36, pd, PLAT_COL)
    t = 0.4
    draw_box_gl(0, 2.5,  ARENA_D/2, ARENA_W, 5, t, WALL_COL, 0.55)
    draw_box_gl(0, 2.5, -ARENA_D/2, ARENA_W, 5, t, WALL_COL, 0.55)
    draw_box_gl( ARENA_W/2, 2.5, 0, t, 5, ARENA_D, WALL_COL, 0.55)
    draw_box_gl(-ARENA_W/2, 2.5, 0, t, 5, ARENA_D, WALL_COL, 0.55)

def draw_crosshair(player):
    glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity(); glOrtho(0,W,H,0,-1,1)
    glMatrixMode(GL_MODELVIEW);  glPushMatrix(); glLoadIdentity()
    glDisable(GL_DEPTH_TEST); glDisable(GL_LIGHTING)
    cx, cy = W//2, H//2
    # Crosshair gap expands with speed
    spd = math.sqrt(player.vel[0]**2 + player.vel[2]**2)
    gap = int(4 + spd * 18)
    s = 9
    glColor4f(1, 1, 1, 0.90); glLineWidth(1.5); glBegin(GL_LINES)
    glVertex2f(cx-s-gap, cy); glVertex2f(cx-gap, cy)
    glVertex2f(cx+gap,   cy); glVertex2f(cx+s+gap, cy)
    glVertex2f(cx, cy-s-gap); glVertex2f(cx, cy-gap)
    glVertex2f(cx, cy+gap);   glVertex2f(cx, cy+s+gap)
    glEnd()
    glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING)
    glMatrixMode(GL_PROJECTION); glPopMatrix()
    glMatrixMode(GL_MODELVIEW);  glPopMatrix()

# ── HUD text GL textures ──────────────────────────────────────────────────────
_font_tex_cache = {}

def _gl_text_tex(font, text, color):
    key = (id(font), text, color)
    if key not in _font_tex_cache:
        surf = font.render(text, True, color)
        data = pygame.image.tostring(surf, "RGBA", True)
        tid  = int(glGenTextures(1)); glBindTexture(GL_TEXTURE_2D, tid)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                     surf.get_width(), surf.get_height(), 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glBindTexture(GL_TEXTURE_2D, 0)
        _font_tex_cache[key] = (tid, surf.get_width(), surf.get_height())
    return _font_tex_cache[key]

def _hud_text(font, text, color, x, y):
    tid, tw, th = _gl_text_tex(font, text, color)
    by = H - y - th
    glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D, tid); glColor4f(1,1,1,1)
    glBegin(GL_QUADS)
    glTexCoord2f(0,0); glVertex2f(x,    by)
    glTexCoord2f(1,0); glVertex2f(x+tw, by)
    glTexCoord2f(1,1); glVertex2f(x+tw, by+th)
    glTexCoord2f(0,1); glVertex2f(x,    by+th)
    glEnd(); glDisable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D, 0)

def draw_hud(player, font_sm, font_med, wire):
    glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity(); glOrtho(0,W,0,H,-1,1)
    glMatrixMode(GL_MODELVIEW);  glPushMatrix(); glLoadIdentity()
    glDisable(GL_DEPTH_TEST); glDisable(GL_LIGHTING)
    glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    x, y, z = player.pos
    spd = math.sqrt(player.vel[0]**2 + player.vel[2]**2)
    map_label = ("MAP: "+os.path.basename(sys.argv[1])) if (len(sys.argv)>1 and GLTF_MAP.parsed) \
                else ("MAP: ex.gltf" if GLTF_MAP.parsed else "MAP: fallback")

    _hud_text(font_sm, f"XYZ  {x:.1f}  {y:.2f}  {z:.1f}", (200,200,200), 14, 14)
    _hud_text(font_sm, f"SPD  {spd*100:.0f}  VY {player.vel[1]*100:+.0f}", (200,200,200), 14, 32)
    _hud_text(font_sm, f"Yaw {player.yaw%360:.1f}°  Pitch {player.pitch:.1f}°", (200,200,200), 14, 50)
    _hud_text(font_sm, map_label, (120,110,180), 14, H-22)
    _hud_text(font_sm, "WASD move · Space jump · Shift dash · Ctrl slide · F wire · ESC quit",
              (90,80,120), 14, H-42)

    # State labels
    if not player.on_ground:
        jumps_left = player.jumps
        col = (80,180,255) if jumps_left > 0 else (200,80,80)
        label = f"AIR  ×{jumps_left}" if jumps_left > 0 else "AIR"
        _hud_text(font_sm, label, col, W-90, 14)

    if player.sliding:
        _hud_text(font_med, "SLIDE", (180,80,255), W//2-42, H-54)
    if player.dashing:
        _hud_text(font_med, "DASH", (80,220,255), W//2-38, H-54)
    if wire:
        _hud_text(font_sm, "WIREFRAME", (255,220,80), W-120, H-22)

    # Cooldown bars
    def abil_bar(cx, y, cd, mx, col, lbl):
        bw, bh = 116, 14; x_ = cx - bw//2
        glColor4f(0.06, 0.05, 0.12, 0.75)
        glBegin(GL_QUADS); by_ = H-y-bh
        glVertex2f(x_, by_); glVertex2f(x_+bw, by_)
        glVertex2f(x_+bw, by_+bh); glVertex2f(x_, by_+bh); glEnd()
        fill = int(bw * (1 - cd/mx))
        glColor4f(col[0], col[1], col[2], 0.85)
        glBegin(GL_QUADS); glVertex2f(x_, by_); glVertex2f(x_+fill, by_)
        glVertex2f(x_+fill, by_+bh); glVertex2f(x_, by_+bh); glEnd()
        _hud_text(font_sm, lbl, (220,220,220), cx-32, y+1)

    abil_bar(W//2-72, H-36, player.dash_cd,  DASH_CD,  (0.31,0.71,1.00), "DASH [Shift]")
    abil_bar(W//2+72, H-36, player.slide_cd, SLIDE_CD, (0.71,0.31,1.00), "SLIDE [Ctrl]")

    glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING)
    glMatrixMode(GL_PROJECTION); glPopMatrix()
    glMatrixMode(GL_MODELVIEW);  glPopMatrix()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN GAME LOOP
# ══════════════════════════════════════════════════════════════════════════════
def main():
    if "--debug" in sys.argv:
        path = next((a for a in sys.argv[1:] if not a.startswith("-")), "ex.gltf")
        debug_gltf(path); return

    gltf_path = next((a for a in sys.argv[1:] if not a.startswith("-")), "ex.gltf")

    pygame.init()
    pygame.display.set_mode((W, H), pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption("GLTF FPS Walker — Movement Shooter")

    _font_tex_cache.clear()
    init_gl()
    font_sm  = pygame.font.SysFont("consolas", 15)
    font_med = pygame.font.SysFont("consolas", 26, bold=True)

    GLTF_MAP.parse(gltf_path)
    if GLTF_MAP.parsed:
        GLTF_MAP.upload()

    player = Player(spawn_pos())
    clock  = pygame.time.Clock()
    mc     = False
    wire   = False

    # edge-detect stateful keys
    prev_jump  = False
    prev_dash  = False
    prev_slide = False

    # ambient stars
    rng = random.Random(77)
    amb = [(rng.uniform(-200,200), rng.uniform(10,80), rng.uniform(-200,200))
           for _ in range(300)]

    print("Click window to capture mouse. ESC releases; ESC again quits.")
    print("Controls: WASD | Space (tap=low/hold=high jump) | Shift dash | Ctrl slide")
    print("Tip: jump just before landing to bunny-hop and carry momentum!")

    while True:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if mc:
                        pygame.mouse.set_visible(True)
                        pygame.event.set_grab(False); mc = False
                    else:
                        pygame.quit(); sys.exit()
                if event.key == pygame.K_f:
                    wire = not wire
                    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE if wire else GL_FILL)
            if event.type == pygame.MOUSEBUTTONDOWN and not mc:
                pygame.mouse.set_visible(False)
                pygame.event.set_grab(True); mc = True

        dx, dy = pygame.mouse.get_rel()
        dyaw   =  dx * MOUSE_SENS if mc else 0.0
        dpitch = -dy * MOUSE_SENS if mc else 0.0
        keys   = pygame.key.get_pressed()

        jump_now  = bool(keys[pygame.K_SPACE])
        dash_now  = bool(keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT])
        slide_now = bool(keys[pygame.K_LCTRL]  or keys[pygame.K_RCTRL])

        inp = {
            "w": bool(keys[pygame.K_w]),
            "s": bool(keys[pygame.K_s]),
            "a": bool(keys[pygame.K_a]),
            "d": bool(keys[pygame.K_d]),
            "jump_press": jump_now  and not prev_jump,   # rising edge
            "jump_held":  jump_now,                       # held for variable height
            "dash":       dash_now  and not prev_dash,
            "slide":      slide_now and not prev_slide,
            "dy": dyaw,
            "dp": dpitch,
        }
        prev_jump  = jump_now
        prev_dash  = dash_now
        prev_slide = slide_now

        player.apply_input(inp)
        player.update()

        # ── Render ────────────────────────────────────────────────────────────
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw_sky()
        set_camera(player)

        # Stars / ambient dust
        glDisable(GL_LIGHTING); glPointSize(2); glBegin(GL_POINTS)
        rng2 = random.Random(77)
        for ax, ay, az in amb:
            br = rng2.uniform(0.3, 1.0)
            glColor3f(br, br, min(1, br+0.15)); glVertex3f(ax, ay, az)
        glEnd(); glEnable(GL_LIGHTING)

        if GLTF_MAP.uploaded:
            GLTF_MAP.draw()
        else:
            draw_fallback_arena()

        draw_crosshair(player)
        draw_hud(player, font_sm, font_med, wire)

        if not mc:
            glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity(); glOrtho(0,W,0,H,-1,1)
            glMatrixMode(GL_MODELVIEW);  glPushMatrix(); glLoadIdentity()
            glDisable(GL_DEPTH_TEST); glDisable(GL_LIGHTING)
            _hud_text(font_med, "Click window to capture mouse", (255,200,60), W//2-200, H//2)
            glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING)
            glMatrixMode(GL_PROJECTION); glPopMatrix()
            glMatrixMode(GL_MODELVIEW);  glPopMatrix()

        pygame.display.flip()

if __name__ == "__main__":
    main()