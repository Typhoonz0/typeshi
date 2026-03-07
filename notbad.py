"""
GLTF FPS Walker  —  Slide Edition  (FIXED: ledge stickiness)
  pip install pygame PyOpenGL PyOpenGL_accelerate pygltflib numpy

  Place ex.gltf + gun.gltf next to this script.
  python gltf_fps.py [yourmap.gltf]
  python gltf_fps.py --debug    (inspect your GLTF)

Controls:
  WASD        Move
  Mouse       Look  (click window to capture)
  Space       Jump  (tap = low, hold = high)
  Ctrl        Slide
  LMB         Shoot
  R           Reload
  F           Wireframe
  ESC         Release mouse / quit
"""

import pygame, sys, math, random, os, ctypes, base64, io, time
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
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
W, H  = 1280, 720
FPS   = 120

# --- Physics -----------------------------------------------------------------
GRAVITY        = 0.0028
JUMP_VEL       = 0.10
JUMP_CUT       = 0.75
JUMP_BUFFER    = 8
MAX_Y          = 2024.0
COYOTE_FRAMES  = 8

# Stair stepping
STEP_UP   = 0.30     # FIX: was 0.45 — lower = less ledge grabbing
STEP_DOWN = 0.30     # FIX: was 0.45

# --- Ground movement ---------------------------------------------------------
GROUND_ACCEL   = 0.014
GROUND_FRICTION= 0.78
MAX_GROUND_SPD = 0.08

# --- Air movement ------------------------------------------------------------
AIR_ACCEL      = 0.004
MAX_AIR_WISH   = 0.08

# --- Slide -------------------------------------------------------------------
SLIDE_VEL      = 0.22
SLIDE_DUR      = 18
SLIDE_CD       = 36

# --- Gun ---------------------------------------------------------------------
GUN_MAG        = 12          # bullets per mag
GUN_FIRE_CD    = 7           # frames between shots (~17 rps at 120fps)
GUN_RELOAD_DUR = 90          # frames for reload animation (~0.75s at 120fps)

# Viewmodel position: right, down, forward relative to eye (CS-style)
GUN_POS        = ( 0.38, -0.19, -0.45)  # further right, lower, slightly closer
GUN_SCALE      = 0.08        # scale the gun.gltf down to viewmodel size

# Recoil: small kick per shot, large kick for reload
RECOIL_PITCH   =  8.0        # degrees up per shot — big gun kick
RECOIL_RECOVER = 0.10        # lerp speed back to rest
RELOAD_KICK_X  =  0.08       # translate down during reload
RELOAD_KICK_ROT= 35.0        # rotate barrel down during reload

# --- Camera ------------------------------------------------------------------
MOUSE_SENS     = 0.10
BASE_FOV       = 100.0
SPEED_FOV_ADD  = 14.0
FOV_LERP       = 0.2222

# --- Arena / Visuals ---------------------------------------------------------
MAP_SCALE = 2.0
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
    floor_mask = valid & (n[:,1] > 0.3)
    if not floor_mask.any():
        return None
    fp0 = p0[floor_mask]; fp1 = p1[floor_mask]; fp2 = p2[floor_mask]
    fn  = n[floor_mask]
    denom = fn[:,1]
    valid2 = denom > 0.01
    if not valid2.any():
        return None
    op = fp0[valid2] - np.array([[rx, ry, rz]], np.float32)
    t_hit = (op * fn[valid2]).sum(axis=1) / denom[valid2]
    in_range = (t_hit >= -0.1) & (t_hit <= max_dist)
    if not in_range.any():
        return None
    t_vals = t_hit[in_range]
    hit_pts = np.stack([np.full(len(t_vals), rx),
                        ry - t_vals,
                        np.full(len(t_vals), rz)], axis=1).astype(np.float32)
    v2idx = np.where(valid2)[0][in_range]
    a = fp0[v2idx]; b = fp1[v2idx]; c = fp2[v2idx]
    inside = _point_in_tri_batch(hit_pts, a, b, c)
    if not inside.any():
        return None
    best_y = float((ry - t_vals[inside]).max())
    return best_y

def _raycast_hitscan(ox, oy, oz, dx, dy, dz, max_dist=400.0):
    """Cast a ray from (ox,oy,oz) in direction (dx,dy,dz).
    Returns (hit_x, hit_y, hit_z, dist) or None."""
    if TRI_GRID.tri_arr is None or len(TRI_GRID.tri_arr) == 0:
        return None
    # broad query along the ray — sample a few points and union their cells
    d_len = math.sqrt(dx*dx + dy*dy + dz*dz)
    if d_len < 1e-9: return None
    dx /= d_len; dy /= d_len; dz /= d_len
    steps = [0.0, max_dist * 0.25, max_dist * 0.5, max_dist * 0.75, max_dist]
    idx_set = set()
    for s in steps:
        px = ox + dx*s; pz = oz + dz*s
        for i in TRI_GRID.query_indices(px, pz, 6.0):
            idx_set.add(int(i))
    if not idx_set:
        return None
    idx = np.array(sorted(idx_set), dtype=np.int32)
    tris = TRI_GRID.tri_arr[idx]
    p0 = tris[:,0,:]; p1 = tris[:,1,:]; p2 = tris[:,2,:]
    e1 = p1-p0; e2 = p2-p0
    n = np.cross(e1, e2)
    nl = np.linalg.norm(n, axis=1, keepdims=True)
    valid = nl[:,0] > 1e-9
    n[valid] /= nl[valid]
    rd = np.array([[dx, dy, dz]], np.float32)
    denom = (n * rd).sum(axis=1)
    front = valid & (denom < -1e-4)   # only front-facing
    if not front.any(): return None
    ro = np.array([[ox, oy, oz]], np.float32)
    t_vals = ((p0[front] - ro) * n[front]).sum(axis=1) / denom[front]
    in_range = (t_vals > 0.05) & (t_vals <= max_dist)
    if not in_range.any(): return None
    t_sub = t_vals[in_range]
    hit_pts = ro + t_sub[:,None] * rd
    fp0 = p0[front][in_range]; fp1 = p1[front][in_range]; fp2 = p2[front][in_range]
    inside = _point_in_tri_batch(hit_pts, fp0, fp1, fp2)
    if not inside.any(): return None
    best_t = float(t_sub[inside].min())
    hx = ox + dx*best_t; hy = oy + dy*best_t; hz = oz + dz*best_t
    return (hx, hy, hz, best_t)
# ══════════════════════════════════════════════════════════════════════════════
def _collide_capsule_fast(prev_pos, new_pos, radius, height, tri_arr, vy=0.0):
    """
    Resolves a vertical capsule against tri_arr.
    KEY FIXES vs original:
      - Wall step-up threshold lowered to STEP_UP (0.30) so ledge sides don't
        auto-climb and stall the player.
      - Only 2 capsule sample heights instead of 4 — reduces over-detection on
        thin ledge faces that were causing the sticky slowdown.
      - Wall push only zeroes velocity for faces that are genuinely blocking
        (depth > 0.06), not shallow grazes.
      - velocity is NOT zeroed on wall hits; caller decides that.
    """
    rx, ry, rz = float(new_pos[0]), float(new_pos[1]), float(new_pos[2])
    on_ground   = False
    on_wall     = False
    hit_ceiling = False

    if len(tri_arr) == 0:
        return on_ground, on_wall, [rx, ry, rz], hit_ceiling

    p0 = tri_arr[:,0,:]; p1 = tri_arr[:,1,:]; p2 = tri_arr[:,2,:]
    e1 = p1-p0; e2 = p2-p0
    n  = np.cross(e1, e2)
    nl = np.linalg.norm(n, axis=1, keepdims=True)
    valid = (nl[:,0] > 1e-9)
    n[valid] /= nl[valid]
    ny = n[:,1]

    # ── FLOOR ────────────────────────────────────────────────────────────────
    # Only check floor when not rising. vy<=0 is the correct gate — the old
    # vy<=0.001 was also true during falls (vy is negative = less than 0.001)
    # which caused the mid-fall snap. Always snap ry so grounded state is stable.
    if vy <= 0:
        floor_mask = valid & (ny > 0.4)
        if floor_mask.any():
            fp0 = p0[floor_mask]; fp1 = p1[floor_mask]; fp2 = p2[floor_mask]
            foot = np.array([[rx, ry, rz]], np.float32)
            cp   = _closest_pts_tris_batch(foot, fp0, fp1, fp2)
            dxz  = np.sqrt((rx-cp[:,0])**2 + (rz-cp[:,2])**2)
            hit  = (dxz <= radius + 0.20) & (cp[:,1] <= ry + 0.12) & (cp[:,1] >= ry - 0.22)
            if hit.any():
                best_y = float(cp[hit, 1].max())
                if best_y <= ry + 0.12:
                    ry = best_y
                    on_ground = True

    # ── CEILING ──────────────────────────────────────────────────────────────
    # FIX: Always check ceiling regardless of vy — arched/lip geometry has
    # angled faces that need resolving even when vy is near zero.
    # Also sample at multiple heights along the capsule so the *middle* of the
    # capsule can't clip through a low arch while only the top is tested.
    ceil_mask = valid & (ny < -0.35)   # FIX: was -0.4, catch shallower arches
    if ceil_mask.any():
        cp0 = p0[ceil_mask]; cp1 = p1[ceil_mask]; cp2 = p2[ceil_mask]
        for ct in (1.0, 0.85, 0.70):   # FIX: test top, upper-mid, mid of capsule
            sample_y = ry + height * ct
            top = np.array([[rx, sample_y, rz]], np.float32)
            cc  = _closest_pts_tris_batch(top, cp0, cp1, cp2)
            dxz_c = np.sqrt((rx-cc[:,0])**2 + (rz-cc[:,2])**2)
            chit = (dxz_c <= radius + 0.15) & \
                   (cc[:,1] >= sample_y - 0.08) & \
                   (cc[:,1] <= sample_y + radius + 0.15)
            if chit.any():
                lowest = float(cc[chit, 1].min())
                new_ry = lowest - height * ct - 0.02
                if new_ry < ry:
                    ry = new_ry
                    hit_ceiling = True

    # ── WALLS ────────────────────────────────────────────────────────────────
    # FIX: only sample at 2 heights (lower torso, upper torso) instead of 4.
    # Sampling at foot-level (t=0.15) was the main cause of ledge-face false hits
    # because the foot sphere overlapped ledge sides exactly where step-up
    # would kick in, causing an infinite push-up-then-push-back loop.
    wall_mask = valid & (np.abs(ny) < 0.5)
    if wall_mask.any():
        wp0   = p0[wall_mask]; wp1 = p1[wall_mask]; wp2 = p2[wall_mask]
        wn    = n[wall_mask]
        w_top = np.maximum(np.maximum(wp0[:,1], wp1[:,1]), wp2[:,1])

        # FIX: minimum penetration depth to act on — ignores grazing contacts
        MIN_PEN = 0.06

        for _iter in range(6):          # FIX: 6 iters instead of 8
            any_pen = False
            for t in (0.35, 0.75):      # FIX: was (0.15, 0.40, 0.65, 0.90)
                sy = ry + height * t
                pt = np.array([[rx, sy, rz]], np.float32)
                cp = _closest_pts_tris_batch(pt, wp0, wp1, wp2)

                dx3 = rx - cp[:,0]
                dz3 = rz - cp[:,2]
                dy3 = sy - cp[:,1]
                dist3 = np.sqrt(dx3*dx3 + dy3*dy3 + dz3*dz3)

                pen = dist3 < radius
                if not pen.any():
                    continue

                depths = np.where(pen, radius - dist3, -1.0)
                i = int(np.argmax(depths))
                d = float(dist3[i])

                # FIX: ignore shallow grazes
                if (radius - d) < MIN_PEN:
                    continue

                # Step-up: face top reachable AND not jumping
                # FIX: threshold uses STEP_UP (0.30) not hardcoded 0.50
                face_rise = w_top[i] - ry
                if 0.0 <= face_rise <= STEP_UP and vy <= 0.001:
                    ry        = float(w_top[i])
                    on_ground = True
                    any_pen   = True
                    continue

                # Sideways push — XZ only
                ox = float(dx3[i]); oz = float(dz3[i])
                dxz = math.sqrt(ox*ox + oz*oz)
                if dxz < 1e-6:
                    fnx = float(wn[i,0]); fnz = float(wn[i,2])
                    fnl = math.sqrt(fnx*fnx + fnz*fnz)
                    if fnl > 1e-6:
                        rx += fnx/fnl * (radius - d + 0.005)
                        rz += fnz/fnl * (radius - d + 0.005)
                else:
                    ov  = radius - d + 0.005
                    rx += (ox / dxz) * ov
                    rz += (oz / dxz) * ov
                on_wall = True
                any_pen = True
            if not any_pen:
                break

    # ── ANTI-CLIP SANITY CHECK ────────────────────────────────────────────────
    # If after all resolution the capsule is deeply inside any triangle's solid
    # side, snap XZ back to prev_pos. This stops jump-into-lip clipping where
    # the ceiling push moves the player far enough that wall resolution places
    # them on the wrong side of the back face.
    for t in (0.35, 0.75):
        sy = ry + height * t
        pt = np.array([[rx, sy, rz]], np.float32)
        cp_all = _closest_pts_tris_batch(pt, p0, p1, p2)
        dist_all = np.sqrt(((pt - cp_all) ** 2).sum(axis=1))
        toward = ((pt - cp_all) * n).sum(axis=1)  # negative = inside solid
        deeply_inside = valid & (dist_all < radius * 0.45) & (toward < 0)
        if deeply_inside.any():
            rx = float(prev_pos[0])
            rz = float(prev_pos[2])
            on_wall = True
            break

    return on_ground, on_wall, [rx, ry, rz], hit_ceiling

def _point_in_tri_batch(pts, a, b, c):
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

    def parse(self, path, skip_trigrid=False):
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
        if not skip_trigrid:
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
            return False, False, list(new_pos), False
        idx = TRI_GRID.query_indices(new_pos[0], new_pos[2], radius + 1.5)
        if len(idx) == 0: return False, False, list(new_pos), False
        subset = TRI_GRID.tri_arr[idx]
        return _collide_capsule_fast(prev_pos, new_pos, radius, height, subset, vy)

GLTF_MAP = GLTFMap()

# ══════════════════════════════════════════════════════════════════════════════
# GUN MODEL  (viewmodel — rendered in separate near-clip pass)
# ══════════════════════════════════════════════════════════════════════════════
class GunModel:
    """Loads gun.gltf and renders it as a right-side viewmodel."""
    def __init__(self):
        self.draw_calls = []
        self.loaded     = False

    def load(self, path="gun.gltf"):
        if not _HAS_GLTF or not os.path.exists(path):
            print(f"[GUN] '{path}' not found — will skip viewmodel"); return
        tmp = GLTFMap()
        if not tmp.parse(path, skip_trigrid=True):
            print(f"[GUN] parse failed: {tmp.parse_error}"); return
        # Re-upload with tmp's cpu_meshes but don't stomp TRI_GRID
        for m in tmp._cpu_meshes:
            try:
                vflat = m["vdata"].flatten().astype(np.float32)
                iflat = m["idata"].flatten().astype(np.uint32)
                vbo = int(glGenBuffers(1)); glBindBuffer(GL_ARRAY_BUFFER, vbo)
                glBufferData(GL_ARRAY_BUFFER, vflat.nbytes, vflat.tobytes(), GL_STATIC_DRAW)
                glBindBuffer(GL_ARRAY_BUFFER, 0)
                ebo = int(glGenBuffers(1)); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
                glBufferData(GL_ELEMENT_ARRAY_BUFFER, iflat.nbytes, iflat.tobytes(), GL_STATIC_DRAW)
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
                tex_id = None
                if m["tex_idx"] is not None:
                    tex_id = tmp._load_texture(m["gltf"], m["tex_idx"], path)
                self.draw_calls.append((vbo, ebo, int(len(iflat)), m["col"], tex_id))
            except Exception as e:
                print(f"[GUN] VBO error: {e}")
        self.loaded = len(self.draw_calls) > 0
        print(f"[GUN] loaded {len(self.draw_calls)} draw calls")

    def draw(self, player, gun_state):
        """Draw the viewmodel in its own projection so it never clips walls."""
        if not self.loaded: return

        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
        gluPerspective(75.0, W/H, 0.01, 10.0)
        glMatrixMode(GL_MODELVIEW);  glPushMatrix(); glLoadIdentity()

        glClear(GL_DEPTH_BUFFER_BIT)

        # ── recoil & reload offsets ───────────────────────────────────────────
        rx_off  = gun_state.recoil_offset_y
        rp_off  = gun_state.recoil_pitch
        rel_off = gun_state.reload_offset
        rel_rot = gun_state.reload_rot

        # ── position: right, down, forward in screen space ────────────────────
        tx = GUN_POS[0]
        ty = GUN_POS[1] + rx_off - rel_off
        tz = GUN_POS[2]

        glTranslatef(tx, ty, tz)

        # recoil pitch kick + reload barrel dip — rotate around screen X only
        glRotatef(rp_off + rel_rot, 1, 0, 0)

        # orient gun: barrel into screen, grip pointing down (CS-style right-hand)
        glRotatef(72, 0, 20000, 170)    # roll so grip points down
        # lag: gun pulls back as you accelerate
        spd_h = math.sqrt(player.vel[0]**2 + player.vel[2]**2)
        lag_z = gun_state.lag_smooth
        # breathing: fades out completely once moving at all
        still = max(0.0, 1.0 - spd_h / (MAX_GROUND_SPD * 0.15))
        t = time.time()
        breath_y = math.sin(t * 1.1)  * 0.004 * still
        breath_x = math.sin(t * 0.55) * 0.0016 * still
        jump_drop = gun_state.jump_drop_smooth
        glTranslatef(lag_z * -0.04 + breath_x, lag_z * -0.02 + breath_y + jump_drop, lag_z * 0.03)

        # scale
        s = GUN_SCALE / MAP_SCALE
        glScalef(s, s, s)

        # ── draw ──────────────────────────────────────────────────────────────
        stride = 8*4
        glEnable(GL_LIGHTING)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glEnableClientState(GL_TEXTURE_COORD_ARRAY)

        for vbo, ebo, count, col, tex_id in self.draw_calls:
            if tex_id is not None:
                glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D, tex_id); glColor4f(1,1,1,1)
            else:
                glDisable(GL_TEXTURE_2D); glColor4f(col[0], col[1], col[2], 1.0)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glVertexPointer(3, GL_FLOAT, stride, ctypes.c_void_p(0))
            glNormalPointer(GL_FLOAT, stride, ctypes.c_void_p(12))
            glTexCoordPointer(2, GL_FLOAT, stride, ctypes.c_void_p(24))
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
            glDrawElements(GL_TRIANGLES, count, GL_UNSIGNED_INT, None)

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glDisableClientState(GL_TEXTURE_COORD_ARRAY)
        glBindTexture(GL_TEXTURE_2D, 0); glDisable(GL_TEXTURE_2D)
        glBindBuffer(GL_ARRAY_BUFFER, 0); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

        glMatrixMode(GL_MODELVIEW);  glPopMatrix()
        glMatrixMode(GL_PROJECTION); glPopMatrix()

GUN_MODEL = GunModel()

# ══════════════════════════════════════════════════════════════════════════════
# BULLET HOLES
# ══════════════════════════════════════════════════════════════════════════════
MAX_HOLES = 64

class BulletHoles:
    def __init__(self):
        self.holes = []   # list of (x,y,z, nx,ny,nz)

    def add(self, pos, normal):
        self.holes.append((pos, normal))
        if len(self.holes) > MAX_HOLES:
            self.holes.pop(0)

    def draw(self):
        if not self.holes: return
        glDisable(GL_LIGHTING)
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(-1.0, -1.0)
        glColor4f(0.05, 0.03, 0.02, 0.92)
        SIZE = 0.06
        for (px,py,pz), (nx,ny,nz) in self.holes:
            # Build two tangent vectors perpendicular to normal
            up = [0,1,0] if abs(ny) < 0.9 else [1,0,0]
            tx = ny*up[2]-nz*up[1]; ty = nz*up[0]-nx*up[2]; tz = nx*up[1]-ny*up[0]
            tl = math.sqrt(tx*tx+ty*ty+tz*tz)
            if tl < 1e-6: continue
            tx/=tl; ty/=tl; tz/=tl
            bx = ny*tz-nz*ty; by_ = nz*tx-nx*tz; bz = nx*ty-ny*tx
            # tiny quad offset slightly off surface
            off = 0.003
            ox=px+nx*off; oy=py+ny*off; oz=pz+nz*off
            glBegin(GL_QUADS)
            glVertex3f(ox-tx*SIZE-bx*SIZE, oy-ty*SIZE-by_*SIZE, oz-tz*SIZE-bz*SIZE)
            glVertex3f(ox+tx*SIZE-bx*SIZE, oy+ty*SIZE-by_*SIZE, oz+tz*SIZE-bz*SIZE)
            glVertex3f(ox+tx*SIZE+bx*SIZE, oy+ty*SIZE+by_*SIZE, oz+tz*SIZE+bz*SIZE)
            glVertex3f(ox-tx*SIZE+bx*SIZE, oy-ty*SIZE+by_*SIZE, oz-tz*SIZE+bz*SIZE)
            glEnd()
        glDisable(GL_POLYGON_OFFSET_FILL)
        glEnable(GL_LIGHTING)

BULLET_HOLES = BulletHoles()

# ══════════════════════════════════════════════════════════════════════════════
# MUZZLE FLASH
# ══════════════════════════════════════════════════════════════════════════════
class MuzzleFlash:
    def __init__(self):
        self.frames = 0

    def fire(self):
        self.frames = 3

    def update(self):
        if self.frames > 0: self.frames -= 1

    def draw(self):
        if self.frames == 0: return
        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
        gluPerspective(75.0, W/H, 0.01, 10.0)
        glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
        glDisable(GL_LIGHTING); glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        alpha = self.frames / 3.0
        glColor4f(1.0, 0.85, 0.3, alpha * 0.7)
        # small star at muzzle position in viewmodel space
        cx = GUN_POS[0] + 0.01
        cy = GUN_POS[1] + 0.04
        cz = GUN_POS[2] - 0.05
        r = 0.018
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(cx, cy, cz)
        for i in range(9):
            a = i * math.pi * 2 / 8
            rr = r if i % 2 == 0 else r * 0.45
            glVertex3f(cx + math.cos(a)*rr, cy + math.sin(a)*rr, cz)
        glEnd()
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING)
        glMatrixMode(GL_MODELVIEW);  glPopMatrix()
        glMatrixMode(GL_PROJECTION); glPopMatrix()

MUZZLE_FLASH = MuzzleFlash()

# ══════════════════════════════════════════════════════════════════════════════
# GUN STATE  (ammo, recoil, reload animation)
# ══════════════════════════════════════════════════════════════════════════════
class GunState:
    def __init__(self):
        self.ammo          = GUN_MAG
        self.fire_cd       = 0
        self.reloading     = False
        self.reload_timer  = 0

        # raw kick targets — set instantly on event
        self._recoil_pitch_tgt   = 0.0
        self._recoil_y_tgt       = 0.0

        # smoothed values actually used for rendering — lerp toward targets
        self.recoil_pitch        = 0.0
        self.recoil_offset_y     = 0.0
        self.reload_offset       = 0.0
        self.reload_rot          = 0.0

        # smooth jump drop
        self._jump_drop_tgt      = 0.0
        self.jump_drop_smooth    = 0.0

        # smooth speed lag
        self._lag_tgt            = 0.0
        self.lag_smooth          = 0.0

    def try_fire(self, player):
        if self.reloading or self.fire_cd > 0 or self.ammo <= 0:
            return None
        self.ammo    -= 1
        self.fire_cd  = GUN_FIRE_CD

        # instant kick targets — rendering lerps toward these
        self._recoil_pitch_tgt += RECOIL_PITCH
        self._recoil_y_tgt     += 0.06

        # tiny camera nudge
        player.pitch_target -= RECOIL_PITCH * 0.08

        MUZZLE_FLASH.fire()

        ld = player.look_dir()
        eye = player.eye_pos()
        result = _raycast_hitscan(eye[0], eye[1], eye[2], ld[0], ld[1], ld[2])
        if result:
            hx, hy, hz, dist = result
            nx, ny_n, nz = -ld[0], -ld[1], -ld[2]
            BULLET_HOLES.add((hx, hy, hz), (nx, ny_n, nz))

        if self.ammo == 0:
            self.try_reload()

        return result

    def try_reload(self):
        if self.reloading or self.ammo == GUN_MAG: return
        self.reloading    = True
        self.reload_timer = GUN_RELOAD_DUR

    def update(self, player):
        if self.fire_cd > 0: self.fire_cd -= 1

        # ── recoil: snap target up, lerp render value toward it, then decay target ──
        self.recoil_pitch    = lerp(self.recoil_pitch,    self._recoil_pitch_tgt, 0.35)
        self.recoil_offset_y = lerp(self.recoil_offset_y, self._recoil_y_tgt,     0.35)
        self._recoil_pitch_tgt = lerp(self._recoil_pitch_tgt, 0.0, 0.08)
        self._recoil_y_tgt     = lerp(self._recoil_y_tgt,     0.0, 0.08)

        # ── reload animation ──────────────────────────────────────────────────
        if self.reloading:
            self.reload_timer -= 1
            t = 1.0 - self.reload_timer / GUN_RELOAD_DUR
            arc = math.sin(t * math.pi)   # smooth 0→1→0 arc
            tgt_off = RELOAD_KICK_X   * arc
            tgt_rot = RELOAD_KICK_ROT * arc
            self.reload_offset = lerp(self.reload_offset, tgt_off, 0.18)
            self.reload_rot    = lerp(self.reload_rot,    tgt_rot, 0.18)
            if self.reload_timer <= 0:
                self.reloading = False
                self.ammo      = GUN_MAG
        else:
            self.reload_offset = lerp(self.reload_offset, 0.0, 0.12)
            self.reload_rot    = lerp(self.reload_rot,    0.0, 0.12)

        # ── jump drop: smooth target based on airborne state ─────────────────
        self._jump_drop_tgt = -0.04 if not player.on_ground else 0.0
        self.jump_drop_smooth = lerp(self.jump_drop_smooth, self._jump_drop_tgt, 0.08)

        # ── speed lag: smooth ─────────────────────────────────────────────────
        spd_h = math.sqrt(player.vel[0]**2 + player.vel[2]**2)
        self._lag_tgt  = spd_h / MAX_GROUND_SPD
        self.lag_smooth = lerp(self.lag_smooth, self._lag_tgt, 0.10)

        MUZZLE_FLASH.update()

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
# MOVEMENT HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def pm_accelerate(vel_xz, wish_dir_xz, wish_speed, accel):
    cur = vel_xz[0]*wish_dir_xz[0] + vel_xz[1]*wish_dir_xz[1]
    add = wish_speed - cur
    if add <= 0:
        return vel_xz
    gain = min(accel, add)
    return [vel_xz[0] + gain*wish_dir_xz[0],
            vel_xz[1] + gain*wish_dir_xz[1]]

def pm_friction(vel_xz, scale):
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
        self.yaw_target   = 0.0
        self.pitch_target = 0.0

        self.on_ground  = False
        self.on_wall    = False
        self.coyote     = 0
        self.jump_held  = False
        self.jump_buffer= 0
        self.grounded_frames = 0
        self.just_jumped     = False

        self.slide_t    = 0
        self.slide_cd   = 0
        self.sliding    = False

        self.bob        = 0.0
        self.bob_amp    = 0.0
        self.tilt       = 0.0
        self.land_squish= 0.0
        self.fov_current= BASE_FOV
        self.eye_y_smooth = None

        self._was_on_ground = False

    def eye_pos(self):
        slide_off = -0.45 if self.sliding else 0.0
        bob_off   = math.sin(self.bob) * self.bob_amp
        target_y  = self.pos[1] + EYE + slide_off
        if self.eye_y_smooth is None:
            self.eye_y_smooth = target_y
        else:
            self.eye_y_smooth = lerp(self.eye_y_smooth, target_y, 0.22)
        return [self.pos[0],
                self.eye_y_smooth + bob_off - self.land_squish,
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
        CAM_SMOOTH = 0.18
        self.yaw_target   = (self.yaw_target + inp.get("dy", 0)) % 360
        self.pitch_target = clamp(self.pitch_target + inp.get("dp", 0), -89, 89)
        dyaw = (self.yaw_target - self.yaw + 180) % 360 - 180
        self.yaw   = (self.yaw + dyaw * CAM_SMOOTH) % 360
        self.pitch = lerp(self.pitch, self.pitch_target, CAM_SMOOTH)

        yr = math.radians(self.yaw_target)
        fw = [math.sin(yr), 0.0, -math.cos(yr)]
        rt = [math.cos(yr), 0.0,  math.sin(yr)]
        wx = wz = 0.0
        if inp.get("w"): wx += fw[0]; wz += fw[2]
        if inp.get("s"): wx -= fw[0]; wz -= fw[2]
        if inp.get("d"): wx += rt[0]; wz += rt[2]
        if inp.get("a"): wx -= rt[0]; wz -= rt[2]
        wlen = math.sqrt(wx*wx + wz*wz)
        if wlen > 0: wx /= wlen; wz /= wlen

        if inp.get("jump_press"):
            self.jump_buffer = JUMP_BUFFER

        if inp.get("slide") and self.on_ground and self.slide_cd == 0 and wlen > 0 and not self.sliding:
            self.sliding  = True
            self.slide_t  = SLIDE_DUR
            self.slide_cd = SLIDE_CD
            self.vel[0]   = wx * SLIDE_VEL
            self.vel[2]   = wz * SLIDE_VEL

        can_jump = self.on_ground or self.coyote > 0
        if self.jump_buffer > 0 and can_jump:
            self.vel[1]          = JUMP_VEL
            self.coyote          = 0
            self.jump_held       = True
            self.jump_buffer     = 0
            self.grounded_frames = 0
            self.just_jumped     = True
            self.sliding  = False
            self.slide_t  = 0

        if self.jump_held and not inp.get("jump_held") and self.vel[1] > 0:
            self.vel[1] *= JUMP_CUT
            self.jump_held = False
        elif not inp.get("jump_held"):
            self.jump_held = False

        if self.sliding:
            self.vel[0] *= 0.93
            self.vel[2] *= 0.93
        elif self.on_ground:
            vxz = pm_friction([self.vel[0], self.vel[2]], GROUND_FRICTION)
            if wlen > 0:
                vxz = pm_accelerate(vxz, [wx, wz], MAX_GROUND_SPD, GROUND_ACCEL)
            self.vel[0] = vxz[0]
            self.vel[2] = vxz[1]
        else:
            if wlen > 0:
                vxz = pm_accelerate(
                    [self.vel[0], self.vel[2]],
                    [wx, wz],
                    MAX_AIR_WISH,
                    AIR_ACCEL
                )
                self.vel[0] = vxz[0]
                self.vel[2] = vxz[1]

        spd_h = math.sqrt(self.vel[0]**2 + self.vel[2]**2)
        bob_target = spd_h * 0.018 if self.on_ground and not self.sliding else 0.0
        self.bob_amp = lerp(self.bob_amp, bob_target, 0.08)
        if self.on_ground and spd_h > 0.04:
            self.bob += spd_h * 2.5
        else:
            self.bob *= 0.90

        self.tilt = lerp(self.tilt, inp.get("dy", 0) * 0.55, 0.14)

        fov_target = BASE_FOV + SPEED_FOV_ADD * clamp(spd_h / MAX_GROUND_SPD, 0, 2.0)
        if self.sliding: fov_target = BASE_FOV + SPEED_FOV_ADD * 1.8
        self.fov_current = lerp(self.fov_current, fov_target, FOV_LERP)

        self.land_squish = lerp(self.land_squish, 0.0, 0.22)

    def update(self):
        if self.slide_t      > 0: self.slide_t  -= 1
        else:                      self.sliding   = False
        if self.slide_cd     > 0: self.slide_cd -= 1
        if self.coyote       > 0: self.coyote   -= 1
        if self.jump_buffer  > 0: self.jump_buffer -= 1

        self.vel[1] -= GRAVITY

        self._prev_pos = list(self.pos)
        new_pos = vadd(self.pos, self.vel)

        prev_g         = self.on_ground
        self.on_ground = False
        self.on_wall   = False

        if GLTF_MAP.parsed:
            dx      = new_pos[0] - self._prev_pos[0]
            dz      = new_pos[2] - self._prev_pos[2]
            dist    = math.sqrt(dx*dx + dz*dz)
            # FIX: include vertical distance in substep count so fast upward
            # jumps into low geometry don't skip past ceiling checks
            dy_     = new_pos[1] - self._prev_pos[1]
            dist3d  = math.sqrt(dx*dx + dy_*dy_ + dz*dz)
            n_steps = max(1, int(math.ceil(dist3d / (P_R * 0.5))))
            cur     = list(self._prev_pos)
            on_g = False; on_w = False

            for s in range(n_steps):
                frac    = (s + 1) / n_steps
                sub_new = [self._prev_pos[0] + dx * frac,
                           self._prev_pos[1] + (new_pos[1] - self._prev_pos[1]) * frac,
                           self._prev_pos[2] + dz * frac]

                on_g, on_w, cur, hit_ceil = GLTF_MAP.collide_capsule(
                    cur, sub_new, P_R, P_H, self.vel[1])

                if hit_ceil:
                    self.vel[1] = min(self.vel[1], 0.0)
                    self.jump_held   = False   # FIX: kill jump hold so no continued upward push
                    self.jump_buffer = 0        # FIX: eat buffered jump on ceiling contact

                if on_g and self.vel[1] < 0:
                    self.vel[1] = 0.0

                # FIX: only kill XZ velocity on a genuine wall block (not just
                # any on_wall flag), and only when NOT near the ground.
                # Previously any wall contact zeroed vel causing the stickiness.
                if on_w and not on_g:
                    if prev_g or on_g:
                        ls  = [cur[0], cur[1] + STEP_UP, cur[2]]
                        ld  = [sub_new[0], cur[1] + STEP_UP, sub_new[2]]
                        og2, ow2, cl2, _ = GLTF_MAP.collide_capsule(ls, ld, P_R, P_H, 0.0)
                        if not ow2:
                            fy = _raycast_down(cl2[0], cl2[1], cl2[2], STEP_UP + 0.1)
                            if fy is not None: cl2[1] = fy
                            cur = cl2; on_g = True; on_w = False
                            continue   # FIX: don't zero vel — we stepped up cleanly
                    # Genuine wall: zero XZ velocity
                    self.vel[0] = 0.0; self.vel[2] = 0.0
                    break

            self.pos = cur
            if on_g: self.on_ground = True
            self.on_wall = on_w

            if self.on_ground:
                self.grounded_frames = 2   # only 2 frames of grace, not 4
                self.just_jumped     = False
            else:
                if self.just_jumped:
                    self.grounded_frames = 0
                elif self.grounded_frames > 0:
                    self.grounded_frames -= 1
                    # Step-down: only fires the first frame off the ground,
                    # only when moving horizontally (walking off a step),
                    # and only when NOT in a real fall (vy not yet significant).
                    # The -0.015 threshold is ~5 frames of gravity accumulation —
                    # beyond that the player is genuinely falling, not on stairs.
                    if self.grounded_frames == 1 and \
                       -0.015 < self.vel[1] <= 0.0:
                        spd_h = math.sqrt(self.vel[0]**2 + self.vel[2]**2)
                        if spd_h > 0.001:
                            fy = _raycast_down(self.pos[0], self.pos[1],
                                               self.pos[2], STEP_DOWN)
                            if fy is not None and 0.001 < (self.pos[1] - fy) <= STEP_DOWN:
                                self.pos[1]    = fy
                                self.on_ground = True
                                self.vel[1]    = 0.0

        else:
            self.pos = new_pos
            for cx, cz, pw, pd, yt in FALLBACK_PLATFORMS:
                if (cx-pw/2-P_R < self.pos[0] < cx+pw/2+P_R and
                        cz-pd/2-P_R < self.pos[2] < cz+pd/2+P_R and
                        self.vel[1] <= 0 and
                        self.pos[1] <= yt+0.12 and self.pos[1] >= yt-0.65):
                    self.pos[1] = yt; self.vel[1] = 0.0; self.on_ground = True
            hw = ARENA_W/2-P_R; hd_ = ARENA_D/2-P_R
            if abs(self.pos[0]) > hw:
                self.vel[0] *= -0.3; self.pos[0] = clamp(self.pos[0], -hw, hw)
            if abs(self.pos[2]) > hd_:
                self.vel[2] *= -0.3; self.pos[2] = clamp(self.pos[2], -hd_, hd_)

        if not prev_g and self.on_ground:
            impact = clamp(-self.vel[1] * 3.0, 0.0, 0.28)
            self.land_squish = impact

        if prev_g and not self.on_ground:
            self.coyote = COYOTE_FRAMES

        self.vel[1] = clamp(self.vel[1], -2.0, JUMP_VEL * 1.1)
        if self.pos[1] > MAX_Y:
            self.pos[1] = MAX_Y
            self.vel[1] = min(self.vel[1], 0.0)

        floor_y = GLTF_MAP.aabb_min[1] if GLTF_MAP.aabb_min else 0.0
        if self.pos[1] < floor_y - 30:
            self.pos = list(spawn_pos())
            self.vel = v3(0, 0, 0)
            self.land_squish = 0.0
            self.sliding = False; self.slide_t = 0

        self._was_on_ground = self.on_ground

# ══════════════════════════════════════════════════════════════════════════════
# GL HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def spawn_pos():
    if GLTF_MAP.aabb_min and GLTF_MAP.aabb_max:
        mn, mx = GLTF_MAP.aabb_min, GLTF_MAP.aabb_max
        cx = (mn[0]+mx[0])/2
        cz = (mn[2]+mx[2])/2
        height_range = mx[1] - mn[1]
        cast_y = mx[1] + 2.0
        offsets = [(0,0),(2,0),(-2,0),(0,2),(0,-2),(4,4),(-4,4),(4,-4),(-4,-4)]
        best = None
        for ox, oz in offsets:
            fy = _raycast_down(cx+ox, cast_y, cz+oz, height_range + 4.0)
            if fy is not None:
                candidate = fy + P_H * 0.6
                if best is None or candidate > best[1]:
                    best = [cx+ox, candidate, cz+oz]
        if best:
            return best
        return [cx, mx[1] + 6.0, cz]
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

def draw_hud(player, font_sm, font_med, wire, gun_state):
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
    _hud_text(font_sm, "WASD move · Space jump · Ctrl slide · LMB shoot · R reload · F wire · ESC quit",
              (90,80,120), 14, H-42)

    if not player.on_ground:
        _hud_text(font_sm, "AIR", (80,180,255), W-60, 14)

    if player.sliding:
        _hud_text(font_med, "SLIDE", (180,80,255), W//2-42, H-54)

    if wire:
        _hud_text(font_sm, "WIREFRAME", (255,220,80), W-120, H-22)

    # ── ammo counter (bottom-right, CS style) ─────────────────────────────
    if gun_state.reloading:
        _hud_text(font_med, "RELOADING", (255, 200, 60), W-200, 18)
    else:
        ammo_col = (255,255,255) if gun_state.ammo > 3 else (255,80,60)
        _hud_text(font_med, f"{gun_state.ammo}  /  {GUN_MAG}", ammo_col, W-160, 18)

    bw, bh = 130, 14
    cx_ = W // 2
    bx  = cx_ - bw // 2
    by_ = H - 36
    glColor4f(0.06, 0.05, 0.12, 0.75)
    glBegin(GL_QUADS)
    glVertex2f(bx, by_); glVertex2f(bx+bw, by_)
    glVertex2f(bx+bw, by_+bh); glVertex2f(bx, by_+bh)
    glEnd()
    fill = int(bw * (1.0 - player.slide_cd / SLIDE_CD)) if SLIDE_CD > 0 else bw
    glColor4f(0.71, 0.31, 1.00, 0.85)
    glBegin(GL_QUADS)
    glVertex2f(bx, by_); glVertex2f(bx+fill, by_)
    glVertex2f(bx+fill, by_+bh); glVertex2f(bx, by_+bh)
    glEnd()
    _hud_text(font_sm, "SLIDE [Ctrl]", (220,220,220), cx_-36, 36-14)

    glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING)
    glMatrixMode(GL_PROJECTION); glPopMatrix()
    glMatrixMode(GL_MODELVIEW);  glPopMatrix()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════
def main():
    if "--debug" in sys.argv:
        path = next((a for a in sys.argv[1:] if not a.startswith("-")), "ex.gltf")
        debug_gltf(path); return

    gltf_path = next((a for a in sys.argv[1:] if not a.startswith("-")), "ex.gltf")

    pygame.init()
    pygame.display.set_mode((W, H), pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption("GLTF FPS Walker — Slide Edition")

    _font_tex_cache.clear()
    init_gl()
    font_sm  = pygame.font.SysFont("consolas", 15)
    font_med = pygame.font.SysFont("consolas", 26, bold=True)

    GLTF_MAP.parse(gltf_path)
    if GLTF_MAP.parsed:
        GLTF_MAP.upload()

    GUN_MODEL.load("gun.gltf")

    player    = Player(spawn_pos())
    gun_state = GunState()
    clock     = pygame.time.Clock()
    mc        = False
    wire      = False

    prev_jump   = False
    prev_slide  = False
    prev_shoot  = False

    rng = random.Random(77)
    amb = [(rng.uniform(-200,200), rng.uniform(10,80), rng.uniform(-200,200))
           for _ in range(300)]

    print("Click window to capture mouse. ESC releases; ESC again quits.")
    print("Controls: WASD | Space jump | Ctrl slide | LMB shoot | R reload")

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
                if event.key == pygame.K_r:
                    gun_state.try_reload()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if not mc:
                    pygame.mouse.set_visible(False)
                    pygame.event.set_grab(True); mc = True
                else:
                    gun_state.try_fire(player)

        dx, dy = pygame.mouse.get_rel()
        dyaw   =  dx * MOUSE_SENS if mc else 0.0
        dpitch = -dy * MOUSE_SENS if mc else 0.0
        keys   = pygame.key.get_pressed()

        jump_now  = bool(keys[pygame.K_SPACE])
        slide_now = bool(keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL])

        inp = {
            "w": bool(keys[pygame.K_w]),
            "s": bool(keys[pygame.K_s]),
            "a": bool(keys[pygame.K_a]),
            "d": bool(keys[pygame.K_d]),
            "jump_press": jump_now  and not prev_jump,
            "jump_held":  jump_now,
            "slide":      slide_now and not prev_slide,
            "dy": dyaw,
            "dp": dpitch,
        }
        prev_jump  = jump_now
        prev_slide = slide_now

        player.apply_input(inp)
        player.update()
        gun_state.update(player)

        # ── Render ────────────────────────────────────────────────────────────
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw_sky()
        set_camera(player)

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

        BULLET_HOLES.draw()

        # viewmodel + muzzle flash drawn last in their own projection pass
        GUN_MODEL.draw(player, gun_state)
        MUZZLE_FLASH.draw()

        draw_crosshair(player)
        draw_hud(player, font_sm, font_med, wire, gun_state)

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