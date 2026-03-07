/**
 * FPS Multiplayer Server (Node.js)
 *
 * npm install ws
 * node server.js [map.gltf] [--port 7777]
 *
 * Drop-in replacement for server.py — same WebSocket protocol,
 * same message types, same snapshot format.
 * Positions are trusted from the client; server only does
 * hitscan + wall-occlusion + health/kill tracking.
 */

"use strict";

const WebSocket = require("ws");
const fs        = require("fs");
const path      = require("path");

// ══════════════════════════════════════════════════════════════════════════════
// CONFIG  (must match client)
// ══════════════════════════════════════════════════════════════════════════════
const TICK_RATE      = 60;
const BROADCAST_RATE = 20;
const DEFAULT_PORT   = 7777;

const GRAVITY         = 0.0028;
const JUMP_VEL        = 0.10;
const P_H             = 1.8;
const P_R             = 0.38;
const EYE             = 1.62;
const MAP_SCALE       = 2.0;
const BULLET_DAMAGE   = 25;
const RESPAWN_TICKS   = 120;

// ══════════════════════════════════════════════════════════════════════════════
// MATH HELPERS
// ══════════════════════════════════════════════════════════════════════════════
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

function vec3(x, y, z) { return [x, y, z]; }
function vsub(a, b) { return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }
function vadd(a, b) { return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]; }
function vdot(a, b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
function vcross(a, b) {
  return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
}
function vlen(a) { return Math.sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]); }
function vnorm(a) {
  const l = vlen(a); return l < 1e-9 ? [0,0,0] : [a[0]/l, a[1]/l, a[2]/l];
}

// ══════════════════════════════════════════════════════════════════════════════
// TRIANGLE / SPATIAL GRID
// ══════════════════════════════════════════════════════════════════════════════
class TriGrid {
  constructor(cell = 3.0) {
    this.cell    = cell;
    this.buckets = new Map(); // "gx,gz" → int[]
    this.tris    = [];        // [ [[x,y,z],[x,y,z],[x,y,z]], ... ]
  }

  build(triangles) {
    this.tris    = triangles;
    this.buckets = new Map();
    const cell   = this.cell;
    for (let i = 0; i < triangles.length; i++) {
      const tri = triangles[i];
      let xMin = Infinity, xMax = -Infinity, zMin = Infinity, zMax = -Infinity;
      for (const v of tri) {
        if (v[0] < xMin) xMin = v[0]; if (v[0] > xMax) xMax = v[0];
        if (v[2] < zMin) zMin = v[2]; if (v[2] > zMax) zMax = v[2];
      }
      const x0 = Math.floor(xMin/cell), x1 = Math.floor(xMax/cell);
      const z0 = Math.floor(zMin/cell), z1 = Math.floor(zMax/cell);
      for (let gx = x0; gx <= x1; gx++) {
        for (let gz = z0; gz <= z1; gz++) {
          const k = `${gx},${gz}`;
          if (!this.buckets.has(k)) this.buckets.set(k, []);
          this.buckets.get(k).push(i);
        }
      }
    }
  }

  queryIndices(px, pz, radius) {
    const cell = this.cell;
    const r    = Math.ceil(radius / cell) + 1;
    const gx   = Math.floor(px / cell);
    const gz   = Math.floor(pz / cell);
    const seen = new Set();
    const out  = [];
    for (let dx = -r; dx <= r; dx++) {
      for (let dz = -r; dz <= r; dz++) {
        const arr = this.buckets.get(`${gx+dx},${gz+dz}`);
        if (!arr) continue;
        for (const i of arr) { if (!seen.has(i)) { seen.add(i); out.push(i); } }
      }
    }
    return out;
  }
}

const TRI_GRID = new TriGrid();

// ── point-in-triangle (barycentric) ──────────────────────────────────────────
function pointInTri(p, a, b, c) {
  const v0 = vsub(c, a), v1 = vsub(b, a), v2 = vsub(p, a);
  const d00 = vdot(v0,v0), d01 = vdot(v0,v1), d02 = vdot(v0,v2);
  const d11 = vdot(v1,v1), d12 = vdot(v1,v2);
  const denom = d00*d11 - d01*d01;
  if (Math.abs(denom) < 1e-12) return false;
  const inv = 1.0 / denom;
  const u   = (d11*d02 - d01*d12) * inv;
  const v   = (d00*d12 - d01*d02) * inv;
  return u >= -0.01 && v >= -0.01 && u+v <= 1.02;
}

// ── ray vs triangle (Möller–Trumbore) ────────────────────────────────────────
function rayTriDist(ro, rd, a, b, c) {
  const e1 = vsub(b, a), e2 = vsub(c, a);
  const h  = vcross(rd, e2);
  const det = vdot(e1, h);
  if (Math.abs(det) < 1e-9) return null;
  const invDet = 1.0 / det;
  const s  = vsub(ro, a);
  const u  = vdot(s, h) * invDet;
  if (u < -0.01 || u > 1.01) return null;
  const q  = vcross(s, e1);
  const v  = vdot(rd, q) * invDet;
  if (v < -0.01 || u+v > 1.02) return null;
  const t  = vdot(e2, q) * invDet;
  return t > 0.05 ? t : null;
}

// ── hitscan ray vs map geometry ───────────────────────────────────────────────
function raycastHitscan(ox, oy, oz, dx, dy, dz, maxDist = 400) {
  const tris = TRI_GRID.tris;
  if (!tris.length) return null;
  const dlen = Math.sqrt(dx*dx + dy*dy + dz*dz);
  if (dlen < 1e-9) return null;
  const rdx = dx/dlen, rdy = dy/dlen, rdz = dz/dlen;
  const ro = [ox, oy, oz], rd = [rdx, rdy, rdz];

  // Collect candidate triangles along ray path
  const steps = [0, maxDist*0.25, maxDist*0.5, maxDist*0.75, maxDist];
  const seen  = new Set();
  const idxs  = [];
  for (const s of steps) {
    for (const i of TRI_GRID.queryIndices(ox+rdx*s, oz+rdz*s, 6.0)) {
      if (!seen.has(i)) { seen.add(i); idxs.push(i); }
    }
  }

  let best = null;
  for (const i of idxs) {
    const [a, b, c] = tris[i];
    const t = rayTriDist(ro, rd, a, b, c);
    if (t !== null && t <= maxDist) {
      if (best === null || t < best) best = t;
    }
  }
  return best;
}

// ══════════════════════════════════════════════════════════════════════════════
// GLTF MAP LOADING  (geometry only, no GL)
// ══════════════════════════════════════════════════════════════════════════════
let aabbMin = null, aabbMax = null;

function loadMap(filePath) {
  if (!fs.existsSync(filePath)) {
    console.log(`[SERVER] Map '${filePath}' not found — no collision`);
    return false;
  }
  console.log(`[SERVER] Loading map '${filePath}' …`);

  let raw;
  try { raw = fs.readFileSync(filePath); }
  catch(e) { console.error("[SERVER] Failed to read map:", e.message); return false; }

  let gltf;
  // Handle both .gltf (JSON) and .glb (binary)
  if (filePath.endsWith(".glb")) {
    gltf = parseGLB(raw);
  } else {
    try { gltf = JSON.parse(raw.toString("utf8")); }
    catch(e) { console.error("[SERVER] Failed to parse GLTF JSON:", e.message); return false; }
  }

  const COMP = {5120:Int8Array,5121:Uint8Array,5122:Int16Array,5123:Uint16Array,5125:Uint32Array,5126:Float32Array};
  const TNCO = {SCALAR:1,VEC2:2,VEC3:3,VEC4:4,MAT2:4,MAT3:9,MAT4:16};

  function getBuf(bi) {
    const buf = gltf.buffers[bi];
    if (!buf.uri) return gltf._binaryBlob || Buffer.alloc(0);
    if (buf.uri.startsWith("data:")) {
      const b64 = buf.uri.split(",")[1];
      return Buffer.from(b64, "base64");
    }
    return fs.readFileSync(path.join(path.dirname(path.resolve(filePath)), buf.uri));
  }

  function readAcc(ai) {
    const acc = gltf.accessors[ai];
    const bvi = acc.bufferView ?? null;
    if (bvi === null) return new Float32Array(acc.count * (TNCO[acc.type]||1));
    const bv  = gltf.bufferViews[bvi];
    const raw = getBuf(bv.buffer);
    const Ctor = COMP[acc.componentType] || Float32Array;
    const nc   = TNCO[acc.type] || 1;
    const bo   = bv.byteOffset || 0;
    const ao   = acc.byteOffset || 0;
    const sz   = Ctor.BYTES_PER_ELEMENT * nc;
    const arr  = new Ctor(raw.buffer, raw.byteOffset + bo + ao, acc.count * nc);
    // Convert to float32
    if (Ctor === Float32Array) return arr;
    const f = new Float32Array(arr.length);
    for (let i = 0; i < arr.length; i++) f[i] = arr[i];
    return f;
  }

  function nodeMat(node) {
    if (node.matrix) {
      // column-major → row-major
      const m = node.matrix;
      return [
        [m[0],m[4],m[8], m[12]],
        [m[1],m[5],m[9], m[13]],
        [m[2],m[6],m[10],m[14]],
        [m[3],m[7],m[11],m[15]],
      ];
    }
    let M = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
    if (node.scale) {
      const [sx,sy,sz] = node.scale;
      M[0][0]=sx; M[1][1]=sy; M[2][2]=sz;
    }
    if (node.rotation) {
      const [qx,qy,qz,qw] = node.rotation;
      const R = [
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw),   0],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw),   0],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy), 0],
        [0,0,0,1]
      ];
      M = mat4mul(R, M);
    }
    if (node.translation) {
      const [tx,ty,tz] = node.translation;
      const T = [[1,0,0,tx],[0,1,0,ty],[0,0,1,tz],[0,0,0,1]];
      M = mat4mul(T, M);
    }
    return M;
  }

  function mat4mul(A, B) {
    const C = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]];
    for (let i=0;i<4;i++) for (let j=0;j<4;j++)
      for (let k=0;k<4;k++) C[i][j] += A[i][k]*B[k][j];
    return C;
  }

  function transformPt(M, p) {
    const x=M[0][0]*p[0]+M[0][1]*p[1]+M[0][2]*p[2]+M[0][3];
    const y=M[1][0]*p[0]+M[1][1]*p[1]+M[1][2]*p[2]+M[1][3];
    const z=M[2][0]*p[0]+M[2][1]*p[1]+M[2][2]*p[2]+M[2][3];
    return [x*MAP_SCALE, y*MAP_SCALE, z*MAP_SCALE];
  }

  const sceneIdx = gltf.scene ?? 0;
  const roots    = (gltf.scenes && gltf.scenes[sceneIdx]?.nodes) || [];
  const identity = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
  const stack    = roots.map(ni => [ni, identity]);
  const triangles = [];
  const allPts    = [];

  while (stack.length) {
    const [ni, pm] = stack.pop();
    const node = gltf.nodes[ni];
    if (!node) continue;
    const mat = mat4mul(pm, nodeMat(node));
    for (const ci of (node.children || [])) stack.push([ci, mat]);
    if (node.mesh == null) continue;
    for (const prim of gltf.meshes[node.mesh].primitives) {
      if (prim.attributes.POSITION == null) continue;
      const posF = readAcc(prim.attributes.POSITION);
      const nv   = posF.length / 3;
      const pts  = [];
      for (let i = 0; i < nv; i++) {
        const p = transformPt(mat, [posF[i*3], posF[i*3+1], posF[i*3+2]]);
        pts.push(p);
        allPts.push(p);
      }
      let idxArr;
      if (prim.indices != null) {
        const raw = readAcc(prim.indices);
        idxArr = Array.from(raw);
      } else {
        idxArr = Array.from({length: nv}, (_,i) => i);
      }
      const mode = prim.mode ?? 4;
      if (mode === 4) {
        for (let i = 0; i+2 < idxArr.length; i += 3)
          triangles.push([pts[idxArr[i]], pts[idxArr[i+1]], pts[idxArr[i+2]]]);
      } else if (mode === 5) { // triangle strip
        for (let i = 0; i+2 < idxArr.length; i++) {
          if (i%2===0) triangles.push([pts[idxArr[i]], pts[idxArr[i+1]], pts[idxArr[i+2]]]);
          else         triangles.push([pts[idxArr[i+1]], pts[idxArr[i]], pts[idxArr[i+2]]]);
        }
      } else if (mode === 6) { // triangle fan
        for (let i = 1; i+1 < idxArr.length; i++)
          triangles.push([pts[idxArr[0]], pts[idxArr[i]], pts[idxArr[i+1]]]);
      }
    }
  }

  if (!allPts.length) { console.log("[SERVER] No geometry found"); return false; }

  aabbMin = [Infinity,Infinity,Infinity];
  aabbMax = [-Infinity,-Infinity,-Infinity];
  for (const p of allPts) {
    for (let i=0;i<3;i++) {
      if (p[i] < aabbMin[i]) aabbMin[i] = p[i];
      if (p[i] > aabbMax[i]) aabbMax[i] = p[i];
    }
  }

  TRI_GRID.build(triangles);
  console.log(`[SERVER] Map loaded — ${triangles.length} triangles`);
  return true;
}

function parseGLB(buf) {
  // GLB header: magic(4) version(4) length(4)
  const magic = buf.readUInt32LE(0);
  if (magic !== 0x46546C67) throw new Error("Not a GLB file");
  let offset = 12;
  let jsonChunk = null, binChunk = null;
  while (offset < buf.length) {
    const chunkLen  = buf.readUInt32LE(offset);
    const chunkType = buf.readUInt32LE(offset+4);
    const chunkData = buf.slice(offset+8, offset+8+chunkLen);
    if (chunkType === 0x4E4F534A) jsonChunk = chunkData;
    else if (chunkType === 0x004E4942) binChunk = chunkData;
    offset += 8 + chunkLen;
  }
  const gltf = JSON.parse(jsonChunk.toString("utf8"));
  if (binChunk) gltf._binaryBlob = binChunk;
  return gltf;
}

// ══════════════════════════════════════════════════════════════════════════════
// SPAWN POSITION
// ══════════════════════════════════════════════════════════════════════════════
function spawnPos() {
  if (aabbMin && aabbMax) {
    const cx = (aabbMin[0]+aabbMax[0])/2;
    const cz = (aabbMin[2]+aabbMax[2])/2;
    return [cx, aabbMax[1]+4.0, cz];
  }
  return [0, 4, 0];
}

// ══════════════════════════════════════════════════════════════════════════════
// PLAYER
// ══════════════════════════════════════════════════════════════════════════════
class ServerPlayer {
  constructor(pid, name) {
    this.pid    = pid;
    this.name   = name;
    this.pos    = spawnPos();
    this.yaw    = 0;
    this.pitch  = 0;
    this.health = 100;
    this.alive  = true;
    this.kills  = 0;
    this.deaths = 0;
    this.respawnTimer = 0;
    this.sliding = false;
    this.inp    = null;
    this.ws     = null;
  }

  takeDamage(amount) {
    if (!this.alive) return;
    this.health = Math.max(0, this.health - amount);
    if (this.health === 0) {
      this.alive = false;
      this.deaths++;
      this.respawnTimer = RESPAWN_TICKS;
    }
  }

  update() {
    if (!this.alive) {
      if (this.respawnTimer > 0) this.respawnTimer--;
      if (this.respawnTimer === 0) {
        this.alive  = true;
        this.health = 100;
        this.pos    = spawnPos();
      }
    }
    // Update yaw/pitch from latest input
    if (this.inp) {
      if (this.inp.yaw   != null) this.yaw   = this.inp.yaw;
      if (this.inp.pitch != null) this.pitch = this.inp.pitch;
      if (this.inp.x     != null) {
        this.pos[0] = this.inp.x;
        this.pos[1] = this.inp.y;
        this.pos[2] = this.inp.z;
      }
      if (this.inp.slide != null) this.sliding = this.inp.slide;
    }
  }

  snapshot() {
    return {
      pid:      this.pid,
      name:     this.name,
      x:        +this.pos[0].toFixed(3),
      y:        +this.pos[1].toFixed(3),
      z:        +this.pos[2].toFixed(3),
      yaw:      +this.yaw.toFixed(2),
      pitch:    +this.pitch.toFixed(2),
      sliding:  this.sliding,
      health:   this.health,
      alive:    this.alive,
      kills:    this.kills,
      deaths:   this.deaths,
    };
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// HITSCAN
// ══════════════════════════════════════════════════════════════════════════════
function processShoot(shooter, players, yaw, pitch) {
  const yr = yaw   * Math.PI / 180;
  const pr = pitch * Math.PI / 180;
  const dx =  Math.cos(pr) * Math.sin(yr);
  const dy =  Math.sin(pr);
  const dz = -Math.cos(pr) * Math.cos(yr);
  const eye = [shooter.pos[0], shooter.pos[1] + EYE*0.9, shooter.pos[2]];

  const hits = [];
  for (const [pid, p] of players) {
    if (pid === shooter.pid || !p.alive) continue;
    const [cx, cy, cz] = p.pos;
    const capR = P_R * 1.1, capH = P_H;
    const ox = eye[0]-cx, oz = eye[2]-cz;
    const a = dx*dx+dz*dz;
    const b = 2*(ox*dx+oz*dz);
    const c = ox*ox+oz*oz - capR*capR;
    const disc = b*b - 4*a*c;
    if (disc < 0 || a < 1e-9) continue;
    let t = (-b - Math.sqrt(disc)) / (2*a);
    if (t < 0) t = (-b + Math.sqrt(disc)) / (2*a);
    if (t < 0 || t > 300) continue;
    const hy = eye[1] + dy*t;
    if (hy < cy || hy > cy+capH) continue;
    // Wall occlusion
    const wallT = raycastHitscan(eye[0], eye[1], eye[2], dx, dy, dz, t+0.5);
    if (wallT !== null && wallT < t - 0.3) continue;
    hits.push([t, pid]);
  }

  hits.sort((a,b) => a[0]-b[0]);
  const hitPids = [];
  for (const [, pid] of hits) {
    const p = players.get(pid);
    p.takeDamage(BULLET_DAMAGE);
    hitPids.push(pid);
    if (p.health <= 0) shooter.kills++;
    break; // one hit per shot
  }
  return hitPids;
}

// ══════════════════════════════════════════════════════════════════════════════
// SERVER STATE
// ══════════════════════════════════════════════════════════════════════════════
const players     = new Map();  // pid → ServerPlayer
let   nextPid     = 1;
const eventsQueue = [];

function broadcast(msg) {
  const data = JSON.stringify(msg);
  for (const [, p] of players) {
    if (p.ws && p.ws.readyState === WebSocket.OPEN) {
      try { p.ws.send(data); } catch(_) {}
    }
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// GAME LOOP
// ══════════════════════════════════════════════════════════════════════════════
const TICK_MS      = 1000 / TICK_RATE;
const BCAST_MS     = 1000 / BROADCAST_RATE;
let   lastBcast    = Date.now();

function gameTick() {
  // Update all players (position comes from client via inp)
  for (const [, p] of players) p.update();

  // Broadcast at reduced rate
  const now = Date.now();
  if (now - lastBcast >= BCAST_MS) {
    lastBcast = now;
    broadcast({
      type:    "state",
      tick:    now,
      players: [...players.values()].map(p => p.snapshot()),
      events:  eventsQueue.splice(0),
    });
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// WEBSOCKET HANDLER
// ══════════════════════════════════════════════════════════════════════════════
function onConnect(ws) {
  const pid    = String(nextPid++);
  const name   = `Player${pid}`;
  const player = new ServerPlayer(pid, name);
  player.ws    = ws;
  players.set(pid, player);
  console.log(`[+] ${name} connected  (total: ${players.size})`);

  ws.send(JSON.stringify({
    type:      "welcome",
    your_pid:  pid,
    spawn:     player.pos,
    tick_rate: TICK_RATE,
  }));

  ws.on("message", raw => {
    let msg;
    try { msg = JSON.parse(raw); } catch(_) { return; }
    const mtype = msg.type || "";

    if (mtype === "input") {
      player.inp = msg;

    } else if (mtype === "shoot") {
      const yaw   = parseFloat(msg.yaw   ?? player.yaw);
      const pitch = parseFloat(msg.pitch ?? player.pitch);
      const hitPids = processShoot(player, players, yaw, pitch);
      if (hitPids.length) {
        eventsQueue.push({ type:"hit", shooter:pid, targets:hitPids });
        for (const hpid of hitPids) {
          const hp = players.get(hpid);
          if (hp && !hp.alive) {
            eventsQueue.push({
              type:        "kill",
              killer:      pid,
              victim:      hpid,
              killer_name: player.name,
              victim_name: hp.name,
            });
          }
        }
      }

    } else if (mtype === "name") {
      player.name = String(msg.name || "").slice(0, 20);
    }
  });

  ws.on("close", () => {
    console.log(`[-] ${name} disconnected`);
    players.delete(pid);
  });

  ws.on("error", err => console.error(`[NET] ${name} error:`, err.message));
}

// ══════════════════════════════════════════════════════════════════════════════
// ENTRY POINT
// ══════════════════════════════════════════════════════════════════════════════
const args = process.argv.slice(2);
let   mapFile  = "ex.gltf";
let   port     = DEFAULT_PORT;
for (let i = 0; i < args.length; i++) {
  if (args[i] === "--port" && args[i+1]) { port = parseInt(args[++i]); }
  else if (!args[i].startsWith("--"))    { mapFile = args[i]; }
}

loadMap(mapFile);

const wss = new WebSocket.Server({ port });
wss.on("connection", onConnect);
setInterval(gameTick, TICK_MS);

console.log(`[SERVER] Listening on ws://0.0.0.0:${port}`);