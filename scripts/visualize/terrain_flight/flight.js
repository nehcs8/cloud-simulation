// Alps terrain flight: free-flight / walk viewer over the Copernicus GLO-30 tile pyramid.
// Chunks of 225 x 225 cells, quadtree LOD (L6 = 4 deg ... L0 = 1 arcsec), heights decoded from
// Terrarium PNGs into float textures, displaced in the vertex shader. Skirts hide LOD cracks.
import * as THREE from 'three';

const meta = await (await fetch('tiles/meta.json')).json();
const DEG = 6371000 * Math.PI / 180;                       // m per degree of latitude
const LAT0 = meta.lat0, LON0 = meta.lon0;
const KX = DEG * Math.cos(LAT0 * Math.PI / 180);            // m per degree of longitude (equirectangular)
const C = meta.chunk, NS = C + 1, PPD = meta.ppd, NL = meta.levels.length;

// sample (r, c) of level L -> lon/lat (see build_tiles.py)
const lonOf = (c, L) => meta.west + ((c + 0.5) * 2 ** L - 0.5) / PPD;
const latOf = (r, L) => meta.north - ((r + 0.5) * 2 ** L - 0.5) / PPD;
const xOfLon = lon => (lon - LON0) * KX, zOfLat = lat => -(lat - LAT0) * DEG;   // north = -z
const lonOfX = x => LON0 + x / KX, latOfZ = z => LAT0 - z / DEG;

const PLACES = [  // name, lat, lon, elevation, bearing from the target to the camera (deg, 0 = from north)
  ['Matterhorn', 45.976, 7.658, 4478, 45], ['Monte Rosa', 45.937, 7.867, 4634, 0],
  ['Mont Blanc', 45.833, 6.865, 4808, 330], ['Chamonix', 45.924, 6.870, 1035, 45],
  ['Eiger north face', 46.577, 8.005, 3967, 0], ['Jungfrau', 46.537, 7.963, 4158, 330],
  ['Zermatt', 46.020, 7.749, 1608, 20], ['Aletsch glacier', 46.450, 8.050, 2500, 160],
  ['St. Moritz', 46.498, 9.838, 1822, 330], ['Bernina', 46.382, 9.908, 4049, 0],
  ['Gotthard pass', 46.559, 8.561, 2106, 180], ['Grossglockner', 47.074, 12.694, 3798, 120],
  ['Zugspitze', 47.421, 10.985, 2962, 0], ['Innsbruck', 47.269, 11.404, 574, 0],
  ['Ortler', 46.509, 10.545, 3905, 60], ['Marmolada', 46.434, 11.851, 3343, 0],
  ['Tre Cime', 46.619, 12.302, 2999, 0], ['Cortina', 46.540, 12.136, 1224, 180],
  ['Bolzano', 46.498, 11.354, 262, 200], ['Lake Garda', 45.650, 10.700, 65, 180],
  ['Lake Como', 46.000, 9.250, 199, 180], ['Lugano', 46.004, 8.951, 273, 180],
  ['Aosta', 45.737, 7.320, 583, 160], ['Gran Paradiso', 45.517, 7.267, 4061, 180],
  ['Monviso', 44.667, 7.090, 3841, 90], ['Ecrins', 44.922, 6.360, 4102, 200],
  ['Grenoble', 45.188, 5.724, 214, 270], ['Triglav', 46.378, 13.837, 2864, 180],
  ['Dachstein', 47.475, 13.606, 2995, 0],
];

// ---------------------------------------------------------------- renderer, sky
const canvas = document.getElementById('view');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, logarithmicDepthBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
renderer.setSize(innerWidth, innerHeight, false);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(70, innerWidth / innerHeight, 1, 900000);
camera.rotation.order = 'YXZ';
addEventListener('resize', () => {
  renderer.setSize(innerWidth, innerHeight, false);
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
});

const HAZE = [1 / 120000, 1 / 45000, 1 / 400000];
let hazeIdx = 0;
const shared = {
  sunDir: { value: new THREE.Vector3() },
  hazeColor: { value: new THREE.Color(0.73, 0.81, 0.90) },
  hazeDensity: { value: HAZE[0] },
  sunMode: { value: 0 },
};
let sunAz = 150, sunEl = 35;   // degrees, azimuth clockwise from north
function setSun() {
  const a = sunAz * Math.PI / 180, e = sunEl * Math.PI / 180;
  shared.sunDir.value.set(Math.sin(a) * Math.cos(e), Math.sin(e), -Math.cos(a) * Math.cos(e));
}
setSun();

const sky = new THREE.Mesh(new THREE.SphereGeometry(1000, 32, 16), new THREE.ShaderMaterial({
  uniforms: { sunDir: shared.sunDir, hazeColor: shared.hazeColor },
  vertexShader: `varying vec3 vDir;
    void main() { vDir = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: `uniform vec3 sunDir; uniform vec3 hazeColor; varying vec3 vDir;
    void main() {
      vec3 d = normalize(vDir);
      vec3 zenith = vec3(0.25, 0.46, 0.80);
      vec3 c = d.y > 0.0 ? mix(hazeColor, zenith, pow(d.y, 0.5)) : hazeColor * (1.0 + 0.3 * d.y);
      float s = max(dot(d, sunDir), 0.0);
      c += vec3(1.0, 0.92, 0.75) * (pow(s, 900.0) * 3.0 + pow(s, 16.0) * 0.18);
      gl_FragColor = vec4(c, 1.0);
    }`,
  side: THREE.BackSide, depthWrite: false, depthTest: false,
}));
sky.renderOrder = -1; sky.frustumCulled = false;
scene.add(sky);

// ---------------------------------------------------------------- terrain shaders, shared grid
const VERT = `
#include <common>
#include <logdepthbuf_pars_vertex>
uniform sampler2D hmap; uniform vec2 origin; uniform vec2 stepv; uniform float skirtDepth;
uniform vec4 sunXf;
attribute vec2 grid; attribute float skirt;
varying vec3 vWorld; varying vec3 vNormal; varying float vFlat; varying vec2 vSunUV;
float H(ivec2 p) { p = clamp(p, ivec2(0), ivec2(${C})); return texelFetch(hmap, p, 0).r; }
void main() {
  ivec2 p = ivec2(grid + 0.5);
  float h = H(p);
  ivec2 a = clamp(p - 1, ivec2(0), ivec2(${C})), b = clamp(p + 1, ivec2(0), ivec2(${C}));
  float dhx = (H(ivec2(b.x, p.y)) - H(ivec2(a.x, p.y))) / (float(b.x - a.x) * stepv.x);
  float dhz = (H(ivec2(p.x, b.y)) - H(ivec2(p.x, a.y))) / (float(b.y - a.y) * stepv.y);
  vNormal = normalize(vec3(-dhx, 1.0, -dhz));
  vFlat = (abs(dhx) + abs(dhz) < 1e-4) ? 1.0 : 0.0;           // lakes are exactly flat in GLO-30
  vWorld = vec3(origin.x + grid.x * stepv.x, h - skirt * skirtDepth, origin.y + grid.y * stepv.y);
  vSunUV = sunXf.xy + grid * sunXf.zw;
  gl_Position = projectionMatrix * viewMatrix * vec4(vWorld, 1.0);
  #include <logdepthbuf_vertex>
}`;
const FRAG = `
#include <common>
#include <logdepthbuf_pars_fragment>
uniform vec3 sunDir; uniform vec3 hazeColor; uniform float hazeDensity;
uniform sampler2D sunTex; uniform float sunMode;
varying vec3 vWorld; varying vec3 vNormal; varying float vFlat; varying vec2 vSunUV;
vec3 magma(float t) {
  t = clamp(t, 0.0, 1.0) * 4.0;
  vec3 c0 = vec3(0.03, 0.02, 0.10), c1 = vec3(0.32, 0.07, 0.43), c2 = vec3(0.72, 0.21, 0.47),
       c3 = vec3(0.98, 0.47, 0.36), c4 = vec3(0.99, 0.95, 0.72);
  return t < 1.0 ? mix(c0, c1, t) : t < 2.0 ? mix(c1, c2, t - 1.0) : t < 3.0 ? mix(c2, c3, t - 2.0) : mix(c3, c4, t - 3.0);
}
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float vnoise(vec2 p) {
  vec2 i = floor(p), f = fract(p), u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1, 0)), u.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), u.x), u.y);
}
vec3 lin(vec3 c) { return pow(c, vec3(2.2)); }
void main() {
  #include <logdepthbuf_fragment>
  vec3 n = normalize(vNormal);
  float h = vWorld.y;
  float nz = 0.6 * vnoise(vWorld.xz / 450.0) + 0.4 * vnoise(vWorld.xz / 70.0);
  float hh = h + (nz - 0.5) * 300.0;
  vec3 c = mix(vec3(0.43, 0.53, 0.29), vec3(0.20, 0.31, 0.17), smoothstep(450.0, 900.0, hh));  // fields -> forest
  c = mix(c, vec3(0.52, 0.57, 0.34), smoothstep(1750.0, 2050.0, hh));                         // alpine meadow
  c = mix(c, vec3(0.60, 0.58, 0.54), smoothstep(2350.0, 2800.0, hh));                         // scree
  c = mix(c, vec3(0.47, 0.44, 0.41), smoothstep(0.86, 0.70, n.y));                            // steep = rock
  c *= 0.88 + 0.24 * nz;
  float snow = smoothstep(2950.0, 3300.0, hh) * smoothstep(0.62, 0.82, n.y);
  c = mix(c, vec3(0.96, 0.97, 1.0), snow);
  bool water = h < 0.5 || vFlat > 0.5;
  float land = 0.0;
  if (sunMode > 0.5) {                     // sun-hours / sunless-days overlay (scripts/sun_alps.py)
    vec4 sv = texture2D(sunTex, vSunUV);
    land = sv.b;
    if (sunMode < 1.5) c = mix(c, magma(sv.r * 255.0 / 254.0), 0.92 * land);
    else {
      float days = sv.g * 255.0;
      vec3 oc = mix(vec3(0.62, 0.78, 0.92), vec3(0.03, 0.15, 0.40), clamp(days / 150.0, 0.0, 1.0));
      c = mix(mix(c, vec3(0.85), 0.5 * land), oc, 0.92 * land * smoothstep(0.3, 1.0, days));
    }
  }
  if (water && land < 0.5) c = vec3(0.14, 0.29, 0.42);
  vec3 sunCol = vec3(1.0, 0.96, 0.88);
  float diff = max(dot(n, sunDir), 0.0);
  float amb = sunMode > 0.5 ? 0.55 : 0.28 + 0.18 * n.y, dk = sunMode > 0.5 ? 0.55 : 1.1;
  vec3 col = lin(c) * (amb * vec3(0.75, 0.85, 1.0) + dk * diff * sunCol);
  vec3 v = normalize(cameraPosition - vWorld);
  if (water) col += pow(max(dot(reflect(-sunDir, vec3(0, 1, 0)), v), 0.0), 300.0) * sunCol * 2.0;
  col = pow(col, vec3(1.0 / 2.2));
  float d = length(vWorld - cameraPosition);
  float f = 1.0 - exp(-d * hazeDensity * exp(-max(min(h, cameraPosition.y), 0.0) / 6000.0));
  gl_FragColor = vec4(mix(col, hazeColor, clamp(f, 0.0, 1.0)), 1.0);
}`;

function makeGeometry() {
  const nG = NS * NS, nV = nG + 4 * NS;
  const grid = new Float32Array(nV * 2), skirt = new Float32Array(nV);
  for (let j = 0; j < NS; j++) for (let i = 0; i < NS; i++) { const v = j * NS + i; grid[2 * v] = i; grid[2 * v + 1] = j; }
  const edges = [[], [], [], []];
  for (let t = 0; t < NS; t++) {
    edges[0].push(t); edges[1].push(C * NS + t); edges[2].push(t * NS); edges[3].push(t * NS + C);
  }
  const idx = [];
  for (let j = 0; j < C; j++) for (let i = 0; i < C; i++) {
    const v = j * NS + i; idx.push(v, v + NS, v + 1, v + 1, v + NS, v + NS + 1);
  }
  edges.forEach((e, k) => {
    const s0 = nG + k * NS;
    e.forEach((v, t) => { grid[2 * (s0 + t)] = grid[2 * v]; grid[2 * (s0 + t) + 1] = grid[2 * v + 1]; skirt[s0 + t] = 1; });
    for (let t = 0; t < C; t++) idx.push(e[t], e[t + 1], s0 + t + 1, e[t], s0 + t + 1, s0 + t);
  });
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(nV * 3), 3));  // unused, required by three
  g.setAttribute('grid', new THREE.BufferAttribute(grid, 2));
  g.setAttribute('skirt', new THREE.BufferAttribute(skirt, 1));
  g.setIndex(idx);
  return g;
}
const GEOM = makeGeometry();

// ---------------------------------------------------------------- chunk cache + LOD
const chunks = new Map();
const key = (L, r, c) => `${L}/${r}/${c}`;
function getChunk(L, r, c) {
  const k = key(L, r, c);
  let ch = chunks.get(k);
  if (!ch) {
    const lv = meta.levels[L], s = 2 ** L / PPD;
    const ox = xOfLon(lonOf(c * C, L)), oz = zOfLat(latOf(r * C, L)), dx = s * KX, dz = s * DEG;
    const box = new THREE.Box3(new THREE.Vector3(ox, lv.zmin[r][c] - 50, oz),
                               new THREE.Vector3(ox + C * dx, lv.zmax[r][c] + 50, oz + C * dz));
    ch = { L, r, c, k, ox, oz, dx, dz, box, size: C * dz, state: 'new', last: 0 };
    chunks.set(k, ch);
  }
  return ch;
}
const exists = (L, r, c) => L >= 0 && L < NL && r >= 0 && c >= 0 && r < meta.levels[L].rows && c < meta.levels[L].cols;

const decodeCtx = new OffscreenCanvas(NS, NS).getContext('2d', { willReadFrequently: true });
let inflight = 0;
const MAX_INFLIGHT = 12;
async function load(ch) {
  ch.state = 'loading'; inflight++;
  try {
    const res = await fetch(`tiles/${ch.k}.png`);
    if (!res.ok) throw new Error(res.status);
    const bmp = await createImageBitmap(await res.blob(), { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
    decodeCtx.drawImage(bmp, 0, 0); bmp.close();
    const px = decodeCtx.getImageData(0, 0, NS, NS).data;
    const hts = new Float32Array(NS * NS);
    for (let i = 0; i < NS * NS; i++) hts[i] = px[4 * i] * 256 + px[4 * i + 1] + px[4 * i + 2] / 256 - 32768;
    const tex = new THREE.DataTexture(hts, NS, NS, THREE.RedFormat, THREE.FloatType);
    tex.minFilter = tex.magFilter = THREE.NearestFilter; tex.needsUpdate = true;
    const mat = new THREE.ShaderMaterial({
      uniforms: {
        hmap: { value: tex }, origin: { value: new THREE.Vector2(ch.ox, ch.oz) },
        stepv: { value: new THREE.Vector2(ch.dx, ch.dz) }, skirtDepth: { value: 40 + 8 * ch.dz },
        sunDir: shared.sunDir, hazeColor: shared.hazeColor, hazeDensity: shared.hazeDensity,
        sunMode: shared.sunMode, sunTex: { value: null }, sunXf: { value: new THREE.Vector4() },
      },
      vertexShader: VERT, fragmentShader: FRAG, side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(GEOM, mat);
    mesh.frustumCulled = false; mesh.matrixAutoUpdate = false; mesh.visible = false;
    scene.add(mesh);
    Object.assign(ch, { mesh, tex, mat, heights: hts, state: 'ready' });
  } catch (e) {
    ch.state = 'missing';
  } finally { inflight--; }
}
function evict(frame) {
  if (chunks.size < 900) return;
  const old = [...chunks.values()].filter(ch => ch.state === 'ready' && ch.L < NL - 1 && ch.last < frame - 60)
    .sort((a, b) => a.last - b.last);
  for (const ch of old.slice(0, chunks.size - 700)) {
    scene.remove(ch.mesh); ch.tex.dispose(); ch.mat.dispose(); chunks.delete(ch.k);
  }
}

const SPLIT = [1.4, 2.2, 3.5];
let splitIdx = 1;
const frustum = new THREE.Frustum(), pv = new THREE.Matrix4();
let drawn = [];
function updateLOD(frame) {
  pv.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
  frustum.setFromProjectionMatrix(pv);
  const draw = [], want = [], K = SPLIT[splitIdx], P = camera.position;
  const need = ch => { if (ch.state === 'new') want.push([ch.box.distanceToPoint(P) / ch.size, ch]); };
  function visit(ch) {
    ch.last = frame;
    if (!frustum.intersectsBox(ch.box)) return;
    if (ch.state !== 'ready') { need(ch); return; }
    if (ch.L > 0 && ch.box.distanceToPoint(P) < K * ch.size) {
      const kids = [];
      for (let dr = 0; dr < 2; dr++) for (let dc = 0; dc < 2; dc++)
        if (exists(ch.L - 1, 2 * ch.r + dr, 2 * ch.c + dc)) kids.push(getChunk(ch.L - 1, 2 * ch.r + dr, 2 * ch.c + dc));
      let ready = true;
      for (const k of kids) {
        k.last = frame;
        if (!frustum.intersectsBox(k.box)) continue;
        if (k.state === 'new' || k.state === 'loading') { ready = false; need(k); }
      }
      if (ready) { kids.forEach(visit); return; }
    }
    draw.push(ch);
  }
  const top = meta.levels[NL - 1];
  for (let r = 0; r < top.rows; r++) for (let c = 0; c < top.cols; c++) visit(getChunk(NL - 1, r, c));
  for (const ch of drawn) ch.mesh.visible = false;
  for (const ch of draw) ch.mesh.visible = true;
  drawn = draw;
  want.sort((a, b) => a[0] - b[0]);
  for (const [, ch] of want) { if (inflight >= MAX_INFLIGHT) break; load(ch); }
  evict(frame);
}

function groundAt(x, z) {   // bilinear height from the finest loaded level, null if nothing loaded
  const lon = lonOfX(x), lat = latOfZ(z);
  for (let L = 0; L < NL; L++) {
    const s = 2 ** L;
    const cf = ((lon - meta.west) * PPD + 0.5) / s - 0.5, rf = ((meta.north - lat) * PPD + 0.5) / s - 0.5;
    if (cf < 0 || rf < 0) return 0;
    const col = Math.floor(cf / C), row = Math.floor(rf / C);
    const ch = chunks.get(key(L, row, col));
    if (!ch || ch.state !== 'ready') continue;
    const i = cf - col * C, j = rf - row * C;
    const i0 = Math.min(Math.floor(i), C - 1), j0 = Math.min(Math.floor(j), C - 1), fx = i - i0, fy = j - j0;
    const H = ch.heights, v = j0 * NS + i0;
    return (H[v] * (1 - fx) + H[v + 1] * fx) * (1 - fy) + (H[v + NS] * (1 - fx) + H[v + NS + 1] * fx) * fy;
  }
  return null;
}

// ---------------------------------------------------------------- sun overlay (data/sun/<run>/viewer, see serve.py)
let sunMeta = null, sunGlobal = null;
const sunTiles = new Map();
let sunInflight = 0;
async function loadRGBA(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(res.status);
  const bmp = await createImageBitmap(await res.blob(), { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
  const cv = new OffscreenCanvas(bmp.width, bmp.height), cx = cv.getContext('2d', { willReadFrequently: true });
  cx.drawImage(bmp, 0, 0);
  const data = cx.getImageData(0, 0, bmp.width, bmp.height).data;
  const tex = new THREE.DataTexture(new Uint8Array(data.buffer), bmp.width, bmp.height, THREE.RGBAFormat);
  tex.minFilter = THREE.LinearFilter; tex.magFilter = THREE.LinearFilter; tex.needsUpdate = true;
  const out = { tex, data, w: bmp.width, h: bmp.height };
  bmp.close();
  return out;
}
(async () => {
  try {
    sunMeta = await (await fetch('sun/meta.json')).json();
    sunGlobal = await loadRGBA('sun/overview.png');
  } catch (e) { sunMeta = null; }
})();
function sunTile(r, c, frame) {
  const k = `${r}_${c}`;
  let t = sunTiles.get(k);
  if (!t) {
    t = { state: 'new', last: frame }; sunTiles.set(k, t);
  }
  t.last = frame;
  if (t.state === 'new' && sunInflight < 2) {
    t.state = 'loading'; sunInflight++;
    loadRGBA(`sun/${k}.png`).then(d => Object.assign(t, d, { state: 'ready' }))
      .catch(() => { t.state = 'missing'; }).finally(() => { sunInflight--; });
  }
  if (sunTiles.size > 30) {   // LRU
    const old = [...sunTiles.entries()].filter(([, v]) => v.state === 'ready' && v.last < frame - 30)
      .sort((a, b) => a[1].last - b[1].last);
    for (const [kk, v] of old.slice(0, sunTiles.size - 24)) { v.tex.dispose(); sunTiles.delete(kk); }
  }
  return t.state === 'ready' ? t : null;
}
// texture transform for a terrain chunk: uv = xf.xy + grid * xf.zw. A raster texel k (grouped by `group`
// cells for the overview) sits at L0 sample off + stride * (k * group + (group - 1) / 2).
function sunXform(ch, x0, y0, group, W, H, v4) {
  const f = 2 ** ch.L, st = sunMeta.stride, off = sunMeta.offset;
  const ax = ch.c * C * f + (f - 1) / 2 - x0, ay = ch.r * C * f + (f - 1) / 2 - y0;
  v4.set((((ax - off) / st - (group - 1) / 2) / group + 0.5) / W,
         (((ay - off) / st - (group - 1) / 2) / group + 0.5) / H,
         f / st / group / W, f / st / group / H);
}
function updateSun(frame) {
  if (!sunMeta || !sunGlobal || shared.sunMode.value === 0) return;
  for (const ch of drawn) {
    const u = ch.mat.uniforms;
    const t = ch.L <= 4 ? sunTile(ch.r >> (4 - ch.L), ch.c >> (4 - ch.L), frame) : null;
    if (t) {
      u.sunTex.value = t.tex;
      sunXform(ch, (ch.c >> (4 - ch.L)) * PPD, (ch.r >> (4 - ch.L)) * PPD, 1, t.w, t.h, u.sunXf.value);
    } else {
      u.sunTex.value = sunGlobal.tex;
      sunXform(ch, 0, 0, sunMeta.overview_factor, sunGlobal.w, sunGlobal.h, u.sunXf.value);
    }
  }
}
function sunAt(x, z) {   // [hours, sunless days] at a world position, or null
  if (!sunMeta) return null;
  const X = (lonOfX(x) - meta.west) * PPD, Y = (meta.north - latOfZ(z)) * PPD;
  const r = Math.floor(Y / PPD), c = Math.floor(X / PPD);
  const t = sunTiles.get(`${r}_${c}`);
  if (!t || t.state !== 'ready') return null;
  const i = Math.round((X - c * PPD - sunMeta.offset) / sunMeta.stride), j = Math.round((Y - r * PPD - sunMeta.offset) / sunMeta.stride);
  if (i < 0 || j < 0 || i >= t.w || j >= t.h) return null;
  const p = 4 * (j * t.w + i);
  if (t.data[p + 2] < 128) return 'sea';
  return [t.data[p] * sunMeta.hours_scale, t.data[p + 1]];
}

// ---------------------------------------------------------------- controls
let yaw = 0, pitch = 0, flySpeed = 2500, walkSpeed = 6, mode = 'fly', vy = 0, locked = false;
const keys = new Set();
const $ = id => document.getElementById(id);

function teleport(lat, lon, elev, bearing = 180) {
  const tx = xOfLon(lon), tz = zOfLat(lat), b = bearing * Math.PI / 180, D = 6000;
  const cx = tx + D * Math.sin(b), cz = tz - D * Math.cos(b), cy = elev + (elev > 2500 ? 700 : 1500);
  camera.position.set(cx, cy, cz);
  const dx = tx - cx, dz = tz - cz;
  yaw = Math.atan2(-dx, -dz); pitch = Math.atan2(elev - cy, Math.hypot(dx, dz));
  if (mode === 'walk') setMode('fly');
}
function setMode(m) { mode = m; vy = 0; $('h-mode').textContent = m; }
function setSunMode(m) {
  if (m > 0 && !sunMeta) { $('h-sun').textContent = 'no sun data (run scripts/sun_alps.py)'; m = 0; }
  shared.sunMode.value = m;
  $('legend').classList.toggle('hidden', m === 0);
  if (m === 1) {
    $('lg-title').textContent = `direct sun per year ${sunMeta.year} (${sunMeta.res_m} m)`;
    $('lg-bar').style.background = 'linear-gradient(90deg,#08051a,#521270,#b83578,#fa785c,#fcf2b8)';
    $('lg-lo').textContent = '0 h'; $('lg-hi').textContent = '4500 h';
  } else if (m === 2) {
    $('lg-title').textContent = `days without direct sun ${sunMeta.year}`;
    $('lg-bar').style.background = 'linear-gradient(90deg,#d9d9d9,#9ec7eb,#082666)';
    $('lg-lo').textContent = '0'; $('lg-hi').textContent = '150+ days';
  }
}

const sel = $('places');
PLACES.forEach((p, i) => { const o = document.createElement('option'); o.value = i; o.textContent = p[0]; sel.appendChild(o); });
sel.addEventListener('change', () => {
  if (sel.value !== '') { const p = PLACES[+sel.value]; teleport(p[1], p[2], p[3], p[4]); }
  sel.value = ''; sel.blur();
});

canvas.addEventListener('click', () => canvas.requestPointerLock());
document.addEventListener('pointerlockchange', () => {
  locked = document.pointerLockElement === canvas;
  $('start').classList.toggle('hidden', locked);
  $('cross').style.display = locked ? 'block' : 'none';
  if (!locked) keys.clear();
});
document.addEventListener('mousemove', e => {
  if (!locked) return;
  yaw -= e.movementX * 0.0022;
  pitch = Math.max(-1.55, Math.min(1.55, pitch - e.movementY * 0.0022));
});
addEventListener('wheel', e => {
  const f = e.deltaY < 0 ? 1.25 : 0.8;
  if (mode === 'fly') flySpeed = Math.max(5, Math.min(100000, flySpeed * f));
  else walkSpeed = Math.max(1, Math.min(100, walkSpeed * f));
}, { passive: true });
addEventListener('keydown', e => {
  if (document.activeElement === sel) return;
  keys.add(e.code);
  if (e.code === 'Space' || e.code.startsWith('Arrow')) e.preventDefault();
  if (e.repeat) return;
  switch (e.code) {
    case 'KeyG': setMode(mode === 'fly' ? 'walk' : 'fly'); break;
    case 'KeyM': $('mini').classList.toggle('hidden'); break;
    case 'KeyH': ['hud', 'help', 'mini'].forEach(id => $(id).classList.toggle('hidden')); break;
    case 'KeyF': hazeIdx = (hazeIdx + 1) % HAZE.length; shared.hazeDensity.value = HAZE[hazeIdx]; break;
    case 'KeyL': splitIdx = (splitIdx + 1) % SPLIT.length; break;
    case 'KeyO': setSunMode((shared.sunMode.value + 1) % 3); break;
    case 'BracketLeft': sunAz -= 15; setSun(); break;
    case 'BracketRight': sunAz += 15; setSun(); break;
    case 'Comma': sunEl = Math.max(2, sunEl - 5); setSun(); break;
    case 'Period': sunEl = Math.min(85, sunEl + 5); setSun(); break;
  }
});
addEventListener('keyup', e => keys.delete(e.code));
addEventListener('blur', () => keys.clear());

// ---------------------------------------------------------------- minimap
const mini = $('minimap'), mctx = mini.getContext('2d');
const MW = 320, MH = Math.round(MW * ((meta.north - meta.south) * DEG) / ((meta.east - meta.west) * KX));
mini.width = MW * devicePixelRatio; mini.height = MH * devicePixelRatio; mini.style.height = MH + 'px';
const ovImg = new Image(); ovImg.src = 'tiles/overview.png';
function drawMini() {
  const s = devicePixelRatio;
  mctx.setTransform(s, 0, 0, s, 0, 0);
  if (ovImg.complete) mctx.drawImage(ovImg, 0, 0, MW, MH); else { mctx.fillStyle = '#234'; mctx.fillRect(0, 0, MW, MH); }
  const lon = lonOfX(camera.position.x), lat = latOfZ(camera.position.z);
  const px = (lon - meta.west) / (meta.east - meta.west) * MW, py = (meta.north - lat) / (meta.north - meta.south) * MH;
  mctx.save(); mctx.translate(px, py); mctx.rotate(-yaw);
  mctx.fillStyle = '#ff3b30'; mctx.strokeStyle = '#fff'; mctx.lineWidth = 1.2;
  mctx.beginPath(); mctx.moveTo(0, -9); mctx.lineTo(5, 6); mctx.lineTo(0, 3); mctx.lineTo(-5, 6); mctx.closePath();
  mctx.fill(); mctx.stroke(); mctx.restore();
}
mini.addEventListener('click', e => {
  const r = mini.getBoundingClientRect();
  const lon = meta.west + (e.clientX - r.left) / r.width * (meta.east - meta.west);
  const lat = meta.north - (e.clientY - r.top) / r.height * (meta.north - meta.south);
  camera.position.set(xOfLon(lon), Math.max(camera.position.y, 3500), zOfLat(lat));
  if (mode === 'walk') setMode('fly');
});

// ---------------------------------------------------------------- start position (URL hash or Matterhorn)
(function () {
  const h = location.hash.slice(1).split(',').map(Number);
  if (h.length === 5 && h.every(Number.isFinite)) {
    camera.position.set(xOfLon(h[1]), h[2], zOfLat(h[0]));
    yaw = -h[3] * Math.PI / 180; pitch = h[4] * Math.PI / 180;
  } else teleport(...PLACES[0].slice(1));
})();

// ---------------------------------------------------------------- loop
const clock = new THREE.Clock();
const fwd = new THREE.Vector3(), right = new THREE.Vector3(), mv = new THREE.Vector3(), prev = new THREE.Vector3();
let frame = 0, fpsT = 0, fpsN = 0, fps = 0, hudT = 0, hashT = 0, measSpeed = 0;
function tick() {
  requestAnimationFrame(tick);
  const dt = Math.min(clock.getDelta(), 0.1);
  frame++;
  prev.copy(camera.position);
  const k = c => (keys.has(c) ? 1 : 0);
  const boost = keys.has('ShiftLeft') || keys.has('ShiftRight') ? 50 : 1;
  // keyboard look (works when the mouse is blocked, e.g. touchpad disabled while typing)
  yaw += (k('ArrowLeft') - k('ArrowRight')) * 1.6 * dt;
  pitch = Math.max(-1.55, Math.min(1.55, pitch + (k('ArrowUp') - k('ArrowDown')) * 1.0 * dt));
  right.set(Math.cos(yaw), 0, -Math.sin(yaw));
  if (mode === 'fly') {
    fwd.set(-Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), -Math.cos(yaw) * Math.cos(pitch));
    mv.set(0, 0, 0).addScaledVector(fwd, k('KeyW') - k('KeyS')).addScaledVector(right, k('KeyD') - k('KeyA'));
    mv.y += k('Space') + k('KeyE') - k('KeyQ');
    if (mv.lengthSq() > 0) camera.position.addScaledVector(mv.normalize(), flySpeed * boost * dt);
    const g = groundAt(camera.position.x, camera.position.z);
    if (g !== null && camera.position.y < g + 3) camera.position.y = g + 3;
  } else {
    fwd.set(-Math.sin(yaw), 0, -Math.cos(yaw));
    mv.set(0, 0, 0).addScaledVector(fwd, k('KeyW') - k('KeyS')).addScaledVector(right, k('KeyD') - k('KeyA'));
    if (mv.lengthSq() > 0) camera.position.addScaledVector(mv.normalize(), walkSpeed * (boost > 1 ? 5 : 1) * dt);
    const g = groundAt(camera.position.x, camera.position.z);
    if (g !== null) {
      vy -= 9.81 * dt; camera.position.y += vy * dt;
      if (camera.position.y <= g + 1.8) { camera.position.y = g + 1.8; vy = keys.has('Space') ? 5 : 0; }
    }
  }
  camera.rotation.set(pitch, yaw, 0);
  camera.updateMatrixWorld();
  sky.position.copy(camera.position);
  updateLOD(frame);
  updateSun(frame);
  renderer.render(scene, camera);

  measSpeed = 0.9 * measSpeed + 0.1 * prev.distanceTo(camera.position) / Math.max(dt, 1e-3);
  fpsN++; fpsT += dt; hudT += dt; hashT += dt;
  if (fpsT > 0.5) { fps = fpsN / fpsT; fpsN = 0; fpsT = 0; }
  if (hudT > 0.15) {
    hudT = 0;
    const P = camera.position, lat = latOfZ(P.z), lon = lonOfX(P.x), g = groundAt(P.x, P.z);
    const hdg = ((-yaw * 180 / Math.PI) % 360 + 360) % 360;
    $('h-pos').textContent = `${lat.toFixed(4)}° N  ${lon.toFixed(4)}° E`;
    $('h-alt').textContent = `${P.y.toFixed(0)} m`;
    $('h-agl').textContent = g === null ? '–' : `${(P.y - g).toFixed(0)} m`;
    $('h-gnd').textContent = g === null ? '–' : `${g.toFixed(0)} m`;
    $('h-hdg').textContent = `${hdg.toFixed(0)}°  pitch ${(pitch * 180 / Math.PI).toFixed(0)}°`;
    $('h-spd').textContent = `${(measSpeed * 3.6).toFixed(0)} km/h  (base ${mode === 'fly' ? flySpeed.toFixed(0) : walkSpeed.toFixed(1)} m/s)`;
    $('h-lod').textContent = `${drawn.length} / ${chunks.size}  (+${inflight})`;
    $('h-fps').textContent = fps.toFixed(0);
    if (shared.sunMode.value > 0) {
      const sv = sunAt(P.x, P.z);
      $('h-sun').textContent = sv === null ? '…' : sv === 'sea' ? 'sea' : `${Math.round(sv[0] / 10) * 10} h/yr, ${sv[1]} sunless d`;
    }
    drawMini();
  }
  if (hashT > 1.0) {
    hashT = 0;
    const P = camera.position, hdg = ((-yaw * 180 / Math.PI) % 360 + 360) % 360;
    history.replaceState(null, '', `#${latOfZ(P.z).toFixed(5)},${lonOfX(P.x).toFixed(5)},${P.y.toFixed(0)},${hdg.toFixed(1)},${(pitch * 180 / Math.PI).toFixed(1)}`);
  }
}
tick();
