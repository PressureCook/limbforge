// 3D anatomical scene viewer (SPEC-2 §9 / V-5). Loads scene.glb + scene.json
// emitted by the pipeline and never recomputes a pose. Vendored three.js
// (V-6): no CDN fetch from a tool that handles webcam footage.
import * as THREE from "three";
import { OrbitControls } from "/static/three/OrbitControls.js";
import { GLTFLoader } from "/static/three/GLTFLoader.js";

const LAYERS = ["tissue", "bone", "mechanism", "force", "labels", "axis", "rom"];

export async function initViewer(host, base) {
  const status = host.querySelector(".v3d-status");
  const say = (m) => { if (status) status.textContent = m; };
  if (!window.WebGLRenderingContext) {
    host.querySelector(".v3d-fallback").style.display = "";
    say("WebGL is unavailable; showing the rendered still instead.");
    return;
  }
  let doc;
  try { doc = await (await fetch(base + "scene.json")).json(); }
  catch (e) { say("scene.json missing - this run predates the 3D viewer"); return; }

  const canvas = host.querySelector("canvas");
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  const W = () => canvas.clientWidth, H = () => Math.max(canvas.clientHeight, 300);
  renderer.setSize(W(), H(), false);
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  const sceneT = new THREE.Scene();
  sceneT.background = new THREE.Color(0x10151c);
  const span = doc.meta.Lp + doc.meta.Ld;
  const cam = new THREE.PerspectiveCamera(40, W() / H(), 0.01, 20);
  cam.position.set(span * 0.45, -span * 0.9, span * 0.8);
  cam.up.set(0, 1, 0);
  const controls = new OrbitControls(cam, canvas);
  controls.target.set(span * 0.15, -span * 0.15, 0);
  controls.listenToKeyEvents(host);
  sceneT.add(new THREE.HemisphereLight(0xffffff, 0x334455, 1.1));
  const dl = new THREE.DirectionalLight(0xffffff, 1.4);
  dl.position.set(1, 2, 2); sceneT.add(dl);

  // groups
  const gBone = new THREE.Group(), gTissue = new THREE.Group(),
        gMech = new THREE.Group(), gLabels = new THREE.Group(),
        gAxis = new THREE.Group(), gRom = new THREE.Group();
  const gDistal = new THREE.Group();          // rotates with theta
  sceneT.add(gBone, gTissue, gMech, gLabels, gAxis, gRom, gDistal);

  const units = {};
  await new Promise((res, rej) => new GLTFLoader().load(base + "scene.glb", (g) => {
    g.scene.traverse((o) => {
      if (!o.isMesh) return;
      const n = o.name || (o.parent && o.parent.name) || "";
      if (o.material) { o.material.transparent = o.material.opacity < 1 ||
        (o.material.name || "").includes("tissue"); }
      if (n.startsWith("unit_")) { units[n] = o; o.visible = false; sceneT.add(o.clone()); }
      else if (n === "bone_distal" || n === "tissue_distal") {
        (n === "bone_distal" ? gDistal : gDistal).add(o.clone());
        if (n === "tissue_distal") o.userData.tissue = true;
      } else if (n === "bone_proximal") gBone.add(o.clone());
      else if (n === "tissue_proximal") gTissue.add(o.clone());
    });
    // distal tissue lives in the rotating group but toggles with tissue layer
    res();
  }, undefined, rej));
  gDistal.children.forEach(c => { if ((c.name||"").includes("tissue")) c.userData.tissue = true; });

  // per-frame stretched primitives
  const mkCyl = (r, color) => new THREE.Mesh(
    new THREE.CylinderGeometry(r, r, 1, 16),
    new THREE.MeshStandardMaterial({ color, roughness: 0.55 }));
  const nAct = doc.frames[0].actuators.length;
  const barrels = [], rods = [], links = [];
  for (let i = 0; i < nAct; i++) {
    const b = mkCyl(0.011, 0x3a6ea5), r = mkCyl(0.004, 0x9db6d0);
    gMech.add(b, r); barrels.push(b); rods.push(r);
  }
  const nLinks = (doc.frames[0].links || []).length;
  for (let i = 0; i < nLinks; i++) {
    const l = mkCyl(0.007, 0x3f7f3f); gMech.add(l); links.push(l);
  }
  const stretch = (mesh, a, b, fracLen = 1.0, fromA = true) => {
    const A = new THREE.Vector3(...a), B = new THREE.Vector3(...b);
    const d = B.clone().sub(A), L = Math.max(d.length(), 1e-6);
    const mid = fromA ? A.clone().addScaledVector(d, fracLen / 2)
                      : B.clone().addScaledVector(d, -fracLen / 2);
    mesh.scale.set(1, L * fracLen, 1);
    mesh.position.copy(mid);
    mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0),
                                       d.normalize());
  };

  // joint axis + ROM arc
  const axisLen = span * 0.35;
  const axis = new THREE.Mesh(new THREE.CylinderGeometry(0.004, 0.004, axisLen, 10),
    new THREE.MeshStandardMaterial({ color: 0xcccc44 }));
  axis.rotation.x = Math.PI / 2; gAxis.add(axis);
  const romMat = new THREE.LineBasicMaterial({ color: 0x888888 });
  const romPts = [];
  const [r0, r1] = doc.rom_deg;
  for (let a = r0; a <= r1; a += 3) {
    const p = Math.PI - a * Math.PI / 180;
    romPts.push(new THREE.Vector3(Math.cos(p) * doc.meta.Ld,
                                  Math.sin(p) * doc.meta.Ld, 0));
  }
  gRom.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(romPts), romMat));

  // HTML overlay labels (screen-projected, so they stay legible)
  const overlay = host.querySelector(".v3d-labels");
  const labelEls = doc.landmarks.map((l) => {
    const el = document.createElement("div");
    el.className = "v3d-label"; el.textContent = l.name;
    overlay.appendChild(el); return el;
  });
  const jointLabel = document.createElement("div");
  jointLabel.className = "v3d-label v3d-joint";
  jointLabel.textContent = `${doc.meta.side} ${doc.meta.joint} — ` +
    `${doc.meta.prox_bone} / ${doc.meta.dist_bone}`;
  overlay.appendChild(jointLabel);

  // ---- animation state ----
  const layerState = Object.fromEntries(LAYERS.map(l => [l, l !== "tissue"]));
  let frame = 0, playing = true, theta = 0;
  const frames = doc.frames;
  const setFrame = (f) => {
    frame = ((f % frames.length) + frames.length) % frames.length;
    const fr = frames[frame];
    theta = fr.theta_deg * Math.PI / 180;
    gDistal.rotation.z = Math.PI - theta;
    fr.actuators.forEach((a, i) => {
      stretch(barrels[i], a.a, a.b, 0.6, true);
      stretch(rods[i], a.a, a.b, 1.0, true);
      if (layerState.force) {
        const u = Math.min(a.util || 0, 1.2);
        barrels[i].material.color.setHSL(0.6 - 0.6 * Math.min(u, 1), 0.75, 0.45);
      } else barrels[i].material.color.set(0x3a6ea5);
    });
    (fr.links || []).forEach((s, i) => stretch(links[i], s[0], s[1]));
    const sl = host.querySelector(".v3d-scrub");
    if (sl && +sl.value !== frame) sl.value = frame;
    const hud = host.querySelector(".v3d-hud");
    if (hud) hud.textContent =
      `θ ${fr.theta_deg.toFixed(1)}°  ·  frame ${frame + 1}/${frames.length}` +
      (nAct ? `  ·  a0 ${(100 * (fr.actuators[0].util || 0)).toFixed(0)}% cap` : "");
  };

  const project = (v) => {
    const p = v.clone().project(cam);
    return [(p.x * 0.5 + 0.5) * W(), (-p.y * 0.5 + 0.5) * H(), p.z < 1];
  };
  const applyLayers = () => {
    gTissue.visible = layerState.tissue;
    gDistal.children.forEach(c => {
      if (c.userData.tissue) c.visible = layerState.tissue;
      else c.visible = layerState.bone;
    });
    gBone.visible = layerState.bone;
    gMech.visible = layerState.mechanism;
    gAxis.visible = layerState.axis;
    gRom.visible = layerState.rom;
    overlay.style.display = layerState.labels ? "" : "none";
    host.querySelectorAll(".v3d-layer").forEach(cb => {
      cb.checked = layerState[cb.dataset.layer];
    });
    say("layers: " + LAYERS.filter(l => layerState[l]).join(", "));
  };

  host.querySelectorAll(".v3d-layer").forEach(cb =>
    cb.addEventListener("change", () => {
      layerState[cb.dataset.layer] = cb.checked; applyLayers();
    }));
  const scrub = host.querySelector(".v3d-scrub");
  if (scrub) {
    scrub.max = frames.length - 1;
    scrub.addEventListener("input", () => { playing = false; setFrame(+scrub.value); });
  }
  const playBtn = host.querySelector(".v3d-play");
  if (playBtn) playBtn.addEventListener("click", () => {
    playing = !playing; playBtn.textContent = playing ? "pause" : "play";
  });
  host.tabIndex = 0;
  host.addEventListener("keydown", (e) => {
    if (e.key === " ") { playing = !playing; e.preventDefault(); }
    else if (e.key === ",") { playing = false; setFrame(frame - 1); }
    else if (e.key === ".") { playing = false; setFrame(frame + 1); }
    else if (/^[1-7]$/.test(e.key)) {
      const l = LAYERS[+e.key - 1]; layerState[l] = !layerState[l]; applyLayers();
    }
  });

  let last = 0;
  const tick = (t) => {
    requestAnimationFrame(tick);
    if (playing && t - last > 70) { setFrame(frame + 1); last = t; }
    if (canvas.width !== W() * renderer.getPixelRatio()) {
      renderer.setSize(W(), H(), false);
      cam.aspect = W() / H(); cam.updateProjectionMatrix();
    }
    controls.update();
    // project labels
    doc.landmarks.forEach((l, i) => {
      let v = new THREE.Vector3(...l.pos);
      if (l.bone === "D") v.applyAxisAngle(new THREE.Vector3(0, 0, 1),
                                           Math.PI - theta);
      const [x, y, ok] = project(v);
      const el = labelEls[i];
      el.style.transform = `translate(${x}px, ${y}px)`;
      el.style.display = ok ? "" : "none";
    });
    const [jx, jy] = project(new THREE.Vector3(0, 0, 0.02));
    jointLabel.style.transform = `translate(${jx}px, ${jy - 26}px)`;
    renderer.render(sceneT, cam);
  };
  applyLayers();
  setFrame(0);
  requestAnimationFrame(tick);
  say(`loaded ${frames.length} frames · ROM ${doc.rom_deg[0]}–${doc.rom_deg[1]}° · RMSE ${doc.rmse_deg}°`);
}
