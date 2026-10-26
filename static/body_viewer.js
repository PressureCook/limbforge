// Interactive full-body inspector: skeleton + every evolved mechanism, with
// clickable joints that open a stats/spec panel. Reads body_scene.json only -
// no pose is recomputed here.
import * as THREE from "three";
import { OrbitControls } from "/static/three/OrbitControls.js";

const JCOLORS = { shoulder: 0xe05c2a, elbow: 0x1f9e77, hip: 0xcf7a2f,
                  knee: 0x2f7fcf, ankle: 0xb06fc4 };

export async function initBodyViewer(host, base) {
  const status = host.querySelector(".bv-status");
  const say = (m) => { if (status) status.textContent = m; };
  let doc;
  try { doc = await (await fetch(base + "body_scene.json")).json(); }
  catch (e) { say("body_scene.json missing"); return; }

  const canvas = host.querySelector("canvas");
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  const W = () => canvas.clientWidth, H = () => Math.max(canvas.clientHeight, 320);
  renderer.setSize(W(), H(), false);
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x10151c);
  const cam = new THREE.PerspectiveCamera(38, W() / H(), 0.01, 30);
  cam.position.set(0.55, -0.25, 2.1);
  const controls = new OrbitControls(cam, canvas);
  controls.target.set(0.1, -0.35, 0);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x334455, 1.1));
  const dl = new THREE.DirectionalLight(0xffffff, 1.3);
  dl.position.set(1, 2, 2); scene.add(dl);

  const capsule = (r, color) => new THREE.Mesh(
    new THREE.CylinderGeometry(r, r * 0.85, 1, 14),
    new THREE.MeshStandardMaterial({ color, roughness: 0.6 }));
  const stretch = (mesh, a, b) => {
    const A = new THREE.Vector3(a[0], a[1], 0), B = new THREE.Vector3(b[0], b[1], 0);
    const d = B.clone().sub(A), L = Math.max(d.length(), 1e-6);
    mesh.scale.set(1, L, 1);
    mesh.position.copy(A.clone().addScaledVector(d, 0.5));
    mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), d.normalize());
  };

  // torso + head anchored to the hip->shoulder offset used by the pipeline
  const torso = capsule(0.055, 0x39424e); scene.add(torso);
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.065, 16, 12),
    new THREE.MeshStandardMaterial({ color: 0x39424e, roughness: 0.6 }));
  scene.add(head);

  const segs = {}, acts = {}, hot = {}, hotList = [];
  for (const j of doc.joints) {
    segs[j] = capsule(0.028, JCOLORS[j] || 0x888888); scene.add(segs[j]);
    acts[j] = capsule(0.011, 0x3a6ea5); scene.add(acts[j]);
    const h = new THREE.Mesh(new THREE.SphereGeometry(0.035, 14, 10),
      new THREE.MeshStandardMaterial({ color: 0xdddddd, roughness: 0.35,
                                       emissive: 0x000000 }));
    h.userData.joint = j; scene.add(h); hot[j] = h; hotList.push(h);
  }

  // ---- inspector panel ----
  const panel = host.querySelector(".bv-panel");
  let selected = null;
  const showPanel = (j) => {
    selected = j;
    for (const k of doc.joints)
      hot[k].material.emissive.set(k === j ? 0x996600 : 0x000000);
    if (!j) { panel.innerHTML = "<em>click a joint sphere to inspect it</em>"; return; }
    const p = doc.panels[j], s = p.summary, hl = p.spec_highlights;
    const cpl = s.coupling || {};
    const carried = (cpl.carried_hardware || []).map(c =>
      `${c.joint} ${c.mass_g} g @ ${c.at_mm} mm`).join(", ") || "none (most distal)";
    const traj = cpl.trajectory
      ? `${cpl.trajectory.from} motion shifts COM ${cpl.trajectory.lever_shift_mm[0]}…${cpl.trajectory.lever_shift_mm[1]} mm`
      : "—";
    panel.innerHTML = `
      <h3 style="margin:0 0 6px">${doc.side} ${j}</h3>
      <div class="bv-grid">
        <span>ROM</span><b>${s.rom_deg[0]}–${s.rom_deg[1]}°</b>
        <span>tracking RMSE</span><b>${s.rmse_deg}°</b>
        <span>force / speed class</span><b>${s.mechanism.F_class_N} N · ${s.mechanism.v_class_mm_s} mm/s</b>
        <span>stroke · retracted</span><b>${s.mechanism.stroke_mm} · ${s.mechanism.L_ret_mm} mm</b>
        <span>drive</span><b>${hl.recommended} ${hl.screw.d_nom_mm ?? "?"}×${hl.screw.lead_mm ?? "?"} mm, ${hl.gear_ratio}:1</b>
        <span>motor</span><b>${hl.motor}${hl.custom_drive ? " ⚙ custom" : ""}</b>
        <span>electrical</span><b>${hl.bus_V} V · ~${hl.mass_g} g · ${hl.battery_h} h</b>
        <span>packaging</span><b>${hl.fits ? "fits" : "DOES NOT FIT"}</b>
        <span>carries (coupled)</span><b>${carried}</b>
        <span>distal-motion load</span><b>${traj}</b>
      </div>
      <div style="margin-top:8px">
        <a href="${base}${p.files.spec}" target="_blank">full spec</a> ·
        <a href="${base}${p.files.tracking}" target="_blank">tracking plot</a> ·
        <a href="${base}${p.files.drawing}" target="_blank">drawing</a>
      </div>
      <img src="${base}${p.files.drawing}" alt="${j} actuator drawing"
           style="width:100%; margin-top:8px; border-radius:6px"
           onerror="this.style.display='none'">`;
  };
  showPanel(null);

  // raycast clicks
  const ray = new THREE.Raycaster(), mouse = new THREE.Vector2();
  canvas.addEventListener("pointerdown", (e) => {
    const r = canvas.getBoundingClientRect();
    mouse.x = ((e.clientX - r.left) / r.width) * 2 - 1;
    mouse.y = -((e.clientY - r.top) / r.height) * 2 + 1;
    ray.setFromCamera(mouse, cam);
    const hit = ray.intersectObjects(hotList)[0];
    if (hit) showPanel(hit.object.userData.joint);
  });
  // keyboard joint cycling for accessibility
  host.tabIndex = 0;
  host.addEventListener("keydown", (e) => {
    if (e.key === "Tab" && !e.shiftKey && doc.joints.length) {
      const i = selected ? doc.joints.indexOf(selected) : -1;
      showPanel(doc.joints[(i + 1) % doc.joints.length]);
      e.preventDefault();
    } else if (e.key === " ") { playing = !playing; e.preventDefault(); }
    else if (e.key === ",") { playing = false; setFrame(frame - 1); }
    else if (e.key === ".") { playing = false; setFrame(frame + 1); }
  });

  // ---- animation ----
  const frames = doc.frames;
  let frame = 0, playing = true;
  const setFrame = (f) => {
    frame = ((f % frames.length) + frames.length) % frames.length;
    const fr = frames[frame];
    for (const j of doc.joints) {
      const e = fr.joints[j];
      if (!e) continue;
      stretch(segs[j], e.pos, e.tip);
      hot[j].position.set(e.pos[0], e.pos[1], 0.02);
      if (e.act) { acts[j].visible = true; stretch(acts[j], e.act[0], e.act[1]); acts[j].position.z = 0.05; }
      else acts[j].visible = false;
      const u = Math.min(e.util || 0, 1);
      acts[j].material.color.setHSL(0.6 - 0.6 * u, 0.75, 0.45);
    }
    const hipE = fr.joints.hip, shE = fr.joints.shoulder;
    if (hipE && shE) {
      stretch(torso, hipE.pos, shE.pos);
      head.position.set(shE.pos[0] + 0.02, shE.pos[1] + 0.12, 0);
    }
    const sl = host.querySelector(".bv-scrub");
    if (sl && +sl.value !== frame) sl.value = frame;
    const hud = host.querySelector(".bv-hud");
    if (hud) hud.textContent = `t ${fr.t.toFixed(2)} s · frame ${frame + 1}/${frames.length}` +
      (selected && fr.joints[selected]
        ? ` · ${selected} θ ${fr.joints[selected].theta_deg}° · ${(100 * fr.joints[selected].util).toFixed(0)}% cap` : "");
  };
  const scrub = host.querySelector(".bv-scrub");
  if (scrub) { scrub.max = frames.length - 1;
    scrub.addEventListener("input", () => { playing = false; setFrame(+scrub.value); }); }
  const playBtn = host.querySelector(".bv-play");
  if (playBtn) playBtn.addEventListener("click", () => {
    playing = !playing; playBtn.textContent = playing ? "pause" : "play"; });

  let last = 0;
  const tick = (t) => {
    requestAnimationFrame(tick);
    if (playing && t - last > 80) { setFrame(frame + 1); last = t; }
    if (canvas.width !== W() * renderer.getPixelRatio()) {
      renderer.setSize(W(), H(), false);
      cam.aspect = W() / H(); cam.updateProjectionMatrix();
    }
    controls.update();
    renderer.render(scene, cam);
  };
  setFrame(0);
  requestAnimationFrame(tick);
  say(`${doc.joints.length} coupled joints · ${frames.length} frames — click a sphere to inspect`);
}
