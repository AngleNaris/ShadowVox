/* ShadowVox 前端：界面状态在这里，真实处理与音频经 QWebChannel 交给 Python。
   交互逻辑沿用 ShadowBuster：记住参数、预设槽、旁路整块变淡、缺项红框、按钮即进度、成功静默。 */
'use strict';
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const switchOn = el => el.getAttribute('aria-checked') === 'true';
const RM = matchMedia('(prefers-reduced-motion: reduce)');
const EASE = 'cubic-bezier(.25,.6,.35,1)';
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const store = {
  get(k, d) { try { const v = localStorage.getItem('sv_' + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem('sv_' + k, JSON.stringify(v)); } catch (e) { /* 存储不可用时只影响记忆 */ } },
};
let bridge = null;
const call = (name, ...args) => new Promise(res => (bridge ? bridge[name](...args, res) : res(undefined)));
const theme = window.ShadowTheme;

/* 读屏播报：先清空，下一帧再写入，保证相同文字也会重读 */
function announce(text) {
  const live = $('#live'); live.textContent = '';
  requestAnimationFrame(() => { live.textContent = text; });
}
/* 缺项：控件内部红框闪一次并播报；不常驻警示 */
function markNeed(el) {
  announce('缺少输入：' + el.dataset.need);
  el.classList.remove('need'); void el.offsetWidth; el.classList.add('need');
  clearTimeout(el._need); el._need = setTimeout(() => el.classList.remove('need'), 1400);
}
const clock = (s, ms = false) => {
  s = Math.max(0, s || 0);
  const m = Math.floor(s / 60), sec = s - m * 60;
  return ms ? `${String(m).padStart(2, '0')}:${sec.toFixed(3).padStart(6, '0')}` : `${String(m).padStart(2, '0')}:${String(Math.floor(sec)).padStart(2, '0')}`;
};

/* ═══ 参数表：唯一来源；默认值与已确认的处理链一致 ═══ */
const pct = v => `${Math.round(v)}%`;
const PARAMS = {
  correction: { name: '修正', min: 0, max: 100, step: 1, def: 50, fmt: pct },
  beauty: { name: '美化', min: 0, max: 100, step: 1, def: 50, fmt: pct, post: true },
  fusion: { name: '融合', min: 0, max: 100, step: 1, def: 50, fmt: pct, post: true },
  ratio: { name: '比例', min: -100, max: 100, step: 1, def: 0,
    fmt: v => v ? `${v > 0 ? '靠前' : '靠后'} ${Math.abs(v)}` : '平衡' },
  mixdown: { name: '混音', min: 0, max: 100, step: 1, def: 100, fmt: pct, post: true },
};
const LEGACY_KEYS = {
  correction: ['strength', 'rhythm_strength', 'denoise_strength', 'rhythm', 'denoise', 'declip'],
  beauty: ['mix_lowcut_hz', 'mix_eq', 'mix_compression', 'mix_deess'],
  fusion: ['mix_duck', 'mix_reverb', 'mix_delay'],
  ratio: ['vocal_gain_db'],
  mixdown: [],
};
const LEGACY_SWITCHES = ['rhythm', 'denoise', 'declip'];
const legacyKeys = Object.values(LEGACY_KEYS).flat();
function savedLegacy() {
  const saved = store.get('legacy_params', null);
  if (saved !== null) return saved;
  const pairs = legacyKeys.map(k => [k, store.get((LEGACY_SWITCHES.includes(k) ? 'switch_' : 'param_') + k, null)]);
  return Object.fromEntries(pairs.filter(([, v]) => v !== null));
}
const HELP = [
  ['修正', '联动干声降噪、削波修复、节奏对齐和音高校正。启用后，按原唱的演唱区间静音干声空档。'],
  ['美化', '调节人声本身的音色和动态，联动低切、均衡、双级压缩、去齿音和增厚。'],
  ['融合', '联动混响、延时和伴奏避让。演唱时给人声腾出空间，高档更容易听到句尾混响。'],
  ['比例', '调节人声相对伴奏的前后位置，范围为 −6 到 +6 dB。中点为平衡，靠前 35 对应 +2.1 dB。'],
  ['混音', '调节人声和伴奏合成后的多段压缩密度。整曲导出固定以 −8 LUFS-I 为目标，真峰值上限为 −0.5 dBTP。持续限幅过重时会保留更多动态，实际响度可能较低。关闭后期总开关时也会执行母带。'],
  ['旧设置', '读数显示「旧设置」时使用原来保存的参数。调整某一项后，该项改用联动参数。'],
];
const AUDITION = '拖动方形调节点可同时调整两轴，也可按 Tab 选中一项，再用方向键微调。开启「实时预览」后，调参会在后台更新播放位置附近的草稿。要保存新设置，请再次生成成品。';

const P = {}, SW = { mix_effects: store.get('switch_mix_effects', true) !== false };
let legacy = savedLegacy();
const hasLegacy = key => LEGACY_KEYS[key].some(k => Object.hasOwn(legacy, k));
for (const [k, d] of Object.entries(PARAMS)) P[k] = snapTo(store.get('param_' + k, d.def), d);
function snapTo(v, d) {
  v = Number(v); if (!Number.isFinite(v)) v = d.def;
  return +clamp(Math.round((v - d.min) / d.step) * d.step + d.min, d.min, d.max).toFixed(4);
}
const params = () => ({ ...P, ...SW, ...legacy, dry_has_backing: switchOn($('#dry-has-backing')) });

/* ═══ 监听音量：上下拖动、滚轮、方向键、双击复位 ═══ */
function makeKnob(el, d, value, onChange) {
  el.classList.add('knob-wrap'); el.tabIndex = 0; el.setAttribute('role', 'slider');
  el.innerHTML = '<span class="knob-ring"></span><span class="knob-cap"></span><span class="knob-tick"></span>';
  el.setAttribute('aria-valuemin', d.min); el.setAttribute('aria-valuemax', d.max);
  const render = () => {
    el.style.setProperty('--v', (value - d.min) / (d.max - d.min));
    el.setAttribute('aria-valuenow', value); el.setAttribute('aria-valuetext', d.fmt(value));
  };
  const set = v => {
    v = snapTo(v, d);
    if (v === value) return;
    value = v; render(); onChange(v);
  };
  let drag = null;
  el.addEventListener('pointerdown', e => {
    if (e.button !== 0) return;
    el.setPointerCapture(e.pointerId); el.focus({ preventScroll: true });
    drag = { y: e.clientY, v: value }; el.classList.add('dragging'); e.preventDefault();
  });
  el.addEventListener('pointermove', e => {
    if (!drag) return;
    set(drag.v + (drag.y - e.clientY) * (d.max - d.min) / (e.shiftKey ? 800 : 200));
  });
  const end = () => { drag = null; el.classList.remove('dragging'); };
  el.addEventListener('pointerup', end); el.addEventListener('pointercancel', end);
  el.addEventListener('wheel', e => { e.preventDefault(); set(value + (e.deltaY < 0 ? d.step : -d.step)); }, { passive: false });
  el.addEventListener('dblclick', () => set(d.def));
  el.addEventListener('keydown', e => {
    const steps = { ArrowUp: 1, ArrowRight: 1, ArrowDown: -1, ArrowLeft: -1, PageUp: 5, PageDown: -5 }[e.key];
    if (steps) set(value + steps * d.step);
    else if (e.key === 'Home') set(d.min);
    else if (e.key === 'End') set(d.max);
    else return;
    e.preventDefault();
  });
  render();
}

/* ═══ 声音处理 ═══ */
const soundInput = key => $(`#sound-${key}`);
const soundValue = key => hasLegacy(key) ? '旧设置' : PARAMS[key].fmt(P[key]);
const soundFraction = key => (P[key] - PARAMS[key].min) / (PARAMS[key].max - PARAMS[key].min);

function setSound(values, reset = false) {
  let edited = false, legacyEdited = false;
  for (const [key, input] of Object.entries(values)) {
    if (soundInput(key).disabled) continue;
    const value = snapTo(input, PARAMS[key]), old = hasLegacy(key);
    if (value === P[key] && !(reset && old)) continue;
    P[key] = value;
    store.set('param_' + key, value);
    if (old) {
      for (const k of LEGACY_KEYS[key]) delete legacy[k];
      legacyEdited = true;
    }
    edited = true;
  }
  if (!edited) return;
  if (legacyEdited) store.set('legacy_params', legacy);
  paint(); changed();
}

function rangeMarkup(key, className, vertical = false) {
  const { min, max, step } = PARAMS[key];
  return `<input id="sound-${key}" data-sound="${key}" class="${className}" type="range"
    min="${min}" max="${max}" step="${step}"${vertical ? ' aria-orientation="vertical"' : ''}>`;
}

function scaleMarks(mark) {
  return Array.from({ length: 21 }, (_, i) => mark(i * 5, i % 5 === 0)).join('');
}

function xyMarkup(x, y) {
  const bipolar = PARAMS[x].min < 0;
  const xLabel = `<label class="axis-label axis-x" for="sound-${x}">${PARAMS[x].name}</label>`;
  const yLabel = `<label class="axis-label axis-y" for="sound-${y}">${PARAMS[y].name}</label>`;
  const xRange = rangeMarkup(x, 'sr-only axis-input'), yRange = rangeMarkup(y, 'sr-only axis-input', true);
  const marks = scaleMarks((p, major) => {
    const length = major ? 2.2 : 1.05;
    return `M${p} 0v${length}M${p} 100v-${length}M0 ${p}h${length}M100 ${p}h-${length}`;
  });
  return `<div class="xy-control" role="group" aria-label="${bipolar ? PARAMS[y].name + '与' + PARAMS[x].name : PARAMS[x].name + '与' + PARAMS[y].name}">
    ${bipolar ? yLabel + xLabel : xLabel + yLabel}
    <div class="xy-field" data-x="${x}" data-y="${y}"><div class="xy-scale">
      <svg class="xy-grid" viewBox="0 0 100 100" aria-hidden="true">
        <defs><pattern id="xy-dots-${x}" width="5" height="5" x="-2.5" y="-2.5" patternUnits="userSpaceOnUse">
          <circle cx="2.5" cy="2.5" r=".24"/>
        </pattern></defs>
        <rect width="100" height="100" fill="url(#xy-dots-${x})"/>
        <path class="scale-marks" d="${marks}"/>
      </svg>
      <span class="xy-guides" aria-hidden="true"></span>
      <div class="xy-point" aria-hidden="true"><i></i></div>
      ${bipolar ? yRange + xRange : xRange + yRange}
      <output class="sound-tip xy-tip" aria-hidden="true"></output>
    </div></div>
  </div>`;
}

function bindXY(field) {
  const { x, y } = field.dataset, scale = $('.xy-scale', field), point = $('.xy-point', field);
  let drag = null;
  const at = (key, fraction) => PARAMS[key].min + clamp(fraction, 0, 1) * (PARAMS[key].max - PARAMS[key].min);
  const move = e => {
    if (drag?.id !== e.pointerId) return;
    const r = scale.getBoundingClientRect();
    setSound({ [x]: at(x, (e.clientX - drag.dx - r.left) / r.width),
      [y]: at(y, (r.bottom - e.clientY + drag.dy) / r.height) });
  };
  field.addEventListener('pointerdown', e => {
    if (e.button !== 0 || drag) return;
    const hit = point.contains(e.target), r = point.getBoundingClientRect();
    drag = { id: e.pointerId, dx: hit ? e.clientX - r.left - r.width / 2 : 0,
      dy: hit ? e.clientY - r.top - r.height / 2 : 0 };
    field.setPointerCapture(e.pointerId);
    field.classList.add('is-dragging');
    soundInput(x).focus({ preventScroll: true });
    field.classList.add('is-pointer-focused');
    if (!hit) move(e);
    e.preventDefault();
  });
  field.addEventListener('pointermove', move);
  field.addEventListener('lostpointercapture', () => { drag = null; field.classList.remove('is-dragging'); });
  const endPointerFocus = () => field.classList.remove('is-pointer-focused');
  field.addEventListener('keydown', endPointerFocus);
  field.addEventListener('focusout', endPointerFocus);
  field.addEventListener('dblclick', () => setSound({ [x]: PARAMS[x].def, [y]: PARAMS[y].def }, true));
}

function buildRacks() {
  const root = $('#racks');
  root.innerHTML = `<section class="rack" data-rack="sound" aria-labelledby="sound-title">
    <div class="rack-head"><h2 class="rack-title" id="sound-title">声音处理</h2><span class="spacer"></span>
      <button class="key" type="button" data-help="sound" title="声音处理说明" aria-label="声音处理说明" aria-haspopup="dialog"><svg class="i"><use href="#i-help"/></svg></button>
      <button class="key key-labeled" type="button" role="switch" data-switch="mix_effects" title="开关美化、融合、混音" aria-controls="sound-beauty sound-fusion sound-mixdown"><svg class="i"><use href="#i-power"/></svg><span>后期总开关</span></button>
    </div><div class="rack-body">
      ${xyMarkup('correction', 'beauty')}${xyMarkup('ratio', 'fusion')}
      <div class="mix-row"><label class="axis-label" for="sound-mixdown">混音</label>
        <div class="mix-fader">
          ${rangeMarkup('mixdown', 'mix-range')}
          <svg class="mix-marks" viewBox="0 0 100 8" preserveAspectRatio="none" aria-hidden="true">
            <path class="scale-marks" d="${scaleMarks((p, major) => `M${p} 0v${major ? 7 : 4}`)}"/>
          </svg>
          <output class="sound-tip mix-tip" aria-hidden="true"></output>
        </div>
      </div>
    </div></section>`;
  $$('.xy-field', root).forEach(bindXY);
  $$('[data-sound]', root).forEach(input => input.addEventListener('input', () => {
    setSound({ [input.dataset.sound]: input.valueAsNumber });
  }));
  const mix = soundInput('mixdown'), fader = mix.parentElement;
  mix.addEventListener('pointermove', e => {
    const r = mix.getBoundingClientRect(), half = $('.mix-marks', fader).getBoundingClientRect().left - r.left;
    const center = r.left + half + (r.width - 2 * half) * soundFraction('mixdown');
    fader.classList.toggle('is-over-thumb', Math.abs(e.clientX - center) <= half);
  });
  mix.addEventListener('pointerleave', () => fader.classList.remove('is-over-thumb'));
  mix.addEventListener('dblclick', () => setSound({ mixdown: PARAMS.mixdown.def }, true));
  $('[data-switch=mix_effects]', root).addEventListener('click', () => {
    SW.mix_effects = !SW.mix_effects;
    store.set('switch_mix_effects', SW.mix_effects);
    paint(); changed();
  });
  $('[data-help]', root).addEventListener('click', e => toggleHelp(e.currentTarget));
}

function paint() {
  $('[data-switch=mix_effects]').setAttribute('aria-checked', String(SW.mix_effects));
  for (const [key, d] of Object.entries(PARAMS)) {
    const input = soundInput(key);
    input.value = P[key];
    input.disabled = !!d.post && !SW.mix_effects;
    input.setAttribute('aria-valuetext', soundValue(key));
    $(`label[for="${input.id}"]`).classList.toggle('is-disabled', input.disabled);
  }
  for (const field of $$('.xy-field')) {
    const { x, y } = field.dataset;
    field.style.setProperty('--x', `${100 * soundFraction(x)}%`);
    field.style.setProperty('--y', `${100 * (1 - soundFraction(y))}%`);
    field.classList.toggle('is-x-only', soundInput(y).disabled);
    field.classList.toggle('is-tip-below', soundFraction(y) > .82);
    $('.xy-tip', field).textContent = soundInput(y).disabled ? soundValue(x) : `↔ ${soundValue(x)}  ↕ ${soundValue(y)}`;
  }
  const fader = soundInput('mixdown').parentElement;
  fader.style.setProperty('--fill', `${100 * soundFraction('mixdown')}%`);
  $('.mix-tip', fader).textContent = soundValue('mixdown');
}

/* 参数变化：已有成品时交给后台更新草稿 */
function changed() {
  if (state.ready && !run.busy && switchOn($('#live-preview'))) bridge?.draft(params());
}

/* ═══ 预设：9 槽；左键空槽保存、左键已存槽应用；右键已存槽待确认（再右键删除、左键覆盖），4 秒自动取消；不存路径 ═══ */
const PRESET_ARM_MS = 4000;
let presets = store.get('presets', {}), armed = null, armTimer = 0;
function buildPresets() {
  const root = $('#presets');
  for (let n = 1; n <= 9; n++) {
    const pad = document.createElement('button'); pad.type = 'button'; pad.className = 'pad'; pad.dataset.slot = n;
    pad.addEventListener('click', () => presetClick(n));
    pad.addEventListener('contextmenu', e => { e.preventDefault(); presetArm(n); });
    root.appendChild(pad);
  }
  renderPresets();
}
function renderPresets() {
  $$('.pad').forEach(pad => {
    const n = +pad.dataset.slot, saved = !!presets[n], isArmed = armed === n;
    pad.dataset.saved = saved; pad.dataset.armed = isArmed;
    pad.innerHTML = isArmed ? '<svg class="i"><use href="#i-close"/></svg>' : String(n);
    pad.setAttribute('aria-label', isArmed ? `预设 ${n} 待确认：再右键删除，左键用当前参数覆盖`
      : `预设 ${n}：${saved ? '已保存，左键应用，右键删除或覆盖' : '空，左键保存当前参数'}`);
  });
}
function disarm() { clearTimeout(armTimer); armed = null; renderPresets(); }
function flashPad(n) { const pad = $(`.pad[data-slot="${n}"]`); pad.classList.remove('flash'); void pad.offsetWidth; pad.classList.add('flash'); }
function savePreset(n) {
  presets[n] = { savedAt: Date.now(), params: { ...P }, switches: { ...SW }, legacy: { ...legacy } };
  store.set('presets', presets); flashPad(n); announce(`已保存预设 ${n}`);
}
function presetClick(n) {
  if (armed === n) { disarm(); savePreset(n); renderPresets(); return; }
  disarm();
  if (!presets[n]) { savePreset(n); renderPresets(); return; }
  const p = presets[n];
  const old = { ...p.params, ...p.switches };
  legacy = { ...(p.legacy ?? Object.fromEntries(legacyKeys.filter(k => k in old).map(k => [k, old[k]]))) };
  store.set('legacy_params', legacy);
  for (const [k, d] of Object.entries(PARAMS)) {
    P[k] = snapTo(p.params[k] ?? d.def, d);
    store.set('param_' + k, P[k]);
  }
  SW.mix_effects = p.switches?.mix_effects !== false;
  store.set('switch_mix_effects', SW.mix_effects);
  paint(); changed(); flashPad(n); announce(`已应用预设 ${n}`);
}
function presetArm(n) {
  if (!presets[n]) return;
  if (armed === n) { delete presets[n]; store.set('presets', presets); disarm(); announce(`已删除预设 ${n}`); return; }
  disarm(); armed = n; renderPresets(); armTimer = setTimeout(disarm, PRESET_ARM_MS);
}

/* ═══ 输入文件与输出目录 ═══ */
const inputs = { song: '', dry: '' };
function setChip(kind, name) {
  const chip = $(`.chip[data-kind="${kind}"]`), label = kind === 'song' ? '原曲' : '演唱干音';
  chip.dataset.filled = !!name; $('.mq', chip).textContent = name || (kind === 'song' ? '原曲' : '干音');
  chip.setAttribute('aria-label', `${label}：${name || '未选择'}`); chip.title = name ? `${label}：${name}` : `${label}（可拖入）`;
}
async function chooseInput(kind, path) {
  if (run.busy) return;
  const res = path ? await call('setInput', kind, path) : await call('pickAudio', kind);
  if (!res) return;
  const { path: full, name } = JSON.parse(res);
  if (kind === 'dry') {
    const option = $('#dry-has-backing');
    if (inputs.dry !== full) option.setAttribute('aria-checked', 'false');
    option.disabled = false;
  }
  inputs[kind] = full; setChip(kind, name); resetResult();
}
function setOutput(path) {
  store.set('output', path);
  const card = $('#out-card'); card.dataset.filled = !!path;
  $('#out-path bdi').textContent = path || '选择输出目录';
  card.setAttribute('aria-label', `输出目录：${path || '未选择'}`);
}
/* 文件名超出时悬停往返滚动，两端各停一拍 */
function bindMarquee(chip) {
  let anim = null;
  chip.addEventListener('pointerenter', () => {
    const mq = $('.mq', chip), span = mq.scrollWidth - $('.chip-name', chip).clientWidth;
    if (span <= 2 || RM.matches) return;
    anim = mq.animate([{ transform: 'none', offset: 0 }, { transform: 'none', offset: .15 },
      { transform: `translateX(${-span}px)`, offset: .85 }, { transform: `translateX(${-span}px)`, offset: 1 }],
    { duration: Math.max(1600, span * 12) * 1.4, iterations: Infinity, direction: 'alternate', easing: 'ease-in-out' });
  });
  chip.addEventListener('pointerleave', () => { anim?.cancel(); anim = null; });
}

/* ═══ 主按钮：按钮即进度条；处理中变停止；6 个阶段刻度嵌在底边 ═══ */
const run = { busy: false, cancelling: false, settling: false, stage: -1, frac: -1, label: '', last: 0, shown: 0, started: 0 };
function goContent(icon, label, meta = '') {
  $$('.go-layer').forEach(layer => {
    $('use', layer).setAttribute('href', icon); $('.go-label', layer).textContent = label; $('.go-meta', layer).textContent = meta;
  });
}
function notches(stage, kind = 'run') {
  $$('#go-notches i').forEach((n, i) => {
    n.className = kind === 'done' || i < stage ? 'done' : i === stage ? (kind === 'error' ? 'error' : 'active') : '';
  });
}
function setBusy(on) {
  run.busy = on; const go = $('#go');
  go.classList.toggle('running', on); go.classList.remove('settled');
  for (const el of [$('#racks'), $('#presets'), $('#out-card'), ...$$('.chip, .dry-option')]) el.inert = on;
  $('#racks').classList.toggle('is-locked', on);
  if (on) {
    Object.assign(run, { cancelling: false, stage: 0, frac: -1, label: '准备处理', last: performance.now(), shown: 0, started: performance.now() });
    go.style.setProperty('--p', 0); notches(0); goContent('#i-stop', '停止'); go.setAttribute('aria-label', '停止处理');
    fx(true); requestAnimationFrame(progressFrame);
  } else {
    go.disabled = false; go.style.setProperty('--p', 0); goContent('#i-play', 'VOX!'); go.setAttribute('aria-label', '生成成品');
    fx(false);
  }
  $('#btn-batch').disabled = on && !batch.running;
}
function progressFrame(now) {
  if (!run.busy) return;
  const part = (run.stage + Math.max(0, run.frac)) / 6;
  const total = batch.running ? batch.jobs.length : 1, index = batch.running ? batch.index : 0;
  const real = (index + part) / total;
  // 长时间无新进度时缓慢逼近当前阶段上界，让进度保持可见的推进
  if (run.frac < 0 && now - run.last > 700) run.shown += ((index + (run.stage + .92) / 6) / total - run.shown) * .004;
  run.shown = Math.max(run.shown, real);
  $('#go').style.setProperty('--p', run.shown.toFixed(4));
  if (!run.cancelling) goContent('#i-stop', '停止', `${run.stage + 1}/6 · ${run.label} · ${Math.round((now - run.started) / 1000)} s`);
  requestAnimationFrame(progressFrame);
}
async function onGo() {
  const go = $('#go');
  if (run.settling) return;
  if (run.busy) {
    if (!run.cancelling) { run.cancelling = true; go.disabled = true; goContent('#i-stop', '停止中…', run.label); bridge.cancel(); }
    return;
  }
  const missing = !inputs.song ? $('#in-song') : !inputs.dry ? $('#in-dry') : null;
  if (missing) { markNeed(missing); return; }
  const out = store.get('output', '');
  if (!out || !(await call('folderExists', out))) { markNeed($('#out-card')); return; }
  if (await call('process', { ...params(), output: out })) setBusy(true);
}
function onStage(index, frac, label) {
  if (!run.busy) return;
  if (index !== run.stage) run.frac = -1;
  run.stage = index; run.frac = frac; run.label = label; run.last = performance.now(); notches(index);
}
function onFinished(json) {
  const data = JSON.parse(json), go = $('#go');
  run.busy = false; run.settling = true; go.classList.add('settled');
  go.style.setProperty('--p', 1); notches(6, 'done'); go.disabled = true;
  goContent('#i-check', `V${data.report.version}`, '已保存');
  announce(`已生成并保存 ${data.report.name}`);
  versions = data.versions; current = data.report.version;
  state.ready = true; renderVersions(); renderReport(data.report, 'reveal');
  // 完成后停顿一拍再复位，防止连点误触
  setTimeout(() => { run.settling = false; go.classList.remove('settled'); setBusy(false); }, 1300);
  for (const el of [$('#racks'), $('#presets'), $('#out-card'), ...$$('.chip, .dry-option')]) el.inert = false;
  fx(false);
}
function onFailed(json) {
  const { message, detail } = JSON.parse(json);
  if (run.busy) {
    const cancelled = run.cancelling;
    notches(run.stage, 'error'); setBusy(false);
    if (cancelled) { announce(message); return; }
  }
  showError(message, detail);
}

/* ═══ 播放器 ═══ */
const state = { ready: false, pos: 0, dur: 0, playing: false, sel: 2, delta: false, seeking: false };
let versions = [], current = null, wave = null;
const SRC_NAMES = ['修音前', '草稿', '成品'];

function resetResult() {
  state.ready = false; versions = []; current = null; wave = null;
  $('#player').dataset.cache = 'empty'; renderPlayer(); renderVersions(); renderReport(null, 'none'); drawWave();
}
function renderPlayer() {
  const player = $('#player'), ready = state.ready && !!wave;
  player.dataset.ready = ready; player.dataset.playing = state.playing && ready;
  for (const id of ['#p-start', '#p-play', '#p-equal-loudness', '#seek']) $(id).disabled = !ready;
  $('#p-play').setAttribute('aria-label', state.playing ? '暂停' : '播放');
  $$('.src[data-src]').forEach(b => {
    b.disabled = !ready || (+b.dataset.src === 1 && !switchOn($('#live-preview')));
    b.setAttribute('aria-pressed', String(!b.disabled && +b.dataset.src === state.sel));
  });
  $('#p-ver-btn').disabled = !ready || versions.length < 2;
  $('#p-ver').textContent = current ? `V${current}` : '';
  const delta = $('#p-delta'); delta.disabled = !ready || state.sel === 0;
  delta.setAttribute('aria-pressed', String(ready && state.delta));
  $('#p-src').textContent = ready ? (state.sel === 2 ? `成品 V${current}` : SRC_NAMES[state.sel]) + (state.delta ? ' · Δ' : '') : '未生成';
  $('#p-total').textContent = ` / ${clock(state.dur)}`;
  if (!state.seeking) setHead(state.pos);
  moveGlider($('#p-sources'), $('.src[aria-pressed="true"]'), '.src-glider');
}
function setHead(pos) {
  const track = $('#track'), x = state.dur ? pos / state.dur * track.clientWidth : 0;
  $('#playhead').style.transform = `translateX(${Math.round(x)}px)`;
  $('#p-time').textContent = clock(pos, true);
  if (!state.seeking) { const seek = $('#seek'); seek.max = state.dur || 1; seek.value = pos; }
  drawWave(pos);
}
function onPlayerState(json) {
  const s = JSON.parse(json), durChanged = Math.abs(s.dur - state.dur) > .01;
  Object.assign(state, { pos: s.pos, dur: s.dur, playing: s.playing, sel: s.sel, delta: s.delta });
  const meter = $$('#p-meter i');
  s.levels.forEach((l, i) => meter[i].style.setProperty('--l', clamp((20 * Math.log10(Math.max(l, 1e-6)) + 48) / 48, 0, 1)));
  if (durChanged) buildRuler();
  renderPlayer();
}
/* 当前音源的单层波形：峰值定轮廓，均方根决定填充的深浅。 */
function drawWave(pos = state.pos) {
  const canvas = $('#wave'), dpr = devicePixelRatio || 1;
  const w = Math.round(canvas.clientWidth * dpr), h = Math.round(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const g = canvas.getContext('2d'); g.clearRect(0, 0, w, h);
  const css = getComputedStyle(document.documentElement), mid = h / 2;
  g.fillStyle = css.getPropertyValue('--line'); g.fillRect(0, Math.floor(mid), w, 1);
  const track = wave?.tracks[state.sel]; if (!track) return;
  const bins = track.rms.length, head = state.dur ? pos / state.dur * w : 0, half = mid - 3 * dpr;
  const played = css.getPropertyValue('--accent'), rest = css.getPropertyValue('--wave');
  g.fillStyle = head > 0 ? played : rest;
  for (let x = 0; x < w; x++) {
    if (x === Math.ceil(head)) g.fillStyle = rest;
    const i0 = Math.floor(x / w * bins), i1 = Math.max(i0 + 1, Math.floor((x + 1) / w * bins));
    let peak = 0, power = 0, frames = 0;
    for (let i = i0; i < i1; i++) {
      peak = Math.max(peak, track.peak[i]);
      const n = Math.floor((i + 1) * wave.frames / bins) - Math.floor(i * wave.frames / bins);
      power += n * track.rms[i] ** 2;
      frames += n;
    }
    if (!peak) continue;
    const rms = Math.sqrt(power / frames);
    g.globalAlpha = .5 + .5 * Math.min(1, rms / peak);
    g.fillRect(x, mid - peak * half, 1, Math.max(1, peak * half * 2));
  }
  g.globalAlpha = 1;
}
function onWaveform(json) {
  const data = JSON.parse(json);
  wave = data.tracks ? data : null;
  if (wave) { state.dur = data.duration; state.sel = data.sel; buildRuler(); }
  renderPlayer();
}
function onWaveSegment(json) {
  const seg = JSON.parse(json), tr = wave?.tracks[seg.index]; if (!tr) return;
  seg.peak.forEach((v, i) => { tr.peak[seg.first + i] = v; tr.rms[seg.first + i] = seg.rms[i]; });
  if (seg.index === state.sel) drawWave();
}
/* 时间尺：主刻度间距约 85px，主刻度间 4 个小刻度 */
function buildRuler() {
  const ruler = $('#ruler'), w = ruler.clientWidth; ruler.textContent = '';
  if (!state.dur || !w) return;
  const step = [1, 2, 5, 10, 15, 30, 60, 120, 300].find(s => s / state.dur * w >= 85) || 600;
  const frag = document.createDocumentFragment();
  for (let s = 0; s <= state.dur + 1e-6; s += step / 4) {
    const x = s / state.dur * w, major = Math.abs(s / step - Math.round(s / step)) < 1e-6;
    const tick = document.createElement('i'); tick.style.left = `${x}px`; if (major) tick.className = 'major'; frag.appendChild(tick);
    if (major) {
      const label = document.createElement('span'); label.textContent = clock(s); label.style.left = `${x}px`;
      if (x > w - 44) label.className = 'end'; frag.appendChild(label);
    }
  }
  ruler.appendChild(frag);
}
function moveGlider(group, target, selector) {
  const glider = $(selector, group);
  group.dataset.active = !!target;
  if (!target) return;
  const to = { left: `${Math.round(target.offsetLeft)}px`, width: `${Math.round(target.offsetWidth)}px` };
  if (glider.style.left === to.left && glider.style.width === to.width) return;
  const first = !glider.style.width || RM.matches;
  if (!first) glider.animate([{ left: glider.style.left, width: glider.style.width }, to], { duration: 200, easing: EASE });
  Object.assign(glider.style, to);
}

/* 版本菜单：成品 V1…Vn，切换后试听与报告同步 */
function renderVersions() {
  const menu = $('#ver-menu'); menu.textContent = '';
  for (const v of [...versions].reverse()) {
    const item = document.createElement('button'); item.type = 'button'; item.className = 'menu-item'; item.setAttribute('role', 'option');
    item.setAttribute('aria-selected', String(v.n === current));
    item.innerHTML = `<b>V${v.n}</b><span></span>`; $('span', item).textContent = v.name; item.title = v.path;
    item.addEventListener('click', () => { menu.hidePopover(); bridge.selectVersion(v.n); bridge.select(2); });
    menu.appendChild(item);
  }
  renderPlayer();
}
function bindVersionMenu() {
  const menu = $('#ver-menu'), btn = $('#p-ver-btn');
  menu.addEventListener('beforetoggle', e => {
    btn.setAttribute('aria-expanded', String(e.newState === 'open'));
    if (e.newState !== 'open') return;
    const r = btn.getBoundingClientRect();
    menu.style.top = `${r.bottom + 6}px`; menu.style.left = `${Math.max(12, r.right - 260)}px`;
  });
  menu.addEventListener('toggle', e => { if (e.newState === 'open') ($('[aria-selected="true"]', menu) || $('.menu-item', menu))?.focus(); else if (menu.contains(document.activeElement) || document.activeElement === document.body) btn.focus(); });
  menu.addEventListener('keydown', e => {
    const items = $$('.menu-item', menu), i = items.indexOf(document.activeElement);
    const next = { ArrowDown: i + 1, ArrowUp: i - 1, Home: 0, End: items.length - 1 }[e.key];
    if (next === undefined) return;
    items[clamp(next, 0, items.length - 1)]?.focus(); e.preventDefault();
  });
}
function onVersionLoaded(json) {
  const r = JSON.parse(json); current = r.version; renderVersions(); renderReport(r, 'morph');
}
function onDraftState(json) {
  const s = JSON.parse(json), player = $('#player');
  if (!switchOn($('#live-preview')) && s.state !== 'empty') return;
  player.dataset.cache = s.state;
  if (s.state === 'error') {
    announce('草稿更新失败，可继续调整或重新生成成品');
    clearTimeout(player._flash); player._flash = setTimeout(() => { if (player.dataset.cache === 'error') player.dataset.cache = 'empty'; }, 1000);
  }
}

function bindPlayer() {
  $('#p-play').addEventListener('click', () => bridge.toggle());
  $('#p-start').addEventListener('click', () => bridge.seek(0));
  const equalLoudness = $('#p-equal-loudness');
  equalLoudness.addEventListener('click', () => {
    const enabled = !switchOn(equalLoudness);
    equalLoudness.setAttribute('aria-checked', String(enabled));
    bridge.setEqualLoudness(enabled);
  });
  $('#p-delta').addEventListener('click', () => bridge.setDelta(!state.delta));
  $$('.src[data-src]').forEach(b => b.addEventListener('click', () => bridge.select(+b.dataset.src)));
  const livePreview = $('#live-preview');
  livePreview.setAttribute('aria-checked', String(store.get('live_preview', true) !== false));
  livePreview.addEventListener('click', () => {
    const enabled = !switchOn(livePreview);
    livePreview.setAttribute('aria-checked', String(enabled));
    store.set('live_preview', enabled);
    bridge?.setLivePreview(enabled);
    if (enabled) changed();
    else onDraftState('{"state":"empty"}');
    renderPlayer();
  });
  const seek = $('#seek');
  seek.addEventListener('input', () => { state.seeking = true; setHead(+seek.value); });
  seek.addEventListener('change', () => {
    const t = +seek.value; state.seeking = false; state.pos = t; bridge.seek(t);
  });
  const track = $('#track'); let tip = null;
  track.addEventListener('pointermove', e => {
    if (!state.ready) return;
    tip ??= track.appendChild(Object.assign(document.createElement('span'), { className: 'hover-time' }));
    const x = clamp(e.clientX - track.getBoundingClientRect().left, 0, track.clientWidth);
    tip.textContent = clock(x / track.clientWidth * state.dur); tip.style.left = `${Math.min(x, track.clientWidth - 48)}px`;
  });
  track.addEventListener('pointerleave', () => { tip?.remove(); tip = null; });
  makeKnob($('#p-volume'), { min: 0, max: 100, step: 5, def: 100, fmt: pct }, 100, v => bridge.setVolume(v / 100));
  // 全局空格播放 / 暂停，保留开关和输入框自身的键盘操作。
  window.addEventListener('keydown', e => {
    if (e.code !== 'Space' || e.repeat || e.target.closest('input, textarea, [role="switch"], .modal:not([hidden])')) return;
    if (!state.ready) return;
    e.preventDefault(); bridge.toggle();
  }, true);
  bindVersionMenu();
}

/* ═══ 报告：共享 SVG、实际结果对比与社交图片导出 ═══ */
let shownReport = null, reportView = null, reportExporting = false;
function renderReport(r, mode = 'none') {
  shownReport = r; reportView = r;
  $('#rp-title').textContent = r ? `V${r.version} · ${r.name}` : '尚无成品';
  $('#rp-title').title = r ? r.name : '';
  $('#rp-reveal').disabled = !r;
  $('#rp-export').disabled = !r || reportExporting;
  $('#rp-listen').disabled = !r?.visual?.pitch?.available;
  $('#rp-note').textContent = r?.note || '生成成品后，可查看四组实际结果并导出分享图。';
  $('#rp-feedback').textContent = '';
  for (const [selector, rows] of [['#rp-summary', r?.summary || []], ['#rp-params', r?.parameters || []]]) {
    const list = $(selector); list.textContent = '';
    for (const [key, value] of rows) {
      const dt = document.createElement('dt'), dd = document.createElement('dd');
      dt.textContent = key; dd.textContent = value; list.append(dt, dd);
    }
  }
  const poster = $('#rp-poster'); poster.dataset.transition = RM.matches ? 'none' : mode;
  poster.innerHTML = ShadowReport.svg(r, true);
}
function drawReport() { /* SVG 按容器缩放，不在窗口缩放时重建图表或抢走焦点。 */ }
function reportSeek(seconds) {
  if (!shownReport || !bridge) return;
  bridge.select(2); bridge.seek(seconds);
  if (!state.playing) bridge.toggle();
  $('#rp-feedback').textContent = `已定位 ${clock(seconds)}，可在播放器切换处理前与成品。`;
  announce('已定位报告中的对照位置');
}
async function exportReport() {
  const model = shownReport;
  if (!model || reportExporting) return;
  reportExporting = true; $('#rp-export').disabled = true;
  $('#rp-feedback').textContent = '正在生成 1080×1440 分享图…';
  try {
    const image = await ShadowReport.png(model);
    const result = JSON.parse(await call('saveReportImage', image, model.version) || '{}');
    if (result.error) throw new Error(result.error);
    $('#rp-feedback').textContent = result.saved ? `V${model.version} 分享图已保存 · 1080×1440 PNG` : '已取消保存';
    if (result.saved) announce('分享图已保存');
  } catch (error) {
    $('#rp-feedback').textContent = '分享图导出失败'; showError('无法导出分享图。', error.message);
  } finally { reportExporting = false; $('#rp-export').disabled = !shownReport; }
}

/* ═══ 视图切换：滑块 200ms；新视图淡入上移；报告页 Esc 返回 ═══ */
function switchTab(name) {
  const report = name === 'report';
  $('#tab-process').setAttribute('aria-pressed', String(!report)); $('#tab-report').setAttribute('aria-pressed', String(report));
  moveGlider($('.tabs'), report ? $('#tab-report') : $('#tab-process'), '.tab-glider');
  const show = report ? $('#pane-report') : $('#pane-process'), hide = report ? $('#pane-process') : $('#pane-report');
  if (!show.hidden) return;
  hide.hidden = true; show.hidden = false; show.classList.remove('enter'); void show.offsetWidth; show.classList.add('enter');
  $('#dock').classList.toggle('hide', report);
  if (report) drawReport(reportView, null, 1, 1);
}

/* ═══ 弹窗：打开与关闭同路径；模态时背景 inert，关闭后焦点回到触发处 ═══ */
const openModals = [];
function openModal(m, trigger) {
  m._trigger = trigger || document.activeElement; m.hidden = false; openModals.push(m);
  for (const el of [$('.topbar'), $('#content'), $('#dock'), $('#help')]) el.inert = true;
  requestAnimationFrame(() => requestAnimationFrame(() => m.classList.add('open')));
  setTimeout(() => ($('[data-initial-focus]:not(:disabled)', m) || $('input', m) ||
    $('.modal-foot .key-primary:not(:disabled)', m) || $('[data-close]:not(.modal-backdrop)', m))?.focus(), 30);
}
function closeModal(m) {
  if (m.hidden || !m.classList.contains('open') && openModals.indexOf(m) < 0) return;
  m.classList.remove('open'); openModals.splice(openModals.indexOf(m), 1);
  setTimeout(() => { if (!m.classList.contains('open')) m.hidden = true; }, RM.matches ? 0 : 170);
  if (!openModals.length) for (const el of [$('.topbar'), $('#content'), $('#dock'), $('#help')]) el.inert = false;
  m._trigger?.focus?.();
}
function bindModals() {
  $$('.modal').forEach(m => {
    m.addEventListener('click', e => { if (e.target.closest('[data-close]')) closeModal(m); });
    m.addEventListener('keydown', e => {
      if (e.key !== 'Tab') return;
      const items = $$('button:not(:disabled), input', m).filter(x => x.offsetParent);
      const first = items[0], last = items[items.length - 1];
      if (e.shiftKey && document.activeElement === first) { last.focus(); e.preventDefault(); }
      else if (!e.shiftKey && document.activeElement === last) { first.focus(); e.preventDefault(); }
    });
  });
  window.addEventListener('keydown', e => {
    if (e.key !== 'Escape') return;
    if (openModals.length) { closeModal(openModals[openModals.length - 1]); e.preventDefault(); }
    else if (!$('#help').hidden) closeHelp();
    else if ($('#tab-report').getAttribute('aria-pressed') === 'true') switchTab('process');
  });
  $('#err-copy').addEventListener('click', async () => {
    const btn = $('#err-copy'), gen = (btn._gen || 0) + 1; btn._gen = gen; btn.disabled = true;
    const ok = await call('copyText', btn._text || '');
    if (btn._gen !== gen) return;
    btn.disabled = false; $('use', btn).setAttribute('href', ok ? '#i-check' : '#i-copy');
    btn.setAttribute('aria-label', ok ? '已复制' : '复制失败，重试'); announce(ok ? '已复制' : '复制失败');
    setTimeout(() => { if (btn._gen === gen) { $('use', btn).setAttribute('href', '#i-copy'); btn.setAttribute('aria-label', '复制错误信息'); } }, 1600);
  });
}
function showError(message, detail) {
  const body = $('#err-body'); body.textContent = message;
  if (detail) { const pre = document.createElement('div'); pre.className = 'detail'; pre.textContent = detail; body.appendChild(pre); }
  $('#err-copy')._text = detail ? `${message}\n\n${detail}` : message;
  if ($('#err-modal').hidden) openModal($('#err-modal'));
}

/* 参数说明：非模态、可拖动；首开居中，之后保持位置；再点同一个问号关闭 */
let helpKey = null, helpPlaced = false;
function toggleHelp(btn) {
  const help = $('#help'), key = btn.dataset.help;
  if (!help.hidden && helpKey === key) { closeHelp(); return; }
  helpKey = key; help._trigger = btn;
  $('#help-title').textContent = '声音处理说明';
  $('#help-body').innerHTML = '<dl></dl><p class="note"></p>';
  for (const [k, v] of HELP) { const dt = document.createElement('dt'), dd = document.createElement('dd'); dt.textContent = k; dd.textContent = v; $('#help-body dl').append(dt, dd); }
  $('#help-body .note').textContent = AUDITION;
  if (help.hidden) {
    help.hidden = false;
    if (!helpPlaced) { help.style.left = `${(innerWidth - 300) / 2}px`; help.style.top = `${Math.max(60, (innerHeight - help.offsetHeight) / 2)}px`; helpPlaced = true; }
    requestAnimationFrame(() => requestAnimationFrame(() => help.classList.add('open')));
  }
  $('#help-close').focus({ preventScroll: true });
}
function closeHelp() {
  const help = $('#help'); help.classList.remove('open');
  setTimeout(() => { if (!help.classList.contains('open')) help.hidden = true; }, RM.matches ? 0 : 170);
  help._trigger?.focus(); helpKey = null;
}
function bindHelp() {
  $('#help-close').addEventListener('click', closeHelp);
  const help = $('#help'), head = $('#help-drag'); let drag = null;
  head.addEventListener('pointerdown', e => {
    if (e.target.closest('button')) return;
    drag = { x: e.clientX - help.offsetLeft, y: e.clientY - help.offsetTop }; head.setPointerCapture(e.pointerId);
  });
  head.addEventListener('pointermove', e => {
    if (!drag) return;
    help.style.left = `${clamp(e.clientX - drag.x, 8, innerWidth - help.offsetWidth - 8)}px`;
    help.style.top = `${clamp(e.clientY - drag.y, 8, innerHeight - 40)}px`;
  });
  head.addEventListener('pointerup', () => { drag = null; });
}

/* 设置：版本 / 分离缓存（两步确认清空）/ VST3 */
function fmtBytes(n) { return n >= 1 << 30 ? `${(n / (1 << 30)).toFixed(2)} GiB` : `${(n / (1 << 20)).toFixed(0)} MiB`; }
async function refreshCache() {
  const info = JSON.parse(await call('cacheInfo')), btn = $('#set-cache-clear');
  $('#set-cache').textContent = fmtBytes(info.bytes); btn.disabled = info.busy || run.busy || !info.bytes;
}
function renderVst(json) {
  const list = $('#vst-list'); list.textContent = '';
  JSON.parse(json).forEach((p, i) => {
    const li = document.createElement('li'); li.innerHTML = '<span></span>'; $('span', li).textContent = p.name; li.title = p.path;
    const open = document.createElement('button'); open.type = 'button'; open.className = 'key';
    open.innerHTML = '<svg class="i"><use href="#i-open"/></svg>'; open.title = '打开插件界面'; open.setAttribute('aria-label', `打开 ${p.name} 界面`);
    open.addEventListener('click', () => bridge.vstOpen(i)); li.appendChild(open); list.appendChild(li);
  });
}
function syncTheme() {
  $$('[data-scheme-choice]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.schemeChoice === theme.mode)));
  $$('[data-palette-choice]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.paletteChoice === theme.palette)));
  $('.theme-custom').dataset.active = theme.palette === 'custom';
  bridge?.setNativeTheme(theme.mode, theme.mode === 'dark' ? '#17131b' : '#fffafb');
  drawWave();
}

function onUpdateInfo(json) {
  const { state: result, current, latest } = JSON.parse(json);
  const messages = {
    current: '已是最新正式版 V' + current,
    ahead: '当前 V' + current + '，公开正式版 V' + latest,
    newer: '发现 V' + latest + '，当前 V' + current,
    unavailable: '目前没有可访问的正式版本',
    limited: 'GitHub 暂时限制请求，请稍后重试',
    error: '无法检查更新，请检查网络后重试',
  };
  $('#update-status').textContent = messages[result];
  $('#check-update').disabled = false;
  $('#open-update').hidden = result !== 'newer';
}

function bindSettings() {
  const swatches = $('#theme-swatches'), picker = $('#theme-custom');
  for (const [id, [label, color]] of Object.entries(theme.palettes)) {
    const button = document.createElement('button');
    button.className = 'key theme-swatch'; button.type = 'button';
    button.title = label; button.setAttribute('aria-label', label);
    button.dataset.paletteChoice = id; button.style.setProperty('--swatch', color);
    button.addEventListener('click', () => theme.apply(theme.mode, id));
    swatches.insertBefore(button, picker.parentElement);
  }
  picker.value = theme.custom;
  picker.addEventListener('input', () => theme.apply(theme.mode, 'custom', picker.value));
  $$('[data-scheme-choice]').forEach(button => button.addEventListener('click', () => theme.apply(button.dataset.schemeChoice, theme.palette)));
  document.addEventListener('shadowvox-theme', syncTheme);
  syncTheme();
  $('#check-update').addEventListener('click', () => {
    $('#open-update').hidden = true;
    if (!bridge) { $('#update-status').textContent = '请在桌面应用中检查更新'; return; }
    $('#check-update').disabled = true; $('#update-status').textContent = '正在检查…';
    bridge.checkUpdate();
  });
  $('#open-update').addEventListener('click', () => bridge?.openUpdatePage());
  $('#btn-settings').addEventListener('click', async () => {
    openModal($('#settings'), $('#btn-settings'));
    if (bridge) {
      $('#set-version').textContent = await call('appVersion');
      $('#set-author').textContent = await call('appAuthor');
      renderVst(await call('vstList')); refreshCache();
    }
  });
  const clear = $('#set-cache-clear'); let confirmTimer = 0;
  clear.addEventListener('click', async () => {
    if (!clear.classList.contains('confirm')) {
      clear.classList.add('confirm'); clear.setAttribute('aria-label', '确认清空分离缓存？再次点击确认'); clear.title = '再次点击确认清空';
      confirmTimer = setTimeout(() => { clear.classList.remove('confirm'); clear.setAttribute('aria-label', '清空分离缓存'); clear.title = '清空分离缓存'; }, 4000);
      return;
    }
    clearTimeout(confirmTimer); clear.classList.remove('confirm'); clear.setAttribute('aria-label', '清空分离缓存');
    await call('clearCache'); refreshCache(); announce('已清空分离缓存');
  });
  $('#vst-add-file').addEventListener('click', async () => renderVst(await call('vstAdd', 'file')));
  $('#vst-add-folder').addEventListener('click', async () => renderVst(await call('vstAdd', 'folder')));
}
function onVstStatus(json) {
  const s = JSON.parse(json); $('#vst-status').textContent = s.text;
  $$('#vst-list .key').forEach(b => { b.disabled = s.busy; });
}

/* 批处理队列与单文件播放器分别保存状态。 */
const batch = { jobs: [], reference: '', running: false, index: 0 };
const fileName = path => path.split(/[\\/]/).pop();

function batchButton(label, action, disabled = false) {
  const button = document.createElement('button');
  button.className = 'key key-labeled'; button.type = 'button';
  button.textContent = label; button.disabled = disabled;
  button.addEventListener('click', action);
  return button;
}

function renderBatch() {
  const list = $('#batch-list'), locked = batch.running || run.busy;
  list.replaceChildren();
  for (const job of batch.jobs) {
    const row = document.createElement('li');
    row.dataset.state = job.state;
    const info = document.createElement('div');
    info.innerHTML = '<strong class="batch-name"></strong><span class="set-desc"></span><span class="set-desc batch-detail"></span>';
    $('.batch-name', info).textContent = fileName(job.dry);
    info.title = job.dry;
    $('.set-desc', info).textContent = job.song ? '原曲 · ' + fileName(job.song) : '请选择原曲';
    $('.batch-detail', info).textContent = job.error || (job.output && fileName(job.output)) ||
      ({ running: '处理中', cancelled: '已停止', pending: '等待处理' }[job.state] ?? '');
    if (job.error) $('.batch-detail', info).title = job.error;
    const actions = document.createElement('div'); actions.className = 'set-line';
    actions.append(batchButton('换原曲', async () => {
      const [song] = JSON.parse(await call('pickBatchFiles', 'song') || '[]');
      if (song) { Object.assign(job, { song, state: 'pending', error: '', output: '', html: '' }); renderBatch(); }
    }, locked));
    const backing = batchButton('含伴奏', () => {
      job.dry_has_backing = !job.dry_has_backing;
      backing.setAttribute('aria-checked', String(job.dry_has_backing));
    }, locked);
    backing.setAttribute('role', 'switch'); backing.setAttribute('aria-checked', String(job.dry_has_backing));
    actions.append(backing);
    if (job.html) actions.append(batchButton('报告', () => bridge.openBatchReport(job.html), batch.running));
    actions.append(batchButton('移除', () => { batch.jobs.splice(batch.jobs.indexOf(job), 1); renderBatch(); }, locked));
    row.append(info, actions); list.append(row);
  }
  const count = batch.jobs.length;
  $('#batch-button-label').textContent = count ? '批处理 · ' + count : '批处理';
  $('#batch-add').disabled = locked;
  $('#batch-clear').disabled = locked || !count;
  $('#batch-reference').disabled = locked || !count;
  $('#batch-start').disabled = batch.running ? run.cancelling : locked || !count;
  $('#batch-start').textContent = batch.running ? (run.cancelling ? '停止中…' : '停止批处理') : '开始批处理';
}

function addBatchFiles(paths) {
  if (run.busy || batch.running) return;
  for (const dry of paths) {
    if (!batch.jobs.some(job => job.dry === dry)) {
      batch.jobs.push({ dry, song: batch.reference || inputs.song, state: 'pending', dry_has_backing: false });
    }
  }
  renderBatch();
}

function openBatch() {
  if (run.busy && !batch.running) return;
  if ($('#batch-dialog').hidden) openModal($('#batch-dialog'), $('#btn-batch'));
  renderBatch();
}

function onBatchItem(json) {
  const item = JSON.parse(json);
  Object.assign(batch.jobs[item.index], item);
  if (item.state === 'running') {
    batch.index = item.index;
    Object.assign(run, { stage: 0, frac: -1, label: fileName(batch.jobs[item.index].dry), last: performance.now() });
    notches(0);
  }
  $('#batch-status').textContent = '处理 ' + (item.index + 1) + ' / ' + batch.jobs.length;
  renderBatch();
}

function onBatchDone(json) {
  const data = JSON.parse(json);
  batch.running = false;
  setBusy(false);
  const success = data.results.filter(item => item.state === 'done').length;
  const failures = data.results.filter(item => item.state === 'error').length;
  const prefix = data.cancelled ? '批处理已停止。' : '批处理完成。';
  $('#batch-status').textContent = prefix + '成功 ' + success + ' 项，失败 ' + failures + ' 项。';
  announce($('#batch-status').textContent);
  renderBatch();
}

function bindBatch() {
  $('#btn-batch').addEventListener('click', openBatch);
  $('#batch-add').addEventListener('click', async () => addBatchFiles(JSON.parse(await call('pickBatchFiles', 'dry') || '[]')));
  $('#batch-clear').addEventListener('click', () => {
    batch.jobs = []; batch.reference = ''; $('#batch-status').textContent = ''; renderBatch();
  });
  $('#batch-reference').addEventListener('click', async () => {
    const [song] = JSON.parse(await call('pickBatchFiles', 'song') || '[]');
    if (!song) return;
    batch.reference = song;
    for (const job of batch.jobs) Object.assign(job, { song, state: 'pending', error: '', output: '', html: '' });
    renderBatch();
  });
  $('#batch-start').addEventListener('click', async () => {
    if (batch.running) { onGo(); renderBatch(); return; }
    if (run.busy || run.settling) return;
    const status = $('#batch-status'), out = store.get('output', '');
    if (!batch.jobs.length || batch.jobs.some(job => !job.song)) { status.textContent = '请为每条任务指定原曲。'; return; }
    if (!out || !(await call('folderExists', out))) { status.textContent = '请先在工作台选择输出目录。'; return; }
    if (run.busy || batch.running) return;
    batch.running = true; batch.index = 0;
    for (const job of batch.jobs) Object.assign(job, { state: 'pending', error: '', output: '', html: '' });
    setBusy(true); status.textContent = '准备处理…'; renderBatch();
    const jobs = batch.jobs.map(({ dry, song, dry_has_backing }) => ({ dry, song, dry_has_backing }));
    if (!(await call('processBatch', jobs, { ...params(), output: out }))) {
      batch.running = false; setBusy(false); status.textContent = '当前无法启动批处理。';
      renderBatch();
    }
  });
  renderBatch();
}

/* 多个干音进入队列，单个文件保留原来的快速入口。 */
function onDrop(json) {
  const { paths, x, y } = JSON.parse(json), el = document.elementFromPoint(x, y);
  if (run.busy) { announce('处理中不能更换文件'); return; }
  const chip = el?.closest('.chip') || el?.closest('.dry-input')?.querySelector('.chip');
  const audioPaths = paths.filter(p => /\.(wav|mp3|flac|m4a|aac|ogg|aiff|wma)$/i.test(p));
  if (audioPaths.length && (el?.closest('#batch-dialog, #btn-batch') || (chip?.dataset.kind === 'dry' && audioPaths.length > 1))) {
    addBatchFiles(audioPaths); openBatch(); return;
  }
  const audio = audioPaths[0];
  if (chip && audio) { chooseInput(chip.dataset.kind, audio); return; }
  if (el?.closest('#out-card')) {
    const dir = paths.find(p => !/\.[a-z0-9]{2,4}$/i.test(p));
    if (dir) { setOutput(dir); return; }
  }
  announce('请把音频拖到原曲或干音框，把文件夹拖到输出目录');
}

/* ═══ 处理特效：粉色方形粒子缓慢上升；开始淡入，结束沿原路淡出 ═══ */
const fxState = { on: false, parts: [], raf: 0 };
function fx(on) {
  const canvas = $('#fx');
  if (on === fxState.on) return;
  fxState.on = on; canvas.classList.toggle('on', on && !RM.matches);
  if (on && !RM.matches && !fxState.raf) fxState.raf = requestAnimationFrame(fxFrame);
}
function fxFrame(now) {
  const canvas = $('#fx'), w = innerWidth, h = innerHeight, dpr = devicePixelRatio || 1;
  if (canvas.width !== w * dpr) { canvas.width = w * dpr; canvas.height = h * dpr; }
  const g = canvas.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
  const band = g.createLinearGradient(0, h, 0, h * .6), a = .07 + .04 * Math.sin(now / 700);
  const colors = getComputedStyle(document.documentElement);
  const accent = colors.getPropertyValue('--accent'), accentHi = colors.getPropertyValue('--accent-hi');
  band.addColorStop(0, 'color-mix(in srgb,' + accent + ' ' + a * 100 + '%, transparent)');
  band.addColorStop(1, 'transparent');
  g.fillStyle = band; g.fillRect(0, h * .6, w, h * .4);
  if (fxState.on && fxState.parts.length < 36 && Math.random() < .3)
    fxState.parts.push({ x: Math.random() * w, y: h + 6, s: 2 + Math.random() * 3, v: .25 + Math.random() * .5, life: 0, max: 4 + Math.random() * 4, hue: Math.random() < .5 });
  fxState.parts = fxState.parts.filter(p => (p.life += 1 / 60) < p.max);
  for (const p of fxState.parts) {
    p.y -= p.v; const alpha = Math.sin(Math.PI * p.life / p.max) * .28;
    g.globalAlpha = alpha; g.fillStyle = p.hue ? accent : accentHi; g.fillRect(p.x, p.y, p.s, p.s);
  }
  g.globalAlpha = 1;
  // 淡出过渡结束后才停止绘制
  if (!fxState.on && (canvas._off ??= now) && now - canvas._off > 450) { fxState.raf = 0; fxState.parts = []; canvas._off = undefined; g.clearRect(0, 0, w, h); return; }
  if (fxState.on) canvas._off = undefined;
  fxState.raf = requestAnimationFrame(fxFrame);
}

/* 独立 EXE 验收读取界面状态 */
window.__svState = () => ({ correction: P.correction,
  sound_values: { ...P },
  strength: params().strength ?? Math.min(1.5*P.correction,50+.5*P.correction),
  mix_enabled: SW.mix_effects, rhythm_enabled: params().rhythm ?? P.correction > 0,
  busy: run.busy, ready: state.ready, tab: $('#tab-report').getAttribute('aria-pressed') === 'true' ? 'report' : 'process' });

/* ═══ 启动 ═══ */
function init() {
  buildRacks(); buildPresets(); paint();
  setChip('song', ''); setChip('dry', ''); setOutput(store.get('output', ''));
  $$('.chip').forEach(chip => { chip.addEventListener('click', () => chooseInput(chip.dataset.kind)); bindMarquee(chip); });
  $('#dry-has-backing').addEventListener('click', ({ currentTarget }) => {
    currentTarget.setAttribute('aria-checked', String(!switchOn(currentTarget)));
    bridge?.invalidate();
    resetResult();
  });
  $('#out-card').addEventListener('click', async () => { const p = await call('pickOutput', store.get('output', '')); if (p) setOutput(p); });
  $('#go').addEventListener('click', onGo);
  $('#tab-process').addEventListener('click', () => switchTab('process'));
  $('#tab-report').addEventListener('click', () => switchTab('report'));
  $('#rp-reveal').addEventListener('click', () => shownReport && bridge.openFolder(shownReport.path));
  $('#rp-export').addEventListener('click', exportReport);
  $('#rp-listen').addEventListener('click', () => reportSeek(shownReport?.visual?.pitch?.window?.start || 0));
  $('#rp-poster').addEventListener('click', event => {
    const target = event.target.closest('[data-seek]'); if (target) reportSeek(+target.dataset.seek);
  });
  bindPlayer(); bindModals(); bindHelp(); bindSettings(); bindBatch();
  renderPlayer(); renderReport(null, 'none');
  requestAnimationFrame(() => moveGlider($('.tabs'), $('#tab-process'), '.tab-glider'));
  addEventListener('resize', () => { buildRuler(); renderPlayer(); drawReport(reportView, null, 1, 1); moveGlider($('.tabs'), $('.tab[aria-pressed="true"]'), '.tab-glider'); });
}
init();
if (window.qt?.webChannelTransport) {
  new QWebChannel(window.qt.webChannelTransport, channel => {
    bridge = channel.objects.bridge;
    bridge.stage.connect(onStage); bridge.finished.connect(onFinished); bridge.failed.connect(onFailed);
    bridge.playerState.connect(onPlayerState); bridge.waveform.connect(onWaveform); bridge.waveformSegment.connect(onWaveSegment);
    bridge.draftState.connect(onDraftState); bridge.versionLoaded.connect(onVersionLoaded);
    bridge.filesDropped.connect(onDrop); bridge.vstStatus.connect(onVstStatus);
    bridge.updateInfo.connect(onUpdateInfo); bridge.batchItem.connect(onBatchItem); bridge.batchDone.connect(onBatchDone);
    syncTheme();
    bridge.setLivePreview(switchOn($('#live-preview')));
  });
}
