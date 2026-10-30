/* ShadowVox 前端：界面状态在这里，真实处理与音频经 QWebChannel 交给 Python。
   交互逻辑沿用 ShadowBuster：记住参数、预设槽、旁路整块变淡、缺项红框、按钮即进度、成功静默。 */
'use strict';
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const RM = matchMedia('(prefers-reduced-motion: reduce)');
const EASE = 'cubic-bezier(.25,.6,.35,1)';
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const store = {
  get(k, d) { try { const v = localStorage.getItem('sv_' + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem('sv_' + k, JSON.stringify(v)); } catch (e) { /* 存储不可用时只影响记忆 */ } },
};
let bridge = null;
const call = (name, ...args) => new Promise(res => (bridge ? bridge[name](...args, res) : res(undefined)));

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
const easeOut = t => 1 - Math.pow(1 - t, 3);
/* 数值动画：结束一定回调；reduced-motion 直接到终态 */
function tween(ms, step) {
  let done = false, start = null;
  const finish = () => { if (!done) { done = true; step(1); } };
  if (RM.matches || ms <= 0) { finish(); return; }
  const frame = now => {
    if (done) return;
    start ??= now;
    const t = Math.min(1, (now - start) / ms);
    if (t >= 1) finish(); else { step(easeOut(t)); requestAnimationFrame(frame); }
  };
  requestAnimationFrame(frame);
  setTimeout(finish, ms + 250);
}
const clock = (s, ms = false) => {
  s = Math.max(0, s || 0);
  const m = Math.floor(s / 60), sec = s - m * 60;
  return ms ? `${String(m).padStart(2, '0')}:${sec.toFixed(3).padStart(6, '0')}` : `${String(m).padStart(2, '0')}:${String(Math.floor(sec)).padStart(2, '0')}`;
};

/* ═══ 参数表：唯一来源；默认值与已确认的处理链一致 ═══ */
const pct = v => `${Math.round(v)}%`;
const PARAMS = {
  strength: { name: '音高', label: '音高修正', min: 0, max: 100, step: 1, def: 75, fmt: pct },
  rhythm_strength: { name: '节奏', label: '节奏强度', min: 0, max: 100, step: 5, def: 100, fmt: pct, power: 'rhythm' },
  denoise_strength: { name: '降噪', label: 'AI 降噪强度', min: 0, max: 100, step: 5, def: 100, fmt: pct, power: 'denoise' },
  vocal_gain_db: { name: '人声音量', label: '人声音量', min: -6, max: 6, step: .5, def: 0, fmt: v => `${v > 0 ? '+' : ''}${v.toFixed(1)} dB` },
  target_lufs: { name: '成品目标响度', label: '整首成品目标综合响度（人声与伴奏）', min: -24, max: -12, step: 1, def: -16, fmt: v => `${v} LUFS`, mix: true },
  true_peak_ceiling: { name: '成品峰值上限', label: '整首成品真峰值上限（人声与伴奏）', min: -3, max: -1, step: .1, def: -1.5, fmt: v => `${v.toFixed(1)} dBTP`, mix: true },
  mix_lowcut_hz: { name: '低切', label: '低切', min: 0, max: 150, step: 5, def: 75, fmt: v => (v ? `${v} Hz` : '关闭') },
  mix_eq: { name: '均衡', label: '均衡', min: 0, max: 200, step: 5, def: 100, fmt: pct },
  mix_compression: { name: '压缩', label: '压缩', min: 0, max: 200, step: 5, def: 100, fmt: pct },
  mix_deess: { name: '去齿', label: '去齿', min: 0, max: 200, step: 5, def: 100, fmt: pct },
  mix_reverb: { name: '混响', label: '混响', min: 0, max: 200, step: 5, def: 100, fmt: pct },
  mix_delay: { name: '延时', label: '延时', min: 0, max: 200, step: 5, def: 100, fmt: pct },
  mix_duck: { name: '伴奏避让', label: '伴奏避让', min: 0, max: 200, step: 5, def: 100, fmt: pct },
};
const SWITCHES = { rhythm: '节奏匹配', denoise: 'AI 降噪', declip: '削波修复', mix_effects: '后期总开关' };
const RACKS = [
  { key: 'cleanup', title: '干声清理', cells: ['@declip', 'denoise_strength'] },
  { key: 'tune', title: '修音', cells: ['rhythm_strength', 'strength'] },
  { key: 'level', title: '人声电平', cells: ['vocal_gain_db'] },
  { key: 'tone', title: '音色与动态', post: true, bypass: 'mix_effects', groups: [
    { title: 'EQ', cells: ['mix_lowcut_hz', 'mix_eq'] },
    { title: '动态', cells: ['mix_compression', 'mix_deess'] }] },
  { key: 'mix', title: '混音', post: true, groups: [
    { title: '空间与伴奏', cells: ['mix_duck', 'mix_reverb', 'mix_delay'] }] },
  { key: 'output', title: '成品输出', post: true, groups: [
    { title: '人声 + 伴奏', cells: ['target_lufs', 'true_peak_ceiling'] }] },
];
const HELP = {
  cleanup: [['削波修复', '先检查原始干声，只在检测到明显数字削波时保守修复；这个开关不会增加削波。归一化或有损编码后的削波可能漏检。'],
    ['降噪', '削波检查后，再在原始干声与 DeepFilterNet3 温和降噪结果之间混合。0% 旁路降噪，100% 使用已确认的温和档；点按名称可开关。']],
  tune: [['节奏', '当前支持中文歌声。先识别唱词、核对字头与韵母，再修音高。100% 使用受限对齐；0% 或关闭时保持原始时间位置。未匹配处保持原位原速，始终保留原始歌词顺序。'],
    ['音高', '只跟随演唱的音高中心修正，保留大部分颤音与滑音；用 PSOLA 直接在原始波形上移调。严重的音高误检处可能有瑕疵。']],
  level: [['人声音量', '修音后、进入音色和空间处理前的人声额外增益，范围 ±6 dB。只调整人声；整首歌的输出响度在最后的“成品输出”面板设定。']],
  tone: [['EQ / 动态', '低切、均衡与压缩、去齿共同处理人声。现有均衡包含压缩前后的环节，因此在同一面板内分组，不改变既有处理顺序。100% 对应已确认的默认效果，0% 旁路该项。'],
    ['后期总开关', '同时开关“音色与动态”“混音”和“成品输出”。关闭后只将人声与伴奏合并，不做这些后期效果；上方人声音量仍可调。']],
  mix: [['伴奏避让', '处理后的人声触发伴奏让位，与人声空间效果汇合成混音。'],
    ['混响 / 延时', '开源 Freeverb 自然流行混响与独立短延时发送；不调用 VST3 效果。'],
    ['开关', '受“音色与动态”标题右侧的后期总开关控制。']],
  output: [['成品目标响度', '人声和伴奏混合后的整首歌目标综合响度，范围 −24 至 −12 LUFS；不单独调整人声。'],
    ['成品峰值上限', '整首歌的真峰值上限，范围 −3 至 −1 dBTP。为保护峰值，实际综合响度可能低于目标；以报告中的实测值为准。'],
    ['开关', '后期总开关关闭时，这两个成品输出设置不参与处理。']],
};
const AUDITION = '调整参数后，草稿会在后台重算当前播放位置附近的片段，属局部近似试听；要保存新设置，请再次生成成品。';

const P = {}, SW = {}, knobs = {};
for (const [k, d] of Object.entries(PARAMS)) P[k] = snapTo(store.get('param_' + k, d.def), d);
for (const k of Object.keys(SWITCHES)) SW[k] = store.get('switch_' + k, true) !== false;
function snapTo(v, d) {
  v = Number(v); if (!Number.isFinite(v)) v = d.def;
  return +clamp(Math.round((v - d.min) / d.step) * d.step + d.min, d.min, d.max).toFixed(4);
}
const params = () => ({ ...P, ...SW });

/* ═══ 旋钮：上下拖动（约 200px 走满，Shift 精调）、滚轮、方向键、双击复位 ═══ */
function makeKnob(el, d, value, onChange) {
  el.classList.add('knob-wrap'); el.tabIndex = 0; el.setAttribute('role', 'slider');
  el.innerHTML = '<span class="knob-ring"></span><span class="knob-cap"></span><span class="knob-tick"></span>';
  el.setAttribute('aria-valuemin', d.min); el.setAttribute('aria-valuemax', d.max);
  const render = () => {
    el.style.setProperty('--v', (value - d.min) / (d.max - d.min));
    el.setAttribute('aria-valuenow', value); el.setAttribute('aria-valuetext', d.fmt(value));
  };
  const set = (v, user) => {
    v = snapTo(v, d); if (v === value) return;
    value = v; render(); if (user) onChange(v);
  };
  let drag = null;
  el.addEventListener('pointerdown', e => {
    if (e.button !== 0) return;
    el.setPointerCapture(e.pointerId); el.focus({ preventScroll: true });
    drag = { y: e.clientY, v: value }; el.classList.add('dragging'); e.preventDefault();
  });
  el.addEventListener('pointermove', e => {
    if (!drag) return;
    set(drag.v + (drag.y - e.clientY) * (d.max - d.min) / (e.shiftKey ? 800 : 200), true);
  });
  const end = () => { drag = null; el.classList.remove('dragging'); };
  el.addEventListener('pointerup', end); el.addEventListener('pointercancel', end);
  el.addEventListener('wheel', e => { e.preventDefault(); set(value + (e.deltaY < 0 ? d.step : -d.step), true); }, { passive: false });
  el.addEventListener('dblclick', () => set(d.def, true));
  el.addEventListener('keydown', e => {
    const steps = { ArrowUp: 1, ArrowRight: 1, ArrowDown: -1, ArrowLeft: -1, PageUp: 5, PageDown: -5 }[e.key];
    if (steps) set(value + steps * d.step, true);
    else if (e.key === 'Home') set(d.min, true);
    else if (e.key === 'End') set(d.max, true);
    else return;
    e.preventDefault();
  });
  render();
  return { get: () => value, set: v => { value = snapTo(v, d); render(); } };
}

/* ═══ 面板：按参数表生成；面板头 = 编号 + 标题 + 说明 + 旁路 ═══ */
function buildRacks() {
  const root = $('#racks');
  RACKS.forEach((rack, i) => {
    const el = document.createElement('div');
    el.className = `rack rack-${rack.key} rise` + (rack.post ? ' rack-post' : '');
    el.style.setProperty('--d', `${120 + i * 50}ms`); el.dataset.rack = rack.key;
    el.innerHTML = `<div class="rack-head"><span class="rack-index">0${i + 1}</span><h2 class="rack-title">${rack.title}</h2><span class="spacer"></span>
      <button class="key" type="button" data-help="${rack.key}" title="参数说明" aria-label="${rack.title} 参数说明" aria-haspopup="dialog"><svg class="i"><use href="#i-help"/></svg></button>
      ${rack.bypass ? `<button class="key key-labeled" type="button" role="switch" data-switch="${rack.bypass}" title="开关全部后期：音色与动态、混音、成品输出" aria-label="${SWITCHES[rack.bypass]}" aria-controls="rack-tone-body rack-mix-body rack-output-body"><svg class="i"><use href="#i-power"/></svg><span>后期总开关</span></button>` : ''}
      </div><div class="rack-body"></div>`;
    const body = $('.rack-body', el);
    body.id = `rack-${rack.key}-body`;
    if (rack.groups) body.classList.add('grouped-body');
    for (const group of rack.groups || [{ cells: rack.cells }]) {
      let host = body;
      if (group.title) {
        const section = document.createElement('section'); section.className = 'control-group';
        section.innerHTML = `<h3 class="control-group-title">${group.title}</h3><div class="control-group-cells"></div>`;
        body.appendChild(section); host = $('.control-group-cells', section);
      }
      for (const key of group.cells) {
        const cell = document.createElement('div'); cell.className = 'cell';
        if (key.startsWith('@')) {
          const k = key.slice(1); cell.dataset.key = k;
          cell.innerHTML = `<button class="toggle-cell" type="button" role="switch" data-switch="${k}" aria-label="${SWITCHES[k]}" title="${SWITCHES[k]}：仅在检测到明显数字削波时修复"><svg class="i"><use href="#i-clip"/></svg></button>
            <span class="cell-name">${SWITCHES[k]}</span><output class="cell-value" data-switch-value="${k}"></output>`;
        } else {
          const d = PARAMS[key]; cell.dataset.key = key;
          const name = d.power
            ? `<button class="cell-name" type="button" role="switch" data-switch="${d.power}" aria-label="${SWITCHES[d.power]}" title="点按开关${SWITCHES[d.power]}"><svg class="i"><use href="#i-power"/></svg>${d.name}</button>`
            : `<span class="cell-name">${d.name}</span>`;
          cell.innerHTML = `<div class="knob-host" aria-label="${d.label}" title="${d.label}：上下拖动、滚轮或方向键，双击复位"></div>${name}<output class="cell-value"></output>`;
          const out = $('output', cell);
          knobs[key] = makeKnob($('.knob-host', cell), d, P[key], v => { P[key] = v; out.textContent = d.fmt(v); store.set('param_' + key, v); changed(); });
          out.textContent = d.fmt(P[key]);
        }
        host.appendChild(cell);
      }
    }
    root.appendChild(el);
  });
  $$('[data-switch]').forEach(b => b.addEventListener('click', () => {
    const k = b.dataset.switch; SW[k] = !SW[k]; store.set('switch_' + k, SW[k]); paint(); changed();
  }));
  $$('[data-help]').forEach(b => b.addEventListener('click', () => toggleHelp(b)));
}

/* 失效态：整块变淡并锁住；开关本身始终可点 */
function paint() {
  $$('[data-switch]').forEach(b => b.setAttribute('aria-checked', String(SW[b.dataset.switch])));
  $$('[data-switch-value]').forEach(o => { o.textContent = SW[o.dataset.switchValue] ? '开' : '关'; });
  $$('.rack-post .rack-body').forEach(body => {
    body.classList.toggle('is-disabled', !SW.mix_effects); body.inert = !SW.mix_effects;
  });
  for (const [key, d] of Object.entries(PARAMS)) {
    const cell = $(`.cell[data-key="${key}"]`), knob = $('.knob-wrap', cell);
    if (d.mix) {
      cell.inert = !SW.mix_effects;
      cell.title = SW.mix_effects ? '' : '开启后期总开关后可用';
    }
    if (d.power) { cell.classList.toggle('off', !SW[d.power]); knob.inert = !SW[d.power]; }
  }
}

/* 参数变化：已有成品时交给后台更新草稿 */
function changed() {
  if (state.ready && !run.busy) bridge?.draft(params());
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
  presets[n] = { savedAt: Date.now(), params: { ...P }, switches: { ...SW } };
  store.set('presets', presets); flashPad(n); announce(`已保存预设 ${n}`);
}
function presetClick(n) {
  if (armed === n) { disarm(); savePreset(n); renderPresets(); return; }
  disarm();
  if (!presets[n]) { savePreset(n); renderPresets(); return; }
  const p = presets[n];
  for (const [k, d] of Object.entries(PARAMS)) {
    if (k in p.params) { P[k] = snapTo(p.params[k], d); knobs[k].set(P[k]); store.set('param_' + k, P[k]); $(`.cell[data-key="${k}"] output`).textContent = d.fmt(P[k]); }
  }
  for (const k of Object.keys(SWITCHES)) if (k in p.switches) { SW[k] = !!p.switches[k]; store.set('switch_' + k, SW[k]); }
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
  for (const el of [$('#racks'), $('#presets'), $('#out-card'), ...$$('.chip')]) el.inert = on;
  $('#racks').classList.toggle('is-locked', on);
  if (on) {
    Object.assign(run, { cancelling: false, stage: 0, frac: -1, label: '准备处理', last: performance.now(), shown: 0, started: performance.now() });
    go.style.setProperty('--p', 0); notches(0); goContent('#i-stop', '停止'); go.setAttribute('aria-label', '停止处理');
    fx(true); requestAnimationFrame(progressFrame);
  } else {
    go.disabled = false; go.style.setProperty('--p', 0); goContent('#i-play', 'VOX!'); go.setAttribute('aria-label', '生成成品');
    fx(false);
  }
}
function progressFrame(now) {
  if (!run.busy) return;
  const real = (run.stage + Math.max(0, run.frac)) / 6;
  // 长时间无新进度时缓慢逼近当前阶段上界，让进度保持可见的推进
  if (run.frac < 0 && now - run.last > 700) run.shown += ((run.stage + .92) / 6 - run.shown) * .004;
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
  for (const el of [$('#racks'), $('#presets'), $('#out-card'), ...$$('.chip')]) el.inert = false;
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
let versions = [], current = null, wave = null, waveMix = { from: 2, to: 2, t: 1 };
const loop = { a: 0, b: 0, on: false, has: false };
const SRC_NAMES = ['修音前', '草稿', '成品'];

function resetResult() {
  state.ready = false; versions = []; current = null; wave = null;
  Object.assign(loop, { on: false, has: false });
  $('#player').dataset.cache = 'empty'; renderPlayer(); renderVersions(); renderReport(null, 'none'); drawWave();
}
function renderPlayer() {
  const player = $('#player'), ready = state.ready && !!wave;
  player.dataset.ready = ready; player.dataset.playing = state.playing && ready; player.dataset.loop = loop.on;
  for (const id of ['#p-start', '#p-play', '#p-loop', '#seek']) $(id).disabled = !ready;
  $('#p-play').setAttribute('aria-label', state.playing ? '暂停' : '播放');
  $('#p-loop').setAttribute('aria-pressed', String(loop.on));
  $$('.src[data-src]').forEach(b => { b.disabled = !ready; b.setAttribute('aria-pressed', String(ready && +b.dataset.src === state.sel)); });
  $('#p-ver-btn').disabled = !ready || versions.length < 2;
  $('#p-ver').textContent = current ? `V${current}` : '';
  const delta = $('#p-delta'); delta.disabled = !ready || state.sel === 0;
  delta.setAttribute('aria-pressed', String(ready && state.delta));
  $('#p-src').textContent = ready ? (state.sel === 2 ? `成品 V${current}` : SRC_NAMES[state.sel]) + (state.delta ? ' · Δ' : '') : '未生成';
  $('#p-total').textContent = ` / ${clock(state.dur)}`;
  if (!state.seeking) setHead(state.pos);
  moveGlider($('#p-sources'), $('.src[aria-pressed="true"]'), '.src-glider');
  renderLoop();
}
function setHead(pos) {
  const track = $('#track'), x = state.dur ? pos / state.dur * track.clientWidth : 0;
  $('#playhead').style.transform = `translateX(${Math.round(x)}px)`;
  $('#p-time').textContent = clock(pos, true);
  if (!state.seeking) { const seek = $('#seek'); seek.max = state.dur || 1; seek.value = pos; }
  drawWave();
}
function onPlayerState(json) {
  const s = JSON.parse(json), durChanged = Math.abs(s.dur - state.dur) > .01;
  const selChanged = s.sel !== state.sel;
  if (selChanged) { waveMix = { from: waveCurrent(), to: s.sel, t: 0 }; tween(160, t => { waveMix.t = t; drawWave(); }); }
  Object.assign(state, { pos: s.pos, dur: s.dur, playing: s.playing, sel: s.sel, delta: s.delta });
  const meter = $$('#p-meter i');
  s.levels.forEach((l, i) => meter[i].style.setProperty('--l', clamp((20 * Math.log10(Math.max(l, 1e-6)) + 48) / 48, 0, 1)));
  if (durChanged) buildRuler();
  renderPlayer();
}
function waveCurrent() { return waveMix.t >= 1 ? waveMix.to : waveMix.from; }

/* 波形：RMS 实心 + 峰值浅色；已播放部分用强调色；切换版本 160ms 过渡 */
function drawWave() {
  const canvas = $('#wave'), w = canvas.clientWidth, h = canvas.clientHeight, dpr = devicePixelRatio || 1;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); }
  const g = canvas.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
  const css = getComputedStyle(document.documentElement), mid = h / 2;
  g.fillStyle = css.getPropertyValue('--line'); g.fillRect(0, Math.floor(mid), w, 1);
  if (!wave || !wave.tracks.length) return;
  const a = wave.tracks[waveMix.from] || wave.tracks[0], b = wave.tracks[waveMix.to] || a, t = waveMix.t;
  const bins = a.rms.length, head = state.dur ? state.pos / state.dur * w : 0, half = mid - 3;
  const colors = { played: [css.getPropertyValue('--accent'), css.getPropertyValue('--accent-hi')],
    rest: [css.getPropertyValue('--wave'), css.getPropertyValue('--line')] };
  for (let x = 0; x < w; x++) {
    const i0 = Math.floor(x / w * bins), i1 = Math.max(i0 + 1, Math.floor((x + 1) / w * bins));
    let peak = 0, rms = 0;
    for (let i = i0; i < i1 && i < bins; i++) {
      peak = Math.max(peak, a.peak[i] * (1 - t) + b.peak[i] * t); rms = Math.max(rms, a.rms[i] * (1 - t) + b.rms[i] * t);
    }
    const [solid, faint] = x < head ? colors.played : colors.rest;
    g.fillStyle = faint; g.fillRect(x, mid - peak * half, 1, Math.max(1, peak * half * 2));
    g.fillStyle = solid; g.fillRect(x, mid - rms * half, 1, Math.max(1, rms * half * 2));
  }
}
function onWaveform(json) {
  const data = JSON.parse(json);
  wave = data.tracks ? data : null;
  if (wave) { state.dur = data.duration; buildRuler(); }
  renderPlayer();
}
function onWaveSegment(json) {
  const seg = JSON.parse(json), tr = wave?.tracks[seg.index]; if (!tr) return;
  seg.peak.forEach((v, i) => { tr.peak[seg.first + i] = v; tr.rms[seg.first + i] = seg.rms[i]; });
  drawWave();
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

/* 循环带：拖空白创建，拖两端调整，拖中间移动；小于 3px 不算拖动 */
function renderLoop() {
  const region = $('#loop-region'), w = $('#loop-lane').clientWidth;
  region.hidden = !loop.has || !state.dur;
  if (region.hidden) return;
  const l = loop.a / state.dur * w, r = loop.b / state.dur * w;
  Object.assign(region.style, { left: `${l}px`, width: `${Math.max(2, r - l)}px` });
  $('#loop-label').textContent = r - l >= 100 ? `${clock(loop.a)}–${clock(loop.b)}` : '';
}
function commitLoop(on = loop.on) {
  loop.on = on && loop.has; bridge?.setLoop(loop.on, loop.a, loop.b); renderPlayer();
}
function bindLoopLane() {
  const lane = $('#loop-lane'); let drag = null;
  const timeAt = x => clamp((x - lane.getBoundingClientRect().left) / lane.clientWidth * state.dur, 0, state.dur);
  lane.addEventListener('pointerdown', e => {
    if (!state.ready || e.button !== 0) return;
    const edge = e.target.dataset.edge, inRegion = e.target.closest('.loop-region');
    drag = { x: e.clientX, mode: edge || (inRegion ? 'move' : 'new'), t0: timeAt(e.clientX), a: loop.a, b: loop.b, has: loop.has, moved: false };
    lane.setPointerCapture(e.pointerId);
  });
  lane.addEventListener('pointermove', e => {
    if (!drag) return;
    if (!drag.moved && Math.abs(e.clientX - drag.x) < 3) return;
    drag.moved = true;
    const t = timeAt(e.clientX), len = drag.b - drag.a;
    if (drag.mode === 'new') { loop.a = Math.min(drag.t0, t); loop.b = Math.max(drag.t0, t); loop.has = true; }
    else if (drag.mode === 'a') loop.a = Math.min(t, loop.b - .05);
    else if (drag.mode === 'b') loop.b = Math.max(t, loop.a + .05);
    else { loop.a = clamp(drag.a + t - drag.t0, 0, state.dur - len); loop.b = loop.a + len; }
    renderLoop();
  });
  const end = cancel => {
    if (!drag) return;
    if (!drag.moved || cancel || loop.b - loop.a < .05) Object.assign(loop, { a: drag.a, b: drag.b, has: drag.has });
    else commitLoop(true);
    drag = null; renderLoop();
  };
  lane.addEventListener('pointerup', () => end(false)); lane.addEventListener('pointercancel', () => end(true));
  lane.addEventListener('dblclick', () => state.ready && openLoopEditor());
  lane.addEventListener('keydown', e => {
    if (!state.ready) return;
    if (e.key === 'Enter') { openLoopEditor(); e.preventDefault(); return; }
    if (!loop.has) return;
    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
      const len = loop.b - loop.a, d = (e.key === 'ArrowLeft' ? -1 : 1) * (e.shiftKey ? 10 : 1);
      loop.a = clamp(loop.a + d, 0, state.dur - len); loop.b = loop.a + len; commitLoop(); e.preventDefault();
    } else if (e.key === 'Delete' || e.key === 'Backspace') { loop.has = false; commitLoop(false); e.preventDefault(); }
  });
}
function parseClock(text) {
  const m = /^\s*(?:(\d+):)?(\d+(?:\.\d+)?)\s*$/.exec(text); return m ? (+(m[1] || 0)) * 60 + (+m[2]) : NaN;
}
function openLoopEditor() {
  $('#loop-a').value = clock(loop.has ? loop.a : 0, true); $('#loop-b').value = clock(loop.has ? loop.b : state.dur, true);
  $('#loop-error').textContent = ''; openModal($('#loop-editor'), $('#loop-lane'));
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
  player.dataset.cache = s.state;
  if (s.state === 'error') {
    announce('草稿更新失败，可继续调整或重新生成成品');
    clearTimeout(player._flash); player._flash = setTimeout(() => { if (player.dataset.cache === 'error') player.dataset.cache = 'empty'; }, 1000);
  }
}

function bindPlayer() {
  $('#p-play').addEventListener('click', () => bridge.toggle());
  $('#p-start').addEventListener('click', () => bridge.seek(loop.on ? loop.a : 0));
  $('#p-loop').addEventListener('click', () => {
    if (!loop.has) Object.assign(loop, { a: 0, b: state.dur, has: true });
    commitLoop(!loop.on);
  });
  $('#p-delta').addEventListener('click', () => bridge.setDelta(!state.delta));
  $$('.src[data-src]').forEach(b => b.addEventListener('click', () => bridge.select(+b.dataset.src)));
  const seek = $('#seek');
  seek.addEventListener('input', () => { state.seeking = true; setHead(+seek.value); });
  seek.addEventListener('change', () => {
    const t = +seek.value; state.seeking = false; state.pos = t; bridge.seek(t);
    if (loop.on && (t < loop.a || t > loop.b)) commitLoop(false);   // 跳出循环区间即关闭循环
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
  // 全局空格播放 / 暂停（输入框除外，忽略按键重复）
  window.addEventListener('keydown', e => {
    if (e.code !== 'Space' || e.repeat || e.target.closest('input, textarea, .modal:not([hidden])')) return;
    if (!state.ready) return;
    e.preventDefault(); bridge.toggle();
  }, true);
  bindLoopLane(); bindVersionMenu();
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
  poster.innerHTML = ShadowReport.svg(r);
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
  setTimeout(() => ($('input', m) || $('.modal-foot .key-primary', m) || $('[data-close]:not(.modal-backdrop)', m))?.focus(), 30);
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
  $('#loop-form').addEventListener('submit', e => {
    e.preventDefault();
    const a = parseClock($('#loop-a').value), b = parseClock($('#loop-b').value);
    if (!(a >= 0) || !(b <= state.dur + .001) || !(b - a >= .05)) { $('#loop-error').textContent = '请输入 分:秒，终点需晚于起点且不超过总长'; return; }
    Object.assign(loop, { a, b: Math.min(b, state.dur), has: true }); commitLoop(true); closeModal($('#loop-editor'));
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
  const rack = RACKS.find(r => r.key === key);
  $('#help-title').textContent = `${rack.title} · 参数说明`;
  $('#help-body').innerHTML = '<dl></dl><p class="note"></p>';
  for (const [k, v] of HELP[key]) { const dt = document.createElement('dt'), dd = document.createElement('dd'); dt.textContent = k; dd.textContent = v; $('#help-body dl').append(dt, dd); }
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
function bindSettings() {
  $('#btn-settings').addEventListener('click', async () => {
    openModal($('#settings'), $('#btn-settings'));
    $('#set-version').textContent = await call('appVersion');
    renderVst(await call('vstList')); refreshCache();
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

/* 拖入：落在原曲 / 干音框则替换该项，落在输出目录上则设为输出目录 */
function onDrop(json) {
  const { paths, x, y } = JSON.parse(json), el = document.elementFromPoint(x, y);
  if (run.busy) { announce('处理中不能更换文件'); return; }
  const chip = el?.closest('.chip'), audio = paths.find(p => /\.(wav|mp3|flac|m4a|aac|ogg|aiff|wma)$/i.test(p));
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
  band.addColorStop(0, `rgba(181,47,104,${a})`); band.addColorStop(1, 'rgba(181,47,104,0)');
  g.fillStyle = band; g.fillRect(0, h * .6, w, h * .4);
  if (fxState.on && fxState.parts.length < 36 && Math.random() < .3)
    fxState.parts.push({ x: Math.random() * w, y: h + 6, s: 2 + Math.random() * 3, v: .25 + Math.random() * .5, life: 0, max: 4 + Math.random() * 4, hue: Math.random() < .5 });
  fxState.parts = fxState.parts.filter(p => (p.life += 1 / 60) < p.max);
  for (const p of fxState.parts) {
    p.y -= p.v; const alpha = Math.sin(Math.PI * p.life / p.max) * .28;
    g.fillStyle = p.hue ? `rgba(181,47,104,${alpha})` : `rgba(208,88,139,${alpha})`; g.fillRect(p.x, p.y, p.s, p.s);
  }
  // 淡出过渡结束后才停止绘制
  if (!fxState.on && (canvas._off ??= now) && now - canvas._off > 450) { fxState.raf = 0; fxState.parts = []; canvas._off = undefined; g.clearRect(0, 0, w, h); return; }
  if (fxState.on) canvas._off = undefined;
  fxState.raf = requestAnimationFrame(fxFrame);
}

/* 独立 EXE 验收读取界面状态 */
window.__svState = () => ({ strength: P.strength, mix_enabled: SW.mix_effects, rhythm_enabled: SW.rhythm,
  busy: run.busy, ready: state.ready, tab: $('#tab-report').getAttribute('aria-pressed') === 'true' ? 'report' : 'process' });

/* ═══ 启动 ═══ */
function init() {
  buildRacks(); buildPresets(); paint();
  setChip('song', ''); setChip('dry', ''); setOutput(store.get('output', ''));
  $$('.chip').forEach(chip => { chip.addEventListener('click', () => chooseInput(chip.dataset.kind)); bindMarquee(chip); });
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
  bindPlayer(); bindModals(); bindHelp(); bindSettings();
  renderPlayer(); renderReport(null, 'none');
  requestAnimationFrame(() => moveGlider($('.tabs'), $('#tab-process'), '.tab-glider'));
  addEventListener('resize', () => { buildRuler(); renderPlayer(); drawReport(reportView, null, 1, 1); moveGlider($('.tabs'), $('.tab[aria-pressed="true"]'), '.tab-glider'); });
}
init();
new QWebChannel(qt.webChannelTransport, channel => {
  bridge = channel.objects.bridge;
  bridge.stage.connect(onStage); bridge.finished.connect(onFinished); bridge.failed.connect(onFailed);
  bridge.playerState.connect(onPlayerState); bridge.waveform.connect(onWaveform); bridge.waveformSegment.connect(onWaveSegment);
  bridge.draftState.connect(onDraftState); bridge.versionLoaded.connect(onVersionLoaded);
  bridge.filesDropped.connect(onDrop); bridge.vstStatus.connect(onVstStatus);
});
