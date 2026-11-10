/* 应用、离线 HTML 和 PNG 共用同一 SVG；分析数据不参与音频处理。 */
(() => {
  'use strict';
  const C = { bg: '#100b13', text: '#fff6fb', before: '#a99fad', muted: '#b8a8bb', rule: '#55445a',
    line: '#392b3c', grid: '#2a1f2e', pink: '#ff4c9f', pale: '#f3b4d2' };
  const themeColors = new Map(Object.entries(C).map(([key, hex]) => [hex, 'var(--rp-' + key + ', ' + hex + ')']));
  // 节号悬挂在页边，标题、说明和绘图区共用内容线；轴标签在内容线左侧的沟槽里右对齐。
  const G = { left: 48, right: 1032, content: 124, gutter: 112, columnEnd: 524, columnStart: 556,
    indent: 76, sections: [204, 604, 940], headingInset: 24 };
  const F = { metric: 24, title: 32, value: 16, label: 13, tick: 13 };
  const SYNC_MS = 30;
  const esc = x => String(x ?? '').replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&apos;' }[ch]));
  const num = (x, digits = 0) => x == null || !Number.isFinite(x) ? '—' : x.toFixed(digits);
  const signed = (x, digits = 0) => x == null ? '—' : `${x > 0 ? '+' : ''}${num(x, digits)}`;
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const clock = t => `${String(Math.floor(t / 60)).padStart(2, '0')}:${String(Math.floor(t % 60)).padStart(2, '0')}`;
  // 雅黑中文字宽约等于字号，西文约一半；仅用于右对齐图例的排布。
  const textWidth = (value, size) => Array.from(String(value)).reduce((n, ch) => n + (ch.charCodeAt(0) > 255 ? 1 : .56), 0) * size;
  const text = (x, y, value, size = F.label, fill = C.text, extra = '') =>
    `<text x="${x}" y="${y}" font-size="${size}" fill="${fill}" ${extra}>${esc(value)}</text>`;
  const line = (x1, y1, x2, y2, color = C.line, width = 1, extra = '') =>
    `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${width}" ${extra}/>`;
  const rect = (x, y, w, h, fill, extra = '') => `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${fill}" ${extra}/>`;
  const dot = (x, y, r, color, extra = '') => `<circle cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="${r}" fill="${color}" ${extra}/>`;
  const triangle = (x, y, color) => `<path d="M${x.toFixed(2)} ${y}l6 9h-12z" fill="${color}"/>`;
  function path(values, xOf, yOf) {
    let pen = false, d = '';
    values.forEach((v, i) => {
      if (v == null || !Number.isFinite(v)) { pen = false; return; }
      d += `${pen ? 'L' : 'M'}${xOf(i).toFixed(2)},${yOf(v).toFixed(2)} `; pen = true;
    });
    return d;
  }
  const stroke = (d, color, width = 2, extra = '') => `<path d="${d}" fill="none" stroke="${color}" stroke-width="${width}" stroke-linejoin="round" ${extra}/>`;
  const headingY = section => G.sections[section] + G.headingInset + 36;
  function heading(n, title, english, section, x = G.left) {
    const y = headingY(section);
    return `<g class="report-heading" data-section="${n}">` +
      text(x, y - 12, n, 44, C.pink, 'font-weight="700" dominant-baseline="central"') +
      text(x + G.indent, y - 12, title, F.title, C.text, 'font-weight="700" dominant-baseline="central"') +
      text(x + G.indent, y + 30, english, F.tick, C.muted, 'letter-spacing="2"') + '</g>';
  }
  // 图例自右向左排布，末项贴齐给定的右边界。
  function legend(right, y, items) {
    let s = '', x = right;
    for (const [kind, color, label] of [...items].reverse()) {
      x -= textWidth(label, F.tick);
      s += text(x, y, label, F.tick, color);
      x -= 32;
      if (kind === 'zone') s += rect(x, y - 15, 24, 18, C.pink, 'opacity=".16"');
      else if (kind === 'bar') s += rect(x + 6, y - 15, 12, 18, color);
      else if (kind === 'tick') s += line(x + 12, y - 16, x + 12, y + 2, color, 3);
      else if (kind === 'target') s += triangle(x + 12, y - 12, color);
      else s += line(x, y - 6, x + 24, y - 6, color, kind === 'dash' ? 2 : 3, kind === 'dash' ? 'stroke-dasharray="6 4"' : '');
      x -= 24;
    }
    return s;
  }
  function change(before, after, unit, y, digits = 0) {
    return `<text x="${G.right}" y="${y}" font-size="${F.metric}" font-weight="700" text-anchor="end"><tspan fill="${C.before}">${num(before, digits)}</tspan><tspan fill="${C.muted}" font-size="15"> → </tspan><tspan fill="${C.pink}">${num(after, digits)}</tspan></text>` +
      text(G.right, y + 24, unit, F.tick, C.muted, 'text-anchor="end"');
  }
  const fill = (value, max) => clamp((value ?? 0) / max, 0, 1);
  // 同一行内：标签、自 0 起的条、数值。
  function inlineBar(x, right, y, label, value, shown, max, color) {
    const start = x + 96, end = right - 150;
    return text(x, y, label, F.label, color) + rect(start, y - 11, end - start, 8, C.line) +
      rect(start, y - 11, (end - start) * fill(value, max), 8, color) +
      text(right, y, shown, F.label, color, 'text-anchor="end"');
  }
  // 窄栏：标签与数值一行，条在下方。
  function stackBar(x, right, y, label, value, shown, max, color, extra = '') {
    return `<g ${extra}>` + text(x, y, label, F.label, color) + text(right, y, shown, F.value, color, 'text-anchor="end"') +
      rect(x, y + 12, right - x, 8, C.line) + rect(x, y + 12, (right - x) * fill(value, max), 8, color) + '</g>';
  }
  const empty = (x, y, message) => text(x, y, message, F.value, C.muted, 'text-anchor="middle"');
  function pitch(p, enabled) {
    let s = heading('01', '音准修正', enabled === false ? 'PITCH / 修音未开启' : 'PITCH / BEFORE → AFTER', 0);
    if (!p?.available) return s + empty(540, 419, p?.reason || '生成后显示实际音高修正结果');
    s += change(p.before_median, p.after_median, '音分 · 偏差中位数', headingY(0), 1);
    const w = p.window, left = G.content, right = G.right, top = 356, bottom = 456;
    const vals = [...w.before, ...w.after, ...w.reference].filter(v => v != null);
    const lo = Math.floor(Math.min(...vals) - .5), hi = Math.max(lo + 3, Math.ceil(Math.max(...vals) + .5));
    const x = i => left + (right - left) * (w.times[i] - w.start) / (w.end - w.start);
    const y = v => bottom - (v - lo) / (hi - lo) * (bottom - top);
    const names = ['C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B'];
    s += text(left, 336, `旋律示例 ${clock(w.start)}–${clock(w.end)}`, F.tick, C.muted) +
      legend(right, 336, [['dash', C.text, '原唱'], ['line', C.before, '处理前'], ['line', C.pink, '成品']]);
    const noteStep = Math.max(1, Math.ceil((hi - lo) / 4));
    s += line(left, top, right, top, C.grid);
    for (let note = lo; note <= hi; note += noteStep) {
      s += line(left, y(note), right, y(note), C.grid) +
        text(G.gutter, y(note) + 6, `${names[((note % 12) + 12) % 12]}${Math.floor(note / 12) - 1}`, F.tick, C.muted, 'text-anchor="end"');
    }
    for (let i = 0; i <= 6; i++) {
      const at = left + (right - left) * i / 6;
      s += line(at, top, at, bottom, C.grid) + text(at, 486, clock(w.start + (w.end - w.start) * i / 6), F.tick, C.muted,
        `text-anchor="${i === 0 ? 'start' : i === 6 ? 'end' : 'middle'}"`);
    }
    s += `<g class="report-trace">${stroke(path(w.reference, x, y), C.text, 2, 'stroke-dasharray="7 5"')}
      ${stroke(path(w.before, x, y), C.before, 2.5)}${stroke(path(w.after, x, y), C.pink, 3.5, 'filter="url(#rp-glow)"')}</g>`;
    const limit = Math.max(50, Math.ceil(Math.max(p.before_p90, p.after_p90) / 50) * 50);
    s += text(left, 524, '全曲 P90 偏差', F.tick, C.muted) +
      inlineBar(left, right, 556, '处理前', p.before_p90, `${num(p.before_p90, 1)} 音分`, limit, C.before) +
      inlineBar(left, right, 584, '成品', p.after_p90, `${num(p.after_p90, 1)} 音分`, limit, C.pink);
    s += `<rect x="${left}" y="${top}" width="${right - left}" height="${bottom - top}" fill="transparent" data-seek="${w.start}"><title>定位试听此片段，播放器可切换处理前与成品</title></rect>`;
    return s;
  }
  // 全部共同入点按 10 ms 分箱：处理前向上、成品向下，同一纵向刻度；±30 ms 为同步区。
  function timing(p, enabled) {
    let s = heading('02', '节奏对齐', enabled === false ? 'TIMING / 仅补偿整体偏移' : 'TIMING / ONSET OFFSET', 1);
    if (!p?.available) return s + empty(540, 815, p?.reason || '生成后显示共同入点的实际偏差');
    s += change(p.before_median_ms, p.after_median_ms, 'ms · 入点偏差中位数', headingY(1));
    const pairs = p.pairs, total = Math.max(1, pairs.length);
    const bound = Math.max(100, Math.ceil(Math.max(...pairs.flatMap(v => [Math.abs(v.before), Math.abs(v.after)])) / 50) * 50);
    const bins = bound / 5 + 1, left = G.content, right = 760, bw = (right - left) / bins;
    const hist = key => { const a = new Array(bins).fill(0); for (const v of pairs) a[Math.round(clamp(v[key], -bound, bound) / 10) + bound / 10]++; return a; };
    const before = hist('before'), after = hist('after'), peak = Math.max(1, ...before, ...after);
    const edge = v => left + (v + bound) / 10 * bw, zero = edge(0) + bw / 2, upBase = 804, downBase = 832, span = 68;
    s += legend(right, 700, [['zone', C.pink, '同步区 ±30 ms']]);
    s += rect(edge(-SYNC_MS), 724, edge(SYNC_MS + 10) - edge(-SYNC_MS), 184, C.pink, 'opacity=".1" data-sync-zone="true"');
    s += text(G.gutter, 770, '处理前', F.tick, C.before, 'text-anchor="end"') + text(G.gutter, 878, '成品', F.tick, C.pink, 'text-anchor="end"');
    s += line(left, upBase, right, upBase, C.rule) + line(left, downBase, right, downBase, C.rule) + line(zero, 724, zero, 908, C.pale, 1.5, 'stroke-dasharray="4 5" data-onset-zero="true"');
    s += '<g class="report-trace">';
    for (let i = 0; i < bins; i++) {
      const v = (i - bound / 10) * 10, inside = Math.abs(v) <= SYNC_MS, bx = left + i * bw + 2, w = bw - 4;
      if (before[i]) s += rect(bx.toFixed(2), (upBase - span * before[i] / peak).toFixed(2), w.toFixed(2), (span * before[i] / peak).toFixed(2), C.before, `opacity="${inside ? 1 : .5}" data-hist="before" data-bin="${v}" data-count="${before[i]}"`);
      if (after[i]) s += rect(bx.toFixed(2), downBase, w.toFixed(2), (span * after[i] / peak).toFixed(2), C.pink, `opacity="${inside ? 1 : .55}" data-hist="after" data-bin="${v}" data-count="${after[i]}"`);
    }
    s += '</g>';
    s += text(left, 818, `提前 ${bound}`, F.tick, C.muted) + text(right, 818, `延后 ${bound}`, F.tick, C.muted, 'text-anchor="end"') +
      text(zero, 818, '原唱 0', F.tick, C.pale, `text-anchor="middle" stroke="${C.bg}" stroke-width="6" paint-order="stroke"`) +
      text(edge(-SYNC_MS), 818, `-${SYNC_MS}`, F.tick, C.muted, 'text-anchor="middle"') + text(edge(SYNC_MS) + bw, 818, `+${SYNC_MS}`, F.tick, C.muted, 'text-anchor="middle"');
    const share = key => pairs.filter(v => Math.abs(v[key]) <= SYNC_MS).length / total * 100;
    s += line(780, 724, 780, 912, C.line) + text(804, 744, `±${SYNC_MS} ms 内占比`, F.label, C.text) +
      stackBar(804, G.right, 788, '处理前', share('before'), `${num(share('before'))}%`, 100, C.before, 'data-sync-share="before"') +
      stackBar(804, G.right, 848, '成品', share('after'), `${num(share('after'))}%`, 100, C.pink, 'data-sync-share="after"') +
      text(804, 912, 'P90 偏差', F.tick, C.muted) +
      `<text x="${G.right}" y="912" font-size="${F.label}" text-anchor="end"><tspan fill="${C.before}">${num(p.before_p90_ms)}</tspan><tspan fill="${C.muted}"> → </tspan><tspan fill="${C.pink}">${num(p.after_p90_ms)} ms</tspan></text>`;
    return s;
  }
  function tone(p) {
    let s = heading('03', '音色塑形', 'VOCAL / SPECTRUM', 2);
    if (!p?.available) return s + empty(286, 1190, '此版本缺少独立人声分析');
    const a = p.before, b = p.after, left = G.content, right = G.columnEnd, top = 1096, bottom = 1170;
    const x = f => left + (right - left) * Math.log(f / 40) / Math.log(16000 / 40);
    const y = v => bottom - clamp((v + 70) / 70, 0, 1) * (bottom - top);
    s += '<g class="report-tone"><title>原始干声与空间效果前的汇总人声按活动段 RMS 电平归一；频段能量增减不等同于主观音质评分。</title>' +
      text(left, 1078, '等电平频谱 · dB', F.tick, C.muted) +
      legend(right, 1078, [['line', C.before, '处理前'], ['line', C.pink, '成品']]);
    p.changes.forEach((v, i) => {
      s += rect(x(v.lo), top, x(v.hi) - x(v.lo), bottom - top, C.pink, 'opacity=".1"');
    });
    for (const v of [-60, -30, 0]) s += line(left, y(v), right, y(v), C.grid) + text(G.gutter, y(v) + 6, v, F.tick, C.muted, 'text-anchor="end"');
    for (const hz of [100, 1000, 10000]) s += line(x(hz), top, x(hz), bottom, C.grid) +
      text(x(hz), 1194, hz >= 1000 ? `${hz / 1000} kHz` : `${hz} Hz`, F.tick, C.muted, 'text-anchor="middle"');
    s += `<g class="report-trace">${stroke(path(a.spectrum, i => x(a.centers[i]), y), C.before, 2.5)}${stroke(path(b.spectrum, i => x(b.centers[i]), y), C.pink, 3.5)}</g></g>`;
    p.changes.forEach((v, i) => {
      const xx = G.left + i * 166, range = v.lo >= 1000 ? `${v.lo / 1000}–${v.hi / 1000} kHz` : `${v.lo}–${v.hi} Hz`;
      s += `<g class="report-band"><title>${esc(range)} 相对能量变化</title>` +
        text(xx, 1228, ['低频', '中频', '高频'][i], F.label) + text(xx, 1254, range, F.tick, C.muted) +
        text(xx, 1286, `${signed(v.db, 1)} dB`, F.value) + '</g>';
    });
    const maximum = Math.max(40, Math.ceil(Math.max(a.dynamic_range_db, b.dynamic_range_db) / 10) * 10);
    s += text(G.left, 1322, '人声能量起伏', F.label, C.text) +
      stackBar(G.left, 272, 1358, '处理前', a.dynamic_range_db, `${num(a.dynamic_range_db, 1)} dB`, maximum, C.before) +
      stackBar(302, G.columnEnd, 1358, '成品', b.dynamic_range_db, `${num(b.dynamic_range_db, 1)} dB`, maximum, C.pink);
    return s;
  }
  // 响度条：处理前刻度在条上方，目标三角在条下方，二者不会重叠。
  function meter(label, value, unit, lo, hi, target, y, before) {
    const x = 632, width = G.right - x, pos = v => x + clamp((v - lo) / (hi - lo), 0, 1) * width;
    let s = text(x, y, label, F.label, C.muted) +
      `<text x="${G.right}" y="${y}" font-size="${F.value}" text-anchor="end">` +
      (before != null ? `<tspan fill="${C.before}" font-size="${F.label}">${num(before, 1)}</tspan><tspan fill="${C.muted}" font-size="${F.label}"> → </tspan>` : '') +
      `<tspan fill="${C.text}">${num(value, 1)} ${unit}</tspan></text>` + rect(x, y + 14, width, 8, C.line);
    if (value != null) s += rect(x, y + 14, pos(value) - x, 8, C.pink);
    if (before != null) s += line(pos(before), y + 9, pos(before), y + 27, C.before, 3);
    if (target != null) s += triangle(pos(target), y + 29, C.text);
    return s;
  }
  function mix(p, before, output, targets, enabled) {
    let s = heading('04', '混音空间', enabled === false ? '混音未开启 · 仅测量输出' : 'MIX / STEREO FIELD', 2, G.columnStart);
    if (p?.before?.available && p?.after?.available) {
      const points = [...p.before.points, ...p.after.points], max = Math.max(1, ...points.flatMap(v => v.map(Math.abs)));
      for (const [key, color, cx, label] of [['before', C.before, 712, '处理前'], ['after', C.pink, 932, '成品']]) {
        const cy = 1160, radius = 72;
        s += `<g data-report-scope="${key}">` + text(cx, 1078, label, F.label, color, 'text-anchor="middle"');
        for (const r of [radius / 2, radius]) s += `<circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="${C.line}"/>`;
        s += line(cx - radius, cy, cx + radius, cy) + line(cx, cy - radius, cx, cy + radius) +
          line(cx - radius * .7, cy - radius * .7, cx + radius * .7, cy + radius * .7, C.line, 1, 'stroke-dasharray="4 6"') +
          line(cx + radius * .7, cy - radius * .7, cx - radius * .7, cy + radius * .7, C.line, 1, 'stroke-dasharray="4 6"') +
          text(cx - radius - 8, cy - radius * .6, 'L', F.tick, C.muted, 'text-anchor="end"') + text(cx + radius + 8, cy - radius * .6, 'R', F.tick, C.muted);
        s += '<g class="report-trace" opacity=".65">';
        for (const [side, mid] of p[key].points) {
          const scale = Math.max(max, Math.hypot(side, mid));
          s += dot(cx - side / scale * radius, cy - mid / scale * radius, 1.5, color);
        }
        s += '</g></g>';
      }
    } else s += empty(832, 1170, '等待成品测量');
    return s + legend(G.right, 1254, [['tick', C.before, '处理前'], ['target', C.text, '目标']]) +
      meter('综合响度', output?.lufs, 'LUFS', -30, -6, targets?.lufs, 1290, before?.lufs) +
      meter('真峰值', output?.true_peak_dbtp, 'dBTP', -12, 0, targets?.peak, 1352, before?.true_peak_dbtp);
  }
  function svg(data, themed = false) {
    const v = data?.visual || {}, output = data?.output, before = data?.before;
    const rawTitle = v.title || (data ? data.name : '你的声音'), glyphs = Array.from(rawTitle || '声音改造');
    const title = glyphs.slice(0, 17).join('') + (glyphs.length > 17 ? '…' : '');
    const titleSize = Math.min(96, 960 / Math.max(1, Array.from(title).length));
    let s = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 1440" width="1080" height="1440" role="img" aria-label="${esc(rawTitle)}的音准、节奏、音色和混音结果报告" class="report-svg">
      <title>${esc(rawTitle)} · SHADOWVOX 声音改造战报</title>
      <style>.report-svg{forced-color-adjust:none;font-variant-numeric:tabular-nums}</style>
      <defs><filter id="rp-glow" x="-10%" y="-25%" width="120%" height="150%"><feGaussianBlur stdDeviation="1.5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>
      <g font-family="Microsoft YaHei, Noto Sans CJK SC, Arial, sans-serif">${rect(0, 0, 1080, 1440, C.bg)}`;
    s += text(G.left, 48, 'SHADOWVOX', 16, C.text, 'font-weight="700" letter-spacing="2"') +
      text(G.right, 48, '声音改造战报', 13, C.text, 'text-anchor="end" letter-spacing="4"') +
      text(G.left, 145, title, titleSize, C.text, 'font-family="Noto Serif CJK SC,Source Han Serif SC,SimSun,serif" font-weight="700"') +
      text(G.left, 187, 'VOICE, RECRAFTED.', 13, C.muted, 'letter-spacing="4"');
    if (data?.version != null) s += text(G.right, 187, `V${data.version}`, 13, C.pale, 'text-anchor="end"');
    for (const yy of G.sections) s += line(G.left, yy, G.right, yy, C.rule);
    s += pitch(v.pitch, v.processing?.pitch) + timing(v.timing, v.processing?.rhythm) +
      line(540, 964, 540, 1380, C.line) + tone(v.tone) + mix(v.mix, before, output, v.targets, v.processing?.mix);
    s += line(G.left, 1396, G.right, 1396, C.rule) +
      text(G.left, 1428, output ? `${num(output.seconds, 1)} s / ${num(output.sample_rate / 1000, 1)} kHz` : '生成成品后显示真实数据', F.tick, C.muted) +
      text(G.right, 1428, '实际音频分析 · SHADOWVOX', F.tick, C.muted, 'text-anchor="end"');
    s += '</g></svg>';
    return themed ? s.replace(/(fill|stroke)="(#[0-9a-f]{6})"/gi, (match, attr, hex) => {
      const color = themeColors.get(hex);
      return color ? attr + '="' + color + '"' : match;
    }) : s;
  }
  async function png(data) {
    const image = new Image();
    image.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg(data));
    await image.decode();
    const canvas = document.createElement('canvas'); canvas.width = 1080; canvas.height = 1440;
    canvas.getContext('2d').drawImage(image, 0, 0, 1080, 1440);
    return canvas.toDataURL('image/png');
  }
  window.ShadowReport = { svg, png };
})();
