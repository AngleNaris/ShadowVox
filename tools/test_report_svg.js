/* 关键声相方位与共享 SVG 安全性；节奏直方图的计数、方向与同步区占比。 */
'use strict';
const assert = require('node:assert/strict');
global.window = {};
require('../src/shadowvox/ui/report.js');
const model = {
  visual: {title: 'A<script>测试</script>', mix: {
    before: {available:true,points:[[1,1]]},after:{available:true,points:[[1,1]]}
  }},
  before: {lufs:-18,true_peak_dbtp:-3},output:{lufs:-16,true_peak_dbtp:-2,seconds:2,sample_rate:44100}
};
const svg = window.ShadowReport.svg(model);
const scope = svg.match(/data-report-scope="before">[\s\S]*?<circle cx="([\d.]+)" cy="([\d.]+)"/);
const point = svg.match(/data-report-scope="before">[\s\S]*?<g class="report-trace"[^>]*><circle cx="([\d.]+)" cy="([\d.]+)"/);
assert(Number(point[1]) < Number(scope[1]) && Number(point[2]) < Number(scope[2]),
  'Left-only signal must appear at upper left, not right.');
assert(svg.includes('&lt;script&gt;'));
assert(!svg.includes('<script>'));
assert(!svg.includes('NaN') && !svg.includes('Infinity') && !svg.includes('undefined'));
const empty = window.ShadowReport.svg(null);
assert(!empty.includes('class="report-trace"'), 'Unavailable results must not fabricate chart traces.');

const pairs = [[-80,-20],[-60,-60],[-40,-10],[-30,0],[10,20],[20,0],[40,10],[60,80]]
  .map(([before,after],i)=>({time:i+1,before,after}));
const timing = window.ShadowReport.svg({visual:{art:'data:image/png;base64,retired', timing:{
  available:true, before_median_ms:40, after_median_ms:30, before_p90_ms:70, after_p90_ms:60, pairs
}}});
assert(!timing.includes('<image') && !timing.includes('data:image/png'), 'Retired art must never be rendered, including legacy payloads.');
assert(!timing.includes('report-onset'), 'Per-onset rows are replaced by the histogram.');
const bars = [...timing.matchAll(/<rect x="([\d.]+)" y="([\d.]+)" width="[\d.]+" height="[\d.]+"[^>]*data-hist="(before|after)" data-bin="(-?\d+)" data-count="(\d+)"/g)]
  .map(m=>({y:Number(m[2]),key:m[3],bin:Number(m[4]),count:Number(m[5])}));
for (const key of ['before','after']) {
  const sum = bars.filter(b=>b.key===key).reduce((n,b)=>n+b.count,0);
  assert.equal(sum, pairs.length, `${key} histogram must contain every common onset.`);
}
const zero = Number(timing.match(/<line[^>]*x1="([\d.]+)"[^>]*data-onset-zero="true"/)[1]);
const axisY = Number(timing.match(/<line x1="[\d.]+" y1="(\d+)"[^>]*\/><line x1="[\d.]+" y1="832"/)[1]);
assert(bars.filter(b=>b.key==='before').every(b=>b.y < axisY), 'Before bars rise above the axis.');
assert(bars.filter(b=>b.key==='after').every(b=>b.y >= axisY), 'After bars hang below the axis.');
assert(zero > 124 && zero < 760, 'Reference zero line sits inside the plot.');
const share = key => Number(timing.match(new RegExp(`data-sync-share="${key}">[\\s\\S]*?font-size="16"[^>]*>(\\d+)%`))[1]);
assert.equal(share('before'), 38, 'Before sync share: 3 of 8 within ±30 ms (rounded).');
assert.equal(share('after'), 75, 'After sync share: 6 of 8 within ±30 ms.');
assert(timing.includes('data-sync-zone="true"'));
console.log('Shared SVG: channel orientation, escaping, unavailable data, art removal, histogram counts, direction and sync share passed.');
