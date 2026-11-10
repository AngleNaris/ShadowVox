/* Apply persisted appearance before the first paint. */
(() => {
  const palettes = {
    rose: ['玫红', '#b52f68'], ember: ['酒红', '#a83b58'], violet: ['暗紫', '#6d55b8'],
    indigo: ['靛蓝', '#5273cf'], moss: ['墨绿', '#47856a'], amber: ['琥珀', '#b07a2e'],
  };
  const root = document.documentElement;
  const read = (key, fallback) => {
    try { return localStorage.getItem('sv_theme_' + key) ?? fallback; } catch { return fallback; }
  };
  const write = (key, value) => {
    try { localStorage.setItem('sv_theme_' + key, value); } catch { /* session only */ }
  };
  const validColor = color => /^#[0-9a-f]{6}$/i.test(color);
  function hue(color) {
    const [r, g, b] = color.slice(1).match(/../g).map(part => parseInt(part, 16));
    const top = Math.max(r, g, b), span = top - Math.min(r, g, b);
    if (!span) return null;
    return 60 * (top === r ? (g - b) / span : top === g ? 2 + (b - r) / span : 4 + (r - g) / span);
  }
  const logoHue = hue(palettes.rose[1]);
  let mode = read('mode', 'light') === 'dark' ? 'dark' : 'light';
  let palette = read('palette', 'rose'), custom = read('custom', '#8a5ea8');
  if (!validColor(custom)) custom = '#8a5ea8';
  if (palette !== 'custom' && !Object.hasOwn(palettes, palette)) palette = 'rose';
  function paint() {
    root.dataset.scheme = mode;
    root.dataset.palette = palette;
    const color = palette === 'custom' ? custom : palettes[palette][1], targetHue = hue(color);
    root.style.setProperty('--theme-seed', color);
    root.style.setProperty('--logo-tone', targetHue === null ? 'grayscale(1)' : 'hue-rotate(' + (targetHue - logoHue) + 'deg)');
  }
  paint();
  window.ShadowTheme = {
    palettes,
    get mode() { return mode; },
    get palette() { return palette; },
    get custom() { return custom; },
    apply(nextMode, nextPalette, nextColor = custom) {
      if (nextMode !== 'light' && nextMode !== 'dark') return;
      if (nextPalette !== 'custom' && !Object.hasOwn(palettes, nextPalette)) return;
      if (!validColor(nextColor)) return;
      mode = nextMode; palette = nextPalette; custom = nextColor;
      write('mode', mode); write('palette', palette); write('custom', custom);
      paint();
      document.dispatchEvent(new Event('shadowvox-theme'));
    },
  };
})();
