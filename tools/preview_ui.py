"""Preview the desktop UI at http://127.0.0.1:8765 with automatic reload."""

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

UI = Path(__file__).resolve().parents[1] / 'src' / 'shadowvox' / 'ui'
QT_SCRIPT = '<script src="qrc:///qtwebchannel/qwebchannel.js"></script>'
PREVIEW = '''
<style>
  .studio { max-width: 900px; margin-inline: auto; }
  .preview-label { color: var(--muted); font-size: 12px; }
</style>
<script>
document.addEventListener('DOMContentLoaded', () => {
  const content = document.querySelector('#content');
  const badge = document.createElement('span');
  badge.className = 'preview-label';
  badge.textContent = '界面预览 · 保存后自动刷新';
  document.querySelector('.tabs').before(badge);
  document.querySelector('#p-src').textContent = '界面预览';
  for (const button of document.querySelectorAll('#btn-settings, #in-song, #in-dry, #out-card, #go')) {
    button.disabled = true;
    button.title = '音频处理和文件选择请使用桌面版';
  }
  document.querySelector('#p-volume').inert = true;
  const scrollKey = 'sv_preview_scroll';
  const previousScroll = sessionStorage.getItem(scrollKey);
  if (previousScroll === null) document.querySelector('#racks').scrollIntoView({ block: 'end' });
  else content.scrollTop = Number(previousScroll);
  async function checkRevision() {
    try {
      const response = await fetch('/__revision', { cache: 'no-store' });
      if (response.ok && (await response.text()) !== '__REVISION__' &&
          !document.querySelector('.xy-field.is-dragging, .mix-range:active')) {
        sessionStorage.setItem(scrollKey, String(content.scrollTop));
        location.reload();
        return;
      }
    } catch (error) { /* Keep the current preview while the server is restarting. */ }
    setTimeout(checkRevision, 800);
  }
  setTimeout(checkRevision, 800);
});
</script>'''


def revision():
    return ':'.join(str(path.stat().st_mtime_ns) for path in sorted(UI.iterdir()) if path.is_file())


class PreviewHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(UI), **kwargs)

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            page = (UI / 'index.html').read_text(encoding='utf-8')
            self.write_page(page.replace(QT_SCRIPT, PREVIEW.replace('__REVISION__', revision())), 'text/html')
        elif self.path == '/__revision':
            self.write_page(revision(), 'text/plain')
        else:
            super().do_GET()

    def write_page(self, text, media_type):
        body = text.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', f'{media_type}; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_request(self, code='-', size='-'):
        if self.path != '/__revision':
            super().log_request(code, size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    with ThreadingHTTPServer(('127.0.0.1', args.port), PreviewHandler) as server:
        print(f'ShadowVox UI preview: http://127.0.0.1:{server.server_port}/', flush=True)
        server.serve_forever()
