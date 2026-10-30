"""仅本机的音频试听服务器；支持单区间 Range 请求以便长音频跳转。"""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re


class AudioHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        self.remaining = None
        path = Path(self.translate_path(self.path))
        requested = self.headers.get("Range", "")
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
        if not match or not path.is_file():
            return super().send_head()
        size = path.stat().st_size
        first, last = match.groups()
        if not first and not last:
            self.send_error(400, "Invalid range")
            return None
        start = int(first) if first else max(0, size-int(last))
        end = min(int(last), size-1) if first and last else size-1
        if start >= size or start > end:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        source = path.open("rb")
        source.seek(start)
        self.remaining = end-start+1
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(self.remaining))
        self.end_headers()
        return source

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def copyfile(self, source, outputfile):
        try:
            if self.remaining is None:
                return super().copyfile(source, outputfile)
            while self.remaining:
                chunk = source.read(min(65536, self.remaining))
                if not chunk:
                    break
                outputfile.write(chunk)
                self.remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # 浏览器切换音频时正常取消旧请求。


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--directory", type=Path,
                        default=Path(__file__).resolve().parents[1] / "out/naturalness")
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port),
                                partial(AudioHandler, directory=str(args.directory.resolve())))
    print(f"http://127.0.0.1:{args.port}/", flush=True)
    server.serve_forever()
