"""Manual check of ShadowVox stable releases."""
import json
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from . import GITHUB_REPO, __version__

RELEASES = f'https://github.com/{GITHUB_REPO}/releases'


def check_update():
    result = {'current': __version__}
    request = Request(f'https://api.github.com/repos/{GITHUB_REPO}/releases/latest',
                      headers={'User-Agent': f'ShadowVox/{__version__}',
                               'Accept': 'application/vnd.github+json'})
    try:
        with urlopen(request, timeout=8) as response:
            release = json.load(response)
    except HTTPError as exc:
        if exc.code == 404:
            return dict(result, state='unavailable')
        if exc.code == 429 or (exc.code == 403 and exc.headers and
                               (exc.headers.get('X-RateLimit-Remaining') == '0' or exc.headers.get('Retry-After'))):
            return dict(result, state='limited')
        return dict(result, state='error')
    except (URLError, TimeoutError, OSError, ValueError):
        return dict(result, state='error')
    tag = release.get('tag_name', '') if isinstance(release, dict) else ''
    match = re.fullmatch(r'[vV]?(\d+)\.(\d+)\.(\d+)', str(tag))
    if not match or release.get('draft') or release.get('prerelease'):
        return dict(result, state='error')
    latest = tuple(map(int, match.groups()))
    current = tuple(map(int, __version__.split('.')))
    state = 'newer' if latest > current else 'current' if latest == current else 'ahead'
    return dict(result, state=state, latest='.'.join(match.groups()), url=f'{RELEASES}/tag/{tag}')
