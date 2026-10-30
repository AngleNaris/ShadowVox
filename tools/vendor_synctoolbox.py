"""把已安装的 Sync Toolbox（MIT）精简复制到 src/shadowvox/vendor/synctoolbox。

只保留整句对齐用到的特征与 MrMsDTW；算法代码不改，仅：
- 去掉仅供绘图的 matplotlib / libfmp 顶层导入，以及 librosa 备选 DTW 的导入（verbose 绘图与 librosa 实现不可用）；
- sklearn 欧氏距离换成等价的 scipy cdist；libfmp 的严格单调路径函数内联；
- numba 去掉 cache=True（单文件程序没有源码文件，无法写缓存）；绘图注解延迟求值。
用法：python tools/vendor_synctoolbox.py
"""
from pathlib import Path
import shutil
import synctoolbox

SOURCE = Path(synctoolbox.__file__).parent
TARGET = Path(__file__).resolve().parents[1] / "src/shadowvox/vendor/synctoolbox"
FILES = ["__init__.py", "dtw/__init__.py", "dtw/anchor.py", "dtw/core.py", "dtw/cost.py", "dtw/mrmsdtw.py",
         "dtw/utils.py", "feature/__init__.py", "feature/chroma.py", "feature/dlnco.py", "feature/filterbank.py",
         "feature/pitch.py", "feature/pitch_onset.py", "feature/utils.py"]
STRICT_PATH = '''

def compute_strict_alignment_path_mask(P):
    """libfmp.c3（MIT）原样内联：由规整路径得到严格单调路径。"""
    P = np.array(P, copy=True)
    N, M = P[-1]
    keep_mask = (P[1:, 0] > P[:-1, 0]) & (P[1:, 1] > P[:-1, 1])
    keep_mask = np.concatenate(([True], keep_mask))
    keep_mask[(P[:, 0] == N) | (P[:, 1] == M)] = False
    keep_mask[-1] = True
    return P[keep_mask, :]
'''
EDITS = {
    "dtw/core.py": [("import librosa\n", ""), ("@jit(nopython=True, cache=True)", "@jit(nopython=True)")],
    "dtw/cost.py": [("from sklearn.metrics.pairwise import euclidean_distances",
                     "from scipy.spatial.distance import cdist as euclidean_distances")],
    "dtw/mrmsdtw.py": [("from synctoolbox.dtw.visualization import sync_visualize_step1, sync_visualize_step2\n", "")],
    "dtw/utils.py": [("from libfmp.c3 import compute_strict_alignment_path_mask\n", "")],
    "feature/pitch.py": [("from libfmp.b import plot_matrix\n", ""), ("import matplotlib.pyplot as plt\n", "")],
    "feature/dlnco.py": [("import matplotlib.pyplot as plt\n", ""),
                         ("from libfmp.b import MultiplePlotsWithColorbar, plot_chromagram, plot_matrix\n", "")],
    "feature/pitch_onset.py": [("import matplotlib.pyplot as plt\n", "")],
    "feature/utils.py": [("from libfmp.c3 import compute_freq_distribution, tuning_similarity\n", "")],
}


def main():
    if TARGET.exists():
        shutil.rmtree(TARGET)
    for name in FILES:
        text = (SOURCE / name).read_text(encoding="utf-8")
        for old, new in EDITS.get(name, []):
            assert old in text, (name, old)
            text = text.replace(old, new)
        if name == "dtw/utils.py":
            text = text.replace("from typing import List\n", "from typing import List\n" + STRICT_PATH, 1)
        if any(old.startswith(("import matplotlib", "from libfmp")) for old, _ in EDITS.get(name, [])):
            # 绘图函数签名里的 plt.Axes 等注解不再在导入时求值。
            text = "from __future__ import annotations\n" + text
        text = text.replace("from synctoolbox.", "from shadowvox.vendor.synctoolbox.")
        assert "libfmp" not in text.split('"""')[0] and "import matplotlib" not in text, name
        path = TARGET / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    (TARGET.parent / "__init__.py").write_text("", encoding="utf-8")
    license_file = next(SOURCE.parent.glob("synctoolbox-*.dist-info/licenses/*"))
    shutil.copyfile(license_file, TARGET / "LICENSE")
    print("vendored", len(FILES), "files ->", TARGET)


if __name__ == "__main__":
    main()
