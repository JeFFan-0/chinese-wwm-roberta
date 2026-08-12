"""pytest 根配置：把仓库根目录加入 sys.path，使测试可 `import src.*`。"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
