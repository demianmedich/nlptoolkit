# coding=utf-8
import os
from typing import Iterable


def get_descendant_files(root_dir: str) -> Iterable[str]:
    for sub_root, sub_dirs, files in os.walk(root_dir):
        for file in files:
            yield os.path.join(sub_root, file)


def get_directory_size(root_dir: str) -> int:
    size = 0
    for file in get_descendant_files(root_dir):
        size += os.path.getsize(file)
    return size
