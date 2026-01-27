# -*- coding: utf-8 -*-
"""
扫描目录下所有音频文件，计算总时长
用法:
    python audio_total_duration.py /path/to/your/folder
"""

import os
import sys
from pathlib import Path
from datetime import timedelta
import warnings

# 忽略 librosa 的未来警告（可选）
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import librosa
except ImportError:
    print("请先安装 librosa：")
    print("   pip install librosa")
    sys.exit(1)

# 支持的音频扩展名（可自行扩展）
AUDIO_EXTENSIONS = {
    '.wav', '.mp3', '.flac', '.ogg', '.m4a', '.opus', '.aac', '.wma',
    '.aiff', '.aif', '.aifc', '.webm', '.oga'
}


def format_duration(seconds: float) -> str:
    """将秒数转换为 小时:分钟:秒 的可读格式"""
    td = timedelta(seconds=int(seconds))
    total_seconds = td.total_seconds()
    hours = td.days * 24 + td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    secs = td.seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def get_audio_duration(file_path: Path) -> float:
    """获取单个音频文件的时长（秒）"""
    try:
        duration = librosa.get_duration(path=str(file_path))
        return duration
    except Exception as e:
        print(f"无法读取 {file_path} : {e}")
        return 0.0


def calculate_total_duration(directory: str):
    root_path = Path(directory).resolve()
    if not root_path.is_dir():
        print(f"错误：{root_path} 不是一个有效的目录")
        return

    print(f"正在扫描目录：{root_path}")
    print("支持的格式：", ", ".join(sorted(AUDIO_EXTENSIONS)))

    total_sec = 0.0
    file_count = 0

    # 递归遍历所有文件
    for file_path in root_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in AUDIO_EXTENSIONS:
            duration = get_audio_duration(file_path)
            total_sec += duration
            file_count += 1
            print(f"  {file_count:4d} | {duration:8.2f}s | {file_path.name}")

    # 汇总
    print("\n" + "="*60)
    print(f"总文件数：{file_count} 个")
    print(f"总时长：{total_sec:.2f} 秒")
    print(f"格式化：{format_duration(total_sec)}")
    if file_count > 0:
        print(f"平均每文件：{total_sec / file_count:.2f} 秒")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法：")
        print(f"  python {sys.argv[0]} /path/to/audio/folder")
        sys.exit(1)

    folder = sys.argv[1]
    calculate_total_duration(folder)