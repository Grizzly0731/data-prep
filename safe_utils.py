# -*- coding: utf-8 -*-
"""异常安全的数据读取与下载工具。

所有函数都不抛异常，而是返回 (结果, 错误说明)，方便调用方决定是否继续。

覆盖的异常：
    - 文件不存在 / 不是文件 / 大小为空
    - 编码错误（依次尝试 utf-8-sig / utf-8 / gbk / gb18030）
    - 文件无有效数据（EmptyDataError）
    - 格式解析失败（ParserError：分隔符或引号不匹配）
    - 列数与期望不一致
    - 下载失败（网络异常、HTTP 错误、超时、内容为空），并回退到本地副本
"""

import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

ENCODINGS = ("utf-8-sig", "utf-8", "gbk", "gb18030")


def safe_read_csv(path, header=0, nrows=None, encodings=ENCODINGS):
    """读取 CSV，失败时返回清晰的原因。

    返回 (df, None) 表示成功；(None, "原因") 表示失败。
    """
    path = Path(path)

    if not path.exists():
        return None, f"文件不存在：{path}"
    if not path.is_file():
        return None, f"不是普通文件：{path}"
    if path.stat().st_size == 0:
        return None, f"文件大小为 0：{path.name}"

    tried = []
    for enc in encodings:
        try:
            df = pd.read_csv(path, header=header, nrows=nrows, encoding=enc)
        except UnicodeDecodeError:
            tried.append(enc)
            continue
        except pd.errors.EmptyDataError:
            return None, f"文件无有效数据（空文件或全是空行）：{path.name}"
        except pd.errors.ParserError as exc:
            return None, f"格式解析失败（分隔符或引号不匹配）：{path.name} -> {str(exc)[:120]}"
        except OSError as exc:
            return None, f"读取失败（I/O 错误）：{path.name} -> {exc}"

        if df is None or df.shape[0] == 0 or df.shape[1] == 0:
            return None, f"数据为空（0 行或 0 列）：{path.name}"
        return df, None

    return None, f"编码错误：试过 {', '.join(tried)} 都无法解码 {path.name}"


def check_columns(df, expected, name=""):
    """检查列数与列名，返回 (是否通过, 说明)。"""
    if expected is None:
        return True, ""
    if df.shape[1] != len(expected):
        return False, (f"列数不一致：期望 {len(expected)} 列，实际 {df.shape[1]} 列"
                       f"（{name}）")
    actual = list(df.columns)
    if actual != list(expected):
        return False, f"列名不一致：期望 {list(expected)}，实际 {actual}（{name}）"
    return True, ""


def safe_download(url, dest, timeout=30, retries=3, fallbacks=(), user_agent="Mozilla/5.0"):
    """下载文件，失败时尝试本地副本。

    返回 (ok, 说明)。
    """
    dest = Path(dest)
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return False, f"无法创建目录 {dest.parent}：{exc}"

    if dest.is_file() and dest.stat().st_size > 0:
        return True, f"已存在，跳过下载：{dest.name}"

    errors = []
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": user_agent})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            if not data:
                raise ValueError("下载内容为空")
            dest.write_bytes(data)
            return True, f"下载成功：{dest.name}（{len(data)} 字节，第 {attempt} 次尝试）"
        except urllib.error.HTTPError as exc:
            errors.append(f"第 {attempt} 次：HTTP {exc.code} {exc.reason}")
        except urllib.error.URLError as exc:
            errors.append(f"第 {attempt} 次：网络错误 {exc.reason}")
        except (TimeoutError, ValueError, OSError) as exc:
            errors.append(f"第 {attempt} 次：{type(exc).__name__}: {exc}")

    for fb in fallbacks:
        fb = Path(fb)
        if fb.is_file() and fb.stat().st_size > 0:
            try:
                dest.write_bytes(fb.read_bytes())
                return True, f"下载失败，已改用本地副本：{fb}"
            except OSError as exc:
                errors.append(f"本地副本复制失败：{exc}")

    return False, "下载失败且找不到本地副本：\n    " + "\n    ".join(errors)
