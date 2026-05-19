# python读取ini配置文件公用
# 作者：dmf
# 时间：2026/3/27
from .core import IniConfig
from .exceptions import (
    IniConfigError,
    SectionNotFoundError,
    KeyNotFoundError,
    FileNotFoundError,
)


def read(path, encoding="utf-8", override=None):
    """
    从指定文件读取 INI 配置，返回嵌套字典。

    参数:
        path: 配置文件路径
        encoding: 文件编码，默认 utf-8
        override: dict，键为 "section.key"，用于覆盖文件中对应值

    返回:
        dict，结构为 {section: {key: value}}
    """
    import configparser

    cp = configparser.ConfigParser(interpolation=None)
    cp.read(path, encoding=encoding)

    result = {}
    for section in cp.sections():
        result[section] = {}
        for key in cp.options(section):
            result[section][key] = cp.get(section, key)

    if override:
        for full_key, value in override.items():
            sec, _, key = full_key.partition(".")
            if sec in result and key in result[sec]:
                result[sec][key] = str(value)

    return result


__all__ = [
    "IniConfig",
    "IniConfigError",
    "SectionNotFoundError",
    "KeyNotFoundError",
    "FileNotFoundError",
    "read",
]
