# python读取ini配置文件公用
# 作者：dmf
# 时间：2026/3/27
import builtins


class IniConfigError(Exception):
    """INI 配置相关的通用错误"""
    pass


class SectionNotFoundError(IniConfigError):
    """指定的节（section）不存在"""
    pass


class KeyNotFoundError(IniConfigError):
    """指定的键（key）不存在，且未提供默认值"""
    pass


class FileNotFoundError(IniConfigError, builtins.FileNotFoundError):
    """配置文件未找到"""
    pass
