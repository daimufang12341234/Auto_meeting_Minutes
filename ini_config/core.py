# python读取ini配置文件公用
# 作者：dmf
# 时间：2026/3/27
import configparser
import os
import stat

from .env_interpolator import interpolate
from .exceptions import (
    IniConfigError,
    FileNotFoundError as IniFileNotFoundError,
)
from .section_proxy import SectionProxy


class IniConfig(object):
    """
    INI 配置文件读取器。

    用法示例::

        config = IniConfig("config.ini")
        db_num = config.get_int("DB_CONN_NUM", "db_connect_num", default=1)
        path = config.get("PROGRAM_ROOT_PATH", "program_root_path", default="..")

        # 链式访问
        db_num = config.DB_CONN_NUM.db_connect_num
    """

    def __init__(self, path=None, encoding="utf-8", auto_reload=False,
                 base_dir=None, override=None):
        """
        参数:
            path: 配置文件路径，支持单个文件或文件列表
            encoding: 文件编码，默认 utf-8
            auto_reload: 是否开启热重载（文件修改后自动重新读取）
            base_dir: 配置文件所在目录，不指定则使用 os.getcwd()
            override: dict，键为 "section.key"，用于覆盖文件中对应值
        """
        if base_dir:
            self._base_dir = os.path.abspath(base_dir)
        else:
            self._base_dir = os.getcwd()

        self._encoding = encoding
        self._auto_reload = auto_reload
        self._override = override or {}
        self._mtime = None

        self._cp = configparser.ConfigParser(interpolation=None)

        if path is not None:
            if isinstance(path, (list, tuple)):
                self._paths = [self._resolve_path(p) for p in path]
            else:
                self._paths = [self._resolve_path(path)]
            self._path = self._paths[0]
            self._load()
        else:
            self._paths = []
            self._path = None

    def _resolve_path(self, path):
        if not os.path.isabs(path):
            return os.path.join(self._base_dir, path)
        return path

    def _load(self):
        for p in self._paths:
            if not os.path.isfile(p):
                raise IniFileNotFoundError("INI file not found: {}".format(p))

        self._cp.read(self._paths, encoding=self._encoding)

        self._apply_override()
        self._interpolate_all()
        self._update_mtime()

    def _apply_override(self):
        for full_key, value in self._override.items():
            sec, _, key = full_key.partition(".")
            if not sec or not key:
                raise IniConfigError("override key must be 'section.key', got: {}".format(full_key))
            if self._cp.has_section(sec):
                self._cp.set(sec, key, str(value))

    def _interpolate_all(self):
        for section in self._cp.sections():
            for key in self._cp.options(section):
                raw = self._cp.get(section, key)
                self._cp.set(section, key, interpolate(raw))

    def _update_mtime(self):
        if self._path and os.path.isfile(self._path):
            self._mtime = os.stat(self._path)[stat.ST_MTIME]
        else:
            self._mtime = None

    def _check_reload(self):
        if not self._auto_reload or not self._path:
            return
        if not os.path.isfile(self._path):
            return
        mtime = os.stat(self._path)[stat.ST_MTIME]
        if mtime != self._mtime:
            self.reload()

    def reload(self):
        """手动重新加载配置文件"""
        self._cp = configparser.ConfigParser(interpolation=None)
        self._load()

    # ------------------------------------------------------------------
    # 读取方法
    # ------------------------------------------------------------------

    def get(self, section, key, default=None):
        """
        读取配置值（字符串）。

        参数:
            section: 节名
            key: 键名
            default: 键不存在时的默认值

        返回:
            字符串值
        """
        self._check_reload()

        if not self._cp.has_section(section):
            return default
        if not self._cp.has_option(section, key):
            return default

        return self._cp.get(section, key)

    def get_int(self, section, key, default=None):
        """
        读取配置值（整数）。

        参数:
            section: 节名
            key: 键名
            default: 键不存在时的默认值，类型应为 int

        返回:
            整数值，解析失败时返回 default
        """
        self._check_reload()

        if not self._cp.has_section(section):
            return default
        if not self._cp.has_option(section, key):
            return default

        try:
            return int(self._cp.get(section, key))
        except (ValueError, TypeError):
            return default

    def has_option(self, section, key):
        """检查指定节中是否存在某个键"""
        return self._cp.has_option(section, key)

    def has_section(self, section):
        """检查是否存在某个节"""
        return self._cp.has_section(section)

    @property
    def sections(self):
        """返回所有节名"""
        return self._cp.sections()

    # ------------------------------------------------------------------
    # 链式访问
    # ------------------------------------------------------------------

    def __getattr__(self, section):
        """支持 config.SECTION.key 链式访问"""
        if section.startswith("_"):
            raise AttributeError(section)
        return SectionProxy(self, section)
