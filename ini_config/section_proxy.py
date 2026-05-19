# python读取ini配置文件公用
# 作者：dmf
# 时间：2026/3/27
import os
import stat

from .env_interpolator import interpolate


class SectionProxy(object):
    """
    通过 config.SECTION.key 的方式链式访问配置。
    """

    def __init__(self, config, section):
        self._config = config
        self._section = section

    def __getattr__(self, key):
        if key.startswith("_"):
            raise AttributeError(key)
        return self._config.get(self._section, key)

    def __setattr__(self, key, value):
        if key.startswith("_"):
            self.__dict__[key] = value
        else:
            raise NotImplementedError("IniConfig is read-only")

    def __contains__(self, key):
        return self._config.has_option(self._section, key)

    def get_int(self, key, default=None):
        """获取整数值"""
        return self._config.get_int(self._section, key, default=default)

    def get(self, key, default=None):
        """获取字符串值"""
        return self._config.get(self._section, key, default=default)

    def items(self):
        """返回该节下所有键值对"""
        if self._section not in self._config._cp.sections():
            return []
        return list(self._config._cp.items(self._section))

    def keys(self):
        """返回该节下所有键名"""
        if self._section not in self._config._cp.sections():
            return []
        return [k for k, v in self._config._cp.items(self._section)]

    def mtime(self):
        """返回配置文件的最后修改时间"""
        return os.stat(self._config._path)[stat.ST_MTIME]

    def reload(self):
        """重新加载配置文件"""
        self._config.reload()
