# python读取ini配置文件公用
# 作者：dmf
# 时间：2026/3/27
import re
import os

_env_pattern = re.compile(r"\$\{([^}:]+)(?::-(.*?))?\}")


def interpolate(text):
    """
    将文本中的 ${VAR} 和 ${VAR:-default} 替换为环境变量值。

    ${VAR}         -> os.environ.get("VAR", "")
    ${VAR:-default} -> os.environ.get("VAR", "default")
    """
    if text is None:
        return None

    def replacer(m):
        var_name = m.group(1)
        default = m.group(2) if m.group(2) is not None else ""
        return os.environ.get(var_name, default)

    return _env_pattern.sub(replacer, text)
