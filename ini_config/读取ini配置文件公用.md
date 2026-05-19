# ini_config

Python INI 配置文件读取公用类，支持 Python 3.6+，零第三方依赖。

## 安装

将 `ini_config` 文件夹复制到项目目录下即可使用，无需安装。

## 快速开始

### 基础读取

```python
from ini_config import IniConfig

config = IniConfig("ZdsLineFaultAnaAgent.ini", encoding="gbk")

# 字符串
path = config.get("PROGRAM_ROOT_PATH", "program_root_path", default="..")

# 整数
db_num = config.get_int("DB_CONN_NUM", "db_connect_num", default=1)
```

### 链式访问

```python
# 通过 config.SECTION.key 直接读取
path = config.PROGRAM_ROOT_PATH.program_root_path       # 返回 str
db_num = config.DB_CONN_NUM.get_int("db_connect_num", default=1)  # 返回 int

# 检查键是否存在
if "n_ifweather" in config.SWITCH_WEATHER:
    pass

# 遍历节内所有键值
for key, value in config.SWITCH_WEATHER.items():
    print(key, value)
```

### 环境变量替换

INI 文件中可使用 `${VAR}` 或 `${VAR:-default}` 语法：

```ini
[DB]
db_path = ${APP_HOME}/data/db.ini
```

### 多文件合并

后声明的文件会覆盖先声明的文件中的同名键值：

```python
config = IniConfig(["default.ini", "user.ini"])
```

### 覆盖特定值（override）

```python
config = IniConfig("config.ini", override={
    "DB.host": "127.0.0.1",
})
```

### 热重载

```python
config = IniConfig("config.ini", auto_reload=True)
value = config.SECTION.key   # 始终读取最新值
```

### 指定编码

```python
config = IniConfig("config.ini", encoding="gbk")  # Windows 中文环境常用 GBK
```

### 手动重新加载

```python
config.reload()
```

## API 参考

### IniConfig(path=None, encoding="utf-8", auto_reload=False, base_dir=None, override=None)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| path | str/list/tuple | None | 配置文件路径 |
| encoding | str | "utf-8" | 文件编码 |
| auto_reload | bool | False | 是否开启热重载 |
| base_dir | str | None | 配置目录，None 时使用 cwd |
| override | dict | None | 覆盖特定配置值 |

### get(section, key, default=None)

读取配置值，返回字符串。

### get_int(section, key, default=None)

读取整数值，等价于 `get(section, key, default=default, value_type=int)`。

### has_option(section, key)

检查指定节中是否存在某个键。

### has_section(section)

检查是否存在某个节。

### sections

返回所有节名列表。

### reload()

手动重新加载配置文件。
