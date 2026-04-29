import hashlib
import inspect


class StaticConfigs:
    def __init__(self, dict_):
        self.__configs = dict_

    def __hash__(self):
       # 检查内部是否存在函数，如果存在，则将函数转换为完整的函数签名
        hash_str = ''
        for key, value in self.__configs.items():
            if callable(value):
                # 使用函数的完整签名+函数代码体来生成哈希值
                hash_str += f"{key}:{inspect.signature(value)}:{inspect.getsource(value)}|"
            else:
                hash_str += f"{key}:{value}|"
        return int(hashlib.md5(hash_str.encode()).hexdigest(), 16)
    
    # 像字典一样访问
    def __getitem__(self, key):
        return self.__configs[key]