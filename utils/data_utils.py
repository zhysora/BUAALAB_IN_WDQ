import codecs
import json
from os.path import join
import pickle
import os
from utils import settings


def load_json(rfdir, rfname):
    with codecs.open(join(rfdir, rfname), 'r', encoding='utf-8') as rf:
        return json.load(rf) # 读取json 文件 返回为 字典 {key: value}


def dump_json(obj, wfpath, wfname, indent=None):
    with codecs.open(join(wfpath, wfname), 'w', encoding='utf-8') as wf:
        json.dump(obj, wf, ensure_ascii=False, indent=indent)



def dump_data(obj, wfpath, wfname): #对象， 文件路径， 文件名
    with open(os.path.join(wfpath, wfname), 'wb') as wf:
        pickle.dump(obj, wf) #将对象序列化， 输出到wf中


def load_data(rfpath, rfname): #读文件 路径名， 问文件名
    with open(os.path.join(rfpath, rfname), 'rb') as rf:
        return pickle.load(rf) #反序列化 得到对象


def serialize_embedding(embedding):
    return pickle.dumps(embedding)


def deserialize_embedding(s): #反序列化 嵌入
    return pickle.loads(s) #从一个file-like Object中直接反序列化出对象


class Singleton:
    """
    A non-thread-safe helper class to ease implementing singletons.
    This should be used as a decorator -- not a metaclass -- to the
    class that should be a singleton.

    The decorated class can define one `__init__` function that
    takes only the `self` argument. Also, the decorated class cannot be
    inherited from. Other than that, there are no restrictions that apply
    to the decorated class.

    To get the singleton instance, use the `Instance` method. Trying
    to use `__call__` will result in a `TypeError` being raised.

    """

    def __init__(self, decorated):
        self._decorated = decorated

    def Instance(self):
        """
        Returns the singleton instance. Upon its first call, it creates a
        new instance of the decorated class and calls its `__init__` method.
        On all subsequent calls, the already created instance is returned.

        """
        try:
            return self._instance
        except AttributeError:
            self._instance = self._decorated()
            return self._instance

    def __call__(self):
        raise TypeError('Singletons must be accessed through `Instance()`.')

    def __instancecheck__(self, inst):
        return isinstance(inst, self._decorated)
