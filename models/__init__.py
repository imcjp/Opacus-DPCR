from .mnist import *

def gen(name,args=None):
    if args is None:
        return eval(name)();
    else:
        return eval(name)(**args);
