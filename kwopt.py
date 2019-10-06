import scipy.optimize
from scipy.special import logit, expit
import numpy as np

class logbarrier:
    @classmethod
    def mangle(cls, x):
        return np.log(x)
   
    @classmethod
    def unmangle(cls, X):
        return np.exp(X)
    
class logitbarrier:
    @classmethod
    def mangle(cls, x):
        return logit(x)
   
    @classmethod
    def unmangle(cls, X):
        return expit(X)

class fixed: pass

def minimizer(f, opt_f=scipy.optimize.minimize, *oargs, **okwargs):
    def minimize(**spec):
        #keys, spec = ikwargs.keys(), ikwargs.values()
        #spec = [spec if isinstance(spec, tuple) else (spec,)]
        for k, s in spec.items():
            if not isinstance(s, tuple):
                spec[k] = (s,)
        
        def mangle(xs):
            Xs = []
            for key, x in xs.items():
                if fixed in spec[key]:
                    continue
                for mangler in spec[key]:
                    if hasattr(mangler, 'mangle'):
                        x = mangler.mangle(x)
                Xs.append(x)
            return Xs

        def unmangle(Xs):
            xs = {}
            Xs = list(np.atleast_1d(Xs))
            for name, s in spec.items():
                if fixed in s:
                    xs[name] = s[0]
                    continue
                x = Xs.pop(0)
                for mangler in s:
                    if hasattr(mangler, 'unmangle'):
                        x = mangler.unmangle(x)
                xs[name] = x
            return xs
            

        def wrapper(x):
            k = unmangle(x)
            return f(**k)
        initial = {k: v[0] for k, v in spec.items()}
        result = opt_f(wrapper, mangle(initial), *oargs, **okwargs)
        result.kwargs = unmangle(result.x)
        return result
    return minimize
