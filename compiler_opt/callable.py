import environment as env
import cbench

class CallableCompOpt:
    def __init__(self, comp = env.default_compiler,benchmark = cbench.default_bench,n = 10):
        self.compiler = comp
        self.bench = benchmark
        self.setup = cbench.CBMeasurementSetup(compiler= self.compiler, bench=self.bench, input_data=1, iterations=5, max_std=0.05,stability_check= 0, out_percentile=0.1, flag_subset=None, parallel=False)
        self.setup.set_subset(list(range(n)))
        self.n = n

    def __call__(self,ind):
        ind_list = ind.tolist()
        opt = env.ind_to_opt([ind_list])
        return self.setup.measure(opt[0])