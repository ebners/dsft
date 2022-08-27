import dsft
import callable

benchmarks = ["susan_c","bzip2d","bzip2e","jpeg_c","jpeg_d"]

for bench in benchmarks:
    print("calculating full benchmark: " + bench)
    sf = dsft.setfunctions.WrapSetFunction(callable.CallableCompOpt(benchmark = bench,n=10),n=10)
    sf_coefs = sf(dsft.utils.get_indicator_set(10))
    sf_sig = dsft.setfunctions.WrapSignal(sf_coefs)
    sf_sig.export_to_csv(bench + "_full.txt")
    sf.call_counter = 0
    print("finished: " + bench)