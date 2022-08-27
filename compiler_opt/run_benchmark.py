from dsft import setfunctions
from dsft import utils
import callable
import time
import numpy as np

benchmarks = ["bitcount","susan_s","susan_e","susan_s","bzip2d","bzip2e","jpeg_c","jpeg_d"]

for bench in benchmarks:
    sf = setfunctions.WrapSetFunction(callable.CallableCompOpt(benchmark = bench,n=10),n=10)

    times = []
    calls = []
    min_optimizations = []
    min_runtimes = []
    min_optimizations_g = []
    min_runtimes_g =[]
    errors = []

    start_full = time.time()
    sf_coefs = sf(utils.get_indicator_set(10)).tolist()
    end_full = time.time()
    full_time = end_full -start_full
    times += [full_time]
    sf_sig = setfunctions.WrapSignal(sf_coefs)
    sf_min_opt = sf_sig.min()
    sf_min_runtime = sf(sf_min_opt)
    min_optimizations += [sf_min_opt]
    min_runtimes += [sf_min_runtime]
    sf_sig.export_to_csv(bench + "10__full.txt")
    sf.call_counter = 0

    start_sf3 = time.time()
    sf3 = sf.transform_sparse(model = '3',eps = 1e-2)
    end_sf3 = time.time()
    sf3_time = end_sf3 -start_sf3
    times += [sf3_time]
    sf3_callcounter = sf.call_counter
    calls += [sf3_callcounter]
    sf3_minopt, _ = sf3.minimize_MIP()
    sf3_minruntime = sf(sf3_minopt)
    sf3_minopt_g , sf3_min_runtime_g = sf3.minimize_greedy(10,10)
    min_optimizations += [sf3_minopt]
    min_runtimes += [sf3_minruntime]
    min_optimizations_g += [sf3_minopt_g]
    min_runtimes_g += [sf3_min_runtime_g]
    errors += [setfunctions.eval_sf(sf,sf3,n=10,n_samples =50, err_types=["rel"])]
    sf3.export_to_csv(bench + "_10"+"_ssft3.txt")

    sf.call_counter = 0

    start_sfW3 = time.time()
    sfW3 = sf.transform_sparse(model = 'W3',eps = 1e-2)
    end_sfW3 = time.time()
    sfW3_time = end_sfW3 -start_sfW3
    times += [sfW3_time]
    sfW3_callcounter = sf.call_counter
    calls += [sfW3_callcounter]
    sfW3_minopt, _ = sfW3.minimize_MIP()
    sfW3_minruntime = sf(sfW3_minopt)
    sfW3_minopt_g , sfW3_min_runtime_g = sfW3.minimize_greedy(10,10)
    min_optimizations += [sfW3_minopt]
    min_runtimes += [sfW3_minruntime]
    min_optimizations_g += [sfW3_minopt_g]
    min_runtimes_g += [sfW3_min_runtime_g]
    errors += [setfunctions.eval_sf(sf,sfW3,n=10,n_samples =50, err_types=["rel"])]
    sfW3.export_to_csv(bench +"_10_"+"_ssftW3.txt")

    sf.call_counter = 0


    start_sf4 = time.time()
    sf4 = sf.transform_sparse(model = '4',eps = 1e-2)
    end_sf4 = time.time()
    sf4_time = end_sf4 -start_sf4
    times += [sf4_time]
    sf4_callcounter = sf.call_counter
    calls += [sf4_callcounter]
    sf4_minopt, _ = sf4.minimize_MIP()
    sf4_minruntime = sf(sf4_minopt)
    sf4_minopt_g , sf4_min_runtime_g = sf4.minimize_greedy(10,10)
    min_optimizations += [sf4_minopt]
    min_runtimes += [sf4_minruntime]
    min_optimizations_g += [sf4_minopt_g]
    min_runtimes_g += [sf4_min_runtime_g]
    errors += [setfunctions.eval_sf(sf,sf4,n=10,n_samples =50, err_types=["rel"])]
    sf4.export_to_csv(bench +"_10_"+ "_ssft4.txt")

    np.savetxt("times_10_"+ bench ,np.asarray(times))
    np.savetxt("calls_10_"+ bench,np.asarray(calls))
    np.savetxt("errors_10_"+ bench ,np.asarray(errors))
    np.savetxt("minopt_MIP_10_"+ bench,np.asarray(min_optimizations))
    np.savetxt("minopt_greedy_10_"+ bench,np.asarray(min_optimizations_g))
    np.savetxt("minruntimes_10_"+ bench,np.asarray(min_runtimes))
    np.savetxt("minruntimes_greedy_10_"+ bench,np.asarray(min_runtimes_g))
    sf.call_counter = 0