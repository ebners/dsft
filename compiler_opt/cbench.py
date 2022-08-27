import functools
import os
import shutil
import subprocess
import copy
import environment as env
import re
import util
from typing import List

"""
A module implementing support for the cBench benchmark suite.
"""


"""The path to the cBench benchmark suite."""
root = os.path.join(env.proj_root,"compiler_opt" ,"benchmarks")

"""A dictionary enlisting all benchmarks available in the suite, 
where the key indicates the name of the benchmark and the value its location.
"""
benchmarks = {
    "bitcount": "automotive_bitcount/src",
    "qsort1" : "automotive_qsort1/src",
    "susan_c" : "automotive_susan_c/src",
    "susan_e" : "automotive_susan_e/src",
    "susan_s" : "automotive_susan_s/src",
    "bzip2d" : "bzip2d/src",
    "bzip2e" : "bzip2e/src",
    "jpeg_c" : "consumer_jpeg_c/src",
    "jpeg_d" : "consumer_jpeg_d/src",
    "lame" : "consumer_lame/src",
    # "mad" : "consumer_mad/src", # not compiling
    "tiff2bw" : "consumer_tiff2bw/src",
    "tiff2rgba" : "consumer_tiff2rgba/src",
    "tiffdither" : "consumer_tiffdither/src",
    "tiffmedian" : "consumer_tiffmedian/src",
    "dijkstra" : "network_dijkstra/src",
    "patricia" : "network_patricia/src",
    # "ghostscript" : "office_ghostscript/src", # not compiling
    # "ispell" : "office_ispell/src", # not compiling
    "rsynth" : "office_rsynth/src",
    # "stringsearch1" : "office_stringsearch1/src", # unstable measurements
    "blowfish_d" : "security_blowfish_d/src",
    "blowfish_e" : "security_blowfish_e/src",
    # "pgp_d" : "security_pgp_d/src", # not compiling
    # "pgp_e" : "security_pgp_e/src", # not compiling
    # "rijndael" : "security_rijndael_d/src", # not working for some optimizations
    "rijndael_e" : "security_rijndael_e/src",
    # "sha" : "security_sha/src", # validity check failing
    "adpcm_c" : "telecom_adpcm_c/src",
    "adpcm_d" : "telecom_adpcm_d/src",
    "CRC32" : "telecom_CRC32/src",
    "gsm" : "telecom_gsm/src"
}

default_bench = "bitcount"

class CBMeasurementSetup(env.MeasurementSetup):

    def __init__(self, compiler: env.CompilerSetup, bench: str, input_data, iterations: int, max_std: float, out_percentile: float, stability_check: int = 2, flag_subset: List[int] = None, parallel=False, use_make=False):
        self.use_make = use_make
        self._validity_check = False
        super().__init__(compiler, bench, input_data, iterations, max_std, out_percentile, stability_check, flag_subset, parallel)

    def set_input_data(self, input_data):
        super().set_input_data(input_data)
        self.reset_rounds()

    def set_bench(self, bench):
        self._validity_check = False
        super().set_bench(bench)
        self.reset_rounds()
        self.src_dir = os.path.join(root, benchmarks[bench])

    def get_default_rounds(self) -> int:
        # read number of rounds of loop_wrap from _ccc_info_datasets
        file = os.path.join(root, benchmarks[self.bench], "_ccc_info_datasets")
        with open(file, "r") as fp:
            max_datasets = int(fp.readline())
            for _ in fp:
                dataset = int(fp.readline())
                fp.readline()
                rounds = int(fp.readline())
                if dataset == self.input_data:
                    return rounds
        print("Warning: dataset not found")
        return 1

    def reset_rounds(self):
        self._rounds = max(self.get_default_rounds() // 10, 1) # one execution should roughly take one second

    def prepare(self, opt: env.Optimization):
        # change directory
        os.chdir(self.src_dir)

        # compile
        generic_makefile = os.path.join(self.src_dir, "Makefile")
        specific_makefile = generic_makefile + "." + self.compiler.type.name.lower()
        # copy correct makefile (also useful for non-makefile compilation)
        shutil.copy(specific_makefile, generic_makefile)
        if self.use_make:
            opt_str = functools.reduce(lambda l, r: l+" "+r, self.resolve_flags(opt), "")
            args = "make ZCC="+self.compiler.path + " CCC_OPTS='" + opt_str + "'"
            subprocess.run(args, check=True, shell=True, capture_output=True)
        else:
            # extract extra flags from makefile
            includes = []
            defines = []
            add_flags = ["-lm"]
            with open(specific_makefile) as fp:
                lines = fp.readlines()
                action_area = False
                for l in lines:
                    l = l.strip()
                    if re.match("all:", l):
                        action_area = True
                    elif re.match("clean:", l):
                        action_area = False
                    elif action_area:
                        for flag in re.findall("-[IDl]\S*", l):
                            add_flags.append(flag)

            self.compiler.compile(
                sources=["*.c"],
                includes=includes,
                opt=opt,
                output="a.out",
                flags=add_flags,
                defines=defines
            )

    def run_benchmark(self) -> float:
        os.chdir(self.src_dir)

        # run benchmarks
        args = ["./__run", str(self.input_data)]
        if self._rounds != -1:
            args.append(str(self._rounds))
        output = subprocess.run(args, check=True, capture_output=True, text=True)
       
        # check validity
        if self._validity_check:
            try:
                subprocess.run("./_ccc_check_output.diff", check=True, shell=True)
            except:
                raise env.OutputValidationError()

        # extract time
        # cBench uses the time command which reports the time in stderr
        line = re.search("real.*", output.stderr).group()
        minutes = float(re.search("\d*m", line).group()[:-1])
        seconds = float(re.search(util.FLOAT_REGEX + "s", line).group()[:-1])
        time = minutes*60 + seconds
        return time / self._rounds

    def prepare_validity_check(self):
        cur_it = self.iterations
        self.set_iterations(1)
        self.measure_O3()
        self.set_iterations(cur_it)
        os.chdir(self.src_dir)
        subprocess.run("./_ccc_check_output.copy", check=True, shell=True)
        self._validity_check = True

    def set_input_data(self, input_data):
        super().set_input_data(input_data)
        if self._validity_check:
            self.prepare_validity_check() # need to reprepare validity check if we set new input data

    def cleanup(self):
        os.chdir(self.src_dir)
        subprocess.run(["make", "clean"], capture_output=True)
        if self._validity_check:
            subprocess.run("./_ccc_check_output.clean", shell=True)
        self.compiler.remove_compilation_artifacts()

gcc_greedy_best_flags = [21, 31, 58, 46, 59, 44, 25, 103, 65, 18, 62, 33, 77, 11, 96, 102, 16, 39, 78, 61, 55, 84, 41, 63, 94, 53, 83, 70, 52, 36, 73, 89, 12, 104, 48, 34, 13, 38, 7, 91, 40, 8, 43, 49, 60, 56, 4, 109, 0, 108, 15, 45, 28, 90, 51, 26, 87, 50, 6, 30, 22, 88, 105, 97, 66, 72, 69, 17, 79, 5, 80, 20, 10, 85, 101, 71, 75, 2, 57, 98, 3, 74, 92, 32, 54, 93, 35, 47, 86, 27, 9, 67, 100, 23, 29, 42, 76, 81, 19, 95, 64, 68, 82, 24, 107, 37, 99, 14, 106, 1]

gcc_individual_best_flags = [21, 44, 52, 3, 29, 1, 18, 49, 96, 37, 16, 107, 15, 24, 45, 65, 74, 84, 92, 2, 32, 36, 39, 47, 50, 51, 56, 64, 66, 71, 81, 98, 101, 103, 108, 5, 7, 8, 13, 14, 17, 19, 33, 43, 48, 54, 61, 62, 75, 78, 80, 85, 89, 106, 22, 25, 31, 34, 38, 40, 53, 57, 67, 68, 69, 70, 72, 73, 77, 83, 90, 93, 94, 100, 109, 0, 4, 10, 11, 12, 20, 26, 30, 41, 42, 59, 63, 79, 91, 95, 97, 99, 102, 104, 9, 23, 28, 35, 55, 60, 76, 82, 87, 88, 105, 6, 27, 46, 58, 86]

gcc_best_flags = gcc_greedy_best_flags
"""A selection of flags which gave the best runtime difference to no flags when measured with the default setup, ordered in increaing order by measured difference."""

default_setup = CBMeasurementSetup(compiler=env.default_compiler, bench=default_bench, input_data=1, iterations=-1, max_std=0.05, out_percentile=0.1, flag_subset=None, parallel=False)
test_setup = copy.copy(default_setup)
test_setup.iterations = 5
test_setup.set_subset([gcc_best_flags[0]])