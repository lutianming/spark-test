#!/usr/bin/env python

import subprocess
import time


def call(call_args):
    start = time.time()
    subprocess.call(call_args)
    return time.time() - start

basedir = '/user/Tianming/linear1000/'
outputdir = 'results/'
cmd = 'spark-submit'

EXECUTOR = 1
BATCHSIZE = 2
SIZE = 3
STEP = 4
CORES = 5

def call_args(files, output, svm, batch, driver_memory='2g',
              executor_memory='2g', cores=1,
              num_executors=3, stepsize=1, numIters=100):
    args = [cmd,
            '--driver-memory', driver_memory,
            '--executor-memory', executor_memory,
            '--num-executors', str(num_executors),
            '--executor-cores', str(cores),
            '--class', 'app',
            '--master', 'yarn-client',
            'spark-test_2.10-0.1.0.jar',
            files,
            svm, str(batch), str(stepsize), str(numIters), output]
    return args

def read_time(filename):
    with open(filename, 'r') as f:
        time = f.readline()
    return int(time)

def test_svm(output, svm, features):
    f = open(output, 'w')

    # test different number of executors
    if EXECUTOR in features:
        f.write('# executors\ttime\n')
        executors = range(8, 16+1)
        for i in executors:
            output = outputdir + 'executor#{0}#{1}.txt'.format(i, svm)
            # memory = 7-i
            # if memory < 1:
            #     memory = 1
            if i <= 12:
                memory = 2000
            else:
                memory = 1500

            args = call_args(basedir + '[0-3].csv',
                             output,
                             svm, 0.1,
                             num_executors=i,
                             executor_memory='{0}m'.format(memory),
                             driver_memory='3g')

            call(args)
            duration = read_time(output)
            f.write('{0}\t{1}\n'.format(i, duration))

    # different minibatch
    if BATCHSIZE in features:
        f.write('# batchsize\ttime\n')
        minibatchs = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.01]
        for i in minibatchs:
            output = outputdir + 'batchsize#{0}#{1}.txt'.format(i, svm)
            args = call_args(basedir + '[0-4].csv',
                             output,
                             svm, str(i),
                             num_executors=4,
                             executor_memory='3g',
                             driver_memory='2g')
            call(args)
            duration = read_time(output)
            f.write('{0}\t{1}\n'.format(i, duration))

    # different filesize
    if SIZE in features:
        f.write('# size\ttime\n')
        for i in range(0, 10):
            if i == 0:
                filename = basedir + '0.csv'
            else:
                filename = (basedir + '[0-{0}].csv').format(i)
            output = outputdir + 'size#{0}#{1}.txt'.format(i, svm)
            args = call_args(filename,
                             output,
                             svm, 1,
                             num_executors=4,
                             executor_memory='6g',
                             driver_memory='2g')
            call(args)
            duration = read_time(output)
            f.write('{0}\t{1}\n'.format(i+1, duration))

    if STEP in features:
        f.write('# step\ttime\n')
        steps = range(1, 6)
        for i in steps:
            if svm == "svm":
                stepsize = i*30
            else:
                stepsize = i
            output = outputdir + 'step#{0}#{1}.txt'.format(stepsize, svm)
            args = call_args(basedir + '[0-4].csv',
                             output,
                             svm, 0.1,
                             num_executors=4,
                             executor_memory='6g',
                             driver_memory='2g',
                             stepsize=stepsize)
            call(args)
            duration = read_time(output)
            f.write('{0}\t{1}\n'.format(i, duration))
    if CORES in features:
        f.write('# core\ttime\n')
        steps = range(1, 5)
        for i in steps:
            output = outputdir + 'core#{0}#{1}.txt'.format(i, svm)
            args = call_args(basedir + '[0-1].csv',
                             output,
                             svm, 0.1,
                             num_executors=4,
                             executor_memory='6g',
                             driver_memory='2g',
                             cores=i,
                             stepsize=stepsize)
            call(args)
            duration = read_time(output)
            f.write('{0}\t{1}\n'.format(i, duration))
    f.close()

def main():
    # features = [EXECUTOR, BATCHSIZE, SIZE, STEP]
    features = [CORES]
    test_svm(outputdir + "grid_pegasos.csv", "pegasos", features)
    test_svm(outputdir + "grid_svm.csv", "svm", features)

if __name__ == '__main__':
    main()
