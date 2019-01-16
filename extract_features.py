# Extract features from MPU dataset:
# https://sites.google.com/view/mobile-phone-use-dataset

import sys
import os
import time
import argparse
import shutil

import pandas as pd
# import numpy as np

from tqdm import tqdm
from datetime import datetime, timedelta

from multiprocessing import Pool, cpu_count
import traceback


##############################################################################################
# FILE PATHS
##############################################################################################

GLOBAL_PATH = "."  # The global path that holds the INPUT_DATA and OUTPUT_DATA folders. e.g. '~/Users/<user-name>/Project'.
INPUT_DATA = "mobile_phone_use"  # The MPU dataset folder
OUTPUT_DATA = "features"  # The folder that will store the extracted features per user

PARTICIPANTS_INFO_FILENAME = "pinfo.csv"  # Should be 'pinfo.csv'
GT_COLUMN = "Esm_TiredAwake"  # Your Ground Truth column of the Esm sensor event.


##############################################################################################
# METHODS
##############################################################################################

def extract_features(pinfo, df, ff):

    # This is the place where the feature extraction should happen (per user).
    # 'pinfo' is a dict with the participant's information.
    # 'df' is the dataframe with the participant's mobile phone use data.
    # 'ff' is an empty dataframe that will store the features. GT_COLUMN is also copied.

    # This example uses the 'Acc' sensor (Accelerometer) and extracts the
    # average acceleration of the last measurement before each Esm event.
    df['Acc_Avg'].fillna(method='ffill', inplace=True)
    ff['ft_last_acc'] = df['Acc_Avg']


def extract_features_per_core(params):

    # unpack parameters
    pinfo, input_data_path, output_data_path = params

    try:
        # prepare paths
        input_file_path = os.path.join(input_data_path, "%s.csv" % pinfo.uuid)
        output_file_path = os.path.join(output_data_path, "%s.csv" % pinfo.uuid)

        # read data file
        df = pd.read_csv(input_file_path, low_memory=False)

        # init ff (features dataframe) and set GT
        ff = df[df.sensor_id == "Esm"][[GT_COLUMN]].copy().dropna()

        # extract features using pinfo, from df to ff.
        extract_features(pinfo, df, ff)

        # sort columns
        sorted_columns = sort_columns(ff.columns)
        ff = ff[sorted_columns]

        # save into csv
        ff.to_csv(output_file_path, index=False)

        # Status ok
        return True

    except KeyboardInterrupt:
        return False

    except Exception:
        e = sys.exc_info()[0]
        msg = sys.exc_info()[1]
        tb = sys.exc_info()[2]
        message = "exception: %s '%s'" % (e, msg)
        tqdm.write(message)
        traceback.print_tb(tb)
        return False


def extract_all_features(pdf, input_data_path, output_data_path, nproc):

    # choose between single core vs multi-core
    if nproc <= 1:

        for _, pinfo in tqdm(pdf.iterrows(), total=len(pdf), desc='User', ncols=80):

            # pack params and extract features
            params = (pinfo, input_data_path, output_data_path)
            status = extract_features_per_core(params)

            # check for KeyboardInterrupt
            if status is False:
                raise KeyboardInterrupt

    else:
        # init pool with nproc
        pool = Pool(processes=nproc)

        # prepare parameters
        params = [(pinfo, input_data_path, output_data_path) for _, pinfo in pdf.iterrows()]

        try:
            for status in tqdm(pool.imap_unordered(extract_features_per_core, params), total=len(pdf), desc='User', ncols=80):

                # check for KeyboardInterrupt
                if status is False:
                    raise KeyboardInterrupt

        except KeyboardInterrupt:
            pool.terminate()


def sort_columns(columns):

    # sort columns by name, GT_COLUMN should be the last column
    columns = sorted(list(columns))
    # columns.insert(0, columns.pop(columns.index("u2")))  # first
    columns.append(columns.pop(columns.index(GT_COLUMN)))  # last

    return columns


def ensure_path(path, clean=False):

    if clean and os.path.exists(path):
        shutil.rmtree(path)

    if not os.path.exists(path):
        os.makedirs(path)


def parse_arguments(args):

    parser = argparse.ArgumentParser(description="extract features (using 'ft_' as a column prefix)")
    parser.add_argument('-p', '--parallel', dest='parallel', type=int, nargs=1, metavar='nproc', default=[0],
                        help='execute in parallel, nproc=number of processors to use.')
    parser.add_argument('-sd', '--sudden-death', dest='sudden_death', action='store', nargs='*', metavar='uuid',
                        help='sudden death: use particular uuid to test the features extraction; either specify the uuid or omit it and it reads out a default one from code (ie. u000)')
    parsed = vars(parser.parse_args(args))

    return parsed


##############################################################################################
# MAIN
##############################################################################################

def main(args):
    '''main function'''

    # detemine number of CPUs
    nproc = args['parallel'][0]
    if nproc <= 0:
        # automatically selects about 80% of the available CPUs
        cpus = cpu_count()
        nproc = int(cpus * 0.8 + 0.5)
    else:
        nproc = min([nproc, cpu_count()])
    print("using %d CPUs" % nproc)

    # get paths
    global_path = os.path.expanduser(GLOBAL_PATH)
    input_data_path = os.path.join(global_path, INPUT_DATA, "data")
    output_data_path = os.path.join(global_path, OUTPUT_DATA)

    # clean and ensure dir
    ensure_path(output_data_path, clean=True)

    # load pinfo.csv
    pinfo_path = os.path.join(global_path, INPUT_DATA, PARTICIPANTS_INFO_FILENAME)
    print(pinfo_path)
    if not os.path.isfile(pinfo_path):
        sys.exit("Participant's info file with name '%s' does not exist." % PARTICIPANTS_INFO_FILENAME)

    # load json file
    with open(pinfo_path) as data_file:
        pdf = pd.read_csv(data_file)

    # determine sudden_death
    sudden_death = args['sudden_death']
    if sudden_death is not None:
        if len(sudden_death) == 0:
            sudden_death = ['u000']  # default user

        # apply sudden_death
        pdf = pdf[pdf.uuid.isin(sudden_death)]

    # begin feature extraction
    extract_all_features(pdf, input_data_path, output_data_path, nproc)


if __name__ == '__main__':

    # parse args
    args = parse_arguments(sys.argv[1:])

    try:
        # track time
        print("Started at: %s" % (datetime.now()))
        start_time = time.time()

        # call main
        main(args)

        # save and report elapsed time
        elapsed_time = time.time() - start_time
        print("\nSuccess! Duration: %s" % str(timedelta(seconds=int(elapsed_time))))

    except(KeyboardInterrupt):
        sys.exit("Interrupted: Exiting on request.")

    except(SystemExit):
        e = sys.exc_info()[0]
        msg = sys.exc_info()[1]
        tb = sys.exc_info()[2]
        message = "exception: %s '%s'" % (e, msg)
        tqdm.write(message)
        traceback.print_tb(tb)
