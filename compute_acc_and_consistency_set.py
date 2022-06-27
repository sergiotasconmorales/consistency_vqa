# Project:
#   VQA
# Description:
#   Script to compute metrics for a set of experiments. 
# Author: 
#   Sergio Tascon-Morales

import pandas as pd
import misc.io as io
from os.path import join as jp
from compute_consistency import compute_consistency
from analyze_val_answers import compute_accuracies

path_configs = 'config/idrid_regions/single/'
# now give range for files to be processed
r_init = 977
r_end = 986
path_output = 'logs/stats'
q3_too = True

def main():

    # create dataframe
    if q3_too:
        df = pd.DataFrame(columns = ['config', 'overall', 'grade', 'whole', 'fovea', 'inside', 'q2', 'q3_1', 'q3_2'])
    else:
        df = pd.DataFrame(columns = ['config', 'overall', 'grade', 'whole', 'fovea', 'inside', 'consistency'])
    i_row = 0
    #iterate 
    for i_config in range(r_init, r_end+1):
        # read config file
        config = io.read_config(jp(path_configs, 'config_' + str(i_config) + '.yaml'))
        config_file_name = 'config_' + str(i_config)

        # first, get consistency
        c = compute_consistency(config, config_file_name, q3_too = q3_too)

        # now, get accuracies
        accs = compute_accuracies(config, config_file_name, 'test') # by default for test set

        if q3_too:
            df.at[i_row] = [str(i_config), accs['overall'], accs['grade'], accs['whole'], accs['fovea'], accs['inside'], c[0], c[1], c[2]] 
        else:
            df.at[i_row] = [str(i_config), accs['overall'], accs['grade'], accs['whole'], accs['fovea'], accs['inside'], c] 
        i_row += 1

    # add means and stds
    if q3_too:
        mean = ['AVG', df['overall'].mean(), df['grade'].mean(), df['whole'].mean(), df['fovea'].mean(), df['inside'].mean(), df['q2'].mean(), df['q3_1'].mean(), df['q3_2'].mean()]
        std = ['STD', df['overall'].std(), df['grade'].std(), df['whole'].std(), df['fovea'].std(), df['inside'].std(), df['q2'].std(), df['q3_1'].std(), df['q3_2'].std()]
    else:
        mean = ['AVG', df['overall'].mean(), df['grade'].mean(), df['whole'].mean(), df['fovea'].mean(), df['inside'].mean(), df['consistency'].mean()]
        std = ['STD', df['overall'].std(), df['grade'].std(), df['whole'].std(), df['fovea'].std(), df['inside'].std(), df['consistency'].std()]
    df.at[i_row] = mean
    i_row += 1
    df.at[i_row] = std

    # save
    if q3_too:
        df.to_csv(jp(path_output, 'configs_' + str(r_init) + 'to' + str(r_end) + '_q3.csv'), index=False)
    else:
        df.to_csv(jp(path_output, 'configs_' + str(r_init) + 'to' + str(r_end) + '.csv'), index=False)


if __name__ == '__main__':
    main()