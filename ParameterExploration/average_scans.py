from pypet import Trajectory
import numpy as np
from os import listdir
from os.path import expanduser
from h5py import File
from time import time

start_time = time()

home_path = expanduser('~')
data_path = home_path + '/Sciebo/ParameterExploration_results/results/'
print_path = home_path + '/Sciebo/ParameterExploration_results/average/'

result_names = ['Corrcoef', 'EW_max', 'Redundancy', 'SCREE_count',
                'broken_stick_count', 'spectral_count', 'Norm_estimate',
                'Norm_exact', 'min_n_exact', 'D_KL',
                'D_KL_inv', 'D_KS', 'D_KS_p', 'MWU', 'MWU_p', 'Space_angle']


def _init_avg_file(name, init_file):
    f = File(print_path + name, 'w-')

    # Parameters
    f.create_group('Parameters')

    # scanned parameters
    scan_params = ['corr', 'T', 'A_size', 'bkgr_corr']
    traj = _load_traj(init_file, params_load=2, result_load=0)
    for scan_param in scan_params:
        param_array = traj.parameters.f_get(scan_param)
        f.create_dataset('Parameters/{}'.format(scan_param), data=param_array)

    runs = f['Parameters/' + scan_params[0]].len()
    print "{} runs".format(runs)

    # fixed parameters (times in ms)
    f.create_dataset('Parameters/N', data=np.array([100]*runs))
    f.create_dataset('Parameters/binsize', data=np.array([2]*runs))
    f.create_dataset('Parameters/rate', data=np.array([50]*runs))

    # Results
    f.create_group('Results')

    for result_name in result_names:
        f.create_dataset('Results/{}/avg'.format(result_name), (runs,), 'float')
        f.create_dataset('Results/{}/var'.format(result_name), (runs,), 'float')

    f['Results'].attrs['base_num'] = 0

    return f


def _load_traj(filename, params_load=0, result_load=2):
    print filename
    traj = Trajectory(filename=data_path + filename)
    traj.f_load(index=-1, load_parameters=params_load,
                load_results=result_load, force=True)
    traj.v_auto_load = True
    return traj


def _load_avg_file(name):
    f = File(print_path + name, 'r')
    return f


def _update_avg_file(file_obj=None, name=None, new_scan=None):
    if file_obj is None:
        assert name is not None
        file_obj = _load_avg_file(name)
    traj = _load_traj(new_scan)

    base_num = file_obj['Results'].attrs['base_num']

    nbr_of_sets = traj.results.f_children()
    for set_it in range(nbr_of_sets):
        set_obj = traj.results['run_set_{:05d}'.format(set_it)]
        nbr_of_runs = set_obj.f_children()

        for run_it in range(nbr_of_runs):
            run = run_it + set_it*1000
            run_obj = set_obj['run_{:08d}'.format(run)]

            for result_name in result_names:
                if result_name == 'D_KL':
                    scan_value = run_obj['D_KL'][0]
                elif result_name == 'D_KL_inv':
                    scan_value = run_obj['D_KL'][1]
                elif result_name == 'D_KS':
                    scan_value = run_obj['D_KS'][0]
                elif result_name == 'D_KS_p':
                    scan_value = run_obj['D_KS'][1]
                elif result_name == 'MWU':
                    scan_value = run_obj['MWU'][0]
                elif result_name == 'MWU_p':
                    scan_value = run_obj['MWU'][1]
                else:
                    if result_name in run_obj:
                        scan_value = run_obj[result_name]
                    else:
                        print 'Name not found in results: {}'\
                              .format(result_name)

                if not base_num:
                    file_obj['Results/{}/avg'.format(result_name)] = scan_value
                    file_obj['Results'].attrs['base_num'] = 1
                else:
                    prev_avg = file_obj['Results/{}/avg'.format(result_name)]
                    new_avg = (base_num*prev_avg + scan_value) / (base_num + 1)
                    file_obj['Results/{}/avg'.format(result_name)] = new_avg

                    prev_var = file_obj['Results/{}/var'.format(result_name)]
                    new_var = ((base_num-1)*prev_var
                            + (scan_value-new_avg)*(scan_value-prev_avg)) \
                            * 1./base_num
                    file_obj['Results/{}/var'.format(result_name)] = new_var

    file_obj['Results'].attrs['base_num'] = base_num + 1

    return file_obj

scan_nbr = 0
for filename in listdir(data_path)[:1]:
    if filename.split('.')[-1] == 'h5':
        # f = _init_avg_file('test_file.h5', filename)
        f = _update_avg_file(name='test_file.h5', new_scan=filename)
        scan_nbr += 1

f.close()

print "--{} scans in {} minutes--".format(scan_nbr, (time() - start_time)/60.)

