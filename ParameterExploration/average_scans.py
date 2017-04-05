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
    print "\nInitialize new file... \n"

    # Parameters
    f.create_group('Parameters')

    # scanned parameters
    scan_params = ['corr', 'T', 'A_size', 'bkgr_corr']
    traj = _load_traj(init_file, params_load=2, result_load=0)
    for scan_param in scan_params:
        param_array = traj.parameters.f_get(scan_param)
        f.create_dataset('Parameters/{}'.format(scan_param), data=param_array)

    runs = f['Parameters/' + scan_params[0]].len()

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
    f['Results'].attrs['processed_files'] = ['']

    return f


def _load_traj(filename, params_load=0, result_load=2):
    print filename
    traj = Trajectory(filename=data_path + filename)
    traj.f_load(index=-1, load_parameters=params_load,
                load_results=result_load, force=True)
    traj.v_auto_load = True
    return traj


def _load_avg_file(name, mode='r+'):
    avg_file = File(print_path + name, mode)
    return avg_file


def _update_avg_file(file_obj=None, name=None, new_scan=None):
    if file_obj is None:
        assert name is not None
        print '\nLoad file...\n'
        file_obj = _load_avg_file(name)

    processed_files = file_obj['Results'].attrs['processed_files']
    if new_scan in processed_files:
        print 'File already regarded in this average!'
        return file_obj

    print '\nLoad new parameter scan...\n'
    traj = _load_traj(new_scan)

    print '\nUpdate file...\n'

    runs = f['Results/{}/avg'.format(result_names[0])].len()
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

                subgroup = 'Results/{}/'.format(result_name)
                avg = file_obj[subgroup + 'avg'][run]
                var = file_obj[subgroup + 'var'][run]

                if np.size(avg) != np.size(scan_value):
                    # print 'Sizes differ ({} | {}) in run {} for {} \n' \
                    #       'will only take the first element!' \
                    #       .format(np.size(avg),
                    #               np.size(scan_value), run, result_name)
                    scan_value = scan_value[0]

                if not base_num:
                      avg = scan_value
                else:
                    prev_avg = avg
                    new_avg = (base_num*prev_avg + scan_value) / (base_num + 1)
                    avg = new_avg

                    prev_var = var
                    new_var = ((base_num-1)*prev_var
                            + (scan_value-new_avg)*(scan_value-prev_avg)) \
                            * 1./base_num
                    var = new_var

                file_obj[subgroup + 'avg'][run] = avg
                file_obj[subgroup + 'var'][run] = var

        print '{} / {}'.format(run+1, runs)

    file_obj['Results'].attrs['base_num'] = base_num + 1
    file_obj['Results'].attrs['processed_files'] = np.append(processed_files,
                                                             new_scan)

    return file_obj

scan_nbr = 0
f = None
average_file_name = 'avg_scanned_params.h5'

if average_file_name not in listdir(print_path):
    f = _init_avg_file(name=average_file_name, init_file=listdir(data_path)[0])
    scan_nbr += .5

for filename in listdir(data_path)[0:2]:
    if filename.split('.')[-1] == 'h5':
        f = _update_avg_file(file_obj=f, name=average_file_name,
                             new_scan=filename)
        scan_nbr += 1

f.close()

print "--{} scans in {} minutes--".format(scan_nbr, (time() - start_time)/60.)

