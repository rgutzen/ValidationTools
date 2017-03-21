from neo.io.hdf5io import NeoHdf5IO
from neo import SpikeTrain


def load(client, path, file_name_list):
    """
    Loads spiketrains from a hdf5 file in the neo data format.

    :param client
        Get the client in the collab with get_bbp_client().document
    :param path: string
        Path to file
    :param file_name_list: list
        List of hdf5 filenames
    :param N:
        Number of returned spiketrains. When less are found in the file empty
        spiketrains are added; when more are found only the first N are
        returned
    :return:
        N   List of N tuples of neo.Spiketrains
    """
    # Load NEST or SpiNNaker data using NeoHdf5IO
    spike_train_list = []
    for file_name in file_name_list:
        # exc. and inh. as tuples, layerwise
        file_path_exc = file_name + 'E.h5'
        file_path_inh = file_name + 'I.h5'
        store_path = './' + path.split('/')[-1] + '_'
        client.download_file(path + '/' + file_path_exc,
                             store_path + file_path_exc)
        client.download_file(path + '/' + file_path_inh,
                             store_path + file_path_inh)
        data = (NeoHdf5IO(store_path + file_path_exc),
                NeoHdf5IO(store_path + file_path_inh))
        spiketrains = [data[0].read_block().list_children_by_class(SpikeTrain),
                       data[1].read_block().list_children_by_class(SpikeTrain)]
        for st_count in range(len(spiketrains[0])):
            spiketrains[0][st_count].annotations['type'] = 'exc'
            spiketrains[0][st_count].annotations['layer'] = \
                file_name.split('_')[-1]
        for st_count in range(len(spiketrains[1])):
            spiketrains[1][st_count].annotations['type'] = 'inh'
            spiketrains[1][st_count].annotations['layer'] = \
                file_name.split('_')[-1]

        spike_train_list += [spiketrains]

        print path.split('/')[-1] + " " + file_name + " ... loaded"

    return spike_train_list
