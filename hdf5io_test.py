
import neo


# Three working path description:

print neo.io.NeoHdf5IO('spikes_L4E.h5')
print neo.io.NeoHdf5IO('./results/NEST_data/spikes_L4E.h5')

COLLAB_PATH = '/home/robin/INM6/Projects/NEST_SpiNNaker_comparison_INM6'
COLLAB_PATH_NEST = COLLAB_PATH + '/results/NEST_data'
print neo.io.NeoHdf5IO(COLLAB_PATH_NEST + '/spikes_L4E.h5')




