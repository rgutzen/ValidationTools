from neo.io import gdfio
import neo
import os
import quantities as pq


def conv_dat(fn, start, stop, n):
    g = gdfio.GdfIO(fn)
    seg = g.read_segment(gdf_id_list=range(n), id_column=1, time_column=0,
                         t_start=start * pq.ms, t_stop=stop * pq.ms)
    # filelist = [f for f in os.listdir(".") if f.endswith(".h5")]
    # for f in filelist:
    #    os.remove(f)
    bl = neo.Block()
    bl.segments.append(seg)
    bl.create_relationship()
    output_fn = os.path.splitext(fn)[0] + '.h5'
    output = neo.io.NeoHdf5IO(output_fn)
    output.write(bl)
    output.close()


layers = ['23', '4', '5', '6']
N = 100
filelist = [f for f in os.listdir(".") if f.endswith(".h5")]
for f in filelist:
    os.remove(f)
for l in layers:
    filename_E = 'spikes_L{}{}.dat'.format(l, 'E')
    filename_I = 'spikes_L{}{}.dat'.format(l, 'I')
    conv_dat(filename_E, 0, 10000, N)
    conv_dat(filename_I, 0, 10000, N)
