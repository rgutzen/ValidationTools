import numpy as np
import matplotlib.pyplot as plt

def plot_raster_pophist_rate(ax, spiketrains_list, pop_hist_list, mean_rate_list,
                             binsize, num_neuron, layers_name, title, colors,
                             fsize):
    ax.set_title(title + ' microcircuit', fontsize=fsize * 3 / 2)
    num_sample = num_neuron
    sel_file_cnt = 0
    yticks_lst = []
    for pop_hist, mean_rate, spiketrains in zip(pop_hist_list[::-1], mean_rate_list[::-1],
                                                spiketrains_list[::-1]):
        tstop = spiketrains[0][0].t_stop.magnitude
        pop_hist_offset = 70
        pop_hist_scale = 3
        ms = 2
        for st_cnt in range(num_sample):
            try:
                ax.plot(spiketrains[0][st_cnt].magnitude,
                        [st_cnt + sel_file_cnt * (
                        2 * num_sample + pop_hist_offset) + \
                         pop_hist_offset + num_sample] * len(
                            spiketrains[0][st_cnt]), ',', color=colors[0], ms=ms)
                ax.plot(spiketrains[1][st_cnt].magnitude,
                        [st_cnt + sel_file_cnt * (
                         2 * num_sample + pop_hist_offset) +
                         pop_hist_offset] * len(spiketrains[1][st_cnt]), ',',
                        color=colors[1], ms=ms)
            except IndexError:
                pass
        lw = 0.8
        ax.plot(pop_hist[0].times.magnitude,
                pop_hist[0].magnitude * pop_hist_scale +
                sel_file_cnt * (2 * num_sample + pop_hist_offset), color=colors[0],
                lw=lw)
        ax.plot(pop_hist[1].times.magnitude,
                pop_hist[1].magnitude * pop_hist_scale + \
                sel_file_cnt * (2 * num_sample + pop_hist_offset), color=colors[1],
                lw=lw)
        scale_rate = 100
        offset_rate = 300
        ax.plot(np.array(mean_rate[0][:num_sample]) * scale_rate + tstop + offset_rate,
                np.arange(num_sample) + num_sample + sel_file_cnt * (
                2 * num_sample + pop_hist_offset) + pop_hist_offset,
                color=colors[0], lw=lw)
        ax.plot(np.array(mean_rate[1][:num_sample]) * scale_rate + tstop + offset_rate,
                np.arange(num_sample) + sel_file_cnt * (
                2 * num_sample + pop_hist_offset) + pop_hist_offset,
                color=colors[1], lw=lw)

        yticks_lst.append(sel_file_cnt * (
        2 * num_sample + pop_hist_offset) + num_sample + pop_hist_offset)
        #yticks_lst.append(sel_file_cnt * (
        #2 * num_sample + pop_hist_offset) + num_sample * 3 / 2 + pop_hist_offset)
        sel_file_cnt += 1
    ax.set_yticks(yticks_lst)
    ax.set_yticklabels(layers_name, fontsize=fsize * 2)
    
    ax.set_xticks(np.append(np.arange(0, tstop, 1000),
                            np.arange(tstop + offset_rate,
                                      tstop + offset_rate + 40 * scale_rate,
                                      10 * scale_rate)))
    ax.set_xticklabels(np.int32(np.append(np.arange(0, tstop, 1000) / 1000.,
                                          np.arange(0, offset_rate + 40, 10))), fontsize=12)
    ax.set_xlim(0, tstop + offset_rate + 30 * scale_rate)
    ax.set_ylim(0, sel_file_cnt * (
    2 * num_sample + pop_hist_offset) + num_sample)
    ax.set_xlabel('Time [s]', fontsize=14)
    ax.xaxis.set_label_coords(0.3, -0.08, transform=ax.transAxes)
    ax.text(0.75, -0.1, 'Firing rate [Hz]', transform=ax.transAxes, fontsize=14)


def plot_measure(NEST_list, SpiNNaker_list, bins_histogram, layers_name, title,
                 colors_NEST, colors_SPINNAKER, fsize, xlim, ylim, xlabel):
    yticks_lst = []
    num_row, num_col = len(layers_name), 2
    plt.subplots_adjust(top=.95, right=.85, left=.1, bottom=.1, hspace=0.3,
                        wspace=0.5)
    lw = 2
    for sim_cnt, simulator in enumerate(title):
        if sim_cnt == 0:
            lst = NEST_list
            clrs = colors_NEST
        else:
            lst = SpiNNaker_list
            clrs = colors_SPINNAKER
        for sel_file_cnt, measure in enumerate(lst):
            ax_tmp = plt.subplot2grid((num_row, num_col), (sel_file_cnt, sim_cnt),
                                      rowspan=1, colspan=1)
            if sel_file_cnt == 0:
                ax_tmp.set_title(title[sim_cnt], fontsize=fsize + 5)
            hist_tmp = (np.histogram(measure[0], bins_histogram, normed=1)[0] *
                        np.diff(bins_histogram)[0],
                        np.histogram(measure[1], bins_histogram, normed=1)[0] *
                        np.diff(bins_histogram)[0])
            ax_tmp.plot(bins_histogram[:-1], hist_tmp[0], color=clrs[0], lw=lw,
                        label='exc')
            ax_tmp.plot(bins_histogram[:-1], hist_tmp[1], color=clrs[1], lw=lw,
                        label='inh')
            ax_tmp.set_ylim(ylim)
            ax_tmp.set_xlim(xlim)
            if sim_cnt == 0:
                ax_tmp.text(-0.3, 0.5, layers_name[sel_file_cnt],
                            transform=ax_tmp.transAxes, fontsize=fsize)

        ax_tmp.set_xlabel(xlabel)


def plot_distribution_comparison(NEST_list, SpiNNaker_list, bins_histogram, layers_name, title,
            colors_NEST, colors_SPINNAKER, fsize, xlim, ylim, xlabel):
    num_row, num_col = len(layers_name), 2
    plt.subplots_adjust(top=.95, right=.85, left=.1, bottom=.1, hspace=0.3,
                        wspace=0.5)
    lw = 2
    hist_data = np.empty((2,4,2),dtype=(float,len(bins_histogram)-1)) #ToDo 2D Array of tuples
    for sim_cnt, simulator in enumerate(title):
        if sim_cnt == 0:
            lst = NEST_list
            clrs = colors_NEST
        else:
            lst = SpiNNaker_list
            clrs = colors_SPINNAKER
        for sel_file_cnt, measure in enumerate(lst):
            ax_tmp = plt.subplot2grid((num_row, num_col), (sel_file_cnt, sim_cnt),
                                      rowspan=1, colspan=1)
            if sel_file_cnt == 0:
                ax_tmp.set_title(title[sim_cnt], fontsize=fsize + 5)

            hist_tmp = (np.histogram(measure[0], bins_histogram, normed=1)[0] *
                        np.diff(bins_histogram)[0],
                        np.histogram(measure[1], bins_histogram, normed=1)[0] *
                        np.diff(bins_histogram)[0])
            ax_tmp.plot(bins_histogram[:-1], hist_tmp[0], color=clrs[0], lw=lw,
                        label='exc')
            ax_tmp.plot(bins_histogram[:-1], hist_tmp[1], color=clrs[1], lw=lw,
                        label='inh')
            ax_tmp.set_ylim(ylim)
            ax_tmp.set_xlim(xlim)
            ax_tmp.tick_params(axis='both', which='major', labelsize=12)
            if sim_cnt == 0:
                ax_tmp.text(-0.3, 0.5, layers_name[sel_file_cnt],
                            transform=ax_tmp.transAxes, fontsize=fsize)
                # ax_tmp.set_yticks([0,0.1,0.2])
                # ax_tmp.set_xticks([0,0.5,1,1.5])

            hist_data[sim_cnt][sel_file_cnt][0] = np.asarray(hist_tmp[0])
            hist_data[sim_cnt][sel_file_cnt][1] = np.asarray(hist_tmp[1])

        ax_tmp.set_xlabel(xlabel, fontsize=14)
    return hist_data
