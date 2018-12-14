#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exports 2D histograms ('images') of the CNN .h5 input file to a pdf. Only for XZ till now.
Used for bug hunting. Comment: test passed, concatenating and shuffling is not bugged."""

import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

def convert_2d_numpy_hists_to_pdf_image(hist, pdf_2d_plots, event_track=None):
    """
    Creates matplotlib 2D histos based on the numpy histogram2D objects and saves them to a pdf file.
    :param list(ndarray(ndim=2)) hists: Contains np.histogram2d objects of all projections [xy, xz, yz, xt, yt, zt].
    :param ndarray(ndim=2) event_track: contains the relevant mc_track info for the event in order to get a nice title for the pdf histos.
                                        [event_id, particle_type, energy, isCC, bjorkeny, dir_x/y/z]
    """
    fig = plt.figure(figsize=(10, 13))
    if event_track is not None:
        particle_type = {16: 'Tau', -16: 'Anti-Tau', 14: 'Muon', -14: 'Anti-Muon', 12: 'Electron', -12: 'Anti-Electron', 'isCC': ['NC', 'CC']}
        event_info = {'event_id': str(int(event_track[0])), 'energy': str(event_track[2]),
                      'particle_type': particle_type[int(event_track[1])], 'interaction_type': particle_type['isCC'][int(event_track[3])]}
        title = event_info['particle_type'] + '-' + event_info['interaction_type'] + ', Event ID: ' + event_info['event_id'] + ', Energy: ' + event_info['energy'] + ' GeV'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        fig.suptitle(title, usetex=False, horizontalalignment='center', size='xx-large', bbox=props)

    axes_xz = plt.subplot2grid((3, 2), (0, 1), title='XZ - projection', xlabel='X Position [m]', ylabel='Z Position [m]', aspect='equal', xlim=(-175, 175), ylim=(-57.8, 292.2))

    def fill_subplot(hist_ab, axes_ab):
        # Mask hist_ab
        h_ab_masked = np.ma.masked_where(hist_ab[0] == 0, hist_ab[0])

        a, b = np.meshgrid(hist_ab[1], hist_ab[2]) #2,1
        plot_ab = axes_ab.pcolormesh(a, b, h_ab_masked.T)

        the_divider = make_axes_locatable(axes_ab)
        color_axis = the_divider.append_axes("right", size="5%", pad=0.1)

        # add color bar
        cbar_ab = plt.colorbar(plot_ab, cax=color_axis, ax=axes_ab)
        cbar_ab.ax.set_ylabel('Hits [#]')

        return plot_ab

    plot_xz = fill_subplot(hist, axes_xz)

    #fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    pdf_2d_plots.savefig(fig)
    plt.close()


def export_hist_2d_to_pdf():
    filepath_input = 'muon-CC_and_elec-CC_9_xz_shuffled.h5'
    pdf_2d_plots = PdfPages('export_2d_images_pdfs/' + filepath_input + '_plots.pdf')
    f = h5py.File(filepath_input, 'r')
    hists = f['x']
    mc_info = f['y']

    for i in range(hists.shape[0]):
        event_id = mc_info[i][0]
        if event_id > 25:
            print(i)
            continue
        x_bin_edges = [-118.551, -96.82727272727273, -75.10354545454545, -53.37981818181818, -31.656090909090906, -9.932363636363633,
                       11.791363636363641, 33.515090909090915, 55.23881818181819, 76.96254545454546, 98.68627272727274, 120.41]
        z_bin_edges = [33.235, 42.56444444444445, 51.89388888888889, 61.22333333333333, 70.55277777777778, 79.88222222222223, 89.21166666666667,
                       98.5411111111111, 107.87055555555555, 117.2, 126.52944444444445, 135.85888888888888, 145.18833333333333, 154.51777777777778,
                       163.84722222222223, 173.17666666666668, 182.50611111111112, 191.83555555555557, 201.165]
        hists_h2d = [hists[i], np.array(x_bin_edges, dtype=np.float64), np.array(z_bin_edges, dtype=np.float64)]
        convert_2d_numpy_hists_to_pdf_image(hists_h2d, pdf_2d_plots, event_track=mc_info[i])

    pdf_2d_plots.close()


if __name__ == '__main__':
    export_hist_2d_to_pdf()
