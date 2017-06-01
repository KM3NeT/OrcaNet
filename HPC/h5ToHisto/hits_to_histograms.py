#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This utility code contains functions that computes 2D/3D histograms based on the file_to_hits.py output"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

import glob


def compute_4d_to_2d_histograms(event_hits, x_bin_edges, y_bin_edges, z_bin_edges, all_4d_to_2d_hists, event_track, do2d_pdf):
    """
    Computes 2D numpy histogram 'images' from the 4D data.
    :param ndarray(ndim=2) event_hits: 2D array that contains the hits (_xyz) data for a certain eventID. [event_id, positions_xyz, time, dom_id]
    :param ndarray(ndim=1) x_bin_edges: bin edges for the X-direction.
    :param ndarray(ndim=1) y_bin_edges: bin edges for the Y-direction.
    :param ndarray(ndim=1) z_bin_edges: bin edges for the Z-direction.
    :param list all_4d_to_2d_hists: contains all 2D histogram projections.
    :param ndarray(ndim=2) event_track: contains the relevant mc_track info for the event in order to get a nice title for the pdf histos.
    :param bool do2d_pdf: if True, generate 2D matplotlib pdf histograms.
    :return: appends the 2D histograms to the all_4d_to_2d_hists list.
    """
    # slice out the coordinates of the current hits
    x = np.array(event_hits[:, 1], np.float32)
    y = np.array(event_hits[:, 2], np.float32)
    z = np.array(event_hits[:, 3], np.float32)

    # create histograms for this event
    hist_xy = np.histogram2d(y, x, bins=(y_bin_edges, x_bin_edges))  # already transposed for later .pdf file, hist[0] = H, hist[1] = yedges, hist[2] = xedges
    hist_xz = np.histogram2d(z, x, bins=(z_bin_edges, x_bin_edges))
    hist_yz = np.histogram2d(z, y, bins=(z_bin_edges, y_bin_edges))

    # transpose back to get classic numpy convention again: x along first dim (vertical), y along second dim (horizontal)
    all_4d_to_2d_hists.append([hist_xy[0].T, hist_xz[0].T, hist_yz[0].T])

    if do2d_pdf:
        convert_2d_numpy_hists_to_pdf_image(hist_xy, hist_xz, hist_yz, event_track=event_track) # slow! takes about 1s per event


def convert_2d_numpy_hists_to_pdf_image(hist_xy, hist_xz, hist_yz, event_track=None):
    """
    Creates matplotlib 2D histos based on the numpy histogram2D objects and saves them to a pdf file.
    :param ndarray(ndim=2) hist_xy: x-y np.histogram2d 
    :param ndarray(ndim=2) hist_xz: x-z np.histogram2d
    :param ndarray(ndim=2) hist_yz: y-z np.histogram2d
    :param ndarray(ndim=2) event_track: contains the relevant mc_track info for the event in order to get a nice title for the pdf histos. [event_id, particle_type, energy, isCC]
    """
    fig = plt.figure(figsize=(10, 10))
    if event_track is not None:
        particle_type = {16: 'Tau', -16: 'Anti-Tau', 14: 'Muon', -14: 'Anti-Muon', 12: 'Electron', -12: 'Anti-Electron', 'isCC': ['NC', 'CC']}
        event_info = {'event_id': str(int(event_track[0])), 'energy': str(event_track[2]),
                      'particle_type': particle_type[int(event_track[1])], 'interaction_type': particle_type['isCC'][int(event_track[3])]}
        title = event_info['particle_type'] + '-' + event_info['interaction_type'] + ', Event ID: ' + event_info['event_id'] + ', Energy: ' + event_info['energy'] + ' GeV'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        fig.suptitle(title, usetex=False, horizontalalignment='center', size='xx-large', bbox=props)

    axes_xy = plt.subplot2grid((2, 2), (0, 0), title='XY - projection', xlabel='X Position [m]', ylabel='Y Position [m]', aspect='equal', xlim=(-175, 175), ylim=(-175, 175))
    axes_xz = plt.subplot2grid((2, 2), (0, 1), title='XZ - projection', xlabel='X Position [m]', ylabel='Z Position [m]', aspect='equal', xlim=(-175, 175), ylim=(-57.8, 292.2))
    axes_yz = plt.subplot2grid((2, 2), (1, 0), title='YZ - projection', xlabel='Y Position [m]', ylabel='Z Position [m]', aspect='equal', xlim=(-175, 175), ylim=(-57.8, 292.2))

    def fill_subplot(hist_ab, axes_ab):
        # Mask hist_ab
        h_ab_masked = np.ma.masked_where(hist_ab[0] == 0, hist_ab[0])

        a, b = np.meshgrid(hist_ab[2], hist_ab[1])
        plot_ab = axes_ab.pcolormesh(a, b, h_ab_masked)

        the_divider = make_axes_locatable(axes_ab)
        color_axis = the_divider.append_axes("right", size="5%", pad=0.1)

        # add color bar
        cbar_ab = plt.colorbar(plot_ab, cax=color_axis, ax=axes_ab)
        cbar_ab.ax.set_ylabel('Hits [#]')

        return plot_ab

    plot_xy = fill_subplot(hist_xy, axes_xy)
    plot_xz = fill_subplot(hist_xz, axes_xz)
    plot_yz = fill_subplot(hist_yz, axes_yz)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    glob.pdf_2d_plots.savefig(fig) #TODO: remove global variable, but how? Need to close pdf object outside of this function (-> as last step of the 2D eventID loop)
    plt.close()

# deprecated
def store_2d_hist_as_pgm(hist, filename):
    # deprecated
    # BUG!!! SAME PGM FOR ALL EVENTIDs

    pgm_file = open(filename, 'w')
    max_hist_value = np.amax(hist[0])

    # write a valid header for a pgm image file
    pgm_file.write("P2\n" + str(hist[0].shape[1]) + " " + str(hist[0].shape[0]) + "\n" + str(int(max_hist_value)) + "\n")
    # write the actual data
    for row in hist[0]:
        for entry in row:
            # write the actual values
            pgm_file.write(str(int(entry)) + " ")

    pgm_file.write("\n")
    pgm_file.close()


def compute_4d_to_3d_histograms(event_hits, x_bin_edges, y_bin_edges, z_bin_edges, all_4d_to_3d_hists):
    """
    Computes 3D numpy histogram 'images' from the 4D data.
    :param ndarray(ndim=2) event_hits: 2D array that contains the hits (_xyz) data for a certain eventID. [event_id, positions_xyz, time, dom_id]
    :param ndarray(ndim=1) x_bin_edges: bin edges for the X-direction. 
    :param ndarray(ndim=1) y_bin_edges: bin edges for the Y-direction.
    :param ndarray(ndim=1) z_bin_edges: bin edges for the Z-direction. 
    :param list all_4d_to_3d_hists: contains all 3D histogram projections.
    :return: appends the 3D histograms to the all_4d_to_3d_hists list.
    """
    hist_xyz = np.histogramdd(np.array(event_hits[:, 1:4], np.float32), bins=(x_bin_edges, y_bin_edges, z_bin_edges))

    all_4d_to_3d_hists.append([hist_xyz[0]])