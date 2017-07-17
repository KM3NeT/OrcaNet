#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This utility code contains functions that computes 2D/3D histograms based on the file_to_hits.py output"""

import matplotlib.pyplot as plt
import numpy as np
#import line_profiler # call with kernprof file.py args
from mpl_toolkits.axes_grid1 import make_axes_locatable

import glob


def get_time_parameters(t, t_start_margin=0.15, t_end_margin=0.15):
    """
    Gets the fundamental time parameters in one place (this function).
    Later on these parameters cut out a certain time span of events specified by t_start and t_end.
    Implemented just for convenience (less lines of code), since it's both used in the 2D and 3D function.
    :param ndarray(ndim=1) t: time column of the event_hits array
    :param float t_start_margin: defines the start time of the selected timespan with t_mean - t_start * t_diff
    :param float t_end_margin: defines the end time of the selected timespan with t_mean + t_start * t_diff
    :return: float t_start, t_end: absolute start and end time that will be used for the later timespan cut.
                                   Events in this timespan are accepted, others are rejected.
    """
    t_min = np.amin(t)
    t_max = np.amax(t)
    t_diff = t_max - t_min
    t_mean = t_min + 0.5 * t_diff

    t_start = t_mean - t_start_margin * t_diff
    t_end = t_mean + t_end_margin * t_diff

    return t_start, t_end
    #return t_min, t_max


def compute_4d_to_2d_histograms(event_hits, x_bin_edges, y_bin_edges, z_bin_edges, n_bins, all_4d_to_2d_hists, event_track, do2d_pdf):
    """
    Computes 2D numpy histogram 'images' from the 4D data.
    Careful: Currently, appending to all_4d_to_2d_hists takes quite a lot of memory (about 200MB for 3500 events).
    In the future, the list should be changed to a numpy ndarray.
    (Which unfortunately would make the code less readable, since an array is needed for each projection...)
    :param ndarray(ndim=2) event_hits: 2D array that contains the hits (_xyz) data for a certain eventID. [event_id, positions_xyz, time, dom_id]
    :param ndarray(ndim=1) x_bin_edges: bin edges for the X-direction.
    :param ndarray(ndim=1) y_bin_edges: bin edges for the Y-direction.
    :param ndarray(ndim=1) z_bin_edges: bin edges for the Z-direction.
    :param tuple n_bins: Contains the number of bins that should be used for each dimension (x,y,z,t).
    :param list all_4d_to_2d_hists: contains all 2D histogram projections.
    :param ndarray(ndim=2) event_track: contains the relevant mc_track info for the event in order to get a nice title for the pdf histos.
    :param bool do2d_pdf: if True, generate 2D matplotlib pdf histograms.
    :return: appends the 2D histograms to the all_4d_to_2d_hists list.
    """
    # slice out the coordinates of the current hits
    x = np.array(event_hits[:, 0], np.float32)
    y = np.array(event_hits[:, 1], np.float32)
    z = np.array(event_hits[:, 2], np.float32)
    t = np.array(event_hits[:, 3], np.float32)

    # analyze time
    t_start, t_end = get_time_parameters(t, t_start_margin=0.15, t_end_margin=0.15)

    # create histograms for this event
    hist_xy = np.histogram2d(x, y, bins=(x_bin_edges, y_bin_edges))  # hist[0] = H, hist[1] = xedges, hist[2] = yedges
    hist_xz = np.histogram2d(x, z, bins=(x_bin_edges, z_bin_edges))
    hist_yz = np.histogram2d(y, z, bins=(y_bin_edges, z_bin_edges))

    #hist_xt = np.histogram2d(x, t, bins=(n_bins[0], n_bins[3]))
    hist_xt = np.histogram2d(x, t, bins=(x_bin_edges, n_bins[3]), range=((min(x_bin_edges), max(x_bin_edges)), (t_start, t_end)))
    hist_yt = np.histogram2d(y, t, bins=(y_bin_edges, n_bins[3]), range=((min(y_bin_edges), max(y_bin_edges)), (t_start, t_end)))
    hist_zt = np.histogram2d(z, t, bins=(z_bin_edges, n_bins[3]), range=((min(z_bin_edges), max(z_bin_edges)), (t_start, t_end)))

    # Format in classical numpy convention: x along first dim (vertical), y along second dim (horizontal)
    all_4d_to_2d_hists.append((np.array(hist_xy[0], dtype=np.uint8),
                               np.array(hist_xz[0], dtype=np.uint8),
                               np.array(hist_yz[0], dtype=np.uint8),
                               np.array(hist_xt[0], dtype=np.uint8),
                               np.array(hist_yt[0], dtype=np.uint8),
                               np.array(hist_zt[0], dtype=np.uint8)))

    if do2d_pdf:
        # Format in classical numpy convention: x along first dim (vertical), y along second dim (horizontal)
        # Need to take that into account in convert_2d_numpy_hists_to_pdf_image()
        # transpose to get typical cartesian convention: y along first dim (vertical), x along second dim (horizontal)
        hists = [hist_xy, hist_xz, hist_yz, hist_xt, hist_yt, hist_zt]
        convert_2d_numpy_hists_to_pdf_image(hists, t_start, t_end, event_track=event_track) # slow! takes about 1s per event


def convert_2d_numpy_hists_to_pdf_image(hists, t_start, t_end, event_track=None):
    """
    Creates matplotlib 2D histos based on the numpy histogram2D objects and saves them to a pdf file.
    :param list(ndarray(ndim=2)) hists: Contains np.histogram2d objects of all projections [xy, xz, yz, xt, yt, zt].
    :param float t_start: absolute start time of the timespan cut.
    :param float t_end: absolute end time of the timespan cut.
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

    t_diff = t_end - t_start

    axes_xy = plt.subplot2grid((3, 2), (0, 0), title='XY - projection', xlabel='X Position [m]', ylabel='Y Position [m]', aspect='equal', xlim=(-175, 175), ylim=(-175, 175))
    axes_xz = plt.subplot2grid((3, 2), (0, 1), title='XZ - projection', xlabel='X Position [m]', ylabel='Z Position [m]', aspect='equal', xlim=(-175, 175), ylim=(-57.8, 292.2))
    axes_yz = plt.subplot2grid((3, 2), (1, 0), title='YZ - projection', xlabel='Y Position [m]', ylabel='Z Position [m]', aspect='equal', xlim=(-175, 175), ylim=(-57.8, 292.2))

    axes_xt = plt.subplot2grid((3, 2), (1, 1), title='XT - projection', xlabel='X Position [m]', ylabel='Time [ns]', aspect='auto',
                               xlim=(-175, 175), ylim=(t_start - 0.1*t_diff, t_end + 0.1*t_diff))
    axes_yt = plt.subplot2grid((3, 2), (2, 0), title='YT - projection', xlabel='Y Position [m]', ylabel='Time [ns]', aspect='auto',
                               xlim=(-175, 175), ylim=(t_start - 0.1*t_diff, t_end + 0.1*t_diff))
    axes_zt = plt.subplot2grid((3, 2), (2, 1), title='ZT - projection', xlabel='Z Position [m]', ylabel='Time [ns]', aspect='auto',
                               xlim=(-57.8, 292.2), ylim=(t_start - 0.1*t_diff, t_end + 0.1*t_diff))

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

    plot_xy = fill_subplot(hists[0], axes_xy)
    plot_xz = fill_subplot(hists[1], axes_xz)
    plot_yz = fill_subplot(hists[2], axes_yz)
    plot_xt = fill_subplot(hists[3], axes_xt)
    plot_yt = fill_subplot(hists[4], axes_yt)
    plot_zt = fill_subplot(hists[5], axes_zt)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    glob.pdf_2d_plots.savefig(fig) #TODO: remove global variable, but how? Need to close pdf object outside of this function (-> as last step of the 2D eventID loop)
    plt.close()


def compute_4d_to_3d_histograms(event_hits, x_bin_edges, y_bin_edges, z_bin_edges, n_bins, all_4d_to_3d_hists):
    """
    Computes 3D numpy histogram 'images' from the 4D data.
    Careful: Currently, appending to all_4d_to_3d_hists takes quite a lot of memory (about 200MB for 3500 events).
    In the future, the list should be changed to a numpy ndarray.
    (Which unfortunately would make the code less readable, since an array is needed for each projection...)
    :param ndarray(ndim=2) event_hits: 2D array that contains the hits (_xyz) data for a certain eventID. [event_id, positions_xyz, time, dom_id]
    :param ndarray(ndim=1) x_bin_edges: bin edges for the X-direction. 
    :param ndarray(ndim=1) y_bin_edges: bin edges for the Y-direction.
    :param ndarray(ndim=1) z_bin_edges: bin edges for the Z-direction.
    :param tuple n_bins: Declares the number of bins that should be used for each dimension (x,y,z,t).
    :param list all_4d_to_3d_hists: contains all 3D histogram projections.
    :return: appends the 3D histograms to the all_4d_to_3d_hists list. [xyz, xyt, xzt, yzt, rzt]
    """
    x = event_hits[:, 0:1]
    y = event_hits[:, 1:2]
    z = event_hits[:, 2:3]
    t = event_hits[:, 3:4]

    t_start, t_end = get_time_parameters(t, t_start_margin=0.15, t_end_margin=0.15)

    hist_xyz = np.histogramdd(np.array(event_hits[:, 0:3], np.float32), bins=(x_bin_edges, y_bin_edges, z_bin_edges))

    hist_xyt = np.histogramdd(np.array(np.concatenate([x, y, t], axis=1), np.float32), bins=(x_bin_edges, y_bin_edges, n_bins[3]),
                              range=((min(x_bin_edges), max(x_bin_edges)), (np.amin(y), np.amax(y)), (t_start, t_end)))
    hist_xzt = np.histogramdd(np.array(np.concatenate([x, z, t], axis=1), np.float32), bins=(x_bin_edges, z_bin_edges, n_bins[3]),
                              range=((min(x_bin_edges), max(x_bin_edges)), (min(z_bin_edges), max(z_bin_edges)), (t_start, t_end)))
    hist_yzt = np.histogramdd(np.array(event_hits[:, 1:4], np.float32), bins=(y_bin_edges, z_bin_edges, n_bins[3]),
                              range=((min(y_bin_edges), max(y_bin_edges)), (min(z_bin_edges), max(z_bin_edges)), (t_start, t_end)))

    # add a rotation-symmetric 3d hist
    x = np.array(event_hits[:, 0:1], np.float32)
    y = np.array(event_hits[:, 1:2], np.float32)
    r = np.sqrt(x * x + y * y)
    zt = np.array(event_hits[:, 2:4], np.float32)
    rzt = np.array(np.concatenate([r, zt], axis=1), np.float32)
    hist_rzt = np.histogramdd(rzt, bins=(n_bins[0], n_bins[2], n_bins[3]), range=((np.amin(r), np.amax(r)), (np.amin(z), np.amax(z)), (t_start, t_end)))

    all_4d_to_3d_hists.append((np.array(hist_xyz[0], dtype=np.uint8),
                               np.array(hist_xyt[0], dtype=np.uint8),
                               np.array(hist_xzt[0], dtype=np.uint8),
                               np.array(hist_yzt[0], dtype=np.uint8),
                               np.array(hist_rzt[0], dtype=np.uint8)))