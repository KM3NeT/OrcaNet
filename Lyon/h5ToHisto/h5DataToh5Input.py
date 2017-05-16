#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This code takes raw simulated .hdf5 files as input in order to generate 2D/3D histograms ('images') that can be used for CNNs."""

import sys
import pandas as pd
import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable

__author__ = 'Michael Moser'
__license__ = 'AGPL'
__version__= '0.1'
__email__ = 'michael.m.moser@fau.de'
__status__ = 'Prototype'
# Heavily based on code from sgeisselsoeder: https://github.com/sgeisselsoeder/km3netHdf5ToHistograms/


def parse_file(fname, fname_geo, do_mc_hits):
    """
    Reads the raw .hdf5 neutrino MC file and returns the hit arrays (event_id [pos_xyz] dom_id time).
    :param str fname: filepath of parsed inputfile.
    :param str fname_geo: filepath of used ORCA geometry file.
    :param bool do_mc_hits: tells the function of the hits (mc_hits + BG) or the mc_hits only should be parsed. 
                            In the case of mc_hits, the dom_id needs to be calculated thanks to the jpp output.
    :return: ndarray(ndim=2) tracks: 2D array containing important MC information for each event_id. [event_id, particle_type, energy, isCC]
    :return: ndarray(ndim=2) hits, hits_xyz: 2D arrays containing (event_id [pos_xyz] dom_id time).
    :return (ndarray(ndim=1), ndarray(ndim=1)) geo_limits: tuple that contains the min and max geometry values for each dimension. 
    ([first_OM_id, xmin, ymin, zmin], [last_OM_id, xmax, ymax, zmax])
    """

    print "Extracting hits from hdf5 file " + fname
    print "Reading detector geometry from file " + fname_geo
    geo = np.loadtxt(fname_geo)

    # derive maximum and minimum x,y,z coordinates of the geometry input [[first_OM_id, xmin, ymin, zmin], [last_OM_id, xmax, ymax, zmax]]
    geo_limits = np.nanmin(geo, axis = 0), np.nanmax(geo, axis = 0)

    print "Reading tracks"
    tracks_full = np.array(pd.read_hdf(fname, 'mc_tracks'))
    print "Filtering primary tracks"
    tracks_primary = tracks_full[np.where(tracks_full[:, 0] != 0.0)[0]]
    # keep the relevant info from the track: event_id particle_type energy isCC
    tracks = extract_relevant_track_info(tracks_primary)

    print "Reading triggered hits"
    if do_mc_hits is True:
        hits_group = np.array(pd.read_hdf(fname, 'mc_hits'))
        mc_hits_get_dom_id(hits_group)
    else:
        hits_group = np.array(pd.read_hdf(fname, 'hits'))

    # keep the relevant info from each hit: event_id dom_id time
    hits = np.array(np.concatenate([hits_group[:, 14:15], hits_group[:, 4:5], hits_group[:, 11:12]], axis=1), np.float32)

    print "Converting hits omid -> XYZ"
    hits_xyz = convert_hits_xyz(hits, geo)

    print "Done converting."
    return tracks, hits, hits_xyz, geo_limits


def mc_hits_get_dom_id(hits_group):
    """
    This function calculates the dom_id of each mc_hit event based on their pmt_id's. 
    After this, the appropriate dom_id is inserted into the hits_group 2D array for each event.
    pmt_id = (dom_id - 1) * 31 + channel_id + 1
    dom_id = (pmt_id-1)/31 + 1
    :param ndarray(ndim=2) hits_group: 2D arrays that contains the full mc_hit information.
    """
    for hit in hits_group:

        pmt_id = int(hit[6])
        dom_id = int((pmt_id-int(1))/int(31)) + 1
        hit[4] = dom_id


def extract_relevant_track_info(tracks):
    """
    Returns the relevant MC information for all tracks. [event_id, particle_type, energy, isCC]
    :param ndarray(ndim=2) tracks: 2D array of the primary mc_tracks info.
    :return: ndarray(ndim=2): returns a 2D array with the relevant mc_tracks info for each event.
    """
    # keep the relevant info from the track: event_id particle_type energy isCC
    return np.array(np.concatenate([tracks[:, 14:15], tracks[:, 13:14], tracks[:, 4:5], tracks[:, 7:8]], axis=1), np.float32)


def convert_hits_xyz(hits, geo):
    """
    Reads the hits array with dom_id's and returns the hits_xyz array with according xyz positions.
    :param ndarray(ndim=2) hits: 2D hits array that contain event_id dom_id time.
    :param  ndarray(ndim=2) geo: 2D geo array that contains the xyz position for each dom_id.
    :return: ndarray(ndim=2) hits_xyz: 2D hits array with xyz position information.
    """
    # write the hits with xyz geometry
    hits_xyz_list = []
    for hit in hits:
            position = geo[int(hit[1])-1]
            # event_id positions_xyz dom_id time
            hits_xyz_list.append([int(hit[0]), position[1], position[2], position[3], int(hit[1]), hit[2]])
    return np.array(hits_xyz_list)


def calculate_bin_edges(n_bins, geo_limits):
    """
    Calculates the bin edges for the later np.histogramdd actions based on the number of specified bins. 
    This is performed in order to get the same bin size for each event regardless of the fact if all bins have a hit or not.
    :param list n_bins: contains the desired number of bins for each dimension. [n_binsx, n_binsy, nbins_z]
    :param geo_limits: contains the min and max values of each geometry dimension. [[first_OM_id, xmin, ymin, zmin], [last_OM_id, xmax, ymax, zmax]]
    :return: ndarray(ndim=1) x_bin_edges, y_bin_edges, z_bin_edges: contains the resulting bin edges for each dimension.
    """
    n_binsx, n_binsy, n_binsz = n_bins[0], n_bins[1], n_bins[2]  # number of bins in x,y,z

    # geometry input [[first_OM_id, xmin, ymin, zmin], [last_OM_id, xmax, ymax, zmax]]
    x_bin_edges = np.linspace(geo_limits[0][1], geo_limits[1][1], num=n_binsx + 1)
    y_bin_edges = np.linspace(geo_limits[0][2], geo_limits[1][2], num=n_binsy + 1)
    z_bin_edges = np.linspace(geo_limits[0][3], geo_limits[1][3], num=n_binsz + 1)

    return x_bin_edges, y_bin_edges, z_bin_edges

# Output related stuff
def compute_4d_to_2d_histograms(event_hits, x_bin_edges, y_bin_edges, z_bin_edges, all_4d_to_2d_hists, event_track, do2d_pdf):
    """
    Computes 2D numpy histogram 'images' from the 4D data.
    :param ndarray(ndim=2) event_hits: 2D array that contains the hits (_xyz) data for a certain eventID. (event_id positions_xyz time dom_id)
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
    :param ndarray(ndim=2) event_track: contains the relevant mc_track info for the event in order to get a nice title for the pdf histos.
    """
    # Create human viewable 2D matplotlib output

    fig = plt.figure(figsize=(10, 10))
    if event_track is not None:
        # event_id particle_type energy isCC
        particle_type = {16: 'Tau', -16: 'Anti-Tau', 14: 'Muon', -14: 'Anti-Muon', 12: 'Electron', -16: 'Anti-Electron', 'isCC': ['NC', 'CC']}
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
    pdf_2d_plots.savefig(fig)
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
    :param ndarray(ndim=2) event_hits: 2D array that contains the hits (_xyz) data for a certain eventID. (event_id positions_xyz time dom_id)
    :param ndarray(ndim=1) x_bin_edges: bin edges for the X-direction. 
    :param ndarray(ndim=1) y_bin_edges: bin edges for the Y-direction.
    :param ndarray(ndim=1) z_bin_edges: bin edges for the Z-direction. 
    :param list all_4d_to_3d_hists: contains all 3D histogram projections.
    :return: appends the 3D histograms to the all_4d_to_3d_hists list.
    """
    # Correct order? 2D was y,x for xy hist
    hist_xyz = np.histogramdd(np.array(event_hits[:, 1:4], np.float32), bins=(x_bin_edges, y_bin_edges, z_bin_edges))

    all_4d_to_3d_hists.append([hist_xyz[0]])

def store_3d_histogram_as_hdf5(all_4d_to_3d_hists, tracks, filename_output):
    """
    Takes the 3D np histograms ('images') as well as the mc_info ('tracks') and saves them to a h5 file. 
    :param list all_4d_to_3d_hists: contains all 3D np histograms (ndarrays(ndim=2))
    :param ndarray(ndim=2) tracks: 2D array containing important MC information for each event_id. [event_id, particle_type, energy, isCC]
    :param filename_output: placeholder filename till production
    """

    hists_3D_xyz = np.array(all_4d_to_3d_hists)[:, 0]

    h5f = h5py.File('Results/4dTo3d/h5/' + filename_output + '_xyz.h5', 'w')
    dset_tracks = h5f.create_dataset('tracks', data=tracks)
    dset_xyz = h5f.create_dataset('xyz', data=hists_3D_xyz)

    h5f.close()

def main(n_bins=list(), do2d=True, do2d_pdf=True, do3d=True, do_mc_hits=False):
    """
    Main code. Reads raw .hdf5 files and creates 2D/3D histogram projections that can be used for a CNN
    :param list n_bins: Declares the number of bins that should be used for each dimension (x,y,z).
    :param bool do2d: Declares if 2D histograms should be created.
    :param bool do2d_pdf: Declares if pdf visualizations of the 2D histograms should be created. Cannot be called if do2d=False.
    :param bool do3d: Declares if 3D histograms should be created.
    :param bool do_mc_hits: Declares if hits (False, mc_hits + BG) or mc_hits (False) should be processed
    """
    if do2d==False and do2d_pdf==True:
        print 'The 2D pdf images cannot be created if do2d==False. Please try again.'
        sys.exit()

    do_mc_hits = do_mc_hits

    do2d = do2d
    do3d = do3d
    do2d_pdf = do2d_pdf

    filename_input = '/sps/km3net/users/mmoser/Data/ORCA_JTE_NEMOWATER/hdf5/muon-CC/3-100GeV/' \
                     'JTE.KM3Sim.gseagen.muon-CC.3-100GeV-9.1E7-1bin-3.0gspec.ORCA115_9m_2016.99.hdf5'
    filename_output = 'test'
    filename_geometry = 'ORCA_Geo_115lines.txt'

    tracks, hits, hits_xyz, geo_limits = parse_file(filename_input, filename_geometry, do_mc_hits)
    all_event_numbers = set(hits[:, 0])

    x_bin_edges, y_bin_edges, z_bin_edges = calculate_bin_edges(n_bins, geo_limits)

    all_4d_to_2d_hists = []
    all_4d_to_3d_hists = []

    print "Generating histograms from the hits in XYZT format for files based on " + filename_input
    if do2d_pdf:
        global pdf_2d_plots
        pdf_2d_plots = PdfPages('Results/4dTo2d/' + filename_output + '_plots.pdf')

    i=0
    for eventID in all_event_numbers:
        print i
        i+=1
        # filter all hits belonging to this event
        event_hits = hits_xyz[np.where(hits_xyz[:, 0] == eventID)[0]]
        event_track = tracks[np.where(tracks[:, 0] == eventID)[0]][0]

        if do2d:
            # compute the 2D histograms
            compute_4d_to_2d_histograms(event_hits, x_bin_edges, y_bin_edges, z_bin_edges, all_4d_to_2d_hists, event_track, do2d_pdf)
            #store_2d_hist_as_pgm(all_4d_to_2d_hists[0], "Results/4dTo2d/xy/hist_" + filename_output + "_event"+str(eventID)+"_XvsY.pgm")

        if do3d:
            compute_4d_to_3d_histograms(event_hits, x_bin_edges, y_bin_edges, z_bin_edges, all_4d_to_3d_hists)

        if i == 50:
            # only for testing
            if do2d_pdf:
                pdf_2d_plots.close()
            break

    if do3d:
        store_3d_histogram_as_hdf5(all_4d_to_3d_hists, tracks, filename_output)




if __name__ == '__main__':
    main(n_bins=[20,20,20], do2d=False, do2d_pdf=False, do3d=True, do_mc_hits=False)
