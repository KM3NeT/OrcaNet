#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This main code takes raw simulated .hdf5 files as input in order to generate 2D/3D histograms ('images') that can be used for CNNs."""

import sys
from matplotlib.backends.backend_pdf import PdfPages

from file_to_hits import *
from hits_to_histograms import *
from histograms_to_files import *
import glob

__author__ = 'Michael Moser'
__license__ = 'AGPL'
__version__= '0.1'
__email__ = 'michael.m.moser@fau.de'
__status__ = 'Prototype'
# Heavily based on code from sgeisselsoeder: https://github.com/sgeisselsoeder/km3netHdf5ToHistograms/


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
        glob.pdf_2d_plots = PdfPages('Results/4dTo2d/' + filename_output + '_plots.pdf')

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
           #  only for testing
            if do2d_pdf:
                glob.pdf_2d_plots.close()
            break
    #if do2d_pdf:
     #   glob.pdf_2d_plots.close()

    if do2d:
        store_histograms_as_hdf5(np.array(all_4d_to_2d_hists)[:, 0], tracks, 'Results/4dTo2d/h5/xy/' + filename_output + '_xy.h5', projection='xy')
        store_histograms_as_hdf5(np.array(all_4d_to_2d_hists)[:, 1], tracks, 'Results/4dTo2d/h5/xz/' + filename_output + '_xz.h5', projection='xz')
        store_histograms_as_hdf5(np.array(all_4d_to_2d_hists)[:, 2], tracks, 'Results/4dTo2d/h5/yz/' + filename_output + '_yz.h5', projection='yz')

    if do3d:
        store_histograms_as_hdf5(np.array(all_4d_to_3d_hists)[:, 0], tracks, 'Results/4dTo3d/h5/xyz/' + filename_output + '_xyz.h5', projection='xyz')


if __name__ == '__main__':
    main(n_bins=[100,100,100], do2d=True, do2d_pdf=True, do3d=True, do_mc_hits=False)
