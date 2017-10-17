#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This main code takes raw simulated .hdf5 files as input in order to generate 2D/3D histograms ('images') that can be used for CNNs."""

import os
import sys
import warnings
#from memory_profiler import profile # for memory profiling, call with @profile; myfunc()
#import line_profiler # call with kernprof -l -v file.py args
from matplotlib.backends.backend_pdf import PdfPages

import glob
from file_to_hits import *
from histograms_to_files import *
from hits_to_histograms import *

__author__ = 'Michael Moser'
__license__ = 'AGPL'
__version__ = '1.0'
__email__ = 'michael.m.moser@fau.de'
__status__ = 'Prototype'


def parse_input(do2d, do2d_pdf):
    """
    Handles input exceptions, warnings and helps.
    :param bool do2d: Boolean flag for creation of 2D histograms.
    :param (bool, int) do2d_pdf: Boolean flag for creation of 2D pdf images.
    :return: str fname: Parsed filename.
    """
    if len(sys.argv) < 2 or str(sys.argv[1]) == "-h":
        print "Usage: python " + str(sys.argv[0]) + " file.h5"
        sys.exit(1)

    if do2d==False and do2d_pdf==True:
        raise ValueError('The 2D pdf images cannot be created if do2d=False. Please try again.')

    if do2d_pdf[0] is True and do2d_pdf[1] > 100:
        warnings.warn('You declared do2d_pdf=(True, int) with int > 100. This will take more than two minutes.'
                      'Do you really want to create pdfs images for so many events?')

    if not os.path.isfile(str(sys.argv[1])):
        raise IOError('The file -' + str(sys.argv[1]) + '- does not exist.')

    fname = str(sys.argv[1])
    return fname


def calculate_bin_edges_test(geo, y_bin_edge, z_bin_edge):
    """
    Tests, if the bins in one direction don't accidentally have more than 'one' OM.
    For the x-direction, an overlapping can not be avoided in an orthogonal space.
    For y and z though, it can!
    For y, every bin should contain the number of lines per y-direction * 18 for 18 doms per line.
    For z, every bin should contain 115 entries, since every z bin contains one storey of the 115 ORCA lines.
    Not 100% accurate, since only the dom positions are used and not the individual pmt positions for a dom.
    """
    geo_y = geo[:, 2]
    geo_z = geo[:, 3]
    hist_y = np.histogram(geo_y, bins=y_bin_edge)
    hist_z = np.histogram(geo_z, bins=z_bin_edge)

    print '----------------------------------------------------------------------------------------------'
    print 'Y-axis: Bin content: ' + str(hist_y[0])
    print 'It should be:        ' + str(np.array(
        [4 * 18, 7 * 18, 9 * 18, 10 * 18, 9 * 18, 10 * 18, 10 * 18, 10 * 18, 11 * 18, 10 * 18, 9 * 18, 8 * 18, 8 * 18]))
    print 'Y-axis: Bin edges: ' + str(hist_y[1])
    print '..............................................................................................'
    print 'Z-axis: Bin content: ' + str(hist_z[0])
    print 'It should have 115 entries everywhere'
    print 'Z-axis: Bin edges: ' + str(hist_z[1])
    print '----------------------------------------------------------------------------------------------'


def calculate_bin_edges(n_bins, fname_geo_limits):
    """
    Calculates the bin edges for the later np.histogramdd actions based on the number of specified bins. 
    This is performed in order to get the same bin size for each event regardless of the fact if all bins have a hit or not.
    :param tuple n_bins: contains the desired number of bins for each dimension. [n_bins_x, n_bins_y, n_bins_z]
    :param str fname_geo_limits: filepath of the .txt ORCA geometry file.
    :return: ndarray(ndim=1) x_bin_edges, y_bin_edges, z_bin_edges: contains the resulting bin edges for each dimension.
    """
    print "Reading detector geometry in order to calculate the detector dimensions from file " + fname_geo_limits
    geo = np.loadtxt(fname_geo_limits)

    # derive maximum and minimum x,y,z coordinates of the geometry input [[first_OM_id, xmin, ymin, zmin], [last_OM_id, xmax, ymax, zmax]]
    geo_limits = np.nanmin(geo, axis = 0), np.nanmax(geo, axis = 0)
    print 'Detector dimensions [[first_OM_id, xmin, ymin, zmin], [last_OM_id, xmax, ymax, zmax]]: ' + str(geo_limits)

    x_bin_edges = np.linspace(geo_limits[0][1] - 9.95, geo_limits[1][1] + 9.95, num=n_bins[0] + 1) #try to get the lines in the bin center 9.95*2 = average x-separation of two lines
    y_bin_edges = np.linspace(geo_limits[0][2] - 9.75, geo_limits[1][2] + 9.75, num=n_bins[1] + 1) # Delta y = 19.483
    z_bin_edges = np.linspace(geo_limits[0][3] - 4.665, geo_limits[1][3] + 4.665, num=n_bins[2] + 1) # Delta z = 9.329

    #calculate_bin_edges_test(geo, y_bin_edges, z_bin_edges) # test disabled by default. Activate it, if you change the offsets in x/y/z-bin-edges

    return x_bin_edges, y_bin_edges, z_bin_edges


def main(n_bins, do2d=True, do2d_pdf=(False, 10), do3d=True, do4d=False, do_mc_hits=False, use_calibrated_file=False, data_cuts=None):
    """
    Main code. Reads raw .hdf5 files and creates 2D/3D histogram projections that can be used for a CNN
    :param tuple(int) n_bins: Declares the number of bins that should be used for each dimension (x,y,z,t).
    :param bool do2d: Declares if 2D histograms should be created.
    :param (bool, int) do2d_pdf: Declares if pdf visualizations of the 2D histograms should be created. Cannot be called if do2d=False.
                                 The event loop will be stopped after the integer specified in the second argument.
    :param bool do3d: Declares if 3D histograms should be created.
    :param bool do4d: Declares if 4D histograms should be created.
    :param bool do_mc_hits: Declares if hits (False, mc_hits + BG) or mc_hits (True) should be processed
    :param bool use_calibrated_file: Declares if the input file is already calibrated (pos_x/y/z, time) or not.
    :param dict data_cuts: Dictionary that contains information about any possible cuts that should be applied.
                           Supports the following cuts: 'triggered', 'energy_lower_limit'
    """
    if data_cuts is None: data_cuts={'triggered': False, 'energy_lower_limit': 0}

    filename_input = parse_input(do2d, do2d_pdf)
    filename = os.path.basename(os.path.splitext(filename_input)[0])
    filename_output = filename.replace(".","_")
    filename_geo_limits = 'ORCA_Geo_115lines.txt' # used for calculating the dimensions of the ORCA can

    geo = None
    if use_calibrated_file is False:
        filename_geometry = 'orca_115strings_av23min20mhorizontal_18OMs_alt9mvertical_v1.detx'  # used for x/y/z calibration
        if os.path.isfile(filename_geometry) is True:
            geo = kp.Geometry(filename='/home/woody/capn/mppi033h/misc/orca_detectors/fixed/' + filename_geometry)
        else:
            raise IOError('The .detx file does not exist in the default path </home/woody/capn/mppi033h/misc/orca_detectors/fixed/>! '
                          'Change the path or add the .detx file to the default path.')

    x_bin_edges, y_bin_edges, z_bin_edges = calculate_bin_edges(n_bins, filename_geo_limits)

    all_4d_to_2d_hists, all_4d_to_3d_hists, all_4d_to_4d_hists = [], [], []
    mc_infos = []

    if do2d_pdf[0] is True:
        glob.pdf_2d_plots = PdfPages('Results/4dTo2d/' + filename_output + '_plots.pdf')

    # Initialize HDF5Pump of the input file
    event_pump = kp.io.hdf5.HDF5Pump(filename=filename_input)
    print "Generating histograms from the hits in XYZT format for files based on " + filename_input
    for i, event_blob in enumerate(event_pump):
        if i % 10 == 0:
            print 'Event No. ' + str(i)

        # filter out all hit and track information belonging that to this event
        event_hits, event_track = get_event_data(event_blob, geo, do_mc_hits, use_calibrated_file, data_cuts)

        if event_track[2] < data_cuts['energy_lower_limit']: # Cutting events with energy < threshold (default=0)
            #print 'Cut an event with an energy of ' + str(event_track[2]) + ' GeV'
            continue

        # event_track: [event_id, particle_type, energy, isCC, bjorkeny, dir_x/y/z, time]
        mc_infos.append(event_track)

        if do2d:
            compute_4d_to_2d_histograms(event_hits, x_bin_edges, y_bin_edges, z_bin_edges, n_bins, all_4d_to_2d_hists, event_track, do2d_pdf[0])

        if do3d:
            compute_4d_to_3d_histograms(event_hits, x_bin_edges, y_bin_edges, z_bin_edges, n_bins, all_4d_to_3d_hists)

        if do4d:
            compute_4d_to_4d_histograms(event_hits, x_bin_edges, y_bin_edges, z_bin_edges, n_bins, all_4d_to_4d_hists)

        if do2d_pdf[0] is True:
            if i >= do2d_pdf[1]:
                glob.pdf_2d_plots.close()
                break

    if do2d:
        store_histograms_as_hdf5(np.stack([hist_tuple[0] for hist_tuple in all_4d_to_2d_hists]), np.array(mc_infos), 'Results/4dTo2d/h5/xy/' + filename_output + '_xy.h5')
        store_histograms_as_hdf5(np.stack([hist_tuple[1] for hist_tuple in all_4d_to_2d_hists]), np.array(mc_infos), 'Results/4dTo2d/h5/xz/' + filename_output + '_xz.h5')
        store_histograms_as_hdf5(np.stack([hist_tuple[2] for hist_tuple in all_4d_to_2d_hists]), np.array(mc_infos), 'Results/4dTo2d/h5/yz/' + filename_output + '_yz.h5')
        store_histograms_as_hdf5(np.stack([hist_tuple[3] for hist_tuple in all_4d_to_2d_hists]), np.array(mc_infos), 'Results/4dTo2d/h5/xt/' + filename_output + '_xt.h5')
        store_histograms_as_hdf5(np.stack([hist_tuple[4] for hist_tuple in all_4d_to_2d_hists]), np.array(mc_infos), 'Results/4dTo2d/h5/yt/' + filename_output + '_yt.h5')
        store_histograms_as_hdf5(np.stack([hist_tuple[5] for hist_tuple in all_4d_to_2d_hists]), np.array(mc_infos), 'Results/4dTo2d/h5/zt/' + filename_output + '_zt.h5')

    if do3d:
        store_histograms_as_hdf5(np.stack([hist_tuple[0] for hist_tuple in all_4d_to_3d_hists]), np.array(mc_infos), 'Results/4dTo3d/h5/xyz/' + filename_output + '_xyz.h5')
        store_histograms_as_hdf5(np.stack([hist_tuple[1] for hist_tuple in all_4d_to_3d_hists]), np.array(mc_infos), 'Results/4dTo3d/h5/xyt/' + filename_output + '_xyt.h5')
        store_histograms_as_hdf5(np.stack([hist_tuple[2] for hist_tuple in all_4d_to_3d_hists]), np.array(mc_infos), 'Results/4dTo3d/h5/xzt/' + filename_output + '_xzt.h5')
        store_histograms_as_hdf5(np.stack([hist_tuple[3] for hist_tuple in all_4d_to_3d_hists]), np.array(mc_infos), 'Results/4dTo3d/h5/yzt/' + filename_output + '_yzt.h5')
        store_histograms_as_hdf5(np.stack([hist_tuple[4] for hist_tuple in all_4d_to_3d_hists]), np.array(mc_infos), 'Results/4dTo3d/h5/rzt/' + filename_output + '_rzt.h5')

    if do4d:
        store_histograms_as_hdf5(np.array(all_4d_to_4d_hists), np.array(mc_infos), 'Results/4dTo4d/h5/xyzt/' + filename_output + '_xyzt.h5', compression=('gzip', 1))


if __name__ == '__main__':
    main(n_bins=(11,13,18,50), do2d=False, do2d_pdf=(False, 100), do3d=False, do4d=True,
         do_mc_hits=False, use_calibrated_file=True, data_cuts = {'triggered': False, 'energy_lower_limit': 0})







