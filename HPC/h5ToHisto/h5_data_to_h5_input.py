#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This main code takes raw simulated .hdf5 files as input in order to generate 2D/3D histograms ('images') that can be used for CNNs."""

import os
import sys
#from memory_profiler import profile # for memory profiling, call with @profile; myfunc()
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
# Heavily based on code from sgeisselsoeder: https://github.com/sgeisselsoeder/km3netHdf5ToHistograms/


def calculate_bin_edges(n_bins, geo_limits):
    """
    Calculates the bin edges for the later np.histogramdd actions based on the number of specified bins. 
    This is performed in order to get the same bin size for each event regardless of the fact if all bins have a hit or not.
    :param list n_bins: contains the desired number of bins for each dimension. [n_bins_x, n_bins_y, n_bins_z]
    :param geo_limits: contains the min and max values of each geometry dimension. [[first_OM_id, xmin, ymin, zmin], [last_OM_id, xmax, ymax, zmax]]
    :return: ndarray(ndim=1) x_bin_edges, y_bin_edges, z_bin_edges: contains the resulting bin edges for each dimension.
    """
    x_bin_edges = np.linspace(geo_limits[0][1] -9.95, geo_limits[1][1]+9.95, num=n_bins[0] + 1) #try to get the lines in the bin center 9.95*2 = average x-separation of two lines
    y_bin_edges = np.linspace(geo_limits[0][2], geo_limits[1][2], num=n_bins[1] + 1) #+- 9.75
    z_bin_edges = np.linspace(geo_limits[0][3], geo_limits[1][3], num=n_bins[2] + 1)

    return x_bin_edges, y_bin_edges, z_bin_edges


def convert_class_to_categorical(particle_type, is_cc, num_classes=4):
    """
    Converts the possible particle types (elec/muon/tau , NC/CC) to a categorical type that can be used as tensorflow input y
    :param int particle_type: Specifies the particle type, i.e. elec/muon/tau (12, 14, 16). Negative values for antiparticles.
    :param int is_cc: Specifies the interaction channel. 0 = NC, 1 = CC.
    :param int num_classes: Specifies the total number of classes that will be discriminated later on by the CNN. I.e. 2 = elec_NC, muon_CC.
    :return: ndarray(ndim=1) categorical: returns the categorical event type. I.e. (particle_type=14, is_cc=1) -> [0,0,1,0] for num_classes=4.
    """
    if num_classes == 2:
        particle_type_dict = {(12, 0): 0, (14, 1): 1}  # 0: elec_NC, 1: elec_CC, 2: muon_CC, 3: tau_CC
    else:
        particle_type_dict = {(12, 0): 0, (12, 1): 1, (14, 1): 2, (16, 1): 3}  # 0: elec_NC, 1: elec_CC, 2: muon_CC, 3: tau_CC

    category = int(particle_type_dict[(abs(particle_type), is_cc)]) # 2
    categorical = np.zeros(num_classes, dtype='int') # (0,0,0,0)
    categorical[category] = 1 # (0,0,1,0)
    return categorical


def main(n_bins, do2d=True, do2d_pdf=True, do3d=True, do_mc_hits=False):
    """
    Main code. Reads raw .hdf5 files and creates 2D/3D histogram projections that can be used for a CNN
    :param list n_bins: Declares the number of bins that should be used for each dimension (x,y,z).
    :param bool do2d: Declares if 2D histograms should be created.
    :param bool do2d_pdf: Declares if pdf visualizations of the 2D histograms should be created. Cannot be called if do2d=False.
    :param bool do3d: Declares if 3D histograms should be created.
    :param bool do_mc_hits: Declares if hits (False, mc_hits + BG) or mc_hits (True) should be processed
    """
    if len(sys.argv) < 2 or str(sys.argv[1]) == "-h":
        print "Usage: python " + str(sys.argv[0]) + " file.h5"
        sys.exit(1)

    if do2d==False and do2d_pdf==True:
        print 'The 2D pdf images cannot be created if do2d==False. Please try again.'
        sys.exit(1)

    if not os.path.isfile(str(sys.argv[1])):
        print 'The file -' + str(sys.argv[1]) + '- does not exist. Exiting.'
        sys.exit(1)

    filename_input = str(sys.argv[1])
    filename = os.path.basename(os.path.splitext(filename_input)[0])
    filename_output = filename.replace(".","_")
    filename_geometry = 'ORCA_Geo_115lines.txt'
    #filename_input = '/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/raw_data/h5/elec-NC/3-100GeV/JTE.KM3Sim.gseagen.elec-NC.3-100GeV-3.4E6-1bin-3.0gspec.ORCA115_9m_2016.81.hdf5'

    tracks, hits_xyz, geo_limits = parse_file(filename_input, filename_geometry, do_mc_hits)
    all_event_numbers = set(hits_xyz[:, 0])

    x_bin_edges, y_bin_edges, z_bin_edges = calculate_bin_edges(n_bins, geo_limits)

    all_4d_to_2d_hists = []
    all_4d_to_3d_hists = []

    print "Generating histograms from the hits in XYZT format for files based on " + filename_input
    if do2d_pdf:
        glob.pdf_2d_plots = PdfPages('Results/4dTo2d/' + filename_output + '_plots.pdf')

    mc_infos = []
    i=0
    for eventID in all_event_numbers:
        if i % 10 == 0:
            print 'Event No. ' + str(i)
        i+=1
        # filter all hits belonging to this event
        event_hits = hits_xyz[np.where(hits_xyz[:, 0] == eventID)[0]]
        event_track = tracks[np.where(tracks[:, 0] == eventID)[0]][0]

        # get categorical event types and save all MC information to mc_infos
        event_categorical_type = convert_class_to_categorical(event_track[1], event_track[3], num_classes=2)
        all_mc_info = np.concatenate([event_track, event_categorical_type]) # [event_id, particle_type, energy, isCC, categorical types]
        mc_infos.append(all_mc_info)

        if do2d:
            compute_4d_to_2d_histograms(event_hits, x_bin_edges, y_bin_edges, z_bin_edges, all_4d_to_2d_hists, event_track, do2d_pdf)

        if do3d:
            compute_4d_to_3d_histograms(event_hits, x_bin_edges, y_bin_edges, z_bin_edges, all_4d_to_3d_hists)

        #if i == 10:
           #  only for testing
         #   if do2d_pdf:
          #      glob.pdf_2d_plots.close()
           # break
    #if do2d_pdf:
     #   glob.pdf_2d_plots.close()

    if do2d:
        store_histograms_as_hdf5(np.stack([hist_tuple[0] for hist_tuple in all_4d_to_2d_hists]), np.array(mc_infos), 'Results/4dTo2d/h5/xy/' + filename_output + '_xy.h5')
        store_histograms_as_hdf5(np.stack([hist_tuple[1] for hist_tuple in all_4d_to_2d_hists]), np.array(mc_infos), 'Results/4dTo2d/h5/xz/' + filename_output + '_xz.h5')
        store_histograms_as_hdf5(np.stack([hist_tuple[2] for hist_tuple in all_4d_to_2d_hists]), np.array(mc_infos), 'Results/4dTo2d/h5/yz/' + filename_output + '_yz.h5')

    if do3d:
        store_histograms_as_hdf5(np.stack([hist_tuple[0] for hist_tuple in all_4d_to_3d_hists]), np.array(mc_infos), 'Results/4dTo3d/h5/xyz/' + filename_output + '_xyz.h5')


if __name__ == '__main__':
    main(n_bins=[11,13,18], do2d=True, do2d_pdf=False, do3d=True, do_mc_hits=False)

