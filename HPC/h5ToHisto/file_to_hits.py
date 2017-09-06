#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This utility code contains functions that read the raw MC .h5 files"""

import os
import pandas as pd
import numpy as np
import km3pipe as kp
#from memory_profiler import profile
#import line_profiler # call with kernprof file.py args


def get_primary_track_index(event_blob):
    """
    Gets the index of the primary (neutrino) track.
    Uses bjorkeny in order to get the primary track, since bjorkeny!=0 for the initial interacting neutrino.
    :param kp.io.HDF5Pump.blob event_blob: HDF5Pump event blob.
    :return: int primary index: Index of the primary track (=neutrino) in the 'McTracks' branch.
    """
    bjorken_y_array = event_blob['McTracks'].bjorkeny
    primary_index = np.where(bjorken_y_array != 0.0)[0][0]
    return primary_index


def get_event_data(event_blob, geo, do_mc_hits, use_calibrated_file, data_cuts):
    """
    Reads a km3pipe blob which contains the information for one event.
    Returns a hit array and a track array that contains all relevant information of the event.
    :param kp.io.HDF5Pump.blob event_blob: Event blob of the HDF5Pump which contains all information for one event.
    :param kp.Geometry geo: km3pipe Geometry instance that contains the geometry information of the detector.
                            Only used if the event_blob is from a non-calibrated file!
    :param bool do_mc_hits: tells the function of the hits (mc_hits + BG) or the mc_hits only should be parsed.
                            In the case of mc_hits, the dom_id needs to be calculated thanks to the jpp output.
    :param bool use_calibrated_file: specifies if a calibrated file is used as an input for the event_blob.
                                     If False, the hits of the event_blob are calibrated based on the geo parameter.
    :param dict data_cuts: specifies if cuts should be applied. Contains the keys 'triggered' and 'energy_lower_limit'.
    :return: ndarray(ndim=2) hits_xyz: 2D array containing the hit information of the event [pos_xyz time].
    :return: ndarray(ndim=1) event_track: 1D array containing important MC information of the event.
                                          [event_id, particle_type, energy, isCC, bjorkeny, dir_x/y/z, time]
    """
    p = get_primary_track_index(event_blob)

    # parse tracks [event_id, particle_type, energy, isCC, bjorkeny, dir_x/y/z, time]
    event_id = event_blob['EventInfo'].event_id[0]
    particle_type = event_blob['McTracks'][p].type
    energy = event_blob['McTracks'][p].energy
    is_cc = event_blob['McTracks'][p].is_cc
    bjorkeny = event_blob['McTracks'][p].bjorkeny
    dir_x = event_blob['McTracks'][p].dir[0]
    dir_y = event_blob['McTracks'][p].dir[1]
    dir_z = event_blob['McTracks'][p].dir[2]
    time = event_blob['McTracks'][p].time

    event_track = np.array([event_id, particle_type, energy, is_cc, bjorkeny, dir_x, dir_y, dir_z, time], dtype=np.float32)

    # parse hits [x, y, z, time]
    if do_mc_hits is True:
        hits = event_blob["McHits"]
    else:
        hits = event_blob["Hits"]

    if use_calibrated_file is False:
        hits = geo.apply(hits)

    if data_cuts['triggered'] is True:
        hits = hits.__array__[hits.triggered.astype(bool)]
        #hits = hits.triggered_hits # alternative, though it only works for the triggered condition!

    pos_x = hits.pos_x.astype('float32')
    pos_y = hits.pos_y.astype('float32')
    pos_z = hits.pos_z.astype('float32')
    time = hits.time.astype('float32')

    ax = np.newaxis
    event_hits = np.concatenate([pos_x[:, ax], pos_y[:, ax], pos_z[:, ax], time[:, ax]], axis=1)

    # event_hits: 2D hits array for one event, event_track: 1D track array containing event information
    return event_hits, event_track


#-------- only legacy code from here on --------
#Legacy code
def get_geometry(filename_geometry, fname_geo_limits):
    """
    Gets fundamental geometry information of the ORCA detector that is used in the simulation.
    At first, the geometry stored in a .txt is used in order to calculate the dimensions of the ORCA can.
    This information is used in the bin calculation later on -> calculate_bin_edges().
    After that, the geometry .detx file is read with km3pipel.
    Later on in the HDF5Pump loop, the kp.Geometry instance will be used in order to convert hits to hits_x/y/z.
    :param str filename_geometry: filepath of the ORCA .detx file.
    :param str fname_geo_limits: filepath of the .txt ORCA geometry file.
    :return: kp.Geometry geo: km3pipe.Geometry instance.
    :return (ndarray(ndim=1), ndarray(ndim=1)) geo_limits: tuple that contains the min and max geometry values for each dimension.
    ([first_OM_id, xmin, ymin, zmin], [last_OM_id, xmax, ymax, zmax])
    """
    print "Reading detector geometry in order to calculate the detector dimensions from file " + fname_geo_limits
    geo = np.loadtxt(fname_geo_limits)

    # derive maximum and minimum x,y,z coordinates of the geometry input [[first_OM_id, xmin, ymin, zmin], [last_OM_id, xmax, ymax, zmax]]
    geo_limits = np.nanmin(geo, axis = 0), np.nanmax(geo, axis = 0)
    print 'Detector dimensions [[first_OM_id, xmin, ymin, zmin], [last_OM_id, xmax, ymax, zmax]]: ' + str(geo_limits)


    if os.path.isfile(filename_geometry) is True:
        geo = kp.Geometry(filename='/home/woody/capn/mppi033h/misc/orca_detectors/fixed/' + filename_geometry)

    #else: #use only if the .detx is not fixed yet by km3pipe
     #   det = kp.hardware.Detector(filename='/home/woody/capn/mppi033h/misc/orca_detectors/' + filename_geometry)
      #  det.write(filename_geometry)
       # geo = kp.Geometry(filename=filename_geometry)

    return geo, geo_limits

# Legacy code
def parse_file(fname, fname_geo, do_mc_hits):
    """
    Reads the raw .hdf5 neutrino MC file and returns the hit arrays (event_id [pos_xyz] dom_id time).
    :param str fname: filepath of parsed inputfile.
    :param str fname_geo: filepath of used ORCA geometry file.
    :param bool do_mc_hits: tells the function of the hits (mc_hits + BG) or the mc_hits only should be parsed. 
                            In the case of mc_hits, the dom_id needs to be calculated thanks to the jpp output.
    :return: ndarray(ndim=2) tracks: 2D array containing important MC information for each event_id.
                                     [event_id, particle_type, energy, isCC, bjorkeny, dir_x/y/z, time]
    :return: ndarray(ndim=2) hits_xyz: 2D array containing [event_id pos_xyz dom_id time].
    :return (ndarray(ndim=1), ndarray(ndim=1)) geo_limits: tuple that contains the min and max geometry values for each dimension. 
    ([first_OM_id, xmin, ymin, zmin], [last_OM_id, xmax, ymax, zmax])
    """
    print "Extracting hits from h5 file " + fname
    print "Reading detector geometry from file " + fname_geo
    geo = np.loadtxt(fname_geo)

    # derive maximum and minimum x,y,z coordinates of the geometry input [[first_OM_id, xmin, ymin, zmin], [last_OM_id, xmax, ymax, zmax]]
    geo_limits = np.nanmin(geo, axis = 0), np.nanmax(geo, axis = 0)
    print 'Detector dimensions [[first_OM_id, xmin, ymin, zmin], [last_OM_id, xmax, ymax, zmax]]: ' + str(geo_limits)

    print "Reading tracks"
    tracks_full = np.array(pd.read_hdf(fname, 'mc_tracks'), np.float32)
    print "Filtering primary tracks"
    tracks_primary = tracks_full[np.where(tracks_full[:, 0] != 0.0)[0]]
    # keep the relevant info from the track: [event_id, particle_type, energy, isCC]
    tracks = extract_relevant_track_info(tracks_primary)

    if do_mc_hits is True:
        print "Reading mc-hits"
        hits_group = np.array(pd.read_hdf(fname, 'mc_hits'), np.float32)
        mc_hits_get_dom_id(hits_group)
    else:
        print "Reading triggered hits"
        hits_group = np.array(pd.read_hdf(fname, 'hits'), np.float32)

    # keep the relevant info from each hit: [event_id, dom_id, time]
    hits = np.array(np.concatenate([hits_group[:, 5:6], hits_group[:, 1:2], hits_group[:, 2:3]], axis=1), np.float32) # new km3pipe version 6.9.1
    del hits_group

    print "Converting hits omid -> XYZ"
    hits_xyz = convert_hits_xyz(hits, geo)
    del hits

    print "Done converting."
    return tracks, hits_xyz, geo_limits

# Legacy code
def extract_relevant_track_info(tracks):
    """
    Returns the relevant MC information for all tracks. [event_id, particle_type, energy, isCC, bjorkeny, dir_x/y/z, time]
    :param ndarray(ndim=2) tracks: 2D array of the primary mc_tracks info.
    :return: ndarray(ndim=2): returns a 2D array with the relevant mc_tracks info for each event.
    """
    return np.array(np.concatenate([tracks[:, 14:15], tracks[:, 13:14], tracks[:, 4:5], tracks[:, 7:8], tracks[:, 0:1], tracks[:, 1:4]], tracks[:, 12:13], axis=1), np.float32)

# Legacy code
def mc_hits_get_dom_id(hits_group):
    """
    This function calculates the dom_id of each mc_hit event based on their pmt_id's. 
    After this, the appropriate dom_id is inserted into the hits_group 2D array for each event.
    pmt_id = (dom_id - 1) * 31 + channel_id + 1
    dom_id = (pmt_id-1)/31 + 1
    :param ndarray(ndim=2) hits_group: 2D arrays that contains the full mc_hit information.
    """
    for hit in hits_group:
        print hit
        pmt_id = int(hit[6])
        dom_id = int((pmt_id-int(1))/int(31)) + 1
        hit[4] = dom_id

# Legacy code
def convert_hits_xyz(hits, geo):
    """
    Reads the hits array with dom_id's and returns the hits_xyz array with according xyz positions.
    :param ndarray(ndim=2) hits: 2D hits array that contain [event_id, dom_id, time].
    :param  ndarray(ndim=2) geo: 2D geo array that contains the xyz position for each dom_id.
    :return: ndarray(ndim=2) hits_xyz: 2D hits array with xyz position information [event_id, pos_x/y/z, time, dom_id].
    """
    hits_xyz_list = []
    for hit in hits:
            position = geo[int(hit[1])-1]
            # hits_xyz_list: [event_id, positions_xyz, time, dom_id]
            hits_xyz_list.append([int(hit[0]), position[1], position[2], position[3], hit[2], int(hit[1])])
    return np.array(hits_xyz_list, np.float32)