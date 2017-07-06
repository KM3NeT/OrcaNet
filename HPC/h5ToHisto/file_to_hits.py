#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This utility code contains functions that read the raw MC .h5 files"""

import pandas as pd
import numpy as np
# Heavily based on code from sgeisselsoeder: https://github.com/sgeisselsoeder/km3netHdf5ToHistograms/

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
        #hits_group = np.array(h5py.File(fname, 'r')['hits'][()], np.float32)
        mc_hits_get_dom_id(hits_group)
    else:
        print "Reading triggered hits"
        hits_group = np.array(pd.read_hdf(fname, 'hits'), np.float32)
        # hits_group = np.array(h5py.File(fname, 'r')['hits'][()], np.float32)
        # hits_group = np.array(h5py.File(fname, 'r')['hits'][:], np.float32)

    # keep the relevant info from each hit: [event_id, dom_id, time]
    #hits = np.array(np.concatenate([hits_group[:, 14], hits_group[:, 4], hits_group[:, 11]], axis=1), np.float32) # old km3pipe version
    hits = np.array(np.concatenate([hits_group[:, 5:6], hits_group[:, 1:2], hits_group[:, 2:3]], axis=1), np.float32) # new km3pipe version 6.9.1
    del hits_group

    print "Converting hits omid -> XYZ"
    hits_xyz = convert_hits_xyz(hits, geo)
    del hits

    print "Done converting."
    return tracks, hits_xyz, geo_limits


def extract_relevant_track_info(tracks):
    """
    Returns the relevant MC information for all tracks. [event_id, particle_type, energy, isCC, bjorkeny, dir_x/y/z, time]
    :param ndarray(ndim=2) tracks: 2D array of the primary mc_tracks info.
    :return: ndarray(ndim=2): returns a 2D array with the relevant mc_tracks info for each event.
    """
    return np.array(np.concatenate([tracks[:, 14:15], tracks[:, 13:14], tracks[:, 4:5], tracks[:, 7:8], tracks[:, 0:1], tracks[:, 1:4]], tracks[:, 12:13], axis=1), np.float32)


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
            # hits_xyz_list: [event_id, positions_xyz, dom_id, time]
            hits_xyz_list.append([int(hit[0]), position[1], position[2], position[3], hit[2], int(hit[1])])
    return np.array(hits_xyz_list, np.float32)

