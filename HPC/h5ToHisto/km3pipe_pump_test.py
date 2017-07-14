import numpy as np
import km3pipe as kp
import h5py
import collections
import os
import sys

from km3modules.common import StatusBar


class MPump(kp.Pump):

    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.filename = self.require('filename')
        if not self.filename :
            raise ValueError("No filename defined")
        self.index = None
        self._reset_index()
        self._n_items = None

        # Open file to read how many entries it has
        if os.path.isfile(self.filename):
            self.h5f = h5py.File(self.filename, 'r')
        else:
            raise IOError("No such file or directory: '{0}'".format(self.filename))

        event_info_dset = self.h5f['event_info']
        self.event_ids = event_info_dset['event_id']
        self._n_events = len(self.event_ids)

        self.hits_indices = self.h5f['/hits/_indices'][:]

    def _reset_index(self):
        """Reset index to default value"""
        self.index = 0

    def __iter__(self):
        return self

    def next(self):
        """Python 2/3 compatibility for iterators"""
        return self.__next__()

    def __next__(self):
        if self.index >= self._n_events:
            self._reset_index()
            raise StopIteration
        blob = self.get_blob(self.index)
        self.index += 1
        return blob

    def get_blob(self, index):
        blob = kp.core.Blob() # Currently, this is just an empty dictionary

        h5f = self.h5f
        event_id = self.event_ids[index]

        idx, n_items = self.hits_indices[event_id]
        end = idx + n_items

        # extract event_info
        dtype_event_info = np.dtype([
            ('det_id', '<i4'),
            ('frame_index', '<u4'),
            ('livetime_sec', '<u8'),
            ('mc_id', '<i4'),
            ('mc_t', '<f8'),
            ('n_events_gen', '<u8'),
            ('n_files_gen', '<u8'),
            ('overlays', '<u4'),
            ('trigger_counter', '<u8'),
            ('trigger_mask', '<u8'),
            ('utc_nanoseconds', '<u8'),
            ('utc_seconds', '<u8'),
            ('weight_w1', '<f8'),
            ('weight_w2', '<f8'),
            ('weight_w3', '<f8'),
            ('run_id', '<u8'),
            ('event_id', '<u4'),
        ])
        event_info = np.recarray(1, dtype=dtype_event_info)
        for attr in dtype_event_info.names:
            event_info[attr] = h5f['event_info'][index][attr]
        blob['EventInfo'] = event_info

        #extract mc_tracks, same method

        # extract hits: x,y,z,time

        # extract mc_hits: x,y,z,time

        sys.exit()

        self.idx += 1
        return blob




    def process(self, blob):
        try:
            hits_idx, n_hits = self.hits_indices[self.idx]
        except IndexError:
            raise StopIteration

        hits = np.recarray(n_hits, dtype=self.dt)
        for attr in self.dt.names:
            data = self.h5f["/hits/" + attr][hits_idx:hits_idx+n_hits]
            hits[attr] = data

        blob["Hits"] = hits

        self.idx += 1
        return blob

    def finish(self):
        self.h5f.close()


filename_input = 'testfile_km3pipe_v7.2.2.h5'
event_pump = MPump(filename=filename_input)

for event_blob in event_pump:
    print event_blob
