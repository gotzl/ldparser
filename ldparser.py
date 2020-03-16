""" Parser for MoTec ld files

Code created through reverse engineering the data format.
"""

import datetime
import struct

import numpy as np


class ldData(object):
    """Container for parsed data of an ld file.

    Allows reading and writing.
    """

    def __init__(self, head, channs):
        self.head = head
        self.channs = channs

    def __getitem__(self, item):
        if not isinstance(item, int):
            col = [n for n, x in enumerate(self.channs) if x.name == item]
            if len(col) != 1:
                raise Exception("Could get column", item, col)
            item = col[0]
        return self.channs[item]

    def __setitem__(self, key, value):
        self.channs[key] = value

    @classmethod
    def frompd(cls, df):
        # type: (pd.DataFrame) -> ldData
        """Create and ldData object from a pandas DataFrame.
        ToDo: Create a mocked ldHeader and channel headers
        """
        import pandas as pd

    @classmethod
    def fromfile(cls, f):
        # type: (str) -> ldData
        """Parse data of an ld file
        """
        return cls(*read_ldfile(f))

    def write(self, f):
        # type: (str) -> ()
        """Write an ld file containing the current header information and channel data
        """
        with open(f, 'wb') as f_:
            self.head.write(f_, len(self.channs))

            f_.seek(self.channs[0].meta_ptr)
            list(map(lambda c:c.write(f_), self.channs))
            list(map(lambda c:f_.write(c.data), self.channs))


class ldEvent(object):
    fmt = '<1152sH'

    def __init__(self, name, venue_ptr, venue):
        self.name, self.venue_ptr, self.venue = name, venue_ptr, venue

    @classmethod
    def fromfile(cls, f):
        # type: (file) -> ldEvent
        """Parses and stores the header information of an ld file
        """
        name, venue_ptr = struct.unpack(ldEvent.fmt, f.read(struct.calcsize(ldEvent.fmt)))
        f.seek(venue_ptr)
        venue = ldVenue.fromfile(f)
        return cls(decode_string(name), venue_ptr, venue)

    def write(self, f):
        f.write(struct.pack(ldEvent.fmt, self.name.encode(), self.venue_ptr))
        f.seek(self.venue_ptr)
        self.venue.write(f)

    def __str__(self):
        return "%s; venue: %s"%(self.name, self.venue)


class ldVenue(object):
    fmt = '<1098sH'

    def __init__(self, name, vehicle_ptr, vehicle):
        self.name, self.vehicle_ptr, self.vehicle = name, vehicle_ptr, vehicle

    @classmethod
    def fromfile(cls, f):
        # type: (file) -> ldVenue
        """Parses and stores the header information of an ld file
        """
        name, vehicle_ptr = struct.unpack(ldVenue.fmt, f.read(struct.calcsize(ldVenue.fmt)))
        f.seek(vehicle_ptr)
        vehicle = ldVehicle.fromfile(f)
        return cls(decode_string(name), vehicle_ptr, vehicle)

    def write(self, f):
        f.write(struct.pack(ldVenue.fmt, self.name.encode(), self.vehicle_ptr))
        f.seek(self.vehicle_ptr)
        self.vehicle.write(f)

    def __str__(self):
        return "%s; vehicle: %s"%(self.name, self.vehicle)


class ldVehicle(object):
    fmt = '<192sI'

    def __init__(self, id, weight):
        self.id, self.weight = id, weight

    @classmethod
    def fromfile(cls, f):
        # type: (file) -> ldVehicle
        """Parses and stores the header information of an ld file
        """
        id, weight = struct.unpack(ldVehicle.fmt, f.read(struct.calcsize(ldVehicle.fmt)))
        return cls(decode_string(id), weight)

    def write(self, f):
        f.write(struct.pack(ldVehicle.fmt, self.id.encode(), self.weight))

    def __str__(self):
        return "%s"%(self.id)


class ldHead(object):
    fmt = '<' + (
        "I4x"     # ldmarker
        "II"      # chann_meta_ptr chann_data_ptr
        "20x"     # ??
        "I"       # event_ptr
        "24x"     # ??
        "HHH"     # device serial, type and version
        "I"       # ?
        "8s"      # ?
        "HH"      # ? version
        "I"       # num_channs
        "4x"      # ??
        "16s"     # date
        "16x"     # ??
        "16s"     # time
        "16x"     # ??
        "64s"     # name
        "64s"     # vehicle
        "64s"     # ??
        "1152s"   # venue
        "I"
    )

    def __init__(self, meta_ptr, data_ptr, aux_ptr, aux, name, vehicle, venue, event, datetime):
        self.meta_ptr, self.data_ptr, self.aux_ptr, self.aux, self.name, self.vehicle, \
        self.venue, self.event, self.datetime = meta_ptr, data_ptr, aux_ptr, aux,\
                                                name, vehicle, venue, event, datetime

    @classmethod
    def fromfile(cls, f):
        # type: (file) -> ldHead
        """Parses and stores the header information of an ld file
        """
        (_, meta_ptr, data_ptr, aux_ptr,
            _, _, _, _,
            _, _, _, n,
            date, time,
            name, vehicle, event, venue,
            _) = struct.unpack(ldHead.fmt, f.read(struct.calcsize(ldHead.fmt)))
        date, time, name, vehicle, event, venue = map(decode_string, [date, time, name, vehicle, event, venue])

        try:
            # first, try to decode datatime with seconds
            _datetime = datetime.datetime.strptime(
                    '%s %s'%(date, time), '%d/%m/%Y %H:%M:%S')
        except ValueError:
            _datetime = datetime.datetime.strptime(
                '%s %s'%(date, time), '%d/%m/%Y %H:%M')

        f.seek(aux_ptr)
        aux = ldEvent.fromfile(f)
        return cls(meta_ptr, data_ptr, aux_ptr, aux, name, vehicle, venue, event, _datetime)

    def write(self, f, n):
        f.write(struct.pack(ldHead.fmt,
                            0x40,
                            self.meta_ptr, self.data_ptr, self.aux_ptr,
                            0x4240, 1, 0xf,
                            8004, "ADL".encode(), 0xadb0, 420, n,
                            self.datetime.date().strftime("%d/%m/%Y").encode(),
                            self.datetime.time().strftime("%H:%M:%S").encode(),
                            self.name.encode(), self.vehicle.encode(), "".encode(), self.venue.encode(),
                            0xc81a4
                            ))

        f.seek(self.aux_ptr)
        self.aux.write(f)

    def __str__(self):
        return 'name:    %s\n' \
                'event:   %s\n' \
                'venue:   %s\n' \
                'vehicle: %s\n' \
                'event_long: %s'%(
            self.name, self.vehicle, self.venue, self.event, self.aux)


class ldChan(object):
    """Channel (meta) data

    Parses and stores the channel meta data of a channel in a ld file.
    Needs the pointer to the channel meta block in the ld file.
    The actual data is read on demand using the 'data' property.
    """

    fmt = '<' + (
        "IIII"    # prev_addr next_addr data_ptr n_data
        "2x"      # counter?
        "HHH"     # datatype datatype rec_freq
        "HxxHH"   # shift ?? scale dec_places 
        "32s"     # name
        "8s"      # short name
        "12s"     # unit
        "40x"     # ?
    )

    def __init__(self, _f, meta_ptr, prev_meta_ptr, next_meta_ptr, data_ptr, data_len,
                 dtype_a, dtype, freq, shift, scale, dec,
                 name, short_name, unit):

        self._f = _f
        self.meta_ptr = meta_ptr
        self._data = None

        (self.prev_meta_ptr, self.next_meta_ptr, self.data_ptr, self.data_len,
        self.dtype_a, self.dtype, self.freq,
        self.shift, self.scale, self.dec,
        self.name, self.short_name, self.unit) = prev_meta_ptr, next_meta_ptr, data_ptr, data_len,\
                                                 dtype_a, dtype, freq,\
                                                 shift, scale, dec,\
                                                 name, short_name, unit

    @classmethod
    def fromfile(cls, _f, meta_ptr):
        # type: (str, int) -> ldChan
        """Parses and stores the header information of an ld channel in a ld file
        """
        with open(_f, 'rb') as f:
            f.seek(meta_ptr)

            (prev_meta_ptr, next_meta_ptr, data_ptr, data_len,
             dtype_a, dtype, freq, shift, scale, dec,
             name, short_name, unit) = struct.unpack(ldChan.fmt, f.read(struct.calcsize(ldChan.fmt)))

        if dtype_a == 0x07:
            dtype = [None, np.float16, None, np.float32][dtype-1]
        else:
            dtype = [None, np.int16, None, np.int32][dtype-1]

        name, short_name, unit = map(decode_string, [name, short_name, unit])
        return cls(_f, meta_ptr, prev_meta_ptr, next_meta_ptr, data_ptr, data_len,
                   dtype_a, dtype, freq, shift, scale, dec,name, short_name, unit)

    def write(self, f):
        if self.dtype == np.float16 or self.dtype == np.float32:
            dtype_a = 0x07
            dtype = {np.float16: 2, np.float32: 4}[self.dtype]
        else:
            dtype_a = 0x0
            dtype = {np.int16: 2, np.int32: 4}[self.dtype]

        f.write(struct.pack(ldChan.fmt,
                            self.prev_meta_ptr, self.next_meta_ptr, self.data_ptr, self.data_len,
                            dtype_a, dtype, self.freq, self.shift, self.scale, self.dec,
                            self.name.encode(), self.short_name.encode(), self.unit.encode()))

    @property
    def data(self):
        # type: () -> np.array
        """ Read the data words of the channel
        """
        if self._data is None:
            # jump to data and read
            with open(self._f, 'rb') as f:
                f.seek(self.data_ptr)
                try:
                    self._data = np.fromfile(f,
                                            count=self.data_len, dtype=self.dtype)

                    self._data = self._data/self.scale * pow(10., -self.dec) + self.shift

                    if len(self._data) != self.data_len:
                        raise ValueError("Not all data read!")

                except ValueError as v:
                    print(v, self.name, self.freq,
                          hex(self.data_ptr), hex(self.data_len),
                          hex(len(self._data)),hex(f.tell()))
                    # raise v
        return self._data

    def __str__(self):
        return 'chan %s (%s) [%s], %i Hz'%(
            self.name,
            self.short_name, self.unit,
            self.freq)


def decode_string(bytes):
    # type: (bytes) -> str
    """decode the bytes and remove trailing zeros
    """
    return bytes.decode('ascii').rstrip('\0').strip()


def read_channels(f_, meta_ptr):
    # type: (str, int) -> list
    """ Read channel data inside ld file

    Cycles through the channels inside an ld file,
     starting with the one where meta_ptr points to.
     Returns a list of ldchan objects.
    """
    chans = []
    while meta_ptr:
        chan_ = ldChan.fromfile(f_, meta_ptr)
        chans.append(chan_)
        meta_ptr = chan_.next_meta_ptr
    return chans


def read_ldfile(f_):
    # type: (str) -> (ldHead, list)
    """ Read an ld file, return header and list of channels
    """
    head_ = ldHead.fromfile(open(f_,'rb'))
    chans = read_channels(f_, head_.meta_ptr)
    return head_, chans


if __name__ == '__main__':
    """ Small test of the parser.
    
    Decodes all ld files in the directory. For each file, creates 
    a plot for data with the same sample frequency.  
    """

    import sys, os, glob
    from itertools import groupby
    import pandas as pd
    import matplotlib.pyplot as plt

    if len(sys.argv)!=2:
        print("Usage: ldparser.py /some/path/")
        exit(1)

    for f in glob.glob('%s/*.ld'%sys.argv[1]):
        print(os.path.basename(f))

        l = ldData.fromfile(f)
        print(l.head)
        print(list(map(str, l)))
        print()

        # create plots for all channels with the same frequency
        for f, g in groupby(l.channs, lambda x:x.freq):
            df = pd.DataFrame({i.name.lower(): i.data for i in g})
            df.plot()
            plt.show()
