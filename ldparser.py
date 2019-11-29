""" Parser for MoTec ld files

Code created through reverse engineering the data format.
So far, the decoding looks as follows:

### header
description       length    offset
-----------------------------------
unknown           0x8       0x000
channel meta ptr  0x4       0x008
channel data ptr  0x4       0x00c
unknown           0x14      0x010
descr ptr         0x4       0x024
unknown           0x1a      0x028
unknown           0x4       0x042
device serial     0x4       0x046
device type       0x8       0x04a
device version    0x4       0x052
unknown           0x4       0x056
unknown           0x4       0x05a
date              0x10      0x05e
unknown           0x10      0x06e
time              0x10      0x07e
unknown           0x10      0x08e
name              0x40      0x09e
vehicle1          0x40      0x0ce
unknown           0x40      0x11e
venue             0x40      0x15e
vehicle2          0x40      0x6e2

### channel meta data
description       length    offset
---------------------------------
prev_addr         0x4       0x00
next_addr         0x4       0x04
data ptr          0x4       0x08
n_data            0x4       0x0c
somecnt           0x2       0x10
datatype          0x2       0x12
datatype          0x2       0x14
rec freq          0x2       0x16
shift             0x2       0x18
unknown           0x2       0x1a
scale             0x2       0x1c
dec_places        0x2       0x1e
name              0x28      0x20
unit              0x0c      0x48
scale             0x2       0x54
unknown           0x2       0x56
unknown           0x1a      0x58
"""

import datetime
import numpy as np

dt32 = np.dtype(np.uint32).newbyteorder('<')
dt16 = np.dtype(np.uint16).newbyteorder('<')


class ldhead(object):
    """Parses and stores the header information of an ld file"""
    def __init__(self, f_):
        # type: (str) -> None
        with open(f_, 'rb') as f:
            f.seek(0x8)

            # meta_ptr: pointer to begin of channel meta info
            # data_ptr: pointer to begin of channel data
            self.meta_ptr, self.data_ptr = np.fromfile(f, dtype=dt32, count=2)

            f.seek(0x14, 1) # jump over unknown
            # pointer to some text
            descr_ = np.fromfile(f, dtype=dt32, count=1)[0]

            f.seek(0x36, 1)
            date = decode_string(f.read(0x10))

            f.seek(0x10,1)
            time = decode_string(f.read(0x10))

            f.seek(0x10,1)
            self.name = decode_string(f.read(0x40))
            self.vehicle = decode_string(f.read(0x40))

            f.seek(0x40,1)
            self.venue = decode_string(f.read(0x40))

            f.seek(descr_)
            self.event = decode_string(f.read(0x10))

            f.seek(0x470, 1)
            descr_ = np.fromfile(f, dtype=dt32, count=1)[0]

            self.descr = None
            if descr_>0:
                f.seek(descr_)
                self.descr = decode_string(f.read(0x10))

            if len(self.vehicle)==0:
                f.seek(0x6e2)
                self.vehicle = decode_string(f.read(0x40))

        try:
            self.datetime = datetime.datetime.strptime(
                    '%s %s'%(date, time), '%d/%m/%Y %H:%M:%S')
        except ValueError:
            self.datetime = datetime.datetime.strptime(
                '%s %s'%(date, time), '%d/%m/%Y %H:%M')

    def __str__(self):
        return 'name:    %s\n' \
               'vehicle: %s\n' \
               'venue: %s\n' \
               'event:   %s descr: %s\n'%(
            self.name, self.vehicle, self.venue, self.event, self.descr)


class ldchan(object):
    """Channel (meta) data

    Parses and stores the channel meta as well as the
    actual data of a channel in a ld file.
    Needs the pointer to the channel meta block in the ld file.
    """
    def __init__(self, f_, meta_ptr, num):
        # type: (str, int, int) -> None
        self.meta_ptr = meta_ptr
        self.num = num

        with open(f_,'rb') as f:
            f.seek(meta_ptr)

            (self.prev_meta_ptr, self.next_meta_ptr, self.data_ptr, self.data_len) =  \
                np.fromfile(f, dtype=dt32, count=4)

            f.seek(2, 1) # some count, not needed?

            dtype_a, dtype, self.freq = np.fromfile(f, dtype=dt16, count=3)
            if dtype_a == 0x07:
                self.dtype = [None, np.float16, None, np.float32][dtype-1]
            else:
                self.dtype = [None, np.int16, None, np.int32][dtype-1]

            self.shift, self.u1, self.scale, self.dec = \
                np.fromfile(f, dtype=np.int16, count=4)#.astype(np.int32)

            self.name = decode_string(f.read(0x20))
            self.short_name = (decode_string(f.read(0x8)))
            self.unit = decode_string(f.read(0xc))

            self.u2, self.u3, self.u4, self.u5 = \
                np.fromfile(f, dtype=np.int16, count=4)#.astype(np.int32)

            # print_hex(f.read(28))

            # jump to data and read
            f.seek(self.data_ptr)
            try:
                self.data = np.fromfile(f,
                    count=self.data_len, dtype=self.dtype)

                self.data = self.data/self.scale * pow(10.,-self.dec) + self.shift

                if len(self.data)!=self.data_len:
                    raise ValueError("Not all data read!")

            except ValueError as v:
                print(v, self.num, self.name, self.freq,
                      hex(self.data_ptr), hex(self.data_len),
                      hex(len(self.data)),hex(f.tell()))
                # raise v

    def __str__(self):
        return 'chan %i: %s (%s) [%s], %i Hz'%(
            self.num, self.name,
            self.short_name, self.unit,
            self.freq)


def print_hex(bytes_, dtype=dt32):
    # type: (bytes, np.dtype) -> None
    """print the bytes as list of hex values
    """
    print(list(map(hex, np.frombuffer(bytes_, dtype=dtype))))


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
        chan_ = ldchan(f_, meta_ptr, len(chans))
        chans.append(chan_)
        meta_ptr = chan_.next_meta_ptr
    return chans


def read_ldfile(f_):
    # type: (str) -> (ldhead, list)
    """ Read an ld file, return header and list of channels
    """
    head_ = ldhead(f_)
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
        head_, chans = read_ldfile(f)

        print(head_)
        print(list(map(str,chans)))
        print()

        # create plots for all channels with the same frequency
        for f, g in groupby(chans, lambda x:x.freq):
            df = pd.DataFrame({i.name.lower(): i.data for i in g})
            df.plot()
            plt.show()
