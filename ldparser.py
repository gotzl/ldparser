""" Parser for MoTec ld files

Code created through reverse engineering the data format.
So far, the decoding looks as follows:

### header
description       length
----------------------
unknown           0x8
channel meta ptr  0x4
channel data ptr  0x4
unknown           0x14
descr ptr         0x4
unknown           0x36
date              0x10
unknown           0x10
time              0x10
unknown           0x10
name              0x40
unknown           0x80
subject           0x40

### channel meta data
description       length
----------------------
prev_addr         0x4
next_addr         0x4
data ptr          0x4
n_data            0x4
somecnt           0x4
datawordsize      0x2
rec freq          0x2
unknown           0x8
name              0x28
unit              0x0c
unknown           0x18
unknown           0x10
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

            f.seek(0x80,1)
            self.subject = decode_string(f.read(0x40))

            f.seek(descr_)
            self.descr1 = decode_string(f.read(0x10))

            f.seek(0x470, 1)
            descr_ = np.fromfile(f, dtype=dt32, count=1)[0]

            f.seek(descr_)
            self.descr2 = decode_string(f.read(0x10))

        self.datetime =  datetime.datetime.strptime(
                    '%s %s'%(date, time), '%d/%m/%Y %H:%M:%S')

    def __str__(self):
        return 'name:    %s\n' \
               'subject: %s\n' \
               'desc1:   %s descr2: %s\n'%(self.name, self.subject, self.descr1, self.descr2)


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

            f.seek(4, 1) # some count, not needed?

            dtype, self.freq = np.fromfile(f, dtype=dt16, count=2)
            self.dtype = [np.float16, np.float32][(dtype//2)-1]

            self.dunno = np.fromfile(f, dtype=dt32, count=2)

            self.name = decode_string(f.read(0x20))
            self.short_name = (decode_string(f.read(0x8)))
            self.unit = decode_string(f.read(0xc))
            # print_hex(f.read(28))

            # jump to data and read
            f.seek(self.data_ptr)
            try:
                self.data = np.fromfile(f,
                    count=self.data_len, dtype=self.dtype)

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
