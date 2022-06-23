#!/usr/bin/env python3

# Author:     Kris Demuynck
# Copyright:  2015-2019 UGent/iMinds

import numpy, gzip, sys


def open_read(fname):
    """Open the SPRaak file <fname> for reading. Return the file handler and the keyset."""
    fd = (
        fname
        if (hasattr(fname, "read"))
        else (
            gzip.open(fname, "rb")
            if (fname.endswith(".gz"))
            else (
                sys.stdin
                if ((fname == "-") or (fname == "stdin"))
                else open(fname, "rb")
            )
        )
    )
    kset = {}
    for line in fd:
        line = line.decode()
        if (line[0:1] == "#") and (line.lstrip("#") in ("\n", "\r\n", "")):
            break
        f = line.rstrip(" \t\n\r").split(None, 1)
        if f:
            kset[f[0]] = f[1] if (len(f) > 1) else None
    return (fd, kset)


def write_mat(fname, mat):
    """Write the numpy matrix <mat> to the SPRaak file <fname>."""
    dim1, dim2 = mat.shape
    type = str(mat.dtype)
    with (
        fname
        if (hasattr(fname, "write"))
        else (
            gzip.open(fname, "wb")
            if (fname.endswith(".gz"))
            else (
                sys.stdout
                if ((fname == "-") or (fname == "stdout"))
                else open(fname, "wb")
            )
        )
    ) as fd:
        fd.write(
            (
                ".spr\nDATA\tPARAM\nTYPE\t%s\nFORMAT\tBIN%s\nDIM1\t%i\nDIM2\t%i\n#\n"
                % (
                    type[0].upper() + "".join(filter(str.isdigit, type)),
                    ("01" if (sys.byteorder == "little") else "10"),
                    dim1,
                    dim2,
                )
            ).encode()
        )
        mat.tofile(fd)
    return mat


type_xlat = {
    "I8": numpy.int8,
    "I16": numpy.int16,
    "I32": numpy.int32,
    "I64": numpy.int64,
    "U8": numpy.uint8,
    "U16": numpy.uint16,
    "U32": numpy.uint32,
    "U64": numpy.uint64,
    "F32": numpy.float32,
    "F64": numpy.float64,
    "SHORT": numpy.int16,
    "INT": numpy.int32,
    "FLOAT": numpy.float32,
    "DOUBLE": numpy.float64,
}


def read_mat(fname, kset=None):
    """Read a numpy matrix from the SPRaak file <fname>. If you need the keyset, then pass along an empty dictionary as second argument -- it will contain the (key,values) pairs on return."""
    with (
        fname
        if (hasattr(fname, "read"))
        else (
            gzip.open(fname, "rb")
            if (fname.endswith(".gz"))
            else (
                sys.stdin
                if ((fname == "-") or (fname == "stdin"))
                else open(fname, "rb")
            )
        )
    ) as fd:
        if kset is None:
            kset = {}
        for line in fd:
            line = line.decode()
            if (line[0:1] == "#") and (line.lstrip("#") in ("\n", "\r\n", "")):
                break
            f = line.rstrip(" \t\n\r").split(None, 1)
            if f:
                kset[f[0]] = f[1] if (len(f) > 1) else None
        dim1, dim2, type = (
            int(kset.get("DIM1", kset.get("NENTRY", "-1"))),
            int(kset.get("DIM2", kset.get("NPARAM", "-1"))),
            kset.get("TYPE", kset.get("DATAFORMAT", "-1")).upper(),
        )
        fmt, dtype = (
            ("ASCII", type_xlat[type[2:]])
            if (type.startswith("A-"))
            else (kset.get("FORMAT", "BIN10"), type_xlat[type])
        )
        mat = numpy.fromfile(
            fd,
            dtype=dtype,
            count=(dim1 * dim2 if (min(dim1, dim2) >= 0) else -1),
            sep=(" " if (fmt == "ASCII") else ""),
        ).reshape((dim1, dim2))
    return mat.byteswap(inplace=True) if (fmt == "BIN10") else mat


if __name__ == "__main__":
    # test code: read + write a matrix
    write_mat(sys.argv[2], read_mat(sys.argv[1]))
