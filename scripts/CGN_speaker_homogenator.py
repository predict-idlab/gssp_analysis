#!/usr/bin/env python3
import subprocess, os, sys, itertools, spr_io, wave, argparse, numpy


def get_audio(x, t0, t1, chan=None):
    if (x.shape[1] == 1) or (chan == 0):
        return x[t0:t1, 0]
    elif chan == 1:
        return x[t0:t1, 1]
    else:
        return (x[t0:t1, 0] + x[t0:t1, 1]) * 0.5


def make_new_data(new_cor, pending, args, fname, x, awd_chan, awd_bw):
    for desc, blk in pending.items():
        if not (isinstance(blk, tuple)):
            continue
        spkr = desc[0] if (isinstance(desc, tuple)) else desc
        if spkr in args.UNK:
            continue
        xname = (fname, spkr.lstrip("#"))
        chan, bw = awd_chan.get(xname, None), awd_bw.get(xname, 16)
        if bw != 16:
            sys.stderr.write("INFO: ingoring telephone speech %s-%s\n" % xname)
            continue
        part = desc[1] if (isinstance(desc, tuple)) else None
        data_blk, seg_blk = blk
        xname = (
            fname + "-" + spkr.lstrip("#") + ("-%04i" % part if (part != None) else "")
        )
        if not (
            isinstance(data_blk[-1], list)
        ):  # end with join to next, but there is no next => compensate
            data_blk[-2][1] -= data_blk[-1] * 0.5
            data_blk.pop()
        pos = 0
        T1 = x.shape[0]
        for blk in data_blk[1:]:  # skip first element: is size indication
            pos += (
                int(blk[1] * 16000 + 0.5) - min(int(blk[0] * 16000 + 0.5), T1)
                if (isinstance(blk, list))
                else -2 * int(blk * 8000 + 0.5)
            )
        assert abs(pos - data_blk[0] * 16000) < len(data_blk) / 2
        # lenght should be te same, except for round-off errors
        if pos < 0:
            sys.stderr.write("ERROR: pos=%i, blk=%s\n" % (pos, repr(data_blk)))
        xs = numpy.zeros(pos, dtype=numpy.float32)
        pos = overlap = 0
        for blk in data_blk[1:]:
            if isinstance(blk, list):
                t0, t1 = int(blk[0] * 16000 + 0.5), min(int(blk[1] * 16000 + 0.5), T1)
                if overlap:
                    t0 += overlap
                    xs[pos - overlap : pos] += (
                        get_audio(x, t0 - overlap, t0, chan)
                        * numpy.arange(overlap, dtype=numpy.float32)
                        / float(overlap)
                    )
                    overlap = 0
                xs[pos : pos + t1 - t0] = get_audio(x, t0, t1, chan)
                pos += t1 - t0
            else:
                overlap = 2 * int(blk * 8000 + 0.5)
                xs[pos - overlap : pos] *= numpy.arange(
                    overlap, 0, -1, dtype=numpy.float32
                ) / float(overlap)
        if args.dst:
            wname = os.path.join(args.dst, xname + ".wav")
            path = os.path.split(wname)[0]
            mkdir = []
            while not (os.path.exists(path)):
                mkdir.append(path)
                path = os.path.split(path)[0]
            while mkdir:
                os.mkdir(mkdir.pop())
            with wave.open(wname, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(xs.astype(numpy.int16).data)
        xs = None
        for seg in seg_blk:
            new_cor.append(
                "%s\t%s\t%.3f %.3f\t%s\n" % (xname, seg[0], seg[1], seg[2], spkr)
            )
            xname = "-"
    return {}


def read_cor_ext(fname):
    """Read missing channel & bandwidth info."""
    chan_xlat = {"L": 0, "R": 1}
    awd_chan_cnt = {}
    awd_bw_cnt = {}
    awd_chan = {}
    awd_bw = {}
    with open(fname, "r") as fd:
        for line in fd:
            a = line.split()
            if len(a) != 6:
                sys.stderr.write(
                    "ERROR: Invalid awd channel info in '%s':\n\t%s\n" % (fname, line)
                )
            a2 = chan_xlat.get(a[2], None)
            awd_chan[(a[0], a[1])] = a2
            key = (a[0], a[1], a2)
            awd_chan_cnt[key] = awd_chan_cnt.get(key, 0) + int(a[5])
            key = (a[0], a[1], int(a[3]))
            awd_bw_cnt[key] = awd_bw_cnt.get(key, 0) + int(a[5])
    # clean-up channel assignment
    for key in tuple(awd_chan):
        L = awd_chan_cnt.get((key[0], key[1], 0), 0)
        R = awd_chan_cnt.get((key[0], key[1], 1), 0)
        if L and R:
            if key[1] == "UNKNOWN":
                sys.stderr.write(
                    "INFO: Unknown speaker in %s maps to two channels -- mixing channels\n"
                    % key[0]
                )
                awd_chan[key] = None
            else:
                awd_chan[key] = 0 if (L >= R) else 1
    # clean-up bandwidth detection
    for (
        key
    ) in (
        awd_chan
    ):  # ignores 'UNKNOWN' speakers for which we could not determine the channel
        cnt8 = awd_bw_cnt.get((key[0], key[1], 8), 0)
        cnt16 = awd_bw_cnt.get((key[0], key[1], 16), 0)
        if (not (cnt8) or not (cnt16)) or (max(cnt8, cnt16) > 4 * min(cnt8, cnt16)):
            awd_bw[key] = 8 if (cnt8 > cnt16) else 16
        elif (cnt16 + 4000) / (cnt8 + 1000) > 2:
            awd_bw[key] = 16
        else:
            sys.stderr.write(
                "WARNING: Speaker %s in %s maps to two bandwidths (cnt8/16=%i/%i) -- assuming telephone\n"
                % (key[1], key[0], cnt8, cnt16)
            )
            awd_bw[key] = 8
    return (awd_chan, awd_bw)


parser = argparse.ArgumentParser(
    description="Copy all audiofiles in CGN to single speaker 16kHz mono wav files and create the corresponding corpus file.",
    formatter_class=argparse.RawTextHelpFormatter,
)
parser.add_argument(
    "-MERGE",
    dest="MERGE",
    action="store_true",
    default=False,
    help="By default, the files are plit in speaker homogeneous part; but different part belong to the same speaker are not merged. Specify this flag to get one file per speaker.",
)
parser.add_argument(
    "-UNK",
    dest="UNK",
    action="store_true",
    default=False,
    help="Also extra audio from the <unknown> speaker(s) -- this data is not guaranteed to be homogenous in speaker-ID.",
)
parser.add_argument(
    "-cor_base",
    dest="cor_base",
    action="store",
    required=False,
    type=str,
    default="/home/speech/krdmuync/exp/ASR_Dutch/cor/",
    help="Where to find the content description of the CGN DBase.",
)
parser.add_argument(
    "-cor",
    dest="cor",
    action="store",
    required=False,
    type=str,
    help="Content description of the data to process (can also be specified as component + region).",
)
parser.add_argument(
    "-cor_ext",
    dest="cor_ext",
    action="store",
    required=False,
    type=str,
    default="/home/speech/krdmuync/exp/ASR_Dutch/cor/info_awd_chan_bw.txt",
    help="Extra info concerning the content (bandwidth, dominant channel).",
)
parser.add_argument(
    "-spkr_ndx",
    dest="spkr_ndx",
    action="store",
    required=False,
    type=int,
    default=4,
    help="Where to find the spkr-ID in the corpus file (field number, base0).",
)
parser.add_argument(
    "-gap_sz",
    dest="gap_sz",
    action="store",
    required=False,
    type=float,
    default=1.0,
    help="Shrink gaps longer than <gap_sz> seconds (use -1 for infinity).",
)
parser.add_argument(
    "-comp",
    dest="comp",
    action="store",
    required=False,
    type=str,
    help="CGN component(s) to process.",
)
parser.add_argument(
    "-region",
    dest="region",
    action="store",
    required=False,
    type=str,
    help="Region(s) (vl/nl) to process.",
)
parser.add_argument(
    "-src",
    dest="src",
    action="store",
    required=False,
    type=str,
    default="/speech/data/cgn/wav",
    help="Where to find the audio.",
)
parser.add_argument(
    "-dst",
    dest="dst",
    action="store",
    required=False,
    type=str,
    help="Where to write the speaker homogenized audio.",
)
parser.add_argument(
    "-ncor",
    dest="ncor",
    action="store",
    required=False,
    type=str,
    help="Where to write the new corpus file.",
)
args = parser.parse_args()

todo = []
if args.comp and args.region:
    todo.extend(
        os.path.join(args.cor_base, "comp-" + c + "." + r + ".cor")
        for c, r in itertools.product(args.comp.split(), args.region.split())
    )
if args.cor:
    todo.append(args.cor)
if not (todo):
    sys.stderr.write(
        "ERROR: nothing to do; either specify argument -cor '<corpus_file>' or specify -comp 'a b ...' and -region 'vl nl'\n"
    )
    exit(1)
cor_ext = {}
awd_chan, awd_bw = read_cor_ext(args.cor_ext) if (args.cor_ext) else (None, None)
spkr_ndx = args.spkr_ndx
gap_sz = args.gap_sz
gap_sz2 = (gap_sz + 0.5) * 0.5
MERGE = 1 if (args.MERGE) else 0
args.UNK = set(() if (args.UNK) else ("#UNKNOWN",))
new_cor = []
for cor in todo:
    sys.stderr.write("INFO: handling %s\n" % cor)
    fd, kset = spr_io.open_read(cor)
    CTB = kset.get("TIMEBASE", "DISCRETE") == "CONTINUOUS"
    FSHIFT = 1.0 if (CTB) else float(kset.get("FSHIFT", "0.01"))
    fname = None
    xcor = []
    for line in fd:
        f = line.decode(encoding="latin1").split()
        if f[0] != "-":
            fname = f[0]
        t0, t1 = float(f[2]), float(f[3])
        t1 = t1 if (CTB) else (t0 + t1) * FSHIFT
        t0 *= FSHIFT
        spkr = f[spkr_ndx]
        if t1 <= t0:
            sys.stderr.write(
                "ERROR: negative length segment %s [%.3f,%.3f]\n" % (fname, t0, t1)
            )
        xcor.append((fname, t0, t1, spkr, f[1]))
    fd = fd.close()
    xcor.sort()
    fname = prev_spkr = None
    T1 = tx = 0.0
    pending = {}
    for f in xcor:
        if f[0] != fname:
            if len(pending):
                pending = make_new_data(
                    new_cor, pending, args, fname, x, awd_chan, awd_bw
                )
            data_blk = seg_blk = prev_spkr = None
            T1 = tx = 0.0
            fname = f[0]
            # sys.stderr.write("INFO: handling %s\n"%fname);
            with wave.open(os.path.join(args.src, fname + ".wav"), "rb") as wav:
                wav_info = wav.getparams()
                assert (wav_info.sampwidth == 2) and (wav_info.framerate == 16000)
                x = (
                    numpy.frombuffer(
                        wav.readframes(wav.getnframes()), dtype=numpy.int16
                    )
                    .astype(numpy.float32)
                    .reshape((-1, wav_info.nchannels))
                )
            sf = int(kset.get("SAMPLEFREQ", "16000"))
        t0, t1, spkr, desc = f[1:5]
        if T1 > t0:  # overlapping speech -- annoying
            sys.stderr.write(
                "WARNING: overlapping speech in '%s' @ [%.3f,%.3f]\n" % (fname, t0, T1)
            )
            T1 = t0
        if spkr == prev_spkr:
            if (gap_sz == -1) or (t0 <= T1 + gap_sz):  # extend the block
                data_blk[0] += t1 - data_blk[-1][1]
                data_blk[-1][1] = t1
            else:  # skip large gap + merge (overlap add)
                data_blk[0] += t1 - t0 + gap_sz
                data_blk[-1][1] += gap_sz2
                data_blk.append(0.5)
                # indicate a 0.5 second overlap add combination
                data_blk.append([t0 - gap_sz2, t1])
                tx += t0 - T1 - gap_sz
        else:
            td = (t0 - T1) * 0.5
            to = min(td, 0.25) * MERGE
            if data_blk != None:
                data_blk[0] += td + to
                data_blk[-1][1] += td + to
                if to:
                    data_blk.append(to * 2)
                    # indicate a <to> second overlap add combination (when merging); if nothing is added, this part of the signal nneds to be removed
            if MERGE:
                if spkr in pending:
                    data_blk, seg_blk = pending[spkr]
                    to1 = data_blk[-1] * 0.5
                    # overlap at end
                    if to1 > to:  # problem: overlap must be smaller
                        data_blk[-1] = to * 2
                        data_blk[-2][1] -= to1 - to
                    else:
                        to = min(to1, to)
                    data_blk[0] += t1 - t0 + td
                    data_blk.append([t0 - td - to, t1])
                    tx = data_blk[0] - t1
                else:
                    data_blk = [t1 - t0 + td, [t0 - td, t1]]
                    seg_blk = []
                    tx = t0 - td
                    pending[spkr] = (data_blk, seg_blk)
            else:
                data_blk = [t1 - t0 + td, [t0 - td, t1]]
                seg_blk = []
                tx = t0 - td
                pending[spkr] = part = pending.get(spkr, 0) + 1
                pending[(spkr, part)] = (data_blk, seg_blk)
        seg_blk.append((desc, t0 - tx, t1 - tx))
        prev_spkr = spkr
        T1 = t1
    if len(pending):
        pending = make_new_data(new_cor, pending, args, fname, x, awd_chan, awd_bw)

if args.ncor:
    with open(args.ncor, "w") as fd:
        fd.write(
            ".spr\nCORPUS\t%s\nTIMEBASE\tCONTINUOUS\nDIM1\t%i\nFIELD_DESC\tfname orthography begin_time end_time spkr-ID\n#\n"
            % ("CGN", len(new_cor))
        )
        fd.writelines(new_cor)
