#!/usr/bin/env python
# -*- coding: latin_1 -*-
import sys, re, os, codecs, gzip, subprocess


class Object(object):
    def __init__(self, **kv):
        self.__dict__.update(kv)


class Msg(object):
    def __init__(self, lvl, dst=sys.stderr, prog=None):
        self.cnt = 0
        self.lvl = lvl
        self.dst = dst
        if prog == None:
            prog = re.sub("^.*/", "", sys.argv[0])
        if prog == "":
            prog = "spr_parser"
        self.prog = prog

    def __call__(self, msg):
        routine = sys._getframe(1)
        self.dst.write(
            "%s %s(%s@%i): %s\n"
            % (self.lvl, self.prog, routine.f_code.co_name, routine.f_lineno, msg)
        )
        self.cnt += 1


error = Msg("ERROR")
warning = Msg("WARNING")
info = Msg("INFO")
progress = Msg("PROGRESS")


class ArgDecode(object):
    def print_help(cfg):
        error.dst.write("%s\n" % error.prog)
        for (op, desc) in cfg.opdesc.iteritems():
            error.dst.write(
                "\t%s\t%s%s\n\t\t%s\n"
                % (
                    op,
                    (
                        "<" + desc[0] + ">"
                        if (isinstance(desc[0], str))
                        else desc[0].__name__ + "()"
                    ),
                    ("(" + str(eval(desc[1])) + ")" if (desc[1] != None) else ""),
                    desc[2],
                )
            )
        sys.exit(1)

    def arg_decode(cfg, argv):
        dst = "argv0"
        deftype = None
        if len(argv) <= 1:
            ArgDecode.print_help(cfg)
        for arg in argv:
            if dst:
                if isinstance(deftype, list):
                    if not (hasattr(cfg, dst)):
                        setattr(cfg, dst, [])
                    getattr(cfg, dst).append(arg)
                elif hasattr(cfg, dst):
                    error("argument %s specified twice" % dst)
                else:
                    setattr(
                        cfg, dst, (arg if (deftype == None) else type(deftype)(arg))
                    )
                dst = None
            elif arg in cfg.opdesc:
                dst, deftype, ignore = cfg.opdesc[arg]
                if deftype != None:
                    deftype = eval(deftype)
                if not (isinstance(dst, str)):
                    dst = (dst)()
                elif isinstance(deftype, bool):
                    setattr(cfg, dst, not (deftype))
                    dst = None
            else:
                error("unknown option %s" % arg)
                ArgDecode.print_help(cfg)
        if isinstance(dst, str):
            error("value for option %s not specified" % dst)
        for (op, dst) in cfg.opdesc.iteritems():
            if (dst[1] != None) and not (hasattr(cfg, dst[0])):
                setattr(cfg, dst[0], eval(dst[1]))

    def __init__(cfg, opdesc):
        cfg.opdesc = opdesc
        cfg.opdesc["-help"] = (cfg.print_help, None, "Print help")


def kfile_open_r(fname):
    if (fname == "-") or (fname == "stdin"):
        fd = sys.stdin
    else:
        if not (os.path.exists(fname)) and os.path.exists(fname + ".gz"):
            fname += ".gz"
        fd = gzip.open(fname, "r") if (fname.endswith(".gz")) else open(fname, "r")
    kv = {}
    while 1:
        line = fd.readline()
        if not (line) or (line.startswith("#") and not (line.rstrip("#\n\r"))):
            break
        line = line.split(None, 1)
        if line:
            kv[line[0]] = line[1].strip() if (len(line) == 2) else ""
    kv.pop(".key", None)
    if kv.pop(".spr", None) != None:
        p = subprocess.Popen(
            ["spr_copy", "-H", "-i", "stdin", "-o", "key:stdout"],
            shell=False,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            close_fds=True,
        )
        p.stdin.write(".spr\n")
        p.stdin.writelines("%s\t%s\n" % (k, v) for k, v in kv.iteritems())
        p.stdin.write("#\n")
        p.stdin = p.stdin.close()
        kv = {"ENDIAN": ("LITTLE" if (kv.get("FORMAT", "BIN01") == "BIN01") else "BIG")}
        for line in p.stdout:
            if line.startswith("#") and not (line.rstrip("#\n\r")):
                break
            line = line.split(None, 1)
            if line:
                kv[line[0]] = line[1].strip() if (len(line) == 2) else ""
        kv.pop(".key", None)
        p.stdout.close()
        p.wait()
        p = None
    return (fd, kv)


def kfile_open_w(fname, kv):
    if (fname == "-") or (fname == "stdout"):
        fd = sys.stdout
    else:
        fd = gzip.open(fname, "w") if (fname.endswith(".gz")) else open(fname, "w")
    fd.write(".key\n")
    fd.writelines(
        ("%s%s%s\n" % (k, ("\t" if (v) else ""), v) for k, v in kv.iteritems())
    )
    fd.write("#\n")
    return fd
