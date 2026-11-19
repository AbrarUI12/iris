"""Stream-extract entries from a truncated ZIP (no central directory) by walking local
file headers. Supports stored/deflate and ZIP64 size extras. Only used for public data.
python scripts/salvage_zip.py archive.zip out_dir [--list] [--prefix lit/]"""
import struct, sys, zlib
from pathlib import Path


def walk(path, out=None, prefix=None, list_only=False):
    counts = {}
    with open(path, "rb") as f:
        while True:
            hdr = f.read(30)
            if len(hdr) < 30 or hdr[:4] != b"PK\x03\x04":
                break
            (_, ver, flag, method, _, _, crc, csize, usize, nlen, xlen) = struct.unpack("<IHHHHHIIIHH", hdr)
            name = f.read(nlen).decode("utf-8", "replace")
            extra = f.read(xlen)
            i = 0
            while i + 4 <= len(extra):
                hid, hsz = struct.unpack("<HH", extra[i:i + 4])
                if hid == 1:
                    vals = extra[i + 4:i + 4 + hsz]
                    j = 0
                    if usize == 0xFFFFFFFF:
                        usize = struct.unpack("<Q", vals[j:j + 8])[0]; j += 8
                    if csize == 0xFFFFFFFF:
                        csize = struct.unpack("<Q", vals[j:j + 8])[0]; j += 8
                i += 4 + hsz
            if flag & 8:
                raise SystemExit("data descriptor entries not supported")
            data = f.read(csize)
            if len(data) < csize:
                break  # truncated entry
            top = name.split("/")[0]
            counts[top] = counts.get(top, 0) + 1
            if list_only or name.endswith("/"):
                continue
            if prefix and not any(name.startswith(p) for p in prefix):
                continue
            raw = zlib.decompress(data, -15) if method == 8 else data
            p = Path(out) / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(raw)
    return counts


if __name__ == "__main__":
    a = sys.argv[1:]
    lst = "--list" in a
    pre = a[a.index("--prefix") + 1].split(",") if "--prefix" in a else None
    print(walk(a[0], None if lst else a[1], pre, lst))
