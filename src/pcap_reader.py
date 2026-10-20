"""Room A (Experiment 2) packet parsing: .pcap -> ordered Nexmon CSI payloads.

This module does packet framing only; the chip-specific CSI reconstruction
lives in src/csi_decode.py. A general-purpose PCAP parser is permitted by
the challenge, but the format here is simple and completely regular, so we
read it directly rather than taking a dependency:

Verified across all 1084 Room A captures:
  * classic libpcap, little-endian, microsecond timestamps, linktype 1
  * every frame is exactly 1084 bytes, with zero malformed frames
  * Ethernet(14) + IPv4(20) + UDP(8) = 42 bytes of header, UDP 5500->5500,
    source MAC is the ASCII string "NEXMON"
  * the remaining 1042 bytes are an 18-byte Nexmon header (magic 0x1111)
    followed by 1024 bytes = 256 packed 32-bit CSI words

Temporal order is the order frames appear in the file, which is preserved
throughout (challenge spec hint 2).

Run:  python -m src.pcap_reader --inspect
"""

from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

from src import config

_PCAP_MAGIC_LE = 0xA1B2C3D4
_PCAP_MAGIC_LE_NANO = 0xA1B23C4D
_GLOBAL_HEADER_BYTES = 24
_PACKET_HEADER_BYTES = 16

# Nexmon CSI header layout (18 bytes), offsets within the UDP payload.
_OFF_MAGIC = 0
_OFF_RSSI = 2
_OFF_FCTL = 3
_OFF_SRC_MAC = 4
_OFF_SEQ = 10
_OFF_CORE_STREAM = 12
_OFF_CHANSPEC = 14
_OFF_CHIP_VER = 16

SEQ_UNSEQUENCED = 0xFFFF


@dataclass(frozen=True)
class CsiFrame:
    """One Nexmon CSI frame, still packed."""

    timestamp: float
    seq: int
    core_stream: int
    chanspec: int
    chip_version: int
    rssi: int
    packed: bytes  # 1024 bytes = 256 packed 32-bit words

    @property
    def is_sequenced(self) -> bool:
        """False for frames whose 802.11 sequence-control field reads 0xFFFF.

        These frames still carry valid CSI; see read_capture().
        """
        return self.seq != SEQ_UNSEQUENCED


def iter_raw_frames(path: Path):
    """Yield (timestamp, frame_bytes) in file order."""
    data = path.read_bytes()
    if len(data) < _GLOBAL_HEADER_BYTES:
        raise ValueError(f"{path.name}: too short to be a pcap file")

    magic = struct.unpack("<I", data[:4])[0]
    if magic == _PCAP_MAGIC_LE:
        divisor = 1e6
    elif magic == _PCAP_MAGIC_LE_NANO:
        divisor = 1e9
    else:
        raise ValueError(f"{path.name}: unsupported pcap magic 0x{magic:08x}")

    offset = _GLOBAL_HEADER_BYTES
    while offset + _PACKET_HEADER_BYTES <= len(data):
        ts_sec, ts_frac, n_included, _ = struct.unpack(
            "<IIII", data[offset : offset + _PACKET_HEADER_BYTES]
        )
        offset += _PACKET_HEADER_BYTES
        frame = data[offset : offset + n_included]
        offset += n_included
        if len(frame) < n_included:  # truncated trailing record
            break
        yield ts_sec + ts_frac / divisor, frame


def parse_frame(timestamp: float, frame: bytes, cfg: config.Config) -> CsiFrame | None:
    """Parse one Ethernet frame into a CsiFrame, or None if it is not CSI."""
    settings = cfg["ingest"]["room_a"]
    payload = frame[settings["udp_payload_offset"] :]

    header_bytes = settings["nexmon_header_bytes"]
    if len(payload) < header_bytes:
        return None
    if struct.unpack("<H", payload[_OFF_MAGIC : _OFF_MAGIC + 2])[0] != settings["magic"]:
        return None

    packed = payload[header_bytes:]
    if len(packed) != cfg["ingest"]["n_subcarriers"] * 4:
        return None

    return CsiFrame(
        timestamp=timestamp,
        seq=struct.unpack("<H", payload[_OFF_SEQ : _OFF_SEQ + 2])[0],
        core_stream=struct.unpack(
            "<H", payload[_OFF_CORE_STREAM : _OFF_CORE_STREAM + 2]
        )[0],
        chanspec=struct.unpack("<H", payload[_OFF_CHANSPEC : _OFF_CHANSPEC + 2])[0],
        chip_version=struct.unpack("<H", payload[_OFF_CHIP_VER : _OFF_CHIP_VER + 2])[0],
        rssi=payload[_OFF_RSSI],
        packed=packed,
    )


def read_capture(
    path: Path, cfg: config.Config | None = None
) -> tuple[list[CsiFrame], dict[str, int]]:
    """Read one capture's CSI frames in temporal order.

    Frames whose sequence-control field reads 0xFFFF are KEPT by default.
    They carry valid CSI: 976-992 of their 1024 payload bytes are non-zero
    and their amplitude distribution matches that of sequenced frames.
    Dropping them leaves a median of 111 observations per capture, which
    puts the official 150-observation window out of reach for every one of
    the 1084 Room A captures. See configs/experiment.yaml.
    """
    cfg = cfg or config.load()
    keep_unsequenced = cfg["ingest"]["room_a"]["keep_unsequenced_frames"]

    frames: list[CsiFrame] = []
    report = {"pcap_frames": 0, "pcap_not_csi": 0, "pcap_unsequenced": 0}

    for timestamp, raw in iter_raw_frames(path):
        report["pcap_frames"] += 1
        parsed = parse_frame(timestamp, raw, cfg)
        if parsed is None:
            report["pcap_not_csi"] += 1
            continue
        if not parsed.is_sequenced:
            report["pcap_unsequenced"] += 1
            if not keep_unsequenced:
                continue
        frames.append(parsed)

    report["pcap_kept"] = len(frames)
    return frames, report


# Source folder name -> canonical activity label.
_ACTIVITY_FOLDERS = {
    "sit": "SIT",
    "stand": "STAND",
    "walk": "WALK",
    "sitdown": "SITDOWN",
    "standup": "STANDUP",
    "fall": "FALL",
}


def iter_source_captures(cfg: config.Config):
    """Yield (path, activity, is_validation) for every Room A capture.

    The supplied validation/ directory is the Room A validation set and is
    never moved into development (challenge spec section 9).
    """
    root = cfg.path("raw_room_a")
    for folder, activity in sorted(_ACTIVITY_FOLDERS.items()):
        for path in sorted((root / folder).glob("*.pcap")):
            yield path, activity, False
        val_dir = root / "validation" / folder
        if val_dir.is_dir():
            for path in sorted(val_dir.glob("*.pcap")):
                yield path, activity, True


def ingest(cfg: config.Config | None = None, limit: int | None = None):
    """Decode every Room A capture into the common intermediate form."""
    import pandas as pd

    from src import amplitude, cleaning, csi_decode

    cfg = cfg or config.load()
    layout = csi_decode.layout_from_config(cfg)
    out_dir = cfg.path("intermediate") / "room_A"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    sources = list(iter_source_captures(cfg))
    if limit is not None:
        sources = sources[:limit]

    for path, activity, is_validation in sources:
        frames, pcap_report = read_capture(path, cfg)
        decoded = csi_decode.unpack_frames(frames, layout)
        clean, qc_report = cleaning.clean_capture(decoded, cfg)

        # "validation" in the id keeps the supplied split visible in the
        # filename for auditing; the split itself lives in the manifest.
        prefix = "roomA_val__" if is_validation else "roomA__"
        capture_id = f"{prefix}{path.stem}"
        n_written = amplitude.write_capture(clean, out_dir / f"{capture_id}.csv")

        records.append(
            {
                "capture_id": capture_id,
                "room": "A",
                "activity": activity,
                "source_format": "pcap",
                "source_file": str(path.relative_to(cfg.path("raw_room_a"))),
                "supplied_validation": is_validation,
                "n_obs_raw": len(decoded),
                "n_obs_clean": n_written,
                **pcap_report,
                **qc_report,
            }
        )

    return pd.DataFrame(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", action="store_true", help="summarise one capture")
    parser.add_argument("--capture", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = config.load()

    if not args.inspect:
        table = ingest(cfg, limit=args.limit)
        out = cfg.path("results") / "ingest_room_A.csv"
        table.to_csv(out, index=False)
        print(f"Room A: {len(table)} captures -> {cfg.path('intermediate') / 'room_A'}")
        print(f"  supplied validation captures: {int(table.supplied_validation.sum())}")
        print("\nper-activity capture counts:")
        print(table["activity"].value_counts().to_string())
        print("\nobservations per capture after QC:")
        print(table["n_obs_clean"].describe()[["min", "50%", "max"]].to_string())
        print(f"\n  unsequenced frames kept: {int(table.pcap_unsequenced.sum())}")
        print(f"  frames rejected by QC  : {int(table.qc_total_removed.sum())}")
        print(f"  captures under 150 obs : {int((table.n_obs_clean < 150).sum())}")
        print(f"\ningest report -> {out}")
        return 0

    path = args.capture or (cfg.path("raw_room_a") / "sit" / "sit1.pcap")
    frames, report = read_capture(path, cfg)
    print(f"{path.name}: {report}")
    if frames:
        first = frames[0]
        print(f"  chanspec     0x{first.chanspec:04x}")
        print(f"  chip version 0x{first.chip_version:04x}")
        print(f"  core/stream  0x{first.core_stream:04x}")
        print(f"  rssi field   0x{first.rssi:02x}")
        print(f"  packed bytes {len(first.packed)} ({len(first.packed) // 4} words)")
        span = frames[-1].timestamp - frames[0].timestamp
        print(f"  duration     {span:.3f} s over {len(frames)} frames")
        print(f"  mean gap     {span / max(len(frames) - 1, 1) * 1000:.2f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
