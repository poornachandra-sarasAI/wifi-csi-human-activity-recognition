"""Chip-specific packed-CSI reconstruction for BCM4366c0 at 80 MHz.

Each CSI observation arrives as 256 packed 32-bit words. Each word encodes
one subcarrier as a shared exponent plus a signed real and a signed
imaginary mantissa:

    bit 31                                                        bit 0
    +------------------------+------------------------+----------------+
    |   real mantissa (nman) |   imag mantissa (nman) |  exponent(nexp)|
    +------------------------+------------------------+----------------+

The field widths are a property of the chip, and the challenge forbids
porting the source authors' implementation, so rather than copying a
layout we RESOLVE it empirically and verifiably.

Resolution test (no labels, no validation data)
-----------------------------------------------
Room A and Room B were recorded with the same chip, bandwidth and
extraction framework. An 802.11ac 80 MHz channel has null subcarriers at
fixed positions (DC and the guard bands), so the positions of the deep
notches in the mean amplitude profile are a property of the radio, not of
the room. The correct layout is therefore the one whose null positions
coincide with Room B's, whose amplitudes are already given.

`resolve_layout()` scores every candidate layout by that agreement and is
what justifies the values recorded in configs/experiment.yaml.

Run:  python -m src.csi_decode --resolve
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src import config

# Candidate (mantissa_bits, exponent_bits) layouts. Each must satisfy
# 2 * nman + nexp <= 32.
CANDIDATE_LAYOUTS = [(12, 6), (11, 6), (10, 6), (12, 5), (11, 5), (10, 5), (9, 5)]


@dataclass(frozen=True)
class Layout:
    mantissa_bits: int
    exponent_bits: int
    apply_exponent: bool

    def __str__(self) -> str:
        exp = "with-exp" if self.apply_exponent else "no-exp"
        return f"nman={self.mantissa_bits} nexp={self.exponent_bits} {exp}"


def _sign_extend(values: np.ndarray, n_bits: int) -> np.ndarray:
    """Interpret n_bits-wide unsigned values as two's-complement signed."""
    half = 1 << (n_bits - 1)
    signed = values.astype(np.int64)
    return np.where(signed >= half, signed - (1 << n_bits), signed)


def unpack(packed: bytes | np.ndarray, layout: Layout) -> np.ndarray:
    """Packed CSI words -> complex CSI. Returns a complex64 array."""
    words = (
        np.frombuffer(packed, dtype="<u4")
        if isinstance(packed, (bytes, bytearray))
        else np.asarray(packed, dtype=np.uint32)
    )

    nman, nexp = layout.mantissa_bits, layout.exponent_bits
    man_mask = (1 << nman) - 1

    exponent = _sign_extend(words & ((1 << nexp) - 1), nexp)
    imag = _sign_extend((words >> nexp) & man_mask, nman)
    real = _sign_extend((words >> (nexp + nman)) & man_mask, nman)

    csi = real.astype(np.float64) + 1j * imag.astype(np.float64)
    if layout.apply_exponent:
        csi = csi * np.power(2.0, exponent.astype(np.float64))
    return csi.astype(np.complex64)


def unpack_frames(frames, layout: Layout) -> np.ndarray:
    """Decode a sequence of CsiFrame objects into an (n_obs, 256) amplitude array."""
    if not frames:
        return np.empty((0, 256), dtype=np.float32)
    stacked = np.frombuffer(b"".join(f.packed for f in frames), dtype="<u4")
    words = stacked.reshape(len(frames), -1)
    return np.abs(unpack(words, layout)).astype(np.float32)


# ---------------------------------------------------------------------
# Empirical layout resolution
# ---------------------------------------------------------------------


def _null_positions(profile: np.ndarray, quantile: float = 0.06) -> np.ndarray:
    """Indices of the deepest notches in a mean-amplitude profile."""
    threshold = np.quantile(profile, quantile)
    return np.flatnonzero(profile <= threshold)


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = set(a.tolist()), set(b.tolist())
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


def room_b_profile(cfg: config.Config, n_captures: int = 60) -> np.ndarray:
    """Mean amplitude per subcarrier over a sample of Room B captures."""
    from src import amplitude

    paths = sorted((cfg.path("intermediate") / "room_B").glob("*.csv"))[:n_captures]
    if not paths:
        raise FileNotFoundError(
            "no Room B intermediate files; run `python -m src.csv_reader` first"
        )
    means = [amplitude.read_capture(p).mean(axis=0) for p in paths]
    return np.mean(means, axis=0)


def room_a_profile(
    cfg: config.Config, layout: Layout, n_captures: int = 60
) -> np.ndarray:
    """Mean amplitude per subcarrier over a sample of Room A captures."""
    from src import pcap_reader

    root = cfg.path("raw_room_a")
    classes = [d for d in sorted(root.iterdir()) if d.is_dir() and d.name != "validation"]
    paths: list[Path] = []
    per_class = max(n_captures // max(len(classes), 1), 1)
    for folder in classes:
        paths.extend(sorted(folder.glob("*.pcap"))[:per_class])

    means = []
    for path in paths:
        frames, _ = pcap_reader.read_capture(path, cfg)
        if frames:
            means.append(unpack_frames(frames, layout).mean(axis=0))
    if not means:
        raise RuntimeError("no Room A frames decoded")
    return np.mean(means, axis=0)


def resolve_layout(cfg: config.Config, n_captures: int = 60) -> list[dict]:
    """Score every candidate layout against Room B's null-subcarrier positions."""
    profile_b = room_b_profile(cfg, n_captures)
    nulls_b = _null_positions(profile_b)

    results = []
    for nman, nexp in CANDIDATE_LAYOUTS:
        for apply_exponent in (False, True):
            layout = Layout(nman, nexp, apply_exponent)
            try:
                profile_a = room_a_profile(cfg, layout, n_captures)
            except Exception as exc:  # pragma: no cover - diagnostic path
                results.append({"layout": str(layout), "error": repr(exc)})
                continue

            nulls_a = _null_positions(profile_a)
            finite = np.isfinite(profile_a).all()
            results.append(
                {
                    "layout": str(layout),
                    "mantissa_bits": nman,
                    "exponent_bits": nexp,
                    "apply_exponent": apply_exponent,
                    "null_overlap": _jaccard(nulls_a, nulls_b),
                    "profile_corr": float(np.corrcoef(profile_a, profile_b)[0, 1])
                    if finite
                    else float("nan"),
                    "median": float(np.median(profile_a)),
                    "max": float(np.max(profile_a)),
                    "finite": bool(finite),
                }
            )

    results.sort(key=lambda r: r.get("null_overlap", -1), reverse=True)
    return results


def layout_from_config(cfg: config.Config | None = None) -> Layout:
    cfg = cfg or config.load()
    settings = cfg["ingest"]["room_a"]
    return Layout(
        mantissa_bits=settings["mantissa_bits"],
        exponent_bits=settings["exponent_bits"],
        apply_exponent=settings["apply_exponent"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolve", action="store_true", help="run the layout test")
    parser.add_argument("--captures", type=int, default=60)
    args = parser.parse_args()

    cfg = config.load()
    if not args.resolve:
        print(f"configured layout: {layout_from_config(cfg)}")
        return 0

    import pandas as pd

    table = pd.DataFrame(resolve_layout(cfg, args.captures))
    cols = [c for c in ["layout", "null_overlap", "profile_corr", "median", "max"] if c in table]
    print("\nCandidate layouts, best null-position agreement with Room B first:\n")
    print(table[cols].to_string(index=False))

    out = cfg.path("results") / "decode_layout_resolution.csv"
    table.to_csv(out, index=False)
    print(f"\nwritten -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
