"""2D laser-engrave occupancy used as a world-space stock decal."""

from __future__ import annotations

import math

import numpy as np

from carveracontroller.addons.stock.simulator.carvers.array_mesh import (
    compress_array,
    compressed_nbytes,
    decompress_array,
)
from carveracontroller.addons.stock.stock_geometry import StockBounds

LASER_MODE_PLANAR = "planar"
LASER_MODE_CYLINDRICAL = "cylindrical"

# Decal is independent of occupancy: finer cells, ~0.1 mm beam, GPU-size cap.
LASER_RES_MULT = 4
LASER_CELL_MIN_MM = 0.025
LASER_CELL_MAX_MM = 0.05
LASER_MAX_DIM = 4096
LASER_BEAM_RADIUS_MIN_MM = 0.08
LASER_BEAM_RADIUS_MAX_MM = 0.12

# Firmware laser PWM is S in 0–1. Gamma-lift so low duty still reads on beige.
LASER_PREVIEW_GAMMA = 0.3


def laser_burn_uint8(power_s: float | None) -> np.uint8:
    """Map laser S (0–1 PWM) to a preview luminance texel.

    Missing S is full power. S above 1 is full unless it looks like mill RPM.
    """
    if power_s is None:
        return np.uint8(255)
    s = float(power_s)
    if s <= 0.0:
        return np.uint8(0)
    if s > 1.0:
        if s > 100.0 + 1e-9:
            return np.uint8(0)
        s = 1.0
    visual = s**LASER_PREVIEW_GAMMA
    return np.uint8(min(255, int(round(visual * 255.0))))


def pick_laser_cell_size_mm(
    bounds: StockBounds,
    occupancy_cell_mm: float,
    *,
    cylindrical: bool = False,
    max_dim: int = LASER_MAX_DIM,
) -> float:
    """Cell size for the world-space decal; at least 4× occupancy, capped for GPU memory."""
    occ = max(float(occupancy_cell_mm), 1e-9)
    cell = min(occ / float(LASER_RES_MULT), LASER_CELL_MAX_MM)
    cell = max(cell, LASER_CELL_MIN_MM)
    sx, sy, sz = bounds.size
    if cylindrical:
        span = max(float(sx), math.pi * min(float(sy), float(sz)), 1e-6)
    else:
        span = max(float(sx), float(sy), 1e-6)
    cap = max(int(max_dim), 8)
    if span / cell > cap:
        cell = span / cap
    return float(cell)


def stroke_parts(item) -> tuple:
    """``(p0, p1, a0, a1, power)`` from a tuple or a carve job."""
    if isinstance(item, tuple):
        p0, p1 = item[0], item[1]
        a0 = float(item[2]) if len(item) > 2 else 0.0
        a1 = float(item[3]) if len(item) > 3 else a0
        power = item[4] if len(item) > 4 else None
        return p0, p1, a0, a1, power
    return item.p0, item.p1, float(item.a0), float(item.a1), item.spindle_s


def stroke_radius_mm(cell_size: float) -> float:
    """Beam width in mm. Tied to the decal cell, not occupancy, so rasters do not blob."""
    return max(LASER_BEAM_RADIUS_MIN_MM, min(LASER_BEAM_RADIUS_MAX_MM, 0.9 * float(cell_size)))


LASER_SNAP_FULL = "full"
LASER_SNAP_SPARSE_FULL = "sparse_full"
LASER_SNAP_DELTA = "delta"
LASER_SNAP_DROP = "drop"

# COO row is (iu, iv, value); 2×uint32 + uint8, padded in a uint32 triple.
_DELTA_BYTES_PER_TEXEL = 9
_DIRTY_LOG_CAP = 262144


class LaserDirtyLog:
    """Index sink shared with native writers. ``state`` is ``[n, overflow]`` int32."""

    __slots__ = ("iu", "iv", "state")

    def __init__(self, cap: int = _DIRTY_LOG_CAP):
        cap = max(int(cap), 1)
        self.iu = np.empty(cap, dtype=np.int32)
        self.iv = np.empty(cap, dtype=np.int32)
        self.state = np.zeros(2, dtype=np.int32)

    @property
    def n(self) -> int:
        return int(self.state[0])

    @property
    def overflow(self) -> bool:
        return bool(self.state[1])

    @property
    def cap(self) -> int:
        return int(self.iu.shape[0])

    def reset(self) -> None:
        self.state[:] = 0

    def native_args(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (self.iu, self.iv, self.state)

    def append_indices(self, iu: np.ndarray, iv: np.ndarray) -> None:
        if self.overflow or iu.size == 0:
            return
        iu = np.asarray(iu, dtype=np.int32).reshape(-1)
        iv = np.asarray(iv, dtype=np.int32).reshape(-1)
        if iu.size != iv.size:
            self.state[1] = 1
            return
        n = self.n
        room = self.cap - n
        if room <= 0:
            self.state[1] = 1
            return
        take = min(int(iu.size), room)
        self.iu[n : n + take] = iu[:take]
        self.iv[n : n + take] = iv[:take]
        self.state[0] = n + take
        if take < int(iu.size):
            self.state[1] = 1


def compress_laser_sparse(rs: np.ndarray, cs: np.ndarray, vals: np.ndarray, shape: tuple[int, ...]) -> object:
    """Pack changed texels as a zlib’d ``(N, 3)`` uint32 table plus the map shape."""
    pack = np.empty((int(rs.size), 3), dtype=np.uint32)
    pack[:, 0] = rs
    pack[:, 1] = cs
    pack[:, 2] = np.asarray(vals, dtype=np.uint8)
    return (tuple(int(s) for s in shape), compress_array(pack))


def decompress_laser_sparse(blob: object) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    """Inverse of :func:`compress_laser_sparse`."""
    shape, packed = blob  # type: ignore[misc]
    pack = decompress_array(packed)
    return pack[:, 0], pack[:, 1], pack[:, 2].astype(np.uint8, copy=False), tuple(int(s) for s in shape)


def laser_diff_payload(prev: np.ndarray | None, curr: np.ndarray) -> object | None:
    """Tagged snapshot of ``curr`` vs ``prev``. ``None`` if unchanged.

    Falls back to a full map when a sparse COO would be larger than half the bitmap.
    Missing or mismatched ``prev`` is a full snapshot (first paint / new shape).
    """
    curr = np.ascontiguousarray(curr)
    if prev is None or prev.shape != curr.shape:
        return (LASER_SNAP_FULL, compress_array(curr))
    changed = prev != curr
    n = int(np.count_nonzero(changed))
    if n == 0:
        return None
    if n * _DELTA_BYTES_PER_TEXEL > curr.nbytes / 2:
        return (LASER_SNAP_FULL, compress_array(curr))
    rs, cs = np.nonzero(changed)
    return (LASER_SNAP_DELTA, compress_laser_sparse(rs, cs, curr[rs, cs], curr.shape))


def _laser_full_payload(curr: np.ndarray) -> object:
    """Dense zlib full, or sparse-full when burned texels are cheap enough."""
    burned = curr != 0
    n = int(np.count_nonzero(burned))
    if n > 0 and n * _DELTA_BYTES_PER_TEXEL <= curr.nbytes / 2:
        rs, cs = np.nonzero(burned)
        return (LASER_SNAP_SPARSE_FULL, compress_laser_sparse(rs, cs, curr[rs, cs], curr.shape))
    return (LASER_SNAP_FULL, compress_array(curr))


def apply_laser_tagged(laser: LaserMap, payload: object) -> None:
    """Apply a tagged laser extra onto ``laser`` (caller ensures the map exists)."""
    if not isinstance(payload, tuple) or not payload:
        laser.restore(payload)
        return
    kind = payload[0]
    if kind == LASER_SNAP_FULL and len(payload) >= 2:
        laser.restore(payload[1])
        return
    if kind == LASER_SNAP_SPARSE_FULL and len(payload) >= 2:
        rs, cs, vals, shape = decompress_laser_sparse(payload[1])
        laser.intensity[:, :] = 0
        if laser.intensity.shape == shape and rs.size:
            laser.intensity[rs, cs] = vals
        laser.reset_dirty_log()
        return
    if kind == LASER_SNAP_DELTA and len(payload) >= 2:
        rs, cs, vals, shape = decompress_laser_sparse(payload[1])
        if laser.intensity.shape != shape:
            laser.intensity[:, :] = 0
            laser.reset_dirty_log()
            return
        laser.intensity[rs, cs] = vals
        laser.reset_dirty_log()
        return
    laser.restore(payload)


def laser_snapshot_nbytes(blob: object) -> int:
    """Byte estimate for a tagged laser extra (full / sparse-full / delta / drop)."""
    if blob is None:
        return 0
    if not isinstance(blob, tuple) or not blob:
        return compressed_nbytes(blob)
    kind = blob[0]
    if kind == LASER_SNAP_DROP:
        return 8
    if kind in (LASER_SNAP_FULL, LASER_SNAP_SPARSE_FULL, LASER_SNAP_DELTA) and len(blob) >= 2:
        data = blob[1]
        if kind in (LASER_SNAP_DELTA, LASER_SNAP_SPARSE_FULL) and isinstance(data, tuple) and len(data) >= 2:
            return compressed_nbytes(data[1])
        return compressed_nbytes(data)
    return compressed_nbytes(blob)


def attach_laser_snapshot(payload: tuple, laser: LaserMap | None) -> tuple:
    """Keep occupancy 2-tuples unchanged until a map exists."""
    if laser is None:
        return payload
    return payload + ((LASER_SNAP_FULL, laser.snapshot()),)


def split_laser_snapshot(payload: object) -> tuple[object, object | None]:
    """Return ``(occupancy_payload, laser_payload_or_none)``."""
    if isinstance(payload, tuple) and len(payload) >= 3 and payload[0] in ("full", "delta"):
        return payload[:2], payload[2]
    return payload, None


class LaserMap:
    """uint8 intensity over planar XY or cylindrical (x, θ).

    Allocated lazily by backends on the first successful paint.
    """

    def __init__(
        self,
        *,
        nx: int,
        nv: int,
        cell_u: float,
        origin_u: float,
        mode: str,
        cell_v: float | None = None,
        origin_v: float = 0.0,
        wrap_v: bool = False,
        v_period: float | None = None,
        v_scale: float = 1.0,
    ):
        self.nx = max(1, int(nx))
        self.nv = max(1, int(nv))
        self.cell_u = float(cell_u)
        self.cell_v = float(cell_v if cell_v is not None else cell_u)
        self.origin_u = float(origin_u)
        self.origin_v = float(origin_v)
        self.mode = str(mode)
        self.wrap_v = bool(wrap_v)
        self.v_period = float(v_period) if v_period is not None else (360.0 if wrap_v else 0.0)
        self.v_scale = float(v_scale) if v_scale else 1.0
        self.intensity = np.zeros((self.nx, self.nv), dtype=np.uint8)
        self._dirty = LaserDirtyLog()

    def dirty_native_args(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._dirty.native_args()

    def reset_dirty_log(self) -> None:
        self._dirty.reset()

    def mark_dirty_overflow(self) -> None:
        self._dirty.state[1] = 1

    def log_dirty_indices(self, iu: np.ndarray, iv: np.ndarray) -> None:
        self._dirty.append_indices(iu, iv)

    @classmethod
    def planar(cls, bounds: StockBounds, nx: int, ny: int, cell_size: float) -> LaserMap:
        return cls(
            nx=nx,
            nv=ny,
            cell_u=cell_size,
            origin_u=bounds.min_x,
            mode=LASER_MODE_PLANAR,
            cell_v=cell_size,
            origin_v=bounds.min_y,
            wrap_v=False,
        )

    @classmethod
    def cylindrical(
        cls,
        bounds: StockBounds,
        nx: int,
        n_theta: int,
        cell_size: float,
        d_theta: float,
    ) -> LaserMap:
        """``v`` is stock θ (deg), matching mill occupancy and the carved-stock shader."""
        return cls(
            nx=nx,
            nv=n_theta,
            cell_u=cell_size,
            origin_u=bounds.min_x,
            mode=LASER_MODE_CYLINDRICAL,
            cell_v=d_theta,
            origin_v=0.0,
            wrap_v=True,
            v_period=360.0,
            v_scale=float(cell_size) / max(float(d_theta), 1e-12),
        )

    def intensity_at(self, u: float, v: float) -> int:
        iu, iv = self._index(u, v)
        if iu is None:
            return 0
        return int(self.intensity[iu, iv])

    def paint_segment(
        self,
        u0: float,
        v0: float,
        u1: float,
        v1: float,
        radius: float,
        allow: np.ndarray | None = None,
        *,
        allow_origin: tuple[int, int] | None = None,
        burn: np.uint8 | int,
    ) -> bool:
        """Burn a 2D capsule. ``allow`` is True where stock still exists (window-aligned)."""
        value = np.uint8(int(burn))
        if not value:
            return False
        return self._stamp(u0, v0, u1, v1, radius, value, allow, allow_origin)

    def clear_segment(self, u0: float, v0: float, u1: float, v1: float, radius: float) -> bool:
        """Zero a 2D capsule (mill-over-engrave on a decoupled grid)."""
        return self._stamp(u0, v0, u1, v1, radius, np.uint8(0), None, None)

    def clear_window(self, iu0: int, iv0: int, mask: np.ndarray) -> bool:
        """Zero a contiguous window where ``mask`` is True (same grid as occupancy)."""
        if mask.size == 0 or not mask.any():
            return False
        rows, cols = mask.shape
        iu0 = int(iu0)
        iv0 = int(iv0)
        slab = self.intensity[iu0 : iu0 + rows, iv0 : iv0 + cols]
        if slab.shape != mask.shape:
            return False
        hit = mask & (slab != 0)
        if not hit.any():
            return False
        rs, cs = np.nonzero(hit)
        self.log_dirty_indices(iu0 + rs, iv0 + cs)
        slab[mask] = 0
        return True

    def clear_indexed(self, iu0: int, iv_indices: np.ndarray, mask: np.ndarray) -> bool:
        """Zero ``intensity[iu0+i, iv_indices[j]]`` where ``mask[i, j]`` (θ subsets / wrap)."""
        if mask.size == 0 or not mask.any():
            return False
        rs, cs = np.nonzero(mask)
        if rs.size == 0:
            return False
        iu = int(iu0) + rs
        iv = np.asarray(iv_indices, dtype=np.int32)[cs]
        if self.wrap_v:
            iv %= self.nv
        valid = (iu >= 0) & (iu < self.nx) & (iv >= 0) & (iv < self.nv)
        if not valid.any():
            return False
        iu = iu[valid]
        iv = iv[valid]
        live = self.intensity[iu, iv] != 0
        if not live.any():
            return False
        iu = iu[live]
        iv = iv[live]
        self.log_dirty_indices(iu, iv)
        self.intensity[iu, iv] = 0
        return True

    def clear_from_coarse_mask(
        self,
        mask: np.ndarray,
        coarse_iu0: int,
        coarse_iv0: int | np.ndarray,
        coarse_cell_u: float,
        coarse_cell_v: float,
        coarse_origin_u: float,
        coarse_origin_v: float,
        *,
        coarse_nv: int | None = None,
    ) -> bool:
        """Zero laser texels whose centers fall in True occupancy cells (any resolution)."""
        if mask.size == 0 or not np.any(mask):
            return False
        rows, cols = mask.shape
        cu = max(float(coarse_cell_u), 1e-12)
        cv = max(float(coarse_cell_v), 1e-12)
        iu0 = int(coarse_iu0)

        if isinstance(coarse_iv0, np.ndarray):
            ncv = int(
                coarse_nv if coarse_nv is not None else (int(np.max(coarse_iv0)) + 1 if coarse_iv0.size else cols)
            )
            ncv = max(ncv, 1)
            iv = np.asarray(coarse_iv0, dtype=np.int32)
            if iv.size != cols:
                return False
            if self.wrap_v:
                iv %= ncv
            full = np.zeros((rows, ncv), dtype=bool)
            full[:, iv] = mask
            mask = full
            cols = ncv
            coarse_iv0_i = 0
            coarse_nv = ncv
        else:
            coarse_iv0_i = int(coarse_iv0)

        u0 = float(coarse_origin_u) + iu0 * cu
        u1 = u0 + rows * cu
        lu0 = max(0, int(math.floor((u0 - self.origin_u) / max(self.cell_u, 1e-12))))
        lu1 = min(self.nx - 1, int(math.floor((u1 - 1e-9 - self.origin_u) / max(self.cell_u, 1e-12))))
        if lu0 > lu1:
            return False
        uu = self.origin_u + (np.arange(lu0, lu1 + 1, dtype=np.float64) + 0.5) * self.cell_u
        ci = np.floor((uu - float(coarse_origin_u)) / cu).astype(np.int32) - iu0
        ok_i = (ci >= 0) & (ci < rows)

        if self.wrap_v:
            period = self.v_period if self.v_period else 360.0
            ncv = int(coarse_nv if coarse_nv is not None else cols)
            vv = self.origin_v + (np.arange(self.nv, dtype=np.float64) + 0.5) * self.cell_v
            vv = np.mod(vv - float(coarse_origin_v), period) + float(coarse_origin_v)
            cj = np.floor((vv - float(coarse_origin_v)) / cv).astype(np.int32)
            if ncv > 0:
                cj %= ncv
            cj = cj - coarse_iv0_i
            ok_j = (cj >= 0) & (cj < cols)
            shape = (lu1 - lu0 + 1, self.nv)
            valid = ok_i[:, None] & ok_j[None, :]
            if not valid.any():
                return False
            sampled = np.zeros(shape, dtype=bool)
            ci_b = np.broadcast_to(ci[:, None], shape)
            cj_b = np.broadcast_to(cj[None, :], shape)
            sampled[valid] = mask[ci_b[valid], cj_b[valid]]
            if not sampled.any():
                return False
            slab = self.intensity[lu0 : lu1 + 1, :]
            hit = sampled & (slab != 0)
            if not hit.any():
                return False
            rs, cs = np.nonzero(hit)
            self.log_dirty_indices(lu0 + rs, cs)
            slab[sampled] = 0
            return True

        iv0 = coarse_iv0_i
        v0 = float(coarse_origin_v) + iv0 * cv
        v1 = v0 + cols * cv
        lv0 = max(0, int(math.floor((v0 - self.origin_v) / max(self.cell_v, 1e-12))))
        lv1 = min(self.nv - 1, int(math.floor((v1 - 1e-9 - self.origin_v) / max(self.cell_v, 1e-12))))
        if lv0 > lv1:
            return False
        vv = self.origin_v + (np.arange(lv0, lv1 + 1, dtype=np.float64) + 0.5) * self.cell_v
        cj = np.floor((vv - float(coarse_origin_v)) / cv).astype(np.int32) - iv0
        ok_j = (cj >= 0) & (cj < cols)
        shape = (lu1 - lu0 + 1, lv1 - lv0 + 1)
        valid = ok_i[:, None] & ok_j[None, :]
        if not valid.any():
            return False
        sampled = np.zeros(shape, dtype=bool)
        ci_b = np.broadcast_to(ci[:, None], shape)
        cj_b = np.broadcast_to(cj[None, :], shape)
        sampled[valid] = mask[ci_b[valid], cj_b[valid]]
        if not sampled.any():
            return False
        slab = self.intensity[lu0 : lu1 + 1, lv0 : lv1 + 1]
        hit = sampled & (slab != 0)
        if not hit.any():
            return False
        rs, cs = np.nonzero(hit)
        self.log_dirty_indices(lu0 + rs, lv0 + cs)
        slab[sampled] = 0
        return True

    def gpu_payload(self) -> tuple[int, int, bytes, str]:
        """``(width, height, luminance_bytes, mode)`` with v as texture T (row)."""
        pixels = np.ascontiguousarray(self.intensity.T)
        return (self.nx, self.nv, pixels.tobytes(), self.mode)

    def snapshot(self) -> object:
        return compress_array(self.intensity)

    def restore(self, payload: object) -> None:
        restored = decompress_array(payload)
        if restored.shape != self.intensity.shape:
            self.intensity[:, :] = 0
        else:
            self.intensity[:, :] = restored
        self.reset_dirty_log()

    def _index(self, u: float, v: float) -> tuple[int | None, int | None]:
        iu = int(math.floor((float(u) - self.origin_u) / max(self.cell_u, 1e-12)))
        if self.wrap_v:
            period = self.v_period if self.v_period else 360.0
            vv = float(v) % period
            if vv < 0:
                vv += period
            iv = int(math.floor((vv - self.origin_v) / max(self.cell_v, 1e-12))) % self.nv
        else:
            iv = int(math.floor((float(v) - self.origin_v) / max(self.cell_v, 1e-12)))
        if not (0 <= iu < self.nx and 0 <= iv < self.nv):
            return None, None
        return iu, iv

    def _stamp(
        self,
        u0: float,
        v0: float,
        u1: float,
        v1: float,
        radius: float,
        value: np.uint8,
        allow: np.ndarray | None,
        allow_origin: tuple[int, int] | None,
    ) -> bool:
        origin = (0, 0) if allow_origin is None else allow_origin
        dirty = self.dirty_native_args()
        if not int(value):
            from carveracontroller.addons.stock.simulator.native import clear_laser_capsules

            return clear_laser_capsules(
                self.intensity,
                [float(u0)],
                [float(v0)],
                [float(u1)],
                [float(v1)],
                float(radius),
                cell_u=self.cell_u,
                cell_v=self.cell_v,
                origin_u=self.origin_u,
                origin_v=self.origin_v,
                wrap_v=self.wrap_v,
                v_period=self.v_period,
                v_scale=self.v_scale,
                dirty=dirty,
            )
        from carveracontroller.addons.stock.simulator.native import paint_laser

        return paint_laser(
            self.intensity,
            [float(u0)],
            [float(v0)],
            [float(u1)],
            [float(v1)],
            [int(value)],
            float(radius),
            cell_u=self.cell_u,
            cell_v=self.cell_v,
            origin_u=self.origin_u,
            origin_v=self.origin_v,
            wrap_v=self.wrap_v,
            v_period=self.v_period,
            v_scale=self.v_scale,
            allow=None if allow is None else np.ascontiguousarray(allow, dtype=np.uint8),
            allow_origin=origin,
            dirty=dirty,
        )

    def _wrapped_v_indices(self, v0: float, v1: float, pad_v: float) -> np.ndarray:
        period = self.v_period if self.v_period else 360.0
        v0u, v1u = _unwrap_v_segment(v0, v1, period)
        span = abs(v1u - v0u) + 2.0 * pad_v
        if span >= period - 1e-6:
            return np.arange(self.nv, dtype=np.int32)
        lo = min(v0u, v1u) - pad_v
        n = int(math.ceil(span / max(self.cell_v, 1e-12))) + 2
        i0 = int(math.floor((lo - self.origin_v) / max(self.cell_v, 1e-12)))
        return (i0 + np.arange(n + 1, dtype=np.int32)) % self.nv


def _unwrap_v_segment(v0: float, v1: float, period: float) -> tuple[float, float]:
    """Return ``(v0, v1)`` unwrapped along the short arc, unless travel is a full turn."""
    period = float(period) if period else 360.0
    v0 = float(v0)
    v1 = float(v1)
    dv = v1 - v0
    if abs(dv) >= period - 1e-6:
        return v0, v0 + period
    dv = (dv + period * 0.5) % period - period * 0.5
    return v0, v0 + dv


class LaserDecalMixin:
    """Lazy laser map owned by a carver backend. Mill-clear is a no-op until first paint."""

    _laser: LaserMap | None
    _laser_dirty: bool
    _laser_cp_baseline: np.ndarray | None
    _laser_cp_had_map: bool
    _burned_keys: set[int] | None

    def _init_laser(self) -> None:
        self._laser = None
        self._laser_dirty = False
        self._laser_cp_baseline = None
        self._laser_cp_had_map = False
        # Empty while the live map is known to be zeros. ``None`` means unknown
        # (after a restore or a dirty-log overflow) and the next base frame scans once.
        self._burned_keys = set()

    def _drop_laser(self) -> None:
        # Keep ``_laser_cp_had_map`` so the next delta CP can emit ``("drop",)``.
        # Mark dirty when a map existed so occupancy reset still drops the GPU texture.
        had = self._laser is not None
        self._laser = None
        self._burned_keys = set()
        if had:
            self._laser_dirty = True

    def _ensure_laser(self) -> LaserMap:
        if self._laser is None:
            self._laser = self._make_laser_map()
            self._burned_keys = set()
        return self._laser

    def _make_laser_map(self) -> LaserMap:
        raise NotImplementedError

    def _unique_dirty(self, laser: LaserMap) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Unique dirty indices and their current values. Caller checks ``n > 0``."""
        dirty = laser._dirty
        iu = dirty.iu[: dirty.n].copy()
        iv = dirty.iv[: dirty.n].copy()
        keys = iu.astype(np.int64) * int(laser.nv) + iv.astype(np.int64)
        _, uniq = np.unique(keys, return_index=True)
        iu = iu[uniq]
        iv = iv[uniq]
        return iu, iv, laser.intensity[iu, iv]

    def _absorb_dirty_into_burned(self) -> None:
        """Fold the open dirty log into the running nonzero-texel set."""
        laser = self._laser
        burned = self._burned_keys
        if laser is None or burned is None:
            return
        dirty = laser._dirty
        if dirty.overflow:
            self._burned_keys = None
            return
        if dirty.n == 0:
            return
        iu, iv, curr = self._unique_dirty(laser)
        keys = iu.astype(np.int64) * int(laser.nv) + iv.astype(np.int64)
        for key, val in zip(keys.tolist(), curr.tolist()):
            if val:
                burned.add(int(key))
            else:
                burned.discard(int(key))

    def _rebuild_burned_from_image(self) -> None:
        laser = self._laser
        if laser is None:
            self._burned_keys = set()
            return
        rs, cs = np.nonzero(laser.intensity)
        nv = int(laser.nv)
        self._burned_keys = set((rs.astype(np.int64) * nv + cs.astype(np.int64)).tolist())

    def _payload_from_burned(self) -> object:
        """Sparse-full (or dense full) from the running burned set. No full-image scan."""
        laser = self._laser
        assert laser is not None
        burned = self._burned_keys
        if burned is None:
            snap = _laser_full_payload(laser.intensity)
            self._rebuild_burned_from_image()
            return snap
        shape = laser.intensity.shape
        if not burned:
            empty_i = np.empty(0, dtype=np.int32)
            empty_v = np.empty(0, dtype=np.uint8)
            return (LASER_SNAP_SPARSE_FULL, compress_laser_sparse(empty_i, empty_i, empty_v, shape))
        nv = int(laser.nv)
        arr = np.fromiter(burned, dtype=np.int64, count=len(burned))
        iu, iv = np.divmod(arr, nv)
        iu = iu.astype(np.int32, copy=False)
        iv = iv.astype(np.int32, copy=False)
        curr = laser.intensity[iu, iv]
        keep = curr != 0
        if not np.all(keep):
            iu = iu[keep]
            iv = iv[keep]
            curr = curr[keep]
            self._burned_keys = set((iu.astype(np.int64) * nv + iv.astype(np.int64)).tolist())
        if int(iu.size) * _DELTA_BYTES_PER_TEXEL > laser.intensity.nbytes / 2:
            return (LASER_SNAP_FULL, compress_array(laser.intensity))
        return (LASER_SNAP_SPARSE_FULL, compress_laser_sparse(iu, iv, curr, shape))

    def _commit_laser_baseline(self, *, force_full: bool = False) -> None:
        laser = self._laser
        if laser is None:
            self._laser_cp_baseline = None
            self._laser_cp_had_map = False
            return
        self._laser_cp_had_map = True
        dirty = laser._dirty
        if (
            force_full
            or self._laser_cp_baseline is None
            or self._laser_cp_baseline.shape != laser.intensity.shape
            or dirty.overflow
        ):
            self._laser_cp_baseline = laser.intensity.copy()
        else:
            n = dirty.n
            if n:
                iu = dirty.iu[:n]
                iv = dirty.iv[:n]
                self._laser_cp_baseline[iu, iv] = laser.intensity[iu, iv]
        dirty.reset()

    def _delta_from_dirty_log(self) -> object | None:
        """Build a COO delta from the dirty log, or fall back to a full scan once."""
        laser = self._laser
        assert laser is not None
        dirty = laser._dirty
        baseline = self._laser_cp_baseline
        needs_scan = dirty.overflow or (
            self._burned_keys is None and (baseline is None or baseline.shape != laser.intensity.shape)
        )
        if needs_scan:
            snap = laser_diff_payload(baseline, laser.intensity)
            self._rebuild_burned_from_image()
            self._commit_laser_baseline(force_full=True)
            return snap
        if baseline is None or baseline.shape != laser.intensity.shape:
            # First image is a complete snapshot. A delta would patch whatever
            # decal the restorer already has, including a stale one.
            self._absorb_dirty_into_burned()
            if not self._burned_keys:
                self._commit_laser_baseline(force_full=True)
                return None
            snap = self._payload_from_burned()
            self._commit_laser_baseline(force_full=True)
            return snap
        n = dirty.n
        if n == 0:
            return None
        iu, iv, curr = self._unique_dirty(laser)
        changed = curr != baseline[iu, iv]
        self._absorb_dirty_into_burned()
        if not np.any(changed):
            self._commit_laser_baseline()
            return None
        iu = iu[changed]
        iv = iv[changed]
        curr = curr[changed]
        if int(iu.size) * _DELTA_BYTES_PER_TEXEL > laser.intensity.nbytes / 2:
            snap = (LASER_SNAP_FULL, laser.snapshot())
        else:
            snap = (LASER_SNAP_DELTA, compress_laser_sparse(iu, iv, curr, laser.intensity.shape))
        self._commit_laser_baseline()
        return snap

    def laser_checkpoint_full(self) -> object | None:
        """Tagged full extra for a GOP base slot, or ``None`` if no map."""
        if self._laser is None:
            self._laser_cp_baseline = None
            self._laser_cp_had_map = False
            return None
        if self._laser._dirty.overflow or self._burned_keys is None:
            snap = _laser_full_payload(self._laser.intensity)
            self._rebuild_burned_from_image()
        else:
            self._absorb_dirty_into_burned()
            snap = self._payload_from_burned()
        self._commit_laser_baseline(force_full=True)
        return snap

    def laser_checkpoint_delta(self) -> object | None:
        """Tagged extra vs the last recorded CP, or ``None`` if unchanged / still empty."""
        if self._laser is None:
            if not self._laser_cp_had_map:
                return None
            self._laser_cp_had_map = False
            self._laser_cp_baseline = None
            return (LASER_SNAP_DROP,)
        return self._delta_from_dirty_log()

    def take_laser_dirty(self) -> bool:
        dirty = bool(self._laser_dirty)
        self._laser_dirty = False
        return dirty

    def laser_gpu_payload(self) -> tuple[int, int, bytes, str] | None:
        if self._laser is None:
            return None
        return self._laser.gpu_payload()

    def _restore_laser_payload(self, laser_payload: object | None, *, occupancy_kind: str = "full") -> None:
        """Apply a tagged laser extra. Omitted extra drops on occupancy-full, else leaves as-is."""
        if laser_payload is None:
            if occupancy_kind == "full":
                self._drop_laser()
                self._laser_cp_baseline = None
                self._laser_cp_had_map = False
                return
            self._burned_keys = None
            self._commit_laser_baseline(force_full=True)
            return
        if isinstance(laser_payload, tuple) and laser_payload and laser_payload[0] == LASER_SNAP_DROP:
            self._drop_laser()
            self._laser_cp_baseline = None
            self._laser_cp_had_map = False
            return
        kind = laser_payload[0] if isinstance(laser_payload, tuple) and laser_payload else None
        if kind == LASER_SNAP_DELTA:
            if self._laser is None:
                self._laser = self._make_laser_map()
                self._burned_keys = set()
            apply_laser_tagged(self._laser, laser_payload)
            self._laser_dirty = True
            self._burned_keys = None
            self._commit_laser_baseline(force_full=True)
            return
        laser = self._ensure_laser()
        apply_laser_tagged(laser, laser_payload)
        self._laser_dirty = True
        self._burned_keys = None
        self._commit_laser_baseline(force_full=True)

    def _snapshot_with_laser(self, payload: tuple, *, full: bool) -> tuple:
        extra = self.laser_checkpoint_full() if full else self.laser_checkpoint_delta()
        if extra is None:
            return payload
        return payload + (extra,)
