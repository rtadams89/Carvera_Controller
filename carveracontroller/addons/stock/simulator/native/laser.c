/* Laser decal: paint a beam capsule, and wipe texels a mill has just cut.
 * Strokes stay separate. Merging them into one chord would swallow raster lead-ins.
 *
 * The decal is a 2D image of "burn marks" laid over the stock. Its two axes are `u` (X) and `v`
 * (Y on flat stock, or an angle in degrees on cylindrical stock, where it wraps around).
 * A beam move is drawn as a capsule: every texel whose center is within the beam radius of the
 * move's line is painted.
 */
#include "stock_carve_internal.h"

#include <math.h>

/* Extra conditions a texel must pass before it is painted. */
typedef struct ScLaserFilter {
    const uint8_t *allow; /* optional window: only texels whose entry is non-zero can be painted */
    int allow_u0, allow_v0; /* decal index of allow[0][0] */
    int allow_rows, allow_cols;
    const ScLaserOcc *occ; /* optional: only paint where stock exists */
    double z_or_r;         /* beam Z or tool radius, used by `occ` */
} ScLaserFilter;

/* Remainder in [0, m) for integers (C's % can be negative). */
static int sc_mod_int(int a, int m) {
    int r;
    if (m <= 0) {
        return 0;
    }
    r = a % m;
    if (r < 0) {
        r += m;
    }
    return r;
}

/* Length of one turn in v; angles default to degrees. */
static double sc_period(double period) {
    return period > 0.0 ? period : 360.0;
}

/* Pick the way around the circle from v0 to v1 that is shorter, so a beam crossing the 0/360 seam doesn't
 * sweep the whole decal. Outputs v0 and an end point within half a turn of it (or a full turn if the
 * move already covers one).
 */
static void sc_unwrap_v(double v0, double v1, double period, double *o0, double *o1) {
    double dv;
    period = sc_period(period);
    dv = v1 - v0;
    if (fabs(dv) >= period - 1e-6) {
        *o0 = v0;
        *o1 = v0 + period;
        return;
    }
    dv = sc_py_mod(dv + period * 0.5, period) - period * 0.5;
    *o0 = v0;
    *o1 = v0 + dv;
}

/* True if the point (uu, vv) is within `radius` of the line from (u0, v0) to (u1, v1).
 * v is scaled to millimeters first, so the beam stays round when v is an angle.
 */
static int sc_capsule_hit(
    double uu, double vv, double u0, double v0, double u1, double v1, double radius, double v_scale) {
    double scale = v_scale != 0.0 ? v_scale : 1.0;
    double du = u1 - u0;
    double dv = (v1 - v0) * scale;
    double len_sq = du * du + dv * dv;
    double r_cov = radius + 1e-9;
    double vv_mm = vv * scale;
    double v0_mm = v0 * scale;
    double t;
    double px;
    double py;
    if (len_sq < 1e-18) {
        /* Zero-length move: just a dot. */
        px = uu - u0;
        py = vv_mm - v0_mm;
        return px * px + py * py <= r_cov * r_cov;
    }
    t = sc_clamp(((uu - u0) * du + (vv_mm - v0_mm) * dv) / len_sq, 0.0, 1.0);
    px = uu - (u0 + t * du);
    py = vv_mm - (v0_mm + t * dv);
    return px * px + py * py <= r_cov * r_cov;
}

/* Same, but the beam's v may be shifted by a whole turn, so also try the point one turn either way. */
static int sc_capsule_hit_wrapped(
    double uu, double vv, double u0, double v0, double u1, double v1, double radius, double period, double v_scale) {
    period = sc_period(period);
    if (sc_capsule_hit(uu, vv, u0, v0, u1, v1, radius, v_scale)) {
        return 1;
    }
    if (sc_capsule_hit(uu, vv + period, u0, v0, u1, v1, radius, v_scale)) {
        return 1;
    }
    return sc_capsule_hit(uu, vv - period, u0, v0, u1, v1, radius, v_scale);
}

/* Is there stock under the point (uu, vv) that the beam can reach? */
static int sc_occ_allows(const ScLaserOcc *occ, double z_or_r, double uu, double vv) {
    double eps;
    float thr;
    int ix;
    int iv;
    float sample;
    if (occ == NULL || occ->kind == 0 || occ->data == NULL || occ->nx <= 0 || occ->nv <= 0) {
        return 1;
    }
    /* Stock counts if it reaches up to (at most) `eps` below the beam height / tool radius. */
    eps = occ->cell > 0.2 ? occ->cell : 0.2;
    thr = (float)(z_or_r - eps);
    ix = sc_clamp_int((int)floor((uu - occ->min_x) / sc_nonzero(occ->cell)), 0, occ->nx - 1);
    if (occ->kind == 1) {
        /* Heightmap: stock exists if the cell isn't "outside" and its top is high enough. */
        iv = sc_clamp_int((int)floor((vv - occ->min_y) / sc_nonzero(occ->cell)), 0, occ->nv - 1);
        sample = occ->data[(size_t)ix * (size_t)occ->nv + (size_t)iv];
        return (double)sample > SC_OUTSIDE * 0.5 && sample >= thr;
    }
    /* Cylindrical: v is an angle, wrapped into the angle bins. Stock exists if the radius is large enough. */
    iv = (int)floor(sc_py_mod(vv, sc_period(occ->period)) / sc_nonzero(occ->d_theta));
    iv = sc_mod_int(iv, occ->nv);
    sample = occ->data[(size_t)ix * (size_t)occ->nv + (size_t)iv];
    return sample >= 0.0f && sample >= thr;
}

static int sc_texel_allowed(const ScLaserFilter *filter, int iu, int iv, double uu, double vv) {
    if (filter == NULL) {
        return 1;
    }
    if (filter->allow != NULL) {
        int ar = iu - filter->allow_u0;
        int ac = iv - filter->allow_v0;
        if (ar < 0 || ac < 0 || ar >= filter->allow_rows || ac >= filter->allow_cols) {
            return 0;
        }
        if (!filter->allow[(size_t)ar * (size_t)filter->allow_cols + (size_t)ac]) {
            return 0;
        }
    }
    return sc_occ_allows(filter->occ, filter->z_or_r, uu, vv);
}

/* Remember that a texel changed. Once the log is full we only note the overflow. */
static void sc_dirty_append(ScLaserDirty *dirty, int iu, int iv) {
    if (dirty == NULL || dirty->overflow) {
        return;
    }
    if (dirty->n >= dirty->cap || dirty->iu == NULL || dirty->iv == NULL) {
        dirty->overflow = 1;
        return;
    }
    dirty->iu[dirty->n] = (int32_t)iu;
    dirty->iv[dirty->n] = (int32_t)iv;
    dirty->n += 1;
}

/* Set a texel to `value`, or clear it when `value` is 0. Painting never lowers a texel.
 * Returns 1 if the texel changed.
 */
static int sc_write_texel(ScLaserDecal *map, int iu, int iv, uint8_t value) {
    uint8_t *p = &map->intensity[(size_t)iu * (size_t)map->nv + (size_t)iv];
    if (value) {
        if (*p >= value) {
            return 0;
        }
        *p = value;
        sc_dirty_append(map->dirty, iu, iv);
        return 1;
    }
    if (!*p) {
        return 0;
    }
    *p = 0;
    sc_dirty_append(map->dirty, iu, iv);
    return 1;
}

/* Write one texel if the filters allow it. Returns 1 if it changed. */
static int sc_stamp_texel(
    ScLaserDecal *map, const ScLaserFilter *filter, int iu, int iv, double uu, double vv, uint8_t value) {
    if (!sc_texel_allowed(filter, iu, iv, uu, vv)) {
        return 0;
    }
    return sc_write_texel(map, iu, iv, value);
}

/* Stamp a capsule onto a decal whose v axis wraps around (angles). */
static int sc_stamp_wrapped(
    ScLaserDecal *map,
    double u0,
    double v0,
    double u1,
    double v1,
    uint8_t value,
    double radius,
    const ScLaserFilter *filter,
    double pad_v,
    int iu0,
    int iu1) {
    double cell_v = sc_nonzero(map->cell_v);
    double v_scale = map->v_scale != 0.0 ? map->v_scale : 1.0;
    double period = sc_period(map->v_period);
    double v0u, v1u, span;
    int full, lo, hi, iu, i;
    int any = 0;
    sc_unwrap_v(v0, v1, period, &v0u, &v1u);
    span = fabs(v1u - v0u) + 2.0 * pad_v;
    full = span >= period - 1e-6;
    if (full) {
        /* The beam covers the whole circle: visit every angle once. */
        lo = 0;
        hi = map->nv - 1;
    } else {
        /* Visit the angles around the beam; indices outside [0, nv) wrap around. */
        double vlo = fmin(v0u, v1u) - pad_v;
        int n = (int)ceil(span / cell_v) + 2;
        lo = (int)floor((vlo - map->origin_v) / cell_v);
        hi = lo + n; /* inclusive: n + 1 samples */
    }
    for (iu = iu0; iu <= iu1; iu++) {
        double uu = map->origin_u + ((double)iu + 0.5) * map->cell_u;
        for (i = lo; i <= hi; i++) {
            int iv = full ? i : sc_mod_int(i, map->nv);
            double vv = map->origin_v + ((double)iv + 0.5) * map->cell_v;
            if (!sc_capsule_hit_wrapped(uu, vv, u0, v0u, u1, v1u, radius, period, v_scale)) {
                continue;
            }
            if (sc_stamp_texel(map, filter, iu, iv, uu, vv, value)) {
                any = 1;
            }
        }
    }
    return any;
}

/* Stamp a capsule onto a plain (non-wrapping) decal. */
static int sc_stamp_planar(
    ScLaserDecal *map,
    double u0,
    double v0,
    double u1,
    double v1,
    uint8_t value,
    double radius,
    const ScLaserFilter *filter,
    double pad_v,
    int iu0,
    int iu1) {
    double cell_v = sc_nonzero(map->cell_v);
    double v_scale = map->v_scale != 0.0 ? map->v_scale : 1.0;
    int iv0 = (int)floor((fmin(v0, v1) - pad_v - map->origin_v) / cell_v);
    int iv1 = (int)floor((fmax(v0, v1) + pad_v - map->origin_v) / cell_v);
    int iu, iv;
    int any = 0;
    if (!sc_clip_range(&iv0, &iv1, map->nv)) {
        return 0;
    }
    for (iu = iu0; iu <= iu1; iu++) {
        double uu = map->origin_u + ((double)iu + 0.5) * map->cell_u;
        for (iv = iv0; iv <= iv1; iv++) {
            double vv = map->origin_v + ((double)iv + 0.5) * map->cell_v;
            if (!sc_capsule_hit(uu, vv, u0, v0, u1, v1, radius, v_scale)) {
                continue;
            }
            if (sc_stamp_texel(map, filter, iu, iv, uu, vv, value)) {
                any = 1;
            }
        }
    }
    return any;
}

/* Paint (or, with value 0, clear) one capsule. Returns 1 if any texel changed. */
static int sc_laser_stamp_one(
    ScLaserDecal *map,
    double u0,
    double v0,
    double u1,
    double v1,
    uint8_t value,
    double radius,
    const ScLaserFilter *filter,
    int *changed) {
    double cell_u;
    double v_scale;
    double pad_u;
    double pad_v;
    int iu0;
    int iu1;
    int any;
    if (map == NULL || map->intensity == NULL || map->nx <= 0 || map->nv <= 0) {
        return 0;
    }
    cell_u = sc_nonzero(map->cell_u);
    v_scale = map->v_scale != 0.0 ? map->v_scale : 1.0;
    /* A beam thinner than a texel would skip texels, so never go below ~one texel wide. */
    if (radius < 0.71 * map->cell_u) {
        radius = 0.71 * map->cell_u;
    }
    if (radius < 0.0) {
        radius = 0.0;
    }
    /* Search a bit beyond the capsule so no touched texel is missed. */
    pad_u = radius + map->cell_u;
    pad_v = radius / sc_nonzero(v_scale) + map->cell_v;
    iu0 = (int)floor((fmin(u0, u1) - pad_u - map->origin_u) / cell_u);
    iu1 = (int)floor((fmax(u0, u1) + pad_u - map->origin_u) / cell_u);
    if (!sc_clip_range(&iu0, &iu1, map->nx)) {
        return 0;
    }
    if (map->wrap_v) {
        any = sc_stamp_wrapped(map, u0, v0, u1, v1, value, radius, filter, pad_v, iu0, iu1);
    } else {
        any = sc_stamp_planar(map, u0, v0, u1, v1, value, radius, filter, pad_v, iu0, iu1);
    }
    if (any && changed != NULL) {
        *changed = 1;
    }
    return any;
}

int sc_laser_paint(
    ScLaserDecal *map,
    const double *u0,
    const double *v0,
    const double *u1,
    const double *v1,
    const uint8_t *burn,
    const double *z_or_r,
    int nseg,
    double radius,
    const uint8_t *allow,
    int allow_u0,
    int allow_v0,
    int allow_rows,
    int allow_cols,
    const ScLaserOcc *occ,
    int *changed) {
    int i;
    if (map == NULL || u0 == NULL || v0 == NULL || u1 == NULL || v1 == NULL || burn == NULL || nseg <= 0) {
        return 0;
    }
    for (i = 0; i < nseg; i++) {
        ScLaserFilter filter;
        if (!burn[i]) {
            continue;
        }
        /* An allow window belongs to one stamp. Ignore it on later segments. */
        filter.allow = i == 0 ? allow : NULL;
        filter.allow_u0 = allow_u0;
        filter.allow_v0 = allow_v0;
        filter.allow_rows = allow_rows;
        filter.allow_cols = allow_cols;
        filter.occ = occ;
        filter.z_or_r = z_or_r != NULL ? z_or_r[i] : 0.0;
        sc_laser_stamp_one(map, u0[i], v0[i], u1[i], v1[i], burn[i], radius, &filter, changed);
    }
    return 0;
}

int sc_laser_clear_capsules(
    ScLaserDecal *map,
    const double *u0,
    const double *v0,
    const double *u1,
    const double *v1,
    int nseg,
    double radius,
    int *changed) {
    int i;
    if (map == NULL || u0 == NULL || v0 == NULL || u1 == NULL || v1 == NULL || nseg <= 0) {
        return 0;
    }
    for (i = 0; i < nseg; i++) {
        sc_laser_stamp_one(map, u0[i], v0[i], u1[i], v1[i], 0, radius, NULL, changed);
    }
    return 0;
}

/* ------------------------------------------------------------------------- */
/* Wiping the decal under a freshly cut stock cell                           */
/* ------------------------------------------------------------------------- */

/* Decal indices [*lo, *hi] of the texels overlapping the interval [edge0, edge1). Returns 0 if there are none. */
static int sc_texel_range(double edge0, double edge1, double origin, double cell, int n, int *lo, int *hi) {
    double c = sc_nonzero(cell);
    *lo = (int)floor((edge0 - origin) / c);
    *hi = (int)floor((edge1 - 1e-9 - origin) / c);
    return sc_clip_range(lo, hi, n);
}

/* Does `center` fall in bin number `index` of bins `size` wide starting at `origin`? */
static int sc_in_bin(double center, double origin, double size, int index) {
    return (int)floor((center - origin) / sc_nonzero(size)) == index;
}

/* Which angle bin `center` falls in, with angles wrapping every `period`. */
static int sc_theta_bin(double center, double origin, double size, double period, int nbin) {
    double step = sc_nonzero(size);
    double vv = sc_py_mod(center - origin, sc_period(period)) + origin;
    int bin = (int)floor((vv - origin) / step);
    return sc_mod_int(bin, nbin);
}

static void sc_clear_at(ScLaserDecal *laser, int iu, int iv, int *changed) {
    if (iu < 0 || iv < 0 || iu >= laser->nx || iv >= laser->nv) {
        return;
    }
    if (sc_write_texel(laser, iu, iv, 0) && changed != NULL) {
        *changed = 1;
    }
}

void sc_laser_clear_coarse_cell(
    ScLaserDecal *laser,
    int ix,
    int coarse_iv,
    double coarse_origin_u,
    double coarse_origin_v,
    double coarse_du,
    double coarse_dv,
    int coarse_nv,
    int *changed) {
    double u0;
    double u1;
    int lu0;
    int lu1;
    int iu;
    if (laser == NULL || laser->intensity == NULL || laser->nx <= 0 || laser->nv <= 0) {
        return;
    }
    if (coarse_du <= 0.0 || coarse_dv <= 0.0) {
        return;
    }
    /* The decal columns overlapping the stock cell. */
    u0 = coarse_origin_u + (double)ix * coarse_du;
    u1 = u0 + coarse_du;
    if (!sc_texel_range(u0, u1, laser->origin_u, laser->cell_u, laser->nx, &lu0, &lu1)) {
        return;
    }
    if (!laser->wrap_v) {
        double v0 = coarse_origin_v + (double)coarse_iv * coarse_dv;
        double v1 = v0 + coarse_dv;
        int lv0;
        int lv1;
        int iv;
        if (!sc_texel_range(v0, v1, laser->origin_v, laser->cell_v, laser->nv, &lv0, &lv1)) {
            return;
        }
        for (iu = lu0; iu <= lu1; iu++) {
            double cu = laser->origin_u + ((double)iu + 0.5) * laser->cell_u;
            if (!sc_in_bin(cu, coarse_origin_u, coarse_du, ix)) {
                continue;
            }
            for (iv = lv0; iv <= lv1; iv++) {
                double cv = laser->origin_v + ((double)iv + 0.5) * laser->cell_v;
                if (!sc_in_bin(cv, coarse_origin_v, coarse_dv, coarse_iv)) {
                    continue;
                }
                sc_clear_at(laser, iu, iv, changed);
            }
        }
        return;
    }
    {
        /* Angles wrap, so work with indices that may run past either end and fold them back. */
        double period = sc_period(laser->v_period);
        double cell_v = sc_nonzero(laser->cell_v);
        double th0 = coarse_origin_v + (double)coarse_iv * coarse_dv;
        double th1 = th0 + coarse_dv;
        int bins = coarse_nv > 0 ? coarse_nv : 1;
        int lo = (int)floor((th0 - laser->origin_v) / cell_v - 0.5) - 2;
        int hi = (int)floor((th1 - laser->origin_v) / cell_v - 0.5) + 2;
        int direct;
        int i;
        if (hi < lo) {
            return;
        }
        /* Wider than the decal itself: no point folding, just scan every angle. */
        direct = (hi - lo) > laser->nv + 4;
        if (direct) {
            lo = 0;
            hi = laser->nv - 1;
        }
        for (iu = lu0; iu <= lu1; iu++) {
            double cu = laser->origin_u + ((double)iu + 0.5) * laser->cell_u;
            if (!sc_in_bin(cu, coarse_origin_u, coarse_du, ix)) {
                continue;
            }
            for (i = lo; i <= hi; i++) {
                int iv = direct ? i : sc_mod_int(i, laser->nv);
                double cv = laser->origin_v + ((double)iv + 0.5) * laser->cell_v;
                if (sc_theta_bin(cv, coarse_origin_v, coarse_dv, period, bins) != coarse_iv) {
                    continue;
                }
                sc_clear_at(laser, iu, iv, changed);
            }
        }
    }
}
