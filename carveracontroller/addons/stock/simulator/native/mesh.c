/* Welded stock meshes in the Kivy vertex layout.
 *
 * Heightmap and cylindrical shells share corner vertices. Voxel chunks keep
 * unshared quads; a 16^3 chunk is small enough that the win is leaving Python.
 * GLES 2 Mesh uploads reject an index list longer than 65535, not only a
 * vertex index that does not fit in uint16, so each part stays under both.
 */
#include "stock_carve.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define SC_MESH_MAX_VERTS 65000
#define SC_MESH_MAX_INDICES 65535 /* Kivy raises above this on OpenGL ES 2 */
#define SC_VERT_FLOATS 12 /* position (3), normal (3), color (4), uv (2) */

/* Heightmap shading and welding tuning. */
#define HM_CLIFF 4.0f              /* neighbors more than this many cells higher/lower don't shape the normal */
#define HM_MERGE 1.0e-4f           /* heights closer than this count as equal */
#define HM_EXAG 3.0f               /* exaggerate slopes in the shading normal so relief is easier to see */
#define HM_MAX_SLOPE 1.73205080757f /* cap on the exaggerated slope (60 degrees) so the normal never goes flat */

typedef struct ScV3 {
    float x, y, z;
} ScV3;

typedef struct ScColor {
    float r, g, b, a;
} ScColor;

static ScColor sc_color(double r, double g, double b, double a) {
    ScColor c;
    c.r = (float)r;
    c.g = (float)g;
    c.b = (float)b;
    c.a = (float)a;
    return c;
}

static ScV3 sc_v3(float x, float y, float z) {
    ScV3 v;
    v.x = x;
    v.y = y;
    v.z = z;
    return v;
}

/* ------------------------------------------------------------------------- */
/* Growing vertex / index buffers                                            */
/* ------------------------------------------------------------------------- */

/* The mesh part being built. A part is closed before it would exceed either GLES Mesh cap. */
typedef struct ScBuf {
    float *v;
    uint16_t *idx;
    int nv;
    int ni;
    int vcap;
    int icap;
} ScBuf;

void sc_mesh_batch_free(ScMeshBatch *batch) {
    int i;
    if (batch == NULL) {
        return;
    }
    for (i = 0; i < batch->nparts; i++) {
        free(batch->parts[i].verts);
        free(batch->parts[i].indices);
    }
    free(batch->parts);
    batch->parts = NULL;
    batch->nparts = 0;
    batch->cap = 0;
}

static void sc_buf_init(ScBuf *b) {
    memset(b, 0, sizeof(*b));
}

static void sc_buf_free(ScBuf *b) {
    free(b->v);
    free(b->idx);
    sc_buf_init(b);
}

/* Move a finished buffer into the batch (the batch now owns its memory). Empty buffers are dropped.
 * Returns 0, or -1 if memory ran out (the buffer is left untouched).
 */
static int sc_batch_add(ScMeshBatch *batch, ScBuf *b) {
    ScMeshPart *parts;
    int ncap;
    if (b->nv <= 0 || b->ni <= 0) {
        sc_buf_free(b);
        return 0;
    }
    if (batch->nparts >= batch->cap) {
        ncap = batch->cap > 0 ? batch->cap * 2 : 4;
        parts = (ScMeshPart *)realloc(batch->parts, (size_t)ncap * sizeof(ScMeshPart));
        if (parts == NULL) {
            return -1;
        }
        batch->parts = parts;
        batch->cap = ncap;
    }
    batch->parts[batch->nparts].verts = b->v;
    batch->parts[batch->nparts].indices = b->idx;
    batch->parts[batch->nparts].nverts = b->nv;
    batch->parts[batch->nparts].nindices = b->ni;
    batch->nparts++;
    sc_buf_init(b);
    return 0;
}

/* Make room for `add_v` more vertices and `add_i` more indices. Returns 0, or -1 on failure. */
static int sc_buf_reserve(ScBuf *b, int add_v, int add_i) {
    int ncap;
    float *nv;
    uint16_t *ni;
    if (b->nv + add_v > b->vcap) {
        ncap = b->vcap > 0 ? b->vcap : 256;
        while (ncap < b->nv + add_v) {
            if (ncap > 8000000) {
                return -1;
            }
            ncap *= 2;
        }
        nv = (float *)realloc(b->v, (size_t)ncap * SC_VERT_FLOATS * sizeof(float));
        if (nv == NULL) {
            return -1;
        }
        b->v = nv;
        b->vcap = ncap;
    }
    if (b->ni + add_i > b->icap) {
        ncap = b->icap > 0 ? b->icap : 512;
        while (ncap < b->ni + add_i) {
            if (ncap > 48000000) {
                return -1;
            }
            ncap *= 2;
        }
        ni = (uint16_t *)realloc(b->idx, (size_t)ncap * sizeof(uint16_t));
        if (ni == NULL) {
            return -1;
        }
        b->idx = ni;
        b->icap = ncap;
    }
    return 0;
}

/* Append a vertex. Returns its index, -2 if the part is full (the caller can split and retry),
 * or -1 if memory ran out.
 */
static int sc_buf_vert(ScBuf *b, ScV3 pos, ScV3 normal, const ScColor *color) {
    float *p;
    if (b->nv >= SC_MESH_MAX_VERTS) {
        return -2;
    }
    if (sc_buf_reserve(b, 1, 0) != 0) {
        return -1;
    }
    p = b->v + b->nv * SC_VERT_FLOATS;
    p[0] = pos.x;
    p[1] = pos.y;
    p[2] = pos.z;
    p[3] = normal.x;
    p[4] = normal.y;
    p[5] = normal.z;
    p[6] = color->r;
    p[7] = color->g;
    p[8] = color->b;
    p[9] = color->a;
    p[10] = 0.f; /* no texture coordinates */
    p[11] = 0.f;
    return b->nv++;
}

static int sc_buf_tri(ScBuf *b, int a, int c, int d) {
    uint16_t *p;
    if (a < 0 || c < 0 || d < 0) {
        return -1;
    }
    if (b->ni + 3 > SC_MESH_MAX_INDICES) {
        return -2;
    }
    if (sc_buf_reserve(b, 0, 3) != 0) {
        return -1;
    }
    p = b->idx + b->ni;
    p[0] = (uint16_t)a;
    p[1] = (uint16_t)c;
    p[2] = (uint16_t)d;
    b->ni += 3;
    return 0;
}

/* Two triangles covering the quad a-c-d-e. */
static int sc_buf_quad(ScBuf *b, int a, int c, int d, int e) {
    int rc;
    if (b->ni + 6 > SC_MESH_MAX_INDICES) {
        return -2;
    }
    rc = sc_buf_tri(b, a, c, d);
    if (rc != 0) {
        return rc;
    }
    return sc_buf_tri(b, a, d, e);
}

/* Add a flat quad with four new vertices that all share one normal. Returns 0, -2 (part full), or -1. */
static int sc_buf_flat_quad(ScBuf *b, const ScV3 corners[4], ScV3 normal, const ScColor *color) {
    int ids[4];
    int k;
    if (b->nv + 4 > SC_MESH_MAX_VERTS || b->ni + 6 > SC_MESH_MAX_INDICES) {
        return -2;
    }
    for (k = 0; k < 4; k++) {
        ids[k] = sc_buf_vert(b, corners[k], normal, color);
    }
    for (k = 0; k < 4; k++) {
        if (ids[k] < 0) {
            return ids[k];
        }
    }
    return sc_buf_quad(b, ids[0], ids[1], ids[2], ids[3]);
}

/* ------------------------------------------------------------------------- */
/* Greedy rectangle merging                                                  */
/* ------------------------------------------------------------------------- */

/* Called for each merged rectangle covering rows [i0, i1) and columns [j0, j1). Returns 0 on success. */
typedef int (*ScRectFn)(void *ctx, int i0, int j0, int i1, int j1);

/* Cover every free cell of a rows x stride grid with as few rectangles as a simple greedy scan finds.
 * `used` marks cells that are already taken (it is updated as rectangles are emitted). Only columns
 * [j_begin, j_end) are scanned. Stops at the first non-zero return of `emit` and passes it on.
 */
static int sc_greedy_rects(
    uint8_t *used, int rows, int stride, int j_begin, int j_end, ScRectFn emit, void *ctx) {
    int i, j;
    for (i = 0; i < rows; i++) {
        j = j_begin;
        while (j < j_end) {
            int w, d, k, rc;
            if (used[i * stride + j]) {
                j++;
                continue;
            }
            /* Grow right along the row as far as cells are free... */
            w = 1;
            while (j + w < j_end && !used[i * stride + (j + w)]) {
                w++;
            }
            /* ...then down, as long as the whole width stays free. */
            d = 1;
            while (i + d < rows) {
                int ok = 1;
                for (k = 0; k < w; k++) {
                    if (used[(i + d) * stride + (j + k)]) {
                        ok = 0;
                        break;
                    }
                }
                if (!ok) {
                    break;
                }
                d++;
            }
            for (k = 0; k < d; k++) {
                int t;
                for (t = 0; t < w; t++) {
                    used[(i + k) * stride + (j + t)] = 1;
                }
            }
            rc = emit(ctx, i, j, i + d, j + w);
            if (rc != 0) {
                return rc;
            }
            j += w;
        }
    }
    return 0;
}

/* ------------------------------------------------------------------------- */
/* Heightmap mesh                                                            */
/* ------------------------------------------------------------------------- */

/* The mesh is a "terrain": one flat top per cell, welded to neighbors at the same height, plus vertical walls
 * ("skirts") wherever a cell is higher than its neighbor, and a flat underside. Side indices used below:
 * 0 = -X, 1 = +X, 2 = -Y, 3 = +Y.
 */
typedef struct HmMesh {
    const float *patch; /* cell heights with a one-cell halo, (gw + 2) x (gh + 2) */
    int stride;         /* gh + 2 */
    int gw, gh;         /* window size in cells */
    int x0, y0;         /* window position in the full grid */
    int grid_nx, grid_ny;
    float origin_x, origin_y;
    float min_z;        /* bottom of the stock; also the "no stock" threshold */
    float cell;
    float cliff;        /* height difference above which a neighbor is ignored when shading */
    ScColor color;
    /* Filled in by sc_mesh_heightmap: */
    uint8_t *valid;     /* [gw * gh] cell has stock */
    int cstride;        /* gh + 1 */
    uint8_t *share;     /* [(gw + 1) * (gh + 1)] corner is welded: all cells around it have the same height */
    float *corner_z;    /* [(gw + 1) * (gh + 1)] height of welded corners */
} HmMesh;

/* A horizontal band of rows being meshed, and the vertex ids already created for its welded corners. */
typedef struct HmStrip {
    int *ids;
    int j0;
    int nrows; /* corner rows in the strip (cells + 1) */
} HmStrip;

static int hm_valid(float h, float min_z) {
    return h > (float)(SC_OUTSIDE * 0.5) && h > min_z + HM_MERGE;
}

/* Height of cell (cx, cy) in window coordinates. The halo makes -1 and gw / gh valid too. */
static float hm_at(const HmMesh *m, int cx, int cy) {
    return m->patch[(cx + 1) * m->stride + (cy + 1)];
}

/* Unit normal for a surface with the given slope, with the slope exaggerated and capped. */
static ScV3 hm_slope_normal(float dzdx, float dzdy) {
    float gx = HM_EXAG * dzdx;
    float gy = HM_EXAG * dzdy;
    float slope = sqrtf(gx * gx + gy * gy);
    float scale = 1.f;
    float x, y, z, len;
    if (slope > HM_MAX_SLOPE) {
        scale = HM_MAX_SLOPE / slope;
    }
    x = -gx * scale;
    y = -gy * scale;
    z = 1.f;
    len = sqrtf(x * x + y * y + z * z);
    if (len < 1e-12f) {
        return sc_v3(0.f, 0.f, 1.f);
    }
    return sc_v3(x / len, y / len, z / len);
}

/* Shading normal of cell (i, j), from its height differences with neighbors that aren't across a cliff. */
static ScV3 hm_cell_normal(const HmMesh *m, int i, int j, float h) {
    float hl = hm_at(m, i - 1, j);
    float hr = hm_at(m, i + 1, j);
    float hd = hm_at(m, i, j - 1);
    float hu = hm_at(m, i, j + 1);
    int ul = hm_valid(hl, m->min_z) && fabsf(hl - h) <= m->cliff;
    int ur = hm_valid(hr, m->min_z) && fabsf(hr - h) <= m->cliff;
    int ud = hm_valid(hd, m->min_z) && fabsf(hd - h) <= m->cliff;
    int uu = hm_valid(hu, m->min_z) && fabsf(hu - h) <= m->cliff;
    float dzdx = 0.f;
    float dzdy = 0.f;
    if (ul && ur) {
        dzdx = (hr - hl) / (2.f * m->cell);
    } else if (ur) {
        dzdx = (hr - h) / m->cell;
    } else if (ul) {
        dzdx = (h - hl) / m->cell;
    }
    if (ud && uu) {
        dzdy = (hu - hd) / (2.f * m->cell);
    } else if (uu) {
        dzdy = (hu - h) / m->cell;
    } else if (ud) {
        dzdy = (h - hd) / m->cell;
    }
    return hm_slope_normal(dzdx, dzdy);
}

/* Vertex at corner (i + di, j + dj) of cell (i, j)'s top face. Returns its id, or a negative sc_buf_vert error. */
static int hm_corner_vert(ScBuf *b, const HmMesh *m, const HmStrip *strip, int i, int j, int di, int dj) {
    int ci = i + di;
    int cj = j + dj;
    int g = ci * m->cstride + cj;
    int slot = ci * strip->nrows + (cj - strip->j0);
    float z;
    ScV3 normal;
    int id;
    if (m->share[g]) {
        /* Welded flat corner: every incident cell has the same height, so the averaged height is exact
         * and the shading normal stays +Z. The vertex is shared with the neighboring cells. */
        if (strip->ids[slot] >= 0) {
            return strip->ids[slot];
        }
        z = m->corner_z[g];
        normal = sc_v3(0.f, 0.f, 1.f);
    } else {
        /* Steps keep crisp per-cell tops at the cell's own height. Averaging neighbor heights here let
         * adjacent cells disagree on a shared edge while staying under the skirt threshold (open slits
         * on walls). */
        float h = hm_at(m, i, j);
        z = h;
        normal = hm_cell_normal(m, i, j, h);
    }
    id = sc_buf_vert(
        b, sc_v3(m->origin_x + (float)ci * m->cell, m->origin_y + (float)cj * m->cell, z), normal, &m->color);
    if (id >= 0 && m->share[g]) {
        strip->ids[slot] = id;
    }
    return id;
}

/* Where a wall below cell (i, j) is needed on one side.
 * A wall is needed at the edge of the whole grid, next to a cell with no stock, or next to a lower cell.
 * It goes down to the neighbor's top, or all the way to the bottom if there is no neighbor stock.
 */
static void hm_wall_info(const HmMesh *m, int side, int i, int j, float h, int *needed, float *z_bottom) {
    static const int di[4] = {-1, 1, 0, 0};
    static const int dj[4] = {0, 0, -1, 1};
    float nh = hm_at(m, i + di[side], j + dj[side]);
    int at_grid_edge;
    switch (side) {
    case 0:
        at_grid_edge = m->x0 + i == 0;
        break;
    case 1:
        at_grid_edge = m->x0 + i + 1 >= m->grid_nx;
        break;
    case 2:
        at_grid_edge = m->y0 + j == 0;
        break;
    default:
        at_grid_edge = m->y0 + j + 1 >= m->grid_ny;
        break;
    }
    *needed = at_grid_edge || !hm_valid(nh, m->min_z) || (h - nh) > HM_MERGE;
    *z_bottom = (at_grid_edge || !hm_valid(nh, m->min_z)) ? m->min_z : nh;
}

/* Vertical wall on one side of the window or a cell, from z_top down to z_bottom.
 * `fixed` is the wall's X (sides 0, 1) or Y (sides 2, 3); `lo`..`hi` is its extent along the other axis.
 */
static int hm_emit_wall(
    ScBuf *b, const HmMesh *m, int side, float fixed, float lo, float hi, float z_top, float z_bottom) {
    ScV3 c[4];
    ScV3 normal;
    switch (side) {
    case 0:
        c[0] = sc_v3(fixed, lo, z_top);
        c[1] = sc_v3(fixed, hi, z_top);
        c[2] = sc_v3(fixed, hi, z_bottom);
        c[3] = sc_v3(fixed, lo, z_bottom);
        normal = sc_v3(-1.f, 0.f, 0.f);
        break;
    case 1:
        c[0] = sc_v3(fixed, hi, z_top);
        c[1] = sc_v3(fixed, lo, z_top);
        c[2] = sc_v3(fixed, lo, z_bottom);
        c[3] = sc_v3(fixed, hi, z_bottom);
        normal = sc_v3(1.f, 0.f, 0.f);
        break;
    case 2:
        c[0] = sc_v3(hi, fixed, z_top);
        c[1] = sc_v3(lo, fixed, z_top);
        c[2] = sc_v3(lo, fixed, z_bottom);
        c[3] = sc_v3(hi, fixed, z_bottom);
        normal = sc_v3(0.f, -1.f, 0.f);
        break;
    default:
        c[0] = sc_v3(lo, fixed, z_top);
        c[1] = sc_v3(hi, fixed, z_top);
        c[2] = sc_v3(hi, fixed, z_bottom);
        c[3] = sc_v3(lo, fixed, z_bottom);
        normal = sc_v3(0.f, 1.f, 0.f);
        break;
    }
    return sc_buf_flat_quad(b, c, normal, &m->color);
}

/* Walls on all sides of cell (i, j) that stand taller than what is next to them. */
static int hm_emit_skirts_cell(ScBuf *b, const HmMesh *m, int i, int j) {
    float h = hm_at(m, i, j);
    float x = m->origin_x + (float)i * m->cell;
    float y = m->origin_y + (float)j * m->cell;
    int side;
    for (side = 0; side < 4; side++) {
        float z_bottom, fixed, lo, hi;
        int needed, rc;
        hm_wall_info(m, side, i, j, h, &needed, &z_bottom);
        if (!needed || !(h > z_bottom + HM_MERGE)) {
            continue;
        }
        switch (side) {
        case 0:
            fixed = x;
            lo = y;
            hi = y + m->cell;
            break;
        case 1:
            fixed = m->origin_x + (float)(i + 1) * m->cell;
            lo = y;
            hi = y + m->cell;
            break;
        case 2:
            fixed = y;
            lo = x;
            hi = x + m->cell;
            break;
        default:
            fixed = m->origin_y + (float)(j + 1) * m->cell;
            lo = x;
            hi = x + m->cell;
            break;
        }
        rc = hm_emit_wall(b, m, side, fixed, lo, hi, h, z_bottom);
        if (rc != 0) {
            return rc;
        }
    }
    return 0;
}

/* The underside: flat quads at min_z over every cell with stock, merged into big rectangles. */
typedef struct HmBottomCtx {
    ScBuf *b;
    const HmMesh *m;
} HmBottomCtx;

static int hm_bottom_rect(void *vctx, int i0, int j0, int i1, int j1) {
    HmBottomCtx *ctx = (HmBottomCtx *)vctx;
    const HmMesh *m = ctx->m;
    float x0 = m->origin_x + (float)i0 * m->cell;
    float y0 = m->origin_y + (float)j0 * m->cell;
    float x1 = m->origin_x + (float)i1 * m->cell;
    float y1 = m->origin_y + (float)j1 * m->cell;
    ScV3 c[4];
    c[0] = sc_v3(x0, y0, m->min_z);
    c[1] = sc_v3(x0, y1, m->min_z);
    c[2] = sc_v3(x1, y1, m->min_z);
    c[3] = sc_v3(x1, y0, m->min_z);
    return sc_buf_flat_quad(ctx->b, c, sc_v3(0.f, 0.f, -1.f), &m->color);
}

/* Underside for the columns [j0, j1) of the window. */
static int hm_emit_bottom(ScBuf *b, const HmMesh *m, int j0, int j1) {
    HmBottomCtx ctx;
    uint8_t *used;
    int i, j, rc;
    used = (uint8_t *)malloc((size_t)m->gw * (size_t)m->gh);
    if (used == NULL) {
        return -1;
    }
    for (i = 0; i < m->gw; i++) {
        for (j = 0; j < m->gh; j++) {
            used[i * m->gh + j] = m->valid[i * m->gh + j] ? 0 : 1;
        }
    }
    ctx.b = b;
    ctx.m = m;
    rc = sc_greedy_rects(used, m->gw, m->gh, j0, j1, hm_bottom_rect, &ctx);
    free(used);
    return rc;
}

/* Mesh the cells in columns [j0, j1) into `b`: tops, walls, and underside. */
static int hm_emit_rows(ScBuf *b, const HmMesh *m, int j0, int j1) {
    HmStrip strip;
    int nids;
    int i, j, rc;
    strip.j0 = j0;
    strip.nrows = j1 - j0 + 1;
    nids = (m->gw + 1) * strip.nrows;
    strip.ids = (int *)malloc((size_t)nids * sizeof(int));
    if (strip.ids == NULL) {
        return -1;
    }
    for (i = 0; i < nids; i++) {
        strip.ids[i] = -1;
    }
    for (i = 0; i < m->gw; i++) {
        for (j = j0; j < j1; j++) {
            int i00, i10, i11, i01;
            if (!m->valid[i * m->gh + j]) {
                continue;
            }
            i00 = hm_corner_vert(b, m, &strip, i, j, 0, 0);
            i10 = hm_corner_vert(b, m, &strip, i, j, 1, 0);
            i11 = hm_corner_vert(b, m, &strip, i, j, 1, 1);
            i01 = hm_corner_vert(b, m, &strip, i, j, 0, 1);
            if (i00 < 0 || i10 < 0 || i11 < 0 || i01 < 0) {
                free(strip.ids);
                return i00 < 0 ? i00 : (i10 < 0 ? i10 : (i11 < 0 ? i11 : i01));
            }
            rc = sc_buf_quad(b, i00, i10, i11, i01);
            if (rc == 0) {
                rc = hm_emit_skirts_cell(b, m, i, j);
            }
            if (rc != 0) {
                free(strip.ids);
                return rc;
            }
        }
    }
    free(strip.ids);
    return hm_emit_bottom(b, m, j0, j1);
}

/* Mesh columns [j0, j1) into one part. If that needs more vertices or indices than a part can hold
 * (-2), split the range in half and try each half.
 */
static int hm_emit_split(ScMeshBatch *out, const HmMesh *m, int j0, int j1) {
    ScBuf buf;
    int rc, mid;
    sc_buf_init(&buf);
    rc = hm_emit_rows(&buf, m, j0, j1);
    if (rc == 0) {
        return sc_batch_add(out, &buf);
    }
    sc_buf_free(&buf);
    if (rc != -2 || j1 - j0 <= 1) {
        return -1;
    }
    mid = j0 + (j1 - j0) / 2;
    if (hm_emit_split(out, m, j0, mid) != 0) {
        return -1;
    }
    return hm_emit_split(out, m, mid, j1);
}

/* Fast path for a window that is one flat plateau: a single top quad, one underside quad, and one wall per
 * side (merged along runs where the neighbor's height is the same). `h` is the plateau height.
 */
static int hm_emit_uniform(ScMeshBatch *out, const HmMesh *m, float h) {
    ScBuf buf;
    float x1 = m->origin_x + (float)m->gw * m->cell;
    float y1 = m->origin_y + (float)m->gh * m->cell;
    ScV3 c[4];
    int side, rc;
    sc_buf_init(&buf);

    c[0] = sc_v3(m->origin_x, m->origin_y, h);
    c[1] = sc_v3(x1, m->origin_y, h);
    c[2] = sc_v3(x1, y1, h);
    c[3] = sc_v3(m->origin_x, y1, h);
    rc = sc_buf_flat_quad(&buf, c, sc_v3(0.f, 0.f, 1.f), &m->color);
    if (rc != 0) {
        sc_buf_free(&buf);
        return -1;
    }
    if (h > m->min_z + HM_MERGE) {
        c[0] = sc_v3(m->origin_x, m->origin_y, m->min_z);
        c[1] = sc_v3(m->origin_x, y1, m->min_z);
        c[2] = sc_v3(x1, y1, m->min_z);
        c[3] = sc_v3(x1, m->origin_y, m->min_z);
        rc = sc_buf_flat_quad(&buf, c, sc_v3(0.f, 0.f, -1.f), &m->color);
        if (rc != 0) {
            sc_buf_free(&buf);
            return -1;
        }
    }
    for (side = 0; side < 4; side++) {
        /* Walk along this side of the window, one border cell at a time. */
        int n = (side < 2) ? m->gh : m->gw;
        int j = 0;
        while (j < n) {
            int bi = 0, bj = 0; /* the border cell next to the wall */
            int needed, w;
            float z_bottom, fixed, lo, hi;
            switch (side) {
            case 0:
                bi = 0;
                bj = j;
                break;
            case 1:
                bi = m->gw - 1;
                bj = j;
                break;
            case 2:
                bi = j;
                bj = 0;
                break;
            default:
                bi = j;
                bj = m->gh - 1;
                break;
            }
            hm_wall_info(m, side, bi, bj, h, &needed, &z_bottom);
            if (!needed || h <= z_bottom + HM_MERGE) {
                j++;
                continue;
            }
            /* Extend the wall over following cells that need one down to the same depth. */
            w = 1;
            while (j + w < n) {
                int next_needed;
                float next_bottom;
                int nbi = side < 2 ? bi : j + w;
                int nbj = side < 2 ? j + w : bj;
                hm_wall_info(m, side, nbi, nbj, h, &next_needed, &next_bottom);
                if (!next_needed) {
                    break;
                }
                if (fabsf(next_bottom - z_bottom) > HM_MERGE) {
                    break;
                }
                w++;
            }
            if (side < 2) {
                fixed = side == 0 ? m->origin_x : x1;
                lo = m->origin_y + (float)j * m->cell;
                hi = m->origin_y + (float)(j + w) * m->cell;
            } else {
                fixed = side == 2 ? m->origin_y : y1;
                lo = m->origin_x + (float)j * m->cell;
                hi = m->origin_x + (float)(j + w) * m->cell;
            }
            rc = hm_emit_wall(&buf, m, side, fixed, lo, hi, h, z_bottom);
            if (rc != 0) {
                sc_buf_free(&buf);
                return -1;
            }
            j += w;
        }
    }
    return sc_batch_add(out, &buf);
}

/* Mark which cells have stock. Reports whether any do, and the lowest / highest / last such height. */
static void hm_find_valid(HmMesh *m, int *any, int *all_valid, float *h_min, float *h_max, float *h_last) {
    int i, j;
    *any = 0;
    *all_valid = 1;
    *h_min = 1e30f;
    *h_max = -1e30f;
    *h_last = 0.f;
    for (i = 0; i < m->gw; i++) {
        for (j = 0; j < m->gh; j++) {
            float h = hm_at(m, i, j);
            int ok = hm_valid(h, m->min_z);
            m->valid[i * m->gh + j] = (uint8_t)ok;
            if (!ok) {
                *all_valid = 0;
                continue;
            }
            *any = 1;
            if (h < *h_min) {
                *h_min = h;
            }
            if (h > *h_max) {
                *h_max = h;
            }
            *h_last = h;
        }
    }
}

/* Decide which grid corners can be welded: those where all (up to four) cells around it with stock have
 * the same height. Their shared height is the average.
 */
static void hm_find_shared_corners(HmMesh *m) {
    int i, j;
    for (i = 0; i <= m->gw; i++) {
        for (j = 0; j <= m->gh; j++) {
            float sum = 0.f;
            float mn = 1e30f;
            float mx = -1e30f;
            int n = 0;
            int ax, ay;
            int slot = i * m->cstride + j;
            for (ax = i - 1; ax <= i; ax++) {
                for (ay = j - 1; ay <= j; ay++) {
                    float h = hm_at(m, ax, ay);
                    if (!hm_valid(h, m->min_z)) {
                        continue;
                    }
                    if (h < mn) {
                        mn = h;
                    }
                    if (h > mx) {
                        mx = h;
                    }
                    sum += h;
                    n++;
                }
            }
            if (n > 0 && (mx - mn) <= HM_MERGE) {
                m->share[slot] = 1;
                m->corner_z[slot] = sum / (float)n;
            } else {
                m->share[slot] = 0;
                m->corner_z[slot] = 0.f;
            }
        }
    }
}

int sc_mesh_heightmap(
    const float *patch,
    int gw,
    int gh,
    int x0,
    int y0,
    int grid_nx,
    int grid_ny,
    double origin_x,
    double origin_y,
    double min_z,
    double cell,
    double cr,
    double cg,
    double cb,
    double ca,
    ScMeshBatch *out) {
    HmMesh m;
    int ncorner, any, all_valid, rc;
    float h_min, h_max, h_last;
    if (out == NULL || patch == NULL || gw <= 0 || gh <= 0 || cell <= 0.0) {
        return -1;
    }
    memset(&m, 0, sizeof(m));
    m.patch = patch;
    m.stride = gh + 2;
    m.gw = gw;
    m.gh = gh;
    m.x0 = x0;
    m.y0 = y0;
    m.grid_nx = grid_nx;
    m.grid_ny = grid_ny;
    m.origin_x = (float)origin_x;
    m.origin_y = (float)origin_y;
    m.min_z = (float)min_z;
    m.cell = (float)cell;
    m.cliff = HM_CLIFF * m.cell;
    m.color = sc_color(cr, cg, cb, ca);
    m.cstride = gh + 1;

    ncorner = (gw + 1) * (gh + 1);
    m.valid = (uint8_t *)malloc((size_t)gw * (size_t)gh);
    m.share = (uint8_t *)malloc((size_t)ncorner);
    m.corner_z = (float *)malloc((size_t)ncorner * sizeof(float));
    if (m.valid == NULL || m.share == NULL || m.corner_z == NULL) {
        rc = -1;
        goto done;
    }

    hm_find_valid(&m, &any, &all_valid, &h_min, &h_max, &h_last);
    if (!any) {
        rc = 0;
        goto done;
    }
    if (all_valid && (h_max - h_min) <= HM_MERGE) {
        rc = hm_emit_uniform(out, &m, h_last);
        goto done;
    }
    hm_find_shared_corners(&m);
    {
        /* Split into strips up front. Retrying a strip that cannot fit would waste the vertices it
         * builds before hitting the uint16 cap. */
        int rows_fit = SC_MESH_MAX_VERTS / (gw + 1);
        int j;
        if (rows_fit > 4) {
            rows_fit = (rows_fit * 2) / 3;
        }
        if (rows_fit < 1) {
            rows_fit = 1;
        }
        rc = 0;
        for (j = 0; j < gh; j += rows_fit) {
            int j1 = j + rows_fit;
            if (j1 > gh) {
                j1 = gh;
            }
            if (hm_emit_split(out, &m, j, j1) != 0) {
                rc = -1;
                break;
            }
        }
    }
done:
    free(m.valid);
    free(m.share);
    free(m.corner_z);
    return rc;
}

/* ------------------------------------------------------------------------- */
/* Cylindrical mesh                                                          */
/* ------------------------------------------------------------------------- */

/* A tube around the X axis: one ring of vertices per X cell, joined into quads, with a disk cap at each
 * end of the stock. A window of the stock may be one slab of a longer tube.
 */
typedef struct CylMesh {
    const float *radii; /* (nx, n_theta) leftover radius per cell; negative means no stock */
    int nx, n_theta;
    int ix0;            /* X index of the window's first row in the full field */
    int nx_total;       /* number of X rows in the full field */
    double min_x, cell;
    double axis_y, axis_z;
    const double *sin_t;
    const double *cos_t;
    double floor_r;     /* radii below this are drawn as floor_r */
    ScColor color;
} CylMesh;

/* Normal of the triangle (a, b, c), as a plain double vector. */
static void cyl_cross(const float *a, const float *b, const float *c, double *n) {
    double bx = (double)b[0] - (double)a[0];
    double by = (double)b[1] - (double)a[1];
    double bz = (double)b[2] - (double)a[2];
    double cx = (double)c[0] - (double)a[0];
    double cy = (double)c[1] - (double)a[1];
    double cz = (double)c[2] - (double)a[2];
    n[0] = by * cz - bz * cy;
    n[1] = bz * cx - bx * cz;
    n[2] = bx * cy - by * cx;
}

/* A quad can be split into triangles along either diagonal. Pick the one that keeps both triangles facing
 * outward from the axis (so a notch in the surface doesn't get a flipped triangle). Returns 1 for the
 * second diagonal (i10-i01), 0 for the first (i00-i11).
 */
static int cyl_use_b(const float *verts, int i00, int i10, int i11, int i01, double ay, double az) {
    const float *p00 = verts + i00 * SC_VERT_FLOATS;
    const float *p10 = verts + i10 * SC_VERT_FLOATS;
    const float *p11 = verts + i11 * SC_VERT_FLOATS;
    const float *p01 = verts + i01 * SC_VERT_FLOATS;
    /* Direction from the axis to the quad's center (in the YZ plane). */
    double hy = 0.25 * ((double)p00[1] + (double)p10[1] + (double)p11[1] + (double)p01[1]) - ay;
    double hz = 0.25 * ((double)p00[2] + (double)p10[2] + (double)p11[2] + (double)p01[2]) - az;
    double n1[3], n2[3];
    double s1, s2, score_a, score_b;
    cyl_cross(p00, p10, p11, n1);
    cyl_cross(p00, p11, p01, n2);
    s1 = n1[1] * hy + n1[2] * hz;
    s2 = n2[1] * hy + n2[2] * hz;
    score_a = s1 < s2 ? s1 : s2;
    cyl_cross(p10, p11, p01, n1);
    cyl_cross(p10, p01, p00, n2);
    s1 = n1[1] * hy + n1[2] * hz;
    s2 = n2[1] * hy + n2[2] * hz;
    score_b = s1 < s2 ? s1 : s2;
    return score_b > score_a;
}

/* The radius to draw for a cell, or 0 if the cell has no stock. */
static int cyl_draw_r(float rr, double floor_r, double *r) {
    if (rr < 0.f) {
        return 0;
    }
    *r = (double)rr < floor_r ? floor_r : (double)rr;
    return 1;
}

/* Flat disk closing one end of the tube at row `local_ix` (global row `global_ix`).
 * `sign` is the direction the cap faces along X (-1 at the start of the stock, +1 at the end).
 * Returns 0, or a negative sc_buf_vert error.
 */
static int cyl_emit_cap(ScBuf *b, const CylMesh *m, int local_ix, int global_ix, float sign) {
    int *ring;
    int it, center, any;
    float x;
    ScV3 normal = sc_v3(sign, 0.f, 0.f);
    ring = (int *)malloc((size_t)m->n_theta * sizeof(int));
    if (ring == NULL) {
        return -1;
    }
    any = 0;
    x = (float)(m->min_x + ((double)global_ix + 0.5) * m->cell);
    for (it = 0; it < m->n_theta; it++) {
        double r;
        int id;
        if (!cyl_draw_r(m->radii[local_ix * m->n_theta + it], m->floor_r, &r)) {
            ring[it] = -1;
            continue;
        }
        id = sc_buf_vert(
            b, sc_v3(x, (float)(m->axis_y + r * m->sin_t[it]), (float)(m->axis_z + r * m->cos_t[it])), normal,
            &m->color);
        if (id < 0) {
            free(ring);
            return id;
        }
        ring[it] = id;
        any = 1;
    }
    if (!any) {
        free(ring);
        return 0;
    }
    center = sc_buf_vert(b, sc_v3(x, (float)m->axis_y, (float)m->axis_z), normal, &m->color);
    if (center < 0) {
        free(ring);
        return center;
    }
    /* Fan of triangles from the center, skipping gaps where a bin has no stock. */
    for (it = 0; it < m->n_theta; it++) {
        int a = ring[it];
        int c = ring[(it + 1) % m->n_theta];
        int rc;
        if (a < 0 || c < 0) {
            continue;
        }
        if (sign < 0.f) {
            rc = sc_buf_tri(b, center, a, c);
        } else {
            rc = sc_buf_tri(b, center, c, a);
        }
        if (rc != 0) {
            free(ring);
            return rc;
        }
    }
    free(ring);
    return 0;
}

/* Radius to draw at cell (ix, it), or `fallback` if the cell is outside the window or has no stock. */
static double cyl_neighbor_r(const CylMesh *m, int ix, int it, double fallback) {
    double r;
    if (ix < 0 || ix >= m->nx) {
        return fallback;
    }
    if (!cyl_draw_r(m->radii[(size_t)ix * (size_t)m->n_theta + (size_t)it], m->floor_r, &r)) {
        return fallback;
    }
    return r;
}

/* Surface normal at a ring vertex, from the neighboring points along X and around the ring. A purely
 * radial normal would light a relief as a smooth bar.
 */
static ScV3 cyl_shell_normal(const CylMesh *m, int local_ix, int global_ix, int it) {
    double rc, rxm, rxp, rtm, rtp;
    double pxm[3], pxp[3], ptm[3], ptp[3];
    double tx0, tx1, tx2, tt0, tt1, tt2;
    double n0, n1, n2, len, outward;
    int lxm, lxp, itm, itp;
    int gxm, gxp;

    rc = cyl_neighbor_r(m, local_ix, it, m->floor_r);
    lxm = local_ix > 0 ? local_ix - 1 : local_ix;
    lxp = local_ix + 1 < m->nx ? local_ix + 1 : local_ix;
    gxm = global_ix + (lxm - local_ix);
    gxp = global_ix + (lxp - local_ix);
    itm = it > 0 ? it - 1 : m->n_theta - 1;
    itp = it + 1 < m->n_theta ? it + 1 : 0;
    rxm = cyl_neighbor_r(m, lxm, it, rc);
    rxp = cyl_neighbor_r(m, lxp, it, rc);
    rtm = cyl_neighbor_r(m, local_ix, itm, rc);
    rtp = cyl_neighbor_r(m, local_ix, itp, rc);

    /* Points before / after along X, and before / after around the ring. */
    pxm[0] = m->min_x + ((double)gxm + 0.5) * m->cell;
    pxm[1] = m->axis_y + rxm * m->sin_t[it];
    pxm[2] = m->axis_z + rxm * m->cos_t[it];
    pxp[0] = m->min_x + ((double)gxp + 0.5) * m->cell;
    pxp[1] = m->axis_y + rxp * m->sin_t[it];
    pxp[2] = m->axis_z + rxp * m->cos_t[it];
    ptm[0] = m->min_x + ((double)global_ix + 0.5) * m->cell;
    ptm[1] = m->axis_y + rtm * m->sin_t[itm];
    ptm[2] = m->axis_z + rtm * m->cos_t[itm];
    ptp[0] = ptm[0];
    ptp[1] = m->axis_y + rtp * m->sin_t[itp];
    ptp[2] = m->axis_z + rtp * m->cos_t[itp];
    tx0 = pxp[0] - pxm[0];
    tx1 = pxp[1] - pxm[1];
    tx2 = pxp[2] - pxm[2];
    tt0 = ptp[0] - ptm[0];
    tt1 = ptp[1] - ptm[1];
    tt2 = ptp[2] - ptm[2];
    /* cross(dP/dx, dP/dtheta) is perpendicular to the surface; flip it if it points inward. */
    n0 = tx1 * tt2 - tx2 * tt1;
    n1 = tx2 * tt0 - tx0 * tt2;
    n2 = tx0 * tt1 - tx1 * tt0;
    outward = n1 * m->sin_t[it] + n2 * m->cos_t[it];
    if (outward < 0.0) {
        n0 = -n0;
        n1 = -n1;
        n2 = -n2;
    }
    len = sqrt(n0 * n0 + n1 * n1 + n2 * n2);
    if (len < 1e-12) {
        return sc_v3(0.f, (float)m->sin_t[it], (float)m->cos_t[it]);
    }
    return sc_v3((float)(n0 / len), (float)(n1 / len), (float)(n2 / len));
}

/* Mesh rows [ix, ix + room) of the window into one part: the shell, plus a cap at either end if asked.
 * Returns 0, or -1 on failure.
 */
static int cyl_emit_part(ScMeshBatch *out, const CylMesh *m, int ix, int room, int cap_lo, int cap_hi) {
    ScBuf buf;
    int *ids;
    int local, it, rc;
    sc_buf_init(&buf);
    ids = (int *)malloc((size_t)room * (size_t)m->n_theta * sizeof(int));
    if (ids == NULL) {
        return -1;
    }
    /* One vertex per cell that has stock. */
    for (local = 0; local < room; local++) {
        int gix = m->ix0 + ix + local;
        float x = (float)(m->min_x + ((double)gix + 0.5) * m->cell);
        for (it = 0; it < m->n_theta; it++) {
            double r;
            int id = -1;
            float rr = m->radii[(ix + local) * m->n_theta + it];
            if (cyl_draw_r(rr, m->floor_r, &r)) {
                ScV3 normal = cyl_shell_normal(m, ix + local, gix, it);
                id = sc_buf_vert(
                    &buf,
                    sc_v3(x, (float)(m->axis_y + r * m->sin_t[it]), (float)(m->axis_z + r * m->cos_t[it])), normal,
                    &m->color);
                if (id < 0) {
                    free(ids);
                    sc_buf_free(&buf);
                    return -1;
                }
            }
            ids[local * m->n_theta + it] = id;
        }
    }
    /* Quads between each pair of neighboring rows, wrapping around the ring. */
    for (local = 0; local < room - 1; local++) {
        for (it = 0; it < m->n_theta; it++) {
            int it1 = (it + 1) % m->n_theta;
            int i00 = ids[local * m->n_theta + it];
            int i10 = ids[(local + 1) * m->n_theta + it];
            int i11 = ids[(local + 1) * m->n_theta + it1];
            int i01 = ids[local * m->n_theta + it1];
            if (i00 < 0 || i10 < 0 || i11 < 0 || i01 < 0) {
                continue;
            }
            if (cyl_use_b(buf.v, i00, i10, i11, i01, m->axis_y, m->axis_z)) {
                rc = sc_buf_quad(&buf, i10, i11, i01, i00);
            } else {
                rc = sc_buf_quad(&buf, i00, i10, i11, i01);
            }
            if (rc != 0) {
                free(ids);
                sc_buf_free(&buf);
                return -1;
            }
        }
    }
    free(ids);
    if (cap_lo && cyl_emit_cap(&buf, m, ix, m->ix0 + ix, -1.f) != 0) {
        sc_buf_free(&buf);
        return -1;
    }
    if (cap_hi && cyl_emit_cap(&buf, m, ix + room - 1, m->ix0 + ix + room - 1, 1.f) != 0) {
        sc_buf_free(&buf);
        return -1;
    }
    if (sc_batch_add(out, &buf) != 0) {
        sc_buf_free(&buf);
        return -1;
    }
    return 0;
}

/* Largest number of points around the cylinder that we can mesh.
 * The smallest possible part is two rows of points plus an end disk on each side.
 * That needs 12 * n_theta indices, so n_theta can't go above 65535 / 12.
 */
#define SC_CYL_MAX_THETA (SC_MESH_MAX_INDICES / 12)

/* How many rows of the cylinder fit in one part, counting `n_caps` end disks.
 * We have to stay under both limits: vertices and indices.
 * The index limit is usually the one we hit first, because each ring of quads
 * between two rows costs n_theta * 6 indices.
 * Always returns at least 2. That is safe because sc_mesh_cylinder already
 * rejects any n_theta too large for two rows (see SC_CYL_MAX_THETA).
 */
static int cyl_room_limit(int n_theta, int n_caps) {
    int room_v, room_i, cap_v, cap_i, per, budget;
    if (n_theta < 1) {
        return 2;
    }
    cap_v = n_caps * (n_theta + 1);
    cap_i = n_caps * n_theta * 3;
    if (cap_v >= SC_MESH_MAX_VERTS) {
        room_v = 2;
    } else {
        room_v = (SC_MESH_MAX_VERTS - cap_v) / n_theta;
    }
    per = n_theta * 6;
    budget = SC_MESH_MAX_INDICES - cap_i;
    if (per < 1 || budget < per) {
        room_i = 2;
    } else {
        room_i = budget / per + 1;
    }
    if (room_v < 2) {
        room_v = 2;
    }
    if (room_i < 2) {
        room_i = 2;
    }
    return room_v < room_i ? room_v : room_i;
}

int sc_mesh_cylinder(
    const float *radii,
    int nx,
    int n_theta,
    int ix0,
    int nx_total,
    double min_x,
    double cell,
    double axis_y,
    double axis_z,
    const double *sin_t,
    const double *cos_t,
    double floor_r,
    double cr,
    double cg,
    double cb,
    double ca,
    ScMeshBatch *out) {
    CylMesh m;
    int ix;
    if (out == NULL || radii == NULL || sin_t == NULL || cos_t == NULL || nx <= 0 || n_theta < 3 ||
        n_theta > SC_CYL_MAX_THETA || cell <= 0.0) {
        return -1;
    }
    m.radii = radii;
    m.nx = nx;
    m.n_theta = n_theta;
    m.ix0 = ix0;
    m.nx_total = nx_total;
    m.min_x = min_x;
    m.cell = cell;
    m.axis_y = axis_y;
    m.axis_z = axis_z;
    m.sin_t = sin_t;
    m.cos_t = cos_t;
    m.floor_r = floor_r;
    m.color = sc_color(cr, cg, cb, ca);

    if (nx < 2) {
        /* A single row has no shell, only the caps (if this row is at an end of the stock). */
        ScBuf buf;
        int rc = 0;
        sc_buf_init(&buf);
        if (ix0 == 0) {
            rc = cyl_emit_cap(&buf, &m, 0, ix0, -1.f);
        }
        if (rc == 0 && ix0 + nx >= nx_total) {
            rc = cyl_emit_cap(&buf, &m, nx - 1, ix0 + nx - 1, 1.f);
        }
        if (rc != 0) {
            sc_buf_free(&buf);
            return -1;
        }
        return sc_batch_add(out, &buf);
    }
    /* Cut the window into parts that fit both GLES caps. Consecutive parts share
     * one row so the shell has no gap between them. Shared rings make the index
     * list grow much faster than the vertex count, so the index cap binds first.
     */
    ix = 0;
    while (ix < nx) {
        int remain = nx - ix;
        int cap_lo = (ix == 0 && ix0 == 0);
        int room, cap_hi;
        int room_with_hi = cyl_room_limit(n_theta, cap_lo + 1);
        if (ix0 + nx >= nx_total && remain <= room_with_hi) {
            room = remain;
            cap_hi = 1;
        } else {
            room = cyl_room_limit(n_theta, cap_lo);
            if (room > remain) {
                room = remain;
            }
            /* This tail fits in one part only when the end disk is left off. Taking it
             * now ends the loop and leaves the tube open. Keep one row for a last part
             * that still has room for the cap (parts overlap by that row). */
            if (ix0 + nx >= nx_total && remain > room_with_hi && room >= remain) {
                room = remain - 1;
            }
            cap_hi = 0;
        }
        if (cyl_emit_part(out, &m, ix, room, cap_lo, cap_hi) != 0) {
            return -1;
        }
        if (ix + room >= nx) {
            break;
        }
        ix += room - 1;
    }
    return 0;
}

/* ------------------------------------------------------------------------- */
/* Voxel chunk mesh                                                          */
/* ------------------------------------------------------------------------- */

/* Is the voxel at (i, j, k) of the chunk solid? Out of range counts as empty. */
static int voxel_solid(const uint8_t *occ, const uint8_t *valid, int cs, int i, int j, int k) {
    int id;
    if (i < 0 || j < 0 || k < 0 || i >= cs || j >= cs || k >= cs) {
        return 0;
    }
    id = (i * cs + j) * cs + k;
    if (valid != NULL && !valid[id]) {
        return 0;
    }
    if (occ == NULL) {
        return 1;
    }
    return occ[id] != 0;
}

/* Is the voxel just outside the chunk solid? mode 1 = yes, 2 = look it up in `face`, else no. */
static int voxel_face_bit(int mode, const uint8_t *face, int cs, int a, int b) {
    if (mode == 1) {
        return 1;
    }
    if (mode == 2 && face != NULL) {
        return face[a * cs + b] != 0;
    }
    return 0;
}

/* Close the current part and start a new one if another quad would pass a GLES cap. */
static int voxel_flush_room(ScMeshBatch *batch, ScBuf *b, int need_v, int need_i) {
    if (b->nv + need_v <= SC_MESH_MAX_VERTS && b->ni + need_i <= SC_MESH_MAX_INDICES) {
        return 0;
    }
    if (b->nv <= 0 && b->ni <= 0) {
        return -1;
    }
    return sc_batch_add(batch, b);
}

static int voxel_emit_quad(ScMeshBatch *batch, ScBuf *b, const ScV3 corners[4], ScV3 normal, const ScColor *color) {
    if (voxel_flush_room(batch, b, 4, 6) != 0) {
        return -1;
    }
    return sc_buf_flat_quad(b, corners, normal, color) != 0 ? -1 : 0;
}

/* The two axes that span a face perpendicular to each axis: X faces span (Y, Z), Y faces (X, Z), Z faces (X, Y). */
static const int VOXEL_AXIS_A[3] = {1, 0, 0};
static const int VOXEL_AXIS_B[3] = {2, 2, 1};

/* Where a merged rectangle sits. */
typedef struct VoxelRectCtx {
    ScMeshBatch *batch;
    ScBuf *buf;
    int axis;        /* which axis the face is perpendicular to */
    int sign;        /* +1 faces the positive direction, -1 the negative one */
    int index;       /* layer along `axis` */
    float origin[3]; /* chunk corner */
    float vs;        /* voxel size */
    ScColor color;
} VoxelRectCtx;

/* Face rectangle covering voxels [i0, i1) x [j0, j1) of the layer. */
static int voxel_emit_rect(void *vctx, int i0, int j0, int i1, int j1) {
    const VoxelRectCtx *r = (const VoxelRectCtx *)vctx;
    int ca = VOXEL_AXIS_A[r->axis];
    int cb = VOXEL_AXIS_B[r->axis];
    float plane = r->origin[r->axis] + (float)(r->index + (r->sign > 0 ? 1 : 0)) * r->vs;
    float a0 = r->origin[ca] + (float)i0 * r->vs;
    float a1 = r->origin[ca] + (float)i1 * r->vs;
    float b0 = r->origin[cb] + (float)j0 * r->vs;
    float b1 = r->origin[cb] + (float)j1 * r->vs;
    /* The four corners in one of two windings, picked so the quad faces the way `sign` says. */
    int reversed = (r->axis == 1) ? (r->sign > 0) : (r->sign < 0);
    float pa[4] = {a0, reversed ? a0 : a1, a1, reversed ? a1 : a0};
    float pb[4] = {b0, reversed ? b1 : b0, b1, reversed ? b0 : b1};
    float normal[3] = {0.f, 0.f, 0.f};
    ScV3 corners[4];
    float p[3];
    int k;
    for (k = 0; k < 4; k++) {
        p[r->axis] = plane;
        p[ca] = pa[k];
        p[cb] = pb[k];
        corners[k] = sc_v3(p[0], p[1], p[2]);
    }
    normal[r->axis] = (float)r->sign;
    return voxel_emit_quad(r->batch, r->buf, corners, sc_v3(normal[0], normal[1], normal[2]), &r->color);
}

/* Turn a mask of exposed voxels in one layer into merged rectangles. */
static int voxel_greedy_mask(const uint8_t *mask, int rows, int cols, VoxelRectCtx *rect) {
    uint8_t *used;
    int i, any, all, rc;
    any = 0;
    all = 1;
    for (i = 0; i < rows * cols; i++) {
        if (mask[i]) {
            any = 1;
        } else {
            all = 0;
        }
    }
    if (!any) {
        return 0;
    }
    if (all) {
        return voxel_emit_rect(rect, 0, 0, rows, cols);
    }
    used = (uint8_t *)malloc((size_t)rows * (size_t)cols);
    if (used == NULL) {
        return -1;
    }
    for (i = 0; i < rows * cols; i++) {
        used[i] = mask[i] ? 0 : 1;
    }
    rc = sc_greedy_rects(used, rows, cols, 0, cols, voxel_emit_rect, rect);
    free(used);
    return rc;
}

/* The voxel coordinates of cell (a, b) in layer `index` perpendicular to `axis`. */
static void voxel_layer_coords(int axis, int index, int a, int b, int *i, int *j, int *k) {
    if (axis == 0) {
        *i = index;
        *j = a;
        *k = b;
    } else if (axis == 1) {
        *i = a;
        *j = index;
        *k = b;
    } else {
        *i = a;
        *j = b;
        *k = index;
    }
}

/* Is the voxel next to (i, j, k) in the +axis (`plus`) or -axis direction solid? Beyond the chunk this asks
 * the neighboring chunk, via face_mode / face.
 */
static int voxel_neighbor(
    const uint8_t *occ,
    const uint8_t *valid,
    int cs,
    int axis,
    int plus,
    int i,
    int j,
    int k,
    int face_mode,
    const uint8_t *face) {
    int ni = i, nj = j, nk = k;
    if (axis == 0) {
        ni = i + (plus ? 1 : -1);
    } else if (axis == 1) {
        nj = j + (plus ? 1 : -1);
    } else {
        nk = k + (plus ? 1 : -1);
    }
    if (ni >= 0 && nj >= 0 && nk >= 0 && ni < cs && nj < cs && nk < cs) {
        return voxel_solid(occ, valid, cs, ni, nj, nk);
    }
    if (axis == 0) {
        return voxel_face_bit(face_mode, face, cs, j, k);
    }
    if (axis == 1) {
        return voxel_face_bit(face_mode, face, cs, i, k);
    }
    return voxel_face_bit(face_mode, face, cs, i, j);
}

int sc_mesh_voxel_chunk(
    const uint8_t *occ,
    const uint8_t *valid,
    int cs,
    const int face_mode[6],
    const uint8_t *const faces[6],
    double ox,
    double oy,
    double oz,
    double voxel,
    double cr,
    double cg,
    double cb,
    double ca,
    ScMeshBatch *out) {
    ScBuf buf;
    VoxelRectCtx rect;
    uint8_t *mask;
    int axis;
    if (out == NULL || cs <= 0 || voxel <= 0.0 || face_mode == NULL) {
        return -1;
    }
    rect.batch = out;
    rect.buf = &buf;
    rect.origin[0] = (float)ox;
    rect.origin[1] = (float)oy;
    rect.origin[2] = (float)oz;
    rect.vs = (float)voxel;
    rect.color = sc_color(cr, cg, cb, ca);
    mask = (uint8_t *)malloc((size_t)cs * (size_t)cs);
    if (mask == NULL) {
        return -1;
    }
    sc_buf_init(&buf);
    /* Faces are numbered 0..5 as +X, -X, +Y, -Y, +Z, -Z. */
    for (axis = 0; axis < 3; axis++) {
        int plus;
        for (plus = 1; plus >= 0; plus--) {
            int face_i = axis * 2 + (plus ? 0 : 1);
            int mode = face_mode[face_i];
            const uint8_t *face = faces != NULL ? faces[face_i] : NULL;
            int i0 = 0;
            int i1 = cs;
            int index;
            /* A solid block only exposes the outer layer on this side. */
            if (occ == NULL && valid == NULL) {
                if (mode == 1) {
                    continue;
                }
                i0 = plus ? cs - 1 : 0;
                i1 = i0 + 1;
            }
            for (index = i0; index < i1; index++) {
                int a, b, any, rc;
                /* Mark the voxels of this layer that are solid and have nothing solid in front of them. */
                any = 0;
                for (a = 0; a < cs; a++) {
                    for (b = 0; b < cs; b++) {
                        int vi, vj, vk, solid, covered;
                        voxel_layer_coords(axis, index, a, b, &vi, &vj, &vk);
                        solid = voxel_solid(occ, valid, cs, vi, vj, vk);
                        covered = voxel_neighbor(occ, valid, cs, axis, plus, vi, vj, vk, mode, face);
                        mask[a * cs + b] = (uint8_t)(solid && !covered);
                        if (mask[a * cs + b]) {
                            any = 1;
                        }
                    }
                }
                if (!any) {
                    continue;
                }
                rect.axis = axis;
                rect.sign = plus ? 1 : -1;
                rect.index = index;
                rc = voxel_greedy_mask(mask, cs, cs, &rect);
                if (rc != 0) {
                    free(mask);
                    sc_buf_free(&buf);
                    return -1;
                }
            }
        }
    }
    free(mask);
    if (sc_batch_add(out, &buf) != 0) {
        sc_buf_free(&buf);
        return -1;
    }
    return 0;
}
