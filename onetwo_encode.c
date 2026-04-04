/* onetwo_encode.c — ONETWO encoding layer.
 * "Don't send the AI data. Send the AI rules OF the data."
 * Values live in T (the substrate). Rules travel.
 *
 * Depends on: BitStream, bs_init, bs_set, bs_push, bs_popcount, bs_corr,
 *             bloom_set, hash32 (all defined in xyzt.c before this #include).
 */

/* ─── Layout: 4096-bit fixed output ─── */
#define OT_TOTAL    4096

/* REPETITION: what repeats */
#define OT_REP_OFF     0
#define OT_RUN_OFF     0       /* run-length spectrum: 256-bit bloom k=3 */
#define OT_RUN_LEN     256
#define OT_AUTO_OFF    256     /* autocorrelation: 32 lags × 8-bit thermometer */
#define OT_AUTO_LEN    256
#define OT_FREQ_OFF    512     /* frequency shape: 256-bit bloom k=3 */
#define OT_FREQ_LEN    256
#define OT_DIV_OFF     768     /* byte diversity: 256-bit thermometer */
#define OT_DIV_LEN     256

/* OPPOSITION: what contrasts */
#define OT_OPP_OFF     1024
#define OT_DELTA_OFF   1024    /* delta spectrum: 256-bit bloom k=3 */
#define OT_DELTA_LEN   256
#define OT_XOR_OFF     1280    /* XOR spectrum: 256-bit thermometer */
#define OT_XOR_LEN     256
#define OT_CLASS_OFF   1536    /* class transitions: 256-bit bloom k=3 */
#define OT_CLASS_LEN   256
#define OT_SYM_OFF     1792    /* symmetry: 256-bit thermometer */
#define OT_SYM_LEN     256

/* NESTING: what contains what */
#define OT_NEST_OFF    2048
#define OT_SCALE_OFF   2048    /* multi-scale similarity: 256-bit bloom k=3 */
#define OT_SCALE_LEN   256
#define OT_DEPTH_OFF   2304    /* delimiter depth: 256-bit bloom k=2 */
#define OT_DEPTH_LEN   256
#define OT_SUBREP_OFF  2560    /* substring repeats: 256-bit bloom k=2 */
#define OT_SUBREP_LEN  256
#define OT_ENTG_OFF    2816    /* entropy gradient: 32 blocks × 8-bit thermometer */
#define OT_ENTG_LEN    256

/* META */
#define OT_META_OFF    3072
#define OT_MLEN_OFF    3072    /* length class: 64-bit thermometer */
#define OT_MLEN_LEN    64
#define OT_MENT_OFF    3136    /* overall entropy: 64-bit thermometer */
#define OT_MENT_LEN    64
#define OT_MUNIQ_OFF   3200    /* unique byte ratio: 64-bit thermometer */
#define OT_MUNIQ_LEN   64
#define OT_MDENS_OFF   3264    /* density: 64-bit thermometer */
#define OT_MDENS_LEN   64
/* 3328-4095: reserved (768 bits) */

/* ─── Character class for opposition analysis ─── */
static int ot_char_class(uint8_t c) {
    if (c >= 'A' && c <= 'Z') return 0; /* upper */
    if (c >= 'a' && c <= 'z') return 1; /* lower */
    if (c >= '0' && c <= '9') return 2; /* digit */
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') return 3; /* space */
    if (c < 32 || c == 127) return 4; /* control */
    return 5; /* punct/other */
}

/* ─── Thermometer: set bits 0..on-1 in a section ─── */
static void ot_thermo(BitStream *bs, int base, int nbits, int on) {
    if (on > nbits) on = nbits;
    for (int i = 0; i < on; i++)
        bs_set(bs, base + i, 1);
}

/* ═══════════════════════════════════════════════
 * onetwo_parse — extract structural RULES from raw data
 * ═══════════════════════════════════════════════ */
static void onetwo_parse(const uint8_t *raw, size_t len, BitStream *out) {
    bs_init(out);
    out->len = OT_TOTAL;
    if (!raw || len == 0) return;

    /* Cap analysis length */
    int n = (int)len;
    if (n > 1024) n = 1024;

    /* ═══ Pre-compute: byte frequency histogram ═══ */
    int freq[256] = {0};
    for (int i = 0; i < n; i++) freq[raw[i]]++;

    int unique = 0;
    for (int c = 0; c < 256; c++) if (freq[c] > 0) unique++;

    /* ═══ SECTION 1: REPETITION (bits 0-1023) ═══ */

    /* 1a. Run-length spectrum (256-bit bloom, k=3)
     * Hash each unique run length found. "This data has runs of length 3, 7." */
    {
        int i = 0;
        int run_lengths_seen[256] = {0};
        while (i < n) {
            int j = i + 1;
            while (j < n && raw[j] == raw[i]) j++;
            int rlen = j - i;
            if (rlen > 0 && rlen <= 255 && !run_lengths_seen[rlen]) {
                run_lengths_seen[rlen] = 1;
                uint8_t buf[4];
                buf[0] = (uint8_t)(rlen & 0xFF);
                buf[1] = (uint8_t)((rlen >> 8) & 0xFF);
                buf[2] = 'R'; buf[3] = 'L';
                bloom_set(out, OT_RUN_OFF, OT_RUN_LEN, buf, 4, 3);
            }
            i = j;
        }
    }

    /* 1b. Autocorrelation (32 lags × 8-bit thermometer = 256 bits)
     * At each lag, how often does raw[i] == raw[i+lag]? */
    {
        int max_lag = 32;
        if (max_lag > n - 1) max_lag = n - 1;
        for (int lag = 1; lag <= max_lag; lag++) {
            int matches = 0;
            int pairs = n - lag;
            if (pairs <= 0) break;
            for (int i = 0; i < pairs; i++)
                if (raw[i] == raw[i + lag]) matches++;
            int score = matches * 8 / pairs; /* 0-8 thermometer */
            if (score > 8) score = 8;
            ot_thermo(out, OT_AUTO_OFF + (lag - 1) * 8, 8, score);
        }
    }

    /* 1c. Frequency shape (256-bit bloom, k=3)
     * Hash (byte_value, quantized_freq_bucket) pairs.
     * Captures WHICH bytes appear at WHICH frequency level. */
    {
        /* Quantize freq into 8 buckets: 0, 1, 2-3, 4-7, 8-15, 16-31, 32-63, 64+ */
        for (int c = 0; c < 256; c++) {
            if (freq[c] == 0) continue;
            int bucket;
            if      (freq[c] == 1)  bucket = 0;
            else if (freq[c] <= 3)  bucket = 1;
            else if (freq[c] <= 7)  bucket = 2;
            else if (freq[c] <= 15) bucket = 3;
            else if (freq[c] <= 31) bucket = 4;
            else if (freq[c] <= 63) bucket = 5;
            else                    bucket = 6;
            uint8_t buf[3] = { (uint8_t)c, (uint8_t)bucket, 'F' };
            bloom_set(out, OT_FREQ_OFF, OT_FREQ_LEN, buf, 3, 3);
        }
    }

    /* 1d. Byte diversity (256-bit thermometer)
     * How many unique byte values appear? 0-256 mapped to 0-256 bits. */
    ot_thermo(out, OT_DIV_OFF, OT_DIV_LEN, unique);

    /* ═══ SECTION 2: OPPOSITION (bits 1024-2047) ═══ */

    /* 2a. Delta spectrum (256-bit bloom, k=3)
     * Transition between successive bytes: classify delta into buckets. */
    if (n >= 2) {
        for (int i = 0; i < n - 1; i++) {
            int delta = (int)raw[i + 1] - (int)raw[i];
            /* Quantize: 16 buckets. Sign bit + magnitude category. */
            int sign = delta >= 0 ? 0 : 1;
            int mag = delta < 0 ? -delta : delta;
            int mag_bucket;
            if      (mag == 0)   mag_bucket = 0;
            else if (mag <= 3)   mag_bucket = 1;
            else if (mag <= 15)  mag_bucket = 2;
            else if (mag <= 31)  mag_bucket = 3;
            else if (mag <= 63)  mag_bucket = 4;
            else if (mag <= 127) mag_bucket = 5;
            else                 mag_bucket = 6;
            uint8_t buf[3] = { (uint8_t)sign, (uint8_t)mag_bucket, 'D' };
            bloom_set(out, OT_DELTA_OFF, OT_DELTA_LEN, buf, 3, 3);
        }
    }

    /* 2b. XOR spectrum (256-bit thermometer)
     * Histogram of XOR between successive bytes, top-32 values × 8 bits. */
    if (n >= 2) {
        int xor_hist[256] = {0};
        for (int i = 0; i < n - 1; i++)
            xor_hist[raw[i] ^ raw[i + 1]]++;

        /* Find top-32 XOR values by frequency */
        int top_vals[32], top_freq[32], n_top = 0;
        for (int v = 0; v < 256; v++) {
            if (xor_hist[v] == 0) continue;
            if (n_top < 32) {
                top_vals[n_top] = v;
                top_freq[n_top] = xor_hist[v];
                n_top++;
            } else {
                int min_k = 0;
                for (int k = 1; k < 32; k++)
                    if (top_freq[k] < top_freq[min_k]) min_k = k;
                if (xor_hist[v] > top_freq[min_k]) {
                    top_vals[min_k] = v;
                    top_freq[min_k] = xor_hist[v];
                }
            }
        }
        /* Thermometer encode each: frequency relative to max */
        int max_freq = 1;
        for (int k = 0; k < n_top; k++)
            if (top_freq[k] > max_freq) max_freq = top_freq[k];
        for (int k = 0; k < n_top; k++) {
            int on = top_freq[k] * 8 / max_freq;
            if (on > 8) on = 8;
            ot_thermo(out, OT_XOR_OFF + k * 8, 8, on);
        }
    }

    /* 2c. Class transitions (256-bit bloom, k=3)
     * Character class bigrams: (class_i, class_i+1) hashed. */
    if (n >= 2) {
        for (int i = 0; i < n - 1; i++) {
            uint8_t ca = (uint8_t)ot_char_class(raw[i]);
            uint8_t cb = (uint8_t)ot_char_class(raw[i + 1]);
            uint8_t buf[3] = { ca, cb, 'C' };
            bloom_set(out, OT_CLASS_OFF, OT_CLASS_LEN, buf, 3, 3);
        }
    }

    /* 2d. Symmetry (256 bits)
     * Compare first half to second half at scales 1, 2, 4, 8.
     * Encode match ratio as thermometer per scale. */
    {
        int half = n / 2;
        if (half >= 2) {
            int scales[] = {1, 2, 4, 8};
            for (int si = 0; si < 4 && scales[si] <= half; si++) {
                int s = scales[si];
                int chunks = half / s;
                if (chunks < 1) continue;
                int matches = 0, total = 0;
                for (int c = 0; c < chunks && c < 32; c++) {
                    int off1 = c * s;
                    int off2 = half + c * s;
                    if (off2 + s > n) break;
                    for (int b = 0; b < s; b++) {
                        if (raw[off1 + b] == raw[off2 + b]) matches++;
                        total++;
                    }
                }
                int score = total > 0 ? matches * 64 / total : 0;
                if (score > 64) score = 64;
                ot_thermo(out, OT_SYM_OFF + si * 64, 64, score);
            }
        }
    }

    /* ═══ SECTION 3: NESTING (bits 2048-3071) ═══ */

    /* 3a. Multi-scale self-similarity (256-bit bloom, k=3)
     * XOR adjacent equal-sized chunks. Hash (scale, quantized_similarity). */
    {
        int scales[] = {2, 4, 8, 16};
        for (int si = 0; si < 4; si++) {
            int s = scales[si];
            if (s > n) break;
            int n_chunks = n / s;
            for (int c = 0; c + 1 < n_chunks && c < 32; c++) {
                int off1 = c * s;
                int off2 = (c + 1) * s;
                int diff = 0;
                for (int b = 0; b < s; b++)
                    diff += __builtin_popcount(raw[off1 + b] ^ raw[off2 + b]);
                /* Quantize similarity: 0=identical, 1=similar, 2=different, 3=opposite */
                int max_diff = s * 8;
                int q = diff * 4 / (max_diff + 1);
                if (q > 3) q = 3;
                uint8_t buf[4] = { (uint8_t)si, (uint8_t)q, (uint8_t)(c & 31), 'S' };
                bloom_set(out, OT_SCALE_OFF, OT_SCALE_LEN, buf, 4, 3);
            }
        }
    }

    /* 3b. Delimiter depth (256-bit bloom, k=2)
     * Track nesting depth for (){}[]. Hash (delimiter_type, depth_reached). */
    {
        int depth[3] = {0}; /* 0=(), 1={}, 2=[] */
        int max_depth[3] = {0};
        for (int i = 0; i < n; i++) {
            int dtype = -1, dir = 0;
            switch (raw[i]) {
                case '(': dtype=0; dir=1; break;
                case ')': dtype=0; dir=-1; break;
                case '{': dtype=1; dir=1; break;
                case '}': dtype=1; dir=-1; break;
                case '[': dtype=2; dir=1; break;
                case ']': dtype=2; dir=-1; break;
            }
            if (dtype >= 0) {
                depth[dtype] += dir;
                if (depth[dtype] < 0) depth[dtype] = 0;
                if (depth[dtype] > max_depth[dtype])
                    max_depth[dtype] = depth[dtype];
            }
        }
        for (int d = 0; d < 3; d++) {
            if (max_depth[d] > 0) {
                for (int lev = 1; lev <= max_depth[d] && lev <= 16; lev++) {
                    uint8_t buf[3] = { (uint8_t)d, (uint8_t)lev, 'N' };
                    bloom_set(out, OT_DEPTH_OFF, OT_DEPTH_LEN, buf, 3, 2);
                }
            }
        }
    }

    /* 3c. Substring repeats (256-bit bloom, k=2)
     * For substring lengths 2,3,4: count unique vs total. Hash repeat counts. */
    {
        int sub_lens[] = {2, 3, 4};
        for (int si = 0; si < 3; si++) {
            int slen = sub_lens[si];
            if (slen > n) continue;
            int n_subs = n - slen + 1;
            if (n_subs <= 0) continue;

            /* Count repeats using hash collisions (bloom-style counting) */
            int repeat_count = 0;
            /* Simple O(n^2) check, capped for performance */
            int cap = n_subs > 128 ? 128 : n_subs;
            for (int i = 0; i < cap; i++) {
                uint32_t h1 = hash32(raw + i, slen);
                for (int j = i + 1; j < cap; j++) {
                    uint32_t h2 = hash32(raw + j, slen);
                    if (h1 == h2) { repeat_count++; break; }
                }
            }
            /* Hash (substring_length, repeat_count_bucket) */
            int bucket;
            if      (repeat_count == 0)  bucket = 0;
            else if (repeat_count <= 3)  bucket = 1;
            else if (repeat_count <= 10) bucket = 2;
            else if (repeat_count <= 30) bucket = 3;
            else                         bucket = 4;
            uint8_t buf[3] = { (uint8_t)slen, (uint8_t)bucket, 'P' };
            bloom_set(out, OT_SUBREP_OFF, OT_SUBREP_LEN, buf, 3, 2);
        }
    }

    /* 3d. Entropy gradient (32 blocks × 8-bit thermometer = 256 bits)
     * Block entropy vs global entropy. Encodes entropy SHAPE, not value. */
    {
        /* Global entropy */
        double global_ent = 0.0;
        for (int c = 0; c < 256; c++) {
            if (freq[c] == 0) continue;
            double p = (double)freq[c] / n;
            global_ent -= p * log2(p);
        }

        int bsz = n >= 64 ? n / 32 : 2;
        if (bsz < 2) bsz = 2;
        int n_blocks = n / bsz;
        if (n_blocks > 32) n_blocks = 32;

        for (int b = 0; b < n_blocks; b++) {
            int off = b * bsz;
            int blen = (off + bsz <= n) ? bsz : n - off;
            if (blen < 1) continue;

            int bf[256] = {0};
            for (int k = 0; k < blen; k++) bf[raw[off + k]]++;

            double bent = 0.0;
            for (int c = 0; c < 256; c++) {
                if (bf[c] == 0) continue;
                double p = (double)bf[c] / blen;
                bent -= p * log2(p);
            }

            /* Ratio: block_entropy / global_entropy, thermometer 0-8 */
            int ratio = global_ent > 0.01 ? (int)(bent * 8 / global_ent) : 0;
            if (ratio > 8) ratio = 8;
            ot_thermo(out, OT_ENTG_OFF + b * 8, 8, ratio);
        }
    }

    /* ═══ SECTION 4: META (bits 3072-4095) ═══ */

    /* 4a. Length class (64-bit thermometer) — log2 of input length */
    {
        int lclass = 0;
        int tmp = n;
        while (tmp > 1) { lclass++; tmp >>= 1; }
        if (lclass > 64) lclass = 64;
        ot_thermo(out, OT_MLEN_OFF, OT_MLEN_LEN, lclass);
    }

    /* 4b. Overall entropy (64-bit thermometer) */
    {
        double global_ent = 0.0;
        for (int c = 0; c < 256; c++) {
            if (freq[c] == 0) continue;
            double p = (double)freq[c] / n;
            global_ent -= p * log2(p);
        }
        /* Max entropy for 256 symbols = 8 bits. Scale 0-8 → 0-64. */
        int ent_score = (int)(global_ent * 8);
        if (ent_score > 64) ent_score = 64;
        ot_thermo(out, OT_MENT_OFF, OT_MENT_LEN, ent_score);
    }

    /* 4c. Unique byte ratio (64-bit thermometer) — unique/256 scaled */
    {
        int uratio = unique * 64 / 256;
        if (uratio > 64) uratio = 64;
        ot_thermo(out, OT_MUNIQ_OFF, OT_MUNIQ_LEN, uratio);
    }

    /* 4d. Density (64-bit thermometer) — set bits / total bits in raw */
    {
        int set_bits = 0;
        for (int i = 0; i < n; i++)
            set_bits += __builtin_popcount(raw[i]);
        int density = set_bits * 64 / (n * 8);
        if (density > 64) density = 64;
        ot_thermo(out, OT_MDENS_OFF, OT_MDENS_LEN, density);
    }
}

/* ═══════════════════════════════════════════════
 * onetwo_generate — pattern bitstream → raw bytes (reverse path)
 * Best-effort reconstruction. Lossy.
 * ═══════════════════════════════════════════════ */
static void onetwo_generate(const BitStream *in, uint8_t *out_buf, size_t *out_len) {
    if (!in || in->len == 0) { *out_len = 0; return; }

    /* Estimate length from META length class thermometer */
    int lclass = 0;
    for (int i = 0; i < OT_MLEN_LEN && i < 64; i++)
        if (bs_get(in, OT_MLEN_OFF + i)) lclass = i + 1;
    int est_len = 1 << lclass;
    if (est_len > 1024) est_len = 1024;
    if (est_len < 1) est_len = 1;

    /* Estimate entropy from META */
    int ent_score = 0;
    for (int i = 0; i < OT_MENT_LEN && i < 64; i++)
        if (bs_get(in, OT_MENT_OFF + i)) ent_score = i + 1;

    /* Estimate diversity from META */
    int div_score = 0;
    for (int i = 0; i < OT_MUNIQ_LEN && i < 64; i++)
        if (bs_get(in, OT_MUNIQ_OFF + i)) div_score = i + 1;
    int n_unique = div_score * 256 / 64;
    if (n_unique < 1) n_unique = 1;
    if (n_unique > 256) n_unique = 256;

    /* Generate bytes: distribute n_unique values across est_len positions.
     * Simple approach: cycle through the estimated unique set. */
    uint8_t palette[256];
    for (int i = 0; i < n_unique; i++)
        palette[i] = (uint8_t)(i * 256 / n_unique);

    for (int i = 0; i < est_len; i++)
        out_buf[i] = palette[i % n_unique];

    /* Apply density: if density thermometer suggests low set-bit ratio, mask high bits */
    int dens_score = 0;
    for (int i = 0; i < OT_MDENS_LEN && i < 64; i++)
        if (bs_get(in, OT_MDENS_OFF + i)) dens_score = i + 1;
    if (dens_score < 32) {
        uint8_t mask = (1 << (dens_score * 8 / 64 + 1)) - 1;
        for (int i = 0; i < est_len; i++)
            out_buf[i] &= mask;
    }

    *out_len = est_len;
}

/* ═══════════════════════════════════════════════
 * onetwo_self_observe — pattern of patterns
 * Feed bitstream's own bytes through onetwo_parse.
 * Noise has no self-similar structure → dies.
 * Signal reinforces → converges.
 * ═══════════════════════════════════════════════ */
static void onetwo_self_observe(const BitStream *bs, BitStream *out) {
    int byte_len = (bs->len + 7) / 8;
    if (byte_len > 1024) byte_len = 1024;
    onetwo_parse((const uint8_t *)bs->w, byte_len, out);
}
