/* SHA-256 (FIPS 180-4), for naming the exact model, template, feature and head files. */
#include "s1.h"

#include <stdlib.h>
#include <string.h>

struct sha256 {
    uint32_t h[8];
    uint8_t  block[64];
    size_t   n_block; /* bytes waiting in block */
    uint64_t n_total; /* bytes hashed so far */
};

static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

static void compress(struct sha256 *s, const uint8_t *block)
{
    uint32_t w[64];
    for (int i = 0; i < 16; i++) {
        w[i] = (uint32_t)block[4 * i] << 24 | (uint32_t)block[4 * i + 1] << 16 |
               (uint32_t)block[4 * i + 2] << 8 | (uint32_t)block[4 * i + 3];
    }
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i]        = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t v[8];
    memcpy(v, s->h, sizeof v);
    for (int i = 0; i < 64; i++) {
        uint32_t s1  = rotr(v[4], 6) ^ rotr(v[4], 11) ^ rotr(v[4], 25);
        uint32_t ch  = (v[4] & v[5]) ^ (~v[4] & v[6]);
        uint32_t t1  = v[7] + s1 + ch + K[i] + w[i];
        uint32_t s0  = rotr(v[0], 2) ^ rotr(v[0], 13) ^ rotr(v[0], 22);
        uint32_t maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
        memmove(v + 1, v, 7 * sizeof v[0]);
        v[4] += t1;
        v[0] = t1 + s0 + maj;
    }
    for (int i = 0; i < 8; i++) {
        s->h[i] += v[i];
    }
}

static void sha256_init(struct sha256 *s)
{
    static const uint32_t h0[8] = { 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };
    memcpy(s->h, h0, sizeof h0);
    s->n_block = 0;
    s->n_total = 0;
}

static void sha256_update(struct sha256 *s, const uint8_t *data, size_t len)
{
    s->n_total += len;
    while (len > 0) {
        size_t take = 64 - s->n_block < len ? 64 - s->n_block : len;
        memcpy(s->block + s->n_block, data, take);
        s->n_block += take;
        data += take;
        len -= take;
        if (s->n_block == 64) {
            compress(s, s->block);
            s->n_block = 0;
        }
    }
}

static void sha256_final(struct sha256 *s, char hex[S1_SHA256_HEX])
{
    uint64_t bits    = s->n_total * 8;
    uint8_t  pad[72] = { 0x80 };
    size_t   n_pad   = (s->n_block < 56 ? 56 : 120) - s->n_block;
    for (int i = 0; i < 8; i++) {
        pad[n_pad + (size_t)i] = (uint8_t)(bits >> (56 - 8 * i));
    }
    sha256_update(s, pad, n_pad + 8);
    for (int i = 0; i < 8; i++) {
        snprintf(hex + 8 * i, 9, "%08x", s->h[i]);
    }
}

int s1_sha256_file(const char *path, char hex[S1_SHA256_HEX])
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "judgly: cannot open %s to hash it\n", path);
        return -1;
    }
    const size_t   cap = 1 << 20;
    uint8_t       *buf = malloc(cap);
    struct sha256  s;
    sha256_init(&s);
    size_t got;
    while (buf && (got = fread(buf, 1, cap, f)) > 0) {
        sha256_update(&s, buf, got);
    }
    int failed = !buf || ferror(f);
    free(buf);
    fclose(f);
    if (failed) {
        fprintf(stderr, "judgly: cannot read %s to hash it\n", path);
        return -1;
    }
    sha256_final(&s, hex);
    return 0;
}
