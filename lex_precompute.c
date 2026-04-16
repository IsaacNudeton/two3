/*
 * lex_precompute.c — Preprocess corpus into .lex annotations
 *
 * Usage:
 *   lex_precompute <corpus.bin> <output.lex>
 *   lex_precompute --dump <corpus.bin>         (human-readable)
 *
 * Isaac & CC — April 2026
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lexer.h"

static const char *CLASS_NAMES[16] = {
    "WS", "NL", "LET", "DIG", "O(", "C)", "O{", "C}",
    "O[", "C]", "OP", "QUO", "HASH", "SEMI", "DC", "OTH"
};

static const char *TOKEN_POS_NAMES[4] = {
    "first", "mid", "last", "single"
};

static void dump_tokens(const uint8_t *data, const LexToken *tokens, int n, int max_lines) {
    int lines = 0;
    printf("%-6s %-4s %-6s %-5s %-4s %-4s %-4s %-4s %-3s %-7s %-4s\n",
           "pos", "byte", "class", "depth", "str", "chr", "lc", "bc", "tp", "tok_pos", "col");
    printf("--------------------------------------------------------------\n");

    for (int i = 0; i < n && lines < max_lines; i++) {
        char ch_repr[8];
        if (data[i] >= 32 && data[i] < 127)
            snprintf(ch_repr, sizeof(ch_repr), "'%c'", data[i]);
        else
            snprintf(ch_repr, sizeof(ch_repr), "x%02X", data[i]);

        int brace_d = tokens[i].depth_id / 16;
        int paren_d = tokens[i].depth_id % 16;
        int in_str = (tokens[i].flags & LEX_FLAG_IN_STRING) ? 1 : 0;
        int in_chr = (tokens[i].flags & LEX_FLAG_IN_CHAR) ? 1 : 0;
        int in_lc  = (tokens[i].flags & LEX_FLAG_IN_LINE_COMMENT) ? 1 : 0;
        int in_bc  = (tokens[i].flags & LEX_FLAG_IN_BLOCK_COMMENT) ? 1 : 0;
        int tp     = (tokens[i].flags & LEX_FLAG_TOKEN_POS_MASK) >> LEX_FLAG_TOKEN_POS_SHIFT;

        printf("%-6d %-4s %-6s %d/%d   %-4d %-4d %-4d %-4d %-3d %-7s %-4d\n",
               i, ch_repr, CLASS_NAMES[tokens[i].byte_class],
               brace_d, paren_d, in_str, in_chr, in_lc, in_bc,
               tp, TOKEN_POS_NAMES[tp], tokens[i].line_pos);

        if (data[i] == '\n') lines++;
    }
}

static void print_stats(const LexToken *tokens, int n) {
    int class_counts[16] = {0};
    int max_brace = 0, max_paren = 0;
    int string_bytes = 0, char_bytes = 0, line_comment_bytes = 0, block_comment_bytes = 0;

    for (int i = 0; i < n; i++) {
        class_counts[tokens[i].byte_class]++;
        int bd = tokens[i].depth_id / 16;
        int pd = tokens[i].depth_id % 16;
        if (bd > max_brace) max_brace = bd;
        if (pd > max_paren) max_paren = pd;
        if (tokens[i].flags & LEX_FLAG_IN_STRING) string_bytes++;
        if (tokens[i].flags & LEX_FLAG_IN_CHAR) char_bytes++;
        if (tokens[i].flags & LEX_FLAG_IN_LINE_COMMENT) line_comment_bytes++;
        if (tokens[i].flags & LEX_FLAG_IN_BLOCK_COMMENT) block_comment_bytes++;
    }

    printf("\n--- Class distribution ---\n");
    for (int c = 0; c < 16; c++) {
        if (class_counts[c] > 0)
            printf("  %-6s: %7d (%5.1f%%)\n", CLASS_NAMES[c], class_counts[c],
                   100.0 * class_counts[c] / n);
    }

    printf("\n--- Depth ---\n");
    printf("  Max brace depth: %d\n", max_brace);
    printf("  Max paren depth: %d\n", max_paren);

    printf("\n--- Literal regions ---\n");
    printf("  In string:        %7d (%5.1f%%)\n", string_bytes, 100.0 * string_bytes / n);
    printf("  In char:          %7d (%5.1f%%)\n", char_bytes, 100.0 * char_bytes / n);
    printf("  In line comment:  %7d (%5.1f%%)\n", line_comment_bytes, 100.0 * line_comment_bytes / n);
    printf("  In block comment: %7d (%5.1f%%)\n", block_comment_bytes, 100.0 * block_comment_bytes / n);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage:\n");
        printf("  lex_precompute <corpus.bin> <output.lex>\n");
        printf("  lex_precompute --dump <corpus.bin>\n");
        return 1;
    }

    int dump_mode = (strcmp(argv[1], "--dump") == 0);
    const char *input_path = dump_mode ? argv[2] : argv[1];
    const char *output_path = dump_mode ? NULL : argv[2];

    /* Read corpus */
    FILE *f = fopen(input_path, "rb");
    if (!f) { printf("Cannot open: %s\n", input_path); return 1; }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *data = (uint8_t*)malloc(fsize);
    fread(data, 1, fsize, f);
    fclose(f);
    printf("Corpus: %s (%ld bytes)\n", input_path, fsize);

    /* Lex */
    LexToken *tokens = (LexToken*)malloc(fsize * sizeof(LexToken));
    lexer_process(data, (int)fsize, tokens);
    printf("Lexed: %ld tokens\n", fsize);

    if (dump_mode) {
        dump_tokens(data, tokens, (int)fsize, 200);
        print_stats(tokens, (int)fsize);
    } else {
        /* Write .lex file */
        FILE *out = fopen(output_path, "wb");
        if (!out) { printf("Cannot write: %s\n", output_path); return 1; }

        LexFileHeader hdr;
        hdr.magic = LEX_MAGIC;
        hdr.corpus_size = (uint32_t)fsize;
        hdr.n_classes = 16;
        hdr.n_context_signals = 5;
        hdr.reserved = 0;
        fwrite(&hdr, sizeof(hdr), 1, out);
        fwrite(tokens, sizeof(LexToken), fsize, out);
        fclose(out);

        printf("Written: %s (%ld bytes header + %ld tokens)\n",
               output_path, (long)sizeof(hdr), fsize);

        print_stats(tokens, (int)fsize);
    }

    free(data);
    free(tokens);
    return 0;
}
