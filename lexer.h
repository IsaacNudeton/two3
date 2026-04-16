/*
 * lexer.h — Structural Context Lexer for {2,3} Architecture
 *
 * Enriches raw byte input with deterministic structural signals:
 *   1. Character class (16 classes, pure lookup)
 *   2. Nesting depth (brace + paren, state machine)
 *   3. String/comment/char literal detection
 *   4. Token boundary position
 *   5. Line position (indentation pattern)
 *
 * Properties:
 *   - Reversible: byte_val field recovers original byte
 *   - Deterministic: same input → same output
 *   - Single left-to-right pass (+ one pass for token boundaries)
 *
 * Isaac & CC — April 2026
 */

#ifndef LEXER_H
#define LEXER_H

#include <stdint.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════
 * Character classes — 16 values, pure function of byte
 * ═══════════════════════════════════════════════════════ */

#define LEX_WHITESPACE     0
#define LEX_NEWLINE        1
#define LEX_LETTER         2
#define LEX_DIGIT          3
#define LEX_OPEN_PAREN     4
#define LEX_CLOSE_PAREN    5
#define LEX_OPEN_BRACE     6
#define LEX_CLOSE_BRACE    7
#define LEX_OPEN_BRACKET   8
#define LEX_CLOSE_BRACKET  9
#define LEX_OPERATOR      10
#define LEX_QUOTE         11
#define LEX_HASH          12
#define LEX_SEMICOLON     13
#define LEX_DOT_COMMA     14
#define LEX_OTHER         15

static const uint8_t LEX_CLASS_TABLE[256] = {
    /* 0x00-0x0F: control chars */
    15, 15, 15, 15, 15, 15, 15, 15, 15,  0,  1, 15,  0,  1, 15, 15,
    /*              \t \n      \f \r          */
    /* 0x10-0x1F: more control */
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    /* 0x20-0x2F:  ! " # $ % & ' ( ) * + , - . / */
     0, 10, 11, 12, 15, 10, 10, 11,  4,  5, 10, 10, 14, 10, 14, 10,
    /* 0x30-0x3F: 0-9 : ; < = > ? */
     3,  3,  3,  3,  3,  3,  3,  3,  3,  3, 10, 13, 10, 10, 10, 10,
    /* 0x40-0x4F: @ A-O */
    15,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
    /* 0x50-0x5F: P-Z [ \ ] ^ _ */
     2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  8, 15,  9, 10,  2,
    /* 0x60-0x6F: ` a-o */
    15,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
    /* 0x70-0x7F: p-z { | } ~ DEL */
     2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  6, 10,  7, 10, 15,
    /* 0x80-0xFF: high bytes → OTHER */
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
};

/* ═══════════════════════════════════════════════════════
 * Per-byte annotation — the lexer's output
 * ═══════════════════════════════════════════════════════ */

typedef struct {
    uint8_t byte_val;    /* original byte (identity) */
    uint8_t byte_class;  /* 0-15 character class */
    uint8_t depth_id;    /* brace_depth*16 + paren_depth (0-255) */
    uint8_t flags;       /* bits: in_string:1, in_char:1, in_line_comment:1,
                                  in_block_comment:1, token_pos:2, unused:2 */
    uint8_t line_pos;    /* column mod 16 */
} LexToken;

/* Flag bit positions */
#define LEX_FLAG_IN_STRING        0x01
#define LEX_FLAG_IN_CHAR          0x02
#define LEX_FLAG_IN_LINE_COMMENT  0x04
#define LEX_FLAG_IN_BLOCK_COMMENT 0x08
#define LEX_FLAG_TOKEN_POS_MASK   0x30
#define LEX_FLAG_TOKEN_POS_SHIFT  4

/* Token positions */
#define LEX_TOKEN_FIRST  0
#define LEX_TOKEN_MIDDLE 1
#define LEX_TOKEN_LAST   2
#define LEX_TOKEN_SINGLE 3  /* single-byte token (first AND last) */

/* ═══════════════════════════════════════════════════════
 * Lexer state machine
 * ═══════════════════════════════════════════════════════ */

typedef struct {
    uint8_t brace_depth;
    uint8_t paren_depth;
    uint8_t in_string;
    uint8_t in_char;
    uint8_t in_line_comment;
    uint8_t in_block_comment;
    uint8_t escape_next;
    uint8_t prev_byte;
    uint16_t col;
} LexerState;

static void lexer_init(LexerState *s) {
    memset(s, 0, sizeof(LexerState));
}

/* Advance the lexer by one byte. Returns the annotation for this byte.
 * Token position is NOT set here — requires a second pass (see lexer_fill_token_pos). */
static LexToken lexer_advance(LexerState *s, uint8_t byte) {
    LexToken tok;
    tok.byte_val = byte;
    tok.byte_class = LEX_CLASS_TABLE[byte];
    tok.flags = 0;
    tok.line_pos = (uint8_t)(s->col & 0x0F);

    /* String/char/comment state — set flags BEFORE updating state,
     * so the opening quote/slash is marked as being in the construct */
    uint8_t in_literal = s->in_string | s->in_char |
                         s->in_line_comment | s->in_block_comment;

    if (s->in_string)        tok.flags |= LEX_FLAG_IN_STRING;
    if (s->in_char)          tok.flags |= LEX_FLAG_IN_CHAR;
    if (s->in_line_comment)  tok.flags |= LEX_FLAG_IN_LINE_COMMENT;
    if (s->in_block_comment) tok.flags |= LEX_FLAG_IN_BLOCK_COMMENT;

    /* Update state machine */
    if (s->escape_next) {
        s->escape_next = 0;
    } else if (byte == '\\' && (s->in_string || s->in_char)) {
        s->escape_next = 1;
    } else if (s->in_string) {
        if (byte == '"') s->in_string = 0;
    } else if (s->in_char) {
        if (byte == '\'') s->in_char = 0;
    } else if (s->in_line_comment) {
        if (byte == '\n') s->in_line_comment = 0;
    } else if (s->in_block_comment) {
        if (byte == '/' && s->prev_byte == '*') s->in_block_comment = 0;
    } else {
        /* Not in any literal — check for openers */
        if (byte == '"')  s->in_string = 1;
        else if (byte == '\'') s->in_char = 1;
        else if (byte == '/' && s->prev_byte == '/') s->in_line_comment = 1;
        else if (byte == '*' && s->prev_byte == '/') s->in_block_comment = 1;

        /* Depth tracking — only outside literals */
        if (byte == '{' && s->brace_depth < 15) s->brace_depth++;
        else if (byte == '}' && s->brace_depth > 0) s->brace_depth--;
        if (byte == '(' && s->paren_depth < 15) s->paren_depth++;
        else if (byte == ')' && s->paren_depth > 0) s->paren_depth--;
    }

    tok.depth_id = (uint8_t)(s->brace_depth * 16 + s->paren_depth);

    /* Column tracking */
    if (byte == '\n') {
        s->col = 0;
    } else {
        s->col++;
    }

    s->prev_byte = byte;
    return tok;
}

/* Second pass: fill in token position (first/middle/last/single).
 * A "token" is a maximal run of same-class bytes. */
static void lexer_fill_token_pos(LexToken *tokens, int n) {
    if (n == 0) return;

    for (int i = 0; i < n; i++) {
        int is_first = (i == 0) || (tokens[i].byte_class != tokens[i-1].byte_class);
        int is_last  = (i == n-1) || (tokens[i].byte_class != tokens[i+1].byte_class);

        uint8_t pos;
        if (is_first && is_last)  pos = LEX_TOKEN_SINGLE;
        else if (is_first)        pos = LEX_TOKEN_FIRST;
        else if (is_last)         pos = LEX_TOKEN_LAST;
        else                      pos = LEX_TOKEN_MIDDLE;

        tokens[i].flags = (tokens[i].flags & ~LEX_FLAG_TOKEN_POS_MASK) |
                          (pos << LEX_FLAG_TOKEN_POS_SHIFT);
    }
}

/* ═══════════════════════════════════════════════════════
 * Batch processing — lex an entire buffer
 * ═══════════════════════════════════════════════════════ */

static void lexer_process(const uint8_t *data, int n, LexToken *out) {
    LexerState state;
    lexer_init(&state);

    for (int i = 0; i < n; i++)
        out[i] = lexer_advance(&state, data[i]);

    lexer_fill_token_pos(out, n);
}

/* ═══════════════════════════════════════════════════════
 * .lex file format
 * ═══════════════════════════════════════════════════════ */

#define LEX_MAGIC 0x4C455831  /* "LEX1" */

typedef struct {
    uint32_t magic;
    uint32_t corpus_size;
    uint16_t n_classes;
    uint16_t n_context_signals;
    uint32_t reserved;
} LexFileHeader;

#endif /* LEXER_H */
