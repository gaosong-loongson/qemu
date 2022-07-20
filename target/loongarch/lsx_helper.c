#include "qemu/osdep.h"
#include "cpu.h"
#include "internals.h"
#include "exec/exec-all.h"
#include "exec/cpu_ldst.h"
#include "exec/helper-proto.h"
#include "fpu/softfloat.h"
#include "tcg.h"
/* Data format min and max values */
#define DF_BITS(df) (1 << ((df) + 3))

#define DF_MAX_INT(df)  (int64_t)((1LL << (DF_BITS(df) - 1)) - 1)
#define M_MAX_INT(m)    (int64_t)((1LL << ((m)         - 1)) - 1)

#define DF_MIN_INT(df)  (int64_t)(-(1LL << (DF_BITS(df) - 1)))
#define M_MIN_INT(m)    (int64_t)(-(1LL << ((m)         - 1)))

#define DF_MAX_UINT(df) (uint64_t)(-1ULL >> (64 - DF_BITS(df)))
#define M_MAX_UINT(m)   (uint64_t)(-1ULL >> (64 - (m)))

#define UNSIGNED(x, df) ((x) & DF_MAX_UINT(df))
#define SIGNED(x, df)                                                   \
    ((((int64_t)x) << (64 - DF_BITS(df))) >> (64 - DF_BITS(df)))

#define BIT_POSITION(x, df) ((uint64_t)(x) % DF_BITS(df))
/* Element-by-element access macros */
#define DF_ELEMENTS(df) (MSA_WRLEN / DF_BITS(df))
#define DF_V_ELEMENTS(df) (128 / DF_BITS(df))

static inline void msa_move_v(wr_t *pwd, wr_t *pws)
{
    pwd->d[0] = pws->d[0];
    pwd->d[1] = pws->d[1];
}

static inline void lsx_move_v(wr_t *pwd, wr_t *pws)
{
    pwd->d[0] = pws->d[0];
    pwd->d[1] = pws->d[1];
}

static inline void lsx_move_x(wr_t *pwd, wr_t *pws)
{
    pwd->d[0] = pws->d[0];
    pwd->d[1] = pws->d[1];
    pwd->d[2] = pws->d[2];
    pwd->d[3] = pws->d[3];
}


void dump_wr(wr_t *p);

void dump_wr(wr_t *p) {
    int i;

    for (i = 0; i < 16; i++) {
        printf("%04x ", p->h[i]);
    }
    printf("\n");
}

/* lsx floating-point vector */
#define float64_from_int32 int32_to_float64
#define Rw(pwr, i) (pwr->w[i])
#define Lw(pwr, i) (pwr->w[i + (128/32) / 2])

#define FLOAT_ONE32 make_float32(0x3f8 << 20)
#define FLOAT_ONE64 make_float64(0x3ffULL << 52)

#define FLOAT_SNAN16(s) (float16_default_nan(s) ^ 0x0220)
        /* 0x7c20 */
#define FLOAT_SNAN32(s) (float32_default_nan(s) ^ 0x00400020)
        /* 0x7f800020 */
#define FLOAT_SNAN64(s) (float64_default_nan(s) ^ 0x0008000000000020ULL)
        /* 0x7ff0000000000020 */

static inline void clear_msacsr_cause(CPULoongArchState *env)
{
    SET_FP_CAUSE(env->fcsr0, 0);
}

static inline void check_msacsr_cause(CPULoongArchState *env, uintptr_t retaddr)
{
    if ((GET_FP_CAUSE(env->fcsr0) &
            (GET_FP_ENABLE(env->fcsr0) | FP_UNIMPLEMENTED)) == 0) {
        UPDATE_FP_FLAGS(env->fcsr0,
                GET_FP_CAUSE(env->fcsr0));
    } else {
        do_raise_exception(env, EXCP_FPE, retaddr);
    }
}

/* Flush-to-zero use cases for update_msacsr() */
#define CLEAR_FS_UNDERFLOW 1
#define CLEAR_IS_INEXACT   2
#define RECIPROCAL_INEXACT 4

static inline int update_msacsr(CPULoongArchState *env, int action, int denormal)
{
    int ieee_ex;

    int c;
    int cause;
    int enable;

    ieee_ex = get_float_exception_flags(&env->fp_status);

    /* QEMU softfloat does not signal all underflow cases */
    if (denormal) {
        ieee_ex |= float_flag_underflow;
    }

    c = ieee_ex_to_mips(ieee_ex);
    enable = GET_FP_ENABLE(env->fcsr0) | FP_UNIMPLEMENTED;

    /* Set Inexact (I) when flushing inputs to zero */
    if ((ieee_ex & float_flag_input_denormal) &&
            (env->fcsr0 & MSACSR_FS_MASK) != 0) {
        if (action & CLEAR_IS_INEXACT) {
            c &= ~FP_INEXACT;
        } else {
            c |=  FP_INEXACT;
        }
    }

    /* Set Inexact (I) and Underflow (U) when flushing outputs to zero */
    if ((ieee_ex & float_flag_output_denormal) &&
            (env->fcsr0 & MSACSR_FS_MASK) != 0) {
        c |= FP_INEXACT;
        if (action & CLEAR_FS_UNDERFLOW) {
            c &= ~FP_UNDERFLOW;
        } else {
            c |=  FP_UNDERFLOW;
        }
    }

    /* Set Inexact (I) when Overflow (O) is not enabled */
    if ((c & FP_OVERFLOW) != 0 && (enable & FP_OVERFLOW) == 0) {
        c |= FP_INEXACT;
    }

    /* Clear Exact Underflow when Underflow (U) is not enabled */
    if ((c & FP_UNDERFLOW) != 0 && (enable & FP_UNDERFLOW) == 0 &&
            (c & FP_INEXACT) == 0) {
        c &= ~FP_UNDERFLOW;
    }

    /*
     * Reciprocal operations set only Inexact when valid and not
     * divide by zero
     */
    if ((action & RECIPROCAL_INEXACT) &&
            (c & (FP_INVALID | FP_DIV0)) == 0) {
        c = FP_INEXACT;
    }

    cause = c & enable;    /* all current enabled exceptions */

    if (cause == 0) {
        /*
         * No enabled exception, update the MSACSR Cause
         * with all current exceptions
         */
        SET_FP_CAUSE(env->fcsr0,
                (GET_FP_CAUSE(env->fcsr0) | c));
    } else {
        /* Current exceptions are enabled */
        if ((env->fcsr0 & MSACSR_NX_MASK) == 0) {
            /*
             * Exception(s) will trap, update MSACSR Cause
             * with all enabled exceptions
             */
            SET_FP_CAUSE(env->fcsr0,
                    (GET_FP_CAUSE(env->fcsr0) | c));
        }
    }

    return c;
}

static inline int get_enabled_exceptions(const CPULoongArchState *env, int c)
{
    int enable = GET_FP_ENABLE(env->fcsr0) | FP_UNIMPLEMENTED;
    return c & enable;
}

#define float16_is_zero(ARG) 0
#define float16_is_zero_or_denormal(ARG) 0
#define float32_from_int64 int64_to_float32

#define IS_DENORMAL(ARG, BITS)                      \
    (!float ## BITS ## _is_zero(ARG)                \
    && float ## BITS ## _is_zero_or_denormal(ARG))

#define MSA_FLOAT_UNOP(DEST, OP, ARG, BITS)                                 \
    do {                                                                    \
        float_status *status = &env->fp_status;               \
        int c;                                                              \
                                                                            \
        set_float_exception_flags(0, status);                               \
        DEST = float ## BITS ## _ ## OP(ARG, status);                       \
        c = update_msacsr(env, 0, IS_DENORMAL(DEST, BITS));                 \
                                                                            \
        if (get_enabled_exceptions(env, c)) {                               \
            DEST = ((FLOAT_SNAN ## BITS(status) >> 6) << 6) | c;            \
        }                                                                   \
    } while (0)

#define MSA_FLOAT_BINOP(DEST, OP, ARG1, ARG2, BITS)                         \
    do {                                                                    \
        float_status *status = &env->fp_status;               \
        int c;                                                              \
                                                                            \
        set_float_exception_flags(0, status);                               \
        DEST = float ## BITS ## _ ## OP(ARG1, ARG2, status);                \
        c = update_msacsr(env, 0, IS_DENORMAL(DEST, BITS));                 \
                                                                            \
        if (get_enabled_exceptions(env, c)) {                               \
            DEST = ((FLOAT_SNAN ## BITS(status) >> 6) << 6) | c;            \
        }                                                                   \
    } while (0)

#define MSA_FLOAT_MULADD(DEST, ARG1, ARG2, ARG3, NEGATE, BITS)              \
    do {                                                                    \
        float_status *status = &env->fp_status;               \
        int c;                                                              \
                                                                            \
        set_float_exception_flags(0, status);                               \
        DEST = float ## BITS ## _muladd(ARG2, ARG3, ARG1, NEGATE, status);  \
        c = update_msacsr(env, 0, IS_DENORMAL(DEST, BITS));                 \
                                                                            \
        if (get_enabled_exceptions(env, c)) {                               \
            DEST = ((FLOAT_SNAN ## BITS(status) >> 6) << 6) | c;            \
        }                                                                   \
    } while (0)

#define MSA_FLOAT_NEGATE_MULADD(DEST, ARG1, ARG2, ARG3, NEGATE, BITS)              \
    do {                                                                    \
        float_status *status = &env->fp_status;               \
        int c;                                                              \
                                                                            \
        set_float_exception_flags(0, status);                               \
        DEST = float ## BITS ## _muladd(ARG2, ARG3, ARG1, NEGATE, status);  \
		if (float##BITS##_is_normal(DEST) || 								\
			float##BITS##_is_zero_or_denormal(DEST) ||						\
			float##BITS##_is_infinity(DEST)) {								\
			DEST = float##BITS##_chs(DEST);									\
		}																	\
        c = update_msacsr(env, 0, IS_DENORMAL(DEST, BITS));                 \
                                                                            \
        if (get_enabled_exceptions(env, c)) {                               \
            DEST = ((FLOAT_SNAN ## BITS(status) >> 6) << 6) | c;            \
        }                                                                   \
    } while (0)

#define MSA_FLOAT_UNOP0(DEST, OP, ARG, BITS)                                \
    do {                                                                    \
        float_status *status = &env->fp_status;               \
        int c;                                                              \
                                                                            \
        set_float_exception_flags(0, status);                               \
        DEST = float ## BITS ## _ ## OP(ARG, status);                       \
        c = update_msacsr(env, CLEAR_FS_UNDERFLOW, 0);                      \
                                                                            \
        if (get_enabled_exceptions(env, c)) {                               \
            DEST = ((FLOAT_SNAN ## BITS(status) >> 6) << 6) | c;            \
        } else if (float ## BITS ## _is_any_nan(ARG)) {                     \
            DEST = 0;                                                       \
        }                                                                   \
    } while (0)


#define LSX_FMA(name, fmt, df, msa_len, fmt_bits, muladd_arg, width)     \
void helper_lsx_ ## name ## _ ## fmt(CPULoongArchState *env, uint32_t wd, \
                              uint32_t ws, uint32_t wt, uint32_t wr) \
{                                                                    \
    wr_t wx, *pwx = &wx;                                             \
    wr_t *pwd = &(env->fpr[wd].wr);                       \
    wr_t *pws = &(env->fpr[ws].wr);                       \
    wr_t *pwt = &(env->fpr[wt].wr);                       \
    wr_t *pwr = &(env->fpr[wr].wr);                       \
    uint32_t i;                                                      \
                                                                     \
    clear_msacsr_cause(env);                                         \
                                                                     \
    for (i = 0; i < msa_len / fmt_bits; i++) {                       \
        MSA_FLOAT_MULADD(pwx->df[i], pwr->df[i],                     \
                       pws->df[i], pwt->df[i], muladd_arg, fmt_bits);\
    }                                                                \
                                                                     \
    check_msacsr_cause(env, GETPC());                                \
    lsx_move_##width(pwd, pwx);                                            \
}

#define LSX_FMA_NEGATE(name, fmt, df, msa_len, fmt_bits, muladd_arg, width)     \
void helper_lsx_ ## name ## _ ## fmt(CPULoongArchState *env, uint32_t wd, \
                              uint32_t ws, uint32_t wt, uint32_t wr) \
{                                                                    \
    wr_t wx, *pwx = &wx;                                             \
    wr_t *pwd = &(env->fpr[wd].wr);                       \
    wr_t *pws = &(env->fpr[ws].wr);                       \
    wr_t *pwt = &(env->fpr[wt].wr);                       \
    wr_t *pwr = &(env->fpr[wr].wr);                       \
    uint32_t i;                                                      \
                                                                     \
    clear_msacsr_cause(env);                                         \
                                                                     \
    for (i = 0; i < msa_len / fmt_bits; i++) {                       \
        MSA_FLOAT_NEGATE_MULADD(pwx->df[i], pwr->df[i],                     \
                       pws->df[i], pwt->df[i], muladd_arg, fmt_bits);\
    }                                                                \
                                                                     \
    check_msacsr_cause(env, GETPC());                                \
    lsx_move_##width(pwd, pwx);                                            \
}

LSX_FMA(vfmadd , s, w, 128, 32, 0, v)                            // helper_lsx_vfmadd_s
LSX_FMA(vfmsub , s, w, 128, 32, float_muladd_negate_c, v)  // helper_lsx_vfmsub_s
LSX_FMA_NEGATE(vfnmadd, s, w, 128, 32, 0, v)   // helper_lsx_vfnmadd_s
LSX_FMA_NEGATE(vfnmsub, s, w, 128, 32, float_muladd_negate_c, v)   // helper_lsx_vfnmsub_s

LSX_FMA(vfmadd , d, d, 128, 64, 0, v)                            // helper_lsx_vfmadd_d
LSX_FMA(vfmsub , d, d, 128, 64, float_muladd_negate_c, v)  // helper_lsx_vfmsub_d
LSX_FMA_NEGATE(vfnmadd, d, d, 128, 64, 0, v)   // helper_lsx_vfnmadd_d
LSX_FMA_NEGATE(vfnmsub, d, d, 128, 64, float_muladd_negate_c, v)   // helper_lsx_vfnmsub_d

LSX_FMA(xvfmadd , s, w, 256, 32, 0, x)                           // helper_lsx_xvfmadd_s
LSX_FMA(xvfmsub , s, w, 256, 32, float_muladd_negate_c, x) // helper_lsx_xvfmsub_s
LSX_FMA_NEGATE(xvfnmadd, s, w, 256, 32, 0, x)  // helper_lsx_xvfnmadd_s
LSX_FMA_NEGATE(xvfnmsub, s, w, 256, 32, float_muladd_negate_c, x)   // helper_lsx_xvfnmsub_s

LSX_FMA(xvfmadd , d, d, 256, 64, 0, x)                           // helper_lsx_xvfmadd_d
LSX_FMA(xvfmsub , d, d, 256, 64, float_muladd_negate_c, x) // helper_lsx_xvfmsub_d
LSX_FMA_NEGATE(xvfnmadd, d, d, 256, 64, 0, x)  // helper_lsx_xvfnmadd_d
LSX_FMA_NEGATE(xvfnmsub, d, d, 256, 64, float_muladd_negate_c, x)   // helper_lsx_xvfnmsub_d
#undef LSX_FMA


#define LSX_FMADDSUB(name, msa_len, op1, op2, fmt, fmt_bits, df, width) \
void helper_lsx_ ## name ## _ ## fmt(CPULoongArchState *env, uint32_t wd, \
                              uint32_t ws, uint32_t wt, uint32_t wr) \
{                                                                    \
    wr_t wx, *pwx = &wx;                                             \
    wr_t *pwd = &(env->fpr[wd].wr);                       \
    wr_t *pws = &(env->fpr[ws].wr);                       \
    wr_t *pwt = &(env->fpr[wt].wr);                       \
    wr_t *pwr = &(env->fpr[wr].wr);                       \
    uint32_t i;                                                      \
                                                                     \
    clear_msacsr_cause(env);                                         \
                                                                     \
    for (i = 0; i < msa_len / fmt_bits / 2; i++) {                   \
        MSA_FLOAT_MULADD(pwx->df[2*i+1], pwr->df[2*i+1],             \
                     pws->df[2*i+1], pwt->df[2*i+1], op1, fmt_bits); \
        MSA_FLOAT_MULADD(pwx->df[2*i+0], pwr->df[2*i+0],             \
                     pws->df[2*i+0], pwt->df[2*i+0], op2, fmt_bits); \
    }                                                                \
                                                                     \
    check_msacsr_cause(env, GETPC());                                \
    lsx_move_##width(pwd, pwx);                                          \
}

LSX_FMADDSUB(vfmaddsub , 128, 0, float_muladd_negate_c, s, 32, w, v)   // helper_lsx_vfmaddsub_s
LSX_FMADDSUB(vfmaddsub , 128, 0, float_muladd_negate_c, d, 64, d, v)   // helper_lsx_vfmaddsub_d
LSX_FMADDSUB(vfmsubadd , 128, float_muladd_negate_c, 0, s, 32, w, v)   // helper_lsx_vfmsubadd_s
LSX_FMADDSUB(vfmsubadd , 128, float_muladd_negate_c, 0, d, 64, d, v)   // helper_lsx_vfmsubadd_d
LSX_FMADDSUB(xvfmaddsub, 256, 0, float_muladd_negate_c, s, 32, w, x)   // helper_lsx_xvfmaddsub_s
LSX_FMADDSUB(xvfmaddsub, 256, 0, float_muladd_negate_c, d, 64, d, x)   // helper_lsx_xvfmaddsub_d
LSX_FMADDSUB(xvfmsubadd, 256, float_muladd_negate_c, 0, s, 32, w, x)   // helper_lsx_xvfmsubadd_s
LSX_FMADDSUB(xvfmsubadd, 256, float_muladd_negate_c, 0, d, 64, d, x)   // helper_lsx_xvfmsubadd_d
#undef LSX_FMADDSUB

#define LSX_XVFADDSUB(name, msa_len, op1, op2, fmt, fmt_bits, df, width)                            \
void helper_lsx_ ## name ## _ ## fmt(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)      \
{                                                                                                   \
    wr_t wx, *pwx = &wx;                                                                            \
    wr_t *pwd = &(env->fpr[wd].wr);                                                      \
    wr_t *pws = &(env->fpr[ws].wr);                                                      \
    wr_t *pwt = &(env->fpr[wt].wr);                                                      \
                                                                                                    \
    uint32_t i;                                                                                     \
    clear_msacsr_cause(env);                                                                        \
    for (i = 0; i < msa_len / fmt_bits / 2; i++) {                                                  \
        MSA_FLOAT_BINOP(pwx->df[2 * i + 1], op1, pws->df[2 * i + 1], pwt->df[2 * i + 1], fmt_bits); \
        MSA_FLOAT_BINOP(pwx->df[2 * i + 0], op2, pws->df[2 * i + 0], pwt->df[2 * i + 0], fmt_bits); \
    }                                                                                               \
    check_msacsr_cause(env, GETPC());                                                               \
    lsx_move_##width(pwd, pwx);                                                                           \
}

LSX_XVFADDSUB(vfaddsub, 128, add, sub, s, 32, w, v)   // helper_lsx_vfaddsub_s
LSX_XVFADDSUB(vfaddsub, 128, add, sub, d, 64, d, v)   // helper_lsx_vfaddsub_d
LSX_XVFADDSUB(vfsubadd, 128, sub, add, s, 32, w, v)   // helper_lsx_vfsubadd_s
LSX_XVFADDSUB(vfsubadd, 128, sub, add, d, 64, d, v)   // helper_lsx_vfsubadd_d
LSX_XVFADDSUB(xvfaddsub, 256, add, sub, s, 32, w, x)   // helper_lsx_xvfaddsub_s
LSX_XVFADDSUB(xvfaddsub, 256, add, sub, d, 64, d, x)   // helper_lsx_xvfaddsub_d
LSX_XVFADDSUB(xvfsubadd, 256, sub, add, s, 32, w, x)   // helper_lsx_xvfsubadd_s
LSX_XVFADDSUB(xvfsubadd, 256, sub, add, d, 64, d, x)   // helper_lsx_xvfsubadd_d
#undef LSX_XVFADDSUB

void helper_lsx_vffint_s_l(CPULoongArchState *env, uint32_t wd,
                         uint32_t ws, uint32_t wt)
{
    wr_t wx, *pwx = &wx;
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);
    uint32_t i;

    clear_msacsr_cause(env);

    for (i = 0; i < 2; i++) {
        MSA_FLOAT_UNOP(pwx->w[i]    , from_int64, pwt->d[i], 32);
        MSA_FLOAT_UNOP(pwx->w[i + 2], from_int64, pws->d[i], 32);
    }
    check_msacsr_cause(env, GETPC());
    msa_move_v(pwd, pwx);
}

void helper_lsx_xvffint_s_l(CPULoongArchState *env, uint32_t wd,
                         uint32_t ws, uint32_t wt)
{
    wr_t wx, *pwx = &wx;
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);
    uint32_t i, j;

    clear_msacsr_cause(env);

    for (i = 0; i < 2; i++) {
        for (i = 0; j < 2; j++) {
            MSA_FLOAT_UNOP(pwx->w[j*4+i]    , from_int64, pwt->d[j*2+i], 32);
            MSA_FLOAT_UNOP(pwx->w[j*4+i + 2], from_int64, pws->d[j*2+i], 32);
        }
    }
    check_msacsr_cause(env, GETPC());
    lsx_move_x(pwd, pwx);
}

#define LSX_VFTINT(name, op)                                     \
void helper_lsx_ ## name ## _w_d(CPULoongArchState *env, uint32_t wd, \
                         uint32_t ws, uint32_t wt)               \
{                                                                \
    wr_t wx, *pwx = &wx;                                         \
    wr_t *pwd = &(env->fpr[wd].wr);                   \
    wr_t *pws = &(env->fpr[ws].wr);                   \
    wr_t *pwt = &(env->fpr[wt].wr);                   \
    uint32_t i;                                                  \
                                                                 \
    clear_msacsr_cause(env);                                     \
                                                                 \
    for (i = 0; i < 2; i++) {                                    \
        MSA_FLOAT_UNOP(pwx->w[i]    , op, pwt->d[i], 64);        \
        MSA_FLOAT_UNOP(pwx->w[i + 2], op, pws->d[i], 64);        \
    }                                                            \
    check_msacsr_cause(env, GETPC());                            \
    msa_move_v(pwd, pwx);                                        \
}
LSX_VFTINT(vftint   , to_int32                   )  // helper_lsx_vftint_w_d
LSX_VFTINT(vftintrm , to_int32_round_down        )  // helper_lsx_vftintrm_w_d
LSX_VFTINT(vftintrp , to_int32_round_up          )  // helper_lsx_vftintrp_w_d
LSX_VFTINT(vftintrz , to_int32_round_to_zero     )  // helper_lsx_vftintrz_w_d
LSX_VFTINT(vftintrne, to_int32_round_nearest_even)  // helper_lsx_vftintrne_w_d
#undef LSX_VFTINT

#define LSX_XVFTINT(name, op)                                     \
void helper_lsx_ ## name ## _w_d(CPULoongArchState *env, uint32_t wd, \
                         uint32_t ws, uint32_t wt)               \
{                                                                \
    wr_t wx, *pwx = &wx;                                         \
    wr_t *pwd = &(env->fpr[wd].wr);                   \
    wr_t *pws = &(env->fpr[ws].wr);                   \
    wr_t *pwt = &(env->fpr[wt].wr);                   \
    uint32_t i, j;                                               \
                                                                 \
    clear_msacsr_cause(env);                                     \
                                                                 \
    for (j = 0; j < 2; j++) {                                    \
        for (i = 0; i < 2; i++) {                                \
            MSA_FLOAT_UNOP(pwx->w[j*4+i]    , op, pwt->d[j*2+i], 64); \
            MSA_FLOAT_UNOP(pwx->w[j*4+i + 2], op, pws->d[j*2+i], 64); \
        }                                                        \
    }                                                            \
    check_msacsr_cause(env, GETPC());                            \
    lsx_move_x(pwd, pwx);                                        \
}
LSX_XVFTINT(xvftint   , to_int32                   )  // helper_lsx_xvftint_w_d
LSX_XVFTINT(xvftintrm , to_int32_round_down        )  // helper_lsx_xvftintrm_w_d
LSX_XVFTINT(xvftintrp , to_int32_round_up          )  // helper_lsx_xvftintrp_w_d
LSX_XVFTINT(xvftintrz , to_int32_round_to_zero     )  // helper_lsx_xvftintrz_w_d
LSX_XVFTINT(xvftintrne, to_int32_round_nearest_even)  // helper_lsx_xvftintrne_w_d
#undef LSX_XVFTINT

#define LSX_FRINT(rm)                                                        \
void helper_lsx_vfrint ## rm ## _df(CPULoongArchState *env, uint32_t df,          \
                         uint32_t wd, uint32_t ws)                           \
{                                                                            \
    wr_t wx, *pwx = &wx;                                                     \
    wr_t *pwd = &(env->fpr[wd].wr);                               \
    wr_t *pws = &(env->fpr[ws].wr);                               \
    uint32_t i;                                                              \
                                                                             \
    clear_msacsr_cause(env);                                                 \
                                                                             \
    switch (df) {                                                            \
    case DF_WORD:                                                            \
        for (i = 0; i < DF_ELEMENTS(DF_WORD); i++) {                         \
            MSA_FLOAT_UNOP(pwx->w[i], round_to_int_ ## rm, pws->w[i], 32);   \
        }                                                                    \
        break;                                                               \
    case DF_DOUBLE:                                                          \
        for (i = 0; i < DF_ELEMENTS(DF_DOUBLE); i++) {                       \
            MSA_FLOAT_UNOP(pwx->d[i], round_to_int_ ## rm, pws->d[i], 64);   \
        }                                                                    \
        break;                                                               \
    default:                                                                 \
        assert(0);                                                           \
    }                                                                        \
                                                                             \
    check_msacsr_cause(env, GETPC());                                        \
                                                                             \
    msa_move_v(pwd, pwx);                                                    \
}
LSX_FRINT(rne)  // helper_lsx_vfrintrne_df
LSX_FRINT(rz)   // helper_lsx_vfrintrz_df
LSX_FRINT(rp)   // helper_lsx_vfrintrp_df
LSX_FRINT(rm)   // helper_lsx_vfrintrm_df
#undef LSX_FRINT

#define LSX_VFFINT_D_W(hl, LR)                                       \
void helper_lsx_vffint ## hl ## _d_w(CPULoongArchState *env,              \
                                       uint32_t wd, uint32_t ws)     \
{                                                                    \
    wr_t wx, *pwx = &wx;                                             \
    wr_t *pwd = &(env->fpr[wd].wr);                       \
    wr_t *pws = &(env->fpr[ws].wr);                       \
    uint32_t i;                                                      \
                                                                     \
    clear_msacsr_cause(env);                                         \
                                                                     \
    for (i = 0; i < 2; i++) {                                        \
        MSA_FLOAT_UNOP(pwx->d[i], from_int32, LR ## w(pws, i), 64);  \
    }                                                                \
    check_msacsr_cause(env, GETPC());                                \
    msa_move_v(pwd, pwx);                                            \
}
LSX_VFFINT_D_W(l, R)
LSX_VFFINT_D_W(h, L)
#undef LSX_VFFINT_D_W

#define LSX_XVFFINT_D_W(hl, LR)                                       \
void helper_lsx_xvffint ## hl ## _d_w(CPULoongArchState *env,              \
                                       uint32_t wd, uint32_t ws)     \
{                                                                    \
    wr_t wx, *pwx = &wx;                                             \
    wr_t *pwd = &(env->fpr[wd].wr);                       \
    wr_t *pws = &(env->fpr[ws].wr);                       \
    uint32_t i, j;                                                   \
                                                                     \
    clear_msacsr_cause(env);                                         \
                                                                     \
    for (j = 0; j < 2; j++) {                                        \
        for (i = 0; i < 2; i++) {                                    \
            MSA_FLOAT_UNOP(pwx->d[j*2+i], from_int32, LR ## w(pws, j*4+i), 64);  \
        }                                                            \
    }                                                                \
    check_msacsr_cause(env, GETPC());                                \
    lsx_move_x(pwd, pwx);                                            \
}
LSX_XVFFINT_D_W(l, R)
LSX_XVFFINT_D_W(h, L)
#undef LSX_XVFFINT_D_W

#define LSX_VFTINT_S(rm)                                                 \
void helper_lsx_vftint ## rm ## _df(CPULoongArchState *env, uint32_t df,      \
                           uint32_t wd, uint32_t ws)                     \
{                                                                        \
    wr_t wx, *pwx = &wx;                                                 \
    wr_t *pwd = &(env->fpr[wd].wr);                           \
    wr_t *pws = &(env->fpr[ws].wr);                           \
    uint32_t i;                                                          \
                                                                         \
    clear_msacsr_cause(env);                                             \
                                                                         \
    switch (df) {                                                        \
    case DF_WORD:                                                        \
        for (i = 0; i < DF_ELEMENTS(DF_WORD); i++) {                     \
            MSA_FLOAT_UNOP0(pwx->w[i], to_int32_ ## rm, pws->w[i], 32);  \
        }                                                                \
        break;                                                           \
    case DF_DOUBLE:                                                      \
        for (i = 0; i < DF_ELEMENTS(DF_DOUBLE); i++) {                   \
            MSA_FLOAT_UNOP0(pwx->d[i], to_int64_ ## rm, pws->d[i], 64);  \
        }                                                                \
        break;                                                           \
    default:                                                             \
        assert(0);                                                       \
    }                                                                    \
                                                                         \
    check_msacsr_cause(env, GETPC());                                    \
                                                                         \
    msa_move_v(pwd, pwx);                                                \
}
LSX_VFTINT_S(rm)  // helper_lsx_vftintrm_df
LSX_VFTINT_S(rp)  // helper_lsx_vftintrp_df
LSX_VFTINT_S(rz)  // helper_lsx_vftintrz_df
LSX_VFTINT_S(rne) // helper_lsx_vftintrne_df
#undef LSX_VFTINT_S

#define LSX_VFTINT_L_S(hl, LR)                                       \
void helper_lsx_vftint ## hl ## _l_s(CPULoongArchState *env,              \
                                       uint32_t wd, uint32_t ws)     \
{                                                                    \
    wr_t wx, *pwx = &wx;                                             \
    wr_t *pwd = &(env->fpr[wd].wr);                       \
    wr_t *pws = &(env->fpr[ws].wr);                       \
    uint32_t i;                                                      \
                                                                     \
    clear_msacsr_cause(env);                                         \
                                                                     \
    for (i = 0; i < 2; i++) {                                        \
        MSA_FLOAT_UNOP(pwx->d[i], to_int64, LR ## w(pws, i), 32);    \
    }                                                                \
    check_msacsr_cause(env, GETPC());                                \
    msa_move_v(pwd, pwx);                                            \
}
LSX_VFTINT_L_S(h, L) // helper_lsx_vftinth_l_s
LSX_VFTINT_L_S(l, R) // helper_lsx_vftintl_l_s
#undef LSX_VFTINT_L_S

#define LSX_XVFTINT_L_S(hl, LR)                                       \
void helper_lsx_xvftint ## hl ## _l_s(CPULoongArchState *env,              \
                                       uint32_t wd, uint32_t ws)     \
{                                                                    \
    wr_t wx, *pwx = &wx;                                             \
    wr_t *pwd = &(env->fpr[wd].wr);                       \
    wr_t *pws = &(env->fpr[ws].wr);                       \
    uint32_t i, j;                                                   \
                                                                     \
    clear_msacsr_cause(env);                                         \
                                                                     \
    for (j = 0; j < 2; j++) {                                        \
        for (i = 0; i < 2; i++) {                                     \
            MSA_FLOAT_UNOP(pwx->d[j*2+i], to_int64, LR ## w(pws, j*4+i), 32); \
        }                                                             \
    }                                                                \
    check_msacsr_cause(env, GETPC());                                \
    lsx_move_x(pwd, pwx);                                            \
}
LSX_XVFTINT_L_S(h, L) // helper_lsx_xvftinth_l_s
LSX_XVFTINT_L_S(l, R) // helper_lsx_xvftintl_l_s
#undef LSX_XVFTINT_L_S

#define LSX_VFTINT_RM_L_S(hl, LR, rm)                                \
void helper_lsx_vftint ## rm ## hl ## _l_s(CPULoongArchState *env,        \
                                       uint32_t wd, uint32_t ws)     \
{                                                                    \
    wr_t wx, *pwx = &wx;                                             \
    wr_t *pwd = &(env->fpr[wd].wr);                       \
    wr_t *pws = &(env->fpr[ws].wr);                       \
    uint32_t i;                                                      \
                                                                     \
    clear_msacsr_cause(env);                                         \
                                                                     \
    for (i = 0; i < 2; i++) {                                        \
        MSA_FLOAT_UNOP(pwx->d[i], to_int64_##rm, LR ## w(pws, i), 32); \
    }                                                                \
    check_msacsr_cause(env, GETPC());                                \
    msa_move_v(pwd, pwx);                                            \
}
LSX_VFTINT_RM_L_S(h, L, rm)  // helper_lsx_vftintrmh_l_s
LSX_VFTINT_RM_L_S(l, R, rm)  // helper_lsx_vftintrml_l_s
LSX_VFTINT_RM_L_S(h, L, rp)  // helper_lsx_vftintrph_l_s
LSX_VFTINT_RM_L_S(l, R, rp)  // helper_lsx_vftintrpl_l_s
LSX_VFTINT_RM_L_S(h, L, rz)  // helper_lsx_vftintrzh_l_s
LSX_VFTINT_RM_L_S(l, R, rz)  // helper_lsx_vftintrzl_l_s
LSX_VFTINT_RM_L_S(h, L, rne) // helper_lsx_vftintrneh_l_s
LSX_VFTINT_RM_L_S(l, R, rne) // helper_lsx_vftintrnel_l_s
#undef LSX_VFTINT_RM_L_S

#define LSX_XVFTINT_RM_L_S(hl, LR, rm)                                \
void helper_lsx_xvftint ## rm ## hl ## _l_s(CPULoongArchState *env,        \
                                       uint32_t wd, uint32_t ws)     \
{                                                                    \
    wr_t wx, *pwx = &wx;                                             \
    wr_t *pwd = &(env->fpr[wd].wr);                       \
    wr_t *pws = &(env->fpr[ws].wr);                       \
    uint32_t i, j;                                                   \
                                                                     \
    clear_msacsr_cause(env);                                         \
                                                                     \
    for (j = 0; j < 2; j++) {                                        \
        for (i = 0; i < 2; i++) {                                        \
            MSA_FLOAT_UNOP(pwx->d[j*2+i], to_int64_##rm, LR ## w(pws, j*4+i), 32); \
        }                                                                \
    }                                                                \
    check_msacsr_cause(env, GETPC());                                \
    lsx_move_x(pwd, pwx);                                            \
}
LSX_XVFTINT_RM_L_S(h, L, rm)  // helper_lsx_xvftintrmh_l_s
LSX_XVFTINT_RM_L_S(l, R, rm)  // helper_lsx_xvftintrml_l_s
LSX_XVFTINT_RM_L_S(h, L, rp)  // helper_lsx_xvftintrph_l_s
LSX_XVFTINT_RM_L_S(l, R, rp)  // helper_lsx_xvftintrpl_l_s
LSX_XVFTINT_RM_L_S(h, L, rz)  // helper_lsx_xvftintrzh_l_s
LSX_XVFTINT_RM_L_S(l, R, rz)  // helper_lsx_xvftintrzl_l_s
LSX_XVFTINT_RM_L_S(h, L, rne) // helper_lsx_xvftintrneh_l_s
LSX_XVFTINT_RM_L_S(l, R, rne) // helper_lsx_xvftintrnel_l_s
#undef LSX_XVFTINT_RM_L_S


static inline int64_t lsx_vssub_u_s_u_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    uint64_t u_arg2 = UNSIGNED(arg2, df);;
    if (arg1 < 0) {
        return 0;
    } else {
        uint64_t u_arg1 = (uint64_t)arg1;
        return (u_arg1 > u_arg2) ?
            (int64_t)(u_arg1 - u_arg2) :
            0;
    }
}

void helper_lsx_vssub_bu_b_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->b[0]  =  lsx_vssub_u_s_u_df(DF_BYTE, pws->b[0],  pwt->b[0]);
    pwd->b[1]  =  lsx_vssub_u_s_u_df(DF_BYTE, pws->b[1],  pwt->b[1]);
    pwd->b[2]  =  lsx_vssub_u_s_u_df(DF_BYTE, pws->b[2],  pwt->b[2]);
    pwd->b[3]  =  lsx_vssub_u_s_u_df(DF_BYTE, pws->b[3],  pwt->b[3]);
    pwd->b[4]  =  lsx_vssub_u_s_u_df(DF_BYTE, pws->b[4],  pwt->b[4]);
    pwd->b[5]  =  lsx_vssub_u_s_u_df(DF_BYTE, pws->b[5],  pwt->b[5]);
    pwd->b[6]  =  lsx_vssub_u_s_u_df(DF_BYTE, pws->b[6],  pwt->b[6]);
    pwd->b[7]  =  lsx_vssub_u_s_u_df(DF_BYTE, pws->b[7],  pwt->b[7]);
    pwd->b[8]  =  lsx_vssub_u_s_u_df(DF_BYTE, pws->b[8],  pwt->b[8]);
    pwd->b[9]  =  lsx_vssub_u_s_u_df(DF_BYTE, pws->b[9],  pwt->b[9]);
    pwd->b[10] =  lsx_vssub_u_s_u_df(DF_BYTE, pws->b[10], pwt->b[10]);
    pwd->b[11] =  lsx_vssub_u_s_u_df(DF_BYTE, pws->b[11], pwt->b[11]);
    pwd->b[12] =  lsx_vssub_u_s_u_df(DF_BYTE, pws->b[12], pwt->b[12]);
    pwd->b[13] =  lsx_vssub_u_s_u_df(DF_BYTE, pws->b[13], pwt->b[13]);
    pwd->b[14] =  lsx_vssub_u_s_u_df(DF_BYTE, pws->b[14], pwt->b[14]);
    pwd->b[15] =  lsx_vssub_u_s_u_df(DF_BYTE, pws->b[15], pwt->b[15]);
}

void helper_lsx_vssub_hu_h_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->h[0] =  lsx_vssub_u_s_u_df(DF_HALF, pws->h[0], pwt->h[0]);
    pwd->h[1] =  lsx_vssub_u_s_u_df(DF_HALF, pws->h[1], pwt->h[1]);
    pwd->h[2] =  lsx_vssub_u_s_u_df(DF_HALF, pws->h[2], pwt->h[2]);
    pwd->h[3] =  lsx_vssub_u_s_u_df(DF_HALF, pws->h[3], pwt->h[3]);
    pwd->h[4] =  lsx_vssub_u_s_u_df(DF_HALF, pws->h[4], pwt->h[4]);
    pwd->h[5] =  lsx_vssub_u_s_u_df(DF_HALF, pws->h[5], pwt->h[5]);
    pwd->h[6] =  lsx_vssub_u_s_u_df(DF_HALF, pws->h[6], pwt->h[6]);
    pwd->h[7] =  lsx_vssub_u_s_u_df(DF_HALF, pws->h[7], pwt->h[7]);
}

void helper_lsx_vssub_wu_w_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->w[0] =  lsx_vssub_u_s_u_df(DF_WORD, pws->w[0], pwt->w[0]);
    pwd->w[1] =  lsx_vssub_u_s_u_df(DF_WORD, pws->w[1], pwt->w[1]);
    pwd->w[2] =  lsx_vssub_u_s_u_df(DF_WORD, pws->w[2], pwt->w[2]);
    pwd->w[3] =  lsx_vssub_u_s_u_df(DF_WORD, pws->w[3], pwt->w[3]);
}

void helper_lsx_vssub_du_d_du(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->d[0] =  lsx_vssub_u_s_u_df(DF_DOUBLE, pws->d[0], pwt->d[0]);
    pwd->d[1] =  lsx_vssub_u_s_u_df(DF_DOUBLE, pws->d[1], pwt->d[1]);
}

void helper_lsx_vaddw_h_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/16; i++) {
        tmp.h[i] = pws->h[i] + (int16_t)pwt->b[i];
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vaddw_w_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);
    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/32; i++) {
        tmp.w[i] = pws->w[i] + (int32_t)pwt->h[i];
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vaddw_d_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);
    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/64; i++) {
        tmp.d[i] = pws->d[i] + (int64_t)pwt->w[i];
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvaddw_h_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/16; i++) {
        tmp.h[i] = pws->h[i] + (int16_t)pwt->b[i];
        tmp.h[i+8] = pws->h[i+8] + (int16_t)pwt->b[i+16];
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvaddw_w_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);
    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/32; i++) {
        tmp.w[i] = pws->w[i] + (int32_t)pwt->h[i];
        tmp.w[i+4] = pws->w[i+4] + (int32_t)pwt->h[i+8];
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvaddw_d_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);
    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/64; i++) {
        tmp.d[i] = pws->d[i] + (int64_t)pwt->w[i];
        tmp.d[i+2] = pws->d[i+2] + (int64_t)pwt->w[i+4];
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

#define LSX_SIGNED_EVEN(a, df) \
        ((((__int128)(a)) << (128 - DF_BITS(df) / 2)) >> (128 - DF_BITS(df) / 2))

#define LSX_UNSIGNED_EVEN(a, df) \
        ((((unsigned __int128)(a)) << (128 - DF_BITS(df) / 2)) >> (128 - DF_BITS(df) / 2))

#define LSX_SIGNED_ODD(a, df) \
        ((((__int128)(a)) << (128 - DF_BITS(df))) >> (128 - DF_BITS(df) / 2))

#define LSX_UNSIGNED_ODD(a, df) \
        ((((unsigned __int128)(a)) << (128 - DF_BITS(df))) >> (128 - DF_BITS(df) / 2))



static inline __int128 lsx_vhaddw_q_d_df(__int128 arg1, __int128 arg2)
{
    return LSX_SIGNED_ODD(arg1, DF_QUAD) + LSX_SIGNED_EVEN(arg2, DF_QUAD);
}

void helper_lsx_vhaddw_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{

    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0]  = lsx_vhaddw_q_d_df(pws->q[0],  pwt->q[0]);
}

static inline __int128 lsx_vhsubw_q_d_df(__int128 arg1, __int128 arg2)
{
    return LSX_SIGNED_ODD(arg1, DF_QUAD) - LSX_SIGNED_EVEN(arg2, DF_QUAD);
}

void helper_lsx_vhsubw_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{

    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0]  = lsx_vhsubw_q_d_df(pws->q[0],  pwt->q[0]);
}

static inline __int128 lsx_vhaddw_qu_du_df(__int128 arg1, __int128 arg2)
{
    return LSX_UNSIGNED_ODD(arg1, DF_QUAD) + LSX_UNSIGNED_EVEN(arg2, DF_QUAD);
}

void helper_lsx_vhaddw_qu_du(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{

    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0]  = lsx_vhaddw_qu_du_df(pws->q[0],  pwt->q[0]);
}

static inline __int128 lsx_vhsubw_qu_du_df(__int128 arg1, __int128 arg2)
{
    return LSX_UNSIGNED_ODD(arg1, DF_QUAD) - LSX_UNSIGNED_EVEN(arg2, DF_QUAD);
}

void helper_lsx_vhsubw_qu_du(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{

    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0]  = lsx_vhsubw_qu_du_df(pws->q[0],  pwt->q[0]);
}

#define LSX_SIGNED_EXTRACT(e, o, a, df)     \
    do {                                \
        e = LSX_SIGNED_EVEN(a, df);         \
        o = LSX_SIGNED_ODD(a, df);          \
    } while (0)

#define LSX_UNSIGNED_EXTRACT(e, o, a, df)   \
    do {                                \
        e = LSX_UNSIGNED_EVEN(a, df);       \
        o = LSX_UNSIGNED_ODD(a, df);        \
    } while (0)

static inline __int128 lsx_vdp2_q_d_df(__int128 arg1, __int128 arg2)
{
    __int128 even_arg1;
    __int128 even_arg2;
    __int128 odd_arg1;
    __int128 odd_arg2;
    LSX_SIGNED_EXTRACT(even_arg1, odd_arg1, arg1, DF_QUAD);
    LSX_SIGNED_EXTRACT(even_arg2, odd_arg2, arg2, DF_QUAD);
    return (even_arg1 * even_arg2) + (odd_arg1 * odd_arg2);
}

void helper_lsx_vdp2_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{

    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0]  = lsx_vdp2_q_d_df(pws->q[0],  pwt->q[0]);
}

static inline __int128 lsx_vdp2_qu_du_df(__int128 arg1, __int128 arg2)
{
    __int128 even_arg1;
    __int128 even_arg2;
    __int128 odd_arg1;
    __int128 odd_arg2;
    LSX_UNSIGNED_EXTRACT(even_arg1, odd_arg1, arg1, DF_QUAD);
    LSX_UNSIGNED_EXTRACT(even_arg2, odd_arg2, arg2, DF_QUAD);
    return (even_arg1 * even_arg2) + (odd_arg1 * odd_arg2);
}

void helper_lsx_vdp2_qu_du(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{

    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0]  = lsx_vdp2_qu_du_df(pws->q[0],  pwt->q[0]);
}

void helper_lsx_vaddw_h_h_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/16; i++) {
        tmp.h[i] = pws->h[i] + (uint16_t)((uint8_t)pwt->b[i]);
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vaddw_w_w_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);
    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/32; i++) {
        tmp.w[i] = pws->w[i] + (uint32_t)((uint16_t)pwt->h[i]);
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vaddw_d_d_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);
    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/64; i++) {
        tmp.d[i] = pws->d[i] + (uint64_t)((uint32_t)pwt->w[i]);
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvaddw_h_h_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/16; i++) {
        tmp.h[i] = pws->h[i] + (uint16_t)((uint8_t)pwt->b[i]);
        tmp.h[i+8] = pws->h[i+8] + (uint16_t)((uint8_t)pwt->b[i+16]);
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvaddw_w_w_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);
    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/32; i++) {
        tmp.w[i] = pws->w[i] + (uint32_t)((uint16_t)pwt->h[i]);
        tmp.w[i+4] = pws->w[i+4] + (uint32_t)((uint16_t)pwt->h[i+8]);
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvaddw_d_d_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);
    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/64; i++) {
        tmp.d[i] = pws->d[i] + (uint64_t)((uint32_t)pwt->w[i]);
        tmp.d[i+2] = pws->d[i+2] + (uint64_t)((uint32_t)pwt->w[i+4]);
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

static inline __int128 lsx_vdp2_s_u_s_df(uint32_t df, __int128 arg1, __int128 arg2)
{
    __int128 even_arg1;
    __int128 even_arg2;
    __int128 odd_arg1;
    __int128 odd_arg2;
    LSX_UNSIGNED_EXTRACT(even_arg1, odd_arg1, arg1, df);
    LSX_SIGNED_EXTRACT(even_arg2, odd_arg2, arg2, df);
    return (even_arg1 * even_arg2) + (odd_arg1 * odd_arg2);
}

void helper_lsx_vdp2_h_bu_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->h[0] =  lsx_vdp2_s_u_s_df(DF_HALF, pws->h[0], pwt->h[0]);
    pwd->h[1] =  lsx_vdp2_s_u_s_df(DF_HALF, pws->h[1], pwt->h[1]);
    pwd->h[2] =  lsx_vdp2_s_u_s_df(DF_HALF, pws->h[2], pwt->h[2]);
    pwd->h[3] =  lsx_vdp2_s_u_s_df(DF_HALF, pws->h[3], pwt->h[3]);
    pwd->h[4] =  lsx_vdp2_s_u_s_df(DF_HALF, pws->h[4], pwt->h[4]);
    pwd->h[5] =  lsx_vdp2_s_u_s_df(DF_HALF, pws->h[5], pwt->h[5]);
    pwd->h[6] =  lsx_vdp2_s_u_s_df(DF_HALF, pws->h[6], pwt->h[6]);
    pwd->h[7] =  lsx_vdp2_s_u_s_df(DF_HALF, pws->h[7], pwt->h[7]);
}

void helper_lsx_vdp2_w_hu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->w[0] =  lsx_vdp2_s_u_s_df(DF_WORD, pws->w[0], pwt->w[0]);
    pwd->w[1] =  lsx_vdp2_s_u_s_df(DF_WORD, pws->w[1], pwt->w[1]);
    pwd->w[2] =  lsx_vdp2_s_u_s_df(DF_WORD, pws->w[2], pwt->w[2]);
    pwd->w[3] =  lsx_vdp2_s_u_s_df(DF_WORD, pws->w[3], pwt->w[3]);
}

void helper_lsx_vdp2_d_wu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->d[0] =  lsx_vdp2_s_u_s_df(DF_DOUBLE, pws->d[0], pwt->d[0]);
    pwd->d[1] =  lsx_vdp2_s_u_s_df(DF_DOUBLE, pws->d[1], pwt->d[1]);
}

void helper_lsx_vdp2_q_du_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{

    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0]  = lsx_vdp2_s_u_s_df(DF_QUAD, pws->q[0],  pwt->q[0]);
}

static inline __int128 lsx_vdp2add_q_d_df(__int128 dest, __int128 arg1, __int128 arg2)
{
    __int128 even_arg1;
    __int128 even_arg2;
    __int128 odd_arg1;
    __int128 odd_arg2;
    LSX_SIGNED_EXTRACT(even_arg1, odd_arg1, arg1, DF_QUAD);
    LSX_SIGNED_EXTRACT(even_arg2, odd_arg2, arg2, DF_QUAD);
    return dest + (even_arg1 * even_arg2) + (odd_arg1 * odd_arg2);
}

void helper_lsx_vdp2add_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{

    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0]  = lsx_vdp2add_q_d_df(pwd->q[0], pws->q[0],  pwt->q[0]);
}

static inline __int128 lsx_vdp2add_q_du_df(__int128 dest, __int128 arg1, __int128 arg2)
{
    __int128 even_arg1;
    __int128 even_arg2;
    __int128 odd_arg1;
    __int128 odd_arg2;
    LSX_UNSIGNED_EXTRACT(even_arg1, odd_arg1, arg1, DF_QUAD);
    LSX_UNSIGNED_EXTRACT(even_arg2, odd_arg2, arg2, DF_QUAD);
    return dest + (even_arg1 * even_arg2) + (odd_arg1 * odd_arg2);
}

void helper_lsx_vdp2add_q_du(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{

    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0]  = lsx_vdp2add_q_du_df(pwd->q[0], pws->q[0],  pwt->q[0]);
}

static inline __int128 lsx_vdp2add_s_u_s_df(uint32_t df, __int128 dest, __int128 arg1, __int128 arg2)
{
    __int128 even_arg1;
    __int128 even_arg2;
    __int128 odd_arg1;
    __int128 odd_arg2;
    LSX_UNSIGNED_EXTRACT(even_arg1, odd_arg1, arg1, df);
    LSX_SIGNED_EXTRACT(even_arg2, odd_arg2, arg2, df);
    return dest + (even_arg1 * even_arg2) + (odd_arg1 * odd_arg2);
}

void helper_lsx_vdp2add_h_bu_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->h[0] =  lsx_vdp2add_s_u_s_df(DF_HALF, pwd->h[0], pws->h[0], pwt->h[0]);
    pwd->h[1] =  lsx_vdp2add_s_u_s_df(DF_HALF, pwd->h[1], pws->h[1], pwt->h[1]);
    pwd->h[2] =  lsx_vdp2add_s_u_s_df(DF_HALF, pwd->h[2], pws->h[2], pwt->h[2]);
    pwd->h[3] =  lsx_vdp2add_s_u_s_df(DF_HALF, pwd->h[3], pws->h[3], pwt->h[3]);
    pwd->h[4] =  lsx_vdp2add_s_u_s_df(DF_HALF, pwd->h[4], pws->h[4], pwt->h[4]);
    pwd->h[5] =  lsx_vdp2add_s_u_s_df(DF_HALF, pwd->h[5], pws->h[5], pwt->h[5]);
    pwd->h[6] =  lsx_vdp2add_s_u_s_df(DF_HALF, pwd->h[6], pws->h[6], pwt->h[6]);
    pwd->h[7] =  lsx_vdp2add_s_u_s_df(DF_HALF, pwd->h[7], pws->h[7], pwt->h[7]);
}

void helper_lsx_vdp2add_w_hu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->w[0] =  lsx_vdp2add_s_u_s_df(DF_WORD, pwd->w[0], pws->w[0], pwt->w[0]);
    pwd->w[1] =  lsx_vdp2add_s_u_s_df(DF_WORD, pwd->w[1], pws->w[1], pwt->w[1]);
    pwd->w[2] =  lsx_vdp2add_s_u_s_df(DF_WORD, pwd->w[2], pws->w[2], pwt->w[2]);
    pwd->w[3] =  lsx_vdp2add_s_u_s_df(DF_WORD, pwd->w[3], pws->w[3], pwt->w[3]);
}

void helper_lsx_vdp2add_d_wu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->d[0] =  lsx_vdp2add_s_u_s_df(DF_DOUBLE, pwd->d[0], pws->d[0], pwt->d[0]);
    pwd->d[1] =  lsx_vdp2add_s_u_s_df(DF_DOUBLE, pwd->d[1], pws->d[1], pwt->d[1]);
}

void helper_lsx_vdp2add_q_du_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{

    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0]  = lsx_vdp2add_s_u_s_df(DF_QUAD, pwd->q[0], pws->q[0],  pwt->q[0]);
}

static inline __int128 lsx_vdp2sub_q_d_df(__int128 dest, __int128 arg1, __int128 arg2)
{
    __int128 even_arg1;
    __int128 even_arg2;
    __int128 odd_arg1;
    __int128 odd_arg2;
    LSX_SIGNED_EXTRACT(even_arg1, odd_arg1, arg1, DF_QUAD);
    LSX_SIGNED_EXTRACT(even_arg2, odd_arg2, arg2, DF_QUAD);
    return dest - ((even_arg1 * even_arg2) + (odd_arg1 * odd_arg2));
}

void helper_lsx_vdp2sub_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{

    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0]  = lsx_vdp2sub_q_d_df(pwd->q[0], pws->q[0],  pwt->q[0]);
}

static inline __int128 lsx_vdp2sub_q_du_df(__int128 dest, __int128 arg1, __int128 arg2)
{
    __int128 even_arg1;
    __int128 even_arg2;
    __int128 odd_arg1;
    __int128 odd_arg2;
    LSX_UNSIGNED_EXTRACT(even_arg1, odd_arg1, arg1, DF_QUAD);
    LSX_UNSIGNED_EXTRACT(even_arg2, odd_arg2, arg2, DF_QUAD);
    return dest - ((even_arg1 * even_arg2) + (odd_arg1 * odd_arg2));
}

void helper_lsx_vdp2sub_q_du(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{

    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0]  = lsx_vdp2sub_q_du_df(pwd->q[0], pws->q[0],  pwt->q[0]);
}

#define BIT_POSITION(x, df) ((uint64_t)(x) % DF_BITS(df))
#define ROTATE_RIGHT(x, n, df) (((uint64_t)(x) >> (n)) | ((uint64_t)(x) << (DF_BITS(df) - (n))))

static inline int64_t lsx_vrotr_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    int32_t b_arg2 = BIT_POSITION(arg2, df);
    return UNSIGNED(ROTATE_RIGHT(u_arg1, b_arg2, df), df);
}

void helper_lsx_vrotr_b(CPULoongArchState *env,
                      uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->b[0]  = lsx_vrotr_df(DF_BYTE, pws->b[0],  pwt->b[0]);
    pwd->b[1]  = lsx_vrotr_df(DF_BYTE, pws->b[1],  pwt->b[1]);
    pwd->b[2]  = lsx_vrotr_df(DF_BYTE, pws->b[2],  pwt->b[2]);
    pwd->b[3]  = lsx_vrotr_df(DF_BYTE, pws->b[3],  pwt->b[3]);
    pwd->b[4]  = lsx_vrotr_df(DF_BYTE, pws->b[4],  pwt->b[4]);
    pwd->b[5]  = lsx_vrotr_df(DF_BYTE, pws->b[5],  pwt->b[5]);
    pwd->b[6]  = lsx_vrotr_df(DF_BYTE, pws->b[6],  pwt->b[6]);
    pwd->b[7]  = lsx_vrotr_df(DF_BYTE, pws->b[7],  pwt->b[7]);
    pwd->b[8]  = lsx_vrotr_df(DF_BYTE, pws->b[8],  pwt->b[8]);
    pwd->b[9]  = lsx_vrotr_df(DF_BYTE, pws->b[9],  pwt->b[9]);
    pwd->b[10] = lsx_vrotr_df(DF_BYTE, pws->b[10], pwt->b[10]);
    pwd->b[11] = lsx_vrotr_df(DF_BYTE, pws->b[11], pwt->b[11]);
    pwd->b[12] = lsx_vrotr_df(DF_BYTE, pws->b[12], pwt->b[12]);
    pwd->b[13] = lsx_vrotr_df(DF_BYTE, pws->b[13], pwt->b[13]);
    pwd->b[14] = lsx_vrotr_df(DF_BYTE, pws->b[14], pwt->b[14]);
    pwd->b[15] = lsx_vrotr_df(DF_BYTE, pws->b[15], pwt->b[15]);
}

void helper_lsx_vrotr_h(CPULoongArchState *env,
                      uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->h[0]  = lsx_vrotr_df(DF_HALF, pws->h[0],  pwt->h[0]);
    pwd->h[1]  = lsx_vrotr_df(DF_HALF, pws->h[1],  pwt->h[1]);
    pwd->h[2]  = lsx_vrotr_df(DF_HALF, pws->h[2],  pwt->h[2]);
    pwd->h[3]  = lsx_vrotr_df(DF_HALF, pws->h[3],  pwt->h[3]);
    pwd->h[4]  = lsx_vrotr_df(DF_HALF, pws->h[4],  pwt->h[4]);
    pwd->h[5]  = lsx_vrotr_df(DF_HALF, pws->h[5],  pwt->h[5]);
    pwd->h[6]  = lsx_vrotr_df(DF_HALF, pws->h[6],  pwt->h[6]);
    pwd->h[7]  = lsx_vrotr_df(DF_HALF, pws->h[7],  pwt->h[7]);
}

void helper_lsx_vrotr_w(CPULoongArchState *env,
                      uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);
    pwd->w[0]  = lsx_vrotr_df(DF_WORD, pws->w[0],  pwt->w[0]);
    pwd->w[1]  = lsx_vrotr_df(DF_WORD, pws->w[1],  pwt->w[1]);
    pwd->w[2]  = lsx_vrotr_df(DF_WORD, pws->w[2],  pwt->w[2]);
    pwd->w[3]  = lsx_vrotr_df(DF_WORD, pws->w[3],  pwt->w[3]);
}

void helper_lsx_vrotr_d(CPULoongArchState *env,
                      uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->d[0]  = lsx_vrotr_df(DF_DOUBLE, pws->d[0],  pwt->d[0]);
    pwd->d[1]  = lsx_vrotr_df(DF_DOUBLE, pws->d[1],  pwt->d[1]);
}

static inline void lsx_vreplve_df(uint32_t df, wr_t *pwd,
                                wr_t *pws, target_ulong rt)
{
    uint32_t n = rt % DF_V_ELEMENTS(df);
    uint32_t i;

    switch (df) {
    case DF_BYTE:
        for (i = 0; i < DF_V_ELEMENTS(DF_BYTE); i++) {
            pwd->b[i] = pws->b[n];
        }
        break;
    case DF_HALF:
        for (i = 0; i < DF_V_ELEMENTS(DF_HALF); i++) {
            pwd->h[i] = pws->h[n];
        }
        break;
    case DF_WORD:
        for (i = 0; i < DF_V_ELEMENTS(DF_WORD); i++) {
            pwd->w[i] = pws->w[n];
        }
        break;
    case DF_DOUBLE:
        for (i = 0; i < DF_V_ELEMENTS(DF_DOUBLE); i++) {
            pwd->d[i] = pws->d[n];
        }
       break;
    default:
        assert(0);
    }
}

void helper_lsx_vreplve_df(CPULoongArchState *env, uint32_t df, uint32_t wd,
                         uint32_t ws, uint32_t rt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    lsx_vreplve_df(df, pwd, pws, env->gpr[rt]);
}

#define CONCATENATE_AND_SLIDE(s, k)             \
    do {                                        \
        for (i = 0; i < s; i++) {               \
            v[i]     = pws->b[s * k + i];       \
            v[i + s] = pwd->b[s * k + i];       \
        }                                       \
        for (i = 0; i < s; i++) {               \
            pwd->b[s * k + i] = v[i + n];       \
        }                                       \
    } while (0)

static inline void lsx_vextrcol_df(uint32_t df, wr_t *pwd,
                              wr_t *pws, target_ulong rt)
{
    uint32_t n = rt % DF_V_ELEMENTS(df);
    uint8_t v[64];
    uint32_t i, k;

    switch (df) {
    case DF_BYTE:
        CONCATENATE_AND_SLIDE(DF_V_ELEMENTS(DF_BYTE), 0);
        break;
    case DF_HALF:
        for (k = 0; k < 2; k++) {
            CONCATENATE_AND_SLIDE(DF_V_ELEMENTS(DF_HALF), k);
        }
        break;
    case DF_WORD:
        for (k = 0; k < 4; k++) {
            CONCATENATE_AND_SLIDE(DF_V_ELEMENTS(DF_WORD), k);
        }
        break;
    case DF_DOUBLE:
        for (k = 0; k < 8; k++) {
            CONCATENATE_AND_SLIDE(DF_V_ELEMENTS(DF_DOUBLE), k);
        }
        break;
    default:
        assert(0);
    }
}

void helper_lsx_vextrcol_df(CPULoongArchState *env, uint32_t df, uint32_t wd,
                       uint32_t ws, uint32_t rt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    lsx_vextrcol_df(df, pwd, pws, env->gpr[rt]);
}

void helper_lsx_vandn_v(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->d[0] = ~pws->d[0] & pwt->d[0];
    pwd->d[1] = ~pws->d[1] & pwt->d[1];
}

void helper_lsx_vorn_v(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->d[0] = pws->d[0] | ~pwt->d[0];
    pwd->d[1] = pws->d[1] | ~pwt->d[1];
}

static inline int64_t lsx_xvseq_s_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    return arg1 == arg2 ? -1 : 0;
}

static inline int64_t lsx_xvsle_s_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    return arg1 <= arg2 ? -1 : 0;
}

static inline int64_t lsx_xvsle_u_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    uint64_t u_arg2 = UNSIGNED(arg2, df);
    return u_arg1 <= u_arg2 ? -1 : 0;
}

static inline int64_t lsx_xvslt_s_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    return arg1 < arg2 ? -1 : 0;
}

static inline int64_t lsx_xvslt_u_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    uint64_t u_arg2 = UNSIGNED(arg2, df);
    return u_arg1 < u_arg2 ? -1 : 0;
}

static inline int64_t lsx_xvadd_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    return arg1 + arg2;
}

static inline int64_t lsx_xvsub_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    return arg1 - arg2;
}

static inline int64_t lsx_xvsadd_s_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    int64_t max_int = DF_MAX_INT(df);
    int64_t min_int = DF_MIN_INT(df);
    if (arg1 < 0) {
        return (min_int - arg1 < arg2) ? arg1 + arg2 : min_int;
    } else {
        return (arg2 < max_int - arg1) ? arg1 + arg2 : max_int;
    }
}

static inline int64_t lsx_xvssub_s_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    int64_t max_int = DF_MAX_INT(df);
    int64_t min_int = DF_MIN_INT(df);
    if (arg2 > 0) {
        return (min_int + arg2 < arg1) ? arg1 - arg2 : min_int;
    } else {
        return (arg1 < max_int + arg2) ? arg1 - arg2 : max_int;
    }
}

static inline uint64_t lsx_xvsadd_u_df(uint32_t df, uint64_t arg1, uint64_t arg2)
{
    uint64_t max_uint = DF_MAX_UINT(df);
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    uint64_t u_arg2 = UNSIGNED(arg2, df);
    return (u_arg1 < max_uint - u_arg2) ? u_arg1 + u_arg2 : max_uint;
}

static inline int64_t lsx_xvssub_u_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    uint64_t u_arg2 = UNSIGNED(arg2, df);
    return (u_arg1 > u_arg2) ? u_arg1 - u_arg2 : 0;
}

static inline int64_t lsx_xvssub_u_u_s_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    uint64_t max_uint = DF_MAX_UINT(df);
    if (arg2 >= 0) {
        uint64_t u_arg2 = (uint64_t)arg2;
        return (u_arg1 > u_arg2) ?
            (int64_t)(u_arg1 - u_arg2) :
            0;
    } else {
        uint64_t u_arg2 = (uint64_t)(-arg2);
        return (u_arg1 < max_uint - u_arg2) ?
            (int64_t)(u_arg1 + u_arg2) :
            (int64_t)max_uint;
    }
}

static inline int64_t lsx_xvssub_u_s_u_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    uint64_t u_arg2 = UNSIGNED(arg2, df);;
    if (arg1 < 0) {
        return 0;
    } else {
        uint64_t u_arg1 = (uint64_t)arg1;
        return (u_arg1 > u_arg2) ?
            (int64_t)(u_arg1 - u_arg2) :
            0;
    }
}

static inline int64_t lsx_xvssub_s_u_u_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    uint64_t u_arg2 = UNSIGNED(arg2, df);
    int64_t max_int = DF_MAX_INT(df);
    int64_t min_int = DF_MIN_INT(df);
    if (u_arg1 > u_arg2) {
        return u_arg1 - u_arg2 < (uint64_t)max_int ?
            (int64_t)(u_arg1 - u_arg2) :
            max_int;
    } else {
        return u_arg2 - u_arg1 < (uint64_t)(-min_int) ?
            (int64_t)(u_arg1 - u_arg2) :
            min_int;
    }
}


static inline __int128 lsx_xvhaddw_s_s_df(uint32_t df, __int128 arg1, __int128 arg2)
{
    return LSX_SIGNED_ODD(arg1, df) + LSX_SIGNED_EVEN(arg2, df);
}

static inline __int128 lsx_xvhsubw_s_s_df(uint32_t df, __int128 arg1, __int128 arg2)
{
    return LSX_SIGNED_ODD(arg1, df) - LSX_SIGNED_EVEN(arg2, df);
}

static inline __int128 lsx_xvhaddw_u_u_df(uint32_t df, __int128 arg1, __int128 arg2)
{
    return LSX_UNSIGNED_ODD(arg1, df) + LSX_UNSIGNED_EVEN(arg2, df);
}

static inline __int128 lsx_xvhsubw_u_u_df(uint32_t df, __int128 arg1, __int128 arg2)
{
    return LSX_UNSIGNED_ODD(arg1, df) - LSX_UNSIGNED_EVEN(arg2, df);
}


static inline int64_t lsx_xvadda_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    uint64_t abs_arg1 = arg1 >= 0 ? arg1 : -arg1;
    uint64_t abs_arg2 = arg2 >= 0 ? arg2 : -arg2;
    return abs_arg1 + abs_arg2;
}

static inline int64_t lsx_xvsadda_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    uint64_t max_int = (uint64_t)DF_MAX_INT(df);
    uint64_t abs_arg1 = arg1 >= 0 ? arg1 : -arg1;
    uint64_t abs_arg2 = arg2 >= 0 ? arg2 : -arg2;
    if (abs_arg1 > max_int || abs_arg2 > max_int) {
        return (int64_t)max_int;
    } else {
        return (abs_arg1 < max_int - abs_arg2) ? abs_arg1 + abs_arg2 : max_int;
    }
}

static inline int64_t lsx_xvabsd_s_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    /* signed compare */
    return (arg1 < arg2) ?
        (uint64_t)(arg2 - arg1) : (uint64_t)(arg1 - arg2);
}

static inline uint64_t lsx_xvabsd_u_df(uint32_t df, uint64_t arg1, uint64_t arg2)
{
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    uint64_t u_arg2 = UNSIGNED(arg2, df);
    /* unsigned compare */
    return (u_arg1 < u_arg2) ?
        (uint64_t)(u_arg2 - u_arg1) : (uint64_t)(u_arg1 - u_arg2);
}

static inline int64_t lsx_xvavg_s_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    /* signed shift */
    return (arg1 >> 1) + (arg2 >> 1) + (arg1 & arg2 & 1);
}

static inline uint64_t lsx_xvavg_u_df(uint32_t df, uint64_t arg1, uint64_t arg2)
{
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    uint64_t u_arg2 = UNSIGNED(arg2, df);
    /* unsigned shift */
    return (u_arg1 >> 1) + (u_arg2 >> 1) + (u_arg1 & u_arg2 & 1);
}

static inline int64_t lsx_xvavgr_s_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    /* signed shift */
    return (arg1 >> 1) + (arg2 >> 1) + ((arg1 | arg2) & 1);
}

static inline uint64_t lsx_xvavgr_u_df(uint32_t df, uint64_t arg1, uint64_t arg2)
{
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    uint64_t u_arg2 = UNSIGNED(arg2, df);
    /* unsigned shift */
    return (u_arg1 >> 1) + (u_arg2 >> 1) + ((u_arg1 | u_arg2) & 1);
}

static inline int64_t lsx_xvmax_s_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    return arg1 > arg2 ? arg1 : arg2;
}

static inline int64_t lsx_xvmin_s_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    return arg1 < arg2 ? arg1 : arg2;
}

static inline int64_t lsx_xvmax_u_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    uint64_t u_arg2 = UNSIGNED(arg2, df);
    return u_arg1 > u_arg2 ? arg1 : arg2;
}

static inline int64_t lsx_xvmin_u_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    uint64_t u_arg2 = UNSIGNED(arg2, df);
    return u_arg1 < u_arg2 ? arg1 : arg2;
}

static inline int64_t lsx_xvmaxa_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    uint64_t abs_arg1 = arg1 >= 0 ? arg1 : -arg1;
    uint64_t abs_arg2 = arg2 >= 0 ? arg2 : -arg2;
    return abs_arg1 > abs_arg2 ? arg1 : arg2;
}

static inline int64_t lsx_xvmina_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    uint64_t abs_arg1 = arg1 >= 0 ? arg1 : -arg1;
    uint64_t abs_arg2 = arg2 >= 0 ? arg2 : -arg2;
    return abs_arg1 < abs_arg2 ? arg1 : arg2;
}

static inline int64_t lsx_xvmul_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    return arg1 * arg2;
}


static inline __int128 lsx_xvdp2_s_s_df(uint32_t df, __int128 arg1, __int128 arg2)
{
    __int128 even_arg1;
    __int128 even_arg2;
    __int128 odd_arg1;
    __int128 odd_arg2;
    LSX_SIGNED_EXTRACT(even_arg1, odd_arg1, arg1, df);
    LSX_SIGNED_EXTRACT(even_arg2, odd_arg2, arg2, df);
    return (even_arg1 * even_arg2) + (odd_arg1 * odd_arg2);
}

static inline __int128 lsx_xvdp2_u_u_df(uint32_t df, __int128 arg1, __int128 arg2)
{
    __int128 even_arg1;
    __int128 even_arg2;
    __int128 odd_arg1;
    __int128 odd_arg2;
    LSX_UNSIGNED_EXTRACT(even_arg1, odd_arg1, arg1, df);
    LSX_UNSIGNED_EXTRACT(even_arg2, odd_arg2, arg2, df);
    return (even_arg1 * even_arg2) + (odd_arg1 * odd_arg2);
}

static inline __int128 lsx_xvdp2_s_u_s_df(uint32_t df, __int128 arg1, __int128 arg2)
{
    __int128 even_arg1;
    __int128 even_arg2;
    __int128 odd_arg1;
    __int128 odd_arg2;
    LSX_UNSIGNED_EXTRACT(even_arg1, odd_arg1, arg1, df);
    LSX_SIGNED_EXTRACT(even_arg2, odd_arg2, arg2, df);
    return (even_arg1 * even_arg2) + (odd_arg1 * odd_arg2);
}

static inline int64_t lsx_xvdiv_s_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    if (arg1 == DF_MIN_INT(df) && arg2 == -1) {
        return DF_MIN_INT(df);
    }
    return arg2 ? arg1 / arg2
                : arg1 >= 0 ? -1 : 1;
}

static inline int64_t lsx_xvmod_s_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    if (arg1 == DF_MIN_INT(df) && arg2 == -1) {
        return 0;
    }
    return arg2 ? arg1 % arg2 : arg1;
}

static inline int64_t lsx_xvdiv_u_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    uint64_t u_arg2 = UNSIGNED(arg2, df);
    return arg2 ? u_arg1 / u_arg2 : -1;
}

static inline int64_t lsx_xvmod_u_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    uint64_t u_arg2 = UNSIGNED(arg2, df);
    return u_arg2 ? u_arg1 % u_arg2 : u_arg1;
}


static inline int64_t lsx_xvsll_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    int32_t b_arg2 = BIT_POSITION(arg2, df);
    return arg1 << b_arg2;
}

static inline int64_t lsx_xvsrl_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    int32_t b_arg2 = BIT_POSITION(arg2, df);
    return u_arg1 >> b_arg2;
}

static inline int64_t lsx_xvsra_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    int32_t b_arg2 = BIT_POSITION(arg2, df);
    return arg1 >> b_arg2;
}


static inline int64_t lsx_xvrotr_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    int32_t b_arg2 = BIT_POSITION(arg2, df);
    return UNSIGNED(ROTATE_RIGHT(u_arg1, b_arg2, df), df);
}

static inline int64_t lsx_xvsrlr_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    int32_t b_arg2 = BIT_POSITION(arg2, df);
    if (b_arg2 == 0) {
        return u_arg1;
    } else {
        uint64_t r_bit = (u_arg1 >> (b_arg2 - 1)) & 1;
        return (u_arg1 >> b_arg2) + r_bit;
    }
}

static inline int64_t lsx_xvsrar_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    int32_t b_arg2 = BIT_POSITION(arg2, df);
    if (b_arg2 == 0) {
        return arg1;
    } else {
        int64_t r_bit = (arg1 >> (b_arg2 - 1)) & 1;
        return (arg1 >> b_arg2) + r_bit;
    }
}

static inline int64_t lsx_xvbitclr_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    int32_t b_arg2 = BIT_POSITION(arg2, df);
    return UNSIGNED(arg1 & (~(1LL << b_arg2)), df);
}

static inline int64_t lsx_xvbitset_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    int32_t b_arg2 = BIT_POSITION(arg2, df);
    return UNSIGNED(arg1 | (1LL << b_arg2), df);
}

static inline int64_t lsx_xvbitrev_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    int32_t b_arg2 = BIT_POSITION(arg2, df);
    return UNSIGNED(arg1 ^ (1LL << b_arg2), df);
}

#define LSX_BINXOP_DF(func) \
void helper_lsx_ ## func ## _df(CPULoongArchState *env, uint32_t df,         \
                                uint32_t wd, uint32_t ws, uint32_t wt)  \
{                                                                       \
    wr_t *pwd = &(env->fpr[wd].wr);                          \
    wr_t *pws = &(env->fpr[ws].wr);                          \
    wr_t *pwt = &(env->fpr[wt].wr);                          \
    uint32_t i;                                                         \
                                                                        \
    switch (df) {                                                       \
    case DF_BYTE:                                                       \
        for (i = 0; i < (256/DF_BITS(df)); i++) {                       \
            pwd->b[i] = lsx_ ## func ## _df(df, pws->b[i], pwt->b[i]);  \
        }                                                               \
        break;                                                          \
    case DF_HALF:                                                       \
        for (i = 0; i < (256/DF_BITS(df)); i++) {                       \
            pwd->h[i] = lsx_ ## func ## _df(df, pws->h[i], pwt->h[i]);  \
        }                                                               \
        break;                                                          \
    case DF_WORD:                                                       \
        for (i = 0; i < (256/DF_BITS(df)); i++) {                       \
            pwd->w[i] = lsx_ ## func ## _df(df, pws->w[i], pwt->w[i]);  \
        }                                                               \
        break;                                                          \
    case DF_DOUBLE:                                                     \
        for (i = 0; i < (256/DF_BITS(df)); i++) {                       \
            pwd->d[i] = lsx_ ## func ## _df(df, pws->d[i], pwt->d[i]);  \
        }                                                               \
        break;                                                          \
    case DF_QUAD:                                                       \
        for (i = 0; i < (256/DF_BITS(df)); i++) {                       \
            pwd->q[i] = lsx_ ## func ## _df(df, pws->q[i], pwt->q[i]);  \
        }                                                               \
        break;                                                          \
    default:                                                            \
        assert(0);                                                      \
    }                                                                   \
}

LSX_BINXOP_DF(xvseq_s)
LSX_BINXOP_DF(xvsle_s)
LSX_BINXOP_DF(xvsle_u)
LSX_BINXOP_DF(xvslt_s)
LSX_BINXOP_DF(xvslt_u)
LSX_BINXOP_DF(xvadd)
LSX_BINXOP_DF(xvsub)
LSX_BINXOP_DF(xvsadd_s)
LSX_BINXOP_DF(xvssub_s)
LSX_BINXOP_DF(xvsadd_u)
LSX_BINXOP_DF(xvssub_u)
LSX_BINXOP_DF(xvssub_u_u_s)
LSX_BINXOP_DF(xvssub_u_s_u)
LSX_BINXOP_DF(xvssub_s_u_u)
LSX_BINXOP_DF(xvhaddw_s_s)
LSX_BINXOP_DF(xvhsubw_s_s)
LSX_BINXOP_DF(xvhaddw_u_u)
LSX_BINXOP_DF(xvhsubw_u_u)
LSX_BINXOP_DF(xvadda)
LSX_BINXOP_DF(xvsadda)
LSX_BINXOP_DF(xvabsd_s)
LSX_BINXOP_DF(xvabsd_u)
LSX_BINXOP_DF(xvavg_s)
LSX_BINXOP_DF(xvavg_u)
LSX_BINXOP_DF(xvavgr_s)
LSX_BINXOP_DF(xvavgr_u)
LSX_BINXOP_DF(xvmax_s)
LSX_BINXOP_DF(xvmin_s)
LSX_BINXOP_DF(xvmax_u)
LSX_BINXOP_DF(xvmin_u)
LSX_BINXOP_DF(xvmaxa)
LSX_BINXOP_DF(xvmina)
LSX_BINXOP_DF(xvmul)

LSX_BINXOP_DF(xvdp2_s_s)
LSX_BINXOP_DF(xvdp2_u_u)
LSX_BINXOP_DF(xvdp2_s_u_s)

LSX_BINXOP_DF(xvdiv_s)
LSX_BINXOP_DF(xvmod_s)
LSX_BINXOP_DF(xvdiv_u)
LSX_BINXOP_DF(xvmod_u)
LSX_BINXOP_DF(xvsll)
LSX_BINXOP_DF(xvsrl)
LSX_BINXOP_DF(xvsra)
LSX_BINXOP_DF(xvrotr)
LSX_BINXOP_DF(xvsrlr)
LSX_BINXOP_DF(xvsrar)

LSX_BINXOP_DF(xvbitclr)
LSX_BINXOP_DF(xvbitset)
LSX_BINXOP_DF(xvbitrev)
#undef LSX_BINXOP_DF


static inline int64_t lsx_xvmadd_df(uint32_t df, int64_t dest, int64_t arg1,
                                   int64_t arg2)
{
    return dest + arg1 * arg2;
}

static inline int64_t lsx_xvmsub_df(uint32_t df, int64_t dest, int64_t arg1,
                                   int64_t arg2)
{
    return dest - arg1 * arg2;
}


static inline __int128 lsx_xvdp2add_s_s_df(uint32_t df, __int128 dest, __int128 arg1, __int128 arg2)
{
    __int128 even_arg1;
    __int128 even_arg2;
    __int128 odd_arg1;
    __int128 odd_arg2;
    LSX_SIGNED_EXTRACT(even_arg1, odd_arg1, arg1, df);
    LSX_SIGNED_EXTRACT(even_arg2, odd_arg2, arg2, df);
    return dest + (even_arg1 * even_arg2) + (odd_arg1 * odd_arg2);
}

static inline __int128 lsx_xvdp2add_s_u_df(uint32_t df, __int128 dest, __int128 arg1, __int128 arg2)
{
    __int128 even_arg1;
    __int128 even_arg2;
    __int128 odd_arg1;
    __int128 odd_arg2;
    LSX_UNSIGNED_EXTRACT(even_arg1, odd_arg1, arg1, df);
    LSX_UNSIGNED_EXTRACT(even_arg2, odd_arg2, arg2, df);
    return dest + (even_arg1 * even_arg2) + (odd_arg1 * odd_arg2);
}

static inline __int128 lsx_xvdp2add_s_u_s_df(uint32_t df, __int128 dest, __int128 arg1, __int128 arg2)
{
    __int128 even_arg1;
    __int128 even_arg2;
    __int128 odd_arg1;
    __int128 odd_arg2;
    LSX_UNSIGNED_EXTRACT(even_arg1, odd_arg1, arg1, df);
    LSX_SIGNED_EXTRACT(even_arg2, odd_arg2, arg2, df);
    return dest + (even_arg1 * even_arg2) + (odd_arg1 * odd_arg2);
}

static inline __int128 lsx_xvdp2sub_s_s_df(uint32_t df, __int128 dest, __int128 arg1, __int128 arg2)
{
    __int128 even_arg1;
    __int128 even_arg2;
    __int128 odd_arg1;
    __int128 odd_arg2;
    LSX_SIGNED_EXTRACT(even_arg1, odd_arg1, arg1, df);
    LSX_SIGNED_EXTRACT(even_arg2, odd_arg2, arg2, df);
    return dest - ((even_arg1 * even_arg2) + (odd_arg1 * odd_arg2));
}

static inline __int128 lsx_xvdp2sub_s_u_df(uint32_t df, __int128 dest, __int128 arg1, __int128 arg2)
{
    __int128 even_arg1;
    __int128 even_arg2;
    __int128 odd_arg1;
    __int128 odd_arg2;
    LSX_UNSIGNED_EXTRACT(even_arg1, odd_arg1, arg1, df);
    LSX_UNSIGNED_EXTRACT(even_arg2, odd_arg2, arg2, df);
    return dest - ((even_arg1 * even_arg2) + (odd_arg1 * odd_arg2));
}


static inline int64_t lsx_xvbstrc12_df(uint32_t df,
                                   int64_t dest, int64_t arg1, int64_t arg2)
{
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    uint64_t u_dest = UNSIGNED(dest, df);
    int32_t sh_d = BIT_POSITION(arg2, df) + 1;
    int32_t sh_a = DF_BITS(df) - sh_d;
    if (sh_d == DF_BITS(df)) {
        return u_arg1;
    } else {
        return UNSIGNED(UNSIGNED(u_dest >> sh_d, df) << sh_d, df) |
               UNSIGNED(UNSIGNED(u_arg1 << sh_a, df) >> sh_a, df);
    }
}

static inline int64_t lsx_xvbstrc21_df(uint32_t df,
                                   int64_t dest, int64_t arg1, int64_t arg2)
{
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    uint64_t u_dest = UNSIGNED(dest, df);
    int32_t sh_d = BIT_POSITION(arg2, df) + 1;
    int32_t sh_a = DF_BITS(df) - sh_d;
    if (sh_d == DF_BITS(df)) {
        return u_arg1;
    } else {
        return UNSIGNED(UNSIGNED(u_dest << sh_d, df) >> sh_d, df) |
               UNSIGNED(UNSIGNED(u_arg1 >> sh_a, df) << sh_a, df);
    }
}


#define LSX_TERXOP_DF(func) \
void helper_lsx_ ## func ## _df(CPULoongArchState *env, uint32_t df, uint32_t wd,       \
                                uint32_t ws, uint32_t wt)                          \
{                                                                                  \
    wr_t *pwd = &(env->fpr[wd].wr);                                     \
    wr_t *pws = &(env->fpr[ws].wr);                                     \
    wr_t *pwt = &(env->fpr[wt].wr);                                     \
    uint32_t i;                                                                    \
                                                                                   \
    switch (df) {                                                                  \
    case DF_BYTE:                                                                  \
        for (i = 0; i < (256/DF_BITS(df)); i++) {                                  \
            pwd->b[i] = lsx_ ## func ## _df(df, pwd->b[i], pws->b[i], pwt->b[i]);  \
        }                                                                          \
        break;                                                                     \
    case DF_HALF:                                                                  \
        for (i = 0; i < (256/DF_BITS(df)); i++) {                                  \
            pwd->h[i] = lsx_ ## func ## _df(df, pwd->h[i], pws->h[i], pwt->h[i]);  \
        }                                                                          \
        break;                                                                     \
    case DF_WORD:                                                                  \
        for (i = 0; i < (256/DF_BITS(df)); i++) {                                  \
            pwd->w[i] = lsx_ ## func ## _df(df, pwd->w[i], pws->w[i], pwt->w[i]);  \
        }                                                                          \
        break;                                                                     \
    case DF_DOUBLE:                                                                \
        for (i = 0; i < (256/DF_BITS(df)); i++) {                                  \
            pwd->d[i] = lsx_ ## func ## _df(df, pwd->d[i], pws->d[i], pwt->d[i]);  \
        }                                                                          \
        break;                                                                     \
    case DF_QUAD:                                                                  \
        for (i = 0; i < (256/DF_BITS(df)); i++) {                                  \
            pwd->q[i] = lsx_ ## func ## _df(df, pwd->d[i], pws->q[i], pwt->q[i]);  \
        }                                                                          \
        break;                                                                     \
    default:                                                                       \
        assert(0);                                                                 \
    }                                                                              \
}

LSX_TERXOP_DF(xvmadd)
LSX_TERXOP_DF(xvmsub)
LSX_TERXOP_DF(xvdp2add_s_s)
LSX_TERXOP_DF(xvdp2add_s_u)
LSX_TERXOP_DF(xvdp2add_s_u_s)
LSX_TERXOP_DF(xvdp2sub_s_s)
LSX_TERXOP_DF(xvdp2sub_s_u)
LSX_TERXOP_DF(xvbstrc12)
LSX_TERXOP_DF(xvbstrc21)
#undef LSX_TERXOP_DF


void helper_lsx_xvpackev_df(CPULoongArchState *env, uint32_t df, uint32_t wd,
                         uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    uint32_t i;

    switch (df) {
    case DF_BYTE:
        for (i = 0; i < (256/16); i++) {
            pwd->b[2*i+1] = pws->b[2*i];
            pwd->b[2*i]   = pwt->b[2*i];
        }
        break;
    case DF_HALF:
        for (i = 0; i < (256/32); i++) {
            pwd->h[2*i+1] = pws->h[2*i];
            pwd->h[2*i]   = pwt->h[2*i];
        }
        break;
    case DF_WORD:
        for (i = 0; i < (256/64); i++) {
            pwd->w[2*i+1] = pws->w[2*i];
            pwd->w[2*i]   = pwt->w[2*i];
        }
        break;
    case DF_DOUBLE:
        for (i = 0; i < (256/128); i++) {
            pwd->d[2*i+1] = pws->d[2*i];
            pwd->d[2*i]   = pwt->d[2*i];
        }
       break;
    default:
        assert(0);
    }
}

void helper_lsx_xvpackod_df(CPULoongArchState *env, uint32_t df, uint32_t wd,
                         uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    uint32_t i;

    switch (df) {
    case DF_BYTE:
        for (i = 0; i < (256/16); i++) {
            pwd->b[2*i+1] = pws->b[2*i+1];
            pwd->b[2*i]   = pwt->b[2*i+1];
        }
        break;
    case DF_HALF:
        for (i = 0; i < (256/32); i++) {
            pwd->h[2*i+1] = pws->h[2*i+1];
            pwd->h[2*i]   = pwt->h[2*i+1];
        }
        break;
    case DF_WORD:
        for (i = 0; i < (256/64); i++) {
            pwd->w[2*i+1] = pws->w[2*i+1];
            pwd->w[2*i]   = pwt->w[2*i+1];
        }
        break;
    case DF_DOUBLE:
        for (i = 0; i < (256/128); i++) {
            pwd->d[2*i+1] = pws->d[2*i+1];
            pwd->d[2*i]   = pwt->d[2*i+1];
        }
       break;
    default:
        assert(0);
    }
}

void helper_lsx_xvilvl_df(CPULoongArchState *env, uint32_t df, uint32_t wd,
                         uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    uint32_t i;

    switch (df) {
    case DF_BYTE:
        for (i = 0; i < (256/16); i++) {
            pwd->b[2*i+1] = pws->b[i];
            pwd->b[2*i]   = pwt->b[i];
        }
        break;
    case DF_HALF:
        for (i = 0; i < (256/32); i++) {
            pwd->h[2*i+1] = pws->h[i];
            pwd->h[2*i]   = pwt->h[i];
        }
        break;
    case DF_WORD:
        for (i = 0; i < (256/64); i++) {
            pwd->w[2*i+1] = pws->w[i];
            pwd->w[2*i]   = pwt->w[i];
        }
        break;
    case DF_DOUBLE:
        for (i = 0; i < (256/128); i++) {
            pwd->d[2*i+1] = pws->d[i];
            pwd->d[2*i]   = pwt->d[i];
        }
       break;
    default:
        assert(0);
    }
}

void helper_lsx_xvilvh_df(CPULoongArchState *env, uint32_t df, uint32_t wd,
                         uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    uint32_t i;

    switch (df) {
    case DF_BYTE:
        for (i = 0; i < (256/16); i++) {
            pwd->b[2*i+1] = pws->b[i+16];
            pwd->b[2*i]   = pwt->b[i+16];
        }
        break;
    case DF_HALF:
        for (i = 0; i < (256/32); i++) {
            pwd->h[2*i+1] = pws->h[i+8];
            pwd->h[2*i]   = pwt->h[i+8];
        }
        break;
    case DF_WORD:
        for (i = 0; i < (256/64); i++) {
            pwd->w[2*i+1] = pws->w[i+4];
            pwd->w[2*i]   = pwt->w[i+4];
        }
        break;
    case DF_DOUBLE:
        for (i = 0; i < (256/128); i++) {
            pwd->d[2*i+1] = pws->d[i+2];
            pwd->d[2*i]   = pwt->d[i+2];
        }
       break;
    default:
        assert(0);
    }
}

void helper_lsx_xvpickev_df(CPULoongArchState *env, uint32_t df, uint32_t wd,
                         uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    uint32_t i;

    switch (df) {
    case DF_BYTE:
        for (i = 0; i < (256/16); i++) {
            pwd->b[i+16] = pws->b[i*2];
            pwd->b[i]    = pwt->b[i*2];
        }
        break;
    case DF_HALF:
        for (i = 0; i < (256/32); i++) {
            pwd->h[i+8] = pws->h[i*2];
            pwd->h[i]   = pwt->h[i*2];
        }
        break;
    case DF_WORD:
        for (i = 0; i < (256/64); i++) {
            pwd->w[i+4] = pws->w[i*2];
            pwd->w[i]   = pwt->w[i*2];
        }
        break;
    case DF_DOUBLE:
        for (i = 0; i < (256/128); i++) {
            pwd->d[i+2] = pws->d[i*2];
            pwd->d[i]   = pwt->d[i*2];
        }
       break;
    default:
        assert(0);
    }
}

void helper_lsx_xvpickod_df(CPULoongArchState *env, uint32_t df, uint32_t wd,
                         uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    uint32_t i;

    switch (df) {
    case DF_BYTE:
        for (i = 0; i < (256/16); i++) {
            pwd->b[i+16] = pws->b[i*2+1];
            pwd->b[i]    = pwt->b[i*2+1];
        }
        break;
    case DF_HALF:
        for (i = 0; i < (256/32); i++) {
            pwd->h[i+8] = pws->h[i*2+1];
            pwd->h[i]   = pwt->h[i*2+1];
        }
        break;
    case DF_WORD:
        for (i = 0; i < (256/64); i++) {
            pwd->w[i+4] = pws->w[i*2+1];
            pwd->w[i]   = pwt->w[i*2+1];
        }
        break;
    case DF_DOUBLE:
        for (i = 0; i < (256/128); i++) {
            pwd->d[i+2] = pws->d[i*2+1];
            pwd->d[i]   = pwt->d[i*2+1];
        }
       break;
    default:
        assert(0);
    }
}


static inline void lsx_xvreplve_df(uint32_t df, wr_t *pwd,
                                wr_t *pws, target_ulong rt)
{
    uint32_t n = rt % DF_ELEMENTS(df);
    uint32_t i;

    switch (df) {
    case DF_BYTE:
        for (i = 0; i < DF_ELEMENTS(DF_BYTE); i++) {
            pwd->b[i] = pws->b[n];
        }
        break;
    case DF_HALF:
        for (i = 0; i < DF_ELEMENTS(DF_HALF); i++) {
            pwd->h[i] = pws->h[n];
        }
        break;
    case DF_WORD:
        for (i = 0; i < DF_ELEMENTS(DF_WORD); i++) {
            pwd->w[i] = pws->w[n];
        }
        break;
    case DF_DOUBLE:
        for (i = 0; i < DF_ELEMENTS(DF_DOUBLE); i++) {
            pwd->d[i] = pws->d[n];
        }
       break;
    default:
        assert(0);
    }
}

void helper_lsx_xvreplve_df(CPULoongArchState *env, uint32_t df, uint32_t wd,
                         uint32_t ws, uint32_t rt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    lsx_xvreplve_df(df, pwd, pws, env->gpr[rt]);
}

#define CONCATENATE_AND_SLIDE(s, k)             \
    do {                                        \
        for (i = 0; i < s; i++) {               \
            v[i]     = pws->b[s * k + i];       \
            v[i + s] = pwd->b[s * k + i];       \
        }                                       \
        for (i = 0; i < s; i++) {               \
            pwd->b[s * k + i] = v[i + n];       \
        }                                       \
    } while (0)

static inline void lsx_xvextrcol_df(uint32_t df, wr_t *pwd,
                              wr_t *pws, target_ulong rt)
{
    uint32_t n = rt % DF_ELEMENTS(df);
    uint8_t v[64];
    uint32_t i, k;

    switch (df) {
    case DF_BYTE:
        CONCATENATE_AND_SLIDE(DF_ELEMENTS(DF_BYTE), 0);
        break;
    case DF_HALF:
        for (k = 0; k < 2; k++) {
            CONCATENATE_AND_SLIDE(DF_ELEMENTS(DF_HALF), k);
        }
        break;
    case DF_WORD:
        for (k = 0; k < 4; k++) {
            CONCATENATE_AND_SLIDE(DF_ELEMENTS(DF_WORD), k);
        }
        break;
    case DF_DOUBLE:
        for (k = 0; k < 8; k++) {
            CONCATENATE_AND_SLIDE(DF_ELEMENTS(DF_DOUBLE), k);
        }
        break;
    default:
        assert(0);
    }
}

void helper_lsx_xvextrcol_df(CPULoongArchState *env, uint32_t df, uint32_t wd,
                       uint32_t ws, uint32_t rt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    lsx_xvextrcol_df(df, pwd, pws, env->gpr[rt]);
}


void helper_lsx_xvand_v(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->d[0] = pws->d[0] & pwt->d[0];
    pwd->d[1] = pws->d[1] & pwt->d[1];
    pwd->d[2] = pws->d[2] & pwt->d[2];
    pwd->d[3] = pws->d[3] & pwt->d[3];
}

void helper_lsx_xvnor_v(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->d[0] = ~(pws->d[0] | pwt->d[0]);
    pwd->d[1] = ~(pws->d[1] | pwt->d[1]);
    pwd->d[2] = ~(pws->d[2] | pwt->d[2]);
    pwd->d[3] = ~(pws->d[3] | pwt->d[3]);
}

void helper_lsx_xvor_v(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->d[0] = pws->d[0] | pwt->d[0];
    pwd->d[1] = pws->d[1] | pwt->d[1];
    pwd->d[2] = pws->d[2] | pwt->d[2];
    pwd->d[3] = pws->d[3] | pwt->d[3];
}

void helper_lsx_xvxor_v(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->d[0] = pws->d[0] ^ pwt->d[0];
    pwd->d[1] = pws->d[1] ^ pwt->d[1];
    pwd->d[2] = pws->d[2] ^ pwt->d[2];
    pwd->d[3] = pws->d[3] ^ pwt->d[3];
}

void helper_lsx_xvandn_v(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->d[0] = ~pws->d[0] & pwt->d[0];
    pwd->d[1] = ~pws->d[1] & pwt->d[1];
    pwd->d[2] = ~pws->d[2] & pwt->d[2];
    pwd->d[3] = ~pws->d[3] & pwt->d[3];
}

void helper_lsx_xvorn_v(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->d[0] = ~pws->d[0] | pwt->d[0];
    pwd->d[1] = ~pws->d[1] | pwt->d[1];
    pwd->d[2] = ~pws->d[2] | pwt->d[2];
    pwd->d[3] = ~pws->d[3] | pwt->d[3];
}


void helper_lsx_vsubw_h_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/16; i++) {
        tmp.h[i] = pws->h[i] - (int16_t)pwt->b[i];
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vsubw_w_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);
    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/32; i++) {
        tmp.w[i] = pws->w[i] - (int32_t)pwt->h[i];
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vsubw_d_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);
    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/64; i++) {
        tmp.d[i] = pws->d[i] - (int64_t)pwt->w[i];
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vsubw_h_h_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/16; i++) {
        tmp.h[i] = pws->h[i] - (uint16_t)((uint8_t)pwt->b[i]);
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vsubw_w_w_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);
    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/32; i++) {
        tmp.w[i] = pws->w[i] - (uint32_t)((uint16_t)pwt->h[i]);
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vsubw_d_d_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);
    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/64; i++) {
        tmp.d[i] = pws->d[i] - (uint64_t)((uint32_t)pwt->w[i]);
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvsubw_h_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/16; i++) {
        tmp.h[i] = pws->h[i] - (int16_t)pwt->b[i];
        tmp.h[i+8] = pws->h[i+8] - (int16_t)pwt->b[i+16];
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvsubw_w_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);
    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/32; i++) {
        tmp.w[i] = pws->w[i] - (int32_t)pwt->h[i];
        tmp.w[i+4] = pws->w[i+4] - (int32_t)pwt->h[i+8];
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvsubw_d_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);
    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/64; i++) {
        tmp.d[i] = pws->d[i] - (int64_t)pwt->w[i];
        tmp.d[i+2] = pws->d[i+2] - (int64_t)pwt->w[i+4];
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvsubw_h_h_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/16; i++) {
        tmp.h[i] = pws->h[i] - (uint16_t)((uint8_t)pwt->b[i]);
        tmp.h[i+8] = pws->h[i+8] - (uint16_t)((uint8_t)pwt->b[i+16]);
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvsubw_w_w_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);
    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/32; i++) {
        tmp.w[i] = pws->w[i] - (uint32_t)((uint16_t)pwt->h[i]);
        tmp.w[i+4] = pws->w[i+4] - (uint32_t)((uint16_t)pwt->h[i+8]);
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvsubw_d_d_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);
    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/64; i++) {
        tmp.d[i] = pws->d[i] - (uint64_t)((uint32_t)pwt->w[i]);
        tmp.d[i+2] = pws->d[i+2] - (uint64_t)((uint32_t)pwt->w[i+4]);
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}


static inline int64_t lsx_adds_s_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    int64_t max_int = DF_MAX_INT(df);
    int64_t min_int = DF_MIN_INT(df);
    if (arg1 < 0) {
        return (min_int - arg1 < arg2) ? arg1 + arg2 : min_int;
    } else {
        return (arg2 < max_int - arg1) ? arg1 + arg2 : max_int;
    }
}

void helper_lsx_vsaddw_h_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/16; i++){
        tmp.h[i] = lsx_adds_s_df(DF_HALF,pws->h[i],(int16_t)pwt->b[i]);
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vsaddw_w_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/32; i++){
        tmp.w[i] = lsx_adds_s_df(DF_WORD,pws->w[i],(int32_t)pwt->h[i]);
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}


void helper_lsx_vsaddw_d_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/64; i++){
        tmp.d[i] = lsx_adds_s_df(DF_DOUBLE,pws->d[i],(int64_t)pwt->w[i]);
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvsaddw_h_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/16; i++){
        tmp.h[i] = lsx_adds_s_df(DF_HALF,pws->h[i],(int16_t)pwt->b[i]);
        tmp.h[i+8] = lsx_adds_s_df(DF_HALF,pws->h[i+8],(int16_t)pwt->b[i+16]);
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvsaddw_w_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/32; i++){
        tmp.w[i] = lsx_adds_s_df(DF_WORD,pws->w[i],(int32_t)pwt->h[i]);
        tmp.w[i+4] = lsx_adds_s_df(DF_WORD,pws->w[i+4],(int32_t)pwt->h[i+8]);
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}


void helper_lsx_xvsaddw_d_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/64; i++){
        tmp.d[i] = lsx_adds_s_df(DF_DOUBLE,pws->d[i],(int64_t)pwt->w[i]);
        tmp.d[i+2] = lsx_adds_s_df(DF_DOUBLE,pws->d[i+2],(int64_t)pwt->w[i+4]);
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}


static inline uint64_t lsx_adds_u_df(uint32_t df, uint64_t arg1, uint64_t arg2)
{
    uint64_t max_uint = DF_MAX_UINT(df);
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    uint64_t u_arg2 = UNSIGNED(arg2, df);
    return (u_arg1 < max_uint - u_arg2) ? u_arg1 + u_arg2 : max_uint;
}


void helper_lsx_vsaddw_hu_hu_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/16; i++){
        tmp.h[i] = lsx_adds_u_df(DF_HALF,pws->h[i],(uint16_t)((uint8_t)pwt->b[i]));
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vsaddw_wu_wu_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/32; i++){
        tmp.w[i] = lsx_adds_u_df(DF_WORD,pws->w[i],(uint32_t)((uint16_t)pwt->h[i]));
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}


void helper_lsx_vsaddw_du_du_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/64; i++){
        tmp.d[i] = lsx_adds_u_df(DF_DOUBLE,pws->d[i],(uint64_t)((uint32_t)pwt->w[i]));
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvsaddw_hu_hu_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/16; i++){
        tmp.h[i] = lsx_adds_u_df(DF_HALF,pws->h[i],(uint16_t)((uint8_t)pwt->b[i]));
        tmp.h[i+8] = lsx_adds_u_df(DF_HALF,pws->h[i+8],(uint16_t)((uint8_t)pwt->b[i+16]));
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvsaddw_wu_wu_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/32; i++){
        tmp.w[i] = lsx_adds_u_df(DF_WORD,pws->w[i],(uint32_t)((uint16_t)pwt->h[i]));
        tmp.w[i+4] = lsx_adds_u_df(DF_WORD,pws->w[i+4],(uint32_t)((uint16_t)pwt->h[i+8]));
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}


void helper_lsx_xvsaddw_du_du_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/64; i++){
        tmp.d[i] = lsx_adds_u_df(DF_DOUBLE,pws->d[i],(uint64_t)((uint32_t)pwt->w[i]));
        tmp.d[i+2] = lsx_adds_u_df(DF_DOUBLE,pws->d[i+2],(uint64_t)((uint32_t)pwt->w[i+4]));
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}


static inline int64_t lsx_subs_s_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    int64_t max_int = DF_MAX_INT(df);
    int64_t min_int = DF_MIN_INT(df);
    if (arg2 > 0) {
        return (min_int + arg2 < arg1) ? arg1 - arg2 : min_int;
    } else {
        return (arg1 < max_int + arg2) ? arg1 - arg2 : max_int;
    }
}


void helper_lsx_vssubw_h_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/16; i++){
        tmp.h[i] = lsx_subs_s_df(DF_HALF,pws->h[i],(int16_t)pwt->b[i]);
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vssubw_w_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/32; i++){
        tmp.w[i] = lsx_subs_s_df(DF_WORD,pws->w[i],(int32_t)pwt->h[i]);
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}


void helper_lsx_vssubw_d_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/64; i++){
        tmp.d[i] = lsx_subs_s_df(DF_DOUBLE,pws->d[i],(int64_t)pwt->w[i]);
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvssubw_h_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/16; i++){
        tmp.h[i] = lsx_subs_s_df(DF_HALF,pws->h[i],(int16_t)pwt->b[i]);
        tmp.h[i+8] = lsx_subs_s_df(DF_HALF,pws->h[i+8],(int16_t)pwt->b[i+16]);
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvssubw_w_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/32; i++){
        tmp.w[i] = lsx_subs_s_df(DF_WORD,pws->w[i],(int32_t)pwt->h[i]);
        tmp.w[i+4] = lsx_subs_s_df(DF_WORD,pws->w[i+4],(int32_t)pwt->h[i+8]);
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}


void helper_lsx_xvssubw_d_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/64; i++){
        tmp.d[i] = lsx_subs_s_df(DF_DOUBLE,pws->d[i],(int64_t)pwt->w[i]);
        tmp.d[i+2] = lsx_subs_s_df(DF_DOUBLE,pws->d[i+2],(int64_t)pwt->w[i+4]);
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}


static inline uint64_t lsx_subs_u_df(uint32_t df, uint64_t arg1, uint64_t arg2)
{
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    uint64_t u_arg2 = UNSIGNED(arg2, df);
    return (u_arg1 > u_arg2) ? u_arg1 - u_arg2 : 0;
}

void helper_lsx_vssubw_hu_hu_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/16; i++){
        tmp.h[i] = lsx_subs_u_df(DF_HALF,pws->h[i],(uint16_t)((uint8_t)pwt->b[i]));
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vssubw_wu_wu_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/32; i++){
        tmp.w[i] = lsx_subs_u_df(DF_WORD,pws->w[i],(uint32_t)((uint16_t)pwt->h[i]));
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}


void helper_lsx_vssubw_du_du_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/64; i++){
        tmp.d[i] = lsx_subs_u_df(DF_DOUBLE,pws->d[i],(uint64_t)((uint32_t)pwt->w[i]));
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvssubw_hu_hu_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/16; i++){
        tmp.h[i] = lsx_subs_u_df(DF_HALF,pws->h[i],(uint16_t)((uint8_t)pwt->b[i]));
        tmp.h[i+8] = lsx_subs_u_df(DF_HALF,pws->h[i+8],(uint16_t)((uint8_t)pwt->b[i+16]));
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvssubw_wu_wu_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/32; i++){
        tmp.w[i] = lsx_subs_u_df(DF_WORD,pws->w[i],(uint32_t)((uint16_t)pwt->h[i]));
        tmp.w[i+4] = lsx_subs_u_df(DF_WORD,pws->w[i+4],(uint32_t)((uint16_t)pwt->h[i+8]));
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}


void helper_lsx_xvssubw_du_du_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/64; i++){
        tmp.d[i] = lsx_subs_u_df(DF_DOUBLE,pws->d[i],(uint64_t)((uint32_t)pwt->w[i]));
        tmp.d[i+2] = lsx_subs_u_df(DF_DOUBLE,pws->d[i+2],(uint64_t)((uint32_t)pwt->w[i+4]));
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vaddwev_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = (int8_t)pws->b[2*i] + (int8_t)pwt->b[2*i];
    }
}

void helper_lsx_vaddwev_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = (int16_t)pws->h[2*i] + (int16_t)pwt->h[2*i];
    }
}


void helper_lsx_vaddwev_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64; i++) {
        pwd->d[i] = (int64_t)(int32_t)pws->w[2*i] + (int64_t)(int32_t)pwt->w[2*i];
    }
}


void helper_lsx_vaddwev_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0] = (__int128)(int64_t)pws->d[0] + (__int128)(int64_t)pwt->d[0];
}

void helper_lsx_vsubwev_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = (int16_t)pws->b[2*i] - (int16_t)pwt->b[2*i];
    }
}

void helper_lsx_vsubwev_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = (int32_t)pws->h[2*i] - (int32_t)pwt->h[2*i];
    }
}


void helper_lsx_vsubwev_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64; i++) {
        pwd->d[i] = (int64_t)(int32_t)pws->w[2*i] - (int64_t)(int32_t)pwt->w[2*i];
    }
}


void helper_lsx_vsubwev_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0] = (__int128)(int64_t)pws->d[0] - (__int128)(int64_t)pwt->d[0];
}


void helper_lsx_vaddwod_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = (int16_t)pws->b[2*i+1] + (int16_t)pwt->b[2*i+1];
    }
}

void helper_lsx_vaddwod_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = (int32_t)pws->h[2*i+1] + (int32_t)pwt->h[2*i+1];
    }
}


void helper_lsx_vaddwod_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64; i++) {
        pwd->d[i] = (int64_t)(int32_t)pws->w[2*i+1] + (int64_t)(int32_t)pwt->w[2*i+1];
    }
}


void helper_lsx_vaddwod_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0] = (__int128)(int64_t)pws->d[1] + (__int128)(int64_t)pwt->d[1];
}

void helper_lsx_vsubwod_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = (int16_t)pws->b[2*i+1] - (int16_t)pwt->b[2*i+1];
    }
}

void helper_lsx_vsubwod_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = (int32_t)pws->h[2*i+1] - (int32_t)pwt->h[2*i+1];
    }
}


void helper_lsx_vsubwod_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64; i++) {
        pwd->d[i] = (int64_t)pws->w[2*i+1] - (int64_t)pwt->w[2*i+1];
    }
}


void helper_lsx_vsubwod_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0] = (__int128)pws->d[1] - (__int128)pwt->d[1];
}

void helper_lsx_xvaddwev_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/16; i++) {
        pwd->h[i] = (int16_t)pws->b[2*i] + (int16_t)pwt->b[2*i];
    }
}

void helper_lsx_xvaddwev_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/32; i++) {
        pwd->w[i] = (int32_t)pws->h[2*i] + (int32_t)pwt->h[2*i];
    }
}


void helper_lsx_xvaddwev_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/64; i++) {
        pwd->d[i] = (int64_t)pws->w[2*i] + (int64_t)pwt->w[2*i];
    }
}


void helper_lsx_xvaddwev_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0] = (__int128)pws->d[0] + (__int128)pwt->d[0];
    pwd->q[1] = (__int128)pws->d[2] + (__int128)pwt->d[2];
}

void helper_lsx_xvsubwev_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/16; i++) {
        pwd->h[i] = (int16_t)pws->b[2*i] - (int16_t)pwt->b[2*i];
    }
}

void helper_lsx_xvsubwev_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/32; i++) {
        pwd->w[i] = (int32_t)pws->h[2*i] - (int32_t)pwt->h[2*i];
    }
}


void helper_lsx_xvsubwev_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/64; i++) {
        pwd->d[i] = (int64_t)pws->w[2*i] - (int64_t)pwt->w[2*i];
    }
}


void helper_lsx_xvsubwev_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0] = (__int128)pws->d[0] - (__int128)pwt->d[0];
    pwd->q[1] = (__int128)pws->d[2] - (__int128)pwt->d[2];
}


void helper_lsx_xvaddwod_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/16; i++) {
        pwd->h[i] = (int16_t)pws->b[2*i+1] + (int16_t)pwt->b[2*i+1];
    }
}

void helper_lsx_xvaddwod_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/32; i++) {
        pwd->w[i] = (int32_t)pws->h[2*i+1] + (int32_t)pwt->h[2*i+1];
    }
}


void helper_lsx_xvaddwod_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/64; i++) {
        pwd->d[i] = (int64_t)pws->w[2*i+1] + (int64_t)pwt->w[2*i+1];
    }
}


void helper_lsx_xvaddwod_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0] = (__int128)pws->d[1] + (__int128)pwt->d[1];
    pwd->q[1] = (__int128)pws->d[3] + (__int128)pwt->d[3];
}

void helper_lsx_xvsubwod_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/16; i++) {
        pwd->h[i] = (int16_t)pws->b[2*i+1] - (int16_t)pwt->b[2*i+1];
    }
}

void helper_lsx_xvsubwod_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/32; i++) {
        pwd->w[i] = (int32_t)pws->h[2*i+1] - (int32_t)pwt->h[2*i+1];
    }
}


void helper_lsx_xvsubwod_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/64; i++) {
        pwd->d[i] = (int64_t)pws->w[2*i+1] - (int64_t)pwt->w[2*i+1];
    }
}


void helper_lsx_xvsubwod_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0] = (__int128)pws->d[1] - (__int128)pwt->d[1];
    pwd->q[1] = (__int128)pws->d[3] - (__int128)pwt->d[3];
}


void helper_lsx_vaddwl_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/16; i++) {
        tmp.h[i] = (int16_t)pws->b[i] + (int16_t)pwt->b[i];
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vaddwl_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/32; i++) {
        tmp.w[i] = (int32_t)pws->h[i] + (int32_t)pwt->h[i];
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}


void helper_lsx_vaddwl_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/64; i++) {
        tmp.d[i] = (int64_t)pws->w[i] + (int64_t)pwt->w[i];
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}


void helper_lsx_vaddwl_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0] = (__int128)pws->d[0] + (__int128)pwt->d[0];
}


void helper_lsx_vsubwl_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/16; i++) {
        tmp.h[i] = (int16_t)pws->b[i] - (int16_t)pwt->b[i];
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vsubwl_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/32; i++) {
        tmp.w[i] = (int32_t)pws->h[i] - (int32_t)pwt->h[i];
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}


void helper_lsx_vsubwl_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/64; i++) {
        tmp.d[i] = (int64_t)pws->w[i] - (int64_t)pwt->w[i];
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}


void helper_lsx_vsubwl_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0] = (__int128)pws->d[0] - (__int128)pwt->d[0];
}

void helper_lsx_vaddwh_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = (int16_t)pws->b[i+128/16] + (int16_t)pwt->b[i+128/16];
    }
}

void helper_lsx_vaddwh_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = (int32_t)pws->h[i+128/32] + (int32_t)pwt->h[i+128/32];
    }
}


void helper_lsx_vaddwh_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64; i++) {
        pwd->d[i] = (int64_t)pws->w[i+128/64] + (int64_t)pwt->w[i+128/64];
    }
}


void helper_lsx_vaddwh_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0] = (__int128)pws->d[1] + (__int128)pwt->d[1];
}


void helper_lsx_vsubwh_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = (int16_t)pws->b[i+128/16] - (int16_t)pwt->b[i+128/16];
    }
}

void helper_lsx_vsubwh_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = (int32_t)pws->h[i+128/32] - (int32_t)pwt->h[i+128/32];
    }
}


void helper_lsx_vsubwh_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64; i++) {
        pwd->d[i] = (int64_t)pws->w[i+128/64] - (int64_t)pwt->w[i+128/64];
    }
}


void helper_lsx_vsubwh_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0] = (__int128)pws->d[1] - (__int128)pwt->d[1];
}

void helper_lsx_xvaddwl_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/16; i++) {
        tmp.h[i] = (int16_t)pws->b[i] + (int16_t)pwt->b[i];
        tmp.h[i+8] = (int16_t)pws->b[i+16] + (int16_t)pwt->b[i+16];
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvaddwl_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/32; i++) {
        tmp.w[i] = (int32_t)pws->h[i] + (int32_t)pwt->h[i];
        tmp.w[i+4] = (int32_t)pws->h[i+8] + (int32_t)pwt->h[i+8];
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}


void helper_lsx_xvaddwl_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/64; i++) {
        tmp.d[i] = (int64_t)pws->w[i] + (int64_t)pwt->w[i];
        tmp.d[i+2] = (int64_t)pws->w[i+4] + (int64_t)pwt->w[i+4];
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}


void helper_lsx_xvaddwl_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0] = (__int128)pws->d[0] + (__int128)pwt->d[0];
    pwd->q[1] = (__int128)pws->d[2] + (__int128)pwt->d[2];
}


void helper_lsx_xvsubwl_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/16; i++) {
        tmp.h[i] = (int16_t)pws->b[i] - (int16_t)pwt->b[i];
        tmp.h[i+8] = (int16_t)pws->b[i+16] - (int16_t)pwt->b[i+16];
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvsubwl_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/32; i++) {
        tmp.w[i] = (int32_t)pws->h[i] - (int32_t)pwt->h[i];
        tmp.w[i+4] = (int32_t)pws->h[i+8] - (int32_t)pwt->h[i+8];
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}


void helper_lsx_xvsubwl_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/64; i++) {
        tmp.d[i] = (int64_t)pws->w[i] - (int64_t)pwt->w[i];
        tmp.d[i+2] = (int64_t)pws->w[i+4] - (int64_t)pwt->w[i+4];
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}


void helper_lsx_xvsubwl_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0] = (__int128)pws->d[0] - (__int128)pwt->d[0];
    pwd->q[1] = (__int128)pws->d[2] - (__int128)pwt->d[2];
}

void helper_lsx_xvaddwh_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = (int16_t)pws->b[i+128/16] + (int16_t)pwt->b[i+128/16];
        pwd->h[i+8] = (int16_t)pws->b[i+128/16+16] + (int16_t)pwt->b[i+128/16+16];
    }
}

void helper_lsx_xvaddwh_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = (int32_t)pws->h[i+128/32] + (int32_t)pwt->h[i+128/32];
        pwd->w[i+4] = (int32_t)pws->h[i+128/32+8] + (int32_t)pwt->h[i+128/32+8];
    }
}


void helper_lsx_xvaddwh_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64; i++) {
        pwd->d[i] = (int64_t)pws->w[i+128/64] + (int64_t)pwt->w[i+128/64];
        pwd->d[i+2] = (int64_t)pws->w[i+128/64+4] + (int64_t)pwt->w[i+128/64+4];
    }
}


void helper_lsx_xvaddwh_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0] = (__int128)pws->d[1] + (__int128)pwt->d[1];
    pwd->q[1] = (__int128)pws->d[3] + (__int128)pwt->d[3];
}


void helper_lsx_xvsubwh_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = (int16_t)pws->b[i+128/16] - (int16_t)pwt->b[i+128/16];
        pwd->h[i+8] = (int16_t)pws->b[i+128/16+16] - (int16_t)pwt->b[i+128/16+16];
    }
}

void helper_lsx_xvsubwh_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = (int32_t)pws->h[i+128/32] - (int32_t)pwt->h[i+128/32];
        pwd->w[i+4] = (int32_t)pws->h[i+128/32+8] - (int32_t)pwt->h[i+128/32+8];
    }
}


void helper_lsx_xvsubwh_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64; i++) {
        pwd->d[i] = (int64_t)pws->w[i+128/64] - (int64_t)pwt->w[i+128/64];
        pwd->d[i+2] = (int64_t)pws->w[i+128/64+4] - (int64_t)pwt->w[i+128/64+4];
    }
}


void helper_lsx_xvsubwh_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0] = (__int128)pws->d[1] - (__int128)pwt->d[1];
    pwd->q[1] = (__int128)pws->d[3] - (__int128)pwt->d[3];
}


void helper_lsx_vaddwev_h_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = (uint8_t)pws->b[2*i] + (uint8_t)pwt->b[2*i];
    }
}

void helper_lsx_vaddwev_w_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = (uint16_t)pws->h[2*i] + (uint16_t)pwt->h[2*i];
    }
}


void helper_lsx_vaddwev_d_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64; i++) {
        pwd->d[i] = (uint64_t)(uint32_t)pws->w[2*i] + (uint64_t)(uint32_t)pwt->w[2*i];
    }
}


void helper_lsx_vaddwev_q_du(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0] = (__uint128_t)(uint64_t)pws->d[0] + (__uint128_t)(uint64_t)pwt->d[0];
}

void helper_lsx_vsubwev_h_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = (uint8_t)pws->b[2*i] - (uint8_t)pwt->b[2*i];
    }
}

void helper_lsx_vsubwev_w_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = (uint16_t)pws->h[2*i] - (uint16_t)pwt->h[2*i];
    }
}


void helper_lsx_vsubwev_d_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64; i++) {
        pwd->d[i] = (uint64_t)(uint32_t)pws->w[2*i] - (uint64_t)(uint32_t)pwt->w[2*i];
    }
}


void helper_lsx_vsubwev_q_du(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0] = (__uint128_t)(uint64_t)pws->d[0] - (uint64_t)pwt->d[0];
}


void helper_lsx_vaddwod_h_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = (uint8_t)pws->b[2*i+1] + (uint8_t)pwt->b[2*i+1];
    }
}

void helper_lsx_vaddwod_w_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = (uint16_t)pws->h[2*i+1] + (uint16_t)pwt->h[2*i+1];
    }
}


void helper_lsx_vaddwod_d_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64; i++) {
        pwd->d[i] = (uint64_t)(uint32_t)pws->w[2*i+1] + (uint64_t)(uint32_t)pwt->w[2*i+1];
    }
}


void helper_lsx_vaddwod_q_du(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0] = (__uint128_t)(uint64_t)pws->d[1] + (__uint128_t)(uint64_t)pwt->d[1];
}

void helper_lsx_vsubwod_h_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = (uint8_t)pws->b[2*i+1] - (uint8_t)pwt->b[2*i+1];
    }
}

void helper_lsx_vsubwod_w_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = (uint16_t)pws->h[2*i+1] - (uint16_t)pwt->h[2*i+1];
    }
}


void helper_lsx_vsubwod_d_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64; i++) {
        pwd->d[i] = (uint64_t)(uint32_t)pws->w[2*i+1] - (uint64_t)(uint32_t)pwt->w[2*i+1];
    }
}


void helper_lsx_vsubwod_q_du(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0] = (__uint128_t)(uint64_t)pws->d[1] - (uint64_t)pwt->d[1];
}

void helper_lsx_xvaddwev_h_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/16; i++) {
        pwd->h[i] = (uint8_t)pws->b[2*i] + (uint8_t)pwt->b[2*i];
    }
}

void helper_lsx_xvaddwev_w_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/32; i++) {
        pwd->w[i] = (uint16_t)pws->h[2*i] + (uint16_t)pwt->h[2*i];
    }
}


void helper_lsx_xvaddwev_d_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/64; i++) {
        pwd->d[i] = (uint64_t)(uint32_t)pws->w[2*i] + (uint64_t)(uint32_t)pwt->w[2*i];
    }
}


void helper_lsx_xvaddwev_q_du(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0] = (__uint128_t)(uint64_t)pws->d[0] + (__uint128_t)(uint64_t)pwt->d[0];
    pwd->q[1] = (__uint128_t)(uint64_t)pws->d[2] + (__uint128_t)(uint64_t)pwt->d[2];
}

void helper_lsx_xvsubwev_h_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/16; i++) {
        pwd->h[i] = (uint8_t)pws->b[2*i] - (uint8_t)pwt->b[2*i];
    }
}

void helper_lsx_xvsubwev_w_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/32; i++) {
        pwd->w[i] = (uint16_t)pws->h[2*i] - (uint16_t)pwt->h[2*i];
    }
}


void helper_lsx_xvsubwev_d_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/64; i++) {
        pwd->d[i] = (uint64_t)(uint32_t)pws->w[2*i] - (uint64_t)(uint32_t)pwt->w[2*i];
    }
}


void helper_lsx_xvsubwev_q_du(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0] = (__uint128_t)(uint64_t)pws->d[0] - (__uint128_t)(uint64_t)pwt->d[0];
    pwd->q[1] = (__uint128_t)(uint64_t)pws->d[2] - (__uint128_t)(uint64_t)pwt->d[2];
}


void helper_lsx_xvaddwod_h_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/16; i++) {
        pwd->h[i] = (uint8_t)pws->b[2*i+1] + (uint8_t)pwt->b[2*i+1];
    }
}

void helper_lsx_xvaddwod_w_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/32; i++) {
        pwd->w[i] = (uint16_t)pws->h[2*i+1] + (uint16_t)pwt->h[2*i+1];
    }
}


void helper_lsx_xvaddwod_d_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/64; i++) {
        pwd->d[i] = (uint64_t)(uint32_t)pws->w[2*i+1] + (uint64_t)(uint32_t)pwt->w[2*i+1];
    }
}


void helper_lsx_xvaddwod_q_du(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0] = (__uint128_t)(uint64_t)pws->d[1] + (__uint128_t)(uint64_t)pwt->d[1];
    pwd->q[1] = (__uint128_t)(uint64_t)pws->d[3] + (__uint128_t)(uint64_t)pwt->d[3];
}

void helper_lsx_xvsubwod_h_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/16; i++) {
        pwd->h[i] = (uint8_t)pws->b[2*i+1] - (uint8_t)pwt->b[2*i+1];
    }
}

void helper_lsx_xvsubwod_w_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/32; i++) {
        pwd->w[i] = (uint16_t)pws->h[2*i+1] - (uint16_t)pwt->h[2*i+1];
    }
}


void helper_lsx_xvsubwod_d_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/64; i++) {
        pwd->d[i] = (uint64_t)(uint32_t)pws->w[2*i+1] - (uint64_t)(uint32_t)pwt->w[2*i+1];
    }
}


void helper_lsx_xvsubwod_q_du(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0] = (__uint128_t)(uint64_t)pws->d[1] - (__uint128_t)(uint64_t)pwt->d[1];
    pwd->q[1] = (__uint128_t)(uint64_t)pws->d[3] - (__uint128_t)(uint64_t)pwt->d[3];
}


void helper_lsx_vaddwl_h_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/16; i++) {
        tmp.h[i] = (uint8_t)pws->b[i] + (uint8_t)pwt->b[i];
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vaddwl_w_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/32; i++) {
        tmp.w[i] = (uint16_t)pws->h[i] + (uint16_t)pwt->h[i];
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}


void helper_lsx_vaddwl_d_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/64; i++) {
        tmp.d[i] = (uint64_t)(uint32_t)pws->w[i] + (uint64_t)(uint32_t)pwt->w[i];
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}


void helper_lsx_vaddwl_q_du(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0] = (__uint128_t)(uint64_t)pws->d[0] + (__uint128_t)(uint64_t)pwt->d[0];
}


void helper_lsx_vsubwl_h_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/16; i++) {
        tmp.h[i] = (uint8_t)pws->b[i] - (uint8_t)pwt->b[i];
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vsubwl_w_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/32; i++) {
        tmp.w[i] = (uint16_t)pws->h[i] - (uint16_t)pwt->h[i];
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}


void helper_lsx_vsubwl_d_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/64; i++) {
        tmp.d[i] = (uint64_t)(uint32_t)pws->w[i] - (uint64_t)(uint32_t)pwt->w[i];
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}


void helper_lsx_vsubwl_q_du(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0] = (__uint128_t)(uint64_t)pws->d[0] - (__uint128_t)(uint64_t)pwt->d[0];
}

void helper_lsx_vaddwh_h_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = (uint8_t)pws->b[i+128/16] + (uint8_t)pwt->b[i+128/16];
    }
}

void helper_lsx_vaddwh_w_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = (uint16_t)pws->h[i+128/32] + (uint16_t)pwt->h[i+128/32];
    }
}


void helper_lsx_vaddwh_d_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64; i++) {
        pwd->d[i] = (uint64_t)(uint32_t)pws->w[i+128/64] + (uint64_t)(uint32_t)pwt->w[i+128/64];
    }
}


void helper_lsx_vaddwh_q_du(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0] = (__uint128_t)(uint64_t)pws->d[1] + (__uint128_t)(uint64_t)pwt->d[1];
}


void helper_lsx_vsubwh_h_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = (uint8_t)pws->b[i+128/16] - (uint8_t)pwt->b[i+128/16];
    }
}

void helper_lsx_vsubwh_w_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = (uint16_t)pws->h[i+128/32] - (uint16_t)pwt->h[i+128/32];
    }
}


void helper_lsx_vsubwh_d_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64; i++) {
        pwd->d[i] = (uint64_t)(uint32_t)pws->w[i+128/64] - (uint64_t)(uint32_t)pwt->w[i+128/64];
    }
}


void helper_lsx_vsubwh_q_du(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0] = (__uint128_t)(uint64_t)pws->d[1] - (__uint128_t)(uint64_t)pwt->d[1];
}

void helper_lsx_xvaddwl_h_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/16; i++) {
        tmp.h[i] = (uint16_t)(uint8_t)pws->b[i] + (uint16_t)(uint8_t)pwt->b[i];
        tmp.h[i+8] = (uint16_t)(uint8_t)pws->b[i+16] + (uint16_t)(uint8_t)pwt->b[i+16];
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvaddwl_w_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/32; i++) {
        tmp.w[i] = (uint32_t)(uint16_t)pws->h[i] + (uint32_t)(uint16_t)pwt->h[i];
        tmp.w[i+4] = (uint32_t)(uint16_t)pws->h[i+8] + (uint32_t)(uint16_t)pwt->h[i+8];
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}


void helper_lsx_xvaddwl_d_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/64; i++) {
        tmp.d[i] = (uint64_t)(uint32_t)pws->w[i] + (uint64_t)(uint32_t)pwt->w[i];
        tmp.d[i+2] = (uint64_t)(uint32_t)pws->w[i+4] + (uint64_t)(uint32_t)pwt->w[i+4];
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}


void helper_lsx_xvaddwl_q_du(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0] = (__uint128_t)(uint64_t)pws->d[0] + (__uint128_t)(uint64_t)pwt->d[0];
    pwd->q[1] = (__uint128_t)(uint64_t)pws->d[2] + (__uint128_t)(uint64_t)pwt->d[2];
}


void helper_lsx_xvsubwl_h_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/16; i++) {
        tmp.h[i] = (uint16_t)(uint8_t)pws->b[i] - (uint16_t)(uint8_t)pwt->b[i];
        tmp.h[i+8] = (uint16_t)(uint8_t)pws->b[i+16] - (uint16_t)(uint8_t)pwt->b[i+16];
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvsubwl_w_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/32; i++) {
        tmp.w[i] = (uint32_t)(uint16_t)pws->h[i] - (uint32_t)(uint16_t)pwt->h[i];
        tmp.w[i+4] = (uint32_t)(uint16_t)pws->h[i+8] - (uint32_t)(uint16_t)pwt->h[i+8];
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}


void helper_lsx_xvsubwl_d_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/64; i++) {
        tmp.d[i] = (uint64_t)(uint32_t)pws->w[i] - (uint64_t)(uint32_t)pwt->w[i];
        tmp.d[i+2] = (uint64_t)(uint32_t)pws->w[i+4] - (uint64_t)(uint32_t)pwt->w[i+4];
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}


void helper_lsx_xvsubwl_q_du(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0] = (__uint128_t)(uint64_t)pws->d[0] - (__uint128_t)(uint64_t)pwt->d[0];
    pwd->q[1] = (__uint128_t)(uint64_t)pws->d[2] - (__uint128_t)(uint64_t)pwt->d[2];
}

void helper_lsx_xvaddwh_h_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = (uint16_t)(uint8_t)pws->b[i+128/16] + (uint16_t)(uint8_t)pwt->b[i+128/16];
        pwd->h[i+8] = (uint16_t)(uint8_t)pws->b[i+128/16+16] + (uint16_t)(uint8_t)pwt->b[i+128/16+16];
    }
}

void helper_lsx_xvaddwh_w_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = (uint32_t)(uint16_t)pws->h[i+128/32] + (uint32_t)(uint16_t)pwt->h[i+128/32];
        pwd->w[i+4] = (uint32_t)(uint16_t)pws->h[i+128/32+8] + (uint32_t)(uint16_t)pwt->h[i+128/32+8];
    }
}


void helper_lsx_xvaddwh_d_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64; i++) {
        pwd->d[i] = (uint64_t)(uint32_t)pws->w[i+128/64] + (uint64_t)(uint32_t)pwt->w[i+128/64];
        pwd->d[i+2] = (uint64_t)(uint32_t)pws->w[i+128/64+4] + (uint64_t)(uint32_t)pwt->w[i+128/64+4];
    }
}


void helper_lsx_xvaddwh_q_du(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0] = (__uint128_t)(uint64_t)pws->d[1] + (__uint128_t)(uint64_t)pwt->d[1];
    pwd->q[1] = (__uint128_t)(uint64_t)pws->d[3] + (__uint128_t)(uint64_t)pwt->d[3];
}


void helper_lsx_xvsubwh_h_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = (uint16_t)(uint8_t)pws->b[i+128/16] - (uint16_t)(uint8_t)pwt->b[i+128/16];
        pwd->h[i+8] = (uint16_t)(uint8_t)pws->b[i+128/16+16] - (uint16_t)(uint8_t)pwt->b[i+128/16+16];
    }
}

void helper_lsx_xvsubwh_w_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = (uint32_t)(uint16_t)pws->h[i+128/32] - (uint32_t)(uint16_t)pwt->h[i+128/32];
        pwd->w[i+4] = (uint32_t)(uint16_t)pws->h[i+128/32+8] - (uint32_t)(uint16_t)pwt->h[i+128/32+8];
    }
}


void helper_lsx_xvsubwh_d_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64; i++) {
        pwd->d[i] = (uint64_t)(uint32_t)pws->w[i+128/64] - (uint64_t)(uint32_t)pwt->w[i+128/64];
        pwd->d[i+2] = (uint64_t)(uint32_t)pws->w[i+128/64+4] - (uint64_t)(uint32_t)pwt->w[i+128/64+4];
    }
}


void helper_lsx_xvsubwh_q_du(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0] = (__uint128_t)(uint64_t)pws->d[1] - (__uint128_t)(uint64_t)pwt->d[1];
    pwd->q[1] = (__uint128_t)(uint64_t)pws->d[3] - (__uint128_t)(uint64_t)pwt->d[3];
}


void helper_lsx_vaddwev_h_bu_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = (uint8_t)pws->b[2*i] + (int8_t)pwt->b[2*i];
    }
}

void helper_lsx_vaddwev_w_hu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = (uint16_t)pws->h[2*i] + (int16_t)pwt->h[2*i];
    }
}


void helper_lsx_vaddwev_d_wu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64; i++) {
        pwd->d[i] = (uint64_t)(uint32_t)pws->w[2*i] + (uint64_t)(int32_t)pwt->w[2*i];
    }
}


void helper_lsx_vaddwev_q_du_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0] = (__uint128_t)(uint64_t)pws->d[0] + (int64_t)pwt->d[0];
}

void helper_lsx_vaddwod_h_bu_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = (uint8_t)pws->b[2*i+1] + (int8_t)pwt->b[2*i+1];
    }
}

void helper_lsx_vaddwod_w_hu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = (uint16_t)pws->h[2*i+1] + (int16_t)pwt->h[2*i+1];
    }
}


void helper_lsx_vaddwod_d_wu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64; i++) {
        pwd->d[i] = (uint64_t)(uint32_t)pws->w[2*i+1] + (int32_t)pwt->w[2*i+1];
    }
}


void helper_lsx_vaddwod_q_du_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0] = (__uint128_t)(uint64_t)pws->d[1] + (int64_t)pwt->d[1];
}

void helper_lsx_vaddwl_h_bu_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/16; i++) {
        tmp.h[i] = (uint8_t)pws->b[i] + (int8_t)pwt->b[i];
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vaddwl_w_hu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/32; i++) {
        tmp.w[i] = (uint16_t)pws->h[i] + (int16_t)pwt->h[i];
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}


void helper_lsx_vaddwl_d_wu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/64; i++) {
        tmp.d[i] = (uint64_t)(uint32_t)pws->w[i] + (int32_t)pwt->w[i];
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}


void helper_lsx_vaddwl_q_du_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0] = (__uint128_t)(uint64_t)pws->d[0] + (int64_t)pwt->d[0];
}


void helper_lsx_vaddwh_h_bu_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = (uint8_t)pws->b[i+128/16] + (int8_t)pwt->b[i+128/16];
    }
}

void helper_lsx_vaddwh_w_hu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = (uint16_t)pws->h[i+128/32] + (int16_t)pwt->h[i+128/32];
    }
}


void helper_lsx_vaddwh_d_wu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64; i++) {
        pwd->d[i] = (uint64_t)(uint32_t)pws->w[i+128/64] + (int32_t)pwt->w[i+128/64];
    }
}

void helper_lsx_vaddwh_q_du_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0] = (__uint128_t)(uint64_t)pws->d[1] + (int64_t)pwt->d[1];
}

void helper_lsx_xvaddwev_h_bu_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/16; i++) {
        pwd->h[i] = (uint8_t)pws->b[2*i] + (int8_t)pwt->b[2*i];
    }
}

void helper_lsx_xvaddwev_w_hu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/32; i++) {
        pwd->w[i] = (uint16_t)pws->h[2*i] + (int16_t)pwt->h[2*i];
    }
}


void helper_lsx_xvaddwev_d_wu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/64; i++) {
        pwd->d[i] = (uint64_t)(uint32_t)pws->w[2*i] + (int64_t)pwt->w[2*i];
    }
}


void helper_lsx_xvaddwev_q_du_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0] = (__uint128_t)(uint64_t)pws->d[0] + (__int128_t)pwt->d[0];
    pwd->q[1] = (__uint128_t)(uint64_t)pws->d[2] + (__int128_t)pwt->d[2];
}

void helper_lsx_xvaddwod_h_bu_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/16; i++) {
        pwd->h[i] = (uint8_t)pws->b[2*i+1] + (int8_t)pwt->b[2*i+1];
    }
}

void helper_lsx_xvaddwod_w_hu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/32; i++) {
        pwd->w[i] = (uint16_t)pws->h[2*i+1] + (int16_t)pwt->h[2*i+1];
    }
}


void helper_lsx_xvaddwod_d_wu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/64; i++) {
        pwd->d[i] = (uint64_t)(uint32_t)pws->w[2*i+1] + (int64_t)pwt->w[2*i+1];
    }
}


void helper_lsx_xvaddwod_q_du_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0] = (__uint128_t)(uint64_t)pws->d[1] + (__int128_t)pwt->d[1];
    pwd->q[1] = (__uint128_t)(uint64_t)pws->d[3] + (__int128_t)pwt->d[3];
}

void helper_lsx_xvaddwl_h_bu_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/16; i++) {
        tmp.h[i] = (uint16_t)(uint8_t)pws->b[i] + (int16_t)pwt->b[i];
        tmp.h[i+8] = (uint16_t)(uint8_t)pws->b[i+16] + (int16_t)pwt->b[i+16];
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvaddwl_w_hu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/32; i++) {
        tmp.w[i] = (uint32_t)(uint16_t)pws->h[i] + (int32_t)pwt->h[i];
        tmp.w[i+4] = (uint32_t)(uint16_t)pws->h[i+8] + (int32_t)pwt->h[i+8];
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}


void helper_lsx_xvaddwl_d_wu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/64; i++) {
        tmp.d[i] = (uint64_t)(uint32_t)pws->w[i] + (int64_t)pwt->w[i];
        tmp.d[i+2] = (uint64_t)(uint32_t)pws->w[i+4] + (int64_t)pwt->w[i+4];
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}


void helper_lsx_xvaddwl_q_du_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0] = (__uint128_t)(uint64_t)pws->d[0] + (__int128_t)pwt->d[0];
    pwd->q[1] = (__uint128_t)(uint64_t)pws->d[2] + (__int128_t)pwt->d[2];
}


void helper_lsx_xvaddwh_h_bu_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = (uint16_t)(uint8_t)pws->b[i+128/16] + (int16_t)pwt->b[i+128/16];
        pwd->h[i+8] = (uint16_t)(uint8_t)pws->b[i+128/16+16] + (int16_t)pwt->b[i+128/16+16];
    }
}

void helper_lsx_xvaddwh_w_hu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = (uint32_t)(uint16_t)pws->h[i+128/32] + (int32_t)pwt->h[i+128/32];
        pwd->w[i+4] = (uint32_t)(uint16_t)pws->h[i+128/32+8] + (int32_t)pwt->h[i+128/32+8];
    }
}


void helper_lsx_xvaddwh_d_wu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64; i++) {
        pwd->d[i] = (uint64_t)(uint32_t)pws->w[i+128/64] + (int64_t)pwt->w[i+128/64];
        pwd->d[i+2] = (uint64_t)(uint32_t)pws->w[i+128/64+4] + (int64_t)pwt->w[i+128/64+4];
    }
}

void helper_lsx_xvaddwh_q_du_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0] = (__uint128_t)(uint64_t)pws->d[1] + (__int128_t)pwt->d[1];
    pwd->q[1] = (__uint128_t)(uint64_t)pws->d[3] + (__int128_t)pwt->d[3];
}


void helper_lsx_vhalfd_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/8; i++) {
        pwd->b[i] =(int8_t)(((int16_t)pws->b[i] - (int16_t)pwt->b[i]) >> 1);
    }
}

void helper_lsx_vhalfd_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = (int16_t)(((int32_t)pws->h[i] - (int32_t)pwt->h[i]) >> 1);
    }
}


void helper_lsx_vhalfd_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = (int32_t)(((int64_t)pws->w[i] - (int64_t)pwt->w[i]) >> 1);
    }
}


void helper_lsx_vhalfd_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64; i++) {
        pwd->d[i] = (int64_t)(((__int128_t)pws->d[i] - (__int128_t)pwt->d[i]) >> 1);
    }
}


void helper_lsx_vhalfd_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/8; i++) {
        pwd->b[i] = (uint8_t)(((uint16_t)(uint8_t)pws->b[i] - (uint16_t)(uint8_t)pwt->b[i]) >> 1);
    }
}

void helper_lsx_vhalfd_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = (uint16_t)(((uint32_t)(uint16_t)pws->h[i] - (uint32_t)(uint16_t)pwt->h[i]) >> 1);
    }
}


void helper_lsx_vhalfd_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = (uint32_t)(((uint64_t)(uint32_t)pws->w[i] - (uint64_t)(uint32_t)pwt->w[i]) >> 1);
    }
}

void helper_lsx_vhalfd_du(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64; i++) {
        pwd->d[i] = (uint64_t)(((__uint128_t)(uint64_t)pws->d[i] - (__uint128_t)(uint64_t)pwt->d[i]) >> 1);
    }
}

void helper_lsx_xvhalfd_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/8; i++) {
        pwd->b[i] = (int8_t)(((int16_t)pws->b[i] - (int16_t)pwt->b[i]) >> 1);
    }
}

void helper_lsx_xvhalfd_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/16; i++) {
        pwd->h[i] = (int16_t)(((int32_t)pws->h[i] - (int32_t)pwt->h[i]) >> 1);
    }
}


void helper_lsx_xvhalfd_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/32; i++) {
        pwd->w[i] = (int32_t)(((int64_t)pws->w[i] - (int64_t)pwt->w[i]) >> 1);
    }
}


void helper_lsx_xvhalfd_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/64; i++) {
        pwd->d[i] =(int64_t)(((__int128_t)pws->d[i] - (__int128_t)pwt->d[i]) >> 1);
    }
}


void helper_lsx_xvhalfd_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/8; i++) {
        pwd->b[i] = (uint8_t)(((uint16_t)(uint8_t)pws->b[i] - (uint16_t)(uint8_t)pwt->b[i]) >> 1);
    }
}

void helper_lsx_xvhalfd_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/16; i++) {
        pwd->h[i] = (uint16_t)(((uint32_t)(uint16_t)pws->h[i] - (uint32_t)(uint16_t)pwt->h[i]) >> 1);
    }
}


void helper_lsx_xvhalfd_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/32; i++) {
        pwd->w[i] = (uint32_t)(((uint64_t)(uint32_t)pws->w[i] - (uint64_t)(uint32_t)pwt->w[i]) >> 1);
    }
}

void helper_lsx_xvhalfd_du(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/64; i++) {
        pwd->d[i] = (uint64_t)(((__uint128_t)(uint64_t)pws->d[i] - (__uint128_t)(uint64_t)pwt->d[i]) >> 1);
    }
}

static inline int64_t lsx_asub_s_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    /* signed compare */
    return (arg1 < arg2) ?
        (uint64_t)(arg2 - arg1) : (uint64_t)(arg1 - arg2);
}


static inline uint64_t lsx_asub_u_df(uint32_t df, uint64_t arg1, uint64_t arg2)
{
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    uint64_t u_arg2 = UNSIGNED(arg2, df);
    /* unsigned compare */
    return (u_arg1 < u_arg2) ?
        (uint64_t)(u_arg2 - u_arg1) : (uint64_t)(u_arg1 - u_arg2);
}

void helper_lsx_vsadw_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = lsx_asub_s_df(DF_BYTE,pws->b[2*i]  ,pwt->b[2*i])
                  + lsx_asub_s_df(DF_BYTE,pws->b[2*i+1],pwt->b[2*i+1]) ;
    }
}


void helper_lsx_vsadw_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = lsx_asub_s_df(DF_HALF,pws->h[2*i]  ,pwt->h[2*i])
                  + lsx_asub_s_df(DF_HALF,pws->h[2*i+1],pwt->h[2*i+1]) ;
    }
}


void helper_lsx_vsadw_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64; i++) {
        pwd->d[i] = lsx_asub_s_df(DF_WORD,pws->w[2*i]  ,pwt->w[2*i])
                  + lsx_asub_s_df(DF_WORD,pws->w[2*i+1],pwt->w[2*i+1]) ;
    }
}

void helper_lsx_vsadw_h_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = lsx_asub_u_df(DF_BYTE,pws->b[2*i]  ,pwt->b[2*i])
                  + lsx_asub_u_df(DF_BYTE,pws->b[2*i+1],pwt->b[2*i+1]) ;
    }
}


void helper_lsx_vsadw_w_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = lsx_asub_u_df(DF_HALF,pws->h[2*i]  ,pwt->h[2*i])
                  + lsx_asub_u_df(DF_HALF,pws->h[2*i+1],pwt->h[2*i+1]) ;
    }
}


void helper_lsx_vsadw_d_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64; i++) {
        pwd->d[i] = lsx_asub_u_df(DF_WORD,pws->w[2*i]  ,pwt->w[2*i])
                  + lsx_asub_u_df(DF_WORD,pws->w[2*i+1],pwt->w[2*i+1]) ;
    }
}

void helper_lsx_vaccsadw_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = lsx_asub_s_df(DF_BYTE,pws->b[2*i]  ,pwt->b[2*i])
                  + lsx_asub_s_df(DF_BYTE,pws->b[2*i+1],pwt->b[2*i+1])
                  + pwd->h[i];
    }
}


void helper_lsx_vaccsadw_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = lsx_asub_s_df(DF_HALF,pws->h[2*i]  ,pwt->h[2*i])
                  + lsx_asub_s_df(DF_HALF,pws->h[2*i+1],pwt->h[2*i+1])
                  + pwd->w[i];
    }
}


void helper_lsx_vaccsadw_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64; i++) {
        pwd->d[i] = lsx_asub_s_df(DF_WORD,pws->w[2*i]  ,pwt->w[2*i])
                  + lsx_asub_s_df(DF_WORD,pws->w[2*i+1],pwt->w[2*i+1])
                  + pwd->d[i];
    }
}

void helper_lsx_vaccsadw_h_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = lsx_asub_u_df(DF_BYTE,pws->b[2*i]  ,pwt->b[2*i])
                  + lsx_asub_u_df(DF_BYTE,pws->b[2*i+1],pwt->b[2*i+1])
                  + pwd->h[i];
    }
}


void helper_lsx_vaccsadw_w_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = lsx_asub_u_df(DF_HALF,pws->h[2*i]  ,pwt->h[2*i])
                  + lsx_asub_u_df(DF_HALF,pws->h[2*i+1],pwt->h[2*i+1])
                  + pwd->w[i];
    }
}


void helper_lsx_vaccsadw_d_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64; i++) {
        pwd->d[i] = lsx_asub_u_df(DF_WORD,pws->w[2*i]  ,pwt->w[2*i])
                  + lsx_asub_u_df(DF_WORD,pws->w[2*i+1],pwt->w[2*i+1])
                  + pwd->d[i];
    }
}

void helper_lsx_xvsadw_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/16; i++) {
        pwd->h[i] = lsx_asub_s_df(DF_BYTE,pws->b[2*i]  ,pwt->b[2*i])
                  + lsx_asub_s_df(DF_BYTE,pws->b[2*i+1],pwt->b[2*i+1]) ;
    }
}


void helper_lsx_xvsadw_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/32; i++) {
        pwd->w[i] = lsx_asub_s_df(DF_HALF,pws->h[2*i]  ,pwt->h[2*i])
                  + lsx_asub_s_df(DF_HALF,pws->h[2*i+1],pwt->h[2*i+1]) ;
    }
}


void helper_lsx_xvsadw_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/64; i++) {
        pwd->d[i] = lsx_asub_s_df(DF_WORD,pws->w[2*i]  ,pwt->w[2*i])
                  + lsx_asub_s_df(DF_WORD,pws->w[2*i+1],pwt->w[2*i+1]) ;
    }
}

void helper_lsx_xvsadw_h_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/16; i++) {
        pwd->h[i] = lsx_asub_u_df(DF_BYTE,pws->b[2*i]  ,pwt->b[2*i])
                  + lsx_asub_u_df(DF_BYTE,pws->b[2*i+1],pwt->b[2*i+1]) ;
    }
}


void helper_lsx_xvsadw_w_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/32; i++) {
        pwd->w[i] = lsx_asub_u_df(DF_HALF,pws->h[2*i]  ,pwt->h[2*i])
                  + lsx_asub_u_df(DF_HALF,pws->h[2*i+1],pwt->h[2*i+1]) ;
    }
}


void helper_lsx_xvsadw_d_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/64; i++) {
        pwd->d[i] = lsx_asub_u_df(DF_WORD,pws->w[2*i]  ,pwt->w[2*i])
                  + lsx_asub_u_df(DF_WORD,pws->w[2*i+1],pwt->w[2*i+1]) ;
    }
}

void helper_lsx_xvaccsadw_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/16; i++) {
        pwd->h[i] = lsx_asub_s_df(DF_BYTE,pws->b[2*i]  ,pwt->b[2*i])
                  + lsx_asub_s_df(DF_BYTE,pws->b[2*i+1],pwt->b[2*i+1])
                  + pwd->h[i];
    }
}


void helper_lsx_xvaccsadw_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/32; i++) {
        pwd->w[i] = lsx_asub_s_df(DF_HALF,pws->h[2*i]  ,pwt->h[2*i])
                  + lsx_asub_s_df(DF_HALF,pws->h[2*i+1],pwt->h[2*i+1])
                  + pwd->w[i];
    }
}


void helper_lsx_xvaccsadw_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/64; i++) {
        pwd->d[i] = lsx_asub_s_df(DF_WORD,pws->w[2*i]  ,pwt->w[2*i])
                  + lsx_asub_s_df(DF_WORD,pws->w[2*i+1],pwt->w[2*i+1])
                  + pwd->d[i];
    }
}

void helper_lsx_xvaccsadw_h_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/16; i++) {
        pwd->h[i] = lsx_asub_u_df(DF_BYTE,pws->b[2*i]  ,pwt->b[2*i])
                  + lsx_asub_u_df(DF_BYTE,pws->b[2*i+1],pwt->b[2*i+1])
                  + pwd->h[i];
    }
}


void helper_lsx_xvaccsadw_w_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/32; i++) {
        pwd->w[i] = lsx_asub_u_df(DF_HALF,pws->h[2*i]  ,pwt->h[2*i])
                  + lsx_asub_u_df(DF_HALF,pws->h[2*i+1],pwt->h[2*i+1])
                  + pwd->w[i];
    }
}


void helper_lsx_xvaccsadw_d_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/64; i++) {
        pwd->d[i] = lsx_asub_u_df(DF_WORD,pws->w[2*i]  ,pwt->w[2*i])
                  + lsx_asub_u_df(DF_WORD,pws->w[2*i+1],pwt->w[2*i+1])
                  + pwd->d[i];
    }
}


static inline int64_t lsx_srl_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    int32_t b_arg2 = BIT_POSITION(arg2, df);
    return u_arg1 >> b_arg2;
}

static inline int64_t lsx_sra_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    int32_t b_arg2 = BIT_POSITION(arg2, df);
    return arg1 >> b_arg2;
}

void helper_lsx_vsrln_b_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    for (i = 0; i < 128/16; i++) {
        shift_temp  = pwt->h[i] & 0xf;
        pwd->b[i]   = (uint16_t)pws->h[i] >> shift_temp;
    }
    pwd->d[1] = 0;
}


void helper_lsx_vsrln_h_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    for (i = 0; i < 128/32; i++) {
        shift_temp  = pwt->w[i] & 0x1f;
        pwd->h[i]   = (uint32_t)pws->w[i] >> shift_temp;
    }
    pwd->d[1] = 0;
}


void helper_lsx_vsrln_w_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    long long shift_temp ;
    for (i = 0; i < 128/64; i++) {
        shift_temp  = pwt->d[i] & 0x3f;
        pwd->w[i]   = (uint64_t)pws->d[i] >> shift_temp;
    }
    pwd->d[1] = 0;
}

void helper_lsx_vsran_b_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    for (i = 0; i < 128/16; i++) {
        shift_temp  = pwt->h[i] & 0xf;
        pwd->b[i]   = (int8_t)((int64_t)pws->h[i] >> shift_temp);
    }
    pwd->d[1] = 0;
}


void helper_lsx_vsran_h_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    for (i = 0; i < 128/32; i++) {
        shift_temp  = pwt->w[i] & 0x1f;
        pwd->h[i]   = (int16_t)((int64_t)pws->w[i] >> shift_temp);
    }
    pwd->d[1] = 0;
}


void helper_lsx_vsran_w_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    long long shift_temp ;
    for (i = 0; i < 128/64; i++) {
        shift_temp  = pwt->d[i] & 0x3f;
        pwd->w[i]   = (int32_t)((int64_t)pws->d[i] >> shift_temp);
    }
    pwd->d[1] = 0;
}

void helper_lsx_xvsrln_b_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    for (i = 0; i < 128/16; i++) {
        shift_temp  = pwt->h[i] & 0xf;
        pwd->b[i]   = (uint16_t)pws->h[i] >> shift_temp;
        shift_temp  = pwt->h[i+8] & 0xf;
        pwd->b[i+16]   = (uint16_t)pws->h[i+8] >> shift_temp;
    }
    pwd->d[1] = 0;
    pwd->d[3] = 0;
}


void helper_lsx_xvsrln_h_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    for (i = 0; i < 128/32; i++) {
        shift_temp  = pwt->w[i] & 0x1f;
        pwd->h[i]   = (uint32_t)pws->w[i] >> shift_temp;
        shift_temp  = pwt->w[i+4] & 0x1f;
        pwd->h[i+8]   = (uint32_t)pws->w[i+4] >> shift_temp;
    }
    pwd->d[1] = 0;
    pwd->d[3] = 0;
}


void helper_lsx_xvsrln_w_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    long long shift_temp ;
    for (i = 0; i < 128/64; i++) {
        shift_temp  = pwt->d[i] & 0x3f;
        pwd->w[i]   = (uint64_t)pws->d[i] >> shift_temp;
        shift_temp  = pwt->d[i+2] & 0x3f;
        pwd->w[i+4]   = (uint64_t)pws->d[i+2] >> shift_temp;
    }
    pwd->d[1] = 0;
    pwd->d[3] = 0;
}

void helper_lsx_xvsran_b_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    for (i = 0; i < 128/16; i++) {
        shift_temp  = pwt->h[i] & 0xf;
        pwd->b[i]   = (int8_t)((int64_t)pws->h[i] >> shift_temp);
        shift_temp  = pwt->h[i+8] & 0xf;
        pwd->b[i+16]   = (int8_t)((int64_t)pws->h[i+8] >> shift_temp);
    }
    pwd->d[1] = 0;
    pwd->d[3] = 0;
}


void helper_lsx_xvsran_h_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    for (i = 0; i < 128/32; i++) {
        shift_temp  = pwt->w[i] & 0x1f;
        pwd->h[i]   = (int16_t)((int64_t)pws->w[i] >> shift_temp);
        shift_temp  = pwt->w[i+4] & 0x1f;
        pwd->h[i+8]   = (int16_t)((int64_t)pws->w[i+4] >> shift_temp);
    }
    pwd->d[1] = 0;
    pwd->d[3] = 0;
}


void helper_lsx_xvsran_w_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    long long shift_temp ;
    for (i = 0; i < 128/64; i++) {
        shift_temp  = pwt->d[i] & 0x3f;
        pwd->w[i]   = (int32_t)((int64_t)pws->d[i] >> shift_temp);
        shift_temp  = pwt->d[i+2] & 0x3f;
        pwd->w[i+4]   = (int32_t)((int64_t)pws->d[i+2] >> shift_temp);
    }
    pwd->d[1] = 0;
    pwd->d[3] = 0;
}


static inline int64_t lsx_srlr_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    int32_t b_arg2 = BIT_POSITION(arg2, df);
    if (b_arg2 == 0) {
        return u_arg1;
    } else {
        uint64_t r_bit = (u_arg1 >> (b_arg2 - 1)) & 1;
        return (u_arg1 >> b_arg2) + r_bit;
    }
}
static inline int64_t lsx_srar_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    int32_t b_arg2 = BIT_POSITION(arg2, df);
    if (b_arg2 == 0) {
        return arg1;
    } else {
        int64_t r_bit = (arg1 >> (b_arg2 - 1)) & 1;
        return (arg1 >> b_arg2) + r_bit;
    }
}

static inline int64_t lsx_sat_s_df(uint32_t df, int64_t arg, uint32_t m)
{
    return arg < M_MIN_INT(m + 1) ? M_MIN_INT(m + 1) :
                                    arg > M_MAX_INT(m + 1) ? M_MAX_INT(m + 1) :
                                                             arg;
}

static inline int64_t lsx_sat_u_df(uint32_t df, int64_t arg, uint32_t m)
{
    uint64_t u_arg = UNSIGNED(arg, df);
    return  u_arg < M_MAX_UINT(m + 1) ? u_arg :
                                        M_MAX_UINT(m + 1);
}

static inline int64_t lsx_sat_s_128(__int128_t arg, uint32_t m)
{
    return arg < M_MIN_INT(m + 1) ? M_MIN_INT(m + 1) :
                                    arg > M_MAX_INT(m + 1) ? M_MAX_INT(m + 1) :
                                                             arg;
}

static inline int64_t lsx_sat_s_128u(__uint128_t u_arg, uint32_t m)
{
    return  u_arg < M_MAX_UINT(m) ? u_arg :
                                    M_MAX_UINT(m);
}


static inline int64_t lsx_sat_u_128(__uint128_t u_arg, uint32_t m)
{
    return  u_arg < M_MAX_UINT(m + 1) ? u_arg :
                                        M_MAX_UINT(m + 1);
}



void helper_lsx_vsrlrn_b_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    for (i = 0; i < 128/16; i++) {
        shift_temp  = pwt->h[i] % 16;
        pwd->b[i]   = lsx_srlr_df(DF_HALF,(uint16_t)pws->h[i],shift_temp);
    }
    pwd->d[1] = 0;
}

void helper_lsx_vsrlrn_h_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    for (i = 0; i < 128/32; i++) {
        shift_temp  = pwt->w[i] % 32;
        pwd->h[i]   = lsx_srlr_df(DF_WORD,(uint32_t)pws->w[i],shift_temp);
    }
    pwd->d[1] = 0;
}


void helper_lsx_vsrlrn_w_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    for (i = 0; i < 128/64; i++) {
        shift_temp  = pwt->d[i] % 64;
        pwd->w[i]   = lsx_srlr_df(DF_DOUBLE,(uint64_t)pws->d[i],shift_temp);
    }
    pwd->d[1] = 0;
}

void helper_lsx_vsrarn_b_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    for (i = 0; i < 128/16; i++) {
        shift_temp  = pwt->h[i] % 16;
        pwd->b[i]   = lsx_srar_df(DF_HALF,(int64_t)pws->h[i],shift_temp);
    }
    pwd->d[1] = 0;
}

void helper_lsx_vsrarn_h_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    for (i = 0; i < 128/32; i++) {
        shift_temp  = pwt->w[i] % 32;
        pwd->h[i]   = lsx_srar_df(DF_WORD,(int64_t)pws->w[i],shift_temp);
    }
    pwd->d[1] = 0;
}


void helper_lsx_vsrarn_w_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    for (i = 0; i < 128/64; i++) {
        shift_temp  = pwt->d[i] % 64;
        pwd->w[i]   = lsx_srar_df(DF_DOUBLE,(int64_t)pws->d[i],shift_temp);
    }
    pwd->d[1] = 0;
}

void helper_lsx_vssrln_b_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    int64_t temp_result;
    for (i = 0; i < 128/16; i++) {
        shift_temp  = pwt->h[i] % 16;
        temp_result = lsx_srl_df(DF_HALF,(uint16_t)pws->h[i],shift_temp);
        pwd->b[i]   = lsx_sat_s_df(DF_HALF,temp_result,7);
    }
    pwd->d[1] = 0;
}


void helper_lsx_vssrln_h_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    int64_t temp_result;
    for (i = 0; i < 128/32; i++) {
        shift_temp  = pwt->w[i] % 32;
        temp_result = lsx_srl_df(DF_WORD,(uint32_t)pws->w[i],shift_temp);
        pwd->h[i]   = lsx_sat_s_df(DF_WORD,temp_result,15);
    }
    pwd->d[1] = 0;
}


void helper_lsx_vssrln_w_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    int64_t temp_result;
    for (i = 0; i < 128/64; i++) {
        shift_temp  = pwt->d[i] % 64;
        temp_result = lsx_srl_df(DF_DOUBLE,(uint64_t)pws->d[i],shift_temp);
        pwd->w[i]   = lsx_sat_s_df(DF_DOUBLE,temp_result,31);
    }
    pwd->d[1] = 0;
}

void helper_lsx_vssran_b_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    int64_t temp_result;
    for (i = 0; i < 128/16; i++) {
        shift_temp  = pwt->h[i] % 16;
        temp_result = lsx_sra_df(DF_HALF,(int64_t)pws->h[i],shift_temp);
        pwd->b[i]   = lsx_sat_s_df(DF_HALF,temp_result,7);
    }
    pwd->d[1] = 0;
}


void helper_lsx_vssran_h_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int64_t shift_temp ;
    int64_t temp_result;
    for (i = 0; i < 128/32; i++) {
        shift_temp  = pwt->w[i] % 32;
        temp_result = lsx_sra_df(DF_WORD,(int64_t)pws->w[i],shift_temp);
        pwd->h[i]   = lsx_sat_s_df(DF_WORD,temp_result,15);
    }
    pwd->d[1] = 0;
}


void helper_lsx_vssran_w_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    int64_t temp_result;
    for (i = 0; i < 128/64; i++) {
        shift_temp  = pwt->d[i] % 64;
        temp_result = lsx_sra_df(DF_DOUBLE,(int64_t)pws->d[i],shift_temp);
        pwd->w[i]   = lsx_sat_s_df(DF_DOUBLE,temp_result,31);
    }
    pwd->d[1] = 0;
}

void helper_lsx_vssrlrn_b_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    int64_t temp_result;
    for (i = 0; i < 128/16; i++) {
        shift_temp  = pwt->h[i] % 16;
        temp_result = lsx_srlr_df(DF_HALF,(uint16_t)pws->h[i],shift_temp);
        pwd->b[i]   = lsx_sat_s_df(DF_HALF,temp_result,7);
    }
    pwd->d[1] = 0;
}


void helper_lsx_vssrlrn_h_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    int64_t temp_result;
    for (i = 0; i < 128/32; i++) {
        shift_temp  = pwt->w[i] % 32;
        temp_result = lsx_srlr_df(DF_WORD,(uint32_t)pws->w[i],shift_temp);
        pwd->h[i]   = lsx_sat_s_df(DF_WORD,temp_result,15);
    }
    pwd->d[1] = 0;
}


void helper_lsx_vssrlrn_w_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    int64_t temp_result;
    for (i = 0; i < 128/64; i++) {
        shift_temp  = pwt->d[i] % 64;
        temp_result = lsx_srlr_df(DF_DOUBLE,(uint64_t)pws->d[i],shift_temp);
        pwd->w[i]   = lsx_sat_s_df(DF_DOUBLE,temp_result,31);
    }
    pwd->d[1] = 0;
}

void helper_lsx_vssrarn_b_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    int64_t temp_result;
    for (i = 0; i < 128/16; i++) {
        shift_temp  = pwt->h[i] % 16;
        temp_result = lsx_srar_df(DF_HALF,(int64_t)pws->h[i],shift_temp);
        pwd->b[i]   = lsx_sat_s_df(DF_HALF,temp_result,7);
    }
    pwd->d[1] = 0;
}


void helper_lsx_vssrarn_h_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    int64_t temp_result;
    for (i = 0; i < 128/32; i++) {
        shift_temp  = pwt->w[i] % 32;
        temp_result = lsx_srar_df(DF_WORD,(int64_t)pws->w[i],shift_temp);
        pwd->h[i]   = lsx_sat_s_df(DF_WORD,temp_result,15);
    }
    pwd->d[1] = 0;
}


void helper_lsx_vssrarn_w_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    int64_t temp_result;
    for (i = 0; i < 128/64; i++) {
        shift_temp  = pwt->d[i] % 64;
        temp_result = lsx_srar_df(DF_DOUBLE,(int64_t)pws->d[i],shift_temp);
        pwd->w[i]   = lsx_sat_s_df(DF_DOUBLE,temp_result,31);
    }
    pwd->d[1] = 0;
}

void helper_lsx_vssrln_bu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    uint64_t temp_result;
    for (i = 0; i < 128/16; i++) {
        shift_temp  = pwt->h[i] % 16;
        temp_result = lsx_srl_df(DF_HALF,(uint16_t)pws->h[i],shift_temp);
        pwd->b[i]   = lsx_sat_u_df(DF_HALF,temp_result,7);
    }
    pwd->d[1] = 0;
}


void helper_lsx_vssrln_hu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    uint64_t temp_result;
    for (i = 0; i < 128/32; i++) {
        shift_temp  = pwt->w[i] % 32;
        temp_result = lsx_srl_df(DF_WORD,(uint32_t)pws->w[i],shift_temp);
        pwd->h[i]   = lsx_sat_u_df(DF_WORD,temp_result,15);
    }
    pwd->d[1] = 0;
}


void helper_lsx_vssrln_wu_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    uint64_t temp_result;
    for (i = 0; i < 128/64; i++) {
        shift_temp  = pwt->d[i] % 64;
        temp_result = lsx_srl_df(DF_DOUBLE,(uint64_t)pws->d[i],shift_temp);
        pwd->w[i]   = lsx_sat_u_df(DF_DOUBLE,temp_result,31);
    }
    pwd->d[1] = 0;
}

void helper_lsx_vssran_bu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    int64_t temp_result;
    for (i = 0; i < 128/16; i++) {
        shift_temp  = pwt->h[i] % 16;
        temp_result = lsx_sra_df(DF_HALF,(int64_t)pws->h[i],shift_temp);
        pwd->b[i]   = lsx_sat_u_df(DF_HALF,temp_result,7);
    }
    pwd->d[1] = 0;
}


void helper_lsx_vssran_hu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    int64_t temp_result;
    for (i = 0; i < 128/32; i++) {
        shift_temp  = pwt->w[i] % 32;
        temp_result = lsx_sra_df(DF_WORD,(int64_t)pws->w[i],shift_temp);
        pwd->h[i]   = lsx_sat_u_df(DF_WORD,temp_result,15);
    }
    pwd->d[1] = 0;
}


void helper_lsx_vssran_wu_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    int64_t temp_result;
    for (i = 0; i < 128/64; i++) {
        shift_temp  = pwt->d[i] % 64;
        temp_result = lsx_sra_df(DF_DOUBLE,(int64_t)pws->d[i],shift_temp);
        pwd->w[i]   = lsx_sat_u_df(DF_DOUBLE,temp_result,31);
    }
    pwd->d[1] = 0;
}

void helper_lsx_vssrlrn_bu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    uint64_t temp_result;
    for (i = 0; i < 128/16; i++) {
        shift_temp  = pwt->h[i] % 16;
        temp_result = lsx_srlr_df(DF_HALF,(uint16_t)pws->h[i],shift_temp);
        pwd->b[i]   = lsx_sat_u_df(DF_HALF,temp_result,7);
    }
    pwd->d[1] = 0;
}


void helper_lsx_vssrlrn_hu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    uint64_t temp_result;
    for (i = 0; i < 128/32; i++) {
        shift_temp  = pwt->w[i] % 32;
        temp_result = lsx_srlr_df(DF_WORD,(uint32_t)pws->w[i],shift_temp);
        pwd->h[i]   = lsx_sat_u_df(DF_WORD,temp_result,15);
    }
    pwd->d[1] = 0;
}


void helper_lsx_vssrlrn_wu_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    uint64_t temp_result;
    for (i = 0; i < 128/64; i++) {
        shift_temp  = pwt->d[i] % 64;
        temp_result = lsx_srlr_df(DF_DOUBLE,(uint64_t)pws->d[i],shift_temp);
        pwd->w[i]   = lsx_sat_u_df(DF_DOUBLE,temp_result,31);
    }
    pwd->d[1] = 0;
}

void helper_lsx_vssrarn_bu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    int64_t temp_result;
    for (i = 0; i < 128/16; i++) {
        shift_temp  = pwt->h[i] % 16;
        temp_result = lsx_srar_df(DF_HALF,(int64_t)pws->h[i],shift_temp);
        pwd->b[i]   = lsx_sat_u_df(DF_HALF,temp_result,7);
    }
    pwd->d[1] = 0;
}


void helper_lsx_vssrarn_hu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    int64_t temp_result;
    for (i = 0; i < 128/32; i++) {
        shift_temp  = pwt->w[i] % 32;
        temp_result = lsx_srar_df(DF_WORD,(int64_t)pws->w[i],shift_temp);
        pwd->h[i]   = lsx_sat_u_df(DF_WORD,temp_result,15);
    }
    pwd->d[1] = 0;
}


void helper_lsx_vssrarn_wu_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    int64_t temp_result;
    for (i = 0; i < 128/64; i++) {
        shift_temp  = pwt->d[i] % 64;
        temp_result = lsx_srar_df(DF_DOUBLE,(int64_t)pws->d[i],shift_temp);
        pwd->w[i]   = lsx_sat_u_df(DF_DOUBLE,temp_result,31);
    }
    pwd->d[1] = 0;
}

void helper_lsx_xvsrlrn_b_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    for (i = 0; i < 256/16; i++) {
        shift_temp  = pwt->h[i] % 16;
        pwd->b[i]   = lsx_srlr_df(DF_HALF,(uint16_t)pws->h[i],shift_temp);
    }
    pwd->q[1] = 0;
}

void helper_lsx_xvsrlrn_h_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    for (i = 0; i < 256/32; i++) {
        shift_temp  = pwt->w[i] % 32;
        pwd->h[i]   = lsx_srlr_df(DF_WORD,(uint32_t)pws->w[i],shift_temp);
    }
    pwd->q[1] = 0;
}


void helper_lsx_xvsrlrn_w_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    for (i = 0; i < 256/64; i++) {
        shift_temp  = pwt->d[i] % 64;
        pwd->w[i]   = lsx_srlr_df(DF_DOUBLE,(uint64_t)pws->d[i],shift_temp);
    }
    pwd->q[1] = 0;
}

void helper_lsx_xvsrarn_b_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    for (i = 0; i < 256/16; i++) {
        shift_temp  = pwt->h[i] % 16;
        pwd->b[i]   = lsx_srar_df(DF_HALF,(int64_t)pws->h[i],shift_temp);
    }
    pwd->q[1] = 0;
}

void helper_lsx_xvsrarn_h_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    for (i = 0; i < 256/32; i++) {
        shift_temp  = pwt->w[i] % 32;
        pwd->h[i]   = lsx_srar_df(DF_WORD,(int64_t)pws->w[i],shift_temp);
    }
    pwd->q[1] = 0;
}


void helper_lsx_xvsrarn_w_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    for (i = 0; i < 256/64; i++) {
        shift_temp  = pwt->d[i] % 64;
        pwd->w[i]   = lsx_srar_df(DF_DOUBLE,(int64_t)pws->d[i],shift_temp);
    }
    pwd->q[1] = 0;
}

void helper_lsx_xvssrln_b_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    int64_t temp_result;
    for (i = 0; i < 256/16; i++) {
        shift_temp  = pwt->h[i] % 16;
        temp_result = lsx_srl_df(DF_HALF,(uint16_t)pws->h[i],shift_temp);
        pwd->b[i]   = lsx_sat_s_df(DF_HALF,temp_result,7);
    }
    pwd->q[1] = 0;
}


void helper_lsx_xvssrln_h_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    int64_t temp_result;
    for (i = 0; i < 256/32; i++) {
        shift_temp  = pwt->w[i] % 32;
        temp_result = lsx_srl_df(DF_WORD,(uint32_t)pws->w[i],shift_temp);
        pwd->h[i]   = lsx_sat_s_df(DF_WORD,temp_result,15);
    }
    pwd->q[1] = 0;
}


void helper_lsx_xvssrln_w_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    int64_t temp_result;
    for (i = 0; i < 256/64; i++) {
        shift_temp  = pwt->d[i] % 64;
        temp_result = lsx_srl_df(DF_DOUBLE,(uint64_t)pws->d[i],shift_temp);
        pwd->w[i]   = lsx_sat_s_df(DF_DOUBLE,temp_result,31);
    }
    pwd->q[1] = 0;
}

void helper_lsx_xvssran_b_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    int64_t temp_result;
    for (i = 0; i < 256/16; i++) {
        shift_temp  = pwt->h[i] % 16;
        temp_result = lsx_sra_df(DF_HALF,(int64_t)pws->h[i],shift_temp);
        pwd->b[i]   = lsx_sat_s_df(DF_HALF,temp_result,7);
    }
    pwd->q[1] = 0;
}


void helper_lsx_xvssran_h_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    int64_t temp_result;
    for (i = 0; i < 256/32; i++) {
        shift_temp  = pwt->w[i] % 32;
        temp_result = lsx_sra_df(DF_WORD,(int64_t)pws->w[i],shift_temp);
        pwd->h[i]   = lsx_sat_s_df(DF_WORD,temp_result,15);
    }
    pwd->q[1] = 0;
}


void helper_lsx_xvssran_w_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    int64_t temp_result;
    for (i = 0; i < 256/64; i++) {
        shift_temp  = pwt->d[i] % 64;
        temp_result = lsx_sra_df(DF_DOUBLE,(int64_t)pws->d[i],shift_temp);
        pwd->w[i]   = lsx_sat_s_df(DF_DOUBLE,temp_result,31);
    }
    pwd->q[1] = 0;
}

void helper_lsx_xvssrlrn_b_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    int64_t temp_result;
    for (i = 0; i < 256/16; i++) {
        shift_temp  = pwt->h[i] % 16;
        temp_result = lsx_srlr_df(DF_HALF,(uint16_t)pws->h[i],shift_temp);
        pwd->b[i]   = lsx_sat_s_df(DF_HALF,temp_result,7);
    }
    pwd->q[1] = 0;
}


void helper_lsx_xvssrlrn_h_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    int64_t temp_result;
    for (i = 0; i < 256/32; i++) {
        shift_temp  = pwt->w[i] % 32;
        temp_result = lsx_srlr_df(DF_WORD,(uint32_t)pws->w[i],shift_temp);
        pwd->h[i]   = lsx_sat_s_df(DF_WORD,temp_result,15);
    }
    pwd->q[1] = 0;
}


void helper_lsx_xvssrlrn_w_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    int64_t temp_result;
    for (i = 0; i < 256/64; i++) {
        shift_temp  = pwt->d[i] % 64;
        temp_result = lsx_srlr_df(DF_DOUBLE,(uint64_t)pws->d[i],shift_temp);
        pwd->w[i]   = lsx_sat_s_df(DF_DOUBLE,temp_result,31);
    }
    pwd->q[1] = 0;
}

void helper_lsx_xvssrarn_b_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    int64_t temp_result;
    for (i = 0; i < 256/16; i++) {
        shift_temp  = pwt->h[i] % 16;
        temp_result = lsx_srar_df(DF_HALF,(int64_t)pws->h[i],shift_temp);
        pwd->b[i]   = lsx_sat_s_df(DF_HALF,temp_result,7);
    }
    pwd->q[1] = 0;
}


void helper_lsx_xvssrarn_h_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    int64_t temp_result;
    for (i = 0; i < 256/32; i++) {
        shift_temp  = pwt->w[i] % 32;
        temp_result = lsx_srar_df(DF_WORD,(int64_t)pws->w[i],shift_temp);
        pwd->h[i]   = lsx_sat_s_df(DF_WORD,temp_result,15);
    }
    pwd->q[1] = 0;
}


void helper_lsx_xvssrarn_w_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    int64_t temp_result;
    for (i = 0; i < 256/64; i++) {
        shift_temp  = pwt->d[i] % 64;
        temp_result = lsx_srar_df(DF_DOUBLE,(int64_t)pws->d[i],shift_temp);
        pwd->w[i]   = lsx_sat_s_df(DF_DOUBLE,temp_result,31);
    }
    pwd->q[1] = 0;
}

void helper_lsx_xvssrln_bu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    uint64_t temp_result;
    for (i = 0; i < 256/16; i++) {
        shift_temp  = pwt->h[i] % 16;
        temp_result = lsx_srl_df(DF_HALF,(uint16_t)pws->h[i],shift_temp);
        pwd->b[i]   = lsx_sat_u_df(DF_HALF,temp_result,7);
    }
    pwd->q[1] = 0;
}


void helper_lsx_xvssrln_hu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    uint64_t temp_result;
    for (i = 0; i < 256/32; i++) {
        shift_temp  = pwt->w[i] % 32;
        temp_result = lsx_srl_df(DF_WORD,(uint32_t)pws->w[i],shift_temp);
        pwd->h[i]   = lsx_sat_u_df(DF_WORD,temp_result,15);
    }
    pwd->q[1] = 0;
}


void helper_lsx_xvssrln_wu_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    uint64_t temp_result;
    for (i = 0; i < 256/64; i++) {
        shift_temp  = pwt->d[i] % 64;
        temp_result = lsx_srl_df(DF_DOUBLE,(uint64_t)pws->d[i],shift_temp);
        pwd->w[i]   = lsx_sat_u_df(DF_DOUBLE,temp_result,31);
    }
    pwd->q[1] = 0;
}

void helper_lsx_xvssran_bu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    int64_t temp_result;
    for (i = 0; i < 256/16; i++) {
        shift_temp  = pwt->h[i] % 16;
        temp_result = lsx_sra_df(DF_HALF,(int64_t)pws->h[i],shift_temp);
        pwd->b[i]   = lsx_sat_u_df(DF_HALF,temp_result,7);
    }
    pwd->q[1] = 0;
}


void helper_lsx_xvssran_hu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    int64_t temp_result;
    for (i = 0; i < 256/32; i++) {
        shift_temp  = pwt->w[i] % 32;
        temp_result = lsx_sra_df(DF_WORD,(int64_t)pws->w[i],shift_temp);
        pwd->h[i]   = lsx_sat_u_df(DF_WORD,temp_result,15);
    }
    pwd->q[1] = 0;
}


void helper_lsx_xvssran_wu_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    int64_t temp_result;
    for (i = 0; i < 256/64; i++) {
        shift_temp  = pwt->d[i] % 64;
        temp_result = lsx_sra_df(DF_DOUBLE,(int64_t)pws->d[i],shift_temp);
        pwd->w[i]   = lsx_sat_u_df(DF_DOUBLE,temp_result,31);
    }
    pwd->q[1] = 0;
}

void helper_lsx_xvssrlrn_bu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    uint64_t temp_result;
    for (i = 0; i < 256/16; i++) {
        shift_temp  = pwt->h[i] % 16;
        temp_result = lsx_srlr_df(DF_HALF,(uint16_t)pws->h[i],shift_temp);
        pwd->b[i]   = lsx_sat_u_df(DF_HALF,temp_result,7);
    }
    pwd->q[1] = 0;
}


void helper_lsx_xvssrlrn_hu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    uint64_t temp_result;
    for (i = 0; i < 256/32; i++) {
        shift_temp  = pwt->w[i] % 32;
        temp_result = lsx_srlr_df(DF_WORD,(uint32_t)pws->w[i],shift_temp);
        pwd->h[i]   = lsx_sat_u_df(DF_WORD,temp_result,15);
    }
    pwd->q[1] = 0;
}


void helper_lsx_xvssrlrn_wu_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    uint64_t temp_result;
    for (i = 0; i < 256/64; i++) {
        shift_temp  = pwt->d[i] % 64;
        temp_result = lsx_srlr_df(DF_DOUBLE,(uint64_t)pws->d[i],shift_temp);
        pwd->w[i]   = lsx_sat_u_df(DF_DOUBLE,temp_result,31);
    }
    pwd->q[1] = 0;
}

void helper_lsx_xvssrarn_bu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    int64_t temp_result;
    for (i = 0; i < 256/16; i++) {
        shift_temp  = pwt->h[i] % 16;
        temp_result = lsx_srar_df(DF_HALF,(int64_t)pws->h[i],shift_temp);
        pwd->b[i]   = lsx_sat_u_df(DF_HALF,temp_result,7);
    }
    pwd->q[1] = 0;
}


void helper_lsx_xvssrarn_hu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    int64_t temp_result;
    for (i = 0; i < 256/32; i++) {
        shift_temp  = pwt->w[i] % 32;
        temp_result = lsx_srar_df(DF_WORD,(int64_t)pws->w[i],shift_temp);
        pwd->h[i]   = lsx_sat_u_df(DF_WORD,temp_result,15);
    }
    pwd->q[1] = 0;
}


void helper_lsx_xvssrarn_wu_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int shift_temp ;
    int64_t temp_result;
    for (i = 0; i < 256/64; i++) {
        shift_temp  = pwt->d[i] % 64;
        temp_result = lsx_srar_df(DF_DOUBLE,(int64_t)pws->d[i],shift_temp);
        pwd->w[i]   = lsx_sat_u_df(DF_DOUBLE,temp_result,31);
    }
    pwd->q[1] = 0;
}








//-----------------------{SYK code}begin------------------/

static inline int64_t lsx_sll_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    int32_t b_arg2 = BIT_POSITION(arg2, df);
    return arg1 << b_arg2;
}

void helper_lsx_vsllwil_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i;
    int shift_temp ;
    int64_t temp_result;
    shift_temp  = ui % 8;
    for (i = 0; i < 128/16; i++) {
        temp_result = lsx_sll_df(DF_HALF,(int64_t)pws->b[i],shift_temp);
        pwd->h[i]   = (int64_t)temp_result;
    }
}

void helper_lsx_vsllwil_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i;
    int shift_temp ;
    int64_t temp_result;
    shift_temp  = ui % 16;
    for (i = 0; i < 128/32; i++) {
        temp_result = lsx_sll_df(DF_WORD,(int64_t)pws->h[i],shift_temp);
        pwd->w[i]   = (int64_t)temp_result;
    }
}


void helper_lsx_vsllwil_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i;
    int shift_temp ;
    int64_t temp_result;
    shift_temp  = ui % 8;
    for (i = 0; i < 128/64; i++) {
        temp_result = lsx_sll_df(DF_DOUBLE,(int64_t)pws->b[i],shift_temp);
        pwd->d[i]   = (int64_t)temp_result;
    }
}

void helper_lsx_vsllwil_hu_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i;
    int shift_temp ;
    uint64_t temp_result;
    shift_temp  = ui % 8;
    for (i = 0; i < 128/16; i++) {
        temp_result = lsx_sll_df(DF_HALF,(uint8_t)pws->b[i],shift_temp);
        pwd->h[i]   = (uint64_t)temp_result;
    }
}

void helper_lsx_vsllwil_wu_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i;
    int shift_temp ;
    uint64_t temp_result;
    shift_temp  = ui % 16;
    for (i = 0; i < 128/32; i++) {
        temp_result = lsx_sll_df(DF_WORD,(uint16_t)pws->h[i],shift_temp);
        pwd->w[i]   = (uint64_t)temp_result;
    }
}


void helper_lsx_vsllwil_du_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i;
    int shift_temp ;
    uint64_t temp_result;
    shift_temp  = ui % 8;
    for (i = 0; i < 128/64; i++) {
        temp_result = lsx_sll_df(DF_DOUBLE,(uint32_t)pws->w[i],shift_temp);
        pwd->d[i]   = (uint64_t)temp_result;
    }
}

void helper_lsx_vextl_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    pwd->q[0] = (__int128_t)pws->d[0];
}

void helper_lsx_vextl_qu_du(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    pwd->q[0] = (uint64_t)pws->d[0];
}


void helper_lsx_vexth_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    for (i = 0; i < 128/16; i++){
        pwd->h[i] = (int16_t)pws->b[i+128/16];
    }
}

void helper_lsx_vexth_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    for (i = 0; i < 128/32; i++){
        pwd->w[i] = (int32_t)pws->h[i+128/32];
    }
}

void helper_lsx_vexth_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    for (i = 0; i < 128/64; i++){
        pwd->d[i] = (int64_t)pws->w[i+128/64];
    }
}

void helper_lsx_vexth_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    pwd->q[0] = (__int128_t)pws->d[1];
}

void helper_lsx_vexth_hu_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    for (i = 0; i < 128/16; i++){
        pwd->h[i] = (uint8_t)pws->b[i+128/16];
    }
}

void helper_lsx_vexth_wu_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    for (i = 0; i < 128/32; i++){
        pwd->w[i] = (uint16_t)pws->h[i+128/32];
    }
}

void helper_lsx_vexth_du_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    for (i = 0; i < 128/64; i++){
        pwd->d[i] = (uint32_t)pws->w[i+128/64];
    }
}

void helper_lsx_vexth_qu_du(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    pwd->q[0] = (uint64_t)pws->d[1];
}

void helper_lsx_xvsllwil_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i;
    int shift_temp ;
    int64_t temp_result;
    shift_temp  = ui % 8;
    for (i = 0; i < 256/16; i++) {
        temp_result = lsx_sll_df(DF_HALF,(int64_t)pws->b[i],shift_temp);
        pwd->h[i]   = (int64_t)temp_result;
    }
}

void helper_lsx_xvsllwil_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i;
    int shift_temp ;
    int64_t temp_result;
    shift_temp  = ui % 16;
    for (i = 0; i < 256/32; i++) {
        temp_result = lsx_sll_df(DF_WORD,(int64_t)pws->h[i],shift_temp);
        pwd->w[i]   = (int64_t)temp_result;
    }
}


void helper_lsx_xvsllwil_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i;
    int shift_temp ;
    int64_t temp_result;
    shift_temp  = ui % 8;
    for (i = 0; i < 256/64; i++) {
        temp_result = lsx_sll_df(DF_DOUBLE,(int64_t)pws->b[i],shift_temp);
        pwd->d[i]   = (int64_t)temp_result;
    }
}

void helper_lsx_xvsllwil_hu_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i;
    int shift_temp ;
    uint64_t temp_result;
    shift_temp  = ui % 8;
    for (i = 0; i < 256/16; i++) {
        temp_result = lsx_sll_df(DF_HALF,(uint8_t)pws->b[i],shift_temp);
        pwd->h[i]   = (uint64_t)temp_result;
    }
}

void helper_lsx_xvsllwil_wu_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i;
    int shift_temp ;
    uint64_t temp_result;
    shift_temp  = ui % 16;
    for (i = 0; i < 256/32; i++) {
        temp_result = lsx_sll_df(DF_WORD,(uint16_t)pws->h[i],shift_temp);
        pwd->w[i]   = (uint64_t)temp_result;
    }
}


void helper_lsx_xvsllwil_du_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i;
    int shift_temp ;
    uint64_t temp_result;
    shift_temp  = ui % 8;
    for (i = 0; i < 256/64; i++) {
        temp_result = lsx_sll_df(DF_DOUBLE,(uint32_t)pws->w[i],shift_temp);
        pwd->d[i]   = (uint64_t)temp_result;
    }
}

void helper_lsx_xvextl_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    pwd->q[0] = (__int128_t)pws->d[0];
    pwd->q[1] = (__int128_t)pws->d[1];
}

void helper_lsx_xvextl_qu_du(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    pwd->q[0] = (uint64_t)pws->d[0];
    pwd->q[1] = (uint64_t)pws->d[1];
}


void helper_lsx_xvexth_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    for (i = 0; i < 256/16; i++){
        pwd->h[i] = (int16_t)pws->b[i+256/16];
    }
}

void helper_lsx_xvexth_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    for (i = 0; i < 256/32; i++){
        pwd->w[i] = (int32_t)pws->h[i+256/32];
    }
}

void helper_lsx_xvexth_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    for (i = 0; i < 256/64; i++){
        pwd->d[i] = (int64_t)pws->w[i+256/64];
    }
}

void helper_lsx_xvexth_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    pwd->q[0] = (__int128_t)pws->d[2];
    pwd->q[1] = (__int128_t)pws->d[3];
}

void helper_lsx_xvexth_hu_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    for (i = 0; i < 256/16; i++){
        pwd->h[i] = (uint8_t)pws->b[i+256/16];
    }
}

void helper_lsx_xvexth_wu_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    for (i = 0; i < 256/32; i++){
        pwd->w[i] = (uint16_t)pws->h[i+256/32];
    }
}

void helper_lsx_xvexth_du_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    for (i = 0; i < 256/64; i++){
        pwd->d[i] = (uint32_t)pws->w[i+256/64];
    }
}

void helper_lsx_xvexth_qu_du(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    pwd->q[0] = (uint64_t)pws->d[2];
    pwd->q[1] = (uint64_t)pws->d[3];
}


static inline __int128_t lsx_round_legacy(__int128_t arg1, uint64_t arg2)
{
    if (arg2 == 0)
        return arg1;
    __uint128_t tmp  = (__uint128_t)1 << (arg2 -1);
    __int128_t temp = (arg1 + tmp) >> arg2;
    return temp << arg2;
}

static inline __int128_t lsx_round_nearest_even(__int128_t arg1, int64_t arg2)
{
    if (arg2 == 0)
        return arg1;
    __int128_t temp = arg1 >> (arg2 - 2);
    __int128_t rest_num = arg1 - (temp << (arg2 - 2));
    int lo_nz = rest_num != 0;
    int m_1_bit = (temp >> 1) % 2;
    int m_bit = (temp >> 2) % 2;
    __uint128_t round_inc;
    if (!m_bit && !m_1_bit)
        round_inc = 0;
    else if (!m_bit && m_1_bit && !lo_nz)
        round_inc = 0;
    else if (m_bit && m_1_bit && lo_nz)
        round_inc = 1;
    else if (m_bit && m_1_bit && !lo_nz)
        round_inc = 1;
    else if (         m_1_bit && lo_nz)
        round_inc = 1;
    else
    {
        round_inc = 0;
        //printf("else\n");
    }
    temp = (arg1 + (round_inc << arg2)) >> arg2;
    return temp << arg2;
}
static inline __int128_t lsx_round_nearest_even_u(__uint128_t arg1, int64_t arg2)
{
    if (arg2 == 0)
        return arg1;
    __uint128_t temp = arg1 >> (arg2 - 2);
    __uint128_t rest_num = arg1 - (temp << (arg2 - 2));
    int lo_nz = rest_num != 0;
    int m_1_bit = (temp >> 1) % 2;
    int m_bit = (temp >> 2) % 2;
    __uint128_t round_inc;
    if (!m_bit && !m_1_bit)
        round_inc = 0;
    else if (!m_bit && m_1_bit && !lo_nz)
        round_inc = 0;
    else if (m_bit && m_1_bit && lo_nz)
        round_inc = 1;
    else if (m_bit && m_1_bit && !lo_nz)
        round_inc = 1;
    else if (         m_1_bit && lo_nz)
        round_inc = 1;
    else
    {
        round_inc = 0;
        //printf("else\n");
    }
    temp = (arg1 + (round_inc << arg2)) >> arg2;
    return temp << arg2;
}


void helper_lsx_vsrlrneni_b_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg [128/8];
    uint64_t temp_v;
    for (i = 0; i < 128/16; i++){
        temp_v = (uint16_t)pws->h[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/16; i++){
        temp_v = (uint16_t)pwd->h[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+128/16] = temp_v;
    }
    for (i = 0; i < 128/8; i++) {
        pwd->b[i] = temp_reg[i];
    }
}

void helper_lsx_vsrlrneni_h_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg[128/16];
    uint64_t temp_v;
    for (i = 0; i < 128/32; i++){
        temp_v = (uint32_t)pws->w[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/32; i++){
        temp_v = (uint32_t)pwd->w[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+128/32] = temp_v;
    }
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = temp_reg[i];
    }
}


void helper_lsx_vsrlrneni_w_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg[128/32];
    uint64_t temp_v;
    for (i = 0; i < 128/64; i++){
        temp_v = (uint64_t)pws->d[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/64; i++){
        temp_v = (uint64_t)pwd->d[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+128/64] = temp_v;
    }
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = temp_reg[i];
    }

}


void helper_lsx_vsrlrneni_d_q(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    __uint128_t temp_reg[2];
    __uint128_t temp_v;
    temp_v = (__uint128_t)pws->q[0];
    if (ui > 0)
        temp_v = lsx_round_nearest_even(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[0] = temp_v;

    temp_v = (__uint128_t)pwd->q[0];
    if (ui > 0)
        temp_v = lsx_round_nearest_even(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[1] = temp_v;
    pwd->d[0] = (uint64_t)temp_reg[0];
    pwd->d[1] = (uint64_t)temp_reg[1];
}

void helper_lsx_vsrarneni_b_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg [128/8];
    int64_t temp_v;
    for (i = 0; i < 128/16; i++){
        temp_v = (int64_t)(int16_t)pws->h[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/16; i++){
        temp_v = (int64_t)(int16_t)pwd->h[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+128/16] = temp_v;
    }
    for (i = 0; i < 128/8; i++) {
        pwd->b[i] = temp_reg[i];
    }
}

void helper_lsx_vsrarneni_h_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg[128/16];
    int64_t temp_v;
    for (i = 0; i < 128/32; i++){
        temp_v = (int64_t)(int32_t)pws->w[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/32; i++){
        temp_v = (int64_t)(int32_t)pwd->w[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+128/32] = temp_v;
    }
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = temp_reg[i];
    }
}


void helper_lsx_vsrarneni_w_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg[128/32];
    int64_t temp_v;
    for (i = 0; i < 128/64; i++){
        temp_v = (int64_t)pws->d[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/64; i++){
        temp_v = (int64_t)pwd->d[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+128/64] = temp_v;
    }
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = temp_reg[i];
    }

}


void helper_lsx_vsrarneni_d_q(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    __int128_t temp_reg[2];
    __int128_t temp_v;
    temp_v = (__int128_t)pws->q[0];
    if (ui > 0)
        temp_v = lsx_round_nearest_even(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[0] = temp_v;

    temp_v = (__int128_t)pwd->q[0];
    if (ui > 0)
        temp_v = lsx_round_nearest_even(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[1] = temp_v;
    pwd->d[0] = temp_reg[0];
    pwd->d[1] = temp_reg[1];
}

void helper_lsx_vsrlni_b_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg [128/8];
    uint64_t temp_v;
    for (i = 0; i < 128/16; i++){
        temp_v = (uint16_t)pws->h[i];
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/16; i++){
        temp_v = (uint16_t)pwd->h[i];
        temp_v = temp_v >> ui;
        temp_reg[i+128/16] = temp_v;
    }
    for (i = 0; i < 128/8; i++) {
        pwd->b[i] = temp_reg[i];
    }
}

void helper_lsx_vsrlni_h_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg[128/16];
    uint64_t temp_v;
    for (i = 0; i < 128/32; i++){
        temp_v = (uint32_t)pws->w[i];
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/32; i++){
        temp_v = (uint32_t)pwd->w[i];
        temp_v = temp_v >> ui;
        temp_reg[i+128/32] = temp_v;
    }
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = temp_reg[i];
    }
}


void helper_lsx_vsrlni_w_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg[128/32];
    uint64_t temp_v;
    for (i = 0; i < 128/64; i++){
        temp_v = (uint64_t)pws->d[i];
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/64; i++){
        temp_v = (uint64_t)pwd->d[i];
        temp_v = temp_v >> ui;
        temp_reg[i+128/64] = temp_v;
    }
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = temp_reg[i];
    }

}


void helper_lsx_vsrlni_d_q(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    __uint128_t temp_reg[2];
    __uint128_t temp_v;
    temp_v = (__uint128_t)pws->q[0];
    temp_v = temp_v >> ui;
    temp_reg[0] = temp_v;

    temp_v = (__uint128_t)pwd->q[0];
    temp_v = temp_v >> ui;
    temp_reg[1] = temp_v;
    pwd->d[0] = (uint64_t)temp_reg[0];
    pwd->d[1] = (uint64_t)temp_reg[1];
}

void helper_lsx_vsrlrni_b_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg [128/8];
    uint64_t temp_v;
    for (i = 0; i < 128/16; i++){
        temp_v = (uint16_t)pws->h[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/16; i++){
        temp_v = (uint16_t)pwd->h[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+128/16] = temp_v;
    }
    for (i = 0; i < 128/8; i++) {
        pwd->b[i] = temp_reg[i];
    }
}

void helper_lsx_vsrlrni_h_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg[128/16];
    uint64_t temp_v;
    for (i = 0; i < 128/32; i++){
        temp_v = (uint32_t)pws->w[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/32; i++){
        temp_v = (uint32_t)pwd->w[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+128/32] = temp_v;
    }
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = temp_reg[i];
    }
}


void helper_lsx_vsrlrni_w_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg[128/32];
    uint64_t temp_v;
    for (i = 0; i < 128/64; i++){
        temp_v = (uint64_t)pws->d[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/64; i++){
        temp_v = (uint64_t)pwd->d[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+128/64] = temp_v;
    }
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = temp_reg[i];
    }

}


void helper_lsx_vsrlrni_d_q(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    __uint128_t temp_reg[2];
    __uint128_t temp_v;
    temp_v = (__uint128_t)pws->q[0];
    if (ui > 0)
        temp_v = lsx_round_legacy(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[0] = temp_v;

    temp_v = (__uint128_t)pwd->q[0];
    if (ui > 0)
        temp_v = lsx_round_legacy(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[1] = temp_v;
    pwd->d[0] = (uint64_t)temp_reg[0];
    pwd->d[1] = (uint64_t)temp_reg[1];
}


void helper_lsx_vssrlni_b_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg [128/8];
    uint64_t temp_v;
    for (i = 0; i < 128/16; i++){
        temp_v = (uint16_t)pws->h[i];
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/16; i++){
        temp_v = (uint16_t)pwd->h[i];
        temp_v = temp_v >> ui;
        temp_reg[i+128/16] = temp_v;
    }
    for (i = 0; i < 128/8; i++) {
        pwd->b[i] = lsx_sat_s_df(DF_HALF,temp_reg[i],7);
    }
}

void helper_lsx_vssrlni_h_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg[128/16];
    uint64_t temp_v;
    for (i = 0; i < 128/32; i++){
        temp_v = (uint32_t)pws->w[i];
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/32; i++){
        temp_v = (uint32_t)pwd->w[i];
        temp_v = temp_v >> ui;
        temp_reg[i+128/32] = temp_v;
    }
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = lsx_sat_s_df(DF_WORD,temp_reg[i],15);
    }
}


void helper_lsx_vssrlni_w_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg[128/32];
    uint64_t temp_v;
    for (i = 0; i < 128/64; i++){
        temp_v = (uint64_t)pws->d[i];
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/64; i++){
        temp_v = (uint64_t)pwd->d[i];
        temp_v = temp_v >> ui;
        temp_reg[i+128/64] = temp_v;
    }
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = lsx_sat_s_128u(temp_reg[i],31);
    }

}


void helper_lsx_vssrlni_d_q(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    __uint128_t temp_reg[2];
    __uint128_t temp_v;
    temp_v = (__uint128_t)pws->q[0];
    temp_v = temp_v >> ui;
    temp_reg[0] = temp_v;

    temp_v = (__uint128_t)pwd->q[0];
    temp_v = temp_v >> ui;
    temp_reg[1] = temp_v;
    pwd->d[0] = lsx_sat_s_128u(temp_reg[0],63);
    pwd->d[1] = lsx_sat_s_128u(temp_reg[1],63);
}


void helper_lsx_vssrlni_bu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg [128/8];
    uint64_t temp_v;
    for (i = 0; i < 128/16; i++){
        temp_v = (uint16_t)pws->h[i];
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/16; i++){
        temp_v = (uint16_t)pwd->h[i];
        temp_v = temp_v >> ui;
        temp_reg[i+128/16] = temp_v;
    }
    for (i = 0; i < 128/8; i++) {
        pwd->b[i] = lsx_sat_u_df(DF_HALF,temp_reg[i],7);
    }
}

void helper_lsx_vssrlni_hu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg[128/16];
    uint64_t temp_v;
    for (i = 0; i < 128/32; i++){
        temp_v = (uint32_t)pws->w[i];
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/32; i++){
        temp_v = (uint32_t)pwd->w[i];
        temp_v = temp_v >> ui;
        temp_reg[i+128/32] = temp_v;
    }
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = lsx_sat_u_df(DF_WORD,temp_reg[i],15);
    }
}


void helper_lsx_vssrlni_wu_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg[128/32];
    uint64_t temp_v;
    for (i = 0; i < 128/64; i++){
        temp_v = (uint64_t)pws->d[i];
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/64; i++){
        temp_v = (uint64_t)pwd->d[i];
        temp_v = temp_v >> ui;
        temp_reg[i+128/64] = temp_v;
    }
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = lsx_sat_u_128(temp_reg[i],31);
    }

}


void helper_lsx_vssrlni_du_q(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    __uint128_t temp_reg[2];
    __uint128_t temp_v;
    temp_v = (__uint128_t)pws->q[0];
    temp_v = temp_v >> ui;
    temp_reg[0] = temp_v;

    temp_v = (__uint128_t)pwd->q[0];
    temp_v = temp_v >> ui;
    temp_reg[1] = temp_v;
    pwd->d[0] = lsx_sat_u_128(temp_reg[0],63);
    pwd->d[1] = lsx_sat_u_128(temp_reg[1],63);
}

void helper_lsx_vssrlrni_b_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg [128/8];
    uint64_t temp_v;
    for (i = 0; i < 128/16; i++){
        temp_v = (uint16_t)pws->h[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/16; i++){
        temp_v = (uint16_t)pwd->h[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+128/16] = temp_v;
    }
    for (i = 0; i < 128/8; i++) {
        pwd->b[i] = lsx_sat_s_df(DF_HALF,temp_reg[i],7);
    }
}

void helper_lsx_vssrlrni_h_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg[128/16];
    uint64_t temp_v;
    for (i = 0; i < 128/32; i++){
        temp_v = (uint32_t)pws->w[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/32; i++){
        temp_v = (uint32_t)pwd->w[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+128/32] = temp_v;
    }
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = lsx_sat_s_df(DF_WORD,temp_reg[i],15);
    }
}


void helper_lsx_vssrlrni_w_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg[128/32];
    uint64_t temp_v;
    for (i = 0; i < 128/64; i++){
        temp_v = (uint64_t)pws->d[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/64; i++){
        temp_v = (uint64_t)pwd->d[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+128/64] = temp_v;
    }
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = lsx_sat_s_128u(temp_reg[i],31);
    }

}


void helper_lsx_vssrlrni_d_q(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    __uint128_t temp_reg[2];
    __uint128_t temp_v;
    temp_v = (__uint128_t)pws->q[0];
    if (ui > 0)
        temp_v = lsx_round_legacy(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[0] = temp_v;

    temp_v = (__uint128_t)pwd->q[0];
    if (ui > 0)
        temp_v = lsx_round_legacy(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[1] = temp_v;
    pwd->d[0] = lsx_sat_s_128u(temp_reg[0],63);
    pwd->d[1] = lsx_sat_s_128u(temp_reg[1],63);
}

void helper_lsx_vssrlrni_bu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg [128/8];
    uint64_t temp_v;
    for (i = 0; i < 128/16; i++){
        temp_v = (uint16_t)pws->h[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/16; i++){
        temp_v = (uint16_t)pwd->h[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+128/16] = temp_v;
    }
    for (i = 0; i < 128/8; i++) {
        pwd->b[i] = lsx_sat_u_df(DF_HALF,temp_reg[i],7);
    }
}

void helper_lsx_vssrlrni_hu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg[128/16];
    uint64_t temp_v;
    for (i = 0; i < 128/32; i++){
        temp_v = (uint32_t)pws->w[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/32; i++){
        temp_v = (uint32_t)pwd->w[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+128/32] = temp_v;
    }
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = lsx_sat_u_df(DF_WORD,temp_reg[i],15);
    }
}


void helper_lsx_vssrlrni_wu_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg[128/32];
    uint64_t temp_v;
    for (i = 0; i < 128/64; i++){
        temp_v = (uint64_t)pws->d[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/64; i++){
        temp_v = (uint64_t)pwd->d[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+128/64] = temp_v;
    }
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = lsx_sat_u_128(temp_reg[i],31);
    }

}


void helper_lsx_vssrlrni_du_q(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    __uint128_t temp_reg[2];
    __uint128_t temp_v;
    temp_v = (__uint128_t)pws->q[0];
    if (ui > 0)
        temp_v = lsx_round_legacy(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[0] = temp_v;

    temp_v = (__uint128_t)pwd->q[0];
    if (ui > 0)
        temp_v = lsx_round_legacy(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[1] = temp_v;
    pwd->d[0] = lsx_sat_u_128(temp_reg[0],63);
    pwd->d[1] = lsx_sat_u_128(temp_reg[1],63);
}

void helper_lsx_vsrani_b_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg [128/8];
    int64_t temp_v;
    for (i = 0; i < 128/16; i++){
        temp_v = (int64_t)(int16_t)pws->h[i];
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/16; i++){
        temp_v = (int64_t)(int16_t)pwd->h[i];
        temp_v = temp_v >> ui;
        temp_reg[i+128/16] = temp_v;
    }
    for (i = 0; i < 128/8; i++) {
        pwd->b[i] = temp_reg[i];
    }
}

void helper_lsx_vsrani_h_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg[128/16];
    int64_t temp_v;
    for (i = 0; i < 128/32; i++){
        temp_v = (int64_t)(int32_t)pws->w[i];
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/32; i++){
        temp_v = (int64_t)(int32_t)pwd->w[i];
        temp_v = temp_v >> ui;
        temp_reg[i+128/32] = temp_v;
    }
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = temp_reg[i];
    }
}


void helper_lsx_vsrani_w_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg[128/32];
    int64_t temp_v;
    for (i = 0; i < 128/64; i++){
        temp_v = (int64_t)pws->d[i];
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/64; i++){
        temp_v = (int64_t)pwd->d[i];
        temp_v = temp_v >> ui;
        temp_reg[i+128/64] = temp_v;
    }
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = temp_reg[i];
    }

}


void helper_lsx_vsrani_d_q(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    __int128_t temp_reg[2];
    __int128_t temp_v;
    temp_v = (__int128_t)pws->q[0];
    temp_v = temp_v >> ui;
    temp_reg[0] = temp_v;

    temp_v = (__int128_t)pwd->q[0];
    temp_v = temp_v >> ui;
    temp_reg[1] = temp_v;
    pwd->d[0] = (int64_t)temp_reg[0];
    pwd->d[1] = (int64_t)temp_reg[1];
}


void helper_lsx_vsrarni_b_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg [128/8];
    int64_t temp_v;
    for (i = 0; i < 128/16; i++){
        temp_v = (int64_t)(int16_t)pws->h[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/16; i++){
        temp_v = (int64_t)(int16_t)pwd->h[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+128/16] = temp_v;
    }
    for (i = 0; i < 128/8; i++) {
        pwd->b[i] = temp_reg[i];
    }
}

void helper_lsx_vsrarni_h_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg[128/16];
    int64_t temp_v;
    for (i = 0; i < 128/32; i++){
        temp_v = (int64_t)(int32_t)pws->w[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/32; i++){
        temp_v = (int64_t)(int32_t)pwd->w[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+128/32] = temp_v;
    }
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = temp_reg[i];
    }
}


void helper_lsx_vsrarni_w_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg[128/32];
    int64_t temp_v;
    for (i = 0; i < 128/64; i++){
        temp_v = (int64_t)pws->d[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/64; i++){
        temp_v = (int64_t)pwd->d[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+128/64] = temp_v;
    }
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = temp_reg[i];
    }

}


void helper_lsx_vsrarni_d_q(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    __int128_t temp_reg[2];
    __int128_t temp_v;
    temp_v = (__int128_t)pws->q[0];
    if (ui > 0)
        temp_v = lsx_round_legacy(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[0] = temp_v;

    temp_v = (__int128_t)pwd->q[0];
    if (ui > 0)
        temp_v = lsx_round_legacy(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[1] = temp_v;
    pwd->d[0] = temp_reg[0];
    pwd->d[1] = temp_reg[1];
}

void helper_lsx_vssrani_b_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg [128/8];
    int64_t temp_v;
    for (i = 0; i < 128/16; i++){
        temp_v = (int64_t)(int16_t)pws->h[i];
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/16; i++){
        temp_v = (int64_t)(int16_t)pwd->h[i];
        temp_v = temp_v >> ui;
        temp_reg[i+128/16] = temp_v;
    }
    for (i = 0; i < 128/8; i++) {
        pwd->b[i] = lsx_sat_s_df(DF_HALF,temp_reg[i],7);
    }
}

void helper_lsx_vssrani_h_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg[128/16];
    int64_t temp_v;
    for (i = 0; i < 128/32; i++){
        temp_v = (int64_t)(int32_t)pws->w[i];
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/32; i++){
        temp_v = (int64_t)(int32_t)pwd->w[i];
        temp_v = temp_v >> ui;
        temp_reg[i+128/32] = temp_v;
    }
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = lsx_sat_s_df(DF_WORD,temp_reg[i],15);
    }
}


void helper_lsx_vssrani_w_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg[128/32];
    int64_t temp_v;
    for (i = 0; i < 128/64; i++){
        temp_v = (int64_t)pws->d[i];
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/64; i++){
        temp_v = (int64_t)pwd->d[i];
        temp_v = temp_v >> ui;
        temp_reg[i+128/64] = temp_v;
    }
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = lsx_sat_s_128(temp_reg[i],31);
    }

}


void helper_lsx_vssrani_d_q(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    __int128_t temp_reg[2];
    __int128_t temp_v;
    temp_v = (__int128_t)pws->q[0];
    temp_v = temp_v >> ui;
    temp_reg[0] = temp_v;

    temp_v = (__uint128_t)pwd->q[0];
    temp_v = temp_v >> ui;
    temp_reg[1] = temp_v;
    pwd->d[0] = lsx_sat_s_128(temp_reg[0],63);
    pwd->d[1] = lsx_sat_s_128(temp_reg[1],63);
}

void helper_lsx_vssrani_bu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg [128/8];
    int64_t temp_v;
    for (i = 0; i < 128/16; i++){
        temp_v = (int64_t)(int16_t)pws->h[i];
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/16; i++){
        temp_v = (int64_t)(int16_t)pwd->h[i];
        temp_v = temp_v >> ui;
        temp_reg[i+128/16] = temp_v;
    }
    for (i = 0; i < 128/8; i++) {
        pwd->b[i] = lsx_sat_u_df(DF_HALF,temp_reg[i],7);
    }
}

void helper_lsx_vssrani_hu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg[128/16];
    int64_t temp_v;
    for (i = 0; i < 128/32; i++){
        temp_v = (int64_t)(int32_t)pws->w[i];
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/32; i++){
        temp_v = (int64_t)(int32_t)pwd->w[i];
        temp_v = temp_v >> ui;
        temp_reg[i+128/32] = temp_v;
    }
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = lsx_sat_u_df(DF_WORD,temp_reg[i],15);
    }
}


void helper_lsx_vssrani_wu_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg[128/32];
    int64_t temp_v;
    for (i = 0; i < 128/64; i++){
        temp_v = (int64_t)pws->d[i];
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/64; i++){
        temp_v = (int64_t)pwd->d[i];
        temp_v = temp_v >> ui;
        temp_reg[i+128/64] = temp_v;
    }
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = lsx_sat_u_128(temp_reg[i],31);
    }

}


void helper_lsx_vssrani_du_q(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    __int128_t temp_reg[2];
    __int128_t temp_v;
    temp_v = (__int128_t)pws->q[0];
    temp_v = temp_v >> ui;
    temp_reg[0] = temp_v;

    temp_v = (__uint128_t)pwd->q[0];
    temp_v = temp_v >> ui;
    temp_reg[1] = temp_v;
    pwd->d[0] = lsx_sat_u_128(temp_reg[0],63);
    pwd->d[1] = lsx_sat_u_128(temp_reg[1],63);
}

void helper_lsx_vssrarni_b_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg [128/8];
    int64_t temp_v;
    for (i = 0; i < 128/16; i++){
        temp_v = (int64_t)(int16_t)pws->h[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/16; i++){
        temp_v = (int64_t)(int16_t)pwd->h[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+128/16] = temp_v;
    }
    for (i = 0; i < 128/8; i++) {
        pwd->b[i] = lsx_sat_s_df(DF_HALF,temp_reg[i],7);
    }
}

void helper_lsx_vssrarni_h_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg[128/16];
    int64_t temp_v;
    for (i = 0; i < 128/32; i++){
        temp_v = (int64_t)(int32_t)pws->w[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/32; i++){
        temp_v = (int64_t)(int32_t)pwd->w[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+128/32] = temp_v;
    }
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = lsx_sat_s_df(DF_WORD,temp_reg[i],15);
    }
}


void helper_lsx_vssrarni_w_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg[128/32];
    int64_t temp_v;
    for (i = 0; i < 128/64; i++){
        temp_v = (int64_t)pws->d[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/64; i++){
        temp_v = (int64_t)pwd->d[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+128/64] = temp_v;
    }
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = lsx_sat_s_128(temp_reg[i],31);
    }
}


void helper_lsx_vssrarni_d_q(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    __int128_t temp_reg[2];
    __int128_t temp_v;
    temp_v = (__int128_t)pws->q[0];
    if (ui > 0)
        temp_v = lsx_round_legacy(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[0] = temp_v;

    temp_v = (__uint128_t)pwd->q[0];
    if (ui > 0)
       temp_v = lsx_round_legacy(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[1] = temp_v;
    pwd->d[0] = lsx_sat_s_128(temp_reg[0],63);
    pwd->d[1] = lsx_sat_s_128(temp_reg[1],63);
}

void helper_lsx_vssrarni_bu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg [128/8];
    int64_t temp_v;
    for (i = 0; i < 128/16; i++){
        temp_v = (int64_t)(int16_t)pws->h[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/16; i++){
        temp_v = (int64_t)(int16_t)pwd->h[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+128/16] = temp_v;
    }
    for (i = 0; i < 128/8; i++) {
        pwd->b[i] = lsx_sat_u_df(DF_HALF,temp_reg[i],7);
    }
}

void helper_lsx_vssrarni_hu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg[128/16];
    int64_t temp_v;
    for (i = 0; i < 128/32; i++){
        temp_v = (int64_t)(int32_t)pws->w[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/32; i++){
        temp_v = (int64_t)(int32_t)pwd->w[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+128/32] = temp_v;
    }
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = lsx_sat_u_df(DF_WORD,temp_reg[i],15);
    }
}


void helper_lsx_vssrarni_wu_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg[128/32];
    int64_t temp_v;
    for (i = 0; i < 128/64; i++){
        temp_v = (int64_t)pws->d[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/64; i++){
        temp_v = (int64_t)pwd->d[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+128/64] = temp_v;
    }
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = lsx_sat_u_128(temp_reg[i],31);
    }
}


void helper_lsx_vssrarni_du_q(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    __int128_t temp_reg[2];
    __int128_t temp_v;
    temp_v = (__int128_t)pws->q[0];
    if (ui > 0)
        temp_v = lsx_round_legacy(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[0] = temp_v;

    temp_v = (__uint128_t)pwd->q[0];
    if (ui > 0)
       temp_v = lsx_round_legacy(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[1] = temp_v;
    pwd->d[0] = lsx_sat_u_128(temp_reg[0],63);
    pwd->d[1] = lsx_sat_u_128(temp_reg[1],63);
}

void helper_lsx_vssrlrneni_b_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg [128/8];
    uint64_t temp_v;
    for (i = 0; i < 128/16; i++){
        temp_v = (uint16_t)pws->h[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/16; i++){
        temp_v = (uint16_t)pwd->h[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+128/16] = temp_v;
    }
    for (i = 0; i < 128/8; i++) {
        pwd->b[i] = lsx_sat_s_df(DF_HALF,temp_reg[i],7);
    }
}

void helper_lsx_vssrlrneni_h_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg[128/16];
    uint64_t temp_v;
    for (i = 0; i < 128/32; i++){
        temp_v = (uint32_t)pws->w[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/32; i++){
        temp_v = (uint32_t)pwd->w[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+128/32] = temp_v;
    }
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = lsx_sat_s_df(DF_WORD,temp_reg[i],15);
    }
}


void helper_lsx_vssrlrneni_w_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg[128/32];
    uint64_t temp_v;
    for (i = 0; i < 128/64; i++){
        temp_v = (uint64_t)pws->d[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/64; i++){
        temp_v = (uint64_t)pwd->d[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+128/64] = temp_v;
    }
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = lsx_sat_s_128u(temp_reg[i],31);
    }

}


void helper_lsx_vssrlrneni_d_q(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    __uint128_t temp_reg[2];
    __uint128_t temp_v;
    temp_v = (__uint128_t)pws->q[0];
    if (ui > 0)
        temp_v = lsx_round_nearest_even(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[0] = temp_v;

    temp_v = (__uint128_t)pwd->q[0];
    if (ui > 0)
        temp_v = lsx_round_nearest_even(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[1] = temp_v;
    pwd->d[0] = lsx_sat_s_128u(temp_reg[0],63);
    pwd->d[1] = lsx_sat_s_128u(temp_reg[1],63);
}

void helper_lsx_vssrlrneni_bu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg [128/8];
    uint64_t temp_v;
    for (i = 0; i < 128/16; i++){
        temp_v = (uint16_t)pws->h[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/16; i++){
        temp_v = (uint16_t)pwd->h[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+128/16] = temp_v;
    }
    for (i = 0; i < 128/8; i++) {
        pwd->b[i] = lsx_sat_u_df(DF_HALF,temp_reg[i],7);
    }
}

void helper_lsx_vssrlrneni_hu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg[128/16];
    uint64_t temp_v;
    for (i = 0; i < 128/32; i++){
        temp_v = (uint32_t)pws->w[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/32; i++){
        temp_v = (uint32_t)pwd->w[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+128/32] = temp_v;
    }
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = lsx_sat_u_df(DF_WORD,temp_reg[i],15);
    }
}


void helper_lsx_vssrlrneni_wu_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg[128/32];
    uint64_t temp_v;
    for (i = 0; i < 128/64; i++){
        temp_v = (uint64_t)pws->d[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/64; i++){
        temp_v = (uint64_t)pwd->d[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+128/64] = temp_v;
    }
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = lsx_sat_u_128(temp_reg[i],31);
    }

}


void helper_lsx_vssrlrneni_du_q(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    __uint128_t temp_reg[2];
    __uint128_t temp_v;
    temp_v = (__uint128_t)pws->q[0];
    if (ui > 0)
        temp_v = lsx_round_nearest_even(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[0] = temp_v;

    temp_v = (__uint128_t)pwd->q[0];
    if (ui > 0)
        temp_v = lsx_round_nearest_even(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[1] = temp_v;
    pwd->d[0] = lsx_sat_u_128(temp_reg[0],63);
    pwd->d[1] = lsx_sat_u_128(temp_reg[1],63);
}

void helper_lsx_vssrarneni_b_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg [128/8];
    int64_t temp_v;
    for (i = 0; i < 128/16; i++){
        temp_v = (int64_t)(int16_t)pws->h[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/16; i++){
        temp_v = (int64_t)(int16_t)pwd->h[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+128/16] = temp_v;
    }
    for (i = 0; i < 128/8; i++) {
        pwd->b[i] = lsx_sat_s_df(DF_HALF,temp_reg[i],7);
    }
}

void helper_lsx_vssrarneni_h_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg[128/16];
    int64_t temp_v;
    for (i = 0; i < 128/32; i++){
        temp_v = (int64_t)(int32_t)pws->w[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/32; i++){
        temp_v = (int64_t)(int32_t)pwd->w[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+128/32] = temp_v;
    }
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = lsx_sat_s_df(DF_WORD,temp_reg[i],15);
    }
}


void helper_lsx_vssrarneni_w_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg[128/32];
    int64_t temp_v;
    for (i = 0; i < 128/64; i++){
        temp_v = (int64_t)pws->d[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/64; i++){
        temp_v = (int64_t)pwd->d[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+128/64] = temp_v;
    }
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = lsx_sat_s_128(temp_reg[i],31);
    }
}


void helper_lsx_vssrarneni_d_q(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    __int128_t temp_reg[2];
    __int128_t temp_v;
    temp_v = (__int128_t)pws->q[0];
    if (ui > 0)
        temp_v = lsx_round_nearest_even(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[0] = temp_v;

    temp_v = (__uint128_t)pwd->q[0];
    if (ui > 0)
       temp_v = lsx_round_nearest_even(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[1] = temp_v;
    pwd->d[0] = lsx_sat_s_128(temp_reg[0],63);
    pwd->d[1] = lsx_sat_s_128(temp_reg[1],63);
}

void helper_lsx_vssrarneni_bu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg [128/8];
    int64_t temp_v;
    for (i = 0; i < 128/16; i++){
        temp_v = (int64_t)(int16_t)pws->h[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/16; i++){
        temp_v = (int64_t)(int16_t)pwd->h[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+128/16] = temp_v;
    }
    for (i = 0; i < 128/8; i++) {
        pwd->b[i] = lsx_sat_u_df(DF_HALF,temp_reg[i],7);
    }
}

void helper_lsx_vssrarneni_hu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg[128/16];
    int64_t temp_v;
    for (i = 0; i < 128/32; i++){
        temp_v = (int64_t)(int32_t)pws->w[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/32; i++){
        temp_v = (int64_t)(int32_t)pwd->w[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+128/32] = temp_v;
    }
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = lsx_sat_u_df(DF_WORD,temp_reg[i],15);
    }
}


void helper_lsx_vssrarneni_wu_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg[128/32];
    int64_t temp_v;
    for (i = 0; i < 128/64; i++){
        temp_v = (int64_t)pws->d[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 128/64; i++){
        temp_v = (int64_t)pwd->d[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+128/64] = temp_v;
    }
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = lsx_sat_u_128(temp_reg[i],31);
    }
}


void helper_lsx_vssrarneni_du_q(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    __int128_t temp_reg[2];
    __int128_t temp_v;
    temp_v = (__int128_t)pws->q[0];
    if (ui > 0)
        temp_v = lsx_round_nearest_even(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[0] = temp_v;

    temp_v = (__int128_t)pwd->q[0];
    if (ui > 0)
       temp_v = lsx_round_nearest_even(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[1] = temp_v;
    pwd->d[0] = lsx_sat_u_128(temp_reg[0],63);
    pwd->d[1] = lsx_sat_u_128(temp_reg[1],63);
}






//--------xv-----/
void helper_lsx_xvsrlrneni_b_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg [256/8];
    uint64_t temp_v;
    for (i = 0; i < 256/16; i++){
        temp_v = (uint16_t)pws->h[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/16; i++){
        temp_v = (uint16_t)pwd->h[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+256/16] = temp_v;
    }
    for (i = 0; i < 256/8; i++) {
        pwd->b[i] = temp_reg[i];
    }
}

void helper_lsx_xvsrlrneni_h_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg[256/16];
    uint64_t temp_v;
    for (i = 0; i < 256/32; i++){
        temp_v = (uint32_t)pws->w[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/32; i++){
        temp_v = (uint32_t)pwd->w[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+256/32] = temp_v;
    }
    for (i = 0; i < 256/16; i++) {
        pwd->h[i] = temp_reg[i];
    }
}


void helper_lsx_xvsrlrneni_w_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg[256/32];
    uint64_t temp_v;
    for (i = 0; i < 256/64; i++){
        temp_v = (uint64_t)pws->d[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/64; i++){
        temp_v = (uint64_t)pwd->d[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+256/64] = temp_v;
    }
    for (i = 0; i < 256/32; i++) {
        pwd->w[i] = temp_reg[i];
    }

}


void helper_lsx_xvsrlrneni_d_q(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    __uint128_t temp_reg[4];
    __uint128_t temp_v;
    temp_v = (__uint128_t)pws->q[0];
    if (ui > 0)
        temp_v = lsx_round_nearest_even(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[0] = temp_v;
    temp_v = (__uint128_t)pws->q[1];
    if (ui > 0)
        temp_v = lsx_round_nearest_even(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[1] = temp_v;


    temp_v = (__uint128_t)pwd->q[0];
    if (ui > 0)
        temp_v = lsx_round_nearest_even(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[2] = temp_v;
    temp_v = (__uint128_t)pwd->q[1];
    if (ui > 0)
        temp_v = lsx_round_nearest_even(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[3] = temp_v;

    pwd->d[0] = (uint64_t)temp_reg[0];
    pwd->d[1] = (uint64_t)temp_reg[1];
    pwd->d[2] = (uint64_t)temp_reg[2];
    pwd->d[3] = (uint64_t)temp_reg[3];
}

void helper_lsx_xvsrarneni_b_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg [256/8];
    int64_t temp_v;
    for (i = 0; i < 256/16; i++){
        temp_v = (int64_t)(int16_t)pws->h[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/16; i++){
        temp_v = (int64_t)(int16_t)pwd->h[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+256/16] = temp_v;
    }
    for (i = 0; i < 256/8; i++) {
        pwd->b[i] = temp_reg[i];
    }
}

void helper_lsx_xvsrarneni_h_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg[256/16];
    int64_t temp_v;
    for (i = 0; i < 256/32; i++){
        temp_v = (int64_t)(int32_t)pws->w[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/32; i++){
        temp_v = (int64_t)(int32_t)pwd->w[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+256/32] = temp_v;
    }
    for (i = 0; i < 256/16; i++) {
        pwd->h[i] = temp_reg[i];
    }
}


void helper_lsx_xvsrarneni_w_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg[256/32];
    int64_t temp_v;
    for (i = 0; i < 256/64; i++){
        temp_v = (int64_t)pws->d[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/64; i++){
        temp_v = (int64_t)pwd->d[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+256/64] = temp_v;
    }
    for (i = 0; i < 256/32; i++) {
        pwd->w[i] = temp_reg[i];
    }

}


void helper_lsx_xvsrarneni_d_q(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    __int128_t temp_reg[4];
    __int128_t temp_v;

    temp_v = (__int128_t)pws->q[0];
    if (ui > 0)
        temp_v = lsx_round_nearest_even(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[0] = temp_v;

    temp_v = (__int128_t)pws->q[1];
    if (ui > 0)
        temp_v = lsx_round_nearest_even(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[1] = temp_v;


    temp_v = (__int128_t)pwd->q[0];
    if (ui > 0)
        temp_v = lsx_round_nearest_even(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[2] = temp_v;

    temp_v = (__int128_t)pwd->q[1];
    if (ui > 0)
        temp_v = lsx_round_nearest_even(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[3] = temp_v;

    pwd->d[0] = temp_reg[0];
    pwd->d[1] = temp_reg[1];
    pwd->d[2] = temp_reg[2];
    pwd->d[3] = temp_reg[3];
}

void helper_lsx_xvsrlni_b_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg [256/8];
    uint64_t temp_v;
    for (i = 0; i < 256/16; i++){
        temp_v = (uint16_t)pws->h[i];
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/16; i++){
        temp_v = (uint16_t)pwd->h[i];
        temp_v = temp_v >> ui;
        temp_reg[i+256/16] = temp_v;
    }
    for (i = 0; i < 256/8; i++) {
        pwd->b[i] = temp_reg[i];
    }
}

void helper_lsx_xvsrlni_h_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg[256/16];
    uint64_t temp_v;
    for (i = 0; i < 256/32; i++){
        temp_v = (uint32_t)pws->w[i];
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/32; i++){
        temp_v = (uint32_t)pwd->w[i];
        temp_v = temp_v >> ui;
        temp_reg[i+256/32] = temp_v;
    }
    for (i = 0; i < 256/16; i++) {
        pwd->h[i] = temp_reg[i];
    }
}


void helper_lsx_xvsrlni_w_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg[256/32];
    uint64_t temp_v;
    for (i = 0; i < 256/64; i++){
        temp_v = (uint64_t)pws->d[i];
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/64; i++){
        temp_v = (uint64_t)pwd->d[i];
        temp_v = temp_v >> ui;
        temp_reg[i+256/64] = temp_v;
    }
    for (i = 0; i < 256/32; i++) {
        pwd->w[i] = temp_reg[i];
    }

}


void helper_lsx_xvsrlni_d_q(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    __uint128_t temp_reg[4];
    __uint128_t temp_v;
    temp_v = (__uint128_t)pws->q[0];
    temp_v = temp_v >> ui;
    temp_reg[0] = temp_v;
    temp_v = (__uint128_t)pws->q[1];
    temp_v = temp_v >> ui;
    temp_reg[1] = temp_v;


    temp_v = (__uint128_t)pwd->q[0];
    temp_v = temp_v >> ui;
    temp_reg[2] = temp_v;

    temp_v = (__uint128_t)pwd->q[1];
    temp_v = temp_v >> ui;
    temp_reg[3] = temp_v;
    pwd->d[0] = (uint64_t)temp_reg[0];
    pwd->d[1] = (uint64_t)temp_reg[1];
    pwd->d[2] = (uint64_t)temp_reg[2];
    pwd->d[3] = (uint64_t)temp_reg[3];
}

void helper_lsx_xvsrlrni_b_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg [256/8];
    uint64_t temp_v;
    for (i = 0; i < 256/16; i++){
        temp_v = (uint16_t)pws->h[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/16; i++){
        temp_v = (uint16_t)pwd->h[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+256/16] = temp_v;
    }
    for (i = 0; i < 256/8; i++) {
        pwd->b[i] = temp_reg[i];
    }
}

void helper_lsx_xvsrlrni_h_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg[256/16];
    uint64_t temp_v;
    for (i = 0; i < 256/32; i++){
        temp_v = (uint32_t)pws->w[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/32; i++){
        temp_v = (uint32_t)pwd->w[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+256/32] = temp_v;
    }
    for (i = 0; i < 256/16; i++) {
        pwd->h[i] = temp_reg[i];
    }
}


void helper_lsx_xvsrlrni_w_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg[256/32];
    uint64_t temp_v;
    for (i = 0; i < 256/64; i++){
        temp_v = (uint64_t)pws->d[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/64; i++){
        temp_v = (uint64_t)pwd->d[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+256/64] = temp_v;
    }
    for (i = 0; i < 256/32; i++) {
        pwd->w[i] = temp_reg[i];
    }

}


void helper_lsx_xvsrlrni_d_q(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    __uint128_t temp_reg[4];
    __uint128_t temp_v;
    temp_v = (__uint128_t)pws->q[0];
    if (ui > 0)
        temp_v = lsx_round_legacy(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[0] = temp_v;
    temp_v = (__uint128_t)pws->q[1];
    if (ui > 0)
        temp_v = lsx_round_legacy(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[1] = temp_v;


    temp_v = (__uint128_t)pwd->q[0];
    if (ui > 0)
        temp_v = lsx_round_legacy(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[2] = temp_v;

    temp_v = (__uint128_t)pwd->q[1];
    if (ui > 0)
        temp_v = lsx_round_legacy(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[3] = temp_v;
    pwd->d[0] = (uint64_t)temp_reg[0];
    pwd->d[1] = (uint64_t)temp_reg[1];
    pwd->d[2] = (uint64_t)temp_reg[2];
    pwd->d[3] = (uint64_t)temp_reg[3];
}


void helper_lsx_xvssrlni_b_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg [256/8];
    uint64_t temp_v;
    for (i = 0; i < 256/16; i++){
        temp_v = (uint16_t)pws->h[i];
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/16; i++){
        temp_v = (uint16_t)pwd->h[i];
        temp_v = temp_v >> ui;
        temp_reg[i+256/16] = temp_v;
    }
    for (i = 0; i < 256/8; i++) {
        pwd->b[i] = lsx_sat_s_df(DF_HALF,temp_reg[i],7);
    }
}

void helper_lsx_xvssrlni_h_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg[256/16];
    uint64_t temp_v;
    for (i = 0; i < 256/32; i++){
        temp_v = (uint32_t)pws->w[i];
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/32; i++){
        temp_v = (uint32_t)pwd->w[i];
        temp_v = temp_v >> ui;
        temp_reg[i+256/32] = temp_v;
    }
    for (i = 0; i < 256/16; i++) {
        pwd->h[i] = lsx_sat_s_df(DF_WORD,temp_reg[i],15);
    }
}


void helper_lsx_xvssrlni_w_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg[256/32];
    uint64_t temp_v;
    for (i = 0; i < 256/64; i++){
        temp_v = (uint64_t)pws->d[i];
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/64; i++){
        temp_v = (uint64_t)pwd->d[i];
        temp_v = temp_v >> ui;
        temp_reg[i+256/64] = temp_v;
    }
    for (i = 0; i < 256/32; i++) {
        pwd->w[i] = lsx_sat_s_128u(temp_reg[i],31);
    }

}


void helper_lsx_xvssrlni_d_q(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    __uint128_t temp_reg[4];
    __uint128_t temp_v;
    temp_v = (__uint128_t)pws->q[0];
    temp_v = temp_v >> ui;
    temp_reg[0] = temp_v;
    temp_v = (__uint128_t)pws->q[1];
    temp_v = temp_v >> ui;
    temp_reg[1] = temp_v;


    temp_v = (__uint128_t)pwd->q[0];
    temp_v = temp_v >> ui;
    temp_reg[2] = temp_v;

    temp_v = (__uint128_t)pwd->q[1];
    temp_v = temp_v >> ui;
    temp_reg[3] = temp_v;
    pwd->d[0] = lsx_sat_s_128u(temp_reg[0],63);
    pwd->d[1] = lsx_sat_s_128u(temp_reg[1],63);
    pwd->d[2] = lsx_sat_s_128u(temp_reg[2],63);
    pwd->d[3] = lsx_sat_s_128u(temp_reg[3],63);
}


void helper_lsx_xvssrlni_bu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg [256/8];
    uint64_t temp_v;
    for (i = 0; i < 256/16; i++){
        temp_v = (uint16_t)pws->h[i];
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/16; i++){
        temp_v = (uint16_t)pwd->h[i];
        temp_v = temp_v >> ui;
        temp_reg[i+256/16] = temp_v;
    }
    for (i = 0; i < 256/8; i++) {
        pwd->b[i] = lsx_sat_u_df(DF_HALF,temp_reg[i],7);
    }
}

void helper_lsx_xvssrlni_hu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg[256/16];
    uint64_t temp_v;
    for (i = 0; i < 256/32; i++){
        temp_v = (uint32_t)pws->w[i];
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/32; i++){
        temp_v = (uint32_t)pwd->w[i];
        temp_v = temp_v >> ui;
        temp_reg[i+256/32] = temp_v;
    }
    for (i = 0; i < 256/16; i++) {
        pwd->h[i] = lsx_sat_u_df(DF_WORD,temp_reg[i],15);
    }
}


void helper_lsx_xvssrlni_wu_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg[256/32];
    uint64_t temp_v;
    for (i = 0; i < 256/64; i++){
        temp_v = (uint64_t)pws->d[i];
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/64; i++){
        temp_v = (uint64_t)pwd->d[i];
        temp_v = temp_v >> ui;
        temp_reg[i+256/64] = temp_v;
    }
    for (i = 0; i < 256/32; i++) {
        pwd->w[i] = lsx_sat_u_128(temp_reg[i],31);
    }

}


void helper_lsx_xvssrlni_du_q(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    __uint128_t temp_reg[4];
    __uint128_t temp_v;
    temp_v = (__uint128_t)pws->q[0];
    temp_v = temp_v >> ui;
    temp_reg[0] = temp_v;
    temp_v = (__uint128_t)pws->q[1];
    temp_v = temp_v >> ui;
    temp_reg[1] = temp_v;


    temp_v = (__uint128_t)pwd->q[0];
    temp_v = temp_v >> ui;
    temp_reg[2] = temp_v;

    temp_v = (__uint128_t)pwd->q[1];
    temp_v = temp_v >> ui;
    temp_reg[3] = temp_v;

    pwd->d[0] = lsx_sat_u_128(temp_reg[0],63);
    pwd->d[1] = lsx_sat_u_128(temp_reg[1],63);
    pwd->d[2] = lsx_sat_u_128(temp_reg[2],63);
    pwd->d[3] = lsx_sat_u_128(temp_reg[3],63);
}

void helper_lsx_xvssrlrni_b_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg [256/8];
    uint64_t temp_v;
    for (i = 0; i < 256/16; i++){
        temp_v = (uint16_t)pws->h[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/16; i++){
        temp_v = (uint16_t)pwd->h[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+256/16] = temp_v;
    }
    for (i = 0; i < 256/8; i++) {
        pwd->b[i] = lsx_sat_s_df(DF_HALF,temp_reg[i],7);
    }
}

void helper_lsx_xvssrlrni_h_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg[256/16];
    uint64_t temp_v;
    for (i = 0; i < 256/32; i++){
        temp_v = (uint32_t)pws->w[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/32; i++){
        temp_v = (uint32_t)pwd->w[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+256/32] = temp_v;
    }
    for (i = 0; i < 256/16; i++) {
        pwd->h[i] = lsx_sat_s_df(DF_WORD,temp_reg[i],15);
    }
}


void helper_lsx_xvssrlrni_w_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg[256/32];
    uint64_t temp_v;
    for (i = 0; i < 256/64; i++){
        temp_v = (uint64_t)pws->d[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/64; i++){
        temp_v = (uint64_t)pwd->d[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+256/64] = temp_v;
    }
    for (i = 0; i < 256/32; i++) {
        pwd->w[i] = lsx_sat_s_128u(temp_reg[i],31);
    }

}


void helper_lsx_xvssrlrni_d_q(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    __uint128_t temp_reg[4];
    __uint128_t temp_v;
    temp_v = (__uint128_t)pws->q[0];
    if (ui > 0)
        temp_v = lsx_round_legacy(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[0] = temp_v;
    temp_v = (__uint128_t)pws->q[1];
    if (ui > 0)
        temp_v = lsx_round_legacy(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[1] = temp_v;


    temp_v = (__uint128_t)pwd->q[0];
    if (ui > 0)
        temp_v = lsx_round_legacy(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[2] = temp_v;

    temp_v = (__uint128_t)pwd->q[1];
    if (ui > 0)
        temp_v = lsx_round_legacy(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[3] = temp_v;

    pwd->d[0] = lsx_sat_s_128u(temp_reg[0],63);
    pwd->d[1] = lsx_sat_s_128u(temp_reg[1],63);
    pwd->d[2] = lsx_sat_s_128u(temp_reg[2],63);
    pwd->d[3] = lsx_sat_s_128u(temp_reg[3],63);
}

void helper_lsx_xvssrlrni_bu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg [256/8];
    uint64_t temp_v;
    for (i = 0; i < 256/16; i++){
        temp_v = (uint16_t)pws->h[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/16; i++){
        temp_v = (uint16_t)pwd->h[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+256/16] = temp_v;
    }
    for (i = 0; i < 256/8; i++) {
        pwd->b[i] = lsx_sat_u_df(DF_HALF,temp_reg[i],7);
    }
}

void helper_lsx_xvssrlrni_hu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg[256/16];
    uint64_t temp_v;
    for (i = 0; i < 256/32; i++){
        temp_v = (uint32_t)pws->w[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/32; i++){
        temp_v = (uint32_t)pwd->w[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+256/32] = temp_v;
    }
    for (i = 0; i < 256/16; i++) {
        pwd->h[i] = lsx_sat_u_df(DF_WORD,temp_reg[i],15);
    }
}


void helper_lsx_xvssrlrni_wu_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg[256/32];
    uint64_t temp_v;
    for (i = 0; i < 256/64; i++){
        temp_v = (uint64_t)pws->d[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/64; i++){
        temp_v = (uint64_t)pwd->d[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+256/64] = temp_v;
    }
    for (i = 0; i < 256/32; i++) {
        pwd->w[i] = lsx_sat_u_128(temp_reg[i],31);
    }

}


void helper_lsx_xvssrlrni_du_q(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    __uint128_t temp_reg[4];
    __uint128_t temp_v;
    temp_v = (__uint128_t)pws->q[0];
    if (ui > 0)
        temp_v = lsx_round_legacy(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[0] = temp_v;
    temp_v = (__uint128_t)pws->q[1];
    if (ui > 0)
        temp_v = lsx_round_legacy(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[1] = temp_v;


    temp_v = (__uint128_t)pwd->q[0];
    if (ui > 0)
        temp_v = lsx_round_legacy(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[2] = temp_v;

    temp_v = (__uint128_t)pwd->q[1];
    if (ui > 0)
        temp_v = lsx_round_legacy(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[3] = temp_v;

    pwd->d[0] = lsx_sat_u_128(temp_reg[0],63);
    pwd->d[1] = lsx_sat_u_128(temp_reg[1],63);
    pwd->d[2] = lsx_sat_u_128(temp_reg[2],63);
    pwd->d[3] = lsx_sat_u_128(temp_reg[3],63);
}

void helper_lsx_xvsrani_b_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg [256/8];
    int64_t temp_v;
    for (i = 0; i < 256/16; i++){
        temp_v = (int64_t)(int16_t)pws->h[i];
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/16; i++){
        temp_v = (int64_t)(int16_t)pwd->h[i];
        temp_v = temp_v >> ui;
        temp_reg[i+256/16] = temp_v;
    }
    for (i = 0; i < 256/8; i++) {
        pwd->b[i] = temp_reg[i];
    }
}

void helper_lsx_xvsrani_h_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg[256/16];
    int64_t temp_v;
    for (i = 0; i < 256/32; i++){
        temp_v = (int64_t)(int32_t)pws->w[i];
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/32; i++){
        temp_v = (int64_t)(int32_t)pwd->w[i];
        temp_v = temp_v >> ui;
        temp_reg[i+256/32] = temp_v;
    }
    for (i = 0; i < 256/16; i++) {
        pwd->h[i] = temp_reg[i];
    }
}


void helper_lsx_xvsrani_w_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg[256/32];
    int64_t temp_v;
    for (i = 0; i < 256/64; i++){
        temp_v = (int64_t)pws->d[i];
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/64; i++){
        temp_v = (int64_t)pwd->d[i];
        temp_v = temp_v >> ui;
        temp_reg[i+256/64] = temp_v;
    }
    for (i = 0; i < 256/32; i++) {
        pwd->w[i] = temp_reg[i];
    }

}


void helper_lsx_xvsrani_d_q(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    __int128_t temp_reg[4];
    __int128_t temp_v;
    temp_v = (__int128_t)pws->q[0];
    temp_v = temp_v >> ui;
    temp_reg[0] = temp_v;
    temp_v = (__int128_t)pws->q[1];
    temp_v = temp_v >> ui;
    temp_reg[1] = temp_v;


    temp_v = (__int128_t)pwd->q[0];
    temp_v = temp_v >> ui;
    temp_reg[2] = temp_v;

    temp_v = (__int128_t)pwd->q[1];
    temp_v = temp_v >> ui;
    temp_reg[3] = temp_v;

    pwd->d[0] = (int64_t)temp_reg[0];
    pwd->d[1] = (int64_t)temp_reg[1];
    pwd->d[2] = (int64_t)temp_reg[2];
    pwd->d[3] = (int64_t)temp_reg[3];
}


void helper_lsx_xvsrarni_b_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg [256/8];
    int64_t temp_v;
    for (i = 0; i < 256/16; i++){
        temp_v = (int64_t)(int16_t)pws->h[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/16; i++){
        temp_v = (int64_t)(int16_t)pwd->h[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+256/16] = temp_v;
    }
    for (i = 0; i < 256/8; i++) {
        pwd->b[i] = temp_reg[i];
    }
}

void helper_lsx_xvsrarni_h_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg[256/16];
    int64_t temp_v;
    for (i = 0; i < 256/32; i++){
        temp_v = (int64_t)(int32_t)pws->w[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/32; i++){
        temp_v = (int64_t)(int32_t)pwd->w[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+256/32] = temp_v;
    }
    for (i = 0; i < 256/16; i++) {
        pwd->h[i] = temp_reg[i];
    }
}


void helper_lsx_xvsrarni_w_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg[256/32];
    int64_t temp_v;
    for (i = 0; i < 256/64; i++){
        temp_v = (int64_t)pws->d[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/64; i++){
        temp_v = (int64_t)pwd->d[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+256/64] = temp_v;
    }
    for (i = 0; i < 256/32; i++) {
        pwd->w[i] = temp_reg[i];
    }

}


void helper_lsx_xvsrarni_d_q(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    __int128_t temp_reg[4];
    __int128_t temp_v;
    temp_v = (__int128_t)pws->q[0];
    if (ui > 0)
        temp_v = lsx_round_legacy(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[0] = temp_v;
    temp_v = (__int128_t)pws->q[1];
    if (ui > 0)
        temp_v = lsx_round_legacy(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[1] = temp_v;


    temp_v = (__int128_t)pwd->q[0];
    if (ui > 0)
        temp_v = lsx_round_legacy(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[2] = temp_v;

    temp_v = (__int128_t)pwd->q[1];
    if (ui > 0)
        temp_v = lsx_round_legacy(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[3] = temp_v;

    pwd->d[0] = temp_reg[0];
    pwd->d[1] = temp_reg[1];
    pwd->d[2] = temp_reg[2];
    pwd->d[3] = temp_reg[3];
}

void helper_lsx_xvssrani_b_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg [256/8];
    int64_t temp_v;
    for (i = 0; i < 256/16; i++){
        temp_v = (int64_t)(int16_t)pws->h[i];
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/16; i++){
        temp_v = (int64_t)(int16_t)pwd->h[i];
        temp_v = temp_v >> ui;
        temp_reg[i+256/16] = temp_v;
    }
    for (i = 0; i < 256/8; i++) {
        pwd->b[i] = lsx_sat_s_df(DF_HALF,temp_reg[i],7);
    }
}

void helper_lsx_xvssrani_h_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg[256/16];
    int64_t temp_v;
    for (i = 0; i < 256/32; i++){
        temp_v = (int64_t)(int32_t)pws->w[i];
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/32; i++){
        temp_v = (int64_t)(int32_t)pwd->w[i];
        temp_v = temp_v >> ui;
        temp_reg[i+256/32] = temp_v;
    }
    for (i = 0; i < 256/16; i++) {
        pwd->h[i] = lsx_sat_s_df(DF_WORD,temp_reg[i],15);
    }
}


void helper_lsx_xvssrani_w_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg[256/32];
    int64_t temp_v;
    for (i = 0; i < 256/64; i++){
        temp_v = (int64_t)pws->d[i];
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/64; i++){
        temp_v = (int64_t)pwd->d[i];
        temp_v = temp_v >> ui;
        temp_reg[i+256/64] = temp_v;
    }
    for (i = 0; i < 256/32; i++) {
        pwd->w[i] = lsx_sat_s_128(temp_reg[i],31);
    }

}


void helper_lsx_xvssrani_d_q(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    __int128_t temp_reg[4];
    __int128_t temp_v;
    temp_v = (__int128_t)pws->q[0];
    temp_v = temp_v >> ui;
    temp_reg[0] = temp_v;
    temp_v = (__int128_t)pws->q[1];
    temp_v = temp_v >> ui;
    temp_reg[1] = temp_v;


    temp_v = (__uint128_t)pwd->q[0];
    temp_v = temp_v >> ui;
    temp_reg[2] = temp_v;
    temp_v = (__uint128_t)pwd->q[1];
    temp_v = temp_v >> ui;
    temp_reg[3] = temp_v;

    pwd->d[0] = lsx_sat_s_128(temp_reg[0],63);
    pwd->d[1] = lsx_sat_s_128(temp_reg[1],63);
    pwd->d[2] = lsx_sat_s_128(temp_reg[2],63);
    pwd->d[3] = lsx_sat_s_128(temp_reg[3],63);
}

void helper_lsx_xvssrani_bu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg [256/8];
    int64_t temp_v;
    for (i = 0; i < 256/16; i++){
        temp_v = (int64_t)(int16_t)pws->h[i];
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/16; i++){
        temp_v = (int64_t)(int16_t)pwd->h[i];
        temp_v = temp_v >> ui;
        temp_reg[i+256/16] = temp_v;
    }
    for (i = 0; i < 256/8; i++) {
        pwd->b[i] = lsx_sat_u_df(DF_HALF,temp_reg[i],7);
    }
}

void helper_lsx_xvssrani_hu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg[256/16];
    int64_t temp_v;
    for (i = 0; i < 256/32; i++){
        temp_v = (int64_t)(int32_t)pws->w[i];
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/32; i++){
        temp_v = (int64_t)(int32_t)pwd->w[i];
        temp_v = temp_v >> ui;
        temp_reg[i+256/32] = temp_v;
    }
    for (i = 0; i < 256/16; i++) {
        pwd->h[i] = lsx_sat_u_df(DF_WORD,temp_reg[i],15);
    }
}


void helper_lsx_xvssrani_wu_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg[256/32];
    int64_t temp_v;
    for (i = 0; i < 256/64; i++){
        temp_v = (int64_t)pws->d[i];
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/64; i++){
        temp_v = (int64_t)pwd->d[i];
        temp_v = temp_v >> ui;
        temp_reg[i+256/64] = temp_v;
    }
    for (i = 0; i < 256/32; i++) {
        pwd->w[i] = lsx_sat_u_128(temp_reg[i],31);
    }

}


void helper_lsx_xvssrani_du_q(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    __int128_t temp_reg[4];
    __int128_t temp_v;
    temp_v = (__int128_t)pws->q[0];
    temp_v = temp_v >> ui;
    temp_reg[0] = temp_v;
    temp_v = (__int128_t)pws->q[1];
    temp_v = temp_v >> ui;
    temp_reg[1] = temp_v;


    temp_v = (__uint128_t)pwd->q[0];
    temp_v = temp_v >> ui;
    temp_reg[2] = temp_v;

    temp_v = (__uint128_t)pwd->q[1];
    temp_v = temp_v >> ui;
    temp_reg[3] = temp_v;

    pwd->d[0] = lsx_sat_u_128(temp_reg[0],63);
    pwd->d[1] = lsx_sat_u_128(temp_reg[1],63);
    pwd->d[2] = lsx_sat_u_128(temp_reg[2],63);
    pwd->d[3] = lsx_sat_u_128(temp_reg[3],63);
}

void helper_lsx_xvssrarni_b_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg [256/8];
    int64_t temp_v;
    for (i = 0; i < 256/16; i++){
        temp_v = (int64_t)(int16_t)pws->h[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/16; i++){
        temp_v = (int64_t)(int16_t)pwd->h[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+256/16] = temp_v;
    }
    for (i = 0; i < 256/8; i++) {
        pwd->b[i] = lsx_sat_s_df(DF_HALF,temp_reg[i],7);
    }
}

void helper_lsx_xvssrarni_h_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg[256/16];
    int64_t temp_v;
    for (i = 0; i < 256/32; i++){
        temp_v = (int64_t)(int32_t)pws->w[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/32; i++){
        temp_v = (int64_t)(int32_t)pwd->w[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+256/32] = temp_v;
    }
    for (i = 0; i < 256/16; i++) {
        pwd->h[i] = lsx_sat_s_df(DF_WORD,temp_reg[i],15);
    }
}


void helper_lsx_xvssrarni_w_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg[256/32];
    int64_t temp_v;
    for (i = 0; i < 256/64; i++){
        temp_v = (int64_t)pws->d[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/64; i++){
        temp_v = (int64_t)pwd->d[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+256/64] = temp_v;
    }
    for (i = 0; i < 256/32; i++) {
        pwd->w[i] = lsx_sat_s_128(temp_reg[i],31);
    }
}


void helper_lsx_xvssrarni_d_q(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    __int128_t temp_reg[4];
    __int128_t temp_v;
    temp_v = (__int128_t)pws->q[0];
    if (ui > 0)
        temp_v = lsx_round_legacy(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[0] = temp_v;
    temp_v = (__int128_t)pws->q[1];
    if (ui > 0)
        temp_v = lsx_round_legacy(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[1] = temp_v;


    temp_v = (__uint128_t)pwd->q[0];
    if (ui > 0)
       temp_v = lsx_round_legacy(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[2] = temp_v;

    temp_v = (__uint128_t)pwd->q[1];
    if (ui > 0)
       temp_v = lsx_round_legacy(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[3] = temp_v;

    pwd->d[0] = lsx_sat_s_128(temp_reg[0],63);
    pwd->d[1] = lsx_sat_s_128(temp_reg[1],63);
    pwd->d[2] = lsx_sat_s_128(temp_reg[2],63);
    pwd->d[3] = lsx_sat_s_128(temp_reg[3],63);
}

void helper_lsx_xvssrarni_bu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg [256/8];
    int64_t temp_v;
    for (i = 0; i < 256/16; i++){
        temp_v = (int64_t)(int16_t)pws->h[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/16; i++){
        temp_v = (int64_t)(int16_t)pwd->h[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+256/16] = temp_v;
    }
    for (i = 0; i < 256/8; i++) {
        pwd->b[i] = lsx_sat_u_df(DF_HALF,temp_reg[i],7);
    }
}

void helper_lsx_xvssrarni_hu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg[256/16];
    int64_t temp_v;
    for (i = 0; i < 256/32; i++){
        temp_v = (int64_t)(int32_t)pws->w[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/32; i++){
        temp_v = (int64_t)(int32_t)pwd->w[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+256/32] = temp_v;
    }
    for (i = 0; i < 256/16; i++) {
        pwd->h[i] = lsx_sat_u_df(DF_WORD,temp_reg[i],15);
    }
}


void helper_lsx_xvssrarni_wu_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg[256/32];
    int64_t temp_v;
    for (i = 0; i < 256/64; i++){
        temp_v = (int64_t)pws->d[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/64; i++){
        temp_v = (int64_t)pwd->d[i];
        if (ui > 0)
            temp_v = lsx_round_legacy(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+256/64] = temp_v;
    }
    for (i = 0; i < 256/32; i++) {
        pwd->w[i] = lsx_sat_u_128(temp_reg[i],31);
    }
}


void helper_lsx_xvssrarni_du_q(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    __int128_t temp_reg[4];
    __int128_t temp_v;
    temp_v = (__int128_t)pws->q[0];
    if (ui > 0)
        temp_v = lsx_round_legacy(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[0] = temp_v;
    temp_v = (__int128_t)pws->q[1];
    if (ui > 0)
        temp_v = lsx_round_legacy(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[1] = temp_v;


    temp_v = (__uint128_t)pwd->q[0];
    if (ui > 0)
       temp_v = lsx_round_legacy(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[2] = temp_v;

    temp_v = (__uint128_t)pwd->q[1];
    if (ui > 0)
       temp_v = lsx_round_legacy(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[3] = temp_v;

    pwd->d[0] = lsx_sat_u_128(temp_reg[0],63);
    pwd->d[1] = lsx_sat_u_128(temp_reg[1],63);
    pwd->d[2] = lsx_sat_u_128(temp_reg[2],63);
    pwd->d[3] = lsx_sat_u_128(temp_reg[3],63);
}

void helper_lsx_xvssrlrneni_b_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg [256/8];
    uint64_t temp_v;
    for (i = 0; i < 256/16; i++){
        temp_v = (uint16_t)pws->h[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/16; i++){
        temp_v = (uint16_t)pwd->h[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+256/16] = temp_v;
    }
    for (i = 0; i < 256/8; i++) {
        pwd->b[i] = lsx_sat_s_df(DF_HALF,temp_reg[i],7);
    }
}

void helper_lsx_xvssrlrneni_h_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg[256/16];
    uint64_t temp_v;
    for (i = 0; i < 256/32; i++){
        temp_v = (uint32_t)pws->w[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/32; i++){
        temp_v = (uint32_t)pwd->w[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+256/32] = temp_v;
    }
    for (i = 0; i < 256/16; i++) {
        pwd->h[i] = lsx_sat_s_df(DF_WORD,temp_reg[i],15);
    }
}


void helper_lsx_xvssrlrneni_w_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg[256/32];
    uint64_t temp_v;
    for (i = 0; i < 256/64; i++){
        temp_v = (uint64_t)pws->d[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/64; i++){
        temp_v = (uint64_t)pwd->d[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+256/64] = temp_v;
    }
    for (i = 0; i < 256/32; i++) {
        pwd->w[i] = lsx_sat_s_128u(temp_reg[i],31);
    }

}


void helper_lsx_xvssrlrneni_d_q(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    __uint128_t temp_reg[4];
    __uint128_t temp_v;
    temp_v = (__uint128_t)pws->q[0];
    if (ui > 0)
        temp_v = lsx_round_nearest_even(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[0] = temp_v;
    temp_v = (__uint128_t)pws->q[1];
    if (ui > 0)
        temp_v = lsx_round_nearest_even(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[1] = temp_v;


    temp_v = (__uint128_t)pwd->q[0];
    if (ui > 0)
        temp_v = lsx_round_nearest_even(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[2] = temp_v;

    temp_v = (__uint128_t)pwd->q[1];
    if (ui > 0)
        temp_v = lsx_round_nearest_even(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[3] = temp_v;

    pwd->d[0] = lsx_sat_s_128u(temp_reg[0],63);
    pwd->d[1] = lsx_sat_s_128u(temp_reg[1],63);
    pwd->d[2] = lsx_sat_s_128u(temp_reg[2],63);
    pwd->d[3] = lsx_sat_s_128u(temp_reg[3],63);
}

void helper_lsx_xvssrlrneni_bu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg [256/8];
    uint64_t temp_v;
    for (i = 0; i < 256/16; i++){
        temp_v = (uint16_t)pws->h[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/16; i++){
        temp_v = (uint16_t)pwd->h[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+256/16] = temp_v;
    }
    for (i = 0; i < 256/8; i++) {
        pwd->b[i] = lsx_sat_u_df(DF_HALF,temp_reg[i],7);
    }
}

void helper_lsx_xvssrlrneni_hu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg[256/16];
    uint64_t temp_v;
    for (i = 0; i < 256/32; i++){
        temp_v = (uint32_t)pws->w[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/32; i++){
        temp_v = (uint32_t)pwd->w[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+256/32] = temp_v;
    }
    for (i = 0; i < 256/16; i++) {
        pwd->h[i] = lsx_sat_u_df(DF_WORD,temp_reg[i],15);
    }
}


void helper_lsx_xvssrlrneni_wu_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    uint64_t temp_reg[256/32];
    uint64_t temp_v;
    for (i = 0; i < 256/64; i++){
        temp_v = (uint64_t)pws->d[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/64; i++){
        temp_v = (uint64_t)pwd->d[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+256/64] = temp_v;
    }
    for (i = 0; i < 256/32; i++) {
        pwd->w[i] = lsx_sat_u_128(temp_reg[i],31);
    }

}


void helper_lsx_xvssrlrneni_du_q(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    __uint128_t temp_reg[4];
    __uint128_t temp_v;
    temp_v = (__uint128_t)pws->q[0];
    if (ui > 0)
        temp_v = lsx_round_nearest_even(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[0] = temp_v;
    temp_v = (__uint128_t)pws->q[1];
    if (ui > 0)
        temp_v = lsx_round_nearest_even(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[1] = temp_v;


    temp_v = (__uint128_t)pwd->q[0];
    if (ui > 0)
        temp_v = lsx_round_nearest_even(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[2] = temp_v;

    temp_v = (__uint128_t)pwd->q[1];
    if (ui > 0)
        temp_v = lsx_round_nearest_even(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[3] = temp_v;

    pwd->d[0] = lsx_sat_u_128(temp_reg[0],63);
    pwd->d[1] = lsx_sat_u_128(temp_reg[1],63);
    pwd->d[2] = lsx_sat_u_128(temp_reg[2],63);
    pwd->d[3] = lsx_sat_u_128(temp_reg[3],63);
}

void helper_lsx_xvssrarneni_b_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg [256/8];
    int64_t temp_v;
    for (i = 0; i < 256/16; i++){
        temp_v = (int64_t)(int16_t)pws->h[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/16; i++){
        temp_v = (int64_t)(int16_t)pwd->h[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+256/16] = temp_v;
    }
    for (i = 0; i < 256/8; i++) {
        pwd->b[i] = lsx_sat_s_df(DF_HALF,temp_reg[i],7);
    }
}

void helper_lsx_xvssrarneni_h_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg[256/16];
    int64_t temp_v;
    for (i = 0; i < 256/32; i++){
        temp_v = (int64_t)(int32_t)pws->w[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/32; i++){
        temp_v = (int64_t)(int32_t)pwd->w[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+256/32] = temp_v;
    }
    for (i = 0; i < 256/16; i++) {
        pwd->h[i] = lsx_sat_s_df(DF_WORD,temp_reg[i],15);
    }
}


void helper_lsx_xvssrarneni_w_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg[256/32];
    int64_t temp_v;
    for (i = 0; i < 256/64; i++){
        temp_v = (int64_t)pws->d[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/64; i++){
        temp_v = (int64_t)pwd->d[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+256/64] = temp_v;
    }
    for (i = 0; i < 256/32; i++) {
        pwd->w[i] = lsx_sat_s_128(temp_reg[i],31);
    }
}


void helper_lsx_xvssrarneni_d_q(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    __int128_t temp_reg[4];
    __int128_t temp_v;
    temp_v = (__int128_t)pws->q[0];
    if (ui > 0)
        temp_v = lsx_round_nearest_even(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[0] = temp_v;
    temp_v = (__int128_t)pws->q[1];
    if (ui > 0)
        temp_v = lsx_round_nearest_even(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[1] = temp_v;


    temp_v = (__uint128_t)pwd->q[0];
    if (ui > 0)
       temp_v = lsx_round_nearest_even(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[2] = temp_v;

    temp_v = (__uint128_t)pwd->q[1];
    if (ui > 0)
       temp_v = lsx_round_nearest_even(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[3] = temp_v;

    pwd->d[0] = lsx_sat_s_128(temp_reg[0],63);
    pwd->d[1] = lsx_sat_s_128(temp_reg[1],63);
    pwd->d[2] = lsx_sat_s_128(temp_reg[2],63);
    pwd->d[3] = lsx_sat_s_128(temp_reg[3],63);
}

void helper_lsx_xvssrarneni_bu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg [256/8];
    int64_t temp_v;
    for (i = 0; i < 256/16; i++){
        temp_v = (int64_t)(int16_t)pws->h[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/16; i++){
        temp_v = (int64_t)(int16_t)pwd->h[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+256/16] = temp_v;
    }
    for (i = 0; i < 256/8; i++) {
        pwd->b[i] = lsx_sat_u_df(DF_HALF,temp_reg[i],7);
    }
}

void helper_lsx_xvssrarneni_hu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg[256/16];
    int64_t temp_v;
    for (i = 0; i < 256/32; i++){
        temp_v = (int64_t)(int32_t)pws->w[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/32; i++){
        temp_v = (int64_t)(int32_t)pwd->w[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+256/32] = temp_v;
    }
    for (i = 0; i < 256/16; i++) {
        pwd->h[i] = lsx_sat_u_df(DF_WORD,temp_reg[i],15);
    }
}


void helper_lsx_xvssrarneni_wu_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    int i;
    int64_t temp_reg[256/32];
    int64_t temp_v;
    for (i = 0; i < 256/64; i++){
        temp_v = (int64_t)pws->d[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i] = temp_v;
    }
    for (i = 0; i < 256/64; i++){
        temp_v = (int64_t)pwd->d[i];
        if (ui > 0)
            temp_v = lsx_round_nearest_even(temp_v,ui);
        temp_v = temp_v >> ui;
        temp_reg[i+256/64] = temp_v;
    }
    for (i = 0; i < 256/32; i++) {
        pwd->w[i] = lsx_sat_u_128(temp_reg[i],31);
    }
}


void helper_lsx_xvssrarneni_du_q(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    __int128_t temp_reg[4];
    __int128_t temp_v;
    temp_v = (__int128_t)pws->q[0];
    if (ui > 0)
        temp_v = lsx_round_nearest_even(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[0] = temp_v;
    temp_v = (__int128_t)pws->q[1];
    if (ui > 0)
        temp_v = lsx_round_nearest_even(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[1] = temp_v;


    temp_v = (__int128_t)pwd->q[0];
    if (ui > 0)
       temp_v = lsx_round_nearest_even(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[2] = temp_v;

    temp_v = (__int128_t)pwd->q[1];
    if (ui > 0)
       temp_v = lsx_round_nearest_even(temp_v,ui);
    temp_v = temp_v >> ui;
    temp_reg[3] = temp_v;

    pwd->d[0] = lsx_sat_u_128(temp_reg[0],63);
    pwd->d[1] = lsx_sat_u_128(temp_reg[1],63);
    pwd->d[2] = lsx_sat_u_128(temp_reg[2],63);
    pwd->d[3] = lsx_sat_u_128(temp_reg[3],63);
}










//-----------------------{SYK code}end--------------------/





////////////////////////// MUL ///////////////////////////////////
//DEBUGE USAGE
#define PRINT_BEFORE printf("%s:\n%08x|%08x|%08x|%08x|%08x|%08x|%08x|%08x\n%08x|%08x|%08x|%08x|%08x|%08x|%08x|%08x---\n",__FUNCTION__,pws->w[7],pws->w[6],pws->w[5],pws->w[4],pws->w[3],pws->w[2],pws->w[1],pws->w[0],pwt->w[7],pwt->w[6],pwt->w[5],pwt->w[4],pwt->w[3],pwt->w[2],pwt->w[1],pwt->w[0]);
#define PRINT_AFTER  printf("%08x|%08x|%08x|%08x|%08x|%08x|%08x|%08x\n\n",pwd->w[7],pwd->w[6],pwd->w[5],pwd->w[4],pwd->w[3],pwd->w[2],pwd->w[1],pwd->w[0]);
void helper_lsx_vmuh_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

	int i;
	for (i = 0; i < 128/8; i++) {
		int16_t res = pws->b[i] * pwt->b[i];
		pwd->b[i] = res >> 8;
	}
}

void helper_lsx_vmuh_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
	for (i = 0; i < 128/16; i++) {
		int32_t res = pws->h[i] * pwt->h[i];
		pwd->h[i] = res >> 16;
	}
}

void helper_lsx_vmuh_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
	for (i = 0; i < 128/32; i++) {
		int64_t res = (int64_t)pws->w[i] * (int64_t)pwt->w[i];
		pwd->w[i] = res >> 32;
	}
}

void helper_lsx_vmuh_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

	int i;
	for (i = 0; i < 128/64; i++) {
		__int128_t res = (__int128_t)pws->d[i] * (__int128_t)pwt->d[i];
		pwd->d[i] = res >> 64;
	}
}

void helper_lsx_vmuh_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
	for (i = 0; i < 128/8; i++) {
		uint16_t res = (uint8_t)pws->b[i] * (uint8_t)pwt->b[i];
		pwd->b[i] = res >> 8;
	}
}

void helper_lsx_vmuh_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
	for (i = 0; i < 128/16; i++) {
		uint32_t res = (uint16_t)pws->h[i] * (uint16_t)pwt->h[i];
		pwd->h[i] = res >> 16;
	}
}

void helper_lsx_vmuh_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
	for (i = 0; i < 128/32; i++) {
		uint64_t res = (uint64_t)(uint32_t)pws->w[i] * (uint64_t)(uint32_t)pwt->w[i];
		pwd->w[i] = res >> 32;
	}
}

void helper_lsx_vmuh_du(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
	for (i = 0; i < 128/64; i++) {
		__uint128_t res = (__uint128_t)(uint64_t)pws->d[i] * (__uint128_t)(uint64_t)pwt->d[i];
		pwd->d[i] = res >> 64;
	}
}

void helper_lsx_vmuh_bu_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
	for (i = 0; i < 128/8; i++) {
		int16_t res = (uint8_t)pws->b[i] * pwt->b[i];
		pwd->b[i] = res >> 8;
	}
}

void helper_lsx_vmuh_hu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
	for (i = 0; i < 128/16; i++) {
		int32_t res = (uint16_t)pws->h[i] * pwt->h[i];
		pwd->h[i] = res >> 16;
	}
}

void helper_lsx_vmuh_wu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
	for (i = 0; i < 128/32; i++) {
		int64_t res = (int64_t)(uint32_t)pws->w[i] * (int64_t)pwt->w[i];
		pwd->w[i] = res >> 32;
	}
}

void helper_lsx_vmuh_du_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
	for (i = 0; i < 128/64; i++) {
		__int128_t res = (__int128_t)(uint64_t)pws->d[i] * (__int128_t)pwt->d[i];
		pwd->d[i] = res >> 64;
	}
}

void helper_lsx_vmulxw_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
	for (i = 0; i < 128/8/2; i++) {
		int16_t res = pws->b[2*i+1] * pwt->b[2*i];
		pwd->h[i] = res;
	}
}

void helper_lsx_vmulxw_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
	for (i = 0; i < 128/16/2; i++) {
		int32_t res = pws->h[2*i+1] * pwt->h[2*i];
		pwd->w[i] = res;
	}
}

void helper_lsx_vmulxw_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
	for (i = 0; i < 128/32/2; i++) {
		int64_t res = (int64_t)pws->w[2*i+1] * (int64_t)pwt->w[2*i];
		pwd->d[i] = res;
	}
}

void helper_lsx_vmulxw_h_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
	for (i = 0; i < 128/8/2; i++) {
		uint16_t res = (uint8_t)pws->b[2*i+1] * (uint8_t)pwt->b[2*i];
		pwd->h[i] = res;
	}
}

void helper_lsx_vmulxw_w_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
	for (i = 0; i < 128/16/2; i++) {
		uint32_t res = (uint16_t)pws->h[2*i+1] * (uint16_t)pwt->h[2*i];
		pwd->w[i] = res;
	}
}

void helper_lsx_vmulxw_d_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
	for (i = 0; i < 128/32/2; i++) {
		uint64_t res = (uint64_t)(uint32_t)pws->w[2*i+1] * (uint64_t)(uint32_t)pwt->w[2*i];
		pwd->d[i] = res;
	}
}


void helper_lsx_xvmuh_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/8; i++) {
        int16_t res = pws->b[i] * pwt->b[i];
        pwd->b[i] = res >> 8;
    }
}

void helper_lsx_xvmuh_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/16; i++) {
        int32_t res = pws->h[i] * pwt->h[i];
        pwd->h[i] = res >> 16;
    }
}

void helper_lsx_xvmuh_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/32; i++) {
        int64_t res = (int64_t)pws->w[i] * (int64_t)pwt->w[i];
        pwd->w[i] = res >> 32;
    }
}

void helper_lsx_xvmuh_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/64; i++) {
        __int128_t res = (__int128_t)pws->d[i] * (__int128_t)pwt->d[i];
        pwd->d[i] = res >> 64;
    }
}

void helper_lsx_xvmuh_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/8; i++) {
        uint16_t res = (uint8_t)pws->b[i] * (uint8_t)pwt->b[i];
        pwd->b[i] = res >> 8;
    }
}

void helper_lsx_xvmuh_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/16; i++) {
        uint32_t res = (uint16_t)pws->h[i] * (uint16_t)pwt->h[i];
        pwd->h[i] = res >> 16;
    }
}

void helper_lsx_xvmuh_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/32; i++) {
        uint64_t res = (uint64_t)(uint32_t)pws->w[i] * (uint64_t)(uint32_t)pwt->w[i];
        pwd->w[i] = res >> 32;
    }
}

void helper_lsx_xvmuh_du(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/64; i++) {
        __uint128_t res = (__uint128_t)(uint64_t)pws->d[i] * (__uint128_t)(uint64_t)pwt->d[i];
        pwd->d[i] = res >> 64;
    }
}

void helper_lsx_xvmuh_bu_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/8; i++) {
        int16_t res = (uint8_t)pws->b[i] * pwt->b[i];
        pwd->b[i] = res >> 8;
    }
}

void helper_lsx_xvmuh_hu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/16; i++) {
        int32_t res = (uint16_t)pws->h[i] * pwt->h[i];
        pwd->h[i] = res >> 16;
    }
}

void helper_lsx_xvmuh_wu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/32; i++) {
        int64_t res = (int64_t)(uint32_t)pws->w[i] * (int64_t)pwt->w[i];
        pwd->w[i] = res >> 32;
    }
}

void helper_lsx_xvmuh_du_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/64; i++) {
        __int128_t res = (__int128_t)(uint64_t)pws->d[i] * (__int128_t)pwt->d[i];
        pwd->d[i] = res >> 64;
    }
}

void helper_lsx_xvmulxw_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/8/2; i++) {
        int16_t res = pws->b[2*i+1] * pwt->b[2*i];
        pwd->h[i] = res;
    }
}

void helper_lsx_xvmulxw_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/16/2; i++) {
        int32_t res = pws->h[2*i+1] * pwt->h[2*i];
        pwd->w[i] = res;
    }
}

void helper_lsx_xvmulxw_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/32/2; i++) {
        int64_t res = (int64_t)pws->w[2*i+1] * (int64_t)pwt->w[2*i];
        pwd->d[i] = res;
    }
}

void helper_lsx_xvmulxw_h_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/8/2; i++) {
        uint16_t res = (uint8_t)pws->b[2*i+1] * (uint8_t)pwt->b[2*i];
        pwd->h[i] = res;
    }
}

void helper_lsx_xvmulxw_w_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/16/2; i++) {
        uint32_t res = (uint16_t)pws->h[2*i+1] * (uint16_t)pwt->h[2*i];
        pwd->w[i] = res;
    }
}

void helper_lsx_xvmulxw_d_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/32/2; i++) {
        uint64_t res = (uint64_t)(uint32_t)pws->w[2*i+1] * (uint64_t)(uint32_t)pwt->w[2*i];
        pwd->d[i] = res;
    }
}

void helper_lsx_vmulwev_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
	for (i = 0; i < 128/8/2; i++) {
		int16_t res = pws->b[2*i] * pwt->b[2*i];
		pwd->h[i] = res;
	}
}

void helper_lsx_vmulwev_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
	for (i = 0; i < 128/16/2; i++) {
		int32_t res = pws->h[2*i] * pwt->h[2*i];
		pwd->w[i] = res;
	}
}

void helper_lsx_vmulwev_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32/2; i++) {
        int64_t res = (int64_t)pws->w[2*i] * (int64_t)pwt->w[2*i];
        pwd->d[i] = res;
    }
}

void helper_lsx_vmulwev_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64/2; i++) {
        __int128_t res = (__int128_t)pws->d[2*i] * (__int128_t)pwt->d[2*i];
        pwd->q[i] = res;
    }
}

void helper_lsx_vmulwod_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/8/2; i++) {
        int16_t res = pws->b[2*i+1] * pwt->b[2*i+1];
        pwd->h[i] = res;
    }
}

void helper_lsx_vmulwod_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16/2; i++) {
        int32_t res = pws->h[2*i+1] * pwt->h[2*i+1];
        pwd->w[i] = res;
    }
}

void helper_lsx_vmulwod_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32/2; i++) {
        int64_t res = (int64_t)pws->w[2*i+1] * (int64_t)pwt->w[2*i+1];
        pwd->d[i] = res;
    }
}

void helper_lsx_vmulwod_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64/2; i++) {
        __int128_t res = (__int128_t)pws->d[2*i+1] * (__int128_t)pwt->d[2*i+1];
        pwd->q[i] = res;
    }
}

void helper_lsx_vmulwl_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/8/2; i++) {
        int16_t res = pws->b[i] * pwt->b[i];
        tmp.h[i] = res;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vmulwl_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/16/2; i++) {
        int32_t res = pws->h[i] * pwt->h[i];
        tmp.w[i] = res;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vmulwl_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/32/2; i++) {
        int64_t res = (int64_t)pws->w[i] * (int64_t)pwt->w[i];
        tmp.d[i] = res;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vmulwl_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/64/2; i++) {
        __int128_t res = (__int128_t)pws->d[i] * (__int128_t)pwt->d[i];
        tmp.q[i] = res;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vmulwh_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/8/2; i++) {
        int16_t res = pws->b[i+8] * pwt->b[i+8];
        pwd->h[i] = res;
    }
}

void helper_lsx_vmulwh_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16/2; i++) {
        int32_t res = pws->h[i+4] * pwt->h[i+4];
        pwd->w[i] = res;
    }
}

void helper_lsx_vmulwh_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32/2; i++) {
        int64_t res = (int64_t)pws->w[i+2] * (int64_t)pwt->w[i+2];
        pwd->d[i] = res;
    }
}

void helper_lsx_vmulwh_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64/2; i++) {
        __int128_t res = (__int128_t)pws->d[i+1] * (__int128_t)pwt->d[i+1];
        pwd->q[i] = res;
    }
}

void helper_lsx_vmulwev_h_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/8/2; i++) {
        uint16_t res = (uint8_t)pws->b[2*i] * (uint8_t)pwt->b[2*i];
        pwd->h[i] = res;
    }
}

void helper_lsx_vmulwev_w_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16/2; i++) {
        uint32_t res = (uint16_t)pws->h[2*i] * (uint16_t)pwt->h[2*i];
        pwd->w[i] = res;
    }
}

void helper_lsx_vmulwev_d_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32/2; i++) {
        uint64_t res = (uint64_t)(uint32_t)pws->w[2*i] * (uint64_t)(uint32_t)pwt->w[2*i];
        pwd->d[i] = res;
    }
}

void helper_lsx_vmulwev_q_du(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64/2; i++) {
        __uint128_t res = (__uint128_t)(uint64_t)pws->d[2*i] * (__uint128_t)(uint64_t)pwt->d[2*i];
        pwd->q[i] = res;
    }
}

void helper_lsx_vmulwod_h_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/8/2; i++) {
        uint16_t res = (uint8_t)pws->b[2*i+1] * (uint8_t)pwt->b[2*i+1];
        pwd->h[i] = res;
    }
}

void helper_lsx_vmulwod_w_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16/2; i++) {
        uint32_t res = (uint16_t)pws->h[2*i+1] * (uint16_t)pwt->h[2*i+1];
        pwd->w[i] = res;
    }
}

void helper_lsx_vmulwod_d_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32/2; i++) {
        uint64_t res = (uint64_t)(uint32_t)pws->w[2*i+1] * (uint64_t)(uint32_t)pwt->w[2*i+1];
        pwd->d[i] = res;
    }
}

void helper_lsx_vmulwod_q_du(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64/2; i++) {
        __uint128_t res = (__uint128_t)(uint64_t)pws->d[2*i+1] * (__uint128_t)(uint64_t)pwt->d[2*i+1];
        pwd->q[i] = res;
    }
}

void helper_lsx_vmulwl_h_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/8/2; i++) {
        uint16_t res = (uint8_t)pws->b[i] * (uint8_t)pwt->b[i];
        tmp.h[i] = res;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vmulwl_w_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/16/2; i++) {
        uint32_t res = (uint16_t)pws->h[i] * (uint16_t)pwt->h[i];
        tmp.w[i] = res;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vmulwl_d_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/32/2; i++) {
        uint64_t res = (uint64_t)(uint32_t)pws->w[i] * (uint64_t)(uint32_t)pwt->w[i];
        tmp.d[i] = res;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vmulwl_q_du(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/64/2; i++) {
        __uint128_t res = (__uint128_t)(uint64_t)pws->d[i] * (__uint128_t)(uint64_t)pwt->d[i];
        tmp.q[i] = res;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vmulwh_h_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/8/2; i++) {
        uint16_t res = (uint8_t)pws->b[i+8] * (uint8_t)pwt->b[i+8];
        pwd->h[i] = res;
    }
}

void helper_lsx_vmulwh_w_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16/2; i++) {
        uint32_t res = (uint16_t)pws->h[i+4] * (uint16_t)pwt->h[i+4];
        pwd->w[i] = res;
    }
}

void helper_lsx_vmulwh_d_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32/2; i++) {
        uint64_t res = (uint64_t)(uint32_t)pws->w[i+2] * (uint64_t)(uint32_t)pwt->w[i+2];
        pwd->d[i] = res;
    }
}

void helper_lsx_vmulwh_q_du(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64/2; i++) {
        __uint128_t res = (__uint128_t)(uint64_t)pws->d[i+1] * (__uint128_t)(uint64_t)pwt->d[i+1];
        pwd->q[i] = res;
    }
}

void helper_lsx_vmulwev_h_bu_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/8/2; i++) {
        int16_t res = (uint8_t)pws->b[2*i] * pwt->b[2*i];
        pwd->h[i] = res;
    }
}

void helper_lsx_vmulwev_w_hu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16/2; i++) {
        int32_t res = (uint16_t)pws->h[2*i] * pwt->h[2*i];
        pwd->w[i] = res;
    }
}

void helper_lsx_vmulwev_d_wu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32/2; i++) {
        int64_t res = (int64_t)(uint32_t)pws->w[2*i] * (int64_t)pwt->w[2*i];
        pwd->d[i] = res;
    }
}

void helper_lsx_vmulwev_q_du_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64/2; i++) {
        __int128_t res = (__int128_t)(uint64_t)pws->d[2*i] * (__int128_t)pwt->d[2*i];
        pwd->q[i] = res;
    }
}

void helper_lsx_vmulwod_h_bu_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/8/2; i++) {
        int16_t res = (uint8_t)pws->b[2*i+1] * pwt->b[2*i+1];
        pwd->h[i] = res;
    }
}

void helper_lsx_vmulwod_w_hu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16/2; i++) {
        int32_t res = (uint16_t)pws->h[2*i+1] * pwt->h[2*i+1];
        pwd->w[i] = res;
    }
}

void helper_lsx_vmulwod_d_wu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32/2; i++) {
        int64_t res = (int64_t)(uint32_t)pws->w[2*i+1] * (int64_t)pwt->w[2*i+1];
        pwd->d[i] = res;
    }
}

void helper_lsx_vmulwod_q_du_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64/2; i++) {
        __int128_t res = (__int128_t)(uint64_t)pws->d[2*i+1] * (__int128_t)pwt->d[2*i+1];
        pwd->q[i] = res;
    }
}

void helper_lsx_vmulwl_h_bu_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/8/2; i++) {
        int16_t res = (uint8_t)pws->b[i] * pwt->b[i];
        tmp.h[i] = res;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vmulwl_w_hu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/16/2; i++) {
        int32_t res = (uint16_t)pws->h[i] * pwt->h[i];
        tmp.w[i] = res;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vmulwl_d_wu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/32/2; i++) {
        int64_t res = (int64_t)(uint32_t)pws->w[i] * (int64_t)pwt->w[i];
        tmp.d[i] = res;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vmulwl_q_du_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/64/2; i++) {
        __int128_t res = (__int128_t)(uint64_t)pws->d[i] * (__int128_t)pwt->d[i];
        tmp.q[i] = res;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vmulwh_h_bu_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/8/2; i++) {
        int16_t res = (uint8_t)pws->b[i+8] * pwt->b[i+8];
        pwd->h[i] = res;
    }
}

void helper_lsx_vmulwh_w_hu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16/2; i++) {
        int32_t res = (uint16_t)pws->h[i+4] * pwt->h[i+4];
        pwd->w[i] = res;
    }
}

void helper_lsx_vmulwh_d_wu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32/2; i++) {
        int64_t res = (int64_t)(uint32_t)pws->w[i+2] * (int64_t)pwt->w[i+2];
        pwd->d[i] = res;
    }
}

void helper_lsx_vmulwh_q_du_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64/2; i++) {
        __int128_t res = (__int128_t)(uint64_t)pws->d[i+1] * (__int128_t)pwt->d[i+1];
        pwd->q[i] = res;
    }
}

void helper_lsx_xvmulwev_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/8/2; i++) {
        int16_t res = pws->b[2*i] * pwt->b[2*i];
        pwd->h[i] = res;
    }
}

void helper_lsx_xvmulwev_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/16/2; i++) {
        int32_t res = pws->h[2*i] * pwt->h[2*i];
        pwd->w[i] = res;
    }
}

void helper_lsx_xvmulwev_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/32/2; i++) {
        int64_t res = (int64_t)pws->w[2*i] * (int64_t)pwt->w[2*i];
        pwd->d[i] = res;
    }
}

void helper_lsx_xvmulwev_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/64/2; i++) {
        __int128_t res = (__int128_t)pws->d[2*i] * (__int128_t)pwt->d[2*i];
        pwd->q[i] = res;
    }
}

void helper_lsx_xvmulwod_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/8/2; i++) {
        int16_t res = pws->b[2*i+1] * pwt->b[2*i+1];
        pwd->h[i] = res;
    }
}

void helper_lsx_xvmulwod_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/16/2; i++) {
        int32_t res = pws->h[2*i+1] * pwt->h[2*i+1];
        pwd->w[i] = res;
    }
}

void helper_lsx_xvmulwod_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/32/2; i++) {
        int64_t res = (int64_t)pws->w[2*i+1] * (int64_t)pwt->w[2*i+1];
        pwd->d[i] = res;
    }
}

void helper_lsx_xvmulwod_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/64/2; i++) {
        __int128_t res = (__int128_t)pws->d[2*i+1] * (__int128_t)pwt->d[2*i+1];
        pwd->q[i] = res;
    }
}

void helper_lsx_xvmulwl_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/8/2; i++) {
        int16_t res_1 = pws->b[i] * pwt->b[i];
        tmp.h[i] = res_1;
        int16_t res_2 = pws->b[i+16] * pwt->b[i+16];
        tmp.h[i+8] = res_2;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvmulwl_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/16/2; i++) {
        int32_t res_1 = pws->h[i] * pwt->h[i];
        tmp.w[i] = res_1;
        int32_t res_2 = pws->h[i+8] * pwt->h[i+8];
        tmp.w[i+4] = res_2;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvmulwl_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/32/2; i++) {
        int64_t res_1 = (int64_t)pws->w[i] * (int64_t)pwt->w[i];
        tmp.d[i] = res_1;
        int64_t res_2 = (int64_t)pws->w[i+4] * (int64_t)pwt->w[i+4];
        tmp.d[i+2] = res_2;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvmulwl_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/64/2; i++) {
        __int128_t res_1 = (__int128_t)pws->d[i] * (__int128_t)pwt->d[i];
        tmp.q[i] = res_1;
        __int128_t res_2 = (__int128_t)pws->d[i+2] * (__int128_t)pwt->d[i+2];
        tmp.q[i+1] = res_2;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvmulwh_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/8/2; i++) {
        int16_t res_1 = pws->b[i+8] * pwt->b[i+8];
        pwd->h[i] = res_1;
        int16_t res_2 = pws->b[i+24] * pwt->b[i+24];
        pwd->h[i+8] = res_2;
    }
}

void helper_lsx_xvmulwh_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16/2; i++) {
        int32_t res_1 = pws->h[i+4] * pwt->h[i+4];
        pwd->w[i] = res_1;
        int32_t res_2 = pws->h[i+12] * pwt->h[i+12];
        pwd->w[i+4] = res_2;
    }
}

void helper_lsx_xvmulwh_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32/2; i++) {
        int64_t res_1 = (int64_t)pws->w[i+2] * (int64_t)pwt->w[i+2];
        pwd->d[i] = res_1;
        int64_t res_2 = (int64_t)pws->w[i+6] * (int64_t)pwt->w[i+6];
        pwd->d[i+2] = res_2;
    }
}

void helper_lsx_xvmulwh_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64/2; i++) {
        __int128_t res_1 = (__int128_t)pws->d[i+1] * (__int128_t)pwt->d[i+1];
        pwd->q[i] = res_1;
        __int128_t res_2 = (__int128_t)pws->d[i+3] * (__int128_t)pwt->d[i+3];
        pwd->q[i+1] = res_2;
    }
}

void helper_lsx_xvmulwev_h_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/8/2; i++) {
        uint16_t res = (uint8_t)pws->b[2*i] * (uint8_t)pwt->b[2*i];
        pwd->h[i] = res;
    }
}

void helper_lsx_xvmulwev_w_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/16/2; i++) {
        uint32_t res = (uint16_t)pws->h[2*i] * (uint16_t)pwt->h[2*i];
        pwd->w[i] = res;
    }
}

void helper_lsx_xvmulwev_d_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/32/2; i++) {
        uint64_t res = (uint64_t)(uint32_t)pws->w[2*i] * (uint64_t)(uint32_t)pwt->w[2*i];
        pwd->d[i] = res;
    }
}

void helper_lsx_xvmulwev_q_du(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/64/2; i++) {
        __uint128_t res = (__uint128_t)(uint64_t)pws->d[2*i] * (__uint128_t)(uint64_t)pwt->d[2*i];
        pwd->q[i] = res;
    }
}

void helper_lsx_xvmulwod_h_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/8/2; i++) {
        uint16_t res = (uint8_t)pws->b[2*i+1] * (uint8_t)pwt->b[2*i+1];
        pwd->h[i] = res;
    }
}

void helper_lsx_xvmulwod_w_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/16/2; i++) {
        uint32_t res = (uint16_t)pws->h[2*i+1] * (uint16_t)pwt->h[2*i+1];
        pwd->w[i] = res;
    }
}

void helper_lsx_xvmulwod_d_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/32/2; i++) {
        uint64_t res = (uint64_t)(uint32_t)pws->w[2*i+1] * (uint64_t)(uint32_t)pwt->w[2*i+1];
        pwd->d[i] = res;
    }
}

void helper_lsx_xvmulwod_q_du(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/64/2; i++) {
        __uint128_t res = (__uint128_t)(uint64_t)pws->d[2*i+1] * (__uint128_t)(uint64_t)pwt->d[2*i+1];
        pwd->q[i] = res;
    }
}

void helper_lsx_xvmulwl_h_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/8/2; i++) {
        uint16_t res_1 = (uint8_t)pws->b[i] * (uint8_t)pwt->b[i];
        tmp.h[i] = res_1;
        uint16_t res_2 = (uint8_t)pws->b[i+16] *(uint8_t)pwt->b[i+16];
        tmp.h[i+8] = res_2;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvmulwl_w_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/16/2; i++) {
        uint32_t res_1 = (uint16_t)pws->h[i] * (uint16_t)pwt->h[i];
        tmp.w[i] = res_1;
        uint32_t res_2 = (uint16_t)pws->h[i+8] * (uint16_t)pwt->h[i+8];
        tmp.w[i+4] = res_2;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvmulwl_d_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/32/2; i++) {
        uint64_t res_1 = (uint64_t)(uint32_t)pws->w[i] * (uint64_t)(uint32_t)pwt->w[i];
        tmp.d[i] = res_1;
        uint64_t res_2 = (uint64_t)(uint32_t)pws->w[i+4] * (uint64_t)(uint32_t)pwt->w[i+4];
        tmp.d[i+2] = res_2;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvmulwl_q_du(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/64/2; i++) {
        __uint128_t res_1 = (__int128_t)(uint64_t)pws->d[i] * (__int128_t)(uint64_t)pwt->d[i];
        tmp.q[i] = res_1;
        __uint128_t res_2 = (__int128_t)(uint64_t)pws->d[i+2] * (__int128_t)(uint64_t)pwt->d[i+2];
        tmp.q[i+1] = res_2;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvmulwh_h_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/8/2; i++) {
        uint16_t res_1 = (uint8_t)pws->b[i+8] * (uint8_t)pwt->b[i+8];
        pwd->h[i] = res_1;
        uint16_t res_2 = (uint8_t)pws->b[i+24] * (uint8_t)pwt->b[i+24];
        pwd->h[i+8] = res_2;
    }
}

void helper_lsx_xvmulwh_w_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16/2; i++) {
        uint32_t res_1 = (uint16_t)pws->h[i+4] * (uint16_t)pwt->h[i+4];
        pwd->w[i] = res_1;
        uint32_t res_2 = (uint16_t)pws->h[i+12] * (uint16_t)pwt->h[i+12];
        pwd->w[i+4] = res_2;
    }
}

void helper_lsx_xvmulwh_d_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32/2; i++) {
        uint64_t res_1 = (uint64_t)(uint32_t)pws->w[i+2] * (uint64_t)(uint32_t)pwt->w[i+2];
        pwd->d[i] = res_1;
        uint64_t res_2 = (uint64_t)(uint32_t)pws->w[i+6] * (uint64_t)(uint32_t)pwt->w[i+6];
        pwd->d[i+2] = res_2;
    }
}

void helper_lsx_xvmulwh_q_du(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64/2; i++) {
        __uint128_t res_1 = (__int128_t)(uint64_t)pws->d[i+1] * (__int128_t)(uint64_t)pwt->d[i+1];
        pwd->q[i] = res_1;
        __uint128_t res_2 = (__int128_t)(uint64_t)pws->d[i+3] * (__int128_t)(uint64_t)pwt->d[i+3];
        pwd->q[i+1] = res_2;
    }
}

void helper_lsx_xvmulwev_h_bu_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/8/2; i++) {
        int16_t res = (uint8_t)pws->b[2*i] * pwt->b[2*i];
        pwd->h[i] = res;
    }
}

void helper_lsx_xvmulwev_w_hu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/16/2; i++) {
        int32_t res = (uint16_t)pws->h[2*i] * pwt->h[2*i];
        pwd->w[i] = res;
    }
}

void helper_lsx_xvmulwev_d_wu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/32/2; i++) {
        int64_t res = (int64_t)(uint32_t)pws->w[2*i] * (int64_t)pwt->w[2*i];
        pwd->d[i] = res;
    }
}

void helper_lsx_xvmulwev_q_du_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/64/2; i++) {
        __int128_t res = (__int128_t)(uint64_t)pws->d[2*i] * (__int128_t)pwt->d[2*i];
        pwd->q[i] = res;
    }
}

void helper_lsx_xvmulwod_h_bu_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/8/2; i++) {
        int16_t res = (uint8_t)pws->b[2*i+1] * pwt->b[2*i+1];
        pwd->h[i] = res;
    }
}

void helper_lsx_xvmulwod_w_hu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/16/2; i++) {
        int32_t res = (uint16_t)pws->h[2*i+1] * pwt->h[2*i+1];
        pwd->w[i] = res;
    }
}

void helper_lsx_xvmulwod_d_wu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/32/2; i++) {
        int64_t res = (int64_t)(uint32_t)pws->w[2*i+1] * (int64_t)pwt->w[2*i+1];
        pwd->d[i] = res;
    }
}

void helper_lsx_xvmulwod_q_du_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/64/2; i++) {
        __int128_t res = (__int128_t)(uint64_t)pws->d[2*i+1] * (__int128_t)pwt->d[2*i+1];
        pwd->q[i] = res;
    }
}

void helper_lsx_xvmulwl_h_bu_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/8/2; i++) {
        int16_t res_1 = (uint8_t)pws->b[i] * pwt->b[i];
        tmp.h[i] = res_1;
        int16_t res_2 = (uint8_t)pws->b[i+16] * pwt->b[i+16];
        tmp.h[i+8] = res_2;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvmulwl_w_hu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/16/2; i++) {
        int32_t res_1 = (uint16_t)pws->h[i] * pwt->h[i];
        tmp.w[i] = res_1;
        int32_t res_2 = (uint16_t)pws->h[i+8] * pwt->h[i+8];
        tmp.w[i+4] = res_2;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvmulwl_d_wu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/32/2; i++) {
        int64_t res_1 = (int64_t)(uint32_t)pws->w[i] * (int64_t)pwt->w[i];
        tmp.d[i] = res_1;
        int64_t res_2 = (int64_t)(uint32_t)pws->w[i+4] * (int64_t)pwt->w[i+4];
        tmp.d[i+2] = res_2;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvmulwl_q_du_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/64/2; i++) {
        __int128_t res_1 = (__int128_t)(uint64_t)pws->d[i] * (__int128_t)pwt->d[i];
        tmp.q[i] = res_1;
        __int128_t res_2 = (__int128_t)(uint64_t)pws->d[i+2] * (__int128_t)pwt->d[i+2];
        tmp.q[i+1] = res_2;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvmulwh_h_bu_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/8/2; i++) {
        int16_t res_1 = (uint8_t)pws->b[i+8] * pwt->b[i+8];
        pwd->h[i] = res_1;
        int16_t res_2 = (uint8_t)pws->b[i+24] * pwt->b[i+24];
        pwd->h[i+8] = res_2;
    }
}

void helper_lsx_xvmulwh_w_hu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16/2; i++) {
        int32_t res_1 = (uint16_t)pws->h[i+4] * pwt->h[i+4];
        pwd->w[i] = res_1;
        int32_t res_2 = (uint16_t)pws->h[i+12] * pwt->h[i+12];
        pwd->w[i+4] = res_2;
    }
}

void helper_lsx_xvmulwh_d_wu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32/2; i++) {
        int64_t res_1 = (int64_t)(uint32_t)pws->w[i+2] * (int64_t)pwt->w[i+2];
        pwd->d[i] = res_1;
        int64_t res_2 = (int64_t)(uint32_t)pws->w[i+6] * (int64_t)pwt->w[i+6];
        pwd->d[i+2] = res_2;
    }
}

void helper_lsx_xvmulwh_q_du_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64/2; i++) {
        __int128_t res_1 = (__int128_t)(uint64_t)pws->d[i+1] * (__int128_t)(uint64_t)pwt->d[i+1];
        pwd->q[i] = res_1;
        __int128_t res_2 = (__int128_t)(uint64_t)pws->d[i+3] * (__int128_t)(uint64_t)pwt->d[i+3];
        pwd->q[i+1] = res_2;
    }
}

void helper_lsx_vmaddwev_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/8/2; i++) {
        int16_t res = pws->b[2*i] * pwt->b[2*i];
        pwd->h[i] += res;
    }
}

void helper_lsx_vmaddwev_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16/2; i++) {
        int32_t res = pws->h[2*i] * pwt->h[2*i];
        pwd->w[i] += res;
    }
}

void helper_lsx_vmaddwev_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32/2; i++) {
        int64_t res = (int64_t)pws->w[2*i] * (int64_t)pwt->w[2*i];
        pwd->d[i] += res;
    }
}

void helper_lsx_vmaddwev_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64/2; i++) {
        __int128_t res = (__int128_t)pws->d[2*i] * (__int128_t)pwt->d[2*i];
        pwd->q[i] += res;
    }
}

void helper_lsx_vmaddwod_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/8/2; i++) {
        int16_t res = pws->b[2*i+1] * pwt->b[2*i+1];
        pwd->h[i] += res;
    }
}

void helper_lsx_vmaddwod_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16/2; i++) {
        int32_t res = pws->h[2*i+1] * pwt->h[2*i+1];
        pwd->w[i] += res;
    }
}

void helper_lsx_vmaddwod_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32/2; i++) {
        int64_t res = (int64_t)pws->w[2*i+1] * (int64_t)pwt->w[2*i+1];
        pwd->d[i] += res;
    }
}

void helper_lsx_vmaddwod_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64/2; i++) {
        __int128_t res = (__int128_t)pws->d[2*i+1] * (__int128_t)pwt->d[2*i+1];
        pwd->q[i] += res;
    }
}

void helper_lsx_vmaddwl_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/8/2; i++) {
        int16_t res = pws->b[i] * pwt->b[i];
        tmp.h[i] += res;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vmaddwl_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/16/2; i++) {
        int32_t res = pws->h[i] * pwt->h[i];
        tmp.w[i] += res;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vmaddwl_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/32/2; i++) {
        int64_t res = (int64_t)pws->w[i] * (int64_t)pwt->w[i];
        tmp.d[i] += res;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vmaddwl_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/64/2; i++) {
        __int128_t res = (__int128_t)pws->d[i] * (__int128_t)pwt->d[i];
        tmp.q[i] += res;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vmaddwh_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/8/2; i++) {
        int16_t res = pws->b[i+8] * pwt->b[i+8];
        pwd->h[i] += res;
    }
}

void helper_lsx_vmaddwh_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16/2; i++) {
        int32_t res = pws->h[i+4] * pwt->h[i+4];
        pwd->w[i] += res;
    }
}

void helper_lsx_vmaddwh_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32/2; i++) {
        int64_t res = (int64_t)pws->w[i+2] * (int64_t)pwt->w[i+2];
        pwd->d[i] += res;
    }
}

void helper_lsx_vmaddwh_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64/2; i++) {
        __int128_t res = (__int128_t)pws->d[i+1] * (__int128_t)pwt->d[i+1];
        pwd->q[i] += res;
    }
}

void helper_lsx_vmaddwev_h_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/8/2; i++) {
        uint16_t res = (uint8_t)pws->b[2*i] * (uint8_t)pwt->b[2*i];
        pwd->h[i] += res;
    }
}

void helper_lsx_vmaddwev_w_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16/2; i++) {
        uint32_t res = (uint16_t)pws->h[2*i] * (uint16_t)pwt->h[2*i];
        pwd->w[i] += res;
    }
}

void helper_lsx_vmaddwev_d_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32/2; i++) {
        uint64_t res = (uint64_t)(uint32_t)pws->w[2*i] * (uint64_t)(uint32_t)pwt->w[2*i];
        pwd->d[i] += res;
    }
}

void helper_lsx_vmaddwev_q_du(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64/2; i++) {
        __uint128_t res = (__uint128_t)(uint64_t)pws->d[2*i] * (__uint128_t)(uint64_t)pwt->d[2*i];
        pwd->q[i] += res;
    }
}

void helper_lsx_vmaddwod_h_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/8/2; i++) {
        uint16_t res = (uint8_t)pws->b[2*i+1] * (uint8_t)pwt->b[2*i+1];
        pwd->h[i] += res;
    }
}

void helper_lsx_vmaddwod_w_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16/2; i++) {
        uint32_t res = (uint16_t)pws->h[2*i+1] * (uint16_t)pwt->h[2*i+1];
        pwd->w[i] += res;
    }
}

void helper_lsx_vmaddwod_d_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32/2; i++) {
        uint64_t res = (uint64_t)(uint32_t)pws->w[2*i+1] * (uint64_t)(uint32_t)pwt->w[2*i+1];
        pwd->d[i] += res;
    }
}

void helper_lsx_vmaddwod_q_du(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64/2; i++) {
        __uint128_t res = (__uint128_t)(uint64_t)pws->d[2*i+1] * (__uint128_t)(uint64_t)pwt->d[2*i+1];
        pwd->q[i] += res;
    }
}

void helper_lsx_vmaddwl_h_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/8/2; i++) {
        uint16_t res = (uint8_t)pws->b[i] * (uint8_t)pwt->b[i];
        tmp.h[i] += res;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vmaddwl_w_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/16/2; i++) {
        uint32_t res = (uint16_t)pws->h[i] * (uint16_t)pwt->h[i];
        tmp.w[i] += res;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vmaddwl_d_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/32/2; i++) {
        uint64_t res = (uint64_t)(uint32_t)pws->w[i] * (uint64_t)(uint32_t)pwt->w[i];
        tmp.d[i] += res;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vmaddwl_q_du(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/64/2; i++) {
        __uint128_t res = (__uint128_t)(uint64_t)pws->d[i] * (__uint128_t)(uint64_t)pwt->d[i];
        tmp.q[i] += res;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vmaddwh_h_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/8/2; i++) {
        uint16_t res = (uint8_t)pws->b[i+8] * (uint8_t)pwt->b[i+8];
        pwd->h[i] += res;
    }
}

void helper_lsx_vmaddwh_w_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16/2; i++) {
        uint32_t res = (uint16_t)pws->h[i+4] * (uint16_t)pwt->h[i+4];
        pwd->w[i] += res;
    }
}

void helper_lsx_vmaddwh_d_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32/2; i++) {
        uint64_t res = (uint64_t)(uint32_t)pws->w[i+2] * (uint64_t)(uint32_t)pwt->w[i+2];
        pwd->d[i] += res;
    }
}

void helper_lsx_vmaddwh_q_du(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64/2; i++) {
        __uint128_t res = (__uint128_t)(uint64_t)pws->d[i+1] * (__uint128_t)(uint64_t)pwt->d[i+1];
        pwd->q[i] += res;
    }
}

void helper_lsx_vmaddwev_h_bu_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/8/2; i++) {
        int16_t res = (uint8_t)pws->b[2*i] * pwt->b[2*i];
        pwd->h[i] += res;
    }
}

void helper_lsx_vmaddwev_w_hu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16/2; i++) {
        int32_t res = (uint16_t)pws->h[2*i] * pwt->h[2*i];
        pwd->w[i] += res;
    }
}

void helper_lsx_vmaddwev_d_wu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32/2; i++) {
        int64_t res = (int64_t)(uint32_t)pws->w[2*i] * (int64_t)pwt->w[2*i];
        pwd->d[i] += res;
    }
}

void helper_lsx_vmaddwev_q_du_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64/2; i++) {
        __int128_t res = (__int128_t)(uint64_t)pws->d[2*i] * (__int128_t)pwt->d[2*i];
        pwd->q[i] += res;
    }
}

void helper_lsx_vmaddwod_h_bu_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/8/2; i++) {
        int16_t res = (uint8_t)pws->b[2*i+1] * pwt->b[2*i+1];
        pwd->h[i] += res;
    }
}

void helper_lsx_vmaddwod_w_hu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16/2; i++) {
        int32_t res = (uint16_t)pws->h[2*i+1] * pwt->h[2*i+1];
        pwd->w[i] += res;
    }
}

void helper_lsx_vmaddwod_d_wu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32/2; i++) {
        int64_t res = (int64_t)(uint32_t)pws->w[2*i+1] * (int64_t)pwt->w[2*i+1];
        pwd->d[i] += res;
    }
}

void helper_lsx_vmaddwod_q_du_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64/2; i++) {
        __int128_t res = (__int128_t)(uint64_t)pws->d[2*i+1] * (__int128_t)pwt->d[2*i+1];
        pwd->q[i] += res;
    }
}

void helper_lsx_vmaddwl_h_bu_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/8/2; i++) {
        int16_t res = (uint8_t)pws->b[i] * pwt->b[i];
        tmp.h[i] += res;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vmaddwl_w_hu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/16/2; i++) {
        int32_t res = (uint16_t)pws->h[i] * pwt->h[i];
        tmp.w[i] += res;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vmaddwl_d_wu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/32/2; i++) {
        int64_t res = (int64_t)(uint32_t)pws->w[i] * (int64_t)pwt->w[i];
        tmp.d[i] += res;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vmaddwl_q_du_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/64/2; i++) {
        __int128_t res = (__int128_t)(uint64_t)pws->d[i] * (__int128_t)pwt->d[i];
        tmp.q[i] += res;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vmaddwh_h_bu_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/8/2; i++) {
        int16_t res = (uint8_t)pws->b[i+8] * pwt->b[i+8];
        pwd->h[i] += res;
    }
}

void helper_lsx_vmaddwh_w_hu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16/2; i++) {
        int32_t res = (uint16_t)pws->h[i+4] * pwt->h[i+4];
        pwd->w[i] += res;
    }
}

void helper_lsx_vmaddwh_d_wu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32/2; i++) {
        int64_t res = (int64_t)(uint32_t)pws->w[i+2] * (int64_t)pwt->w[i+2];
        pwd->d[i] += res;
    }
}

void helper_lsx_vmaddwh_q_du_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64/2; i++) {
        __int128_t res = (__int128_t)(uint64_t)pws->d[i+1] * (__int128_t)pwt->d[i+1];
        pwd->q[i] += res;
    }
}

void helper_lsx_xvmaddwev_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/8/2; i++) {
        int16_t res = pws->b[2*i] * pwt->b[2*i];
        pwd->h[i] += res;
    }
}

void helper_lsx_xvmaddwev_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/16/2; i++) {
        int32_t res = pws->h[2*i] * pwt->h[2*i];
        pwd->w[i] += res;
    }
}

void helper_lsx_xvmaddwev_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/32/2; i++) {
        int64_t res = (int64_t)pws->w[2*i] * (int64_t)pwt->w[2*i];
        pwd->d[i] += res;
    }
}

void helper_lsx_xvmaddwev_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/64/2; i++) {
        __int128_t res = (__int128_t)pws->d[2*i] * (__int128_t)pwt->d[2*i];
        pwd->q[i] += res;
    }
}

void helper_lsx_xvmaddwod_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/8/2; i++) {
        int16_t res = pws->b[2*i+1] * pwt->b[2*i+1];
        pwd->h[i] += res;
    }
}

void helper_lsx_xvmaddwod_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/16/2; i++) {
        int32_t res = pws->h[2*i+1] * pwt->h[2*i+1];
        pwd->w[i] += res;
    }
}

void helper_lsx_xvmaddwod_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/32/2; i++) {
        int64_t res = (int64_t)pws->w[2*i+1] * (int64_t)pwt->w[2*i+1];
        pwd->d[i] += res;
    }
}

void helper_lsx_xvmaddwod_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/64/2; i++) {
        __int128_t res = (__int128_t)pws->d[2*i+1] * (__int128_t)pwt->d[2*i+1];
        pwd->q[i] += res;
    }
}

void helper_lsx_xvmaddwl_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/8/2; i++) {
        int16_t res_1 = pws->b[i] * pwt->b[i];
        tmp.h[i] += res_1;
        int16_t res_2 = pws->b[i+16] * pwt->b[i+16];
        tmp.h[i+8] += res_2;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvmaddwl_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/16/2; i++) {
        int32_t res_1 = pws->h[i] * pwt->h[i];
        tmp.w[i] += res_1;
        int32_t res_2 = pws->h[i+8] * pwt->h[i+8];
        tmp.w[i+4] += res_2;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvmaddwl_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/32/2; i++) {
        int64_t res_1 = (int64_t)pws->w[i] * (int64_t)pwt->w[i];
        tmp.d[i] += res_1;
        int64_t res_2 = (int64_t)pws->w[i+4] * (int64_t)pwt->w[i+4];
        tmp.d[i+2] += res_2;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvmaddwl_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/64/2; i++) {
        __int128_t res_1 = (__int128_t)pws->d[i] * (__int128_t)pwt->d[i];
        tmp.q[i] += res_1;
        __int128_t res_2 = (__int128_t)pws->d[i+2] * (__int128_t)pwt->d[i+2];
        tmp.q[i+1] += res_2;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvmaddwh_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/8/2; i++) {
        int16_t res_1 = pws->b[i+8] * pwt->b[i+8];
        pwd->h[i] += res_1;
        int16_t res_2 = pws->b[i+24] * pwt->b[i+24];
        pwd->h[i+8] += res_2;
    }
}

void helper_lsx_xvmaddwh_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16/2; i++) {
        int32_t res_1 = pws->h[i+4] * pwt->h[i+4];
        pwd->w[i] += res_1;
        int32_t res_2 = pws->h[i+12] * pwt->h[i+12];
        pwd->w[i+4] += res_2;
    }
}

void helper_lsx_xvmaddwh_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32/2; i++) {
        int64_t res_1 = (int64_t)pws->w[i+2] * (int64_t)pwt->w[i+2];
        pwd->d[i] += res_1;
        int64_t res_2 = (int64_t)pws->w[i+6] * (int64_t)pwt->w[i+6];
        pwd->d[i+2] += res_2;
    }
}

void helper_lsx_xvmaddwh_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64/2; i++) {
        __int128_t res_1 = (__int128_t)pws->d[i+1] * (__int128_t)pwt->d[i+1];
        pwd->q[i] += res_1;
        __int128_t res_2 = (__int128_t)pws->d[i+3] * (__int128_t)pwt->d[i+3];
        pwd->q[i+1] += res_2;
    }
}

void helper_lsx_xvmaddwev_h_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/8/2; i++) {
        uint16_t res = (uint8_t)pws->b[2*i] * (uint8_t)pwt->b[2*i];
        pwd->h[i] += res;
    }
}

void helper_lsx_xvmaddwev_w_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/16/2; i++) {
        uint32_t res = (uint16_t)pws->h[2*i] * (uint16_t)pwt->h[2*i];
        pwd->w[i] += res;
    }
}

void helper_lsx_xvmaddwev_d_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/32/2; i++) {
        uint64_t res = (uint64_t)(uint32_t)pws->w[2*i] * (uint64_t)(uint32_t)pwt->w[2*i];
        pwd->d[i] += res;
    }
}

void helper_lsx_xvmaddwev_q_du(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/64/2; i++) {
        __uint128_t res = (__uint128_t)(uint64_t)pws->d[2*i] * (__uint128_t)(uint64_t)pwt->d[2*i];
        pwd->q[i] += res;
    }
}

void helper_lsx_xvmaddwod_h_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/8/2; i++) {
        uint16_t res = (uint8_t)pws->b[2*i+1] * (uint8_t)pwt->b[2*i+1];
        pwd->h[i] += res;
    }
}

void helper_lsx_xvmaddwod_w_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/16/2; i++) {
        uint32_t res = (uint16_t)pws->h[2*i+1] * (uint16_t)pwt->h[2*i+1];
        pwd->w[i] += res;
    }
}

void helper_lsx_xvmaddwod_d_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/32/2; i++) {
        uint64_t res = (uint64_t)(uint32_t)pws->w[2*i+1] * (uint64_t)(uint32_t)pwt->w[2*i+1];
        pwd->d[i] += res;
    }
}

void helper_lsx_xvmaddwod_q_du(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/64/2; i++) {
        __uint128_t res = (__uint128_t)(uint64_t)pws->d[2*i+1] * (__uint128_t)(uint64_t)pwt->d[2*i+1];
        pwd->q[i] += res;
    }
}

void helper_lsx_xvmaddwl_h_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/8/2; i++) {
        uint16_t res_1 = (uint8_t)pws->b[i] * (uint8_t)pwt->b[i];
        tmp.h[i] += res_1;
        uint16_t res_2 = (uint8_t)pws->b[i+16] * (uint8_t)pwt->b[i+16];
        tmp.h[i+8] += res_2;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvmaddwl_w_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/16/2; i++) {
        uint32_t res_1 = (uint16_t)pws->h[i] * (uint16_t)pwt->h[i];
        tmp.w[i] += res_1;
        uint32_t res_2 = (uint16_t)pws->h[i+8] * (uint16_t)pwt->h[i+8];
        tmp.w[i+4] += res_2;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvmaddwl_d_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/32/2; i++) {
        uint64_t res_1 = (uint64_t)(uint32_t)pws->w[i] * (uint64_t)(uint32_t)pwt->w[i];
        tmp.d[i] += res_1;
        uint64_t res_2 = (uint64_t)(uint32_t)pws->w[i+4] * (uint64_t)(uint32_t)pwt->w[i+4];
        tmp.d[i+2] += res_2;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvmaddwl_q_du(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/64/2; i++) {
        __uint128_t res_1 = (__uint128_t)(uint64_t)pws->d[i] * (__uint128_t)(uint64_t)pwt->d[i];
        tmp.q[i] += res_1;
        __uint128_t res_2 = (__uint128_t)(uint64_t)pws->d[i+2] * (__uint128_t)(uint64_t)pwt->d[i+2];
        tmp.q[i+1] += res_2;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvmaddwh_h_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/8/2; i++) {
        uint16_t res_1 = (uint8_t)pws->b[i+8] * (uint8_t)pwt->b[i+8];
        pwd->h[i] += res_1;
        uint16_t res_2 = (uint8_t)pws->b[i+24] * (uint8_t)pwt->b[i+24];
        pwd->h[i+8] += res_2;
    }
}

void helper_lsx_xvmaddwh_w_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16/2; i++) {
        uint32_t res_1 = (uint16_t)pws->h[i+4] * (uint16_t)pwt->h[i+4];
        pwd->w[i] += res_1;
        uint32_t res_2 = (uint16_t)pws->h[i+12] * (uint16_t)pwt->h[i+12];
        pwd->w[i+4] += res_2;
    }
}

void helper_lsx_xvmaddwh_d_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32/2; i++) {
        uint64_t res_1 = (uint64_t)(uint32_t)pws->w[i+2] * (uint64_t)(uint32_t)pwt->w[i+2];
        pwd->d[i] += res_1;
        uint64_t res_2 = (uint64_t)(uint32_t)pws->w[i+6] * (uint64_t)(uint32_t)pwt->w[i+6];
        pwd->d[i+2] += res_2;
    }
}

void helper_lsx_xvmaddwh_q_du(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64/2; i++) {
        __uint128_t res_1 = (__uint128_t)(uint64_t)pws->d[i+1] * (__uint128_t)(uint64_t)pwt->d[i+1];
        pwd->q[i] += res_1;
        __uint128_t res_2 = (__uint128_t)(uint64_t)pws->d[i+3] * (__uint128_t)(uint64_t)pwt->d[i+3];
        pwd->q[i+1] += res_2;
    }
}

void helper_lsx_xvmaddwev_h_bu_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/8/2; i++) {
        int16_t res = (uint8_t)pws->b[2*i] * pwt->b[2*i];
        pwd->h[i] += res;
    }
}

void helper_lsx_xvmaddwev_w_hu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/16/2; i++) {
        int32_t res = (uint16_t)pws->h[2*i] * pwt->h[2*i];
        pwd->w[i] += res;
    }
}

void helper_lsx_xvmaddwev_d_wu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/32/2; i++) {
        int64_t res = (int64_t)(uint32_t)pws->w[2*i] * (int64_t)pwt->w[2*i];
        pwd->d[i] += res;
    }
}

void helper_lsx_xvmaddwev_q_du_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/64/2; i++) {
        __int128_t res = (__int128_t)(uint64_t)pws->d[2*i] * (__int128_t)pwt->d[2*i];
        pwd->q[i] += res;
    }
}

void helper_lsx_xvmaddwod_h_bu_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/8/2; i++) {
        int16_t res = (uint8_t)pws->b[2*i+1] * pwt->b[2*i+1];
        pwd->h[i] += res;
    }
}

void helper_lsx_xvmaddwod_w_hu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/16/2; i++) {
        int32_t res = (uint16_t)pws->h[2*i+1] * pwt->h[2*i+1];
        pwd->w[i] += res;
    }
}

void helper_lsx_xvmaddwod_d_wu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/32/2; i++) {
        int64_t res = (int64_t)(uint32_t)pws->w[2*i+1] * (int64_t)pwt->w[2*i+1];
        pwd->d[i] += res;
    }
}

void helper_lsx_xvmaddwod_q_du_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/64/2; i++) {
        __int128_t res = (__int128_t)(uint64_t)pws->d[2*i+1] * (__int128_t)pwt->d[2*i+1];
        pwd->q[i] += res;
    }
}

void helper_lsx_xvmaddwl_h_bu_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/8/2; i++) {
        int16_t res_1 = (uint8_t)pws->b[i] * pwt->b[i];
        tmp.h[i] += res_1;
        int16_t res_2 = (uint8_t)pws->b[i+16] * pwt->b[i+16];
        tmp.h[i+8] += res_2;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvmaddwl_w_hu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/16/2; i++) {
        int32_t res_1 = (uint16_t)pws->h[i] * pwt->h[i];
        tmp.w[i] += res_1;
        int32_t res_2 = (uint16_t)pws->h[i+8] * pwt->h[i+8];
        tmp.w[i+4] += res_2;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvmaddwl_d_wu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/32/2; i++) {
        int64_t res_1 = (int64_t)(uint32_t)pws->w[i] * (int64_t)pwt->w[i];
        tmp.d[i] += res_1;
        int64_t res_2 = (int64_t)(uint32_t)pws->w[i+4] * (int64_t)pwt->w[i+4];
        tmp.d[i+2] += res_2;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvmaddwl_q_du_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/64/2; i++) {
        __int128_t res_1 = (__int128_t)(uint64_t)pws->d[i] * (__int128_t)pwt->d[i];
        tmp.q[i] += res_1;
        __int128_t res_2 = (__int128_t)(uint64_t)pws->d[i+2] * (__int128_t)pwt->d[i+2];
        tmp.q[i+1] += res_2;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvmaddwh_h_bu_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/8/2; i++) {
        int16_t res_1 = (uint8_t)pws->b[i+8] * pwt->b[i+8];
        pwd->h[i] += res_1;
        int16_t res_2 = (uint8_t)pws->b[i+24] * pwt->b[i+24];
        pwd->h[i+8] += res_2;
    }
}

void helper_lsx_xvmaddwh_w_hu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16/2; i++) {
        int32_t res_1 = (uint16_t)pws->h[i+4] * pwt->h[i+4];
        pwd->w[i] += res_1;
        int32_t res_2 = (uint16_t)pws->h[i+12] * pwt->h[i+12];
        pwd->w[i+4] += res_2;
    }
}

void helper_lsx_xvmaddwh_d_wu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32/2; i++) {
        int64_t res_1 = (int64_t)(uint32_t)pws->w[i+2] * (int64_t)pwt->w[i+2];
        pwd->d[i] += res_1;
        int64_t res_2 = (int64_t)(uint32_t)pws->w[i+6] * (int64_t)pwt->w[i+6];
        pwd->d[i+2] += res_2;
    }
}

void helper_lsx_xvmaddwh_q_du_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64/2; i++) {
        __int128_t res_1 = (__int128_t)(uint64_t)pws->d[i+1] * (__int128_t)pwt->d[i+1];
        pwd->q[i] += res_1;
        __int128_t res_2 = (__int128_t)(uint64_t)pws->d[i+3] * (__int128_t)pwt->d[i+3];
        pwd->q[i+1] += res_2;
    }
}

void helper_lsx_vdp4_w_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

   	int i;
	wr_t tmp;
	for (i = 0; i < 128/8; i++) {
		tmp.h[i] = pws->b[i] * pwt->b[i];
	}
	for (i = 0; i < 128/8/4; i++) {
		int32_t res = tmp.h[i] + tmp.h[i+1] + tmp.h[i+2] + tmp.h[i+3];
		pwd->w[i] = res;
	}
}

void helper_lsx_vdp4_d_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    for (i = 0; i < 128/16; i++) {
        tmp.w[i] = pws->h[i] * pwt->h[i];
    }
    for (i = 0; i < 128/16/4; i++) {
        int64_t res = (int64_t)tmp.w[i] + (int64_t)tmp.w[i+1] + (int64_t)tmp.w[i+2] + (int64_t)tmp.w[i+3];
        pwd->d[i] = res;
    }
}

void helper_lsx_vdp4_q_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    for (i = 0; i < 128/32; i++) {
        tmp.d[i] = (int64_t)pws->w[i] * (int64_t)pwt->w[i];
    }
    for (i = 0; i < 128/32/4; i++) {
        __int128_t res = (__int128_t)tmp.d[i] + (__int128_t)tmp.d[i+1] + (__int128_t)tmp.d[i+2] + (__int128_t)tmp.d[i+3];
        pwd->q[i] = res;
    }
}

void helper_lsx_vdp4_w_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    for (i = 0; i < 128/8; i++) {
        tmp.h[i] = (uint8_t)pws->b[i] * (uint8_t)pwt->b[i];
    }
    for (i = 0; i < 128/8/4; i++) {
        uint32_t res = (uint16_t)tmp.h[i] + (uint16_t)tmp.h[i+1] + (uint16_t)tmp.h[i+2] + (uint16_t)tmp.h[i+3];
        pwd->w[i] = res;
    }
}

void helper_lsx_vdp4_d_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    for (i = 0; i < 128/16; i++) {
        tmp.w[i] = (uint16_t)pws->h[i] * (uint16_t)pwt->h[i];
    }
    for (i = 0; i < 128/16/4; i++) {
        uint64_t res = (uint64_t)(uint32_t)tmp.w[i] + (uint64_t)(uint32_t)tmp.w[i+1] + (uint64_t)(uint32_t)tmp.w[i+2] + (uint64_t)(uint32_t)tmp.w[i+3];
        pwd->d[i] = res;
    }
}

void helper_lsx_vdp4_q_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    for (i = 0; i < 128/32; i++) {
        tmp.d[i] = (int64_t)(uint32_t)pws->w[i] * (int64_t)(uint32_t)pwt->w[i];
    }
    for (i = 0; i < 128/32/4; i++) {
        __uint128_t res = (__uint128_t)(uint64_t)tmp.d[i] + (__uint128_t)(uint64_t)tmp.d[i+1] + (__uint128_t)(uint64_t)tmp.d[i+2] + (__uint128_t)(uint64_t)tmp.d[i+3];
        pwd->q[i] = res;
    }
}

void helper_lsx_vdp4_w_bu_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    for (i = 0; i < 128/8; i++) {
        tmp.h[i] = (uint8_t)pws->b[i] * pwt->b[i];
    }
    for (i = 0; i < 128/8/4; i++) {
        int32_t res = tmp.h[i] + tmp.h[i+1] + tmp.h[i+2] + tmp.h[i+3];
        pwd->w[i] = res;
    }
}

void helper_lsx_vdp4_d_hu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    for (i = 0; i < 128/16; i++) {
        tmp.w[i] = (uint16_t)pws->h[i] * pwt->h[i];
    }
    for (i = 0; i < 128/16/4; i++) {
        int64_t res = (int64_t)tmp.w[i] + (int64_t)tmp.w[i+1] + (int64_t)tmp.w[i+2] + (int64_t)tmp.w[i+3];
        pwd->d[i] = res;
    }
}

void helper_lsx_vdp4_q_wu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    for (i = 0; i < 128/32; i++) {
        tmp.d[i] = (int64_t)(uint32_t)pws->w[i] * (int64_t)pwt->w[i];
    }
    for (i = 0; i < 128/32/4; i++) {
        __int128_t res = (__int128_t)tmp.d[i] + (__int128_t)tmp.d[i+1] + (__int128_t)tmp.d[i+2] + (__int128_t)tmp.d[i+3];
        pwd->q[i] = res;
    }
}

void helper_lsx_vdp4add_w_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    for (i = 0; i < 128/8; i++) {
        tmp.h[i] = pws->b[i] * pwt->b[i];
    }
    for (i = 0; i < 128/8/4; i++) {
        int32_t res = tmp.h[i] + tmp.h[i+1] + tmp.h[i+2] + tmp.h[i+3];
        pwd->w[i] += res;
    }
}

void helper_lsx_vdp4add_d_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    for (i = 0; i < 128/16; i++) {
        tmp.w[i] = pws->h[i] * pwt->h[i];
    }
    for (i = 0; i < 128/16/4; i++) {
        int64_t res = (int64_t)tmp.w[i] + (int64_t)tmp.w[i+1] + (int64_t)tmp.w[i+2] + (int64_t)tmp.w[i+3];
        pwd->d[i] += res;
    }
}

void helper_lsx_vdp4add_q_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    for (i = 0; i < 128/32; i++) {
        tmp.d[i] = (int64_t)pws->w[i] * (int64_t)pwt->w[i];
    }
    for (i = 0; i < 128/32/4; i++) {
        __int128_t res = (__int128_t)tmp.d[i] + (__int128_t)tmp.d[i+1] + (__int128_t)tmp.d[i+2] + (__int128_t)tmp.d[i+3];
        pwd->q[i] += res;
    }
}

void helper_lsx_vdp4add_w_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    for (i = 0; i < 128/8; i++) {
        tmp.h[i] = (uint8_t)pws->b[i] * (uint8_t)pwt->b[i];
    }
    for (i = 0; i < 128/8/4; i++) {
        uint32_t res = (uint16_t)tmp.h[i] + (uint16_t)tmp.h[i+1] + (uint16_t)tmp.h[i+2] + (uint16_t)tmp.h[i+3];
        pwd->w[i] += res;
    }
}

void helper_lsx_vdp4add_d_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    for (i = 0; i < 128/16; i++) {
        tmp.w[i] = (uint16_t)pws->h[i] * (uint16_t)pwt->h[i];
    }
    for (i = 0; i < 128/16/4; i++) {
        uint64_t res = (uint64_t)(uint32_t)tmp.w[i] + (uint64_t)(uint32_t)tmp.w[i+1] + (uint64_t)(uint32_t)tmp.w[i+2] + (uint64_t)(uint32_t)tmp.w[i+3];
        pwd->d[i] += res;
    }
}

void helper_lsx_vdp4add_q_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    for (i = 0; i < 128/32; i++) {
        tmp.d[i] = (int64_t)(uint32_t)pws->w[i] * (int64_t)(uint32_t)pwt->w[i];
    }
    for (i = 0; i < 128/32/4; i++) {
        __uint128_t res = (__uint128_t)(uint64_t)tmp.d[i] + (__uint128_t)(uint64_t)tmp.d[i+1] + (__uint128_t)(uint64_t)tmp.d[i+2] + (__uint128_t)(uint64_t)tmp.d[i+3];
        pwd->q[i] += res;
    }
}

void helper_lsx_vdp4add_w_bu_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    for (i = 0; i < 128/8; i++) {
        tmp.h[i] = (uint8_t)pws->b[i] * pwt->b[i];
    }
    for (i = 0; i < 128/8/4; i++) {
        int32_t res = tmp.h[i] + tmp.h[i+1] + tmp.h[i+2] + tmp.h[i+3];
        pwd->w[i] += res;
    }
}

void helper_lsx_vdp4add_d_hu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    for (i = 0; i < 128/16; i++) {
        tmp.w[i] = (uint16_t)pws->h[i] * pwt->h[i];
    }
    for (i = 0; i < 128/16/4; i++) {
        int64_t res = (int64_t)tmp.w[i] + (int64_t)tmp.w[i+1] + (int64_t)tmp.w[i+2] + (int64_t)tmp.w[i+3];
        pwd->d[i] += res;
    }
}

void helper_lsx_vdp4add_q_wu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t tmp;
    for (i = 0; i < 128/32; i++) {
        tmp.d[i] = (int64_t)(uint32_t)pws->w[i] * (int64_t)pwt->w[i];
    }
    for (i = 0; i < 128/32/4; i++) {
        __int128_t res = (__int128_t)tmp.d[i] + (__int128_t)tmp.d[i+1] + (__int128_t)tmp.d[i+2] + (__int128_t)tmp.d[i+3];
        pwd->q[i] += res;
    }
}

typedef union {
    int8_t  b[2 * MSA_WRLEN / 8];
    int16_t h[2 * MSA_WRLEN / 16];
    int32_t w[2 * MSA_WRLEN / 32];
    int64_t d[2 * MSA_WRLEN / 64];
    __int128 q[2 * MSA_WRLEN / 128];
}wr_t_2;

void helper_lsx_xvdp4_w_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t_2 tmp;
    for (i = 0; i < 256/8; i++) {
        tmp.h[i] = pws->b[i] * pwt->b[i];
    }
    for (i = 0; i < 256/8/4; i++) {
        int32_t res = tmp.h[i] + tmp.h[i+1] + tmp.h[i+2] + tmp.h[i+3];
        pwd->w[i] = res;
    }
}

void helper_lsx_xvdp4_d_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t_2 tmp;
    for (i = 0; i < 256/16; i++) {
        tmp.w[i] = pws->h[i] * pwt->h[i];
    }
    for (i = 0; i < 256/16/4; i++) {
        int64_t res = (int64_t)tmp.w[i] + (int64_t)tmp.w[i+1] + (int64_t)tmp.w[i+2] + (int64_t)tmp.w[i+3];
        pwd->d[i] = res;
    }
}

void helper_lsx_xvdp4_q_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t_2 tmp;
    for (i = 0; i < 256/32; i++) {
        tmp.d[i] = (int64_t)pws->w[i] * (int64_t)pwt->w[i];
    }
    for (i = 0; i < 256/32/4; i++) {
        __int128_t res = (__int128_t)tmp.d[i] + (__int128_t)tmp.d[i+1] + (__int128_t)tmp.d[i+2] + (__int128_t)tmp.d[i+3];
        pwd->q[i] = res;
    }
}

void helper_lsx_xvdp4_w_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t_2 tmp;
    for (i = 0; i < 256/8; i++) {
        tmp.h[i] = (uint8_t)pws->b[i] * (uint8_t)pwt->b[i];
    }
    for (i = 0; i < 256/8/4; i++) {
        uint32_t res = (uint16_t)tmp.h[i] + (uint16_t)tmp.h[i+1] + (uint16_t)tmp.h[i+2] + (uint16_t)tmp.h[i+3];
        pwd->w[i] = res;
    }
}

void helper_lsx_xvdp4_d_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t_2 tmp;
    for (i = 0; i < 256/16; i++) {
        tmp.w[i] = (uint16_t)pws->h[i] * (uint16_t)pwt->h[i];
    }
    for (i = 0; i < 256/16/4; i++) {
        uint64_t res = (uint64_t)(uint32_t)tmp.w[i] + (uint64_t)(uint32_t)tmp.w[i+1] + (uint64_t)(uint32_t)tmp.w[i+2] + (uint64_t)(uint32_t)tmp.w[i+3];
        pwd->d[i] = res;
    }
}

void helper_lsx_xvdp4_q_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t_2 tmp;
    for (i = 0; i < 256/32; i++) {
        tmp.d[i] = (int64_t)(uint32_t)pws->w[i] * (int64_t)(uint32_t)pwt->w[i];
    }
    for (i = 0; i < 256/32/4; i++) {
        __uint128_t res = (__uint128_t)(uint64_t)tmp.d[i] + (__uint128_t)(uint64_t)tmp.d[i+1] + (__uint128_t)(uint64_t)tmp.d[i+2] + (__uint128_t)(uint64_t)tmp.d[i+3];
        pwd->q[i] = res;
    }
}

void helper_lsx_xvdp4_w_bu_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t_2 tmp;
    for (i = 0; i < 256/8; i++) {
        tmp.h[i] = (uint8_t)pws->b[i] * pwt->b[i];
    }
    for (i = 0; i < 256/8/4; i++) {
        int32_t res = tmp.h[i] + tmp.h[i+1] + tmp.h[i+2] + tmp.h[i+3];
        pwd->w[i] = res;
    }
}

void helper_lsx_xvdp4_d_hu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t_2 tmp;
    for (i = 0; i < 256/16; i++) {
        tmp.w[i] = (uint16_t)pws->h[i] * pwt->h[i];
    }
    for (i = 0; i < 256/16/4; i++) {
        int64_t res = (int64_t)tmp.w[i] + (int64_t)tmp.w[i+1] + (int64_t)tmp.w[i+2] + (int64_t)tmp.w[i+3];
        pwd->d[i] = res;
    }
}

void helper_lsx_xvdp4_q_wu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t_2 tmp;
    for (i = 0; i < 256/32; i++) {
        tmp.d[i] = (int64_t)(uint32_t)pws->w[i] * (int64_t)pwt->w[i];
    }
    for (i = 0; i < 256/32/4; i++) {
        __int128_t res = (__int128_t)tmp.d[i] + (__int128_t)tmp.d[i+1] + (__int128_t)tmp.d[i+2] + (__int128_t)tmp.d[i+3];
        pwd->q[i] = res;
    }
}

void helper_lsx_xvdp4add_w_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t_2 tmp;
    for (i = 0; i < 256/8; i++) {
        tmp.h[i] = pws->b[i] * pwt->b[i];
    }
    for (i = 0; i < 256/8/4; i++) {
        int32_t res = tmp.h[i] + tmp.h[i+1] + tmp.h[i+2] + tmp.h[i+3];
        pwd->w[i] += res;
    }
}

void helper_lsx_xvdp4add_d_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t_2 tmp;
    for (i = 0; i < 256/16; i++) {
        tmp.w[i] = pws->h[i] * pwt->h[i];
    }
    for (i = 0; i < 256/16/4; i++) {
        int64_t res = (int64_t)tmp.w[i] + (int64_t)tmp.w[i+1] + (int64_t)tmp.w[i+2] + (int64_t)tmp.w[i+3];
        pwd->d[i] += res;
    }
}

void helper_lsx_xvdp4add_q_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t_2 tmp;
    for (i = 0; i < 256/32; i++) {
        tmp.d[i] = (int64_t)pws->w[i] * (int64_t)pwt->w[i];
    }
    for (i = 0; i < 256/32/4; i++) {
        __int128_t res = (__int128_t)tmp.d[i] + (__int128_t)tmp.d[i+1] + (__int128_t)tmp.d[i+2] + (__int128_t)tmp.d[i+3];
        pwd->q[i] += res;
    }
}

void helper_lsx_xvdp4add_w_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t_2 tmp;
    for (i = 0; i < 256/8; i++) {
        tmp.h[i] = (uint8_t)pws->b[i] * (uint8_t)pwt->b[i];
    }
    for (i = 0; i < 256/8/4; i++) {
        uint32_t res = (uint16_t)tmp.h[i] + (uint16_t)tmp.h[i+1] + (uint16_t)tmp.h[i+2] + (uint16_t)tmp.h[i+3];
        pwd->w[i] += res;
    }
}

void helper_lsx_xvdp4add_d_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t_2 tmp;
    for (i = 0; i < 256/16; i++) {
        tmp.w[i] = (uint16_t)pws->h[i] * (uint16_t)pwt->h[i];
    }
    for (i = 0; i < 256/16/4; i++) {
        uint64_t res = (uint64_t)(uint32_t)tmp.w[i] + (uint64_t)(uint32_t)tmp.w[i+1] + (uint64_t)(uint32_t)tmp.w[i+2] + (uint64_t)(uint32_t)tmp.w[i+3];
        pwd->d[i] += res;
    }
}

void helper_lsx_xvdp4add_q_wu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t_2 tmp;
    for (i = 0; i < 256/32; i++) {
        tmp.d[i] = (int64_t)(uint32_t)pws->w[i] * (int64_t)(uint32_t)pwt->w[i];
    }
    for (i = 0; i < 256/32/4; i++) {
        __uint128_t res = (__uint128_t)(uint64_t)tmp.d[i] + (__uint128_t)(uint64_t)tmp.d[i+1] + (__uint128_t)(uint64_t)tmp.d[i+2] + (__uint128_t)(uint64_t)tmp.d[i+3];
        pwd->q[i] += res;
    }
}

void helper_lsx_xvdp4add_w_bu_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t_2 tmp;
    for (i = 0; i < 256/8; i++) {
        tmp.h[i] = (uint8_t)pws->b[i] * pwt->b[i];
    }
    for (i = 0; i < 256/8/4; i++) {
        int32_t res = tmp.h[i] + tmp.h[i+1] + tmp.h[i+2] + tmp.h[i+3];
        pwd->w[i] += res;
    }
}

void helper_lsx_xvdp4add_d_hu_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t_2 tmp;
    for (i = 0; i < 256/16; i++) {
        tmp.w[i] = (uint16_t)pws->h[i] * pwt->h[i];
    }
    for (i = 0; i < 256/16/4; i++) {
        int64_t res = (int64_t)tmp.w[i] + (int64_t)tmp.w[i+1] + (int64_t)tmp.w[i+2] + (int64_t)tmp.w[i+3];
        pwd->d[i] += res;
    }
}

void helper_lsx_xvdp4add_q_wu_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    wr_t_2 tmp;
    for (i = 0; i < 256/32; i++) {
        tmp.d[i] = (int64_t)(uint32_t)pws->w[i] * (int64_t)pwt->w[i];
    }
    for (i = 0; i < 256/32/4; i++) {
        __int128_t res = (__int128_t)tmp.d[i] + (__int128_t)tmp.d[i+1] + (__int128_t)tmp.d[i+2] + (__int128_t)tmp.d[i+3];
        pwd->q[i] += res;
    }
}

////////////////////////// MUL ///////////////////////////////////

void helper_lsx_vldi(CPULoongArchState *env, uint32_t wd, uint32_t ui)
{
    int sel = (ui >> 12) & 0x1;
    wr_t *pwd = &(env->fpr[wd].wr);
    uint32_t i;

    if (sel) {
        /* VSETI.D */
        int mode = (ui >> 8) & 0xf;
        uint64_t imm = (ui & 0xff);
#define EXPAND_BYTE(bit)    \
    ((uint64_t)(bit ? 0xff : 0))
        for (i = 0; i < 2; i++) {
            switch (mode) {
            case 0:
                pwd->d[i] = (imm << 32) | imm;
                break;
            case 1:
                pwd->d[i] = (imm << 24) | (imm << 8);
                break;
            case 2:
                pwd->d[i] = (imm << 48) | (imm << 16);
                break;
            case 3:
                pwd->d[i] = (imm << 56) | (imm << 24);
                break;
            case 4:
                pwd->d[i] = (imm << 48) | (imm << 32) |
                            (imm << 16) | imm;
                break;
            case 5:
                pwd->d[i] = (imm << 56) | (imm << 40) |
                            (imm << 24) | (imm << 8);
                break;
            case 6:
                pwd->d[i] = (imm << 40) | ((uint64_t)0xff << 32) |
                            (imm << 8) | 0xff;
                break;
            case 7:
                pwd->d[i] = (imm << 48) | ((uint64_t)0xffff << 32) |
                            (imm << 16) | 0xffff;
                break;
            case 8:
                pwd->d[i] = (imm << 56) | (imm << 48) | (imm << 40) |
                            (imm << 32) | (imm << 24) | (imm << 16) |
                            (imm << 8) | imm;
                break;
            case 9: {
                uint64_t b0,b1,b2,b3,b4,b5,b6,b7;
                b0 = imm & 0x1;
                b1 = (imm & 0x2) >> 1;
                b2 = (imm & 0x4) >> 2;
                b3 = (imm & 0x8) >> 3;
                b4 = (imm & 0x10) >> 4;
                b5 = (imm & 0x20) >> 5;
                b6 = (imm & 0x40) >> 6;
                b7 = (imm & 0x80) >> 7;
                pwd->d[i] = (EXPAND_BYTE(b7) << 56) |
                            (EXPAND_BYTE(b6) << 48) |
                            (EXPAND_BYTE(b5) << 40) |
                            (EXPAND_BYTE(b4) << 32) |
                            (EXPAND_BYTE(b3) << 24) |
                            (EXPAND_BYTE(b2) << 16) |
                            (EXPAND_BYTE(b1) <<  8) |
                            EXPAND_BYTE(b0);
                break;
            }
            case 10: {
                uint64_t b6, b7;
                uint64_t t0, t1;
                b6 = (imm & 0x40) >> 6;
                b7 = (imm & 0x80) >> 7;
                t0 = (imm & 0x3f);
                t1 = (b7 << 6) | ((1-b6) << 5) | (uint64_t)(b6 ? 0x1f : 0);
                pwd->d[i] = (t1 << 57) | (t0 << 51) |
                            (t1 << 25) | (t0 << 19);
                break;
            }
            case 11: {
                uint64_t b6,b7;
                uint64_t t0, t1;
                b6 = (imm & 0x40) >> 6;
                b7 = (imm & 0x80) >> 7;
                t0 = (imm & 0x3f);
                t1 = (b7 << 6) | ((1-b6) << 5) | (b6 ? 0x1f : 0);
                pwd->d[i] = (t1 << 25) | (t0 << 19);
                break;
            }
            case 12: {
                uint64_t b6,b7;
                uint64_t t0, t1;
                b6 = (imm & 0x40) >> 6;
                b7 = (imm & 0x80) >> 7;
                t0 = (imm & 0x3f);
                t1 = (b7 << 6) | ((1-b6) << 5) | (b6 ? 0x1f : 0);
                pwd->d[i] = (t1 << 54) | (t0 << 48);
                break;
            }
            default:
                assert(0);
            }
        }
#undef EXPAND_BYTE
    } else {
        /* LDI.df */
        uint32_t df = (ui >> 10) & 0x3;
        int32_t s10 = ((int32_t)(ui << 22)) >> 22;
        const int VLEN = 128;

        switch (df) {
        case DF_BYTE:
            for (i = 0; i < VLEN / DF_BITS(DF_BYTE); i++) {
                pwd->b[i] = (int8_t)s10;
            }
            break;
        case DF_HALF:
            for (i = 0; i < VLEN / DF_BITS(DF_HALF); i++) {
                pwd->h[i] = (int16_t)s10;
            }
            break;
        case DF_WORD:
            for (i = 0; i < VLEN / DF_BITS(DF_WORD); i++) {
                pwd->w[i] = (int32_t)s10;
            }
            break;
        case DF_DOUBLE:
            for (i = 0; i < VLEN / DF_BITS(DF_DOUBLE); i++) {
                pwd->d[i] = (int64_t)s10;
            }
           break;
        default:
            assert(0);
        }
    }
}

void helper_lsx_xvldi(CPULoongArchState *env, uint32_t wd, uint32_t ui)
{
    uint32_t df = (ui >> 10) & 0x3;
    int32_t s10 = ((int32_t)(ui << 22)) >> 22;

    wr_t *pwd = &(env->fpr[wd].wr);
    uint32_t i;
    const int VLEN = 256;

    switch (df) {
    case DF_BYTE:
        for (i = 0; i < VLEN / DF_BITS(DF_BYTE); i++) {
            pwd->b[i] = (int8_t)s10;
        }
        break;
    case DF_HALF:
        for (i = 0; i < VLEN / DF_BITS(DF_HALF); i++) {
            pwd->h[i] = (int16_t)s10;
        }
        break;
    case DF_WORD:
        for (i = 0; i < VLEN / DF_BITS(DF_WORD); i++) {
            pwd->w[i] = (int32_t)s10;
        }
        break;
    case DF_DOUBLE:
        for (i = 0; i < VLEN / DF_BITS(DF_DOUBLE); i++) {
            pwd->d[i] = (int64_t)s10;
        }
       break;
    default:
        assert(0);
    }
}

void helper_lsx_vrandsign_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int8_t r = 0x80;
    int m;
    for (i = 0; i < 128/8; i++) {
        r = r & pws->b[i];
    }
    m = pwt->b[0] & 0xf;
    pwd->b[m] = (r & 0x80) ? 0xff : 0x00;
}

void helper_lsx_vrandsign_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int16_t r = 0x8000;
    int m;
    for (i = 0; i < 128/16; i++) {
        r = r & pws->h[i];
    }
    m = pwt->h[0] & 0x7;
    pwd->h[m] = (r & 0x8000) ? 0xffff : 0x0000;
}

void helper_lsx_xvrandsign_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int8_t r = 0x80;
    int m;
    for (i = 0; i < 256/8; i++) {
        r = r & pws->b[i];
    }
    m = pwt->b[0] & 0x1f;
    pwd->b[m] = (r & 0x80) ? 0xff : 0x00;
}

void helper_lsx_xvrandsign_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int16_t r = 0x8000;
    int m;
    for (i = 0; i < 256/16; i++) {
        r = r & pws->h[i];
    }
    m = pwt->h[0] & 0xf;
    pwd->h[m] = (r & 0x8000) ? 0xffff : 0x0000;
}

void helper_lsx_vrorsign_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int8_t r = 0x0;
    int m;
    for (i = 0; i < 128/8; i++) {
        r = r | pws->b[i];
    }
    m = pwt->b[0] & 0xf;
    pwd->b[m] = (r & 0x80) ? 0xff : 0x00;
}

void helper_lsx_vrorsign_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int16_t r = 0x0;
    int m;
    for (i = 0; i < 128/16; i++) {
        r = r | pws->h[i];
    }
    m = pwt->h[0] & 0x7;
    pwd->h[m] = (r & 0x8000) ? 0xffff : 0x0000;
}

void helper_lsx_xvrorsign_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int8_t r1 = 0x0;
    int8_t r2 = 0x0;
    int m1, m2;
    for (i = 0; i < 128/8; i++) {
        r1 = r1 | pws->b[i];
        r2 = r2 | pws->b[i+16];
    }
    m1 = pwt->b[0] & 0xf;
    pwd->b[m1] = (r1 & 0x80) ? 0xff : 0x00;
    m2 = pwt->b[16] & 0xf;
    pwd->b[m2+16] = (r2 & 0x80) ? 0xff : 0x00;
}

void helper_lsx_xvrorsign_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int16_t r1 = 0x0;
    int16_t r2 = 0x0;
    int m1, m2;
    for (i = 0; i < 128/16; i++) {
        r1 = r1 | pws->h[i];
        r2 = r2 | pws->h[i+8];
    }
    m1 = pwt->b[0] & 0x7;
    pwd->h[m1] = (r1 & 0x8000) ? 0xffff : 0x0000;
    m2 = pwt->b[16] & 0x7;
    pwd->h[m2+8] = (r2 & 0x8000) ? 0xffff : 0x0000;
}

void helper_lsx_vfrstp_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int m;
    for (i = 0; i < 128/8; i++) {
        if(pws->b[i] & 0x80)
            break;
    }
    m = pwt->b[0] & 0xf;
    pwd->b[m] = (int8_t)i;
}

void helper_lsx_vfrstp_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int m;
    for (i = 0; i < 128/16; i++) {
        if(pws->h[i] & 0x8000)
            break;
    }
    m = pwt->h[0] & 0x7;
    pwd->h[m] = (int16_t)i;
}

void helper_lsx_xvfrstp_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int m;
    for (i = 0; i < 256/8; i++) {
        if(pws->b[i] & 0x80)
            break;
    }
    m = pwt->b[0] & 0x1f;
    pwd->b[m] = (int8_t)i;
}

void helper_lsx_xvfrstp_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int m;
    for (i = 0; i < 256/16; i++) {
        if(pws->h[i] & 0x8000)
            break;
    }
    m = pwt->h[0] & 0xf;
    pwd->h[m] = (int16_t)i;
}

void helper_lsx_vclrstrr_v(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    target_ulong rs = env->gpr[wt];

    int i;
    int m;
    m = rs & 0xf;
    for (i = 0; i < 128/8; i++) {
        if(i<=m)
            pwd->b[i] = pws->b[i];
        else
            pwd->b[i] = 0x00;
    }
}

void helper_lsx_xvclrstrr_v(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    target_ulong rs = env->gpr[wt];

    int i;
    int m;
    m = rs & 0x1f;
    for (i = 0; i < 256/8; i++) {
        if(i<=m)
            pwd->b[i] = pws->b[i];
        else
            pwd->b[i] = 0x00;
    }
}

void helper_lsx_vclrstrv_v(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int m;
    m = pwt->b[0] & 0xf;
    for (i = 0; i < 128/8; i++) {
        if(i<=m)
            pwd->b[i] = pws->b[i];
        else
            pwd->b[i] = 0x00;
    }
}

void helper_lsx_xvclrstrv_v(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    int m;
    m = pwt->b[0] & 0x1f;
    for (i = 0; i < 256/8; i++) {
        if(i<=m)
            pwd->b[i] = pws->b[i];
        else
            pwd->b[i] = 0x00;
    }
}

void helper_lsx_vadd_q(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0] = pws->q[0] + pwt->q[0];
}

void helper_lsx_vsub_q(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0] = pws->q[0] - pwt->q[0];
}

void helper_lsx_xvadd_q(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0] = pws->q[0] + pwt->q[0];
    pwd->q[1] = pws->q[1] + pwt->q[1];
}

void helper_lsx_xvsub_q(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0] = pws->q[0] - pwt->q[0];
    pwd->q[1] = pws->q[1] - pwt->q[1];
}

void helper_lsx_vsigncov_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/8; i++) {
        pwd->b[i] = (pws->b[i] == 0x00) ? 0 : (pws->b[i] < 0) ? -pwt->b[i] : pwt->b[i];
    }
}

void helper_lsx_vsigncov_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = (pws->h[i] == 0x00) ? 0 : (pws->h[i] < 0) ? -pwt->h[i] : pwt->h[i];
    }
}

void helper_lsx_vsigncov_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = (pws->w[i] == 0x00) ? 0 : (pws->w[i] < 0) ? -pwt->w[i] : pwt->w[i];
    }
}
void helper_lsx_vsigncov_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64; i++) {
        pwd->d[i] = (pws->d[i] == 0x00) ? 0 : (pws->d[i] < 0) ? -pwt->d[i] : pwt->d[i];
    }
}

void helper_lsx_xvsigncov_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/8; i++) {
        pwd->b[i] = (pws->b[i] == 0x00) ? 0 : (pws->b[i] < 0) ? -pwt->b[i] : pwt->b[i];
    }
}

void helper_lsx_xvsigncov_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/16; i++) {
        pwd->h[i] = (pws->h[i] == 0x00) ? 0 : (pws->h[i] < 0) ? -pwt->h[i] : pwt->h[i];
    }
}

void helper_lsx_xvsigncov_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/32; i++) {
        pwd->w[i] = (pws->w[i] == 0x00) ? 0 : (pws->w[i] < 0) ? -pwt->w[i] : pwt->w[i];
    }
}
void helper_lsx_xvsigncov_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/64; i++) {
        pwd->d[i] = (pws->d[i] == 0x00) ? 0 : (pws->d[i] < 0) ? -pwt->d[i] : pwt->d[i];
    }
}

void helper_lsx_vhadd4_h_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
	wr_t tmp;
	tmp.q[0] = pwd->q[0];
	tmp.q[1] = pwd->q[1];
    for (i = 0; i < 4; i++) {
        tmp.h[i] = (uint16_t)(uint8_t)pwt->b[4*i] + (uint16_t)(uint8_t)pwt->b[4*i+1] + (uint16_t)(uint8_t)pwt->b[4*i+2] + (uint16_t)(uint8_t)pwt->b[4*i+3];
        tmp.h[i+4] = (uint16_t)(uint8_t)pws->b[4*i] + (uint16_t)(uint8_t)pws->b[4*i+1] + (uint16_t)(uint8_t)pws->b[4*i+2] + (uint16_t)(uint8_t)pws->b[4*i+3];
    }
	pwd->q[0] = tmp.q[0];
	pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvhadd4_h_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
	wr_t tmp;
	tmp.q[0] = pwd->q[0];
	tmp.q[1] = pwd->q[1];
    for (i = 0; i < 4; i++) {
        tmp.h[i] = (uint16_t)(uint8_t)pwt->b[4*i] + (uint16_t)(uint8_t)pwt->b[4*i+1] + (uint16_t)(uint8_t)pwt->b[4*i+2] + (uint16_t)(uint8_t)pwt->b[4*i+3];
        tmp.h[i+4] = (uint16_t)(uint8_t)pws->b[4*i] + (uint16_t)(uint8_t)pws->b[4*i+1] + (uint16_t)(uint8_t)pws->b[4*i+2] + (uint16_t)(uint8_t)pws->b[4*i+3];
        tmp.h[i+8] = (uint16_t)(uint8_t)pwt->b[4*i+16] + (uint16_t)(uint8_t)pwt->b[4*i+1+16] + (uint16_t)(uint8_t)pwt->b[4*i+2+16] + (uint16_t)(uint8_t)pwt->b[4*i+3+16];
        tmp.h[i+12] = (uint16_t)(uint8_t)pws->b[4*i+16] + (uint16_t)(uint8_t)pws->b[4*i+1+16] + (uint16_t)(uint8_t)pws->b[4*i+2+16] + (uint16_t)(uint8_t)pws->b[4*i+3+16];
    }
	pwd->q[0] = tmp.q[0];
	pwd->q[1] = tmp.q[1];
}

void helper_lsx_vshuf4_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int j;
	wr_t tmp;
	tmp.q[0] = pwd->q[0];
	tmp.q[1] = pwd->q[1];
    for (j = 0; j < 4; j++){
        tmp.w[j] = pws->w[pwt->w[j] & 0x00000003];
    }
	pwd->q[0] = tmp.q[0];
	pwd->q[1] = tmp.q[1];
}

void helper_lsx_vshuf2_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int j;
	wr_t tmp;
	tmp.q[0] = pwd->q[0];
	tmp.q[1] = pwd->q[1];
    for (j = 0; j < 2; j++){
        tmp.d[j] = pws->d[pwt->d[j] & 0x0000000000000001];
    }
	pwd->q[0] = tmp.q[0];
	pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvshuf4_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int j;
	wr_t tmp;
	tmp.q[0] = pwd->q[0];
	tmp.q[1] = pwd->q[1];
    for (j = 0; j < 4; j++){
        tmp.w[j] = pws->w[pwt->w[j] & 0x00000003];
        tmp.w[j+4] = pws->w[(pwt->w[j+4] & 0x00000003)+4];
    }
	pwd->q[0] = tmp.q[0];
	pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvshuf2_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int j;
	wr_t tmp;
	tmp.q[0] = pwd->q[0];
	tmp.q[1] = pwd->q[1];
    for (j = 0; j < 2; j++){
        tmp.d[j] = pws->d[pwt->d[j] & 0x0000000000000001];
        tmp.d[j+2] = pws->d[(pwt->d[j+2] & 0x0000000000000001)+2];
    }
	pwd->q[0] = tmp.q[0];
	pwd->q[1] = tmp.q[1];
}
#if !defined(CONFIG_USER_ONLY)
#define MEMOP_IDX(DF)                                           \
        TCGMemOpIdx oi = make_memop_idx(MO_TE | DF | MO_UNALN,  \
                                        cpu_mmu_index(env, false));
#else
#define MEMOP_IDX(DF)
#endif

void helper_lsx_xvld(CPULoongArchState *env, uint32_t wd, target_ulong addr)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    MEMOP_IDX(DF_BYTE)
#if !defined(CONFIG_USER_ONLY)
    pwd->b[0]  = helper_ret_ldub_mmu(env, addr + (0  << DF_BYTE), oi, GETPC());
    pwd->b[1]  = helper_ret_ldub_mmu(env, addr + (1  << DF_BYTE), oi, GETPC());
    pwd->b[2]  = helper_ret_ldub_mmu(env, addr + (2  << DF_BYTE), oi, GETPC());
    pwd->b[3]  = helper_ret_ldub_mmu(env, addr + (3  << DF_BYTE), oi, GETPC());
    pwd->b[4]  = helper_ret_ldub_mmu(env, addr + (4  << DF_BYTE), oi, GETPC());
    pwd->b[5]  = helper_ret_ldub_mmu(env, addr + (5  << DF_BYTE), oi, GETPC());
    pwd->b[6]  = helper_ret_ldub_mmu(env, addr + (6  << DF_BYTE), oi, GETPC());
    pwd->b[7]  = helper_ret_ldub_mmu(env, addr + (7  << DF_BYTE), oi, GETPC());
    pwd->b[8]  = helper_ret_ldub_mmu(env, addr + (8  << DF_BYTE), oi, GETPC());
    pwd->b[9]  = helper_ret_ldub_mmu(env, addr + (9  << DF_BYTE), oi, GETPC());
    pwd->b[10] = helper_ret_ldub_mmu(env, addr + (10 << DF_BYTE), oi, GETPC());
    pwd->b[11] = helper_ret_ldub_mmu(env, addr + (11 << DF_BYTE), oi, GETPC());
    pwd->b[12] = helper_ret_ldub_mmu(env, addr + (12 << DF_BYTE), oi, GETPC());
    pwd->b[13] = helper_ret_ldub_mmu(env, addr + (13 << DF_BYTE), oi, GETPC());
    pwd->b[14] = helper_ret_ldub_mmu(env, addr + (14 << DF_BYTE), oi, GETPC());
    pwd->b[15] = helper_ret_ldub_mmu(env, addr + (15 << DF_BYTE), oi, GETPC());
    pwd->b[16] = helper_ret_ldub_mmu(env, addr + (16 << DF_BYTE), oi, GETPC());
    pwd->b[17] = helper_ret_ldub_mmu(env, addr + (17 << DF_BYTE), oi, GETPC());
    pwd->b[18] = helper_ret_ldub_mmu(env, addr + (18 << DF_BYTE), oi, GETPC());
    pwd->b[19] = helper_ret_ldub_mmu(env, addr + (19 << DF_BYTE), oi, GETPC());
    pwd->b[20] = helper_ret_ldub_mmu(env, addr + (20 << DF_BYTE), oi, GETPC());
    pwd->b[21] = helper_ret_ldub_mmu(env, addr + (21 << DF_BYTE), oi, GETPC());
    pwd->b[22] = helper_ret_ldub_mmu(env, addr + (22 << DF_BYTE), oi, GETPC());
    pwd->b[23] = helper_ret_ldub_mmu(env, addr + (23 << DF_BYTE), oi, GETPC());
    pwd->b[24] = helper_ret_ldub_mmu(env, addr + (24 << DF_BYTE), oi, GETPC());
    pwd->b[25] = helper_ret_ldub_mmu(env, addr + (25 << DF_BYTE), oi, GETPC());
    pwd->b[26] = helper_ret_ldub_mmu(env, addr + (26 << DF_BYTE), oi, GETPC());
    pwd->b[27] = helper_ret_ldub_mmu(env, addr + (27 << DF_BYTE), oi, GETPC());
    pwd->b[28] = helper_ret_ldub_mmu(env, addr + (28 << DF_BYTE), oi, GETPC());
    pwd->b[29] = helper_ret_ldub_mmu(env, addr + (29 << DF_BYTE), oi, GETPC());
    pwd->b[30] = helper_ret_ldub_mmu(env, addr + (30 << DF_BYTE), oi, GETPC());
    pwd->b[31] = helper_ret_ldub_mmu(env, addr + (31 << DF_BYTE), oi, GETPC());
#else
    pwd->b[0]  = cpu_ldub_data(env, addr + (0  << DF_BYTE));
    pwd->b[1]  = cpu_ldub_data(env, addr + (1  << DF_BYTE));
    pwd->b[2]  = cpu_ldub_data(env, addr + (2  << DF_BYTE));
    pwd->b[3]  = cpu_ldub_data(env, addr + (3  << DF_BYTE));
    pwd->b[4]  = cpu_ldub_data(env, addr + (4  << DF_BYTE));
    pwd->b[5]  = cpu_ldub_data(env, addr + (5  << DF_BYTE));
    pwd->b[6]  = cpu_ldub_data(env, addr + (6  << DF_BYTE));
    pwd->b[7]  = cpu_ldub_data(env, addr + (7  << DF_BYTE));
    pwd->b[8]  = cpu_ldub_data(env, addr + (8  << DF_BYTE));
    pwd->b[9]  = cpu_ldub_data(env, addr + (9  << DF_BYTE));
    pwd->b[10] = cpu_ldub_data(env, addr + (10 << DF_BYTE));
    pwd->b[11] = cpu_ldub_data(env, addr + (11 << DF_BYTE));
    pwd->b[12] = cpu_ldub_data(env, addr + (12 << DF_BYTE));
    pwd->b[13] = cpu_ldub_data(env, addr + (13 << DF_BYTE));
    pwd->b[14] = cpu_ldub_data(env, addr + (14 << DF_BYTE));
    pwd->b[15] = cpu_ldub_data(env, addr + (15 << DF_BYTE));
    pwd->b[16] = cpu_ldub_data(env, addr + (16 << DF_BYTE));
    pwd->b[17] = cpu_ldub_data(env, addr + (17 << DF_BYTE));
    pwd->b[18] = cpu_ldub_data(env, addr + (18 << DF_BYTE));
    pwd->b[19] = cpu_ldub_data(env, addr + (19 << DF_BYTE));
    pwd->b[20] = cpu_ldub_data(env, addr + (20 << DF_BYTE));
    pwd->b[21] = cpu_ldub_data(env, addr + (21 << DF_BYTE));
    pwd->b[22] = cpu_ldub_data(env, addr + (22 << DF_BYTE));
    pwd->b[23] = cpu_ldub_data(env, addr + (23 << DF_BYTE));
    pwd->b[24] = cpu_ldub_data(env, addr + (24 << DF_BYTE));
    pwd->b[25] = cpu_ldub_data(env, addr + (25 << DF_BYTE));
    pwd->b[26] = cpu_ldub_data(env, addr + (26 << DF_BYTE));
    pwd->b[27] = cpu_ldub_data(env, addr + (27 << DF_BYTE));
    pwd->b[28] = cpu_ldub_data(env, addr + (28 << DF_BYTE));
    pwd->b[29] = cpu_ldub_data(env, addr + (29 << DF_BYTE));
    pwd->b[30] = cpu_ldub_data(env, addr + (30 << DF_BYTE));
    pwd->b[31] = cpu_ldub_data(env, addr + (31 << DF_BYTE));
#endif
}

#define B_PAGESPAN(x) \
        ((((x) & ~TARGET_PAGE_MASK) + B_WRLEN / 8 - 1) >= TARGET_PAGE_SIZE)

static inline void ensure_b_writable_pages(CPULoongArchState *env,
                                         target_ulong addr,
                                         int mmu_idx,
                                         uintptr_t retaddr)
{
#ifndef CONFIG_USER_ONLY
    /* FIXME: Probe the actual accesses (pass and use a size) */
    if (unlikely(B_PAGESPAN(addr))) {
        /* first page */
        probe_write(env, addr, 0, mmu_idx, retaddr);
        /* second page */
        addr = (addr & TARGET_PAGE_MASK) + TARGET_PAGE_SIZE;
        probe_write(env, addr, 0, mmu_idx, retaddr);
    }
#endif
}

#define H_PAGESPAN(x) \
        ((((x) & ~TARGET_PAGE_MASK) + H_WRLEN / 8 - 1) >= TARGET_PAGE_SIZE)

static inline void ensure_h_writable_pages(CPULoongArchState *env,
                                         target_ulong addr,
                                         int mmu_idx,
                                         uintptr_t retaddr)
{
#ifndef CONFIG_USER_ONLY
    /* FIXME: Probe the actual accesses (pass and use a size) */
    if (unlikely(H_PAGESPAN(addr))) {
        /* first page */
        probe_write(env, addr, 0, mmu_idx, retaddr);
        /* second page */
        addr = (addr & TARGET_PAGE_MASK) + TARGET_PAGE_SIZE;
        probe_write(env, addr, 0, mmu_idx, retaddr);
    }
#endif
}

#define W_PAGESPAN(x) \
        ((((x) & ~TARGET_PAGE_MASK) + W_WRLEN / 8 - 1) >= TARGET_PAGE_SIZE)

static inline void ensure_w_writable_pages(CPULoongArchState *env,
                                         target_ulong addr,
                                         int mmu_idx,
                                         uintptr_t retaddr)
{
#ifndef CONFIG_USER_ONLY
    /* FIXME: Probe the actual accesses (pass and use a size) */
    if (unlikely(W_PAGESPAN(addr))) {
        /* first page */
        probe_write(env, addr, 0, mmu_idx, retaddr);
        /* second page */
        addr = (addr & TARGET_PAGE_MASK) + TARGET_PAGE_SIZE;
        probe_write(env, addr, 0, mmu_idx, retaddr);
    }
#endif
}

#define D_PAGESPAN(x) \
        ((((x) & ~TARGET_PAGE_MASK) + D_WRLEN / 8 - 1) >= TARGET_PAGE_SIZE)

static inline void ensure_d_writable_pages(CPULoongArchState *env,
                                         target_ulong addr,
                                         int mmu_idx,
                                         uintptr_t retaddr)
{
#ifndef CONFIG_USER_ONLY
    /* FIXME: Probe the actual accesses (pass and use a size) */
    if (unlikely(D_PAGESPAN(addr))) {
        /* first page */
        probe_write(env, addr, 0, mmu_idx, retaddr);
        /* second page */
        addr = (addr & TARGET_PAGE_MASK) + TARGET_PAGE_SIZE;
        probe_write(env, addr, 0, mmu_idx, retaddr);
    }
#endif
}

#define LSX_PAGESPAN(x) \
        ((((x) & ~TARGET_PAGE_MASK) + LSX_WRLEN / 8 - 1) >= TARGET_PAGE_SIZE)

static inline void ensure_lsx_writable_pages(CPULoongArchState *env,
                                         target_ulong addr,
                                         int mmu_idx,
                                         uintptr_t retaddr)
{
#ifndef CONFIG_USER_ONLY
    /* FIXME: Probe the actual accesses (pass and use a size) */
    if (unlikely(LSX_PAGESPAN(addr))) {
        /* first page */
        probe_write(env, addr, 0, mmu_idx, retaddr);
        /* second page */
        addr = (addr & TARGET_PAGE_MASK) + TARGET_PAGE_SIZE;
        probe_write(env, addr, 0, mmu_idx, retaddr);
    }
#endif
}

#define LASX_PAGESPAN(x) \
        ((((x) & ~TARGET_PAGE_MASK) + LASX_WRLEN / 8 - 1) >= TARGET_PAGE_SIZE)

static inline void ensure_lasx_writable_pages(CPULoongArchState *env,
                                         target_ulong addr,
                                         int mmu_idx,
                                         uintptr_t retaddr)
{
#ifndef CONFIG_USER_ONLY
    /* FIXME: Probe the actual accesses (pass and use a size) */
    if (unlikely(LASX_PAGESPAN(addr))) {
        /* first page */
        probe_write(env, addr, 0, mmu_idx, retaddr);
        /* second page */
        addr = (addr & TARGET_PAGE_MASK) + TARGET_PAGE_SIZE;
        probe_write(env, addr, 0, mmu_idx, retaddr);
    }
#endif
}

void helper_lsx_xvst(CPULoongArchState *env, uint32_t wd, target_ulong addr)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    int mmu_idx = cpu_mmu_index(env, false);

    MEMOP_IDX(DF_BYTE)
    ensure_lasx_writable_pages(env, addr, mmu_idx, GETPC());
#if !defined(CONFIG_USER_ONLY)
    helper_ret_stb_mmu(env, addr + (0  << DF_BYTE), pwd->b[0],  oi, GETPC());
    helper_ret_stb_mmu(env, addr + (1  << DF_BYTE), pwd->b[1],  oi, GETPC());
    helper_ret_stb_mmu(env, addr + (2  << DF_BYTE), pwd->b[2],  oi, GETPC());
    helper_ret_stb_mmu(env, addr + (3  << DF_BYTE), pwd->b[3],  oi, GETPC());
    helper_ret_stb_mmu(env, addr + (4  << DF_BYTE), pwd->b[4],  oi, GETPC());
    helper_ret_stb_mmu(env, addr + (5  << DF_BYTE), pwd->b[5],  oi, GETPC());
    helper_ret_stb_mmu(env, addr + (6  << DF_BYTE), pwd->b[6],  oi, GETPC());
    helper_ret_stb_mmu(env, addr + (7  << DF_BYTE), pwd->b[7],  oi, GETPC());
    helper_ret_stb_mmu(env, addr + (8  << DF_BYTE), pwd->b[8],  oi, GETPC());
    helper_ret_stb_mmu(env, addr + (9  << DF_BYTE), pwd->b[9],  oi, GETPC());
    helper_ret_stb_mmu(env, addr + (10 << DF_BYTE), pwd->b[10], oi, GETPC());
    helper_ret_stb_mmu(env, addr + (11 << DF_BYTE), pwd->b[11], oi, GETPC());
    helper_ret_stb_mmu(env, addr + (12 << DF_BYTE), pwd->b[12], oi, GETPC());
    helper_ret_stb_mmu(env, addr + (13 << DF_BYTE), pwd->b[13], oi, GETPC());
    helper_ret_stb_mmu(env, addr + (14 << DF_BYTE), pwd->b[14], oi, GETPC());
    helper_ret_stb_mmu(env, addr + (15 << DF_BYTE), pwd->b[15], oi, GETPC());
    helper_ret_stb_mmu(env, addr + (16 << DF_BYTE), pwd->b[16], oi, GETPC());
    helper_ret_stb_mmu(env, addr + (17 << DF_BYTE), pwd->b[17], oi, GETPC());
    helper_ret_stb_mmu(env, addr + (18 << DF_BYTE), pwd->b[18], oi, GETPC());
    helper_ret_stb_mmu(env, addr + (19 << DF_BYTE), pwd->b[19], oi, GETPC());
    helper_ret_stb_mmu(env, addr + (20 << DF_BYTE), pwd->b[20], oi, GETPC());
    helper_ret_stb_mmu(env, addr + (21 << DF_BYTE), pwd->b[21], oi, GETPC());
    helper_ret_stb_mmu(env, addr + (22 << DF_BYTE), pwd->b[22], oi, GETPC());
    helper_ret_stb_mmu(env, addr + (23 << DF_BYTE), pwd->b[23], oi, GETPC());
    helper_ret_stb_mmu(env, addr + (24 << DF_BYTE), pwd->b[24], oi, GETPC());
    helper_ret_stb_mmu(env, addr + (25 << DF_BYTE), pwd->b[25], oi, GETPC());
    helper_ret_stb_mmu(env, addr + (26 << DF_BYTE), pwd->b[26], oi, GETPC());
    helper_ret_stb_mmu(env, addr + (27 << DF_BYTE), pwd->b[27], oi, GETPC());
    helper_ret_stb_mmu(env, addr + (28 << DF_BYTE), pwd->b[28], oi, GETPC());
    helper_ret_stb_mmu(env, addr + (29 << DF_BYTE), pwd->b[29], oi, GETPC());
    helper_ret_stb_mmu(env, addr + (30 << DF_BYTE), pwd->b[30], oi, GETPC());
    helper_ret_stb_mmu(env, addr + (31 << DF_BYTE), pwd->b[31], oi, GETPC());
#else
    cpu_stb_data(env, addr + (0  << DF_BYTE), pwd->b[0]);
    cpu_stb_data(env, addr + (1  << DF_BYTE), pwd->b[1]);
    cpu_stb_data(env, addr + (2  << DF_BYTE), pwd->b[2]);
    cpu_stb_data(env, addr + (3  << DF_BYTE), pwd->b[3]);
    cpu_stb_data(env, addr + (4  << DF_BYTE), pwd->b[4]);
    cpu_stb_data(env, addr + (5  << DF_BYTE), pwd->b[5]);
    cpu_stb_data(env, addr + (6  << DF_BYTE), pwd->b[6]);
    cpu_stb_data(env, addr + (7  << DF_BYTE), pwd->b[7]);
    cpu_stb_data(env, addr + (8  << DF_BYTE), pwd->b[8]);
    cpu_stb_data(env, addr + (9  << DF_BYTE), pwd->b[9]);
    cpu_stb_data(env, addr + (10 << DF_BYTE), pwd->b[10]);
    cpu_stb_data(env, addr + (11 << DF_BYTE), pwd->b[11]);
    cpu_stb_data(env, addr + (12 << DF_BYTE), pwd->b[12]);
    cpu_stb_data(env, addr + (13 << DF_BYTE), pwd->b[13]);
    cpu_stb_data(env, addr + (14 << DF_BYTE), pwd->b[14]);
    cpu_stb_data(env, addr + (15 << DF_BYTE), pwd->b[15]);
    cpu_stb_data(env, addr + (16 << DF_BYTE), pwd->b[16]);
    cpu_stb_data(env, addr + (17 << DF_BYTE), pwd->b[17]);
    cpu_stb_data(env, addr + (18 << DF_BYTE), pwd->b[18]);
    cpu_stb_data(env, addr + (19 << DF_BYTE), pwd->b[19]);
    cpu_stb_data(env, addr + (20 << DF_BYTE), pwd->b[20]);
    cpu_stb_data(env, addr + (21 << DF_BYTE), pwd->b[21]);
    cpu_stb_data(env, addr + (22 << DF_BYTE), pwd->b[22]);
    cpu_stb_data(env, addr + (23 << DF_BYTE), pwd->b[23]);
    cpu_stb_data(env, addr + (24 << DF_BYTE), pwd->b[24]);
    cpu_stb_data(env, addr + (25 << DF_BYTE), pwd->b[25]);
    cpu_stb_data(env, addr + (26 << DF_BYTE), pwd->b[26]);
    cpu_stb_data(env, addr + (27 << DF_BYTE), pwd->b[27]);
    cpu_stb_data(env, addr + (28 << DF_BYTE), pwd->b[28]);
    cpu_stb_data(env, addr + (29 << DF_BYTE), pwd->b[29]);
    cpu_stb_data(env, addr + (30 << DF_BYTE), pwd->b[30]);
    cpu_stb_data(env, addr + (31 << DF_BYTE), pwd->b[31]);
#endif
}

#define PRINT_WR(wr) \
    printf("%s: %08lx %08lx %08lx %08lx", "wr", wr->d[0], wr->d[1], wr->d[2], wr->d[3]); \

void helper_lsx_vshuf_b(CPULoongArchState *env, uint32_t wa, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwa = &(env->fpr[wa].wr);
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);
    uint32_t i;
    uint32_t n = 128/8;
	wr_t tmp;
	tmp.q[0] = pwd->q[0];
	//tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/8 ; i++) {
        uint32_t k = (pwa->b[i] & 0x3f) % (2 * n);
        tmp.b[i] =
            (pwa->b[i] & 0xc0) ? 0 : k < n ? pwt->b[k] : pws->b[k - n];
    }
	pwd->q[0] = tmp.q[0];
	//pwd->q[1] = tmp.q[1];
    if (0) {
        PRINT_WR(pwa)
        PRINT_WR(pwd)
        PRINT_WR(pws)
        PRINT_WR(pwt)
    }
}

void helper_lsx_xvshuf_b(CPULoongArchState *env, uint32_t wa, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwa = &(env->fpr[wa].wr);
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);
    uint32_t i;
    uint32_t n = 128/8;
	wr_t tmp;
	tmp.q[0] = pwd->q[0];
	tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/8 ; i++) {
        uint32_t k1 = (pwa->b[i] & 0x3f) % (2 * n);
        tmp.b[i] =
            (pwa->b[i] & 0xc0) ? 0 : k1 < n ? pwt->b[k1] : pws->b[k1 - n];
        uint32_t k2 = (pwa->b[i+16] & 0x3f) % (2 * n);
        tmp.b[i+16] =
            (pwa->b[i+16] & 0xc0) ? 0 : k2 < n ? pwt->b[k2+16] : pws->b[k2 - n+16];
    }
	pwd->q[0] = tmp.q[0];
	pwd->q[1] = tmp.q[1];
    if (0) {
        PRINT_WR(pwa)
        PRINT_WR(pwd)
        PRINT_WR(pws)
        PRINT_WR(pwt)
    }
}

void helper_lsx_vextr_v(CPULoongArchState *env, uint32_t ui5, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);
    uint32_t i;
    uint32_t n = 128/8;
    uint32_t m = ui5 % 16;
	wr_t tmp;
	tmp.q[0] = pwd->q[0];
	tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/8 ; i++) {
        tmp.b[i] = (i + m) < n ? pwt->b[i + m] : pws->b[i + m - n];
    }
	pwd->q[0] = tmp.q[0];
	pwd->q[1] = tmp.q[1];
    if (0) {
        PRINT_WR(pwd)
        PRINT_WR(pws)
        PRINT_WR(pwt)
    }
}

void helper_lsx_xvextr_v(CPULoongArchState *env, uint32_t ui5, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);
    uint32_t i;
    uint32_t n = 128/8;
    uint32_t m = ui5 % 16;
	wr_t tmp;
	tmp.q[0] = pwd->q[0];
	tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/8 ; i++) {
        tmp.b[i] = (i + m) < n ? pwt->b[i + m] : pws->b[i + m - n];
        tmp.b[i+16] = (i + m) < n ? pwt->b[i + m+16] : pws->b[i + m - n+16];
    }
	pwd->q[0] = tmp.q[0];
	pwd->q[1] = tmp.q[1];
    if (0) {
        PRINT_WR(pwd)
        PRINT_WR(pws)
        PRINT_WR(pwt)
    }
}

void helper_lsx_vbitsel_v(CPULoongArchState *env, uint32_t va, uint32_t vd, uint32_t vj, uint32_t vk)
{
    wr_t *pwd = &(env->fpr[vd].wr);
    wr_t *pwa = &(env->fpr[va].wr);
    wr_t *pws = &(env->fpr[vj].wr);
    wr_t *pwt = &(env->fpr[vk].wr);

    pwd->d[0] = UNSIGNED(                                                     \
        (pws->d[0] & (~pwa->d[0])) | (pwt->d[0] & pwa->d[0]), DF_DOUBLE);
    pwd->d[1] = UNSIGNED(                                                     \
        (pws->d[1] & (~pwa->d[1])) | (pwt->d[1] & pwa->d[1]), DF_DOUBLE);
}

#undef PRINT_WR

void helper_lsx_vldrepl_d(CPULoongArchState *env, uint32_t wd, target_ulong addr)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    uint64_t data;
    MEMOP_IDX(DF_DOUBLE)
#if !defined(CONFIG_USER_ONLY)
    data = helper_ret_ldq_mmu(env, addr, oi, GETPC());
#else
    data = cpu_ldq_data(env, addr);
#endif
    pwd->d[0] = data;
    pwd->d[1] = data;
}

// TODO: check the endianness problem
void helper_lsx_vldrepl_w(CPULoongArchState *env, uint32_t wd, target_ulong addr)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    uint32_t data;
    MEMOP_IDX(DF_WORD)
#if !defined(CONFIG_USER_ONLY)
    data = helper_ret_ldul_mmu(env, addr + (0 << DF_WORD), oi, GETPC());
#else
    data = cpu_ldl_data(env, addr + (0 << DF_WORD));
#endif
    pwd->w[0] = data;
    pwd->w[1] = data;
    pwd->w[2] = data;
    pwd->w[3] = data;
}

// TODO: check the endianness problem
void helper_lsx_vldrepl_h(CPULoongArchState *env, uint32_t wd, target_ulong addr)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    uint16_t data;
    MEMOP_IDX(DF_HALF)
#if !defined(CONFIG_USER_ONLY)
    data = helper_ret_lduw_mmu(env, addr + (0 << DF_HALF), oi, GETPC());
#else
    data = cpu_lduw_data(env, addr + (0 << DF_HALF));
#endif
    int i;
    for (i = 0; i < 8; i++) {
        pwd->h[i] = data;
    }
}

// TODO: check the endianness problem
void helper_lsx_vldrepl_b(CPULoongArchState *env, uint32_t wd, target_ulong addr)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    uint8_t data;
    MEMOP_IDX(DF_BYTE)
#if !defined(CONFIG_USER_ONLY)
    data = helper_ret_ldub_mmu(env, addr + (0  << DF_BYTE), oi, GETPC());
#else
    data = cpu_ldub_data(env, addr + (0  << DF_BYTE));
#endif
    int i;
    for (i = 0; i < 16; i++) {
        pwd->b[i] = data;
    }
}
void helper_lsx_vstelm_d(CPULoongArchState *env, uint32_t wd, target_ulong addr, uint32_t sel)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    int mmu_idx = cpu_mmu_index(env, false);

    MEMOP_IDX(DF_DOUBLE)
    ensure_d_writable_pages(env, addr, mmu_idx, GETPC());
#if !defined(CONFIG_USER_ONLY)
    helper_ret_stq_mmu(env, addr + (0 << DF_DOUBLE), pwd->d[sel], oi, GETPC());
#else
    cpu_stq_data(env, addr + (0 << DF_DOUBLE), pwd->d[sel]);
#endif
}

// TODO: check the endianness problem
void helper_lsx_vstelm_w(CPULoongArchState *env, uint32_t wd, target_ulong addr, uint32_t sel)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    int mmu_idx = cpu_mmu_index(env, false);

    MEMOP_IDX(DF_WORD)
    ensure_w_writable_pages(env, addr, mmu_idx, GETPC());
#if !defined(CONFIG_USER_ONLY)
    helper_ret_stl_mmu(env, addr + (0 << DF_WORD), pwd->w[sel], oi, GETPC());
#else
    cpu_stl_data(env, addr + (0 << DF_WORD), pwd->w[sel]);
#endif
}

// TODO: check the endianness problem
void helper_lsx_vstelm_h(CPULoongArchState *env, uint32_t wd, target_ulong addr, uint32_t sel)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    int mmu_idx = cpu_mmu_index(env, false);

    MEMOP_IDX(DF_HALF)
    ensure_h_writable_pages(env, addr, mmu_idx, GETPC());
#if !defined(CONFIG_USER_ONLY)
    helper_ret_stw_mmu(env, addr + (0 << DF_HALF), pwd->h[sel], oi, GETPC());
#else
    cpu_stw_data(env, addr + (0 << DF_HALF), pwd->h[sel]);
#endif
}

// TODO: check the endianness problem
void helper_lsx_vstelm_b(CPULoongArchState *env, uint32_t wd, target_ulong addr, uint32_t sel)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    int mmu_idx = cpu_mmu_index(env, false);

    MEMOP_IDX(DF_BYTE)
    ensure_b_writable_pages(env, addr, mmu_idx, GETPC());
#if !defined(CONFIG_USER_ONLY)
    helper_ret_stb_mmu(env, addr + (0 << DF_BYTE), pwd->b[sel], oi, GETPC());
#else
    cpu_stb_data(env, addr + (0 << DF_BYTE), pwd->b[sel]);
#endif
}

void helper_lsx_xvldrepl_d(CPULoongArchState *env, uint32_t wd, target_ulong addr)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    uint64_t data;
    MEMOP_IDX(DF_DOUBLE)
#if !defined(CONFIG_USER_ONLY)
    data = helper_ret_ldq_mmu(env, addr, oi, GETPC());
#else
    data = cpu_ldq_data(env, addr);
#endif
    pwd->d[0] = data;
    pwd->d[1] = data;
    pwd->d[2] = data;
    pwd->d[3] = data;
}

// TODO: check the endianness problem
void helper_lsx_xvldrepl_w(CPULoongArchState *env, uint32_t wd, target_ulong addr)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    uint32_t data;
    MEMOP_IDX(DF_WORD)
#if !defined(CONFIG_USER_ONLY)
    data = helper_ret_ldul_mmu(env, addr + (0 << DF_WORD), oi, GETPC());
#else
    data = cpu_ldl_data(env, addr + (0 << DF_WORD));
#endif
    int i;
    for (i = 0; i < 8; i++) {
        pwd->w[i] = data;
    }
}

// TODO: check the endianness problem
void helper_lsx_xvldrepl_h(CPULoongArchState *env, uint32_t wd, target_ulong addr)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    uint16_t data;
    MEMOP_IDX(DF_HALF)
#if !defined(CONFIG_USER_ONLY)
    data = helper_ret_lduw_mmu(env, addr + (0 << DF_HALF), oi, GETPC());
#else
    data = cpu_lduw_data(env, addr + (0 << DF_HALF));
#endif
    int i;
    for (i = 0; i < 16; i++) {
        pwd->h[i] = data;
    }
}

// TODO: check the endianness problem
void helper_lsx_xvldrepl_b(CPULoongArchState *env, uint32_t wd, target_ulong addr)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    uint8_t data;
    MEMOP_IDX(DF_BYTE)
#if !defined(CONFIG_USER_ONLY)
    data = helper_ret_ldub_mmu(env, addr + (0  << DF_BYTE), oi, GETPC());
#else
    data = cpu_ldub_data(env, addr + (0  << DF_BYTE));
#endif
    int i;
    for (i = 0; i < 32; i++) {
        pwd->b[i] = data;
    }
}

void helper_lsx_xvstelm_d(CPULoongArchState *env, uint32_t wd, target_ulong addr, uint32_t sel)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    int mmu_idx = cpu_mmu_index(env, false);
    uint32_t idx = sel;

    MEMOP_IDX(DF_DOUBLE)
    ensure_d_writable_pages(env, addr, mmu_idx, GETPC());
#if !defined(CONFIG_USER_ONLY)
    helper_ret_stq_mmu(env, addr + (0 << DF_DOUBLE), pwd->d[idx], oi, GETPC());
#else
    cpu_stq_data(env, addr + (0 << DF_DOUBLE), pwd->d[idx]);
#endif
}

// TODO: check the endianness problem
void helper_lsx_xvstelm_w(CPULoongArchState *env, uint32_t wd, target_ulong addr, uint32_t sel)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    int mmu_idx = cpu_mmu_index(env, false);
    uint32_t idx = sel;

    MEMOP_IDX(DF_WORD)
    ensure_w_writable_pages(env, addr, mmu_idx, GETPC());
#if !defined(CONFIG_USER_ONLY)
    helper_ret_stl_mmu(env, addr + (0 << DF_WORD), pwd->w[idx], oi, GETPC());
#else
    cpu_stl_data(env, addr + (0 << DF_WORD), pwd->w[idx]);
#endif
}

// TODO: check the endianness problem
void helper_lsx_xvstelm_h(CPULoongArchState *env, uint32_t wd, target_ulong addr, uint32_t sel)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    int mmu_idx = cpu_mmu_index(env, false);
    uint32_t idx = sel;

    MEMOP_IDX(DF_HALF)
    ensure_h_writable_pages(env, addr, mmu_idx, GETPC());
#if !defined(CONFIG_USER_ONLY)
    helper_ret_stw_mmu(env, addr + (0 << DF_HALF), pwd->h[idx], oi, GETPC());
#else
    cpu_stw_data(env, addr + (0 << DF_HALF), pwd->h[idx]);
#endif
}

// TODO: check the endianness problem
void helper_lsx_xvstelm_b(CPULoongArchState *env, uint32_t wd, target_ulong addr, uint32_t sel)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    int mmu_idx = cpu_mmu_index(env, false);
    uint32_t idx = sel;

    MEMOP_IDX(DF_BYTE)
    ensure_b_writable_pages(env, addr, mmu_idx, GETPC());
#if !defined(CONFIG_USER_ONLY)
    helper_ret_stb_mmu(env, addr + (0 << DF_BYTE), pwd->b[idx], oi, GETPC());
#else
    cpu_stb_data(env, addr + (0 << DF_BYTE), pwd->b[idx]);
#endif
}

#define UNSIGNED_h_b(x) \
    (((uint16_t)x) & (uint16_t)(-1ULL >> (16 - 8)))
#define UNSIGNED_w_b(x) \
    (((uint32_t)x) & (uint32_t)(-1ULL >> (32 - 8)))
#define UNSIGNED_d_b(x) \
    (((uint64_t)x) & (uint64_t)(-1ULL >> (64 - 8)))
#define UNSIGNED_w_h(x) \
    (((uint32_t)x) & (uint32_t)(-1ULL >> (32 - 16)))
#define UNSIGNED_d_h(x) \
    (((uint64_t)x) & (uint64_t)(-1ULL >> (64 - 16)))
#define UNSIGNED_d_w(x) \
    (((uint64_t)x) & (uint64_t)(-1ULL >> (64 - 32)))

#define SIGNED_h_b(x) \
    ((((int16_t)x) << (16 - 8)) >> (16 - 8))
#define SIGNED_w_b(x) \
    ((((int32_t)x) << (32 - 8)) >> (32 - 8))
#define SIGNED_d_b(x) \
    ((((int64_t)x) << (64 - 8)) >> (64 - 8))
#define SIGNED_w_h(x) \
    ((((int32_t)x) << (32 - 16)) >> (32 - 16))
#define SIGNED_d_h(x) \
    ((((int64_t)x) << (64 - 16)) >> (64 - 16))
#define SIGNED_d_w(x) \
    ((((int64_t)x) << (64 - 32)) >> (64 - 32))
#define VEXT2XV(name, US, SDF, DDF, sdf, ddf)                       \
void helper_lsx_##name(CPULoongArchState *env, uint32_t wd, uint32_t ws) \
{                                                                   \
    wr_t wx, *pwx = &wx;                                            \
    wr_t *pwd = &(env->fpr[wd].wr);                      \
    wr_t *pws = &(env->fpr[ws].wr);                      \
    uint32_t i;                                                     \
    for (i = 0; i < 256/DF_BITS(DDF); i++) {                        \
        pwx->ddf[i] = US##_##ddf##_##sdf(pws->sdf[i]);              \
    }                                                               \
    lsx_move_x(pwd, pwx);                                           \
}
VEXT2XV(vext2xv_h_b  , SIGNED  , DF_BYTE, DF_HALF  , b, h)
VEXT2XV(vext2xv_w_b  , SIGNED  , DF_BYTE, DF_WORD  , b, w)
VEXT2XV(vext2xv_d_b  , SIGNED  , DF_BYTE, DF_DOUBLE, b, d)
VEXT2XV(vext2xv_w_h  , SIGNED  , DF_HALF, DF_WORD  , h, w)
VEXT2XV(vext2xv_d_h  , SIGNED  , DF_HALF, DF_DOUBLE, h, d)
VEXT2XV(vext2xv_d_w  , SIGNED  , DF_WORD, DF_DOUBLE, w, d)
VEXT2XV(vext2xv_hu_bu, UNSIGNED, DF_BYTE, DF_HALF  , b, h)
VEXT2XV(vext2xv_wu_bu, UNSIGNED, DF_BYTE, DF_WORD  , b, w)
VEXT2XV(vext2xv_du_bu, UNSIGNED, DF_BYTE, DF_DOUBLE, b, d)
VEXT2XV(vext2xv_wu_hu, UNSIGNED, DF_HALF, DF_WORD  , h, w)
VEXT2XV(vext2xv_du_hu, UNSIGNED, DF_HALF, DF_DOUBLE, h, d)
VEXT2XV(vext2xv_du_wu, UNSIGNED, DF_WORD, DF_DOUBLE, w, d)
#undef VEXT2XV

void helper_lsx_xvhseli_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t sel)
{
    wr_t wx, *pwx = &wx;
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    pwx->d[0] = pws->d[sel & 0x1];
    pwx->d[1] = pwd->d[((sel >> 1) & 0x1)];
    pwx->d[2] = pws->d[((sel >> 2) & 0x1) + 2];
    pwx->d[3] = pwd->d[((sel >> 3) & 0x1) + 2];
    lsx_move_x(pwd, pwx);
}

void helper_lsx_xvinsgr2vr_w(CPULoongArchState *env, uint32_t wd, uint32_t rs_num, uint32_t n)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    target_ulong rs = env->gpr[rs_num];
    n %= 8;
    pwd->w[n] = (int32_t)rs;
}

void helper_lsx_xvinsgr2vr_d(CPULoongArchState *env, uint32_t wd, uint32_t rs_num, uint32_t n)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    target_ulong rs = env->gpr[rs_num];
    n %= 4;
    pwd->d[n] = (int64_t)rs;
}

void helper_lsx_xvpickve2gr_w(CPULoongArchState *env, uint32_t rd, uint32_t ws, uint32_t n)
{
    n %= 8;
    env->gpr[rd] = (int32_t)env->fpr[ws].wr.w[n];
}

void helper_lsx_xvpickve2gr_d(CPULoongArchState *env, uint32_t rd, uint32_t ws, uint32_t n)
{
    n %= 4;
    env->gpr[rd] = (int64_t)env->fpr[ws].wr.d[n];
}

void helper_lsx_xvpickve2gr_wu(CPULoongArchState *env, uint32_t rd, uint32_t ws, uint32_t n)
{
    n %= 8;
    env->gpr[rd] = (uint32_t)env->fpr[ws].wr.w[n];
}

void helper_lsx_xvpickve_df(CPULoongArchState *env, uint32_t df, uint32_t wd,
                         uint32_t ws, uint32_t n)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    uint32_t i;

    switch (df) {
    case DF_WORD:
        pwd->w[0] = (int32_t)pws->w[n];
        for (i = 1; i < DF_ELEMENTS(DF_WORD); i++) {
             pwd->w[i] = 0;
        }
        break;
    case DF_DOUBLE:
        pwd->d[0] = (int64_t)pws->d[n];
        for (i = 1; i < DF_ELEMENTS(DF_DOUBLE); i++) {
             pwd->d[i] = 0;
        }
        break;
    default:
        assert(0);
    }
}

void helper_lsx_xvreplve0_df(CPULoongArchState *env, uint32_t df, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    uint32_t i;

    switch (df) {
    case DF_BYTE:
        for (i = 0; i < DF_ELEMENTS(DF_BYTE); i++) {
             pwd->b[i] = (int8_t)pws->b[0];
        }
        break;
    case DF_HALF:
        for (i = 0; i < DF_ELEMENTS(DF_HALF); i++) {
             pwd->h[i] = (int16_t)pws->h[0];
        }
        break;
    case DF_WORD:
        for (i = 0; i < DF_ELEMENTS(DF_WORD); i++) {
             pwd->w[i] = (int32_t)pws->w[0];
        }
        break;
    case DF_DOUBLE:
        for (i = 0; i < DF_ELEMENTS(DF_DOUBLE); i++) {
             pwd->d[i] = (int64_t)pws->d[0];
        }
        break;
    case DF_QUAD:
        for (i = 0; i < DF_ELEMENTS(DF_QUAD); i++) {
             pwd->q[i] = (__int128)pws->q[0];
        }
        break;
    default:
        assert(0);
    }
}

void helper_lsx_xvseli_df(CPULoongArchState *env, uint32_t df, uint32_t wd,
                         uint32_t ws, uint32_t n)
{
    wr_t wx, *pwx = &wx;
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    uint32_t i;

    switch (df) {
    case DF_WORD:
        for (i = 0; i < DF_ELEMENTS(DF_WORD); i++) {
             pwx->w[i] = ((n >> i) & 0x1) ? (int32_t)pws->w[i] : (int32_t)pwd->w[i];
        }
        break;
    case DF_DOUBLE:
        for (i = 0; i < DF_ELEMENTS(DF_DOUBLE); i++) {
             pwx->d[i] = ((n >> i) & 0x1) ? (int64_t)pws->d[i] : (int64_t)pwd->d[i];
        }
        break;
    default:
        assert(0);
    }
    lsx_move_x(pwd, pwx);
}

void helper_lsx_vpermi_w(CPULoongArchState *env, uint32_t wd,
                         uint32_t ws, uint32_t n)
{
    wr_t wx, *pwx = &wx;
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    uint32_t i, imm;

    for (i = 0; i < DF_V_ELEMENTS(DF_WORD); i++) {
        imm = (n >> (i*2)) & 0x3;
        switch (imm) {
        case 0x0:
            pwx->w[i] = (int32_t)pws->w[0];
            break;
        case 0x1:
            pwx->w[i] = (int32_t)pws->w[1];
            break;
        case 0x2:
            pwx->w[i] = (int32_t)pws->w[2];
            break;
        case 0x3:
            pwx->w[i] = (int32_t)pws->w[3];
            break;
        default:
            pwx->d[i] = 0;
        }
    }
    msa_move_v(pwd, pwx);
}

void helper_lsx_xvpermi_w(CPULoongArchState *env, uint32_t wd,
                         uint32_t ws, uint32_t n)
{
    wr_t wx, *pwx = &wx;
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    uint32_t i, j, imm;

    for (j = 0; j < 2; j++)
    {
        for (i = 0; i < DF_V_ELEMENTS(DF_WORD); i++) {
            imm = (n >> (i*2)) & 0x3;
            switch (imm) {
            case 0x0:
                pwx->w[2*j+i] = (int32_t)pws->w[2*j+0];
                break;
            case 0x1:
                pwx->w[2*j+i] = (int32_t)pws->w[2*j+1];
                break;
            case 0x2:
                pwx->w[2*j+i] = (int32_t)pws->w[2*j+2];
                break;
            case 0x3:
                pwx->w[2*j+i] = (int32_t)pws->w[2*j+3];
                break;
            default:
                pwx->d[2*j+i] = 0;
            }
        }
    }
    lsx_move_x(pwd, pwx);
}

void helper_lsx_xvpermi_d(CPULoongArchState *env, uint32_t wd,
                         uint32_t ws, uint32_t n)
{
    wr_t wx, *pwx = &wx;
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    uint32_t i, imm;

    for (i = 0; i < DF_ELEMENTS(DF_DOUBLE); i++) {
        imm = (n >> (i*2)) & 0x3;
        switch (imm) {
        case 0x0:
            pwx->d[i] = (int64_t)pws->d[0];
            break;
        case 0x1:
            pwx->d[i] = (int64_t)pws->d[1];
            break;
        case 0x2:
            pwx->d[i] = (int64_t)pws->d[2];
            break;
        case 0x3:
            pwx->d[i] = (int64_t)pws->d[3];
            break;
        default:
            pwx->d[i] = 0;
        }
    }
    lsx_move_x(pwd, pwx);
}

void helper_lsx_xvpermi_q(CPULoongArchState *env, uint32_t wd,
                         uint32_t ws, uint32_t n)
{
    wr_t wx, *pwx = &wx;
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    uint32_t i, imm;

    for (i = 0; i < DF_ELEMENTS(DF_QUAD); i++) {
        imm = (n >> (i*4)) & 0xf;
        if ((imm >> 3) & 0x1) {
            pwx->q[i] = 0;
        } else {
            switch (imm & 0x3) {
            case 0x0:
                pwx->q[i] = (__int128)pws->q[0];
                break;
            case 0x1:
                pwx->q[i] = (__int128)pws->q[1];
                break;
            case 0x2:
                pwx->q[i] = (__int128)pwd->q[0];
                break;
            case 0x3:
                pwx->q[i] = (__int128)pwd->q[1];
                break;
            default:
                pwx->q[i] = 0;
            }
        }
    }
    lsx_move_x(pwd, pwx);
}

#define LSX_DO_B LSX_DO(b)
#define LSX_DO_H LSX_DO(h)
#define LSX_DO_W LSX_DO(w)
#define LSX_DO_D LSX_DO(d)

#define LSX_LOOP_B LSX_LOOP(B)
#define LSX_LOOP_H LSX_LOOP(H)
#define LSX_LOOP_W LSX_LOOP(W)
#define LSX_LOOP_D LSX_LOOP(D)

#define LSX_LOOP_COND_B LSX_LOOP_COND(DF_BYTE)
#define LSX_LOOP_COND_H LSX_LOOP_COND(DF_HALF)
#define LSX_LOOP_COND_W LSX_LOOP_COND(DF_WORD)
#define LSX_LOOP_COND_D LSX_LOOP_COND(DF_DOUBLE)

#define LSX_LOOP_COND(DF) (DF_V_ELEMENTS(DF))

#define LSX_LOOP(DF) \
    do { \
        for (i = 0; i < (LSX_LOOP_COND_ ## DF) ; i++) { \
            LSX_DO_ ## DF; \
        } \
    } while (0)

#define LSX_FN_DF(FUNC)                                             \
void helper_lsx_##FUNC(CPULoongArchState *env, uint32_t df, uint32_t wd, \
        uint32_t ws, uint32_t wt)                                   \
{                                                                   \
    wr_t *pwd = &(env->fpr[wd].wr);                      \
    wr_t *pws = &(env->fpr[ws].wr);                      \
    wr_t *pwt = &(env->fpr[wt].wr);                      \
    wr_t wx, *pwx = &wx;                                            \
    uint32_t i;                                                     \
    switch (df) {                                                   \
    case DF_BYTE:                                                   \
        LSX_LOOP_B;                                                 \
        break;                                                      \
    case DF_HALF:                                                   \
        LSX_LOOP_H;                                                 \
        break;                                                      \
    case DF_WORD:                                                   \
        LSX_LOOP_W;                                                 \
        break;                                                      \
    case DF_DOUBLE:                                                 \
        LSX_LOOP_D;                                                 \
        break;                                                      \
    default:                                                        \
        assert(0);                                                  \
    }                                                               \
    lsx_move_v(pwd, pwx);                                           \
}
#define LSX_DO(DF)                                                          \
    do {                                                                    \
        uint32_t n = DF_V_ELEMENTS(df);                                     \
        uint32_t k = (pwd->DF[i] & 0x3f) % (2 * n);                         \
        pwx->DF[i] =                                                        \
            (pwd->DF[i] & 0xc0) ? 0 : k < n ? pwt->DF[k] : pws->DF[k - n];  \
    } while (0)
LSX_FN_DF(vshuf_df)
#undef LSX_DO
#undef LSX_FN_DF
#undef LSX_LOOP_COND

static inline int64_t lsx_vadd_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    return arg1 + arg2;
}

static inline int64_t lsx_vsub_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    return arg1 - arg2;
}

static inline int64_t lsx_vseq_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    return arg1 == arg2 ? -1 : 0;
}

static inline int64_t lsx_vsle_s_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    return arg1 <= arg2 ? -1 : 0;
}

static inline int64_t lsx_vsle_u_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    uint64_t u_arg2 = UNSIGNED(arg2, df);
    return u_arg1 <= u_arg2 ? -1 : 0;
}

static inline int64_t lsx_vslt_s_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    return arg1 < arg2 ? -1 : 0;
}

static inline int64_t lsx_vslt_u_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    uint64_t u_arg2 = UNSIGNED(arg2, df);
    return u_arg1 < u_arg2 ? -1 : 0;
}

static inline int64_t lsx_vmax_s_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    return arg1 > arg2 ? arg1 : arg2;
}

static inline int64_t lsx_vmin_s_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    return arg1 < arg2 ? arg1 : arg2;
}

static inline int64_t lsx_vmax_u_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    uint64_t u_arg2 = UNSIGNED(arg2, df);
    return u_arg1 > u_arg2 ? arg1 : arg2;
}

static inline int64_t lsx_vmin_u_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    uint64_t u_arg2 = UNSIGNED(arg2, df);
    return u_arg1 < u_arg2 ? arg1 : arg2;
}

#define LSX_BINOP_IMM_DF(helper, func)                                  \
void helper_lsx_ ## helper ## _df(CPULoongArchState *env, uint32_t df,       \
                        uint32_t wd, uint32_t ws, int32_t u5)           \
{                                                                       \
    wr_t *pwd = &(env->fpr[wd].wr);                          \
    wr_t *pws = &(env->fpr[ws].wr);                          \
    uint32_t i;                                                         \
                                                                        \
    switch (df) {                                                       \
    case DF_BYTE:                                                       \
        for (i = 0; i < DF_V_ELEMENTS(DF_BYTE); i++) {                  \
            pwd->b[i] = lsx_ ## func ## _df(df, pws->b[i], u5);         \
        }                                                               \
        break;                                                          \
    case DF_HALF:                                                       \
        for (i = 0; i < DF_V_ELEMENTS(DF_HALF); i++) {                  \
            pwd->h[i] = lsx_ ## func ## _df(df, pws->h[i], u5);         \
        }                                                               \
        break;                                                          \
    case DF_WORD:                                                       \
        for (i = 0; i < DF_V_ELEMENTS(DF_WORD); i++) {                  \
            pwd->w[i] = lsx_ ## func ## _df(df, pws->w[i], u5);         \
        }                                                               \
        break;                                                          \
    case DF_DOUBLE:                                                     \
        for (i = 0; i < DF_V_ELEMENTS(DF_DOUBLE); i++) {                \
            pwd->d[i] = lsx_ ## func ## _df(df, pws->d[i], u5);         \
        }                                                               \
        break;                                                          \
    default:                                                            \
        assert(0);                                                      \
    }                                                                   \
}

LSX_BINOP_IMM_DF(vseqi, vseq)
LSX_BINOP_IMM_DF(vslei_s, vsle_s)
LSX_BINOP_IMM_DF(vslei_u, vsle_u)
LSX_BINOP_IMM_DF(vslti_s, vslt_s)
LSX_BINOP_IMM_DF(vslti_u, vslt_u)
LSX_BINOP_IMM_DF(vaddi_u, vadd)
LSX_BINOP_IMM_DF(vsubi_u, vsub)
LSX_BINOP_IMM_DF(vmaxi_s, vmax_s)
LSX_BINOP_IMM_DF(vmini_s, vmin_s)
LSX_BINOP_IMM_DF(vmaxi_u, vmax_u)
LSX_BINOP_IMM_DF(vmini_u, vmin_u)
#undef LSX_BINOP_IMM_DF

void helper_lsx_vbsll_v(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t u5)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

	wr_t tmp;
	tmp.q[0] = pwd->q[0];
	tmp.q[1] = pwd->q[1];
    for(int i = 0; i < 16; i++) {
        tmp.b[i]  = (i < u5)? 0 : pws->b[i - u5];
	}
	pwd->q[0] = tmp.q[0];
	pwd->q[1] = tmp.q[1];
}

void helper_lsx_vbsrl_v(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t u5)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    for(int i = 0; i < 16; i++)
        pwd->b[i]  = (i + u5 > 15)? 0 : pws->b[i + u5];
}

static inline int64_t lsx_vclz_df(uint32_t df, int64_t arg)
{
    uint64_t x, y;
    int n, c;

    x = UNSIGNED(arg, df);
    n = DF_BITS(df);
    c = DF_BITS(df) / 2;

    do {
        y = x >> c;
        if (y != 0) {
            n = n - c;
            x = y;
        }
        c = c >> 1;
    } while (c != 0);

    return n - x;
}

static inline int64_t lsx_vclo_df(uint32_t df, int64_t arg)
{
    return lsx_vclz_df(df, UNSIGNED((~arg), df));
}

void helper_lsx_vclo_b(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    pwd->b[0]  = lsx_vclo_df(DF_BYTE, pws->b[0]);
    pwd->b[1]  = lsx_vclo_df(DF_BYTE, pws->b[1]);
    pwd->b[2]  = lsx_vclo_df(DF_BYTE, pws->b[2]);
    pwd->b[3]  = lsx_vclo_df(DF_BYTE, pws->b[3]);
    pwd->b[4]  = lsx_vclo_df(DF_BYTE, pws->b[4]);
    pwd->b[5]  = lsx_vclo_df(DF_BYTE, pws->b[5]);
    pwd->b[6]  = lsx_vclo_df(DF_BYTE, pws->b[6]);
    pwd->b[7]  = lsx_vclo_df(DF_BYTE, pws->b[7]);
    pwd->b[8]  = lsx_vclo_df(DF_BYTE, pws->b[8]);
    pwd->b[9]  = lsx_vclo_df(DF_BYTE, pws->b[9]);
    pwd->b[10] = lsx_vclo_df(DF_BYTE, pws->b[10]);
    pwd->b[11] = lsx_vclo_df(DF_BYTE, pws->b[11]);
    pwd->b[12] = lsx_vclo_df(DF_BYTE, pws->b[12]);
    pwd->b[13] = lsx_vclo_df(DF_BYTE, pws->b[13]);
    pwd->b[14] = lsx_vclo_df(DF_BYTE, pws->b[14]);
    pwd->b[15] = lsx_vclo_df(DF_BYTE, pws->b[15]);
}

void helper_lsx_vclo_h(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    pwd->h[0]  = lsx_vclo_df(DF_HALF, pws->h[0]);
    pwd->h[1]  = lsx_vclo_df(DF_HALF, pws->h[1]);
    pwd->h[2]  = lsx_vclo_df(DF_HALF, pws->h[2]);
    pwd->h[3]  = lsx_vclo_df(DF_HALF, pws->h[3]);
    pwd->h[4]  = lsx_vclo_df(DF_HALF, pws->h[4]);
    pwd->h[5]  = lsx_vclo_df(DF_HALF, pws->h[5]);
    pwd->h[6]  = lsx_vclo_df(DF_HALF, pws->h[6]);
    pwd->h[7]  = lsx_vclo_df(DF_HALF, pws->h[7]);
}

void helper_lsx_vclo_w(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    pwd->w[0]  = lsx_vclo_df(DF_WORD, pws->w[0]);
    pwd->w[1]  = lsx_vclo_df(DF_WORD, pws->w[1]);
    pwd->w[2]  = lsx_vclo_df(DF_WORD, pws->w[2]);
    pwd->w[3]  = lsx_vclo_df(DF_WORD, pws->w[3]);
}

void helper_lsx_vclo_d(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    pwd->d[0]  = lsx_vclo_df(DF_DOUBLE, pws->d[0]);
    pwd->d[1]  = lsx_vclo_df(DF_DOUBLE, pws->d[1]);
}

void helper_lsx_vclz_b(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    pwd->b[0]  = lsx_vclz_df(DF_BYTE, pws->b[0]);
    pwd->b[1]  = lsx_vclz_df(DF_BYTE, pws->b[1]);
    pwd->b[2]  = lsx_vclz_df(DF_BYTE, pws->b[2]);
    pwd->b[3]  = lsx_vclz_df(DF_BYTE, pws->b[3]);
    pwd->b[4]  = lsx_vclz_df(DF_BYTE, pws->b[4]);
    pwd->b[5]  = lsx_vclz_df(DF_BYTE, pws->b[5]);
    pwd->b[6]  = lsx_vclz_df(DF_BYTE, pws->b[6]);
    pwd->b[7]  = lsx_vclz_df(DF_BYTE, pws->b[7]);
    pwd->b[8]  = lsx_vclz_df(DF_BYTE, pws->b[8]);
    pwd->b[9]  = lsx_vclz_df(DF_BYTE, pws->b[9]);
    pwd->b[10] = lsx_vclz_df(DF_BYTE, pws->b[10]);
    pwd->b[11] = lsx_vclz_df(DF_BYTE, pws->b[11]);
    pwd->b[12] = lsx_vclz_df(DF_BYTE, pws->b[12]);
    pwd->b[13] = lsx_vclz_df(DF_BYTE, pws->b[13]);
    pwd->b[14] = lsx_vclz_df(DF_BYTE, pws->b[14]);
    pwd->b[15] = lsx_vclz_df(DF_BYTE, pws->b[15]);
}

void helper_lsx_vclz_h(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    pwd->h[0]  = lsx_vclz_df(DF_HALF, pws->h[0]);
    pwd->h[1]  = lsx_vclz_df(DF_HALF, pws->h[1]);
    pwd->h[2]  = lsx_vclz_df(DF_HALF, pws->h[2]);
    pwd->h[3]  = lsx_vclz_df(DF_HALF, pws->h[3]);
    pwd->h[4]  = lsx_vclz_df(DF_HALF, pws->h[4]);
    pwd->h[5]  = lsx_vclz_df(DF_HALF, pws->h[5]);
    pwd->h[6]  = lsx_vclz_df(DF_HALF, pws->h[6]);
    pwd->h[7]  = lsx_vclz_df(DF_HALF, pws->h[7]);
}

void helper_lsx_vclz_w(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    pwd->w[0]  = lsx_vclz_df(DF_WORD, pws->w[0]);
    pwd->w[1]  = lsx_vclz_df(DF_WORD, pws->w[1]);
    pwd->w[2]  = lsx_vclz_df(DF_WORD, pws->w[2]);
    pwd->w[3]  = lsx_vclz_df(DF_WORD, pws->w[3]);
}

void helper_lsx_vclz_d(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    pwd->d[0]  = lsx_vclz_df(DF_DOUBLE, pws->d[0]);
    pwd->d[1]  = lsx_vclz_df(DF_DOUBLE, pws->d[1]);
}

static inline int64_t lsx_vpcnt_df(uint32_t df, int64_t arg)
{
    uint64_t x;

    x = UNSIGNED(arg, df);

    x = (x & 0x5555555555555555ULL) + ((x >>  1) & 0x5555555555555555ULL);
    x = (x & 0x3333333333333333ULL) + ((x >>  2) & 0x3333333333333333ULL);
    x = (x & 0x0F0F0F0F0F0F0F0FULL) + ((x >>  4) & 0x0F0F0F0F0F0F0F0FULL);
    x = (x & 0x00FF00FF00FF00FFULL) + ((x >>  8) & 0x00FF00FF00FF00FFULL);
    x = (x & 0x0000FFFF0000FFFFULL) + ((x >> 16) & 0x0000FFFF0000FFFFULL);
    x = (x & 0x00000000FFFFFFFFULL) + ((x >> 32));

    return x;
}

void helper_lsx_vpcnt_b(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    pwd->b[0]  = lsx_vpcnt_df(DF_BYTE, pws->b[0]);
    pwd->b[1]  = lsx_vpcnt_df(DF_BYTE, pws->b[1]);
    pwd->b[2]  = lsx_vpcnt_df(DF_BYTE, pws->b[2]);
    pwd->b[3]  = lsx_vpcnt_df(DF_BYTE, pws->b[3]);
    pwd->b[4]  = lsx_vpcnt_df(DF_BYTE, pws->b[4]);
    pwd->b[5]  = lsx_vpcnt_df(DF_BYTE, pws->b[5]);
    pwd->b[6]  = lsx_vpcnt_df(DF_BYTE, pws->b[6]);
    pwd->b[7]  = lsx_vpcnt_df(DF_BYTE, pws->b[7]);
    pwd->b[8]  = lsx_vpcnt_df(DF_BYTE, pws->b[8]);
    pwd->b[9]  = lsx_vpcnt_df(DF_BYTE, pws->b[9]);
    pwd->b[10] = lsx_vpcnt_df(DF_BYTE, pws->b[10]);
    pwd->b[11] = lsx_vpcnt_df(DF_BYTE, pws->b[11]);
    pwd->b[12] = lsx_vpcnt_df(DF_BYTE, pws->b[12]);
    pwd->b[13] = lsx_vpcnt_df(DF_BYTE, pws->b[13]);
    pwd->b[14] = lsx_vpcnt_df(DF_BYTE, pws->b[14]);
    pwd->b[15] = lsx_vpcnt_df(DF_BYTE, pws->b[15]);
}

void helper_lsx_vpcnt_h(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    pwd->h[0]  = lsx_vpcnt_df(DF_HALF, pws->h[0]);
    pwd->h[1]  = lsx_vpcnt_df(DF_HALF, pws->h[1]);
    pwd->h[2]  = lsx_vpcnt_df(DF_HALF, pws->h[2]);
    pwd->h[3]  = lsx_vpcnt_df(DF_HALF, pws->h[3]);
    pwd->h[4]  = lsx_vpcnt_df(DF_HALF, pws->h[4]);
    pwd->h[5]  = lsx_vpcnt_df(DF_HALF, pws->h[5]);
    pwd->h[6]  = lsx_vpcnt_df(DF_HALF, pws->h[6]);
    pwd->h[7]  = lsx_vpcnt_df(DF_HALF, pws->h[7]);
}

void helper_lsx_vpcnt_w(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    pwd->w[0]  = lsx_vpcnt_df(DF_WORD, pws->w[0]);
    pwd->w[1]  = lsx_vpcnt_df(DF_WORD, pws->w[1]);
    pwd->w[2]  = lsx_vpcnt_df(DF_WORD, pws->w[2]);
    pwd->w[3]  = lsx_vpcnt_df(DF_WORD, pws->w[3]);
}

void helper_lsx_vpcnt_d(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    pwd->d[0]  = lsx_vpcnt_df(DF_DOUBLE, pws->d[0]);
    pwd->d[1]  = lsx_vpcnt_df(DF_DOUBLE, pws->d[1]);
}

void helper_lsx_vneg_b(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    pwd->b[0]  = -pws->b[0];
    pwd->b[1]  = -pws->b[1];
    pwd->b[2]  = -pws->b[2];
    pwd->b[3]  = -pws->b[3];
    pwd->b[4]  = -pws->b[4];
    pwd->b[5]  = -pws->b[5];
    pwd->b[6]  = -pws->b[6];
    pwd->b[7]  = -pws->b[7];
    pwd->b[8]  = -pws->b[8];
    pwd->b[9]  = -pws->b[9];
    pwd->b[10] = -pws->b[10];
    pwd->b[11] = -pws->b[11];
    pwd->b[12] = -pws->b[12];
    pwd->b[13] = -pws->b[13];
    pwd->b[14] = -pws->b[14];
    pwd->b[15] = -pws->b[15];
}

void helper_lsx_vneg_h(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    pwd->h[0]  = -pws->h[0];
    pwd->h[1]  = -pws->h[1];
    pwd->h[2]  = -pws->h[2];
    pwd->h[3]  = -pws->h[3];
    pwd->h[4]  = -pws->h[4];
    pwd->h[5]  = -pws->h[5];
    pwd->h[6]  = -pws->h[6];
    pwd->h[7]  = -pws->h[7];
}

void helper_lsx_vneg_w(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    pwd->w[0]  = -pws->w[0];
    pwd->w[1]  = -pws->w[1];
    pwd->w[2]  = -pws->w[2];
    pwd->w[3]  = -pws->w[3];
}

void helper_lsx_vneg_d(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    pwd->d[0]  = -pws->d[0];
    pwd->d[1]  = -pws->d[1];
}

void helper_lsx_vmskltz_b(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    pwd->d[0]  = 0;
    pwd->d[1]  = 0;

    pwd->h[0]  = (((0x80 & pws->b[0]) != 0) << 0) |
                 (((0x80 & pws->b[1]) != 0) << 1) |
                 (((0x80 & pws->b[2]) != 0) << 2) |
                 (((0x80 & pws->b[3]) != 0) << 3) |
                 (((0x80 & pws->b[4]) != 0) << 4) |
                 (((0x80 & pws->b[5]) != 0) << 5) |
                 (((0x80 & pws->b[6]) != 0) << 6) |
                 (((0x80 & pws->b[7]) != 0) << 7) |
                 (((0x80 & pws->b[8]) != 0) << 8) |
                 (((0x80 & pws->b[9]) != 0) << 9) |
                 (((0x80 & pws->b[10]) != 0) << 10) |
                 (((0x80 & pws->b[11]) != 0) << 11) |
                 (((0x80 & pws->b[12]) != 0) << 12) |
                 (((0x80 & pws->b[13]) != 0) << 13) |
                 (((0x80 & pws->b[14]) != 0) << 14) |
                 (((0x80 & pws->b[15]) != 0) << 15) ;
}

void helper_lsx_vmskltz_h(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    pwd->d[0]  = 0;
    pwd->d[1]  = 0;

    pwd->h[0]  = (((0x8000 & pws->h[0]) != 0) << 0) |
                 (((0x8000 & pws->h[1]) != 0) << 1) |
                 (((0x8000 & pws->h[2]) != 0) << 2) |
                 (((0x8000 & pws->h[3]) != 0) << 3) |
                 (((0x8000 & pws->h[4]) != 0) << 4) |
                 (((0x8000 & pws->h[5]) != 0) << 5) |
                 (((0x8000 & pws->h[6]) != 0) << 6) |
                 (((0x8000 & pws->h[7]) != 0) << 7) ;
}

void helper_lsx_vmskltz_w(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    pwd->d[0]  = 0;
    pwd->d[1]  = 0;

    pwd->w[0]  = (((0x80000000 & pws->w[0]) != 0) << 0) |
                 (((0x80000000 & pws->w[1]) != 0) << 1) |
                 (((0x80000000 & pws->w[2]) != 0) << 2) |
                 (((0x80000000 & pws->w[3]) != 0) << 3) ;
}

void helper_lsx_vmskltz_d(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    pwd->d[0]  = 0;
    pwd->d[1]  = 0;

    pwd->d[0]  = (((0x8000000000000000 & pws->d[0]) != 0) << 0) |
                 (((0x8000000000000000 & pws->d[1]) != 0) << 1) ;
}

void helper_lsx_vmskgez_b(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    pwd->d[0]  = 0;
    pwd->d[1]  = 0;

    pwd->h[0]  = ( !(pws->b[0]  < 0) << 0 ) |
                 ( !(pws->b[1]  < 0) << 1 ) |
                 ( !(pws->b[2]  < 0) << 2 ) |
                 ( !(pws->b[3]  < 0) << 3 ) |
                 ( !(pws->b[4]  < 0) << 4 ) |
                 ( !(pws->b[5]  < 0) << 5 ) |
                 ( !(pws->b[6]  < 0) << 6 ) |
                 ( !(pws->b[7]  < 0) << 7 ) |
                 ( !(pws->b[8]  < 0) << 8 ) |
                 ( !(pws->b[9]  < 0) << 9 ) |
                 ( !(pws->b[10] < 0) << 10) |
                 ( !(pws->b[11] < 0) << 11) |
                 ( !(pws->b[12] < 0) << 12) |
                 ( !(pws->b[13] < 0) << 13) |
                 ( !(pws->b[14] < 0) << 14) |
                 ( !(pws->b[15] < 0) << 15) ;
}

void helper_lsx_vmsknz_b(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    pwd->d[0]  = 0;
    pwd->d[1]  = 0;

    pwd->h[0]  = ( (pws->b[0]  != 0) << 0 ) |
                 ( (pws->b[1]  != 0) << 1 ) |
                 ( (pws->b[2]  != 0) << 2 ) |
                 ( (pws->b[3]  != 0) << 3 ) |
                 ( (pws->b[4]  != 0) << 4 ) |
                 ( (pws->b[5]  != 0) << 5 ) |
                 ( (pws->b[6]  != 0) << 6 ) |
                 ( (pws->b[7]  != 0) << 7 ) |
                 ( (pws->b[8]  != 0) << 8 ) |
                 ( (pws->b[9]  != 0) << 9 ) |
                 ( (pws->b[10] != 0) << 10) |
                 ( (pws->b[11] != 0) << 11) |
                 ( (pws->b[12] != 0) << 12) |
                 ( (pws->b[13] != 0) << 13) |
                 ( (pws->b[14] != 0) << 14) |
                 ( (pws->b[15] != 0) << 15) ;
}

void helper_lsx_vmskcopy_b(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    pwd->b[0]  = (pws->h[0] & 0x0001) != 0;
    pwd->b[1]  = (pws->h[0] & 0x0002) != 0;
    pwd->b[2]  = (pws->h[0] & 0x0004) != 0;
    pwd->b[3]  = (pws->h[0] & 0x0008) != 0;
    pwd->b[4]  = (pws->h[0] & 0x0010) != 0;
    pwd->b[5]  = (pws->h[0] & 0x0020) != 0;
    pwd->b[6]  = (pws->h[0] & 0x0040) != 0;
    pwd->b[7]  = (pws->h[0] & 0x0080) != 0;
    pwd->b[8]  = (pws->h[0] & 0x0100) != 0;
    pwd->b[9]  = (pws->h[0] & 0x0200) != 0;
    pwd->b[10] = (pws->h[0] & 0x0400) != 0;
    pwd->b[11] = (pws->h[0] & 0x0800) != 0;
    pwd->b[12] = (pws->h[0] & 0x1000) != 0;
    pwd->b[13] = (pws->h[0] & 0x2000) != 0;
    pwd->b[14] = (pws->h[0] & 0x4000) != 0;
    pwd->b[15] = (pws->h[0] & 0x8000) != 0;
}

void helper_lsx_vmskfill_b(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    pwd->b[0]  = ((pws->h[0] & 0x0001) != 0)? 0xff : 0x00;
    pwd->b[1]  = ((pws->h[0] & 0x0002) != 0)? 0xff : 0x00;
    pwd->b[2]  = ((pws->h[0] & 0x0004) != 0)? 0xff : 0x00;
    pwd->b[3]  = ((pws->h[0] & 0x0008) != 0)? 0xff : 0x00;
    pwd->b[4]  = ((pws->h[0] & 0x0010) != 0)? 0xff : 0x00;
    pwd->b[5]  = ((pws->h[0] & 0x0020) != 0)? 0xff : 0x00;
    pwd->b[6]  = ((pws->h[0] & 0x0040) != 0)? 0xff : 0x00;
    pwd->b[7]  = ((pws->h[0] & 0x0080) != 0)? 0xff : 0x00;
    pwd->b[8]  = ((pws->h[0] & 0x0100) != 0)? 0xff : 0x00;
    pwd->b[9]  = ((pws->h[0] & 0x0200) != 0)? 0xff : 0x00;
    pwd->b[10] = ((pws->h[0] & 0x0400) != 0)? 0xff : 0x00;
    pwd->b[11] = ((pws->h[0] & 0x0800) != 0)? 0xff : 0x00;
    pwd->b[12] = ((pws->h[0] & 0x1000) != 0)? 0xff : 0x00;
    pwd->b[13] = ((pws->h[0] & 0x2000) != 0)? 0xff : 0x00;
    pwd->b[14] = ((pws->h[0] & 0x4000) != 0)? 0xff : 0x00;
    pwd->b[15] = ((pws->h[0] & 0x8000) != 0)? 0xff : 0x00;
}

void helper_lsx_vreplgr2vr_df(CPULoongArchState *env, uint32_t df, uint32_t wd,
                              uint32_t rs)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    uint32_t i;
    switch (df) {
    case DF_BYTE:
        for (i = 0; i < DF_V_ELEMENTS(DF_BYTE); i++) {
            pwd->b[i] = (int8_t)env->gpr[rs];
        }
        break;
    case DF_HALF:
        for (i = 0; i < DF_V_ELEMENTS(DF_HALF); i++) {
            pwd->h[i] = (int16_t)env->gpr[rs];
        }
        break;
    case DF_WORD:
        for (i = 0; i < DF_V_ELEMENTS(DF_WORD); i++) {
            pwd->w[i] = (int32_t)env->gpr[rs];
        }
        break;
    case DF_DOUBLE:
        for (i = 0; i < DF_V_ELEMENTS(DF_DOUBLE); i++) {
            pwd->d[i] = (int64_t)env->gpr[rs];
        }
       break;
    default:
        assert(0);
    }
}

void helper_lsx_vinsgr2vr_b(CPULoongArchState *env, uint32_t wd,
                          uint32_t rs_num, uint32_t n)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    target_ulong rs = env->gpr[rs_num];
    n %= 16;
    pwd->b[n] = (int8_t)rs;
}

void helper_lsx_vinsgr2vr_h(CPULoongArchState *env, uint32_t wd,
                          uint32_t rs_num, uint32_t n)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    target_ulong rs = env->gpr[rs_num];
    n %= 8;
    pwd->h[n] = (int16_t)rs;
}

void helper_lsx_vinsgr2vr_w(CPULoongArchState *env, uint32_t wd,
                          uint32_t rs_num, uint32_t n)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    target_ulong rs = env->gpr[rs_num];
    n %= 4;
    pwd->w[n] = (int32_t)rs;
}

void helper_lsx_vinsgr2vr_d(CPULoongArchState *env, uint32_t wd,
                          uint32_t rs_num, uint32_t n)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    target_ulong rs = env->gpr[rs_num];
    n %= 2;
    pwd->d[n] = (int64_t)rs;
}

void helper_lsx_vpickve2gr_du(CPULoongArchState *env, uint32_t rd,
                              uint32_t ws, uint32_t n)
{
    n %= 2;
    env->gpr[rd] = (uint64_t)env->fpr[ws].wr.d[n];
}

void helper_lsx_vreplvei_df(CPULoongArchState *env, uint32_t df, uint32_t wd,
                          uint32_t ws, uint32_t n)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    lsx_vreplve_df(df, pwd, pws, n);
}

#define CONCATENATE_AND_SLIDE(s, k)             \
    do {                                        \
        for (i = 0; i < s; i++) {               \
            v[i]     = pws->b[s * k + i];       \
            v[i + s] = pwd->b[s * k + i];       \
        }                                       \
        for (i = 0; i < s; i++) {               \
            pwd->b[s * k + i] = v[i + n];       \
        }                                       \
    } while (0)

void helper_lsx_vextrcoli_df(CPULoongArchState *env, uint32_t df, uint32_t wd,
                        uint32_t ws, uint32_t n)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    lsx_vextrcol_df(df, pwd, pws, n);
}

void helper_lsx_vseteqz_v(CPULoongArchState *env, uint32_t cd, uint32_t ws)
{
    wr_t *pws = &(env->fpr[ws].wr);
    env->cf[cd & 0x7] = (pws->q[0] == 0);
}

void helper_lsx_vsetnez_v(CPULoongArchState *env, uint32_t cd, uint32_t ws)
{
    wr_t *pws = &(env->fpr[ws].wr);
    env->cf[cd & 0x7] = (pws->q[0] != 0);
}

void helper_lsx_vsetanyeqz_b(CPULoongArchState *env, uint32_t cd, uint32_t ws)
{
    wr_t *pws = &(env->fpr[ws].wr);
    env->cf[cd & 0x7] = (pws->b[0] == 0) ||
                        (pws->b[1] == 0) ||
                        (pws->b[2] == 0) ||
                        (pws->b[3] == 0) ||
                        (pws->b[4] == 0) ||
                        (pws->b[5] == 0) ||
                        (pws->b[6] == 0) ||
                        (pws->b[7] == 0) ||
                        (pws->b[8] == 0) ||
                        (pws->b[9] == 0) ||
                        (pws->b[10] == 0) ||
                        (pws->b[11] == 0) ||
                        (pws->b[12] == 0) ||
                        (pws->b[13] == 0) ||
                        (pws->b[14] == 0) ||
                        (pws->b[15] == 0) ;
}

void helper_lsx_vsetanyeqz_h(CPULoongArchState *env, uint32_t cd, uint32_t ws)
{
    wr_t *pws = &(env->fpr[ws].wr);
    env->cf[cd & 0x7] = (pws->h[0] == 0) ||
                        (pws->h[1] == 0) ||
                        (pws->h[2] == 0) ||
                        (pws->h[3] == 0) ||
                        (pws->h[4] == 0) ||
                        (pws->h[5] == 0) ||
                        (pws->h[6] == 0) ||
                        (pws->h[7] == 0) ;
}

void helper_lsx_vsetanyeqz_w(CPULoongArchState *env, uint32_t cd, uint32_t ws)
{
    wr_t *pws = &(env->fpr[ws].wr);
    env->cf[cd & 0x7] = (pws->w[0] == 0) ||
                        (pws->w[1] == 0) ||
                        (pws->w[2] == 0) ||
                        (pws->w[3] == 0) ;
}

void helper_lsx_vsetanyeqz_d(CPULoongArchState *env, uint32_t cd, uint32_t ws)
{
    wr_t *pws = &(env->fpr[ws].wr);
    env->cf[cd & 0x7] = (pws->d[0] == 0) ||
                                   (pws->d[1] == 0) ;
}

void helper_lsx_vsetallnez_b(CPULoongArchState *env, uint32_t cd, uint32_t ws)
{
    wr_t *pws = &(env->fpr[ws].wr);
    env->cf[cd & 0x7] = (pws->b[0] != 0) &&
                        (pws->b[1] != 0) &&
                        (pws->b[2] != 0) &&
                        (pws->b[3] != 0) &&
                        (pws->b[4] != 0) &&
                        (pws->b[5] != 0) &&
                        (pws->b[6] != 0) &&
                        (pws->b[7] != 0) &&
                        (pws->b[8] != 0) &&
                        (pws->b[9] != 0) &&
                        (pws->b[10] != 0) &&
                        (pws->b[11] != 0) &&
                        (pws->b[12] != 0) &&
                        (pws->b[13] != 0) &&
                        (pws->b[14] != 0) &&
                        (pws->b[15] != 0) ;
}

void helper_lsx_vsetallnez_h(CPULoongArchState *env, uint32_t cd, uint32_t ws)
{
    wr_t *pws = &(env->fpr[ws].wr);
    env->cf[cd & 0x7] = (pws->h[0] != 0) &&
                        (pws->h[1] != 0) &&
                        (pws->h[2] != 0) &&
                        (pws->h[3] != 0) &&
                        (pws->h[4] != 0) &&
                        (pws->h[5] != 0) &&
                        (pws->h[6] != 0) &&
                        (pws->h[7] != 0) ;
}

void helper_lsx_vsetallnez_w(CPULoongArchState *env, uint32_t cd, uint32_t ws)
{
    wr_t *pws = &(env->fpr[ws].wr);
    env->cf[cd & 0x7] = (pws->w[0] != 0) &&
                        (pws->w[1] != 0) &&
                        (pws->w[2] != 0) &&
                        (pws->w[3] != 0) ;
}

void helper_lsx_vsetallnez_d(CPULoongArchState *env, uint32_t cd, uint32_t ws)
{
    wr_t *pws = &(env->fpr[ws].wr);
    env->cf[cd & 0x7] = (pws->d[0] != 0) && (pws->d[1] != 0);
}

static inline int64_t lsx_vbitclr_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    int32_t b_arg2 = BIT_POSITION(arg2, df);
    return UNSIGNED(arg1 & (~(1LL << b_arg2)), df);
}

static inline int64_t lsx_vbitset_df(uint32_t df, int64_t arg1,
        int64_t arg2)
{
    int32_t b_arg2 = BIT_POSITION(arg2, df);
    return UNSIGNED(arg1 | (1LL << b_arg2), df);
}

static inline int64_t lsx_vbitrev_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    int32_t b_arg2 = BIT_POSITION(arg2, df);
    return UNSIGNED(arg1 ^ (1LL << b_arg2), df);
}

static inline int64_t lsx_vsat_s_df(uint32_t df, int64_t arg, uint32_t m)
{
    return arg < M_MIN_INT(m + 1) ? M_MIN_INT(m + 1) :
                                    arg > M_MAX_INT(m + 1) ? M_MAX_INT(m + 1) :
                                                             arg;
}

static inline int64_t lsx_vsat_u_df(uint32_t df, int64_t arg, uint32_t m)
{
    uint64_t u_arg = UNSIGNED(arg, df);
    return  u_arg < M_MAX_UINT(m + 1) ? u_arg :
                                        M_MAX_UINT(m + 1);
}

static inline int64_t lsx_vsll_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    int32_t b_arg2 = BIT_POSITION(arg2, df);
    return arg1 << b_arg2;
}

static inline int64_t lsx_vsrl_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    int32_t b_arg2 = BIT_POSITION(arg2, df);
    return u_arg1 >> b_arg2;
}

static inline int64_t lsx_vsra_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    int32_t b_arg2 = BIT_POSITION(arg2, df);
    return arg1 >> b_arg2;
}

static inline int64_t lsx_vsrlr_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    int32_t b_arg2 = BIT_POSITION(arg2, df);
    if (b_arg2 == 0) {
        return u_arg1;
    } else {
        uint64_t r_bit = (u_arg1 >> (b_arg2 - 1)) & 1;
        return (u_arg1 >> b_arg2) + r_bit;
    }
}

static inline int64_t lsx_vsrar_df(uint32_t df, int64_t arg1, int64_t arg2)
{
    int32_t b_arg2 = BIT_POSITION(arg2, df);
    if (b_arg2 == 0) {
        return arg1;
    } else {
        int64_t r_bit = (arg1 >> (b_arg2 - 1)) & 1;
        return (arg1 >> b_arg2) + r_bit;
    }
}

#define LSX_BINOP_IMMU_DF(helper, func)                                  \
void helper_lsx_ ## helper ## _df(CPULoongArchState *env, uint32_t df, uint32_t wd, \
                       uint32_t ws, uint32_t u5)                        \
{                                                                       \
    wr_t *pwd = &(env->fpr[wd].wr);                          \
    wr_t *pws = &(env->fpr[ws].wr);                          \
    uint32_t i;                                                         \
                                                                        \
    switch (df) {                                                       \
    case DF_BYTE:                                                       \
        for (i = 0; i < DF_V_ELEMENTS(DF_BYTE); i++) {                    \
            pwd->b[i] = lsx_ ## func ## _df(df, pws->b[i], u5);         \
        }                                                               \
        break;                                                          \
    case DF_HALF:                                                       \
        for (i = 0; i < DF_V_ELEMENTS(DF_HALF); i++) {                    \
            pwd->h[i] = lsx_ ## func ## _df(df, pws->h[i], u5);         \
        }                                                               \
        break;                                                          \
    case DF_WORD:                                                       \
        for (i = 0; i < DF_V_ELEMENTS(DF_WORD); i++) {                    \
            pwd->w[i] = lsx_ ## func ## _df(df, pws->w[i], u5);         \
        }                                                               \
        break;                                                          \
    case DF_DOUBLE:                                                     \
        for (i = 0; i < DF_V_ELEMENTS(DF_DOUBLE); i++) {                  \
            pwd->d[i] = lsx_ ## func ## _df(df, pws->d[i], u5);         \
        }                                                               \
        break;                                                          \
    default:                                                            \
        assert(0);                                                      \
    }                                                                   \
}
LSX_BINOP_IMMU_DF(vbitclri, vbitclr)
LSX_BINOP_IMMU_DF(vbitseti, vbitset)
LSX_BINOP_IMMU_DF(vbitrevi, vbitrev)
LSX_BINOP_IMMU_DF(vsat_s, vsat_s)
LSX_BINOP_IMMU_DF(vsat_u, vsat_u)
LSX_BINOP_IMMU_DF(vslli, vsll)
LSX_BINOP_IMMU_DF(vsrli, vsrl)
LSX_BINOP_IMMU_DF(vsrai, vsra)
LSX_BINOP_IMMU_DF(vrotri, vrotr)
LSX_BINOP_IMMU_DF(vsrlri, vsrlr)
LSX_BINOP_IMMU_DF(vsrari, vsrar)
#undef LSX_BINOP_IMMU_DF

static inline int64_t lsx_vbstrc12_df(uint32_t df,
                                   int64_t dest, int64_t arg1, int64_t arg2)
{
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    uint64_t u_dest = UNSIGNED(dest, df);
    int32_t sh_d = BIT_POSITION(arg2, df) + 1;
    int32_t sh_a = DF_BITS(df) - sh_d;
    if (sh_d == DF_BITS(df)) {
        return u_arg1;
    } else {
        return UNSIGNED(UNSIGNED(u_dest >> sh_d, df) << sh_d, df) |
               UNSIGNED(UNSIGNED(u_arg1 << sh_a, df) >> sh_a, df);
    }
}

static inline int64_t lsx_vbstrc21_df(uint32_t df,
                                   int64_t dest, int64_t arg1, int64_t arg2)
{
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    uint64_t u_dest = UNSIGNED(dest, df);
    int32_t sh_d = BIT_POSITION(arg2, df) + 1;
    int32_t sh_a = DF_BITS(df) - sh_d;
    if (sh_d == DF_BITS(df)) {
        return u_arg1;
    } else {
        return UNSIGNED(UNSIGNED(u_dest << sh_d, df) >> sh_d, df) |
               UNSIGNED(UNSIGNED(u_arg1 >> sh_a, df) << sh_a, df);
    }
}

#define LSX_TEROP_IMMU_DF(helper, func)                                  \
void helper_lsx_ ## helper ## _df(CPULoongArchState *env, uint32_t df,       \
                                  uint32_t wd, uint32_t ws, uint32_t u5) \
{                                                                       \
    wr_t *pwd = &(env->fpr[wd].wr);                          \
    wr_t *pws = &(env->fpr[ws].wr);                          \
    uint32_t i;                                                         \
                                                                        \
    switch (df) {                                                       \
    case DF_BYTE:                                                       \
        for (i = 0; i < DF_V_ELEMENTS(DF_BYTE); i++) {                    \
            pwd->b[i] = lsx_ ## func ## _df(df, pwd->b[i], pws->b[i],   \
                                            u5);                        \
        }                                                               \
        break;                                                          \
    case DF_HALF:                                                       \
        for (i = 0; i < DF_V_ELEMENTS(DF_HALF); i++) {                    \
            pwd->h[i] = lsx_ ## func ## _df(df, pwd->h[i], pws->h[i],   \
                                            u5);                        \
        }                                                               \
        break;                                                          \
    case DF_WORD:                                                       \
        for (i = 0; i < DF_V_ELEMENTS(DF_WORD); i++) {                    \
            pwd->w[i] = lsx_ ## func ## _df(df, pwd->w[i], pws->w[i],   \
                                            u5);                        \
        }                                                               \
        break;                                                          \
    case DF_DOUBLE:                                                     \
        for (i = 0; i < DF_V_ELEMENTS(DF_DOUBLE); i++) {                  \
            pwd->d[i] = lsx_ ## func ## _df(df, pwd->d[i], pws->d[i],   \
                                            u5);                        \
        }                                                               \
        break;                                                          \
    default:                                                            \
        assert(0);                                                      \
    }                                                                   \
}
LSX_TEROP_IMMU_DF(vbstrc12i, vbstrc12)
LSX_TEROP_IMMU_DF(vbstrc21i, vbstrc21)
#undef LSX_TEROP_IMMU_DF

#define SHF_POS(i, imm) (((i) & 0xfc) + (((imm) >> (2 * ((i) & 0x03))) & 0x03))

void helper_lsx_vshuf4i_df(CPULoongArchState *env, uint32_t df, uint32_t wd,
                       uint32_t ws, uint32_t imm)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t wx, *pwx = &wx;
    uint32_t i;
    switch (df) {
    case DF_BYTE:
        for (i = 0; i < DF_V_ELEMENTS(DF_BYTE); i++) {
            pwx->b[i] = pws->b[SHF_POS(i, imm)];
        }
        break;
    case DF_HALF:
        for (i = 0; i < DF_V_ELEMENTS(DF_HALF); i++) {
            pwx->h[i] = pws->h[SHF_POS(i, imm)];
        }
        break;
    case DF_WORD:
        for (i = 0; i < DF_V_ELEMENTS(DF_WORD); i++) {
            pwx->w[i] = pws->w[SHF_POS(i, imm)];
        }
        break;
    case DF_DOUBLE:
        pwx->d[0] = ((imm & 0x03) == 0x00) ? pwd->d[0] :
                    ((imm & 0x03) == 0x01) ? pwd->d[1] :
                    ((imm & 0x03) == 0x02) ? pws->d[0] :
                                             pws->d[1] ;

        pwx->d[1] = ((imm & 0x0c) == 0x00) ? pwd->d[0] :
                    ((imm & 0x0c) == 0x04) ? pwd->d[1] :
                    ((imm & 0x0c) == 0x08) ? pws->d[0] :
                                             pws->d[1] ;
        break;
    default:
        assert(0);
    }
    lsx_move_v(pwd, pwx);
}

#define BIT_SELECT(dest, arg1, arg2, df) \
            UNSIGNED((arg1 & (~dest)) | (arg2 & dest), df)

#define BIT_MOVE_IF_ZERO(dest, arg1, arg2, df) \
            UNSIGNED((dest & arg2) | (arg1 & (~arg2)), df)

#define BIT_MOVE_IF_NOT_ZERO(dest, arg1, arg2, df) \
            UNSIGNED(((dest & (~arg2)) | (arg1 & arg2)), df)

#define LSX_FN_IMM8(FUNC, DEST, OPERATION)                              \
void helper_lsx_ ## FUNC(CPULoongArchState *env, uint32_t wd, uint32_t ws,   \
        uint32_t i8)                                                    \
{                                                                       \
    wr_t *pwd = &(env->fpr[wd].wr);                          \
    wr_t *pws = &(env->fpr[ws].wr);                          \
    uint32_t i;                                                         \
    for (i = 0; i < DF_V_ELEMENTS(DF_BYTE); i++) {                      \
        DEST = OPERATION;                                               \
    }                                                                   \
}

LSX_FN_IMM8(vbitseli_b, pwd->b[i],
        BIT_SELECT(pwd->b[i], pws->b[i], i8, DF_BYTE))

LSX_FN_IMM8(vbitmvzi_b, pwd->b[i],
        BIT_MOVE_IF_ZERO(pwd->b[i], pws->b[i], i8, DF_BYTE))

LSX_FN_IMM8(vbitmvnzi_b, pwd->b[i],
        BIT_MOVE_IF_NOT_ZERO(pwd->b[i], pws->b[i], i8, DF_BYTE))

LSX_FN_IMM8(vandi_b, pwd->b[i], pws->b[i] & i8)
LSX_FN_IMM8(vori_b, pwd->b[i], pws->b[i] | i8)
LSX_FN_IMM8(vxori_b, pwd->b[i], pws->b[i] ^ i8)
LSX_FN_IMM8(vnori_b, pwd->b[i], ~(pws->b[i] | i8))

#undef BIT_SELECT
#undef BIT_MOVE_IF_ZERO
#undef BIT_MOVE_IF_NOT_ZERO
#undef LSX_FN_IMM8

void helper_lsx_xvshuf_df(CPULoongArchState *env, uint32_t df, uint32_t wd,
                          uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);
    wr_t wx, *pwx = &wx;
    uint32_t i;
    switch (df) {
    case DF_BYTE:
        for (i = 0; i < DF_ELEMENTS(DF_BYTE) ; i++) {
            uint32_t n = DF_V_ELEMENTS(DF_BYTE);
            uint32_t k = (pwd->b[i] & 0x3f) % (2 * n);
            if(i < n)
                pwx->b[i] =
                    (pwd->b[i] & 0xc0) ? 0 : k < n ? pwt->b[k] : pws->b[k - n];
            else
                pwx->b[i] =
                    (pwd->b[i] & 0xc0) ? 0 : k < n ? pwt->b[k + n] : pws->b[k];
        }
        break;
    case DF_HALF:
        for (i = 0; i < DF_ELEMENTS(DF_HALF) ; i++) {
            uint32_t n = DF_V_ELEMENTS(DF_HALF);
            uint32_t k = (pwd->h[i] & 0x3f) % (2 * n);
            if(i < n)
                pwx->h[i] =
                    (pwd->h[i] & 0xc0) ? 0 : k < n ? pwt->h[k] : pws->h[k - n];
            else
                pwx->h[i] =
                    (pwd->h[i] & 0xc0) ? 0 : k < n ? pwt->h[k + n] : pws->h[k];
        }
        break;
    case DF_WORD:
        for (i = 0; i < DF_ELEMENTS(DF_WORD) ; i++) {
            uint32_t n = DF_V_ELEMENTS(DF_WORD);
            uint32_t k = (pwd->w[i] & 0x3f) % (2 * n);
            if(i < n)
                pwx->w[i] =
                    (pwd->w[i] & 0xc0) ? 0 : k < n ? pwt->w[k] : pws->w[k - n];
            else
                pwx->w[i] =
                    (pwd->w[i] & 0xc0) ? 0 : k < n ? pwt->w[k + n] : pws->w[k];
        }
        break;
    case DF_DOUBLE:
        for (i = 0; i < DF_ELEMENTS(DF_DOUBLE) ; i++) {
            uint32_t n = DF_V_ELEMENTS(DF_DOUBLE);
            uint32_t k = (pwd->d[i] & 0x3f) % (2 * n);
            if(i < n)
                pwx->d[i] =
                    (pwd->d[i] & 0xc0) ? 0 : k < n ? pwt->d[k] : pws->d[k - n];
            else
                pwx->d[i] =
                    (pwd->d[i] & 0xc0) ? 0 : k < n ? pwt->d[k + n] : pws->d[k];
        }
        break;
    default:
        assert(0);
    }
    lsx_move_x(pwd, pwx);
}

void helper_lsx_xvperm_df(CPULoongArchState *env, uint32_t df, uint32_t wd,
                          uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);
    uint32_t i;
    switch (df) {
    case DF_WORD:
        for (i = 0; i < DF_ELEMENTS(DF_WORD) ; i++) {
            uint32_t index = pwt->w[i] & 0x7;
            pwd->w[i] = pws->w[index];
        }
        break;
    default:
        assert(0);
    }
}

#define LSX_XBINOP_IMM_DF(helper, func)                                 \
void helper_lsx_ ## helper ## _df(CPULoongArchState *env, uint32_t df,       \
                        uint32_t wd, uint32_t ws, int32_t u5)           \
{                                                                       \
    wr_t *pwd = &(env->fpr[wd].wr);                          \
    wr_t *pws = &(env->fpr[ws].wr);                          \
    uint32_t i;                                                         \
                                                                        \
    switch (df) {                                                       \
    case DF_BYTE:                                                       \
        for (i = 0; i < DF_ELEMENTS(DF_BYTE); i++) {                    \
            pwd->b[i] = lsx_ ## func ## _df(df, pws->b[i], u5);         \
        }                                                               \
        break;                                                          \
    case DF_HALF:                                                       \
        for (i = 0; i < DF_ELEMENTS(DF_HALF); i++) {                    \
            pwd->h[i] = lsx_ ## func ## _df(df, pws->h[i], u5);         \
        }                                                               \
        break;                                                          \
    case DF_WORD:                                                       \
        for (i = 0; i < DF_ELEMENTS(DF_WORD); i++) {                    \
            pwd->w[i] = lsx_ ## func ## _df(df, pws->w[i], u5);         \
        }                                                               \
        break;                                                          \
    case DF_DOUBLE:                                                     \
        for (i = 0; i < DF_ELEMENTS(DF_DOUBLE); i++) {                  \
            pwd->d[i] = lsx_ ## func ## _df(df, pws->d[i], u5);         \
        }                                                               \
        break;                                                          \
    default:                                                            \
        assert(0);                                                      \
    }                                                                   \
}
LSX_XBINOP_IMM_DF(xvseqi, vseq)
LSX_XBINOP_IMM_DF(xvslei_s, vsle_s)
LSX_XBINOP_IMM_DF(xvslei_u, vsle_u)
LSX_XBINOP_IMM_DF(xvslti_s, vslt_s)
LSX_XBINOP_IMM_DF(xvslti_u, vslt_u)
LSX_XBINOP_IMM_DF(xvaddi_u, vadd)
LSX_XBINOP_IMM_DF(xvsubi_u, vsub)
LSX_XBINOP_IMM_DF(xvmaxi_s, vmax_s)
LSX_XBINOP_IMM_DF(xvmini_s, vmin_s)
LSX_XBINOP_IMM_DF(xvmaxi_u, vmax_u)
LSX_XBINOP_IMM_DF(xvmini_u, vmin_u)
#undef LSX_XBINOP_IMM_DF

void helper_lsx_xvbsll_v(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t u5)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

	wr_t tmp;
	tmp.q[0] = pwd->q[0];
	tmp.q[1] = pwd->q[1];
    for(int i = 0;  i < 16; i++) {
        tmp.b[i] = (i < u5)? 0 : pws->b[i-u5];
	}

    for(int i = 16; i < 32; i++) {
        tmp.b[i] = (i < u5 + 16)? 0 : pws->b[i-u5];
	}
	pwd->q[0] = tmp.q[0];
	pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvbsrl_v(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t u5)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    for(int i = 0;  i < 16; i++)
        pwd->b[i] = (i + u5 > 15)? 0 : pws->b[i+u5];

    for(int i = 16; i < 32; i++)
        pwd->b[i] = (i + u5 > 31)? 0 : pws->b[i+u5];
}

void helper_lsx_xvclo_b(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    for(int i = 0; i < 32; i++)
        pwd->b[i] = lsx_vclo_df(DF_BYTE, pws->b[i]);
}

void helper_lsx_xvclo_h(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    for(int i = 0; i < 16; i++)
        pwd->h[i]  = lsx_vclo_df(DF_HALF, pws->h[i]);
}

void helper_lsx_xvclo_w(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    for(int i = 0; i < 8; i++)
        pwd->w[i]  = lsx_vclo_df(DF_WORD, pws->w[i]);
}

void helper_lsx_xvclo_d(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    for(int i = 0; i < 4; i++)
        pwd->d[i]  = lsx_vclo_df(DF_DOUBLE, pws->d[i]);
}


void helper_lsx_xvclz_b(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    for(int i = 0; i < 32; i++)
        pwd->b[i] = lsx_vclz_df(DF_BYTE, pws->b[i]);
}

void helper_lsx_xvclz_h(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    for(int i = 0; i < 16; i++)
        pwd->h[i]  = lsx_vclz_df(DF_HALF, pws->h[i]);
}

void helper_lsx_xvclz_w(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    for(int i = 0; i < 8; i++)
        pwd->w[i]  = lsx_vclz_df(DF_WORD, pws->w[i]);
}

void helper_lsx_xvclz_d(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    for(int i = 0; i < 4; i++)
        pwd->d[i]  = lsx_vclz_df(DF_DOUBLE, pws->d[i]);
}

void helper_lsx_xvpcnt_b(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    for(int i = 0; i < 32; i++)
        pwd->b[i] = lsx_vpcnt_df(DF_BYTE, pws->b[i]);
}

void helper_lsx_xvpcnt_h(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    for(int i = 0; i < 16; i++)
        pwd->h[i]  = lsx_vpcnt_df(DF_HALF, pws->h[i]);
}

void helper_lsx_xvpcnt_w(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    for(int i = 0; i < 8; i++)
        pwd->w[i]  = lsx_vpcnt_df(DF_WORD, pws->w[i]);
}

void helper_lsx_xvpcnt_d(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    for(int i = 0; i < 4; i++)
        pwd->d[i]  = lsx_vpcnt_df(DF_DOUBLE, pws->d[i]);
}

void helper_lsx_xvneg_b(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    for(int i = 0; i < 32; i++)
        pwd->b[i] = -pws->b[i];
}

void helper_lsx_xvneg_h(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    for(int i = 0; i < 16; i++)
        pwd->h[i]  = -pws->h[i];
}

void helper_lsx_xvneg_w(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    for(int i = 0; i < 8; i++)
        pwd->w[i]  = -pws->w[i];
}

void helper_lsx_xvneg_d(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    for(int i = 0; i < 4; i++)
        pwd->d[i]  = -pws->d[i];
}

void helper_lsx_xvmskltz_b(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    pwd->d[0]  = 0;
    pwd->d[1]  = 0;
    pwd->d[2]  = 0;
    pwd->d[3]  = 0;

    pwd->h[0]  = (((0x80 & pws->b[0]) != 0) << 0) |
                 (((0x80 & pws->b[1]) != 0) << 1) |
                 (((0x80 & pws->b[2]) != 0) << 2) |
                 (((0x80 & pws->b[3]) != 0) << 3) |
                 (((0x80 & pws->b[4]) != 0) << 4) |
                 (((0x80 & pws->b[5]) != 0) << 5) |
                 (((0x80 & pws->b[6]) != 0) << 6) |
                 (((0x80 & pws->b[7]) != 0) << 7) |
                 (((0x80 & pws->b[8]) != 0) << 8) |
                 (((0x80 & pws->b[9]) != 0) << 9) |
                 (((0x80 & pws->b[10]) != 0) << 10) |
                 (((0x80 & pws->b[11]) != 0) << 11) |
                 (((0x80 & pws->b[12]) != 0) << 12) |
                 (((0x80 & pws->b[13]) != 0) << 13) |
                 (((0x80 & pws->b[14]) != 0) << 14) |
                 (((0x80 & pws->b[15]) != 0) << 15) ;

    pwd->h[8]  = (((0x80 & pws->b[16]) != 0) << 0) |
                 (((0x80 & pws->b[17]) != 0) << 1) |
                 (((0x80 & pws->b[18]) != 0) << 2) |
                 (((0x80 & pws->b[19]) != 0) << 3) |
                 (((0x80 & pws->b[20]) != 0) << 4) |
                 (((0x80 & pws->b[21]) != 0) << 5) |
                 (((0x80 & pws->b[22]) != 0) << 6) |
                 (((0x80 & pws->b[23]) != 0) << 7) |
                 (((0x80 & pws->b[24]) != 0) << 8) |
                 (((0x80 & pws->b[25]) != 0) << 9) |
                 (((0x80 & pws->b[26]) != 0) << 10) |
                 (((0x80 & pws->b[27]) != 0) << 11) |
                 (((0x80 & pws->b[28]) != 0) << 12) |
                 (((0x80 & pws->b[29]) != 0) << 13) |
                 (((0x80 & pws->b[30]) != 0) << 14) |
                 (((0x80 & pws->b[31]) != 0) << 15) ;
}

void helper_lsx_xvmskltz_h(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    pwd->d[0]  = 0;
    pwd->d[1]  = 0;
    pwd->d[2]  = 0;
    pwd->d[3]  = 0;

    pwd->h[0]  = (((0x8000 & pws->h[0]) != 0) << 0) |
                 (((0x8000 & pws->h[1]) != 0) << 1) |
                 (((0x8000 & pws->h[2]) != 0) << 2) |
                 (((0x8000 & pws->h[3]) != 0) << 3) |
                 (((0x8000 & pws->h[4]) != 0) << 4) |
                 (((0x8000 & pws->h[5]) != 0) << 5) |
                 (((0x8000 & pws->h[6]) != 0) << 6) |
                 (((0x8000 & pws->h[7]) != 0) << 7) ;

    pwd->h[8]  = (((0x8000 & pws->h[8]) != 0) << 0) |
                 (((0x8000 & pws->h[9]) != 0) << 1) |
                 (((0x8000 & pws->h[10]) != 0) << 2) |
                 (((0x8000 & pws->h[11]) != 0) << 3) |
                 (((0x8000 & pws->h[12]) != 0) << 4) |
                 (((0x8000 & pws->h[13]) != 0) << 5) |
                 (((0x8000 & pws->h[14]) != 0) << 6) |
                 (((0x8000 & pws->h[15]) != 0) << 7) ;
}

void helper_lsx_xvmskltz_w(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    pwd->d[0]  = 0;
    pwd->d[1]  = 0;
    pwd->d[2]  = 0;
    pwd->d[3]  = 0;

    pwd->w[0]  = (((0x80000000 & pws->w[0]) != 0) << 0) |
                 (((0x80000000 & pws->w[1]) != 0) << 1) |
                 (((0x80000000 & pws->w[2]) != 0) << 2) |
                 (((0x80000000 & pws->w[3]) != 0) << 3) ;

    pwd->w[4]  = (((0x80000000 & pws->w[4]) != 0) << 0) |
                 (((0x80000000 & pws->w[5]) != 0) << 1) |
                 (((0x80000000 & pws->w[6]) != 0) << 2) |
                 (((0x80000000 & pws->w[7]) != 0) << 3) ;
}

void helper_lsx_xvmskltz_d(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    pwd->d[0]  = 0;
    pwd->d[1]  = 0;
    pwd->d[2]  = 0;
    pwd->d[3]  = 0;

    pwd->d[0]  = (((0x8000000000000000 & pws->d[0]) != 0) << 0) |
                 (((0x8000000000000000 & pws->d[1]) != 0) << 1) ;

    pwd->d[2]  = (((0x8000000000000000 & pws->d[2]) != 0) << 0) |
                 (((0x8000000000000000 & pws->d[3]) != 0) << 1) ;
}

void helper_lsx_xvmskgez_b(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    pwd->d[0]  = 0;
    pwd->d[1]  = 0;
    pwd->d[2]  = 0;
    pwd->d[3]  = 0;

    pwd->h[0]  = ( !(pws->b[0]  < 0) << 0 ) |
                 ( !(pws->b[1]  < 0) << 1 ) |
                 ( !(pws->b[2]  < 0) << 2 ) |
                 ( !(pws->b[3]  < 0) << 3 ) |
                 ( !(pws->b[4]  < 0) << 4 ) |
                 ( !(pws->b[5]  < 0) << 5 ) |
                 ( !(pws->b[6]  < 0) << 6 ) |
                 ( !(pws->b[7]  < 0) << 7 ) |
                 ( !(pws->b[8]  < 0) << 8 ) |
                 ( !(pws->b[9]  < 0) << 9 ) |
                 ( !(pws->b[10] < 0) << 10) |
                 ( !(pws->b[11] < 0) << 11) |
                 ( !(pws->b[12] < 0) << 12) |
                 ( !(pws->b[13] < 0) << 13) |
                 ( !(pws->b[14] < 0) << 14) |
                 ( !(pws->b[15] < 0) << 15) ;

    pwd->h[8]  = ( !(pws->b[16] < 0) << 0 ) |
                 ( !(pws->b[17] < 0) << 1 ) |
                 ( !(pws->b[18] < 0) << 2 ) |
                 ( !(pws->b[19] < 0) << 3 ) |
                 ( !(pws->b[20] < 0) << 4 ) |
                 ( !(pws->b[21] < 0) << 5 ) |
                 ( !(pws->b[22] < 0) << 6 ) |
                 ( !(pws->b[23] < 0) << 7 ) |
                 ( !(pws->b[24] < 0) << 8 ) |
                 ( !(pws->b[25] < 0) << 9 ) |
                 ( !(pws->b[26] < 0) << 10) |
                 ( !(pws->b[27] < 0) << 11) |
                 ( !(pws->b[28] < 0) << 12) |
                 ( !(pws->b[29] < 0) << 13) |
                 ( !(pws->b[30] < 0) << 14) |
                 ( !(pws->b[31] < 0) << 15) ;
}

void helper_lsx_xvmsknz_b(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    pwd->d[0]  = 0;
    pwd->d[1]  = 0;
    pwd->d[2]  = 0;
    pwd->d[3]  = 0;

    pwd->h[0]  = ( (pws->b[0]  != 0) << 0 ) |
                 ( (pws->b[1]  != 0) << 1 ) |
                 ( (pws->b[2]  != 0) << 2 ) |
                 ( (pws->b[3]  != 0) << 3 ) |
                 ( (pws->b[4]  != 0) << 4 ) |
                 ( (pws->b[5]  != 0) << 5 ) |
                 ( (pws->b[6]  != 0) << 6 ) |
                 ( (pws->b[7]  != 0) << 7 ) |
                 ( (pws->b[8]  != 0) << 8 ) |
                 ( (pws->b[9]  != 0) << 9 ) |
                 ( (pws->b[10] != 0) << 10) |
                 ( (pws->b[11] != 0) << 11) |
                 ( (pws->b[12] != 0) << 12) |
                 ( (pws->b[13] != 0) << 13) |
                 ( (pws->b[14] != 0) << 14) |
                 ( (pws->b[15] != 0) << 15) ;

    pwd->h[8]  = ( (pws->b[16] != 0) << 0 ) |
                 ( (pws->b[17] != 0) << 1 ) |
                 ( (pws->b[18] != 0) << 2 ) |
                 ( (pws->b[19] != 0) << 3 ) |
                 ( (pws->b[20] != 0) << 4 ) |
                 ( (pws->b[21] != 0) << 5 ) |
                 ( (pws->b[22] != 0) << 6 ) |
                 ( (pws->b[23] != 0) << 7 ) |
                 ( (pws->b[24] != 0) << 8 ) |
                 ( (pws->b[25] != 0) << 9 ) |
                 ( (pws->b[26] != 0) << 10) |
                 ( (pws->b[27] != 0) << 11) |
                 ( (pws->b[28] != 0) << 12) |
                 ( (pws->b[29] != 0) << 13) |
                 ( (pws->b[30] != 0) << 14) |
                 ( (pws->b[31] != 0) << 15) ;
}

void helper_lsx_xvmskcopy_b(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    pwd->b[0]  = (pws->h[0] & 0x0001) != 0;
    pwd->b[1]  = (pws->h[0] & 0x0002) != 0;
    pwd->b[2]  = (pws->h[0] & 0x0004) != 0;
    pwd->b[3]  = (pws->h[0] & 0x0008) != 0;
    pwd->b[4]  = (pws->h[0] & 0x0010) != 0;
    pwd->b[5]  = (pws->h[0] & 0x0020) != 0;
    pwd->b[6]  = (pws->h[0] & 0x0040) != 0;
    pwd->b[7]  = (pws->h[0] & 0x0080) != 0;
    pwd->b[8]  = (pws->h[0] & 0x0100) != 0;
    pwd->b[9]  = (pws->h[0] & 0x0200) != 0;
    pwd->b[10] = (pws->h[0] & 0x0400) != 0;
    pwd->b[11] = (pws->h[0] & 0x0800) != 0;
    pwd->b[12] = (pws->h[0] & 0x1000) != 0;
    pwd->b[13] = (pws->h[0] & 0x2000) != 0;
    pwd->b[14] = (pws->h[0] & 0x4000) != 0;
    pwd->b[15] = (pws->h[0] & 0x8000) != 0;

    pwd->b[16] = (pws->h[8] & 0x0001) != 0;
    pwd->b[17] = (pws->h[8] & 0x0002) != 0;
    pwd->b[18] = (pws->h[8] & 0x0004) != 0;
    pwd->b[19] = (pws->h[8] & 0x0008) != 0;
    pwd->b[20] = (pws->h[8] & 0x0010) != 0;
    pwd->b[21] = (pws->h[8] & 0x0020) != 0;
    pwd->b[22] = (pws->h[8] & 0x0040) != 0;
    pwd->b[23] = (pws->h[8] & 0x0080) != 0;
    pwd->b[24] = (pws->h[8] & 0x0100) != 0;
    pwd->b[25] = (pws->h[8] & 0x0200) != 0;
    pwd->b[26] = (pws->h[8] & 0x0400) != 0;
    pwd->b[27] = (pws->h[8] & 0x0800) != 0;
    pwd->b[28] = (pws->h[8] & 0x1000) != 0;
    pwd->b[29] = (pws->h[8] & 0x2000) != 0;
    pwd->b[30] = (pws->h[8] & 0x4000) != 0;
    pwd->b[31] = (pws->h[8] & 0x8000) != 0;
}

void helper_lsx_xvmskfill_b(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    pwd->b[0]  = ((pws->h[0] & 0x0001) != 0)? 0xff : 0x00;
    pwd->b[1]  = ((pws->h[0] & 0x0002) != 0)? 0xff : 0x00;
    pwd->b[2]  = ((pws->h[0] & 0x0004) != 0)? 0xff : 0x00;
    pwd->b[3]  = ((pws->h[0] & 0x0008) != 0)? 0xff : 0x00;
    pwd->b[4]  = ((pws->h[0] & 0x0010) != 0)? 0xff : 0x00;
    pwd->b[5]  = ((pws->h[0] & 0x0020) != 0)? 0xff : 0x00;
    pwd->b[6]  = ((pws->h[0] & 0x0040) != 0)? 0xff : 0x00;
    pwd->b[7]  = ((pws->h[0] & 0x0080) != 0)? 0xff : 0x00;
    pwd->b[8]  = ((pws->h[0] & 0x0100) != 0)? 0xff : 0x00;
    pwd->b[9]  = ((pws->h[0] & 0x0200) != 0)? 0xff : 0x00;
    pwd->b[10] = ((pws->h[0] & 0x0400) != 0)? 0xff : 0x00;
    pwd->b[11] = ((pws->h[0] & 0x0800) != 0)? 0xff : 0x00;
    pwd->b[12] = ((pws->h[0] & 0x1000) != 0)? 0xff : 0x00;
    pwd->b[13] = ((pws->h[0] & 0x2000) != 0)? 0xff : 0x00;
    pwd->b[14] = ((pws->h[0] & 0x4000) != 0)? 0xff : 0x00;
    pwd->b[15] = ((pws->h[0] & 0x8000) != 0)? 0xff : 0x00;

    pwd->b[16] = ((pws->h[8] & 0x0001) != 0)? 0xff : 0x00;
    pwd->b[17] = ((pws->h[8] & 0x0002) != 0)? 0xff : 0x00;
    pwd->b[18] = ((pws->h[8] & 0x0004) != 0)? 0xff : 0x00;
    pwd->b[19] = ((pws->h[8] & 0x0008) != 0)? 0xff : 0x00;
    pwd->b[20] = ((pws->h[8] & 0x0010) != 0)? 0xff : 0x00;
    pwd->b[21] = ((pws->h[8] & 0x0020) != 0)? 0xff : 0x00;
    pwd->b[22] = ((pws->h[8] & 0x0040) != 0)? 0xff : 0x00;
    pwd->b[23] = ((pws->h[8] & 0x0080) != 0)? 0xff : 0x00;
    pwd->b[24] = ((pws->h[8] & 0x0100) != 0)? 0xff : 0x00;
    pwd->b[25] = ((pws->h[8] & 0x0200) != 0)? 0xff : 0x00;
    pwd->b[26] = ((pws->h[8] & 0x0400) != 0)? 0xff : 0x00;
    pwd->b[27] = ((pws->h[8] & 0x0800) != 0)? 0xff : 0x00;
    pwd->b[28] = ((pws->h[8] & 0x1000) != 0)? 0xff : 0x00;
    pwd->b[29] = ((pws->h[8] & 0x2000) != 0)? 0xff : 0x00;
    pwd->b[30] = ((pws->h[8] & 0x4000) != 0)? 0xff : 0x00;
    pwd->b[31] = ((pws->h[8] & 0x8000) != 0)? 0xff : 0x00;
}

void helper_lsx_xvseteqz_v(CPULoongArchState *env, uint32_t cd, uint32_t ws)
{
    wr_t *pws = &(env->fpr[ws].wr);
    env->cf[cd & 0x7] = (pws->q[0] == 0 && pws->q[1] == 0);
}

void helper_lsx_xvsetnez_v(CPULoongArchState *env, uint32_t cd, uint32_t ws)
{
    wr_t *pws = &(env->fpr[ws].wr);
    env->cf[cd & 0x7] = (pws->q[0] != 0 || pws->q[1] != 0);
}

void helper_lsx_xvsetanyeqz_b(CPULoongArchState *env, uint32_t cd, uint32_t ws)
{
    wr_t *pws = &(env->fpr[ws].wr);
    env->cf[cd & 0x7] = (pws->b[0]  == 0) ||
                        (pws->b[1]  == 0) ||
                        (pws->b[2]  == 0) ||
                        (pws->b[3]  == 0) ||
                        (pws->b[4]  == 0) ||
                        (pws->b[5]  == 0) ||
                        (pws->b[6]  == 0) ||
                        (pws->b[7]  == 0) ||
                        (pws->b[8]  == 0) ||
                        (pws->b[9]  == 0) ||
                        (pws->b[10] == 0) ||
                        (pws->b[11] == 0) ||
                        (pws->b[12] == 0) ||
                        (pws->b[13] == 0) ||
                        (pws->b[14] == 0) ||
                        (pws->b[15] == 0) ||
                        (pws->b[16] == 0) ||
                        (pws->b[17] == 0) ||
                        (pws->b[18] == 0) ||
                        (pws->b[19] == 0) ||
                        (pws->b[20] == 0) ||
                        (pws->b[21] == 0) ||
                        (pws->b[22] == 0) ||
                        (pws->b[23] == 0) ||
                        (pws->b[24] == 0) ||
                        (pws->b[25] == 0) ||
                        (pws->b[26] == 0) ||
                        (pws->b[27] == 0) ||
                        (pws->b[28] == 0) ||
                        (pws->b[29] == 0) ||
                        (pws->b[30] == 0) ||
                        (pws->b[31] == 0) ;
}

void helper_lsx_xvsetanyeqz_h(CPULoongArchState *env, uint32_t cd, uint32_t ws)
{
    wr_t *pws = &(env->fpr[ws].wr);
    env->cf[cd & 0x7] = (pws->h[0]  == 0) ||
                        (pws->h[1]  == 0) ||
                        (pws->h[2]  == 0) ||
                        (pws->h[3]  == 0) ||
                        (pws->h[4]  == 0) ||
                        (pws->h[5]  == 0) ||
                        (pws->h[6]  == 0) ||
                        (pws->h[7]  == 0) ||
                        (pws->h[8]  == 0) ||
                        (pws->h[9]  == 0) ||
                        (pws->h[10] == 0) ||
                        (pws->h[11] == 0) ||
                        (pws->h[12] == 0) ||
                        (pws->h[13] == 0) ||
                        (pws->h[14] == 0) ||
                        (pws->h[15] == 0) ;
}

void helper_lsx_xvsetanyeqz_w(CPULoongArchState *env, uint32_t cd, uint32_t ws)
{
    wr_t *pws = &(env->fpr[ws].wr);
    env->cf[cd & 0x7] = (pws->w[0] == 0) ||
                        (pws->w[1] == 0) ||
                        (pws->w[2] == 0) ||
                        (pws->w[3] == 0) ||
                        (pws->w[4] == 0) ||
                        (pws->w[5] == 0) ||
                        (pws->w[6] == 0) ||
                        (pws->w[7] == 0) ;
}

void helper_lsx_xvsetanyeqz_d(CPULoongArchState *env, uint32_t cd, uint32_t ws)
{
    wr_t *pws = &(env->fpr[ws].wr);
    env->cf[cd & 0x7] = (pws->d[0] == 0) ||
                        (pws->d[1] == 0) ||
                        (pws->d[2] == 0) ||
                        (pws->d[3] == 0) ;
}

void helper_lsx_xvsetallnez_b(CPULoongArchState *env, uint32_t cd, uint32_t ws)
{
    wr_t *pws = &(env->fpr[ws].wr);
    env->cf[cd & 0x7] = (pws->b[0]  != 0) &&
                        (pws->b[1]  != 0) &&
                        (pws->b[2]  != 0) &&
                        (pws->b[3]  != 0) &&
                        (pws->b[4]  != 0) &&
                        (pws->b[5]  != 0) &&
                        (pws->b[6]  != 0) &&
                        (pws->b[7]  != 0) &&
                        (pws->b[8]  != 0) &&
                        (pws->b[9]  != 0) &&
                        (pws->b[10] != 0) &&
                        (pws->b[11] != 0) &&
                        (pws->b[12] != 0) &&
                        (pws->b[13] != 0) &&
                        (pws->b[14] != 0) &&
                        (pws->b[15] != 0) &&
                        (pws->b[16] != 0) &&
                        (pws->b[17] != 0) &&
                        (pws->b[18] != 0) &&
                        (pws->b[19] != 0) &&
                        (pws->b[20] != 0) &&
                        (pws->b[21] != 0) &&
                        (pws->b[22] != 0) &&
                        (pws->b[23] != 0) &&
                        (pws->b[24] != 0) &&
                        (pws->b[25] != 0) &&
                        (pws->b[26] != 0) &&
                        (pws->b[27] != 0) &&
                        (pws->b[28] != 0) &&
                        (pws->b[29] != 0) &&
                        (pws->b[30] != 0) &&
                        (pws->b[31] != 0) ;
}

void helper_lsx_xvsetallnez_h(CPULoongArchState *env, uint32_t cd, uint32_t ws)
{
    wr_t *pws = &(env->fpr[ws].wr);
    env->cf[cd & 0x7] = (pws->h[0]  != 0) &&
                        (pws->h[1]  != 0) &&
                        (pws->h[2]  != 0) &&
                        (pws->h[3]  != 0) &&
                        (pws->h[4]  != 0) &&
                        (pws->h[5]  != 0) &&
                        (pws->h[6]  != 0) &&
                        (pws->h[7]  != 0) &&
                        (pws->h[8]  != 0) &&
                        (pws->h[9]  != 0) &&
                        (pws->h[10] != 0) &&
                        (pws->h[11] != 0) &&
                        (pws->h[12] != 0) &&
                        (pws->h[13] != 0) &&
                        (pws->h[14] != 0) &&
                        (pws->h[15] != 0) ;
}

void helper_lsx_xvsetallnez_w(CPULoongArchState *env, uint32_t cd, uint32_t ws)
{
    wr_t *pws = &(env->fpr[ws].wr);
    env->cf[cd & 0x7] = (pws->w[0] != 0) &&
                        (pws->w[1] != 0) &&
                        (pws->w[2] != 0) &&
                        (pws->w[3] != 0) &&
                        (pws->w[4] != 0) &&
                        (pws->w[5] != 0) &&
                        (pws->w[6] != 0) &&
                        (pws->w[7] != 0) ;
}

void helper_lsx_xvsetallnez_d(CPULoongArchState *env, uint32_t cd, uint32_t ws)
{
    wr_t *pws = &(env->fpr[ws].wr);
    env->cf[cd & 0x7] = (pws->d[0] != 0) &&
                        (pws->d[1] != 0) &&
                        (pws->d[2] != 0) &&
                        (pws->d[3] != 0) ;
}

#define LSX_XBINOP_IMMU_DF(helper, func)                                  \
void helper_lsx_ ## helper ## _df(CPULoongArchState *env, uint32_t df, uint32_t wd, \
                       uint32_t ws, uint32_t u5)                        \
{                                                                       \
    wr_t *pwd = &(env->fpr[wd].wr);                          \
    wr_t *pws = &(env->fpr[ws].wr);                          \
    uint32_t i;                                                         \
                                                                        \
    switch (df) {                                                       \
    case DF_BYTE:                                                       \
        for (i = 0; i < DF_ELEMENTS(DF_BYTE); i++) {                    \
            pwd->b[i] = lsx_ ## func ## _df(df, pws->b[i], u5);         \
        }                                                               \
        break;                                                          \
    case DF_HALF:                                                       \
        for (i = 0; i < DF_ELEMENTS(DF_HALF); i++) {                    \
            pwd->h[i] = lsx_ ## func ## _df(df, pws->h[i], u5);         \
        }                                                               \
        break;                                                          \
    case DF_WORD:                                                       \
        for (i = 0; i < DF_ELEMENTS(DF_WORD); i++) {                    \
            pwd->w[i] = lsx_ ## func ## _df(df, pws->w[i], u5);         \
        }                                                               \
        break;                                                          \
    case DF_DOUBLE:                                                     \
        for (i = 0; i < DF_ELEMENTS(DF_DOUBLE); i++) {                  \
            pwd->d[i] = lsx_ ## func ## _df(df, pws->d[i], u5);         \
        }                                                               \
        break;                                                          \
    default:                                                            \
        assert(0);                                                      \
    }                                                                   \
}
LSX_XBINOP_IMMU_DF(xvbitclri, vbitclr)
LSX_XBINOP_IMMU_DF(xvbitseti, vbitset)
LSX_XBINOP_IMMU_DF(xvbitinvi, vbitrev)
LSX_XBINOP_IMMU_DF(xvsat_s, vsat_s)
LSX_XBINOP_IMMU_DF(xvsat_u, vsat_u)
LSX_XBINOP_IMMU_DF(xvslli, vsll)
LSX_XBINOP_IMMU_DF(xvsrli, vsrl)
LSX_XBINOP_IMMU_DF(xvsrai, vsra)
LSX_XBINOP_IMMU_DF(xvrotri, vrotr)
LSX_XBINOP_IMMU_DF(xvsrlri, vsrlr)
LSX_XBINOP_IMMU_DF(xvsrari, vsrar)
#undef LSX_XBINOP_IMMU_DF

#define LSX_XTEROP_IMMU_DF(helper, func)                                \
void helper_lsx_ ## helper ## _df(CPULoongArchState *env, uint32_t df,       \
                                 uint32_t wd, uint32_t ws, uint32_t u5) \
{                                                                       \
    wr_t *pwd = &(env->fpr[wd].wr);                          \
    wr_t *pws = &(env->fpr[ws].wr);                          \
    uint32_t i;                                                         \
                                                                        \
    switch (df) {                                                       \
    case DF_BYTE:                                                       \
        for (i = 0; i < DF_ELEMENTS(DF_BYTE); i++) {                    \
            pwd->b[i] = lsx_ ## func ## _df(df, pwd->b[i], pws->b[i],   \
                                            u5);                        \
        }                                                               \
        break;                                                          \
    case DF_HALF:                                                       \
        for (i = 0; i < DF_ELEMENTS(DF_HALF); i++) {                    \
            pwd->h[i] = lsx_ ## func ## _df(df, pwd->h[i], pws->h[i],   \
                                            u5);                        \
        }                                                               \
        break;                                                          \
    case DF_WORD:                                                       \
        for (i = 0; i < DF_ELEMENTS(DF_WORD); i++) {                    \
            pwd->w[i] = lsx_ ## func ## _df(df, pwd->w[i], pws->w[i],   \
                                            u5);                        \
        }                                                               \
        break;                                                          \
    case DF_DOUBLE:                                                     \
        for (i = 0; i < DF_ELEMENTS(DF_DOUBLE); i++) {                  \
            pwd->d[i] = lsx_ ## func ## _df(df, pwd->d[i], pws->d[i],   \
                                            u5);                        \
        }                                                               \
        break;                                                          \
    default:                                                            \
        assert(0);                                                      \
    }                                                                   \
}
LSX_XTEROP_IMMU_DF(xvbstrc12i, vbstrc12)
LSX_XTEROP_IMMU_DF(xvbstrc21i, vbstrc21)
#undef LSX_XTEROP_IMMU_DF


#define XSHF_POS(i, imm) (((i) & 0xfc) + (((imm) >> (2 * ((i) & 0x03))) & 0x03))

void helper_lsx_xvshuf4i_df(CPULoongArchState *env, uint32_t df, uint32_t wd,
                       uint32_t ws, uint32_t imm)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t wx, *pwx = &wx;
    uint32_t i;
    switch (df) {
    case DF_BYTE:
        for (i = 0; i < DF_ELEMENTS(DF_BYTE); i++) {
            if(i < 16)
                pwx->b[i] = pws->b[XSHF_POS(i, imm)];
            else
                pwx->b[i] = pws->b[XSHF_POS(i-16, imm) + 16];
        }
        break;
    case DF_HALF:
        for (i = 0; i < DF_ELEMENTS(DF_HALF); i++) {
            if(i < 8)
                pwx->h[i] = pws->h[XSHF_POS(i, imm)];
            else
                pwx->h[i] = pws->h[XSHF_POS(i-8, imm) + 8];
        }
        break;
    case DF_WORD:
        for (i = 0; i < DF_ELEMENTS(DF_WORD); i++) {
            if(i < 4)
                pwx->w[i] = pws->w[XSHF_POS(i, imm)];
            else
                pwx->w[i] = pws->w[XSHF_POS(i-4, imm) + 4];
        }
        break;
    case DF_DOUBLE:
        pwx->d[0] = ((imm & 0x03) == 0x00) ? pwd->d[0] :
                    ((imm & 0x03) == 0x01) ? pwd->d[1] :
                    ((imm & 0x03) == 0x02) ? pws->d[0] :
                                             pws->d[1] ;

        pwx->d[1] = ((imm & 0x0c) == 0x00) ? pwd->d[0] :
                    ((imm & 0x0c) == 0x04) ? pwd->d[1] :
                    ((imm & 0x0c) == 0x08) ? pws->d[0] :
                                             pws->d[1] ;

        pwx->d[2] = ((imm & 0x03) == 0x00) ? pwd->d[2] :
                    ((imm & 0x03) == 0x01) ? pwd->d[3] :
                    ((imm & 0x03) == 0x02) ? pws->d[2] :
                                             pws->d[3] ;

        pwx->d[3] = ((imm & 0x0c) == 0x00) ? pwd->d[2] :
                    ((imm & 0x0c) == 0x04) ? pwd->d[3] :
                    ((imm & 0x0c) == 0x08) ? pws->d[2] :
                                             pws->d[3] ;
        break;
    default:
        assert(0);
    }
    lsx_move_x(pwd, pwx);
}

#define BIT_XSELECT(dest, arg1, arg2, df) \
            UNSIGNED((arg1 & (~dest)) | (arg2 & dest), df)

#define BIT_XMOVE_IF_ZERO(dest, arg1, arg2, df) \
            UNSIGNED((dest & arg2) | (arg1 & (~arg2)), df)

#define BIT_XMOVE_IF_NOT_ZERO(dest, arg1, arg2, df) \
            UNSIGNED(((dest & (~arg2)) | (arg1 & arg2)), df)

#define LSX_XFN_IMM8(FUNC, DEST, OPERATION)                           \
void helper_lsx_ ## FUNC(CPULoongArchState *env, uint32_t wd, uint32_t ws, \
        uint32_t i8)                                                  \
{                                                                     \
    wr_t *pwd = &(env->fpr[wd].wr);                        \
    wr_t *pws = &(env->fpr[ws].wr);                        \
    uint32_t i;                                                       \
    for (i = 0; i < DF_ELEMENTS(DF_BYTE); i++) {                      \
        DEST = OPERATION;                                             \
    }                                                                 \
}

LSX_XFN_IMM8(xvbitseli_b, pwd->b[i],
        BIT_XSELECT(pwd->b[i], pws->b[i], i8, DF_BYTE))

LSX_XFN_IMM8(xvbitmvzi_b, pwd->b[i],
        BIT_XMOVE_IF_ZERO(pwd->b[i], pws->b[i], i8, DF_BYTE))

LSX_XFN_IMM8(xvbitmvnzi_b, pwd->b[i],
        BIT_XMOVE_IF_NOT_ZERO(pwd->b[i], pws->b[i], i8, DF_BYTE))

LSX_XFN_IMM8(xvandi_b, pwd->b[i], pws->b[i] & i8)
LSX_XFN_IMM8(xvori_b, pwd->b[i], pws->b[i] | i8)
LSX_XFN_IMM8(xvxori_b, pwd->b[i], pws->b[i] ^ i8)
LSX_XFN_IMM8(xvnori_b, pwd->b[i], ~(pws->b[i] | i8))

#undef BIT_XSELECT
#undef BIT_XMOVE_IF_ZERO
#undef BIT_XMOVE_IF_NOT_ZERO
#undef LSX_XFN_IMM8

int64_t XOR_multp_half(__int128 in1, __int128 in2, int n_bits, int select);
__int128 XOR_multp_all(__int128 in1, __int128 in2, int n_bits);
int64_t XOR_multp_half(__int128 in1, __int128 in2, int n_bits, int select){
    int i;
    __int128 temp = 0;
    __int128 bitmask = 1;
    __int128 mask = 0;
    for(i = 0; i < n_bits; i++){
        if(in1 & bitmask){
             temp = temp ^ (in2 << i);
        }
        bitmask = bitmask << 1;
        mask = mask | bitmask;
    }
    int64_t result;
    if(select == 1){
        result = (int64_t)((temp & (mask << n_bits)) >> n_bits);
    }
    else {
        result = (int64_t)(temp & mask);
    }
    return result;
}

__int128 XOR_multp_all(__int128 in1, __int128 in2, int n_bits){
    int i;
    __int128 temp = 0;
    __int128 bitmask = 1;
    __int128 mask = 0;
    for(i = 0; i < n_bits; i++){
        if(in1 & bitmask){
             temp = temp ^ (in2 << i);
        }
        bitmask = bitmask << 1;
        mask = mask | bitmask;
    }
    return temp & (mask | mask << n_bits);
}

void helper_lsx_vpmul_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = (int32_t)XOR_multp_half((__int128)pws->w[i],(__int128)pwt->w[i],32,0);
    }
}

void helper_lsx_vpmul_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64; i++) {
        pwd->d[i] = (int64_t)XOR_multp_half((__int128)pws->d[i],(__int128)pwt->d[i],64,0);
    }
}

void helper_lsx_xvpmul_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/32; i++) {
        pwd->w[i] = (int32_t)XOR_multp_half((__int128)pws->w[i],(__int128)pwt->w[i],32,0);
    }
}

void helper_lsx_xvpmul_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/64; i++) {
        pwd->d[i] = (int64_t)XOR_multp_half((__int128)pws->d[i],(__int128)pwt->d[i],64,0);
    }
}

void helper_lsx_vpmuh_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = (int32_t)XOR_multp_half((__int128)pws->w[i],(__int128)pwt->w[i],32,1);
    }
}

void helper_lsx_vpmuh_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64; i++) {
        pwd->d[i] = (int64_t)XOR_multp_half((__int128)pws->d[i],(__int128)pwt->d[i],64,1);
    }
}

void helper_lsx_xvpmuh_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/32; i++) {
        pwd->w[i] = (int32_t)XOR_multp_half((__int128)pws->w[i],(__int128)pwt->w[i],32,1);
    }
}

void helper_lsx_xvpmuh_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/64; i++) {
        pwd->d[i] = (int64_t)XOR_multp_half((__int128)pws->d[i],(__int128)pwt->d[i],64,1);
    }
}

void helper_lsx_vpmulacc_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = (int32_t)XOR_multp_half((__int128)pws->w[i],(__int128)pwt->w[i],32,0) ^ pwd->w[i];
    }
}

void helper_lsx_vpmulacc_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64; i++) {
        pwd->d[i] = (int64_t)XOR_multp_half((__int128)pws->d[i],(__int128)pwt->d[i],64,0) ^ pwd->d[i];
    }
}

void helper_lsx_xvpmulacc_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/32; i++) {
        pwd->w[i] = (int32_t)XOR_multp_half((__int128)pws->w[i],(__int128)pwt->w[i],32,0) ^ pwd->w[i];
    }
}

void helper_lsx_xvpmulacc_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/64; i++) {
        pwd->d[i] = (int64_t)XOR_multp_half((__int128)pws->d[i],(__int128)pwt->d[i],64,0) ^ pwd->d[i];
    }
}

void helper_lsx_vpmuhacc_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = (int32_t)XOR_multp_half((__int128)pws->w[i],(__int128)pwt->w[i],32,1) ^ pwd->w[i];
    }
}

void helper_lsx_vpmuhacc_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64; i++) {
        pwd->d[i] = (int64_t)XOR_multp_half((__int128)pws->d[i],(__int128)pwt->d[i],64,1) ^ pwd->d[i];
    }
}

void helper_lsx_xvpmuhacc_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/32; i++) {
        pwd->w[i] = (int32_t)XOR_multp_half((__int128)pws->w[i],(__int128)pwt->w[i],32,1) ^ pwd->w[i];
    }
}

void helper_lsx_xvpmuhacc_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/64; i++) {
        pwd->d[i] = (int64_t)XOR_multp_half((__int128)pws->d[i],(__int128)pwt->d[i],64,1) ^ pwd->d[i];
    }
}

void helper_lsx_vpmulwl_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = (int16_t)XOR_multp_all((__int128)pws->b[i],(__int128)pwt->b[i],8);
    }
}

void helper_lsx_vpmulwl_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = (int32_t)XOR_multp_all((__int128)pws->h[i],(__int128)pwt->h[i],16);
    }
}

void helper_lsx_vpmulwl_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64; i++) {
        pwd->d[i] = (int64_t)XOR_multp_all((__int128)pws->w[i],(__int128)pwt->w[i],32);
    }
}

void helper_lsx_vpmulwl_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/128; i++) {
        pwd->q[i] = XOR_multp_all((__int128)pws->d[i],(__int128)pwt->d[i],64);
    }
}

void helper_lsx_xvpmulwl_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/16; i++) {
        pwd->h[i] = (int16_t)XOR_multp_all((__int128)pws->b[i],(__int128)pwt->b[i],8);
    }
}

void helper_lsx_xvpmulwl_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/32; i++) {
        pwd->w[i] = (int32_t)XOR_multp_all((__int128)pws->h[i],(__int128)pwt->h[i],16);
    }
}

void helper_lsx_xvpmulwl_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/64; i++) {
        pwd->d[i] = (int64_t)XOR_multp_all((__int128)pws->w[i],(__int128)pwt->w[i],32);
    }
}

void helper_lsx_xvpmulwl_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/128; i++) {
        pwd->q[i] = XOR_multp_all((__int128)pws->d[i],(__int128)pwt->d[i],64);
    }
}

void helper_lsx_vpmulwh_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = (int16_t)XOR_multp_all((__int128)pws->b[i+8],(__int128)pwt->b[i+8],8);
    }
}

void helper_lsx_vpmulwh_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = (int32_t)XOR_multp_all((__int128)pws->h[i+4],(__int128)pwt->h[i+4],16);
    }
}

void helper_lsx_vpmulwh_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64; i++) {
        pwd->d[i] = (int64_t)XOR_multp_all((__int128)pws->w[i+2],(__int128)pwt->w[i+2],32);
    }
}

void helper_lsx_vpmulwh_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/128; i++) {
        pwd->q[i] = XOR_multp_all((__int128)pws->d[i+1],(__int128)pwt->d[i+1],64);
    }
}

void helper_lsx_xvpmulwh_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/16; i++) {
        pwd->h[i] = (int16_t)XOR_multp_all((__int128)pws->b[i+8],(__int128)pwt->b[i+8],8);
    }
}

void helper_lsx_xvpmulwh_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/32; i++) {
        pwd->w[i] = (int32_t)XOR_multp_all((__int128)pws->h[i+4],(__int128)pwt->h[i+4],16);
    }
}

void helper_lsx_xvpmulwh_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/64; i++) {
        pwd->d[i] = (int64_t)XOR_multp_all((__int128)pws->w[i+2],(__int128)pwt->w[i+2],32);
    }
}

void helper_lsx_xvpmulwh_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/128; i++) {
        pwd->q[i] = XOR_multp_all((__int128)pws->d[i+1],(__int128)pwt->d[i+1],64);
    }
}

void helper_lsx_vpmaddwl_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = (int16_t)XOR_multp_all((__int128)pws->b[i],(__int128)pwt->b[i],8 ^ pwd->h[i]);
    }
}

void helper_lsx_vpmaddwl_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = (int32_t)XOR_multp_all((__int128)pws->h[i],(__int128)pwt->h[i],16) ^ pwd->w[i];
    }
}

void helper_lsx_vpmaddwl_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64; i++) {
        pwd->d[i] = (int64_t)XOR_multp_all((__int128)pws->w[i],(__int128)pwt->w[i],32) ^ pwd->d[i];
    }
}

void helper_lsx_vpmaddwl_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/128; i++) {
        pwd->q[i] = XOR_multp_all((__int128)pws->d[i],(__int128)pwt->d[i],64) ^ pwd->q[i];
    }
}

void helper_lsx_xvpmaddwl_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/16; i++) {
        pwd->h[i] = (int16_t)XOR_multp_all((__int128)pws->b[i],(__int128)pwt->b[i],8 ^ pwd->h[i]);
    }
}

void helper_lsx_xvpmaddwl_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/32; i++) {
        pwd->w[i] = (int32_t)XOR_multp_all((__int128)pws->h[i],(__int128)pwt->h[i],16) ^ pwd->w[i];
    }
}

void helper_lsx_xvpmaddwl_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/64; i++) {
        pwd->d[i] = (int64_t)XOR_multp_all((__int128)pws->w[i],(__int128)pwt->w[i],32) ^ pwd->d[i];
    }
}

void helper_lsx_xvpmaddwl_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/128; i++) {
        pwd->q[i] = XOR_multp_all((__int128)pws->d[i],(__int128)pwt->d[i],64) ^ pwd->q[i];
    }
}

void helper_lsx_vpmaddwh_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/16; i++) {
        pwd->h[i] = (int16_t)XOR_multp_all((__int128)pws->b[i+8],(__int128)pwt->b[i+8],8) ^ pwd->h[i];
    }
}

void helper_lsx_vpmaddwh_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = (int32_t)XOR_multp_all((__int128)pws->h[i+4],(__int128)pwt->h[i+4],16) ^ pwd->w[i];
    }
}

void helper_lsx_vpmaddwh_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64; i++) {
        pwd->d[i] = (int64_t)XOR_multp_all((__int128)pws->w[i+2],(__int128)pwt->w[i+2],32) ^ pwd->d[i];
    }
}

void helper_lsx_vpmaddwh_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/128; i++) {
        pwd->q[i] = XOR_multp_all((__int128)pws->d[i+1],(__int128)pwt->d[i+1],64) ^ pwd->q[i];
    }
}

void helper_lsx_xvpmaddwh_h_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/16; i++) {
        pwd->h[i] = (int16_t)XOR_multp_all((__int128)pws->b[i+8],(__int128)pwt->b[i+8],8) ^ pwd->h[i];
    }
}

void helper_lsx_xvpmaddwh_w_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/32; i++) {
        pwd->w[i] = (int32_t)XOR_multp_all((__int128)pws->h[i+4],(__int128)pwt->h[i+4],16) ^ pwd->w[i];
    }
}

void helper_lsx_xvpmaddwh_d_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/64; i++) {
        pwd->d[i] = (int64_t)XOR_multp_all((__int128)pws->w[i+2],(__int128)pwt->w[i+2],32) ^ pwd->d[i];
    }
}

void helper_lsx_xvpmaddwh_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/128; i++) {
        pwd->q[i] = XOR_multp_all((__int128)pws->d[i+1],(__int128)pwt->d[i+1],64) ^ pwd->q[i];
    }
}

void helper_lsx_vpdp2_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/128; i++) {
        pwd->q[i] = XOR_multp_all((__int128)pws->d[i],(__int128)pwt->d[i],64) ^ XOR_multp_all((__int128)pws->d[i+1],(__int128)pwt->d[i+1],64);
    }
}

void helper_lsx_vpdp2add_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/128; i++) {
        pwd->q[i] = XOR_multp_all((__int128)pws->d[i],(__int128)pwt->d[i],64) ^ XOR_multp_all((__int128)pws->d[i+1],(__int128)pwt->d[i+1],64) ^ pwd->q[i];
    }
}

void helper_lsx_xvpdp2_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/128; i++) {
        pwd->q[i] = XOR_multp_all((__int128)pws->d[i],(__int128)pwt->d[i],64) ^ XOR_multp_all((__int128)pws->d[i+1],(__int128)pwt->d[i+1],64);
    }
}

void helper_lsx_xvpdp2add_q_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/128; i++) {
        pwd->q[i] = XOR_multp_all((__int128)pws->d[i],(__int128)pwt->d[i],64) ^ XOR_multp_all((__int128)pws->d[i+1],(__int128)pwt->d[i+1],64) ^ pwd->q[i];
    }
}

void helper_lsx_vcdp4_re_d_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->d[0] = (int64_t)pws->h[0] * (int64_t)pwt->h[0]
              - (int64_t)pws->h[1] * (int64_t)pwt->h[1]
              + (int64_t)pws->h[2] * (int64_t)pwt->h[2]
              - (int64_t)pws->h[3] * (int64_t)pwt->h[3]
              + (int64_t)pws->h[4] * (int64_t)pwt->h[4]
              - (int64_t)pws->h[5] * (int64_t)pwt->h[5]
              + (int64_t)pws->h[6] * (int64_t)pwt->h[6]
              - (int64_t)pws->h[7] * (int64_t)pwt->h[7];
    pwd->d[1] = 0;
}

void helper_lsx_vcdp4_im_d_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->d[0] = 0;
    pwd->d[1] = (int64_t)pws->h[0] * (int64_t)pwt->h[0]
              + (int64_t)pws->h[1] * (int64_t)pwt->h[1]
              + (int64_t)pws->h[2] * (int64_t)pwt->h[2]
              + (int64_t)pws->h[3] * (int64_t)pwt->h[3]
              + (int64_t)pws->h[4] * (int64_t)pwt->h[4]
              + (int64_t)pws->h[5] * (int64_t)pwt->h[5]
              + (int64_t)pws->h[6] * (int64_t)pwt->h[6]
              + (int64_t)pws->h[7] * (int64_t)pwt->h[7];
}

void helper_lsx_vcdp4add_re_d_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->d[0]+= (int64_t)pws->h[0] * (int64_t)pwt->h[0]
              - (int64_t)pws->h[1] * (int64_t)pwt->h[1]
              + (int64_t)pws->h[2] * (int64_t)pwt->h[2]
              - (int64_t)pws->h[3] * (int64_t)pwt->h[3]
              + (int64_t)pws->h[4] * (int64_t)pwt->h[4]
              - (int64_t)pws->h[5] * (int64_t)pwt->h[5]
              + (int64_t)pws->h[6] * (int64_t)pwt->h[6]
              - (int64_t)pws->h[7] * (int64_t)pwt->h[7];
    pwd->d[1]+= 0;
}

void helper_lsx_vcdp4add_im_d_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->d[0]+= 0;
    pwd->d[1]+= (int64_t)pws->h[0] * (int64_t)pwt->h[0]
              + (int64_t)pws->h[1] * (int64_t)pwt->h[1]
              + (int64_t)pws->h[2] * (int64_t)pwt->h[2]
              + (int64_t)pws->h[3] * (int64_t)pwt->h[3]
              + (int64_t)pws->h[4] * (int64_t)pwt->h[4]
              + (int64_t)pws->h[5] * (int64_t)pwt->h[5]
              + (int64_t)pws->h[6] * (int64_t)pwt->h[6]
              + (int64_t)pws->h[7] * (int64_t)pwt->h[7];
}

void helper_lsx_xvcdp4_re_d_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->d[0] = (int64_t)pws->h[ 0] * (int64_t)pwt->h[ 0]
              - (int64_t)pws->h[ 1] * (int64_t)pwt->h[ 1]
              + (int64_t)pws->h[ 2] * (int64_t)pwt->h[ 2]
              - (int64_t)pws->h[ 3] * (int64_t)pwt->h[ 3]
              + (int64_t)pws->h[ 4] * (int64_t)pwt->h[ 4]
              - (int64_t)pws->h[ 5] * (int64_t)pwt->h[ 5]
              + (int64_t)pws->h[ 6] * (int64_t)pwt->h[ 6]
              - (int64_t)pws->h[ 7] * (int64_t)pwt->h[ 7]
              + (int64_t)pws->h[ 8] * (int64_t)pwt->h[ 8]
              - (int64_t)pws->h[ 9] * (int64_t)pwt->h[ 9]
              + (int64_t)pws->h[10] * (int64_t)pwt->h[10]
              - (int64_t)pws->h[11] * (int64_t)pwt->h[11]
              + (int64_t)pws->h[12] * (int64_t)pwt->h[12]
              - (int64_t)pws->h[13] * (int64_t)pwt->h[13]
              + (int64_t)pws->h[14] * (int64_t)pwt->h[14]
              - (int64_t)pws->h[15] * (int64_t)pwt->h[15];
    pwd->d[1] = 0;
    pwd->d[2] = 0;
    pwd->d[3] = 0;
}

void helper_lsx_xvcdp4_im_d_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->d[0] = 0;
    pwd->d[1] = (int64_t)pws->h[ 0] * (int64_t)pwt->h[ 0]
              + (int64_t)pws->h[ 1] * (int64_t)pwt->h[ 1]
              + (int64_t)pws->h[ 2] * (int64_t)pwt->h[ 2]
              + (int64_t)pws->h[ 3] * (int64_t)pwt->h[ 3]
              + (int64_t)pws->h[ 4] * (int64_t)pwt->h[ 4]
              + (int64_t)pws->h[ 5] * (int64_t)pwt->h[ 5]
              + (int64_t)pws->h[ 6] * (int64_t)pwt->h[ 6]
              + (int64_t)pws->h[ 7] * (int64_t)pwt->h[ 7]
              + (int64_t)pws->h[ 8] * (int64_t)pwt->h[ 8]
              + (int64_t)pws->h[ 9] * (int64_t)pwt->h[ 9]
              + (int64_t)pws->h[10] * (int64_t)pwt->h[10]
              + (int64_t)pws->h[11] * (int64_t)pwt->h[11]
              + (int64_t)pws->h[12] * (int64_t)pwt->h[12]
              + (int64_t)pws->h[13] * (int64_t)pwt->h[13]
              + (int64_t)pws->h[14] * (int64_t)pwt->h[14]
              + (int64_t)pws->h[15] * (int64_t)pwt->h[15];
    pwd->d[2] = 0;
    pwd->d[3] = 0;
}

void helper_lsx_xvcdp4add_re_d_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->d[0]+= (int64_t)pws->h[ 0] * (int64_t)pwt->h[ 0]
              - (int64_t)pws->h[ 1] * (int64_t)pwt->h[ 1]
              + (int64_t)pws->h[ 2] * (int64_t)pwt->h[ 2]
              - (int64_t)pws->h[ 3] * (int64_t)pwt->h[ 3]
              + (int64_t)pws->h[ 4] * (int64_t)pwt->h[ 4]
              - (int64_t)pws->h[ 5] * (int64_t)pwt->h[ 5]
              + (int64_t)pws->h[ 6] * (int64_t)pwt->h[ 6]
              - (int64_t)pws->h[ 7] * (int64_t)pwt->h[ 7]
              + (int64_t)pws->h[ 8] * (int64_t)pwt->h[ 8]
              - (int64_t)pws->h[ 9] * (int64_t)pwt->h[ 9]
              + (int64_t)pws->h[10] * (int64_t)pwt->h[10]
              - (int64_t)pws->h[11] * (int64_t)pwt->h[11]
              + (int64_t)pws->h[12] * (int64_t)pwt->h[12]
              - (int64_t)pws->h[13] * (int64_t)pwt->h[13]
              + (int64_t)pws->h[14] * (int64_t)pwt->h[14]
              - (int64_t)pws->h[15] * (int64_t)pwt->h[15];
    pwd->d[1]+= 0;
    pwd->d[2]+= 0;
    pwd->d[3]+= 0;
}

void helper_lsx_xvcdp4add_im_d_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->d[0]+= 0;
    pwd->d[1]+= (int64_t)pws->h[ 0] * (int64_t)pwt->h[ 0]
              + (int64_t)pws->h[ 1] * (int64_t)pwt->h[ 1]
              + (int64_t)pws->h[ 2] * (int64_t)pwt->h[ 2]
              + (int64_t)pws->h[ 3] * (int64_t)pwt->h[ 3]
              + (int64_t)pws->h[ 4] * (int64_t)pwt->h[ 4]
              + (int64_t)pws->h[ 5] * (int64_t)pwt->h[ 5]
              + (int64_t)pws->h[ 6] * (int64_t)pwt->h[ 6]
              + (int64_t)pws->h[ 7] * (int64_t)pwt->h[ 7]
              + (int64_t)pws->h[ 8] * (int64_t)pwt->h[ 8]
              + (int64_t)pws->h[ 9] * (int64_t)pwt->h[ 9]
              + (int64_t)pws->h[10] * (int64_t)pwt->h[10]
              + (int64_t)pws->h[11] * (int64_t)pwt->h[11]
              + (int64_t)pws->h[12] * (int64_t)pwt->h[12]
              + (int64_t)pws->h[13] * (int64_t)pwt->h[13]
              + (int64_t)pws->h[14] * (int64_t)pwt->h[14]
              + (int64_t)pws->h[15] * (int64_t)pwt->h[15];
    pwd->d[2]+= 0;
    pwd->d[3]+= 0;
}

void helper_lsx_vcdp2_re_q_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0] = (__int128)pws->w[0] * (__int128)pwt->w[0]
              - (__int128)pws->w[1] * (__int128)pwt->w[1]
              + (__int128)pws->w[2] * (__int128)pwt->w[2]
              - (__int128)pws->w[3] * (__int128)pwt->w[3];
}

void helper_lsx_vcdp2_im_q_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0] = (__int128)pws->w[0] * (__int128)pwt->w[0]
              + (__int128)pws->w[1] * (__int128)pwt->w[1]
              + (__int128)pws->w[2] * (__int128)pwt->w[2]
              + (__int128)pws->w[3] * (__int128)pwt->w[3];
}

void helper_lsx_vcdp2add_re_q_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0]+= (__int128)pws->w[0] * (__int128)pwt->w[0]
              - (__int128)pws->w[1] * (__int128)pwt->w[1]
              + (__int128)pws->w[2] * (__int128)pwt->w[2]
              - (__int128)pws->w[3] * (__int128)pwt->w[3];
}

void helper_lsx_vcdp2add_im_q_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0]+= (__int128)pws->w[0] * (__int128)pwt->w[0]
              + (__int128)pws->w[1] * (__int128)pwt->w[1]
              + (__int128)pws->w[2] * (__int128)pwt->w[2]
              + (__int128)pws->w[3] * (__int128)pwt->w[3];
}

void helper_lsx_xvcdp2_re_q_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0] = (__int128)pws->w[0] * (__int128)pwt->w[0]
              - (__int128)pws->w[1] * (__int128)pwt->w[1]
              + (__int128)pws->w[2] * (__int128)pwt->w[2]
              - (__int128)pws->w[3] * (__int128)pwt->w[3]
              + (__int128)pws->w[4] * (__int128)pwt->w[4]
              - (__int128)pws->w[5] * (__int128)pwt->w[5]
              + (__int128)pws->w[6] * (__int128)pwt->w[6]
              - (__int128)pws->w[7] * (__int128)pwt->w[7];
    pwd->q[1] = 0;
}

void helper_lsx_xvcdp2_im_q_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0] = 0;
    pwd->q[1] = (__int128)pws->w[0] * (__int128)pwt->w[0]
              + (__int128)pws->w[1] * (__int128)pwt->w[1]
              + (__int128)pws->w[2] * (__int128)pwt->w[2]
              + (__int128)pws->w[3] * (__int128)pwt->w[3]
              + (__int128)pws->w[4] * (__int128)pwt->w[4]
              + (__int128)pws->w[5] * (__int128)pwt->w[5]
              + (__int128)pws->w[6] * (__int128)pwt->w[6]
              + (__int128)pws->w[7] * (__int128)pwt->w[7];
}

void helper_lsx_xvcdp2add_re_q_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0]+= (__int128)pws->w[0] * (__int128)pwt->w[0]
              - (__int128)pws->w[1] * (__int128)pwt->w[1]
              + (__int128)pws->w[2] * (__int128)pwt->w[2]
              - (__int128)pws->w[3] * (__int128)pwt->w[3]
              + (__int128)pws->w[4] * (__int128)pwt->w[4]
              - (__int128)pws->w[5] * (__int128)pwt->w[5]
              + (__int128)pws->w[6] * (__int128)pwt->w[6]
              - (__int128)pws->w[7] * (__int128)pwt->w[7];
    pwd->q[1]+= 0;
}

void helper_lsx_xvcdp2add_im_q_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    pwd->q[0]+= 0;
    pwd->q[1]+= (__int128)pws->w[0] * (__int128)pwt->w[0]
              + (__int128)pws->w[1] * (__int128)pwt->w[1]
              + (__int128)pws->w[2] * (__int128)pwt->w[2]
              + (__int128)pws->w[3] * (__int128)pwt->w[3]
              + (__int128)pws->w[4] * (__int128)pwt->w[4]
              + (__int128)pws->w[5] * (__int128)pwt->w[5]
              + (__int128)pws->w[6] * (__int128)pwt->w[6]
              + (__int128)pws->w[7] * (__int128)pwt->w[7];
}

void helper_lsx_vsignsel_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/32; i++) {
        pwd->w[i] = (pwd->w[i] & 0x80000000) ? pwt->w[i] : pws->w[i];
    }
}

void helper_lsx_vsignsel_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 128/64; i++) {
        pwd->d[i] = (pwd->d[i] & 0x8000000000000000) ? pwt->d[i] : pws->d[i];
    }
}

void helper_lsx_vrandsigni_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i;
    int8_t r = 0x80;
    int m;
    for (i = 0; i < 128/8; i++) {
        r = r & pws->b[i];
    }
    m = wt & 0xf;
    pwd->b[m] = (r & 0x80) ? 0xff : 0x00;
}

void helper_lsx_vrandsigni_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i;
    int16_t r = 0x8000;
    int m;
    for (i = 0; i < 128/16; i++) {
        r = r & pws->h[i];
    }
    m = wt & 0x7;
    pwd->h[m] = (r & 0x8000) ? 0xffff : 0x0000;
}

void helper_lsx_vrorsigni_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i;
    int8_t r = 0x0;
    int m;
    for (i = 0; i < 128/8; i++) {
        r = r | pws->b[i];
    }
    m = wt & 0xf;
    pwd->b[m] = (r & 0x80) ? 0xff : 0x00;
}

void helper_lsx_vrorsigni_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i;
    int16_t r = 0x0;
    int m;
    for (i = 0; i < 128/16; i++) {
        r = r | pws->h[i];
    }
    m = wt & 0x7;
    pwd->h[m] = (r & 0x8000) ? 0xffff : 0x0000;
}

void helper_lsx_vfrstpi_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i;
    int m;
    for (i = 0; i < 128/8; i++) {
        if(pws->b[i] & 0x80)
            break;
    }
    m = wt & 0xf;
    pwd->b[m] = (int8_t)i;
}

void helper_lsx_vfrstpi_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i;
    int m;
    for (i = 0; i < 128/16; i++) {
        if(pws->h[i] & 0x8000)
            break;
    }
    m = wt & 0x7;
    pwd->h[m] = (int16_t)i;
}

void helper_lsx_vclrstri_v(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i;
    int m;
    m = wt & 0xf;
    for (i = 0; i < 128/8; i++) {
        if(i<=m)
            pwd->b[i] = pws->b[i];
        else
            pwd->b[i] = 0x00;
    }
}

void helper_lsx_vmepatmsk_v(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);

    int i,j;
    int mode = ws;
    for (i = 0; i < 4; i++) {
        for (j = 0; j < 4; j++){
            switch(mode){
                case 0 :
                    pwd->b[4*i+j] = wt + j;
                    break;
                case 1 :
                    pwd->b[4*i+j] = wt + i + j;
                    break;
                case 2 :
                    pwd->b[4*i+j] = wt + i + j + 4;
                    break;
                case 3 :
                    pwd->b[4*i+j] = wt + (4 * i) + j;
                    break;
                default : assert(0);
            }
        }
    }
}

void helper_lsx_vfrstm_b(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

	wr_t tmp;
    tmp.q[0] = 0;
    int i;
    for (i = 0; i < 128/8; i++) {
        if(pws->b[i] & 0x80)
            break;
    }
    tmp.b[i] = 0xff;

	pwd->q[0] = tmp.q[0];
	pwd->q[1] = tmp.q[1];
}

void helper_lsx_vfrstm_h(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

	wr_t tmp;
    tmp.q[0] = 0;
    int i;
    for (i = 0; i < 128/16; i++) {
        if(pws->h[i] & 0x8000)
            break;
    }
    tmp.h[i] = 0xffff;

	pwd->q[0] = tmp.q[0];
	pwd->q[1] = tmp.q[1];
}

void helper_lsx_vextl_w_b(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/32; i++) {
        tmp.b[4*i  ] = pws->b[i];
        tmp.b[4*i+1] = (pws->b[i] & 0x80) ? 0xff : 0x00;
        tmp.b[4*i+2] = (pws->b[i] & 0x80) ? 0xff : 0x00;
        tmp.b[4*i+3] = (pws->b[i] & 0x80) ? 0xff : 0x00;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vextl_d_b(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/64; i++) {
        tmp.b[8*i  ] = pws->b[i];
        tmp.b[8*i+1] = (pws->b[i] & 0x80) ? 0xff : 0x00;
        tmp.b[8*i+2] = (pws->b[i] & 0x80) ? 0xff : 0x00;
        tmp.b[8*i+3] = (pws->b[i] & 0x80) ? 0xff : 0x00;
        tmp.b[8*i+4] = (pws->b[i] & 0x80) ? 0xff : 0x00;
        tmp.b[8*i+5] = (pws->b[i] & 0x80) ? 0xff : 0x00;
        tmp.b[8*i+6] = (pws->b[i] & 0x80) ? 0xff : 0x00;
        tmp.b[8*i+7] = (pws->b[i] & 0x80) ? 0xff : 0x00;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vextl_d_h(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/64; i++) {
        tmp.h[4*i  ] =  pws->h[i];
        tmp.h[4*i+1] = (pws->h[i] & 0x8000) ? 0xffff : 0x0000;
        tmp.h[4*i+2] = (pws->h[i] & 0x8000) ? 0xffff : 0x0000;
        tmp.h[4*i+3] = (pws->h[i] & 0x8000) ? 0xffff : 0x0000;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vextl_w_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/32; i++) {
        tmp.b[4*i  ] = pws->b[i];
        tmp.b[4*i+1] = 0x00;
        tmp.b[4*i+2] = 0x00;
        tmp.b[4*i+3] = 0x00;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vextl_d_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/64; i++) {
        tmp.b[8*i  ] = pws->b[i];
        tmp.b[8*i+1] = 0x00;
        tmp.b[8*i+2] = 0x00;
        tmp.b[8*i+3] = 0x00;
        tmp.b[8*i+4] = 0x00;
        tmp.b[8*i+5] = 0x00;
        tmp.b[8*i+6] = 0x00;
        tmp.b[8*i+7] = 0x00;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vextl_d_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/64; i++) {
        tmp.h[4*i  ] = pws->h[i];
        tmp.h[4*i+1] = 0x0000;
        tmp.h[4*i+2] = 0x0000;
        tmp.h[4*i+3] = 0x0000;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_vhadd8_d_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i;
    for (i = 0; i < 128/64; i++) {
        pwd->d[i] = (int64_t)(uint8_t)pws->b[8 * i    ]
                  + (int64_t)(uint8_t)pws->b[8 * i + 1]
                  + (int64_t)(uint8_t)pws->b[8 * i + 2]
                  + (int64_t)(uint8_t)pws->b[8 * i + 3]
                  + (int64_t)(uint8_t)pws->b[8 * i + 4]
                  + (int64_t)(uint8_t)pws->b[8 * i + 5]
                  + (int64_t)(uint8_t)pws->b[8 * i + 6]
                  + (int64_t)(uint8_t)pws->b[8 * i + 7];
    }
}

void helper_lsx_vhminpos_w_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i;
    uint16_t ss,tt;
    for (i = 0; i < 128/32; i++) {
        ss = (uint16_t)pws->h[2*i+1];
        tt = (uint16_t)pws->h[2*i];
        pwd->h[2*i  ] = (ss<tt) ? ss : tt;
        pwd->h[2*i+1] = (ss<tt) ? 2*i+1 : 2*i;
    }
}

void helper_lsx_vhminpos_d_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i;
    uint16_t ss,tt,h1,h3;
    for (i = 0; i < 128/64; i++) {
        ss = (uint16_t)pws->h[4*i+2];
        tt = (uint16_t)pws->h[4*i];
        h1 = (uint16_t)pws->h[4*i+1];
        h3 = (uint16_t)pws->h[4*i+3];
        pwd->h[4*i  ] = (ss<tt) ? ss : tt;
        pwd->h[4*i+1] = (ss<tt) ? h3 : h1;
        pwd->w[2*i+1] = 0;
    }
}

void helper_lsx_vhminpos_q_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

	wr_t tmp;
    tmp.q[0] = 0;
    uint16_t ss,tt,h1,h5;
    ss = (uint16_t)pws->h[4];
    tt = (uint16_t)pws->h[0];
    h1 = (uint16_t)pws->h[1];
    h5 = (uint16_t)pws->h[5];
    tmp.h[0] = (ss<tt) ? ss : tt;
    tmp.h[1] = (ss<tt) ? h5 : h1;

	pwd->q[0] = tmp.q[0];
}

void helper_lsx_vclrtail_b(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i;
    int flag = 0;
    for (i = 0; i < 128/8; i++) {
        if(pws->b[i]==0)
            flag = 1;
        if(flag)
            pwd->b[i] = 0x00;
        else
            pwd->b[i] = pws->b[i];
    }
}

void helper_lsx_vclrtail_h(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i;
    int flag = 0;
    for (i = 0; i < 128/16; i++) {
        if(pws->h[i]==0)
            flag = 1;
        if(flag)
            pwd->h[i] = 0x0000;
        else
            pwd->h[i] = pws->h[i];
    }
}

void helper_lsx_vextrins_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui8)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int ins,extr;
    ins  = (ui8 >> 4) & 0x1;
    extr =  ui8       & 0x1;
    pwd->d[ins] = pws->d[extr];
}

void helper_lsx_vextrins_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui8)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int ins,extr;
    ins  = (ui8 >> 4) & 0x3;
    extr =  ui8       & 0x3;
    pwd->w[ins] = pws->w[extr];
}

void helper_lsx_vextrins_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui8)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int ins,extr;
    ins  = (ui8 >> 4) & 0x7;
    extr =  ui8       & 0x7;
    pwd->h[ins] = pws->h[extr];
}

void helper_lsx_vextrins_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui8)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int ins,extr;
    ins  = (ui8 >> 4) & 0xf;
    extr =  ui8       & 0xf;
    pwd->b[ins] = pws->b[extr];
}

void helper_lsx_vshufi1_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui8)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i,k,l;
    k = ui8;
    for(i=0;i<4;++i){
        l = k & 0x3;
        pwd->b[i] = pws->b[l];
        k = k >> 2;
    }
}

void helper_lsx_vshufi2_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui8)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i,k,l;
    k = ui8;
    for(i=0;i<4;++i){
        l = k & 0x3;
        pwd->b[i+4] = pws->b[l+4];
        k = k >> 2;
    }
}

void helper_lsx_vshufi3_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui8)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i,k,l;
    k = ui8;
    for(i=0;i<4;++i){
        l = k & 0x3;
        pwd->b[i+8] = pws->b[l+8];
        k = k >> 2;
    }
}

void helper_lsx_vshufi4_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui8)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i,k,l;
    k = ui8;
    for(i=0;i<4;++i){
        l = k & 0x3;
        pwd->b[i+12] = pws->b[l+12];
        k = k >> 2;
    }
}

void helper_lsx_vshufi1_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui8)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i,k,l;
    k = ui8;
    for(i=0;i<4;++i){
        l = k & 0x3;
        pwd->h[i] = pws->h[l];
        k = k >> 2;
    }
}

void helper_lsx_vshufi2_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui8)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i,k,l;
    k = ui8;
    for(i=0;i<4;++i){
        l = k & 0x3;
        pwd->h[i+4] = pws->h[l+4];
        k = k >> 2;
    }
}

void helper_lsx_vseli_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui8)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i,k;
    k = ui8;
    for(i=0;i<128/16;++i){
        if(k & 0x1){
            pwd->h[i] = pws->h[i];
        }
        k = k >> 1;
    }
}

void helper_lsx_vseli_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui8)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i,k;
    k = ui8;
    for(i=0;i<128/32;++i){
        if(k & 0x1){
            pwd->w[i] = pws->w[i];
        }
        k = k >> 1;
    }
}

void helper_lsx_vseli_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui8)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i,k;
    k = ui8;
    for(i=0;i<128/64;++i){
        if(k & 0x1){
            pwd->d[i] = pws->d[i];
        }
        k = k >> 1;
    }
}

void helper_lsx_xvsignsel_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/32; i++) {
        pwd->w[i] = (pwd->w[i] & 0x80000000) ? pwt->w[i] : pws->w[i];
    }
}

void helper_lsx_xvsignsel_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    wr_t *pwt = &(env->fpr[wt].wr);

    int i;
    for (i = 0; i < 256/64; i++) {
        pwd->d[i] = (pwd->d[i] & 0x8000000000000000) ? pwt->d[i] : pws->d[i];
    }
}

void helper_lsx_xvrandsigni_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i;
    int8_t r = 0x80;
    int m;
    for (i = 0; i < 256/8; i++) {
        r = r & pws->b[i];
    }
    m = wt & 0x1f;
    pwd->b[m] = (r & 0x80) ? 0xff : 0x00;
}

void helper_lsx_xvrandsigni_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i;
    int16_t r = 0x8000;
    int m;
    for (i = 0; i < 256/16; i++) {
        r = r & pws->h[i];
    }
    m = wt & 0xf;
    pwd->h[m] = (r & 0x8000) ? 0xffff : 0x0000;
}

void helper_lsx_xvrorsigni_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i;
    int8_t r1 = 0x0;
    int8_t r2 = 0x0;
    int m;
    for (i = 0; i < 128/8; i++) {
        r1 = r1 | pws->b[i];
        r2 = r2 | pws->b[i+16];
    }
    m = wt & 0xf;
    pwd->b[m] = (r1 & 0x80) ? 0xff : 0x00;
    pwd->b[m+16] = (r2 & 0x80) ? 0xff : 0x00;
}

void helper_lsx_xvrorsigni_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i;
    int16_t r1 = 0x0;
    int16_t r2 = 0x0;
    int m;
    for (i = 0; i < 128/16; i++) {
        r1 = r1 | pws->h[i];
        r2 = r2 | pws->h[i+8];
    }
    m = wt & 0x7;
    pwd->h[m] = (r1 & 0x8000) ? 0xffff : 0x0000;
    pwd->h[m+8] = (r2 & 0x8000) ? 0xffff : 0x0000;
}

void helper_lsx_xvfrstpi_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i;
    int m;
    for (i = 0; i < 256/8; i++) {
        if(pws->b[i] & 0x80)
            break;
    }
    m = wt & 0x1f;
    pwd->b[m] = (int8_t)i;
}

void helper_lsx_xvfrstpi_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i;
    int m;
    for (i = 0; i < 256/16; i++) {
        if(pws->h[i] & 0x8000)
            break;
    }
    m = wt & 0xf;
    pwd->h[m] = (int16_t)i;
}

void helper_lsx_xvclrstri_v(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i;
    int m;
    m = wt & 0x1f;
    for (i = 0; i < 256/8; i++) {
        if(i<=m)
            pwd->b[i] = pws->b[i];
        else
            pwd->b[i] = 0x00;
    }
}

void helper_lsx_xvmepatmsk_v(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t wt)
{
    wr_t *pwd = &(env->fpr[wd].wr);

    int i,j;
    int mode = ws;
    for (i = 0; i < 4; i++) {
        for (j = 0; j < 4; j++){
            switch(mode){
                case 0 :
                    pwd->b[4*i+j] = wt + j;
                    break;
                case 1 :
                    pwd->b[4*i+j] = wt + i + j;
                    break;
                case 2 :
                    pwd->b[4*i+j] = wt + i + j + 4;
                    break;
                case 3 :
                    pwd->b[4*i+j] = wt + (4 * i) + j;
                    break;
                default : assert(0);
            }
        }
    }
}

void helper_lsx_xvfrstm_b(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

	wr_t tmp;
    tmp.q[0] = 0;
    tmp.q[1] = 0;
    int i;
    for (i = 0; i < 128/8; i++) {
        if(pws->b[i] & 0x80)
            break;
    }
    tmp.b[i] = 0xff;

    for (i = 0; i < 128/8; i++) {
        if(pws->b[i+16] & 0x80)
            break;
    }
    tmp.b[i+16] = 0xff;

	pwd->q[0] = tmp.q[0];
	pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvfrstm_h(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

	wr_t tmp;
    tmp.q[0] = 0;
    tmp.q[1] = 0;
    int i;
    for (i = 0; i < 128/16; i++) {
        if(pws->h[i] & 0x8000)
            break;
    }
    tmp.h[i] = 0xffff;

    for (i = 0; i < 128/16; i++) {
        if(pws->h[i+8] & 0x8000)
            break;
    }
    tmp.h[i+8] = 0xffff;

	pwd->q[0] = tmp.q[0];
	pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvextl_w_b(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/32; i++) {
        tmp.b[4*i  ] = pws->b[i];
        tmp.b[4*i+1] = (pws->b[i] & 0x80) ? 0xff : 0x00;
        tmp.b[4*i+2] = (pws->b[i] & 0x80) ? 0xff : 0x00;
        tmp.b[4*i+3] = (pws->b[i] & 0x80) ? 0xff : 0x00;

        tmp.b[4*i+16  ] = pws->b[i+16];
        tmp.b[4*i+1+16] = (pws->b[i+16] & 0x80) ? 0xff : 0x00;
        tmp.b[4*i+2+16] = (pws->b[i+16] & 0x80) ? 0xff : 0x00;
        tmp.b[4*i+3+16] = (pws->b[i+16] & 0x80) ? 0xff : 0x00;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvextl_d_b(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/64; i++) {
        tmp.b[8*i  ] = pws->b[i];
        tmp.b[8*i+1] = (pws->b[i] & 0x80) ? 0xff : 0x00;
        tmp.b[8*i+2] = (pws->b[i] & 0x80) ? 0xff : 0x00;
        tmp.b[8*i+3] = (pws->b[i] & 0x80) ? 0xff : 0x00;
        tmp.b[8*i+4] = (pws->b[i] & 0x80) ? 0xff : 0x00;
        tmp.b[8*i+5] = (pws->b[i] & 0x80) ? 0xff : 0x00;
        tmp.b[8*i+6] = (pws->b[i] & 0x80) ? 0xff : 0x00;
        tmp.b[8*i+7] = (pws->b[i] & 0x80) ? 0xff : 0x00;

        tmp.b[8*i+16  ] = pws->b[i+16];
        tmp.b[8*i+1+16] = (pws->b[i+16] & 0x80) ? 0xff : 0x00;
        tmp.b[8*i+2+16] = (pws->b[i+16] & 0x80) ? 0xff : 0x00;
        tmp.b[8*i+3+16] = (pws->b[i+16] & 0x80) ? 0xff : 0x00;
        tmp.b[8*i+4+16] = (pws->b[i+16] & 0x80) ? 0xff : 0x00;
        tmp.b[8*i+5+16] = (pws->b[i+16] & 0x80) ? 0xff : 0x00;
        tmp.b[8*i+6+16] = (pws->b[i+16] & 0x80) ? 0xff : 0x00;
        tmp.b[8*i+7+16] = (pws->b[i+16] & 0x80) ? 0xff : 0x00;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvextl_d_h(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/64; i++) {
        tmp.h[4*i  ] =  pws->h[i];
        tmp.h[4*i+1] = (pws->h[i] & 0x8000) ? 0xffff : 0x0000;
        tmp.h[4*i+2] = (pws->h[i] & 0x8000) ? 0xffff : 0x0000;
        tmp.h[4*i+3] = (pws->h[i] & 0x8000) ? 0xffff : 0x0000;

        tmp.h[4*i+8  ] =  pws->h[i+8];
        tmp.h[4*i+1+8] = (pws->h[i+8] & 0x8000) ? 0xffff : 0x0000;
        tmp.h[4*i+2+8] = (pws->h[i+8] & 0x8000) ? 0xffff : 0x0000;
        tmp.h[4*i+3+8] = (pws->h[i+8] & 0x8000) ? 0xffff : 0x0000;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvextl_w_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/32; i++) {
        tmp.b[4*i  ] = pws->b[i];
        tmp.b[4*i+1] = 0x00;
        tmp.b[4*i+2] = 0x00;
        tmp.b[4*i+3] = 0x00;

        tmp.b[4*i+16  ] = pws->b[i+16];
        tmp.b[4*i+1+16] = 0x00;
        tmp.b[4*i+2+16] = 0x00;
        tmp.b[4*i+3+16] = 0x00;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvextl_d_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/64; i++) {
        tmp.b[8*i  ] = pws->b[i];
        tmp.b[8*i+1] = 0x00;
        tmp.b[8*i+2] = 0x00;
        tmp.b[8*i+3] = 0x00;
        tmp.b[8*i+4] = 0x00;
        tmp.b[8*i+5] = 0x00;
        tmp.b[8*i+6] = 0x00;
        tmp.b[8*i+7] = 0x00;

        tmp.b[8*i+16  ] = pws->b[i+16];
        tmp.b[8*i+1+16] = 0x00;
        tmp.b[8*i+2+16] = 0x00;
        tmp.b[8*i+3+16] = 0x00;
        tmp.b[8*i+4+16] = 0x00;
        tmp.b[8*i+5+16] = 0x00;
        tmp.b[8*i+6+16] = 0x00;
        tmp.b[8*i+7+16] = 0x00;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvextl_d_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i;
    wr_t tmp;
    tmp.q[0] = pwd->q[0];
    tmp.q[1] = pwd->q[1];
    for (i = 0; i < 128/64; i++) {
        tmp.h[4*i  ] = pws->h[i];
        tmp.h[4*i+1] = 0x0000;
        tmp.h[4*i+2] = 0x0000;
        tmp.h[4*i+3] = 0x0000;

        tmp.h[4*i+8  ] = pws->h[i+8];
        tmp.h[4*i+1+8] = 0x0000;
        tmp.h[4*i+2+8] = 0x0000;
        tmp.h[4*i+3+8] = 0x0000;
    }
    pwd->q[0] = tmp.q[0];
    pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvhadd8_d_bu(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i;
    for (i = 0; i < 256/64; i++) {
        pwd->d[i] = (int64_t)(uint8_t)pws->b[8 * i    ]
                  + (int64_t)(uint8_t)pws->b[8 * i + 1]
                  + (int64_t)(uint8_t)pws->b[8 * i + 2]
                  + (int64_t)(uint8_t)pws->b[8 * i + 3]
                  + (int64_t)(uint8_t)pws->b[8 * i + 4]
                  + (int64_t)(uint8_t)pws->b[8 * i + 5]
                  + (int64_t)(uint8_t)pws->b[8 * i + 6]
                  + (int64_t)(uint8_t)pws->b[8 * i + 7];
    }
}

void helper_lsx_xvhminpos_w_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i;
    uint16_t ss,tt;
    for (i = 0; i < 128/32; i++) {
        ss = (uint16_t)pws->h[2*i+1];
        tt = (uint16_t)pws->h[2*i];
        pwd->h[2*i  ] = (ss<tt) ? ss : tt;
        pwd->h[2*i+1] = (ss<tt) ? 2*i+1 : 2*i;

        ss = (uint16_t)pws->h[2*i+1+8];
        tt = (uint16_t)pws->h[2*i+8];
        pwd->h[2*i+8  ] = (ss<tt) ? ss : tt;
        pwd->h[2*i+1+8] = (ss<tt) ? 2*i+1 : 2*i;
    }
}

void helper_lsx_xvhminpos_d_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i;
    uint16_t ss,tt,h1,h3;
    for (i = 0; i < 128/64; i++) {
        ss = (uint16_t)pws->h[4*i+2];
        tt = (uint16_t)pws->h[4*i];
        h1 = (uint16_t)pws->h[4*i+1];
        h3 = (uint16_t)pws->h[4*i+3];
        pwd->h[4*i  ] = (ss<tt) ? ss : tt;
        pwd->h[4*i+1] = (ss<tt) ? h3 : h1;
        pwd->w[2*i+1] = 0;

        ss = (uint16_t)pws->h[4*i+2+8];
        tt = (uint16_t)pws->h[4*i+8];
        h1 = (uint16_t)pws->h[4*i+1+8];
        h3 = (uint16_t)pws->h[4*i+3+8];
        pwd->h[4*i+8  ] = (ss<tt) ? ss : tt;
        pwd->h[4*i+1+8] = (ss<tt) ? h3 : h1;
        pwd->w[2*i+1+4] = 0;
    }
}

void helper_lsx_xvhminpos_q_hu(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

	wr_t tmp;
    tmp.q[0] = 0;
    tmp.q[1] = 0;
    int i;
    uint16_t ss,tt,h1,h5;
    for (i = 0; i < 128/128; i++) {
     ss = (uint16_t)pws->h[8*i+4];
     tt = (uint16_t)pws->h[8*i  ];
     h1 = (uint16_t)pws->h[8*i+1];
     h5 = (uint16_t)pws->h[8*i+5];
     tmp.h[8*i  ] = (ss<tt) ? ss : tt;
     tmp.h[8*i+1] = (ss<tt) ? h5 : h1;

     ss = (uint16_t)pws->h[8*i+4+8];
     tt = (uint16_t)pws->h[8*i+8  ];
     h1 = (uint16_t)pws->h[8*i+1+8];
     h5 = (uint16_t)pws->h[8*i+5+8];
     tmp.h[8*i+8  ] = (ss<tt) ? ss : tt;
     tmp.h[8*i+1+8] = (ss<tt) ? h5 : h1;
    }
	pwd->q[0] = tmp.q[0];
	pwd->q[1] = tmp.q[1];
}

void helper_lsx_xvclrtail_b(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i;
    int flag = 0;
    for (i = 0; i < 128/8; i++) {
        if(pws->b[i]==0)
            flag = 1;
        if(flag)
            pwd->b[i] = 0x00;
        else
            pwd->b[i] = pws->b[i];
    }

    flag = 0;

    for (i = 16; i < 16 + 128/8; i++) {
        if(pws->b[i]==0)
            flag = 1;
        if(flag)
            pwd->b[i] = 0x00;
        else
            pwd->b[i] = pws->b[i];
    }
}

void helper_lsx_xvclrtail_h(CPULoongArchState *env, uint32_t wd, uint32_t ws)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i;
    int flag = 0;
    for (i = 0; i < 128/16; i++) {
        if(pws->h[i]==0)
            flag = 1;
        if(flag)
            pwd->h[i] = 0x0000;
        else
            pwd->h[i] = pws->h[i];
    }

    flag = 0;

    for (i = 8; i < 8 + 128/16; i++) {
        if(pws->h[i]==0)
            flag = 1;
        if(flag)
            pwd->h[i] = 0x0000;
        else
            pwd->h[i] = pws->h[i];
    }
}

void helper_lsx_xvbitrevi_df(CPULoongArchState *env, uint32_t df, uint32_t wd,
                       uint32_t ws, uint32_t u5)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);
    uint32_t i;
    switch (df) {
    case DF_BYTE:
        for (i = 0; i < DF_ELEMENTS(DF_BYTE); i++) {
            pwd->b[i] = lsx_vbitrev_df(df, pws->b[i], u5);
        }
        break;
    case DF_HALF:
        for (i = 0; i < DF_ELEMENTS(DF_HALF); i++) {
            pwd->h[i] = lsx_vbitrev_df(df, pws->h[i], u5);
        }
        break;
    case DF_WORD:
        for (i = 0; i < DF_ELEMENTS(DF_WORD); i++) {
            pwd->w[i] = lsx_vbitrev_df(df, pws->w[i], u5);
        }
        break;
    case DF_DOUBLE:
        for (i = 0; i < DF_ELEMENTS(DF_DOUBLE); i++) {
            pwd->d[i] = lsx_vbitrev_df(df, pws->d[i], u5);
        }
        break;
    default:
        assert(0);
    }
}

void helper_lsx_xvextrins_d(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui8)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int ins,extr;
    ins  = (ui8 >> 4) & 0x3;
    extr =  ui8       & 0x3;
    pwd->d[ins] = pws->d[extr];
}

void helper_lsx_xvextrins_w(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui8)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int ins,extr;
    ins  = (ui8 >> 4) & 0x7;
    extr =  ui8       & 0x7;
    pwd->w[ins] = pws->w[extr];
}

void helper_lsx_xvextrins_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui8)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int ins,extr;
    ins  = (ui8 >> 4) & 0xf;
    extr =  ui8       & 0xf;
    pwd->h[ins] = pws->h[extr];
}

void helper_lsx_xvextrins_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui8)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int ins,extr;
    ins  = (ui8 >> 4) & 0x1f;
    extr =  ui8       & 0x1f;
    pwd->b[ins] = pws->b[extr];
}

void helper_lsx_xvshufi1_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui8)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i,k,l,j;
    for(j=0;j<2;++j){
        k = ui8;
        for(i=0;i<4;++i){
            l = k & 0x3;
            pwd->h[j*8+i] = pws->h[j*8+l];
            k = k >> 2;
        }
    }
}

void helper_lsx_xvshufi2_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui8)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i,k,l,j;
    for(j=0;j<2;++j){
        k = ui8;
        for(i=0;i<4;++i){
            l = k & 0x3;
            pwd->h[j*8+i+4] = pws->h[j*8+l+4];
            k = k >> 2;
        }
    }
}

void helper_lsx_xvshufi1_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui8)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i,k,l,j;
    for(j=0;j<2;++j){
        k = ui8;
        for(i=0;i<4;++i){
            l = k & 0x3;
            pwd->b[j*16+i] = pws->b[j*16+l];
            k = k >> 2;
        }
    }
}

void helper_lsx_xvshufi2_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui8)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i,k,l,j;
    for(j=0;j<2;++j){
        k = ui8;
        for(i=0;i<4;++i){
            l = k & 0x3;
            pwd->b[j*16+i+4] = pws->b[j*16+l+4];
            k = k >> 2;
        }
    }
}

void helper_lsx_xvshufi3_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui8)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i,k,l,j;
    for(j=0;j<2;++j){
        k = ui8;
        for(i=0;i<4;++i){
            l = k & 0x3;
            pwd->b[j*16+i+8] = pws->b[j*16+l+8];
            k = k >> 2;
        }
    }
}

void helper_lsx_xvshufi4_b(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui8)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i,k,l,j;
    for(j=0;j<2;++j){
        k = ui8;
        for(i=0;i<4;++i){
            l = k & 0x3;
            pwd->b[j*16+i+12] = pws->b[j*16+l+12];
            k = k >> 2;
        }
    }
}

void helper_lsx_xvseli_h(CPULoongArchState *env, uint32_t wd, uint32_t ws, uint32_t ui8)
{
    wr_t *pwd = &(env->fpr[wd].wr);
    wr_t *pws = &(env->fpr[ws].wr);

    int i,k;
    k = ui8;
    for(i=0;i<256/16;++i){
        if(k & 0x1){
            pwd->h[i] = pws->h[i];
        }
        k = k >> 1;
    }
}
