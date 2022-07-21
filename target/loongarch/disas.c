/* SPDX-License-Identifier: GPL-2.0-or-later */
/*
 * QEMU LoongArch Disassembler
 *
 * Copyright (c) 2021 Loongson Technology Corporation Limited.
 */

#include "qemu/osdep.h"
#include "disas/dis-asm.h"
#include "qemu/bitops.h"
#include "cpu-csr.h"
#include "cpu.h"

const char * const vregnames[] = {
    "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
    "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
    "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
    "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
};

const char * const xregnames[] = {
    "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7",
    "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15",
    "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23",
    "x24", "x25", "x26", "x27", "x28", "x29", "x30", "x31",
};

const char * const fccregnames[8] = {
  "$fcc0", "$fcc1", "$fcc2", "$fcc3", "$fcc4", "$fcc5", "$fcc6", "$fcc7",
};

typedef struct {
    disassemble_info *info;
    uint64_t pc;
    uint32_t insn;
} DisasContext;

static inline int plus_1(DisasContext *ctx, int x)
{
    return x + 1;
}

static inline int shl_2(DisasContext *ctx, int x)
{
    return x << 2;
}

#define CSR_NAME(REG) \
    [LOONGARCH_CSR_##REG] = (#REG)

static const char * const csr_names[] = {
    CSR_NAME(CRMD),
    CSR_NAME(PRMD),
    CSR_NAME(EUEN),
    CSR_NAME(MISC),
    CSR_NAME(ECFG),
    CSR_NAME(ESTAT),
    CSR_NAME(ERA),
    CSR_NAME(BADV),
    CSR_NAME(BADI),
    CSR_NAME(EENTRY),
    CSR_NAME(TLBIDX),
    CSR_NAME(TLBEHI),
    CSR_NAME(TLBELO0),
    CSR_NAME(TLBELO1),
    CSR_NAME(ASID),
    CSR_NAME(PGDL),
    CSR_NAME(PGDH),
    CSR_NAME(PGD),
    CSR_NAME(PWCL),
    CSR_NAME(PWCH),
    CSR_NAME(STLBPS),
    CSR_NAME(RVACFG),
    CSR_NAME(CPUID),
    CSR_NAME(PRCFG1),
    CSR_NAME(PRCFG2),
    CSR_NAME(PRCFG3),
    CSR_NAME(SAVE(0)),
    CSR_NAME(SAVE(1)),
    CSR_NAME(SAVE(2)),
    CSR_NAME(SAVE(3)),
    CSR_NAME(SAVE(4)),
    CSR_NAME(SAVE(5)),
    CSR_NAME(SAVE(6)),
    CSR_NAME(SAVE(7)),
    CSR_NAME(SAVE(8)),
    CSR_NAME(SAVE(9)),
    CSR_NAME(SAVE(10)),
    CSR_NAME(SAVE(11)),
    CSR_NAME(SAVE(12)),
    CSR_NAME(SAVE(13)),
    CSR_NAME(SAVE(14)),
    CSR_NAME(SAVE(15)),
    CSR_NAME(TID),
    CSR_NAME(TCFG),
    CSR_NAME(TVAL),
    CSR_NAME(CNTC),
    CSR_NAME(TICLR),
    CSR_NAME(LLBCTL),
    CSR_NAME(IMPCTL1),
    CSR_NAME(IMPCTL2),
    CSR_NAME(TLBRENTRY),
    CSR_NAME(TLBRBADV),
    CSR_NAME(TLBRERA),
    CSR_NAME(TLBRSAVE),
    CSR_NAME(TLBRELO0),
    CSR_NAME(TLBRELO1),
    CSR_NAME(TLBREHI),
    CSR_NAME(TLBRPRMD),
    CSR_NAME(MERRCTL),
    CSR_NAME(MERRINFO1),
    CSR_NAME(MERRINFO2),
    CSR_NAME(MERRENTRY),
    CSR_NAME(MERRERA),
    CSR_NAME(MERRSAVE),
    CSR_NAME(CTAG),
    CSR_NAME(DMW(0)),
    CSR_NAME(DMW(1)),
    CSR_NAME(DMW(2)),
    CSR_NAME(DMW(3)),
    CSR_NAME(DBG),
    CSR_NAME(DERA),
    CSR_NAME(DSAVE),
};

static const char *get_csr_name(unsigned num)
{
    return ((num < ARRAY_SIZE(csr_names)) && (csr_names[num] != NULL)) ?
           csr_names[num] : "Undefined CSR";
}

#define output(C, INSN, FMT, ...)                                   \
{                                                                   \
    (C)->info->fprintf_func((C)->info->stream, "%08x   %-9s\t" FMT, \
                            (C)->insn, INSN, ##__VA_ARGS__);        \
}

#include "decode-insns.c.inc"

int print_insn_loongarch(bfd_vma memaddr, struct disassemble_info *info)
{
    bfd_byte buffer[4];
    uint32_t insn;
    int status;

    status = (*info->read_memory_func)(memaddr, buffer, 4, info);
    if (status != 0) {
        (*info->memory_error_func)(status, memaddr, info);
        return -1;
    }
    insn = bfd_getl32(buffer);
    DisasContext ctx = {
        .info = info,
        .pc = memaddr,
        .insn = insn
    };

    if (!decode(&ctx, insn)) {
        output(&ctx, "illegal", "");
    }
    return 4;
}

static void output_r_i(DisasContext *ctx, arg_r_i *a, const char *mnemonic)
{
    output(ctx, mnemonic, "r%d, %d", a->rd, a->imm);
}

static void output_rrr(DisasContext *ctx, arg_rrr *a, const char *mnemonic)
{
    output(ctx, mnemonic, "r%d, r%d, r%d", a->rd, a->rj, a->rk);
}

static void output_rr_i(DisasContext *ctx, arg_rr_i *a, const char *mnemonic)
{
    output(ctx, mnemonic, "r%d, r%d, %d", a->rd, a->rj, a->imm);
}

static void output_rrr_sa(DisasContext *ctx, arg_rrr_sa *a,
                          const char *mnemonic)
{
    output(ctx, mnemonic, "r%d, r%d, r%d, %d", a->rd, a->rj, a->rk, a->sa);
}

static void output_rr(DisasContext *ctx, arg_rr *a, const char *mnemonic)
{
    output(ctx, mnemonic, "r%d, r%d", a->rd, a->rj);
}

static void output_rr_ms_ls(DisasContext *ctx, arg_rr_ms_ls *a,
                          const char *mnemonic)
{
    output(ctx, mnemonic, "r%d, r%d, %d, %d", a->rd, a->rj, a->ms, a->ls);
}

static void output_hint_r_i(DisasContext *ctx, arg_hint_r_i *a,
                            const char *mnemonic)
{
    output(ctx, mnemonic, "%d, r%d, %d", a->hint, a->rj, a->imm);
}

static void output_i(DisasContext *ctx, arg_i *a, const char *mnemonic)
{
    output(ctx, mnemonic, "%d", a->imm);
}

static void output_rr_jk(DisasContext *ctx, arg_rr_jk *a,
                         const char *mnemonic)
{
    output(ctx, mnemonic, "r%d, r%d", a->rj, a->rk);
}

static void output_ff(DisasContext *ctx, arg_ff *a, const char *mnemonic)
{
    output(ctx, mnemonic, "f%d, f%d", a->fd, a->fj);
}

static void output_fff(DisasContext *ctx, arg_fff *a, const char *mnemonic)
{
    output(ctx, mnemonic, "f%d, f%d, f%d", a->fd, a->fj, a->fk);
}

static void output_ffff(DisasContext *ctx, arg_ffff *a, const char *mnemonic)
{
    output(ctx, mnemonic, "f%d, f%d, f%d, f%d", a->fd, a->fj, a->fk, a->fa);
}

static void output_fffc(DisasContext *ctx, arg_fffc *a, const char *mnemonic)
{
    output(ctx, mnemonic, "f%d, f%d, f%d, %d", a->fd, a->fj, a->fk, a->ca);
}

static void output_fr(DisasContext *ctx, arg_fr *a, const char *mnemonic)
{
    output(ctx, mnemonic, "f%d, r%d", a->fd, a->rj);
}

static void output_rf(DisasContext *ctx, arg_rf *a, const char *mnemonic)
{
    output(ctx, mnemonic, "r%d, f%d", a->rd, a->fj);
}

static void output_fcsrd_r(DisasContext *ctx, arg_fcsrd_r *a,
                           const char *mnemonic)
{
    output(ctx, mnemonic, "fcsr%d, r%d", a->fcsrd, a->rj);
}

static void output_r_fcsrs(DisasContext *ctx, arg_r_fcsrs *a,
                           const char *mnemonic)
{
    output(ctx, mnemonic, "r%d, fcsr%d", a->rd, a->fcsrs);
}

static void output_cf(DisasContext *ctx, arg_cf *a, const char *mnemonic)
{
    output(ctx, mnemonic, "fcc%d, f%d", a->cd, a->fj);
}

static void output_fc(DisasContext *ctx, arg_fc *a, const char *mnemonic)
{
    output(ctx, mnemonic, "f%d, fcc%d", a->fd, a->cj);
}

static void output_cr(DisasContext *ctx, arg_cr *a, const char *mnemonic)
{
    output(ctx, mnemonic, "fcc%d, r%d", a->cd, a->rj);
}

static void output_rc(DisasContext *ctx, arg_rc *a, const char *mnemonic)
{
    output(ctx, mnemonic, "r%d, fcc%d", a->rd, a->cj);
}

static void output_frr(DisasContext *ctx, arg_frr *a, const char *mnemonic)
{
    output(ctx, mnemonic, "f%d, r%d, r%d", a->fd, a->rj, a->rk);
}

static void output_fr_i(DisasContext *ctx, arg_fr_i *a, const char *mnemonic)
{
    output(ctx, mnemonic, "f%d, r%d, %d", a->fd, a->rj, a->imm);
}

static void output_r_offs(DisasContext *ctx, arg_r_offs *a,
                          const char *mnemonic)
{
    output(ctx, mnemonic, "r%d, %d # 0x%" PRIx64, a->rj, a->offs,
           ctx->pc + a->offs);
}

static void output_c_offs(DisasContext *ctx, arg_c_offs *a,
                          const char *mnemonic)
{
    output(ctx, mnemonic, "fcc%d, %d # 0x%" PRIx64, a->cj, a->offs,
           ctx->pc + a->offs);
}

static void output_offs(DisasContext *ctx, arg_offs *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%d # 0x%" PRIx64, a->offs, ctx->pc + a->offs);
}

static void output_rr_offs(DisasContext *ctx, arg_rr_offs *a,
                           const char *mnemonic)
{
    output(ctx, mnemonic, "r%d, r%d, %d # 0x%" PRIx64, a->rj,
           a->rd, a->offs, ctx->pc + a->offs);
}

static void output_r_csr(DisasContext *ctx, arg_r_csr *a,
                         const char *mnemonic)
{
    output(ctx, mnemonic, "r%d, %d # %s", a->rd, a->csr, get_csr_name(a->csr));
}

static void output_rr_csr(DisasContext *ctx, arg_rr_csr *a,
                          const char *mnemonic)
{
    output(ctx, mnemonic, "r%d, r%d, %d # %s",
           a->rd, a->rj, a->csr, get_csr_name(a->csr));
}

static void output_empty(DisasContext *ctx, arg_empty *a,
                         const char *mnemonic)
{
    output(ctx, mnemonic, "");
}

static void output_i_rr(DisasContext *ctx, arg_i_rr *a, const char *mnemonic)
{
    output(ctx, mnemonic, "%d, r%d, r%d", a->imm, a->rj, a->rk);
}

static void output_cop_r_i(DisasContext *ctx, arg_cop_r_i *a,
                           const char *mnemonic)
{
    output(ctx, mnemonic, "%d, r%d, %d", a->cop, a->rj, a->imm);
}

static void output_j_i(DisasContext *ctx, arg_j_i *a, const char *mnemonic)
{
    output(ctx, mnemonic, "r%d, %d", a->rj, a->imm);
}

#define INSN(insn, type)                                    \
static bool trans_##insn(DisasContext *ctx, arg_##type * a) \
{                                                           \
    output_##type(ctx, a, #insn);                           \
    return true;                                            \
}


/* Vec */

#define INSN_VEC(insn, type)                                    \
static bool trans_##insn(DisasContext *ctx, arg_fmt_##type * a) \
{                                                               \
    output_##type(ctx, a, #insn);                               \
    return true;                                                \
}


static void output_cdvj(DisasContext *ctx, arg_fmt_cdvj *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s", fccregnames[a->cd], vregnames[a->vj]);
}

static void output_rdvjui1(DisasContext *ctx, arg_fmt_rdvjui1 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, 0x%x", regnames[a->rd], vregnames[a->vj],  a->ui1);
}

static void output_rdvjui2(DisasContext *ctx, arg_fmt_rdvjui2 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, 0x%x", regnames[a->rd], vregnames[a->vj],  a->ui2);
}

static void output_rdvjui3(DisasContext *ctx, arg_fmt_rdvjui3 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, 0x%x", regnames[a->rd], vregnames[a->vj],  a->ui3);
}

static void output_rdvjui4(DisasContext *ctx, arg_fmt_rdvjui4 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, 0x%x", regnames[a->rd], vregnames[a->vj],  a->ui4);
}


static void output_vdi13(DisasContext *ctx, arg_fmt_vdi13 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, 0x%x", vregnames[a->vd], a->i13);
}

static void output_vdmodeui5(DisasContext *ctx, arg_fmt_vdmodeui5 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, 0x%x, 0x%x", vregnames[a->vd], a->mode, a->ui5);
}

static void output_vdrj(DisasContext *ctx, arg_fmt_vdrj *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s", vregnames[a->vd], regnames[a->rj]);
}

static void output_vdrjsi10(DisasContext *ctx, arg_fmt_vdrjsi10 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, 0x%x", vregnames[a->vd], regnames[a->rj], a->si10);
}

static void output_vdrjsi11(DisasContext *ctx, arg_fmt_vdrjsi11 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, 0x%x", vregnames[a->vd], regnames[a->rj], a->si11);
}

static void output_vdrjsi12(DisasContext *ctx, arg_fmt_vdrjsi12 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, 0x%x", vregnames[a->vd], regnames[a->rj], a->si12);
}

static void output_vdrjsi8idx1(DisasContext *ctx, arg_fmt_vdrjsi8idx1 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, 0x%x, 0x%x", vregnames[a->vd], regnames[a->rj], a->si8, a->idx1);
}

static void output_vdrjsi8idx2(DisasContext *ctx, arg_fmt_vdrjsi8idx2 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, 0x%x, 0x%x", vregnames[a->vd], regnames[a->rj], a->si8, a->idx2);
}

static void output_vdrjsi8idx3(DisasContext *ctx, arg_fmt_vdrjsi8idx3 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, 0x%x, 0x%x", vregnames[a->vd], regnames[a->rj], a->si8, a->idx3);
}

static void output_vdrjsi8idx4(DisasContext *ctx, arg_fmt_vdrjsi8idx4 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, 0x%x, 0x%x", vregnames[a->vd], regnames[a->rj], a->si8, a->idx4);
}

static void output_vdrjsi9(DisasContext *ctx, arg_fmt_vdrjsi9 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, 0x%x", vregnames[a->vd], regnames[a->rj], a->si9);
}

static void output_vdrjui1(DisasContext *ctx, arg_fmt_vdrjui1 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, 0x%x", vregnames[a->vd], regnames[a->rj], a->ui1);
}

static void output_vdrjui2(DisasContext *ctx, arg_fmt_vdrjui2 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, 0x%x", vregnames[a->vd], regnames[a->rj], a->ui2);
}

static void output_vdrjui3(DisasContext *ctx, arg_fmt_vdrjui3 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, 0x%x", vregnames[a->vd], regnames[a->rj], a->ui3);
}

static void output_vdrjui4(DisasContext *ctx, arg_fmt_vdrjui4 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, 0x%x", vregnames[a->vd], regnames[a->rj], a->ui4);
}

static void output_vdvj(DisasContext *ctx, arg_fmt_vdvj *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s", vregnames[a->vd], vregnames[a->vj]);
}

static void output_vdvjrk(DisasContext *ctx, arg_fmt_vdvjrk *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, %s", vregnames[a->vd], vregnames[a->vj], regnames[a->rk]);
}

static void output_vdvjsi5(DisasContext *ctx, arg_fmt_vdvjsi5 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, 0x%x", vregnames[a->vd], vregnames[a->vj], a->si5);
}

static void output_vdvjui1(DisasContext *ctx, arg_fmt_vdvjui1 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, 0x%x", vregnames[a->vd], vregnames[a->vj], a->ui1);
}

static void output_vdvjui2(DisasContext *ctx, arg_fmt_vdvjui2 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, 0x%x", vregnames[a->vd], vregnames[a->vj], a->ui2);
}

static void output_vdvjui3(DisasContext *ctx, arg_fmt_vdvjui3 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, 0x%x", vregnames[a->vd], vregnames[a->vj], a->ui3);
}

static void output_vdvjui4(DisasContext *ctx, arg_fmt_vdvjui4 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, 0x%x", vregnames[a->vd], vregnames[a->vj], a->ui4);
}

static void output_vdvjui5(DisasContext *ctx, arg_fmt_vdvjui5 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, 0x%x", vregnames[a->vd], vregnames[a->vj], a->ui5);
}

static void output_vdvjui6(DisasContext *ctx, arg_fmt_vdvjui6 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, 0x%x", vregnames[a->vd], vregnames[a->vj], a->ui6);
}

static void output_vdvjui7(DisasContext *ctx, arg_fmt_vdvjui7 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, 0x%x", vregnames[a->vd], vregnames[a->vj], a->ui7);
}

static void output_vdvjui8(DisasContext *ctx, arg_fmt_vdvjui8 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, 0x%x", vregnames[a->vd], vregnames[a->vj], a->ui8);
}

static void output_vdvjvk(DisasContext *ctx, arg_fmt_vdvjvk *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, %s", vregnames[a->vd], vregnames[a->vj], vregnames[a->vk]);
}

static void output_vdvjvkfcond(DisasContext *ctx, arg_fmt_vdvjvkfcond *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, %s, 0x%x", vregnames[a->vd], vregnames[a->vj], vregnames[a->vk], a->fcond);
}

static void output_vdvjvkva(DisasContext *ctx, arg_fmt_vdvjvkva *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, %s, %s", vregnames[a->vd], vregnames[a->vj], vregnames[a->vk], vregnames[a->va]);
}

static void output_vdvjvkvui5(DisasContext *ctx, arg_fmt_vdvjvkvui5 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, %s, 0x%x", vregnames[a->vd], vregnames[a->vj], vregnames[a->vk], a->vui5);
}

static void output_cdxj(DisasContext *ctx, arg_fmt_cdxj *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s", fccregnames[a->cd], xregnames[a->xj]);
}

static void output_rdxjui2(DisasContext *ctx, arg_fmt_rdxjui2 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, 0x%x", regnames[a->rd], xregnames[a->xj], a->ui2);
}

static void output_rdxjui3(DisasContext *ctx, arg_fmt_rdxjui3 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, 0x%x", regnames[a->rd], xregnames[a->xj], a->ui3);
}

static void output_xdi13(DisasContext *ctx, arg_fmt_xdi13 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, 0x%x", xregnames[a->xd], a->i13);
}

static void output_xdmodeui5(DisasContext *ctx, arg_fmt_xdmodeui5 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, 0x%x, 0x%x", xregnames[a->xd], a->mode, a->ui5);
}

static void output_xdrj(DisasContext *ctx, arg_fmt_xdrj *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s", xregnames[a->xd], regnames[a->rj]);
}

static void output_xdrjsi10(DisasContext *ctx, arg_fmt_xdrjsi10 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, 0x%x", xregnames[a->xd], regnames[a->rj], a->si10);
}

static void output_xdrjsi11(DisasContext *ctx, arg_fmt_xdrjsi11 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, 0x%x", xregnames[a->xd], regnames[a->rj], a->si11);
}

static void output_xdrjsi12(DisasContext *ctx, arg_fmt_xdrjsi12 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, 0x%x", xregnames[a->xd], regnames[a->rj], a->si12);
}

static void output_xdrjsi8idx(DisasContext *ctx, arg_fmt_xdrjsi8idx *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, 0x%x, 0x%x", xregnames[a->xd], regnames[a->rj], a->si8, a->idx);
}

static void output_xdrjsi8idx2(DisasContext *ctx, arg_fmt_xdrjsi8idx2 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, 0x%x, 0x%x", xregnames[a->xd], regnames[a->rj], a->si8, a->idx2);
}

static void output_xdrjsi8idx3(DisasContext *ctx, arg_fmt_xdrjsi8idx3 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, 0x%x, 0x%x", xregnames[a->xd], regnames[a->rj], a->si8, a->idx3);
}

static void output_xdrjsi8idx4(DisasContext *ctx, arg_fmt_xdrjsi8idx4 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, 0x%x, 0x%x", xregnames[a->xd], regnames[a->rj], a->si8, a->idx4);
}

static void output_xdrjsi9(DisasContext *ctx, arg_fmt_xdrjsi9 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, 0x%x", xregnames[a->xd], regnames[a->rj], a->si9);
}

static void output_xdrjui2(DisasContext *ctx, arg_fmt_xdrjui2 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, 0x%x", xregnames[a->xd], regnames[a->rj], a->ui2);
}

static void output_xdrjui3(DisasContext *ctx, arg_fmt_xdrjui3 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, 0x%x", xregnames[a->xd], regnames[a->rj], a->ui3);
}

static void output_xdxj(DisasContext *ctx, arg_fmt_xdxj *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s", xregnames[a->xd], xregnames[a->xj]);
}

static void output_xdxjrk(DisasContext *ctx, arg_fmt_xdxjrk *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, %s", xregnames[a->xd], xregnames[a->xj], regnames[a->rk]);
}

static void output_xdxjsi5(DisasContext *ctx, arg_fmt_xdxjsi5 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, 0x%x", xregnames[a->xd], xregnames[a->xj], a->si5);
}

static void output_xdxjui1(DisasContext *ctx, arg_fmt_xdxjui1 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, 0x%x", xregnames[a->xd], xregnames[a->xj], a->ui1);
}

static void output_xdxjui2(DisasContext *ctx, arg_fmt_xdxjui2 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, 0x%x", xregnames[a->xd], xregnames[a->xj], a->ui2);
}

static void output_xdxjui3(DisasContext *ctx, arg_fmt_xdxjui3 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, 0x%x", xregnames[a->xd], xregnames[a->xj], a->ui3);
}

static void output_xdxjui4(DisasContext *ctx, arg_fmt_xdxjui4 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, 0x%x", xregnames[a->xd], xregnames[a->xj], a->ui4);
}

static void output_xdxjui5(DisasContext *ctx, arg_fmt_xdxjui5 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, 0x%x", xregnames[a->xd], xregnames[a->xj], a->ui5);
}

static void output_xdxjui6(DisasContext *ctx, arg_fmt_xdxjui6 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, 0x%x", xregnames[a->xd], xregnames[a->xj], a->ui6);
}

static void output_xdxjui7(DisasContext *ctx, arg_fmt_xdxjui7 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, 0x%x", xregnames[a->xd], xregnames[a->xj], a->ui7);
}

static void output_xdxjui8(DisasContext *ctx, arg_fmt_xdxjui8 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, 0x%x", xregnames[a->xd], xregnames[a->xj], a->ui8);
}

static void output_xdxjxk(DisasContext *ctx, arg_fmt_xdxjxk *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, %s", xregnames[a->xd], xregnames[a->xj], xregnames[a->xk]);
}

static void output_xdxjxkfcond(DisasContext *ctx, arg_fmt_xdxjxkfcond *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, %s, 0x%x", xregnames[a->xd], xregnames[a->xj], xregnames[a->xk], a->fcond);
}

static void output_xdxjxkxa(DisasContext *ctx, arg_fmt_xdxjxkxa *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, %s, %s", xregnames[a->xd], xregnames[a->xj], xregnames[a->xk], xregnames[a->xa]);
}

static void output_xdxjxkvui5(DisasContext *ctx, arg_fmt_xdxjxkvui5 *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, %s, 0x%x", xregnames[a->xd], xregnames[a->xj], xregnames[a->xk], a->vui5);
}


static void output_vdrjrk(DisasContext *ctx, arg_fmt_vdrjrk *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, %s", vregnames[a->vd], regnames[a->rj], regnames[a->rk]);
}

static void output_xdrjrk(DisasContext *ctx, arg_fmt_xdrjrk *a,
                        const char *mnemonic)
{
    output(ctx, mnemonic, "%s, %s, %s", xregnames[a->xd], regnames[a->rj], regnames[a->rk]);
}

INSN(clo_w,        rr)
INSN(clz_w,        rr)
INSN(cto_w,        rr)
INSN(ctz_w,        rr)
INSN(clo_d,        rr)
INSN(clz_d,        rr)
INSN(cto_d,        rr)
INSN(ctz_d,        rr)
INSN(revb_2h,      rr)
INSN(revb_4h,      rr)
INSN(revb_2w,      rr)
INSN(revb_d,       rr)
INSN(revh_2w,      rr)
INSN(revh_d,       rr)
INSN(bitrev_4b,    rr)
INSN(bitrev_8b,    rr)
INSN(bitrev_w,     rr)
INSN(bitrev_d,     rr)
INSN(ext_w_h,      rr)
INSN(ext_w_b,      rr)
INSN(rdtimel_w,    rr)
INSN(rdtimeh_w,    rr)
INSN(rdtime_d,     rr)
INSN(cpucfg,       rr)
INSN(asrtle_d,     rr_jk)
INSN(asrtgt_d,     rr_jk)
INSN(alsl_w,       rrr_sa)
INSN(alsl_wu,      rrr_sa)
INSN(bytepick_w,   rrr_sa)
INSN(bytepick_d,   rrr_sa)
INSN(add_w,        rrr)
INSN(add_d,        rrr)
INSN(sub_w,        rrr)
INSN(sub_d,        rrr)
INSN(slt,          rrr)
INSN(sltu,         rrr)
INSN(maskeqz,      rrr)
INSN(masknez,      rrr)
INSN(nor,          rrr)
INSN(and,          rrr)
INSN(or,           rrr)
INSN(xor,          rrr)
INSN(orn,          rrr)
INSN(andn,         rrr)
INSN(sll_w,        rrr)
INSN(srl_w,        rrr)
INSN(sra_w,        rrr)
INSN(sll_d,        rrr)
INSN(srl_d,        rrr)
INSN(sra_d,        rrr)
INSN(rotr_w,       rrr)
INSN(rotr_d,       rrr)
INSN(mul_w,        rrr)
INSN(mulh_w,       rrr)
INSN(mulh_wu,      rrr)
INSN(mul_d,        rrr)
INSN(mulh_d,       rrr)
INSN(mulh_du,      rrr)
INSN(mulw_d_w,     rrr)
INSN(mulw_d_wu,    rrr)
INSN(div_w,        rrr)
INSN(mod_w,        rrr)
INSN(div_wu,       rrr)
INSN(mod_wu,       rrr)
INSN(div_d,        rrr)
INSN(mod_d,        rrr)
INSN(div_du,       rrr)
INSN(mod_du,       rrr)
INSN(crc_w_b_w,    rrr)
INSN(crc_w_h_w,    rrr)
INSN(crc_w_w_w,    rrr)
INSN(crc_w_d_w,    rrr)
INSN(crcc_w_b_w,   rrr)
INSN(crcc_w_h_w,   rrr)
INSN(crcc_w_w_w,   rrr)
INSN(crcc_w_d_w,   rrr)
INSN(break,        i)
INSN(syscall,      i)
INSN(alsl_d,       rrr_sa)
INSN(slli_w,       rr_i)
INSN(slli_d,       rr_i)
INSN(srli_w,       rr_i)
INSN(srli_d,       rr_i)
INSN(srai_w,       rr_i)
INSN(srai_d,       rr_i)
INSN(rotri_w,      rr_i)
INSN(rotri_d,      rr_i)
INSN(bstrins_w,    rr_ms_ls)
INSN(bstrpick_w,   rr_ms_ls)
INSN(bstrins_d,    rr_ms_ls)
INSN(bstrpick_d,   rr_ms_ls)
INSN(fadd_s,       fff)
INSN(fadd_d,       fff)
INSN(fsub_s,       fff)
INSN(fsub_d,       fff)
INSN(fmul_s,       fff)
INSN(fmul_d,       fff)
INSN(fdiv_s,       fff)
INSN(fdiv_d,       fff)
INSN(fmax_s,       fff)
INSN(fmax_d,       fff)
INSN(fmin_s,       fff)
INSN(fmin_d,       fff)
INSN(fmaxa_s,      fff)
INSN(fmaxa_d,      fff)
INSN(fmina_s,      fff)
INSN(fmina_d,      fff)
INSN(fscaleb_s,    fff)
INSN(fscaleb_d,    fff)
INSN(fcopysign_s,  fff)
INSN(fcopysign_d,  fff)
INSN(fabs_s,       ff)
INSN(fabs_d,       ff)
INSN(fneg_s,       ff)
INSN(fneg_d,       ff)
INSN(flogb_s,      ff)
INSN(flogb_d,      ff)
INSN(fclass_s,     ff)
INSN(fclass_d,     ff)
INSN(fsqrt_s,      ff)
INSN(fsqrt_d,      ff)
INSN(frecip_s,     ff)
INSN(frecip_d,     ff)
INSN(frsqrt_s,     ff)
INSN(frsqrt_d,     ff)
INSN(fmov_s,       ff)
INSN(fmov_d,       ff)
INSN(movgr2fr_w,   fr)
INSN(movgr2fr_d,   fr)
INSN(movgr2frh_w,  fr)
INSN(movfr2gr_s,   rf)
INSN(movfr2gr_d,   rf)
INSN(movfrh2gr_s,  rf)
INSN(movgr2fcsr,   fcsrd_r)
INSN(movfcsr2gr,   r_fcsrs)
INSN(movfr2cf,     cf)
INSN(movcf2fr,     fc)
INSN(movgr2cf,     cr)
INSN(movcf2gr,     rc)
INSN(fcvt_s_d,     ff)
INSN(fcvt_d_s,     ff)
INSN(ftintrm_w_s,  ff)
INSN(ftintrm_w_d,  ff)
INSN(ftintrm_l_s,  ff)
INSN(ftintrm_l_d,  ff)
INSN(ftintrp_w_s,  ff)
INSN(ftintrp_w_d,  ff)
INSN(ftintrp_l_s,  ff)
INSN(ftintrp_l_d,  ff)
INSN(ftintrz_w_s,  ff)
INSN(ftintrz_w_d,  ff)
INSN(ftintrz_l_s,  ff)
INSN(ftintrz_l_d,  ff)
INSN(ftintrne_w_s, ff)
INSN(ftintrne_w_d, ff)
INSN(ftintrne_l_s, ff)
INSN(ftintrne_l_d, ff)
INSN(ftint_w_s,    ff)
INSN(ftint_w_d,    ff)
INSN(ftint_l_s,    ff)
INSN(ftint_l_d,    ff)
INSN(ffint_s_w,    ff)
INSN(ffint_s_l,    ff)
INSN(ffint_d_w,    ff)
INSN(ffint_d_l,    ff)
INSN(frint_s,      ff)
INSN(frint_d,      ff)
INSN(slti,         rr_i)
INSN(sltui,        rr_i)
INSN(addi_w,       rr_i)
INSN(addi_d,       rr_i)
INSN(lu52i_d,      rr_i)
INSN(andi,         rr_i)
INSN(ori,          rr_i)
INSN(xori,         rr_i)
INSN(fmadd_s,      ffff)
INSN(fmadd_d,      ffff)
INSN(fmsub_s,      ffff)
INSN(fmsub_d,      ffff)
INSN(fnmadd_s,     ffff)
INSN(fnmadd_d,     ffff)
INSN(fnmsub_s,     ffff)
INSN(fnmsub_d,     ffff)
INSN(fsel,         fffc)
INSN(addu16i_d,    rr_i)
INSN(lu12i_w,      r_i)
INSN(lu32i_d,      r_i)
INSN(pcaddi,       r_i)
INSN(pcalau12i,    r_i)
INSN(pcaddu12i,    r_i)
INSN(pcaddu18i,    r_i)
INSN(ll_w,         rr_i)
INSN(sc_w,         rr_i)
INSN(ll_d,         rr_i)
INSN(sc_d,         rr_i)
INSN(ldptr_w,      rr_i)
INSN(stptr_w,      rr_i)
INSN(ldptr_d,      rr_i)
INSN(stptr_d,      rr_i)
INSN(ld_b,         rr_i)
INSN(ld_h,         rr_i)
INSN(ld_w,         rr_i)
INSN(ld_d,         rr_i)
INSN(st_b,         rr_i)
INSN(st_h,         rr_i)
INSN(st_w,         rr_i)
INSN(st_d,         rr_i)
INSN(ld_bu,        rr_i)
INSN(ld_hu,        rr_i)
INSN(ld_wu,        rr_i)
INSN(preld,        hint_r_i)
INSN(fld_s,        fr_i)
INSN(fst_s,        fr_i)
INSN(fld_d,        fr_i)
INSN(fst_d,        fr_i)
INSN(ldx_b,        rrr)
INSN(ldx_h,        rrr)
INSN(ldx_w,        rrr)
INSN(ldx_d,        rrr)
INSN(stx_b,        rrr)
INSN(stx_h,        rrr)
INSN(stx_w,        rrr)
INSN(stx_d,        rrr)
INSN(ldx_bu,       rrr)
INSN(ldx_hu,       rrr)
INSN(ldx_wu,       rrr)
INSN(fldx_s,       frr)
INSN(fldx_d,       frr)
INSN(fstx_s,       frr)
INSN(fstx_d,       frr)
INSN(amswap_w,     rrr)
INSN(amswap_d,     rrr)
INSN(amadd_w,      rrr)
INSN(amadd_d,      rrr)
INSN(amand_w,      rrr)
INSN(amand_d,      rrr)
INSN(amor_w,       rrr)
INSN(amor_d,       rrr)
INSN(amxor_w,      rrr)
INSN(amxor_d,      rrr)
INSN(ammax_w,      rrr)
INSN(ammax_d,      rrr)
INSN(ammin_w,      rrr)
INSN(ammin_d,      rrr)
INSN(ammax_wu,     rrr)
INSN(ammax_du,     rrr)
INSN(ammin_wu,     rrr)
INSN(ammin_du,     rrr)
INSN(amswap_db_w,  rrr)
INSN(amswap_db_d,  rrr)
INSN(amadd_db_w,   rrr)
INSN(amadd_db_d,   rrr)
INSN(amand_db_w,   rrr)
INSN(amand_db_d,   rrr)
INSN(amor_db_w,    rrr)
INSN(amor_db_d,    rrr)
INSN(amxor_db_w,   rrr)
INSN(amxor_db_d,   rrr)
INSN(ammax_db_w,   rrr)
INSN(ammax_db_d,   rrr)
INSN(ammin_db_w,   rrr)
INSN(ammin_db_d,   rrr)
INSN(ammax_db_wu,  rrr)
INSN(ammax_db_du,  rrr)
INSN(ammin_db_wu,  rrr)
INSN(ammin_db_du,  rrr)
INSN(dbar,         i)
INSN(ibar,         i)
INSN(fldgt_s,      frr)
INSN(fldgt_d,      frr)
INSN(fldle_s,      frr)
INSN(fldle_d,      frr)
INSN(fstgt_s,      frr)
INSN(fstgt_d,      frr)
INSN(fstle_s,      frr)
INSN(fstle_d,      frr)
INSN(ldgt_b,       rrr)
INSN(ldgt_h,       rrr)
INSN(ldgt_w,       rrr)
INSN(ldgt_d,       rrr)
INSN(ldle_b,       rrr)
INSN(ldle_h,       rrr)
INSN(ldle_w,       rrr)
INSN(ldle_d,       rrr)
INSN(stgt_b,       rrr)
INSN(stgt_h,       rrr)
INSN(stgt_w,       rrr)
INSN(stgt_d,       rrr)
INSN(stle_b,       rrr)
INSN(stle_h,       rrr)
INSN(stle_w,       rrr)
INSN(stle_d,       rrr)
INSN(beqz,         r_offs)
INSN(bnez,         r_offs)
INSN(bceqz,        c_offs)
INSN(bcnez,        c_offs)
INSN(jirl,         rr_offs)
INSN(b,            offs)
INSN(bl,           offs)
INSN(beq,          rr_offs)
INSN(bne,          rr_offs)
INSN(blt,          rr_offs)
INSN(bge,          rr_offs)
INSN(bltu,         rr_offs)
INSN(bgeu,         rr_offs)
INSN(csrrd,        r_csr)
INSN(csrwr,        r_csr)
INSN(csrxchg,      rr_csr)
INSN(iocsrrd_b,    rr)
INSN(iocsrrd_h,    rr)
INSN(iocsrrd_w,    rr)
INSN(iocsrrd_d,    rr)
INSN(iocsrwr_b,    rr)
INSN(iocsrwr_h,    rr)
INSN(iocsrwr_w,    rr)
INSN(iocsrwr_d,    rr)
INSN(tlbsrch,      empty)
INSN(tlbrd,        empty)
INSN(tlbwr,        empty)
INSN(tlbfill,      empty)
INSN(tlbclr,       empty)
INSN(tlbflush,     empty)
INSN(invtlb,       i_rr)
INSN(cacop,        cop_r_i)
INSN(lddir,        rr_i)
INSN(ldpte,        j_i)
INSN(ertn,         empty)
INSN(idle,         i)
INSN(dbcl,         i)

/* vector */
INSN_VEC(vldx,             vdrjrk)
INSN_VEC(vstx,             vdrjrk)
INSN_VEC(xvldx,            xdrjrk)
INSN_VEC(xvstx,            xdrjrk)
INSN_VEC(vld,              vdrjsi12)
INSN_VEC(vst,              vdrjsi12)
INSN_VEC(xvld,             xdrjsi12)
INSN_VEC(xvst,             xdrjsi12)
INSN_VEC(vldrepl_d,        vdrjsi9)
INSN_VEC(vldrepl_w,        vdrjsi10)
INSN_VEC(vldrepl_h,        vdrjsi11)
INSN_VEC(vldrepl_b,        vdrjsi12)
INSN_VEC(vfmadd_s,         vdvjvkva)
INSN_VEC(vfmadd_d,         vdvjvkva)
INSN_VEC(vfmsub_s,         vdvjvkva)
INSN_VEC(vfmsub_d,         vdvjvkva)
INSN_VEC(vfnmadd_s,        vdvjvkva)
INSN_VEC(vfnmadd_d,        vdvjvkva)
INSN_VEC(vfnmsub_s,        vdvjvkva)
INSN_VEC(vfnmsub_d,        vdvjvkva)
INSN_VEC(xvfmadd_s,        xdxjxkxa)
INSN_VEC(xvfmadd_d,        xdxjxkxa)
INSN_VEC(xvfmsub_s,        xdxjxkxa)
INSN_VEC(xvfmsub_d,        xdxjxkxa)
INSN_VEC(xvfnmadd_s,       xdxjxkxa)
INSN_VEC(xvfnmadd_d,       xdxjxkxa)
INSN_VEC(xvfnmsub_s,       xdxjxkxa)
INSN_VEC(xvfnmsub_d,       xdxjxkxa)
INSN_VEC(vbitsel_v,        vdvjvkva)
INSN_VEC(xvbitsel_v,       xdxjxkxa)
INSN_VEC(vshuf_b,          vdvjvkva)
INSN_VEC(xvshuf_b,         xdxjxkxa)
INSN_VEC(vextr_v,          vdvjvkvui5)
INSN_VEC(xvextr_v,         xdxjxkvui5)
INSN_VEC(vfmaddsub_s,      vdvjvkva)
INSN_VEC(vfmaddsub_d,      vdvjvkva)
INSN_VEC(vfmsubadd_s,      vdvjvkva)
INSN_VEC(vfmsubadd_d,      vdvjvkva)
INSN_VEC(xvfmaddsub_s,     xdxjxkxa)
INSN_VEC(xvfmaddsub_d,     xdxjxkxa)
INSN_VEC(xvfmsubadd_s,     xdxjxkxa)
INSN_VEC(xvfmsubadd_d,     xdxjxkxa)
INSN_VEC(vstelm_d,         vdrjsi8idx1)
INSN_VEC(vstelm_w,         vdrjsi8idx2)
INSN_VEC(vstelm_h,         vdrjsi8idx3)
INSN_VEC(vstelm_b,         vdrjsi8idx4)
INSN_VEC(xvldrepl_d,       xdrjsi9)
INSN_VEC(xvldrepl_w,       xdrjsi10)
INSN_VEC(xvldrepl_h,       xdrjsi11)
INSN_VEC(xvldrepl_b,       xdrjsi12)
INSN_VEC(xvstelm_d,        xdrjsi8idx2)
INSN_VEC(xvstelm_w,        xdrjsi8idx3)
INSN_VEC(xvstelm_h,        xdrjsi8idx4)
INSN_VEC(xvstelm_b,        xdrjsi8idx)
INSN_VEC(vseq_b,           vdvjvk)
INSN_VEC(vseq_h,           vdvjvk)
INSN_VEC(vseq_w,           vdvjvk)
INSN_VEC(vseq_d,           vdvjvk)
INSN_VEC(vsle_b,           vdvjvk)
INSN_VEC(vsle_h,           vdvjvk)
INSN_VEC(vsle_w,           vdvjvk)
INSN_VEC(vsle_d,           vdvjvk)
INSN_VEC(vsle_bu,          vdvjvk)
INSN_VEC(vsle_hu,          vdvjvk)
INSN_VEC(vsle_wu,          vdvjvk)
INSN_VEC(vsle_du,          vdvjvk)
INSN_VEC(vslt_b,           vdvjvk)
INSN_VEC(vslt_h,           vdvjvk)
INSN_VEC(vslt_w,           vdvjvk)
INSN_VEC(vslt_d,           vdvjvk)
INSN_VEC(vslt_bu,          vdvjvk)
INSN_VEC(vslt_hu,          vdvjvk)
INSN_VEC(vslt_wu,          vdvjvk)
INSN_VEC(vslt_du,          vdvjvk)
INSN_VEC(vadd_b,           vdvjvk)
INSN_VEC(vadd_h,           vdvjvk)
INSN_VEC(vadd_w,           vdvjvk)
INSN_VEC(vadd_d,           vdvjvk)
INSN_VEC(vsub_b,           vdvjvk)
INSN_VEC(vsub_h,           vdvjvk)
INSN_VEC(vsub_w,           vdvjvk)
INSN_VEC(vsub_d,           vdvjvk)
INSN_VEC(vaddwev_h_b,      vdvjvk)
INSN_VEC(vaddwev_w_h,      vdvjvk)
INSN_VEC(vaddwev_d_w,      vdvjvk)
INSN_VEC(vaddwev_q_d,      vdvjvk)
INSN_VEC(vsubwev_h_b,      vdvjvk)
INSN_VEC(vsubwev_w_h,      vdvjvk)
INSN_VEC(vsubwev_d_w,      vdvjvk)
INSN_VEC(vsubwev_q_d,      vdvjvk)
INSN_VEC(vaddwod_h_b,      vdvjvk)
INSN_VEC(vaddwod_w_h,      vdvjvk)
INSN_VEC(vaddwod_d_w,      vdvjvk)
INSN_VEC(vaddwod_q_d,      vdvjvk)
INSN_VEC(vsubwod_h_b,      vdvjvk)
INSN_VEC(vsubwod_w_h,      vdvjvk)
INSN_VEC(vsubwod_d_w,      vdvjvk)
INSN_VEC(vsubwod_q_d,      vdvjvk)
INSN_VEC(vaddwev_h_bu,     vdvjvk)
INSN_VEC(vaddwev_w_hu,     vdvjvk)
INSN_VEC(vaddwev_d_wu,     vdvjvk)
INSN_VEC(vaddwev_q_du,     vdvjvk)
INSN_VEC(vsubwev_h_bu,     vdvjvk)
INSN_VEC(vsubwev_w_hu,     vdvjvk)
INSN_VEC(vsubwev_d_wu,     vdvjvk)
INSN_VEC(vsubwev_q_du,     vdvjvk)
INSN_VEC(vaddwod_h_bu,     vdvjvk)
INSN_VEC(vaddwod_w_hu,     vdvjvk)
INSN_VEC(vaddwod_d_wu,     vdvjvk)
INSN_VEC(vaddwod_q_du,     vdvjvk)
INSN_VEC(vsubwod_h_bu,     vdvjvk)
INSN_VEC(vsubwod_w_hu,     vdvjvk)
INSN_VEC(vsubwod_d_wu,     vdvjvk)
INSN_VEC(vsubwod_q_du,     vdvjvk)
INSN_VEC(vaddwev_h_bu_b,   vdvjvk)
INSN_VEC(vaddwev_w_hu_h,   vdvjvk)
INSN_VEC(vaddwev_d_wu_w,   vdvjvk)
INSN_VEC(vaddwev_q_du_d,   vdvjvk)
INSN_VEC(vaddwod_h_bu_b,   vdvjvk)
INSN_VEC(vaddwod_w_hu_h,   vdvjvk)
INSN_VEC(vaddwod_d_wu_w,   vdvjvk)
INSN_VEC(vaddwod_q_du_d,   vdvjvk)
INSN_VEC(vsadd_b,          vdvjvk)
INSN_VEC(vsadd_h,          vdvjvk)
INSN_VEC(vsadd_w,          vdvjvk)
INSN_VEC(vsadd_d,          vdvjvk)
INSN_VEC(vssub_b,          vdvjvk)
INSN_VEC(vssub_h,          vdvjvk)
INSN_VEC(vssub_w,          vdvjvk)
INSN_VEC(vssub_d,          vdvjvk)
INSN_VEC(vsadd_bu,         vdvjvk)
INSN_VEC(vsadd_hu,         vdvjvk)
INSN_VEC(vsadd_wu,         vdvjvk)
INSN_VEC(vsadd_du,         vdvjvk)
INSN_VEC(vssub_bu,         vdvjvk)
INSN_VEC(vssub_hu,         vdvjvk)
INSN_VEC(vssub_wu,         vdvjvk)
INSN_VEC(vssub_du,         vdvjvk)
INSN_VEC(vhaddw_h_b,       vdvjvk)
INSN_VEC(vhaddw_w_h,       vdvjvk)
INSN_VEC(vhaddw_d_w,       vdvjvk)
INSN_VEC(vhaddw_q_d,       vdvjvk)
INSN_VEC(vhsubw_h_b,       vdvjvk)
INSN_VEC(vhsubw_w_h,       vdvjvk)
INSN_VEC(vhsubw_d_w,       vdvjvk)
INSN_VEC(vhsubw_q_d,       vdvjvk)
INSN_VEC(vhaddw_hu_bu,     vdvjvk)
INSN_VEC(vhaddw_wu_hu,     vdvjvk)
INSN_VEC(vhaddw_du_wu,     vdvjvk)
INSN_VEC(vhaddw_qu_du,     vdvjvk)
INSN_VEC(vhsubw_hu_bu,     vdvjvk)
INSN_VEC(vhsubw_wu_hu,     vdvjvk)
INSN_VEC(vhsubw_du_wu,     vdvjvk)
INSN_VEC(vhsubw_qu_du,     vdvjvk)
INSN_VEC(vadda_b,          vdvjvk)
INSN_VEC(vadda_h,          vdvjvk)
INSN_VEC(vadda_w,          vdvjvk)
INSN_VEC(vadda_d,          vdvjvk)
INSN_VEC(vabsd_b,          vdvjvk)
INSN_VEC(vabsd_h,          vdvjvk)
INSN_VEC(vabsd_w,          vdvjvk)
INSN_VEC(vabsd_d,          vdvjvk)
INSN_VEC(vabsd_bu,         vdvjvk)
INSN_VEC(vabsd_hu,         vdvjvk)
INSN_VEC(vabsd_wu,         vdvjvk)
INSN_VEC(vabsd_du,         vdvjvk)
INSN_VEC(vavg_b,           vdvjvk)
INSN_VEC(vavg_h,           vdvjvk)
INSN_VEC(vavg_w,           vdvjvk)
INSN_VEC(vavg_d,           vdvjvk)
INSN_VEC(vavg_bu,          vdvjvk)
INSN_VEC(vavg_hu,          vdvjvk)
INSN_VEC(vavg_wu,          vdvjvk)
INSN_VEC(vavg_du,          vdvjvk)
INSN_VEC(vavgr_b,          vdvjvk)
INSN_VEC(vavgr_h,          vdvjvk)
INSN_VEC(vavgr_w,          vdvjvk)
INSN_VEC(vavgr_d,          vdvjvk)
INSN_VEC(vavgr_bu,         vdvjvk)
INSN_VEC(vavgr_hu,         vdvjvk)
INSN_VEC(vavgr_wu,         vdvjvk)
INSN_VEC(vavgr_du,         vdvjvk)
INSN_VEC(vmax_b,           vdvjvk)
INSN_VEC(vmax_h,           vdvjvk)
INSN_VEC(vmax_w,           vdvjvk)
INSN_VEC(vmax_d,           vdvjvk)
INSN_VEC(vmin_b,           vdvjvk)
INSN_VEC(vmin_h,           vdvjvk)
INSN_VEC(vmin_w,           vdvjvk)
INSN_VEC(vmin_d,           vdvjvk)
INSN_VEC(vmax_bu,          vdvjvk)
INSN_VEC(vmax_hu,          vdvjvk)
INSN_VEC(vmax_wu,          vdvjvk)
INSN_VEC(vmax_du,          vdvjvk)
INSN_VEC(vmin_bu,          vdvjvk)
INSN_VEC(vmin_hu,          vdvjvk)
INSN_VEC(vmin_wu,          vdvjvk)
INSN_VEC(vmin_du,          vdvjvk)
INSN_VEC(vmul_b,           vdvjvk)
INSN_VEC(vmul_h,           vdvjvk)
INSN_VEC(vmul_w,           vdvjvk)
INSN_VEC(vmul_d,           vdvjvk)
INSN_VEC(vmuh_b,           vdvjvk)
INSN_VEC(vmuh_h,           vdvjvk)
INSN_VEC(vmuh_w,           vdvjvk)
INSN_VEC(vmuh_d,           vdvjvk)
INSN_VEC(vmuh_bu,          vdvjvk)
INSN_VEC(vmuh_hu,          vdvjvk)
INSN_VEC(vmuh_wu,          vdvjvk)
INSN_VEC(vmuh_du,          vdvjvk)
INSN_VEC(vmulwev_h_b,      vdvjvk)
INSN_VEC(vmulwev_w_h,      vdvjvk)
INSN_VEC(vmulwev_d_w,      vdvjvk)
INSN_VEC(vmulwev_q_d,      vdvjvk)
INSN_VEC(vmulwod_h_b,      vdvjvk)
INSN_VEC(vmulwod_w_h,      vdvjvk)
INSN_VEC(vmulwod_d_w,      vdvjvk)
INSN_VEC(vmulwod_q_d,      vdvjvk)
INSN_VEC(vmulwev_h_bu,     vdvjvk)
INSN_VEC(vmulwev_w_hu,     vdvjvk)
INSN_VEC(vmulwev_d_wu,     vdvjvk)
INSN_VEC(vmulwev_q_du,     vdvjvk)
INSN_VEC(vmulwod_h_bu,     vdvjvk)
INSN_VEC(vmulwod_w_hu,     vdvjvk)
INSN_VEC(vmulwod_d_wu,     vdvjvk)
INSN_VEC(vmulwod_q_du,     vdvjvk)
INSN_VEC(vmulwev_h_bu_b,   vdvjvk)
INSN_VEC(vmulwev_w_hu_h,   vdvjvk)
INSN_VEC(vmulwev_d_wu_w,   vdvjvk)
INSN_VEC(vmulwev_q_du_d,   vdvjvk)
INSN_VEC(vmulwod_h_bu_b,   vdvjvk)
INSN_VEC(vmulwod_w_hu_h,   vdvjvk)
INSN_VEC(vmulwod_d_wu_w,   vdvjvk)
INSN_VEC(vmulwod_q_du_d,   vdvjvk)
INSN_VEC(vmadd_b,          vdvjvk)
INSN_VEC(vmadd_h,          vdvjvk)
INSN_VEC(vmadd_w,          vdvjvk)
INSN_VEC(vmadd_d,          vdvjvk)
INSN_VEC(vmsub_b,          vdvjvk)
INSN_VEC(vmsub_h,          vdvjvk)
INSN_VEC(vmsub_w,          vdvjvk)
INSN_VEC(vmsub_d,          vdvjvk)
INSN_VEC(vmaddwev_h_b,     vdvjvk)
INSN_VEC(vmaddwev_w_h,     vdvjvk)
INSN_VEC(vmaddwev_d_w,     vdvjvk)
INSN_VEC(vmaddwev_q_d,     vdvjvk)
INSN_VEC(vmaddwod_h_b,     vdvjvk)
INSN_VEC(vmaddwod_w_h,     vdvjvk)
INSN_VEC(vmaddwod_d_w,     vdvjvk)
INSN_VEC(vmaddwod_q_d,     vdvjvk)
INSN_VEC(vmaddwev_h_bu,    vdvjvk)
INSN_VEC(vmaddwev_w_hu,    vdvjvk)
INSN_VEC(vmaddwev_d_wu,    vdvjvk)
INSN_VEC(vmaddwev_q_du,    vdvjvk)
INSN_VEC(vmaddwod_h_bu,    vdvjvk)
INSN_VEC(vmaddwod_w_hu,    vdvjvk)
INSN_VEC(vmaddwod_d_wu,    vdvjvk)
INSN_VEC(vmaddwod_q_du,    vdvjvk)
INSN_VEC(vmaddwev_h_bu_b,  vdvjvk)
INSN_VEC(vmaddwev_w_hu_h,  vdvjvk)
INSN_VEC(vmaddwev_d_wu_w,  vdvjvk)
INSN_VEC(vmaddwev_q_du_d,  vdvjvk)
INSN_VEC(vmaddwod_h_bu_b,  vdvjvk)
INSN_VEC(vmaddwod_w_hu_h,  vdvjvk)
INSN_VEC(vmaddwod_d_wu_w,  vdvjvk)
INSN_VEC(vmaddwod_q_du_d,  vdvjvk)
INSN_VEC(vdiv_b,           vdvjvk)
INSN_VEC(vdiv_h,           vdvjvk)
INSN_VEC(vdiv_w,           vdvjvk)
INSN_VEC(vdiv_d,           vdvjvk)
INSN_VEC(vmod_b,           vdvjvk)
INSN_VEC(vmod_h,           vdvjvk)
INSN_VEC(vmod_w,           vdvjvk)
INSN_VEC(vmod_d,           vdvjvk)
INSN_VEC(vdiv_bu,          vdvjvk)
INSN_VEC(vdiv_hu,          vdvjvk)
INSN_VEC(vdiv_wu,          vdvjvk)
INSN_VEC(vdiv_du,          vdvjvk)
INSN_VEC(vmod_bu,          vdvjvk)
INSN_VEC(vmod_hu,          vdvjvk)
INSN_VEC(vmod_wu,          vdvjvk)
INSN_VEC(vmod_du,          vdvjvk)
INSN_VEC(vsll_b,           vdvjvk)
INSN_VEC(vsll_h,           vdvjvk)
INSN_VEC(vsll_w,           vdvjvk)
INSN_VEC(vsll_d,           vdvjvk)
INSN_VEC(vsrl_b,           vdvjvk)
INSN_VEC(vsrl_h,           vdvjvk)
INSN_VEC(vsrl_w,           vdvjvk)
INSN_VEC(vsrl_d,           vdvjvk)
INSN_VEC(vsra_b,           vdvjvk)
INSN_VEC(vsra_h,           vdvjvk)
INSN_VEC(vsra_w,           vdvjvk)
INSN_VEC(vsra_d,           vdvjvk)
INSN_VEC(vrotr_b,          vdvjvk)
INSN_VEC(vrotr_h,          vdvjvk)
INSN_VEC(vrotr_w,          vdvjvk)
INSN_VEC(vrotr_d,          vdvjvk)
INSN_VEC(vsrlr_b,          vdvjvk)
INSN_VEC(vsrlr_h,          vdvjvk)
INSN_VEC(vsrlr_w,          vdvjvk)
INSN_VEC(vsrlr_d,          vdvjvk)
INSN_VEC(vsrar_b,          vdvjvk)
INSN_VEC(vsrar_h,          vdvjvk)
INSN_VEC(vsrar_w,          vdvjvk)
INSN_VEC(vsrar_d,          vdvjvk)
INSN_VEC(vsrln_b_h,        vdvjvk)
INSN_VEC(vsrln_h_w,        vdvjvk)
INSN_VEC(vsrln_w_d,        vdvjvk)
INSN_VEC(vsran_b_h,        vdvjvk)
INSN_VEC(vsran_h_w,        vdvjvk)
INSN_VEC(vsran_w_d,        vdvjvk)
INSN_VEC(vsrlrn_b_h,       vdvjvk)
INSN_VEC(vsrlrn_h_w,       vdvjvk)
INSN_VEC(vsrlrn_w_d,       vdvjvk)
INSN_VEC(vsrarn_b_h,       vdvjvk)
INSN_VEC(vsrarn_h_w,       vdvjvk)
INSN_VEC(vsrarn_w_d,       vdvjvk)
INSN_VEC(vssrln_b_h,       vdvjvk)
INSN_VEC(vssrln_h_w,       vdvjvk)
INSN_VEC(vssrln_w_d,       vdvjvk)
INSN_VEC(vssran_b_h,       vdvjvk)
INSN_VEC(vssran_h_w,       vdvjvk)
INSN_VEC(vssran_w_d,       vdvjvk)
INSN_VEC(vssrlrn_b_h,      vdvjvk)
INSN_VEC(vssrlrn_h_w,      vdvjvk)
INSN_VEC(vssrlrn_w_d,      vdvjvk)
INSN_VEC(vssrarn_b_h,      vdvjvk)
INSN_VEC(vssrarn_h_w,      vdvjvk)
INSN_VEC(vssrarn_w_d,      vdvjvk)
INSN_VEC(vssrln_bu_h,      vdvjvk)
INSN_VEC(vssrln_hu_w,      vdvjvk)
INSN_VEC(vssrln_wu_d,      vdvjvk)
INSN_VEC(vssran_bu_h,      vdvjvk)
INSN_VEC(vssran_hu_w,      vdvjvk)
INSN_VEC(vssran_wu_d,      vdvjvk)
INSN_VEC(vssrlrn_bu_h,     vdvjvk)
INSN_VEC(vssrlrn_hu_w,     vdvjvk)
INSN_VEC(vssrlrn_wu_d,     vdvjvk)
INSN_VEC(vssrarn_bu_h,     vdvjvk)
INSN_VEC(vssrarn_hu_w,     vdvjvk)
INSN_VEC(vssrarn_wu_d,     vdvjvk)
INSN_VEC(vbitclr_b,        vdvjvk)
INSN_VEC(vbitclr_h,        vdvjvk)
INSN_VEC(vbitclr_w,        vdvjvk)
INSN_VEC(vbitclr_d,        vdvjvk)
INSN_VEC(vbitset_b,        vdvjvk)
INSN_VEC(vbitset_h,        vdvjvk)
INSN_VEC(vbitset_w,        vdvjvk)
INSN_VEC(vbitset_d,        vdvjvk)
INSN_VEC(vbitrev_b,        vdvjvk)
INSN_VEC(vbitrev_h,        vdvjvk)
INSN_VEC(vbitrev_w,        vdvjvk)
INSN_VEC(vbitrev_d,        vdvjvk)
INSN_VEC(vpackev_b,        vdvjvk)
INSN_VEC(vpackev_h,        vdvjvk)
INSN_VEC(vpackev_w,        vdvjvk)
INSN_VEC(vpackev_d,        vdvjvk)
INSN_VEC(vpackod_b,        vdvjvk)
INSN_VEC(vpackod_h,        vdvjvk)
INSN_VEC(vpackod_w,        vdvjvk)
INSN_VEC(vpackod_d,        vdvjvk)
INSN_VEC(vilvl_b,          vdvjvk)
INSN_VEC(vilvl_h,          vdvjvk)
INSN_VEC(vilvl_w,          vdvjvk)
INSN_VEC(vilvl_d,          vdvjvk)
INSN_VEC(vilvh_b,          vdvjvk)
INSN_VEC(vilvh_h,          vdvjvk)
INSN_VEC(vilvh_w,          vdvjvk)
INSN_VEC(vilvh_d,          vdvjvk)
INSN_VEC(vpickev_b,        vdvjvk)
INSN_VEC(vpickev_h,        vdvjvk)
INSN_VEC(vpickev_w,        vdvjvk)
INSN_VEC(vpickev_d,        vdvjvk)
INSN_VEC(vpickod_b,        vdvjvk)
INSN_VEC(vpickod_h,        vdvjvk)
INSN_VEC(vpickod_w,        vdvjvk)
INSN_VEC(vpickod_d,        vdvjvk)
INSN_VEC(vreplve_b,        vdvjrk)
INSN_VEC(vreplve_h,        vdvjrk)
INSN_VEC(vreplve_w,        vdvjrk)
INSN_VEC(vreplve_d,        vdvjrk)
INSN_VEC(vand_v,           vdvjvk)
INSN_VEC(vor_v,            vdvjvk)
INSN_VEC(vxor_v,           vdvjvk)
INSN_VEC(vnor_v,           vdvjvk)
INSN_VEC(vandn_v,          vdvjvk)
INSN_VEC(vorn_v,           vdvjvk)
INSN_VEC(vfrstp_b,         vdvjvk)
INSN_VEC(vfrstp_h,         vdvjvk)
INSN_VEC(vadd_q,           vdvjvk)
INSN_VEC(vsub_q,           vdvjvk)
INSN_VEC(vsigncov_b,       vdvjvk)
INSN_VEC(vsigncov_h,       vdvjvk)
INSN_VEC(vsigncov_w,       vdvjvk)
INSN_VEC(vsigncov_d,       vdvjvk)
INSN_VEC(vfadd_s,          vdvjvk)
INSN_VEC(vfadd_d,          vdvjvk)
INSN_VEC(vfsub_s,          vdvjvk)
INSN_VEC(vfsub_d,          vdvjvk)
INSN_VEC(vfmul_s,          vdvjvk)
INSN_VEC(vfmul_d,          vdvjvk)
INSN_VEC(vfdiv_s,          vdvjvk)
INSN_VEC(vfdiv_d,          vdvjvk)
INSN_VEC(vfmax_s,          vdvjvk)
INSN_VEC(vfmax_d,          vdvjvk)
INSN_VEC(vfmin_s,          vdvjvk)
INSN_VEC(vfmin_d,          vdvjvk)
INSN_VEC(vfmaxa_s,         vdvjvk)
INSN_VEC(vfmaxa_d,         vdvjvk)
INSN_VEC(vfmina_s,         vdvjvk)
INSN_VEC(vfmina_d,         vdvjvk)
INSN_VEC(vfscaleb_s,       vdvjvk)
INSN_VEC(vfscaleb_d,       vdvjvk)
INSN_VEC(vfcvt_h_s,        vdvjvk)
INSN_VEC(vfcvt_s_d,        vdvjvk)
INSN_VEC(vffint_s_l,       vdvjvk)
INSN_VEC(vftint_w_d,       vdvjvk)
INSN_VEC(vftintrm_w_d,     vdvjvk)
INSN_VEC(vftintrp_w_d,     vdvjvk)
INSN_VEC(vftintrz_w_d,     vdvjvk)
INSN_VEC(vftintrne_w_d,    vdvjvk)
INSN_VEC(vshuf_h,          vdvjvk)
INSN_VEC(vshuf_w,          vdvjvk)
INSN_VEC(vshuf_d,          vdvjvk)
INSN_VEC(vseqi_b,          vdvjsi5)
INSN_VEC(vseqi_h,          vdvjsi5)
INSN_VEC(vseqi_w,          vdvjsi5)
INSN_VEC(vseqi_d,          vdvjsi5)
INSN_VEC(vslei_b,          vdvjsi5)
INSN_VEC(vslei_h,          vdvjsi5)
INSN_VEC(vslei_w,          vdvjsi5)
INSN_VEC(vslei_d,          vdvjsi5)
INSN_VEC(vslei_bu,         vdvjui5)
INSN_VEC(vslei_hu,         vdvjui5)
INSN_VEC(vslei_wu,         vdvjui5)
INSN_VEC(vslei_du,         vdvjui5)
INSN_VEC(vslti_b,          vdvjsi5)
INSN_VEC(vslti_h,          vdvjsi5)
INSN_VEC(vslti_w,          vdvjsi5)
INSN_VEC(vslti_d,          vdvjsi5)
INSN_VEC(vslti_bu,         vdvjui5)
INSN_VEC(vslti_hu,         vdvjui5)
INSN_VEC(vslti_wu,         vdvjui5)
INSN_VEC(vslti_du,         vdvjui5)
INSN_VEC(vaddi_bu,         vdvjui5)
INSN_VEC(vaddi_hu,         vdvjui5)
INSN_VEC(vaddi_wu,         vdvjui5)
INSN_VEC(vaddi_du,         vdvjui5)
INSN_VEC(vsubi_bu,         vdvjui5)
INSN_VEC(vsubi_hu,         vdvjui5)
INSN_VEC(vsubi_wu,         vdvjui5)
INSN_VEC(vsubi_du,         vdvjui5)
INSN_VEC(vbsll_v,          vdvjui5)
INSN_VEC(vbsrl_v,          vdvjui5)
INSN_VEC(vmaxi_b,          vdvjsi5)
INSN_VEC(vmaxi_h,          vdvjsi5)
INSN_VEC(vmaxi_w,          vdvjsi5)
INSN_VEC(vmaxi_d,          vdvjsi5)
INSN_VEC(vmini_b,          vdvjsi5)
INSN_VEC(vmini_h,          vdvjsi5)
INSN_VEC(vmini_w,          vdvjsi5)
INSN_VEC(vmini_d,          vdvjsi5)
INSN_VEC(vmaxi_bu,         vdvjui5)
INSN_VEC(vmaxi_hu,         vdvjui5)
INSN_VEC(vmaxi_wu,         vdvjui5)
INSN_VEC(vmaxi_du,         vdvjui5)
INSN_VEC(vmini_bu,         vdvjui5)
INSN_VEC(vmini_hu,         vdvjui5)
INSN_VEC(vmini_wu,         vdvjui5)
INSN_VEC(vmini_du,         vdvjui5)
INSN_VEC(vfrstpi_b,        vdvjui5)
INSN_VEC(vfrstpi_h,        vdvjui5)
INSN_VEC(vclrstri_v,       vdvjui5)
INSN_VEC(vmepatmsk_v,      vdmodeui5)
INSN_VEC(vclo_b,           vdvj)
INSN_VEC(vclo_h,           vdvj)
INSN_VEC(vclo_w,           vdvj)
INSN_VEC(vclo_d,           vdvj)
INSN_VEC(vclz_b,           vdvj)
INSN_VEC(vclz_h,           vdvj)
INSN_VEC(vclz_w,           vdvj)
INSN_VEC(vclz_d,           vdvj)
INSN_VEC(vpcnt_b,          vdvj)
INSN_VEC(vpcnt_h,          vdvj)
INSN_VEC(vpcnt_w,          vdvj)
INSN_VEC(vpcnt_d,          vdvj)
INSN_VEC(vneg_b,           vdvj)
INSN_VEC(vneg_h,           vdvj)
INSN_VEC(vneg_w,           vdvj)
INSN_VEC(vneg_d,           vdvj)
INSN_VEC(vmskltz_b,        vdvj)
INSN_VEC(vmskltz_h,        vdvj)
INSN_VEC(vmskltz_w,        vdvj)
INSN_VEC(vmskltz_d,        vdvj)
INSN_VEC(vmskgez_b,        vdvj)
INSN_VEC(vmsknz_b,         vdvj)
INSN_VEC(vmskcopy_b,       vdvj)
INSN_VEC(vmskfill_b,       vdvj)
INSN_VEC(vfrstm_b,         vdvj)
INSN_VEC(vfrstm_h,         vdvj)
INSN_VEC(vseteqz_v,        cdvj)
INSN_VEC(vsetnez_v,        cdvj)
INSN_VEC(vsetanyeqz_b,     cdvj)
INSN_VEC(vsetanyeqz_h,     cdvj)
INSN_VEC(vsetanyeqz_w,     cdvj)
INSN_VEC(vsetanyeqz_d,     cdvj)
INSN_VEC(vsetallnez_b,     cdvj)
INSN_VEC(vsetallnez_h,     cdvj)
INSN_VEC(vsetallnez_w,     cdvj)
INSN_VEC(vsetallnez_d,     cdvj)
INSN_VEC(vflogb_s,         vdvj)
INSN_VEC(vflogb_d,         vdvj)
INSN_VEC(vfclass_s,        vdvj)
INSN_VEC(vfclass_d,        vdvj)
INSN_VEC(vfsqrt_s,         vdvj)
INSN_VEC(vfsqrt_d,         vdvj)
INSN_VEC(vfrecip_s,        vdvj)
INSN_VEC(vfrecip_d,        vdvj)
INSN_VEC(vfrsqrt_s,        vdvj)
INSN_VEC(vfrsqrt_d,        vdvj)
INSN_VEC(vfrint_s,         vdvj)
INSN_VEC(vfrint_d,         vdvj)
INSN_VEC(vfrintrm_s,       vdvj)
INSN_VEC(vfrintrm_d,       vdvj)
INSN_VEC(vfrintrp_s,       vdvj)
INSN_VEC(vfrintrp_d,       vdvj)
INSN_VEC(vfrintrz_s,       vdvj)
INSN_VEC(vfrintrz_d,       vdvj)
INSN_VEC(vfrintrne_s,      vdvj)
INSN_VEC(vfrintrne_d,      vdvj)
INSN_VEC(vextl_w_b,        vdvj)
INSN_VEC(vextl_d_b,        vdvj)
INSN_VEC(vextl_d_h,        vdvj)
INSN_VEC(vextl_w_bu,       vdvj)
INSN_VEC(vextl_d_bu,       vdvj)
INSN_VEC(vextl_d_hu,       vdvj)
INSN_VEC(vhadd8_d_bu,      vdvj)
INSN_VEC(vhminpos_w_hu,    vdvj)
INSN_VEC(vhminpos_d_hu,    vdvj)
INSN_VEC(vhminpos_q_hu,    vdvj)
INSN_VEC(vclrtail_b,       vdvj)
INSN_VEC(vclrtail_h,       vdvj)
INSN_VEC(vfcvtl_s_h,       vdvj)
INSN_VEC(vfcvth_s_h,       vdvj)
INSN_VEC(vfcvtl_d_s,       vdvj)
INSN_VEC(vfcvth_d_s,       vdvj)
INSN_VEC(vffint_s_w,       vdvj)
INSN_VEC(vffint_s_wu,      vdvj)
INSN_VEC(vffint_d_l,       vdvj)
INSN_VEC(vffint_d_lu,      vdvj)
INSN_VEC(vffintl_d_w,      vdvj)
INSN_VEC(vffinth_d_w,      vdvj)
INSN_VEC(vftint_w_s,       vdvj)
INSN_VEC(vftint_l_d,       vdvj)
INSN_VEC(vftintrm_w_s,     vdvj)
INSN_VEC(vftintrm_l_d,     vdvj)
INSN_VEC(vftintrp_w_s,     vdvj)
INSN_VEC(vftintrp_l_d,     vdvj)
INSN_VEC(vftintrz_w_s,     vdvj)
INSN_VEC(vftintrz_l_d,     vdvj)
INSN_VEC(vftintrne_w_s,    vdvj)
INSN_VEC(vftintrne_l_d,    vdvj)
INSN_VEC(vftint_wu_s,      vdvj)
INSN_VEC(vftint_lu_d,      vdvj)
INSN_VEC(vftintrz_wu_s,    vdvj)
INSN_VEC(vftintrz_lu_d,    vdvj)
INSN_VEC(vftintl_l_s,      vdvj)
INSN_VEC(vftinth_l_s,      vdvj)
INSN_VEC(vftintrml_l_s,    vdvj)
INSN_VEC(vftintrmh_l_s,    vdvj)
INSN_VEC(vftintrpl_l_s,    vdvj)
INSN_VEC(vftintrph_l_s,    vdvj)
INSN_VEC(vftintrzl_l_s,    vdvj)
INSN_VEC(vftintrzh_l_s,    vdvj)
INSN_VEC(vftintrnel_l_s,   vdvj)
INSN_VEC(vftintrneh_l_s,   vdvj)
INSN_VEC(vexth_h_b,        vdvj)
INSN_VEC(vexth_w_h,        vdvj)
INSN_VEC(vexth_d_w,        vdvj)
INSN_VEC(vexth_q_d,        vdvj)
INSN_VEC(vexth_hu_bu,      vdvj)
INSN_VEC(vexth_wu_hu,      vdvj)
INSN_VEC(vexth_du_wu,      vdvj)
INSN_VEC(vexth_qu_du,      vdvj)
INSN_VEC(vreplgr2vr_b,     vdrj)
INSN_VEC(vreplgr2vr_h,     vdrj)
INSN_VEC(vreplgr2vr_w,     vdrj)
INSN_VEC(vreplgr2vr_d,     vdrj)
INSN_VEC(vrotri_b,         vdvjui3)
INSN_VEC(vrotri_h,         vdvjui4)
INSN_VEC(vrotri_w,         vdvjui5)
INSN_VEC(vrotri_d,         vdvjui6)
INSN_VEC(vsrlri_b,         vdvjui3)
INSN_VEC(vsrlri_h,         vdvjui4)
INSN_VEC(vsrlri_w,         vdvjui5)
INSN_VEC(vsrlri_d,         vdvjui6)
INSN_VEC(vsrari_b,         vdvjui3)
INSN_VEC(vsrari_h,         vdvjui4)
INSN_VEC(vsrari_w,         vdvjui5)
INSN_VEC(vsrari_d,         vdvjui6)
INSN_VEC(vinsgr2vr_b,      vdrjui4)
INSN_VEC(vinsgr2vr_h,      vdrjui3)
INSN_VEC(vinsgr2vr_w,      vdrjui2)
INSN_VEC(vinsgr2vr_d,      vdrjui1)
INSN_VEC(vpickve2gr_b,     rdvjui4)
INSN_VEC(vpickve2gr_h,     rdvjui3)
INSN_VEC(vpickve2gr_w,     rdvjui2)
INSN_VEC(vpickve2gr_d,     rdvjui1)
INSN_VEC(vpickve2gr_bu,    rdvjui4)
INSN_VEC(vpickve2gr_hu,    rdvjui3)
INSN_VEC(vpickve2gr_wu,    rdvjui2)
INSN_VEC(vpickve2gr_du,    rdvjui1)
INSN_VEC(vreplvei_b,       vdvjui4)
INSN_VEC(vreplvei_h,       vdvjui3)
INSN_VEC(vreplvei_w,       vdvjui2)
INSN_VEC(vreplvei_d,       vdvjui1)
INSN_VEC(vextrcoli_b,      vdvjui4)
INSN_VEC(vextrcoli_h,      vdvjui3)
INSN_VEC(vextrcoli_w,      vdvjui2)
INSN_VEC(vextrcoli_d,      vdvjui1)
INSN_VEC(vsllwil_h_b,      vdvjui3)
INSN_VEC(vsllwil_w_h,      vdvjui4)
INSN_VEC(vsllwil_d_w,      vdvjui5)
INSN_VEC(vextl_q_d,        vdvj)
INSN_VEC(vsllwil_hu_bu,    vdvjui3)
INSN_VEC(vsllwil_wu_hu,    vdvjui4)
INSN_VEC(vsllwil_du_wu,    vdvjui5)
INSN_VEC(vextl_qu_du,      vdvj)
INSN_VEC(vbitclri_b,       vdvjui3)
INSN_VEC(vbitclri_h,       vdvjui4)
INSN_VEC(vbitclri_w,       vdvjui5)
INSN_VEC(vbitclri_d,       vdvjui6)
INSN_VEC(vbitseti_b,       vdvjui3)
INSN_VEC(vbitseti_h,       vdvjui4)
INSN_VEC(vbitseti_w,       vdvjui5)
INSN_VEC(vbitseti_d,       vdvjui6)
INSN_VEC(vbitrevi_b,       vdvjui3)
INSN_VEC(vbitrevi_h,       vdvjui4)
INSN_VEC(vbitrevi_w,       vdvjui5)
INSN_VEC(vbitrevi_d,       vdvjui6)
INSN_VEC(vbstrc12i_b,      vdvjui3)
INSN_VEC(vbstrc12i_h,      vdvjui4)
INSN_VEC(vbstrc12i_w,      vdvjui5)
INSN_VEC(vbstrc12i_d,      vdvjui6)
INSN_VEC(vbstrc21i_b,      vdvjui3)
INSN_VEC(vbstrc21i_h,      vdvjui4)
INSN_VEC(vbstrc21i_w,      vdvjui5)
INSN_VEC(vbstrc21i_d,      vdvjui6)
INSN_VEC(vsat_b,           vdvjui3)
INSN_VEC(vsat_h,           vdvjui4)
INSN_VEC(vsat_w,           vdvjui5)
INSN_VEC(vsat_d,           vdvjui6)
INSN_VEC(vsat_bu,          vdvjui3)
INSN_VEC(vsat_hu,          vdvjui4)
INSN_VEC(vsat_wu,          vdvjui5)
INSN_VEC(vsat_du,          vdvjui6)
INSN_VEC(vslli_b,          vdvjui3)
INSN_VEC(vslli_h,          vdvjui4)
INSN_VEC(vslli_w,          vdvjui5)
INSN_VEC(vslli_d,          vdvjui6)
INSN_VEC(vsrli_b,          vdvjui3)
INSN_VEC(vsrli_h,          vdvjui4)
INSN_VEC(vsrli_w,          vdvjui5)
INSN_VEC(vsrli_d,          vdvjui6)
INSN_VEC(vsrai_b,          vdvjui3)
INSN_VEC(vsrai_h,          vdvjui4)
INSN_VEC(vsrai_w,          vdvjui5)
INSN_VEC(vsrai_d,          vdvjui6)
INSN_VEC(vsrlrneni_b_h,    vdvjui4)
INSN_VEC(vsrlrneni_h_w,    vdvjui5)
INSN_VEC(vsrlrneni_w_d,    vdvjui6)
INSN_VEC(vsrlrneni_d_q,    vdvjui7)
INSN_VEC(vsrarneni_b_h,    vdvjui4)
INSN_VEC(vsrarneni_h_w,    vdvjui5)
INSN_VEC(vsrarneni_w_d,    vdvjui6)
INSN_VEC(vsrarneni_d_q,    vdvjui7)
INSN_VEC(vsrlni_b_h,       vdvjui4)
INSN_VEC(vsrlni_h_w,       vdvjui5)
INSN_VEC(vsrlni_w_d,       vdvjui6)
INSN_VEC(vsrlni_d_q,       vdvjui7)
INSN_VEC(vsrlrni_b_h,      vdvjui4)
INSN_VEC(vsrlrni_h_w,      vdvjui5)
INSN_VEC(vsrlrni_w_d,      vdvjui6)
INSN_VEC(vsrlrni_d_q,      vdvjui7)
INSN_VEC(vssrlni_b_h,      vdvjui4)
INSN_VEC(vssrlni_h_w,      vdvjui5)
INSN_VEC(vssrlni_w_d,      vdvjui6)
INSN_VEC(vssrlni_d_q,      vdvjui7)
INSN_VEC(vssrlni_bu_h,     vdvjui4)
INSN_VEC(vssrlni_hu_w,     vdvjui5)
INSN_VEC(vssrlni_wu_d,     vdvjui6)
INSN_VEC(vssrlni_du_q,     vdvjui7)
INSN_VEC(vssrlrni_b_h,     vdvjui4)
INSN_VEC(vssrlrni_h_w,     vdvjui5)
INSN_VEC(vssrlrni_w_d,     vdvjui6)
INSN_VEC(vssrlrni_d_q,     vdvjui7)
INSN_VEC(vssrlrni_bu_h,    vdvjui4)
INSN_VEC(vssrlrni_hu_w,    vdvjui5)
INSN_VEC(vssrlrni_wu_d,    vdvjui6)
INSN_VEC(vssrlrni_du_q,    vdvjui7)
INSN_VEC(vsrani_b_h,       vdvjui4)
INSN_VEC(vsrani_h_w,       vdvjui5)
INSN_VEC(vsrani_w_d,       vdvjui6)
INSN_VEC(vsrani_d_q,       vdvjui7)
INSN_VEC(vsrarni_b_h,      vdvjui4)
INSN_VEC(vsrarni_h_w,      vdvjui5)
INSN_VEC(vsrarni_w_d,      vdvjui6)
INSN_VEC(vsrarni_d_q,      vdvjui7)
INSN_VEC(vssrani_b_h,      vdvjui4)
INSN_VEC(vssrani_h_w,      vdvjui5)
INSN_VEC(vssrani_w_d,      vdvjui6)
INSN_VEC(vssrani_d_q,      vdvjui7)
INSN_VEC(vssrani_bu_h,     vdvjui4)
INSN_VEC(vssrani_hu_w,     vdvjui5)
INSN_VEC(vssrani_wu_d,     vdvjui6)
INSN_VEC(vssrani_du_q,     vdvjui7)
INSN_VEC(vssrarni_b_h,     vdvjui4)
INSN_VEC(vssrarni_h_w,     vdvjui5)
INSN_VEC(vssrarni_w_d,     vdvjui6)
INSN_VEC(vssrarni_d_q,     vdvjui7)
INSN_VEC(vssrarni_bu_h,    vdvjui4)
INSN_VEC(vssrarni_hu_w,    vdvjui5)
INSN_VEC(vssrarni_wu_d,    vdvjui6)
INSN_VEC(vssrarni_du_q,    vdvjui7)
INSN_VEC(vssrlrneni_b_h,   vdvjui4)
INSN_VEC(vssrlrneni_h_w,   vdvjui5)
INSN_VEC(vssrlrneni_w_d,   vdvjui6)
INSN_VEC(vssrlrneni_d_q,   vdvjui7)
INSN_VEC(vssrlrneni_bu_h,  vdvjui4)
INSN_VEC(vssrlrneni_hu_w,  vdvjui5)
INSN_VEC(vssrlrneni_wu_d,  vdvjui6)
INSN_VEC(vssrlrneni_du_q,  vdvjui7)
INSN_VEC(vssrarneni_b_h,   vdvjui4)
INSN_VEC(vssrarneni_h_w,   vdvjui5)
INSN_VEC(vssrarneni_w_d,   vdvjui6)
INSN_VEC(vssrarneni_d_q,   vdvjui7)
INSN_VEC(vssrarneni_bu_h,  vdvjui4)
INSN_VEC(vssrarneni_hu_w,  vdvjui5)
INSN_VEC(vssrarneni_wu_d,  vdvjui6)
INSN_VEC(vssrarneni_du_q,  vdvjui7)
INSN_VEC(vextrins_d,       vdvjui8)
INSN_VEC(vextrins_w,       vdvjui8)
INSN_VEC(vextrins_h,       vdvjui8)
INSN_VEC(vextrins_b,       vdvjui8)
INSN_VEC(vshuf4i_b,        vdvjui8)
INSN_VEC(vshuf4i_h,        vdvjui8)
INSN_VEC(vshuf4i_w,        vdvjui8)
INSN_VEC(vshuf4i_d,        vdvjui8)
INSN_VEC(vshufi1_b,        vdvjui8)
INSN_VEC(vshufi2_b,        vdvjui8)
INSN_VEC(vshufi3_b,        vdvjui8)
INSN_VEC(vshufi4_b,        vdvjui8)
INSN_VEC(vshufi1_h,        vdvjui8)
INSN_VEC(vshufi2_h,        vdvjui8)
INSN_VEC(vseli_h,          vdvjui8)
INSN_VEC(vseli_w,          vdvjui8)
INSN_VEC(vseli_d,          vdvjui8)
INSN_VEC(vbitseli_b,       vdvjui8)
INSN_VEC(vbitmvzi_b,       vdvjui8)
INSN_VEC(vbitmvnzi_b,      vdvjui8)
INSN_VEC(vandi_b,          vdvjui8)
INSN_VEC(vori_b,           vdvjui8)
INSN_VEC(vxori_b,          vdvjui8)
INSN_VEC(vnori_b,          vdvjui8)
INSN_VEC(vldi,             vdi13)
INSN_VEC(vpermi_w,         vdvjui8)
INSN_VEC(xvseq_b,          xdxjxk)
INSN_VEC(xvseq_h,          xdxjxk)
INSN_VEC(xvseq_w,          xdxjxk)
INSN_VEC(xvseq_d,          xdxjxk)
INSN_VEC(xvsle_b,          xdxjxk)
INSN_VEC(xvsle_h,          xdxjxk)
INSN_VEC(xvsle_w,          xdxjxk)
INSN_VEC(xvsle_d,          xdxjxk)
INSN_VEC(xvsle_bu,         xdxjxk)
INSN_VEC(xvsle_hu,         xdxjxk)
INSN_VEC(xvsle_wu,         xdxjxk)
INSN_VEC(xvsle_du,         xdxjxk)
INSN_VEC(xvslt_b,          xdxjxk)
INSN_VEC(xvslt_h,          xdxjxk)
INSN_VEC(xvslt_w,          xdxjxk)
INSN_VEC(xvslt_d,          xdxjxk)
INSN_VEC(xvslt_bu,         xdxjxk)
INSN_VEC(xvslt_hu,         xdxjxk)
INSN_VEC(xvslt_wu,         xdxjxk)
INSN_VEC(xvslt_du,         xdxjxk)
INSN_VEC(xvadd_b,          xdxjxk)
INSN_VEC(xvadd_h,          xdxjxk)
INSN_VEC(xvadd_w,          xdxjxk)
INSN_VEC(xvadd_d,          xdxjxk)
INSN_VEC(xvsub_b,          xdxjxk)
INSN_VEC(xvsub_h,          xdxjxk)
INSN_VEC(xvsub_w,          xdxjxk)
INSN_VEC(xvsub_d,          xdxjxk)
INSN_VEC(xvaddwev_h_b,     xdxjxk)
INSN_VEC(xvaddwev_w_h,     xdxjxk)
INSN_VEC(xvaddwev_d_w,     xdxjxk)
INSN_VEC(xvaddwev_q_d,     xdxjxk)
INSN_VEC(xvsubwev_h_b,     xdxjxk)
INSN_VEC(xvsubwev_w_h,     xdxjxk)
INSN_VEC(xvsubwev_d_w,     xdxjxk)
INSN_VEC(xvsubwev_q_d,     xdxjxk)
INSN_VEC(xvaddwod_h_b,     xdxjxk)
INSN_VEC(xvaddwod_w_h,     xdxjxk)
INSN_VEC(xvaddwod_d_w,     xdxjxk)
INSN_VEC(xvaddwod_q_d,     xdxjxk)
INSN_VEC(xvsubwod_h_b,     xdxjxk)
INSN_VEC(xvsubwod_w_h,     xdxjxk)
INSN_VEC(xvsubwod_d_w,     xdxjxk)
INSN_VEC(xvsubwod_q_d,     xdxjxk)
INSN_VEC(xvaddwev_h_bu,    xdxjxk)
INSN_VEC(xvaddwev_w_hu,    xdxjxk)
INSN_VEC(xvaddwev_d_wu,    xdxjxk)
INSN_VEC(xvaddwev_q_du,    xdxjxk)
INSN_VEC(xvsubwev_h_bu,    xdxjxk)
INSN_VEC(xvsubwev_w_hu,    xdxjxk)
INSN_VEC(xvsubwev_d_wu,    xdxjxk)
INSN_VEC(xvsubwev_q_du,    xdxjxk)
INSN_VEC(xvaddwod_h_bu,    xdxjxk)
INSN_VEC(xvaddwod_w_hu,    xdxjxk)
INSN_VEC(xvaddwod_d_wu,    xdxjxk)
INSN_VEC(xvaddwod_q_du,    xdxjxk)
INSN_VEC(xvsubwod_h_bu,    xdxjxk)
INSN_VEC(xvsubwod_w_hu,    xdxjxk)
INSN_VEC(xvsubwod_d_wu,    xdxjxk)
INSN_VEC(xvsubwod_q_du,    xdxjxk)
INSN_VEC(xvaddwev_h_bu_b,  xdxjxk)
INSN_VEC(xvaddwev_w_hu_h,  xdxjxk)
INSN_VEC(xvaddwev_d_wu_w,  xdxjxk)
INSN_VEC(xvaddwev_q_du_d,  xdxjxk)
INSN_VEC(xvaddwod_h_bu_b,  xdxjxk)
INSN_VEC(xvaddwod_w_hu_h,  xdxjxk)
INSN_VEC(xvaddwod_d_wu_w,  xdxjxk)
INSN_VEC(xvaddwod_q_du_d,  xdxjxk)
INSN_VEC(xvsadd_b,         xdxjxk)
INSN_VEC(xvsadd_h,         xdxjxk)
INSN_VEC(xvsadd_w,         xdxjxk)
INSN_VEC(xvsadd_d,         xdxjxk)
INSN_VEC(xvssub_b,         xdxjxk)
INSN_VEC(xvssub_h,         xdxjxk)
INSN_VEC(xvssub_w,         xdxjxk)
INSN_VEC(xvssub_d,         xdxjxk)
INSN_VEC(xvsadd_bu,        xdxjxk)
INSN_VEC(xvsadd_hu,        xdxjxk)
INSN_VEC(xvsadd_wu,        xdxjxk)
INSN_VEC(xvsadd_du,        xdxjxk)
INSN_VEC(xvssub_bu,        xdxjxk)
INSN_VEC(xvssub_hu,        xdxjxk)
INSN_VEC(xvssub_wu,        xdxjxk)
INSN_VEC(xvssub_du,        xdxjxk)
INSN_VEC(xvhaddw_h_b,      xdxjxk)
INSN_VEC(xvhaddw_w_h,      xdxjxk)
INSN_VEC(xvhaddw_d_w,      xdxjxk)
INSN_VEC(xvhaddw_q_d,      xdxjxk)
INSN_VEC(xvhsubw_h_b,      xdxjxk)
INSN_VEC(xvhsubw_w_h,      xdxjxk)
INSN_VEC(xvhsubw_d_w,      xdxjxk)
INSN_VEC(xvhsubw_q_d,      xdxjxk)
INSN_VEC(xvhaddw_hu_bu,    xdxjxk)
INSN_VEC(xvhaddw_wu_hu,    xdxjxk)
INSN_VEC(xvhaddw_du_wu,    xdxjxk)
INSN_VEC(xvhaddw_qu_du,    xdxjxk)
INSN_VEC(xvhsubw_hu_bu,    xdxjxk)
INSN_VEC(xvhsubw_wu_hu,    xdxjxk)
INSN_VEC(xvhsubw_du_wu,    xdxjxk)
INSN_VEC(xvhsubw_qu_du,    xdxjxk)
INSN_VEC(xvadda_b,         xdxjxk)
INSN_VEC(xvadda_h,         xdxjxk)
INSN_VEC(xvadda_w,         xdxjxk)
INSN_VEC(xvadda_d,         xdxjxk)
INSN_VEC(xvabsd_b,         xdxjxk)
INSN_VEC(xvabsd_h,         xdxjxk)
INSN_VEC(xvabsd_w,         xdxjxk)
INSN_VEC(xvabsd_d,         xdxjxk)
INSN_VEC(xvabsd_bu,        xdxjxk)
INSN_VEC(xvabsd_hu,        xdxjxk)
INSN_VEC(xvabsd_wu,        xdxjxk)
INSN_VEC(xvabsd_du,        xdxjxk)
INSN_VEC(xvavg_b,          xdxjxk)
INSN_VEC(xvavg_h,          xdxjxk)
INSN_VEC(xvavg_w,          xdxjxk)
INSN_VEC(xvavg_d,          xdxjxk)
INSN_VEC(xvavg_bu,         xdxjxk)
INSN_VEC(xvavg_hu,         xdxjxk)
INSN_VEC(xvavg_wu,         xdxjxk)
INSN_VEC(xvavg_du,         xdxjxk)
INSN_VEC(xvavgr_b,         xdxjxk)
INSN_VEC(xvavgr_h,         xdxjxk)
INSN_VEC(xvavgr_w,         xdxjxk)
INSN_VEC(xvavgr_d,         xdxjxk)
INSN_VEC(xvavgr_bu,        xdxjxk)
INSN_VEC(xvavgr_hu,        xdxjxk)
INSN_VEC(xvavgr_wu,        xdxjxk)
INSN_VEC(xvavgr_du,        xdxjxk)
INSN_VEC(xvmax_b,          xdxjxk)
INSN_VEC(xvmax_h,          xdxjxk)
INSN_VEC(xvmax_w,          xdxjxk)
INSN_VEC(xvmax_d,          xdxjxk)
INSN_VEC(xvmin_b,          xdxjxk)
INSN_VEC(xvmin_h,          xdxjxk)
INSN_VEC(xvmin_w,          xdxjxk)
INSN_VEC(xvmin_d,          xdxjxk)
INSN_VEC(xvmax_bu,         xdxjxk)
INSN_VEC(xvmax_hu,         xdxjxk)
INSN_VEC(xvmax_wu,         xdxjxk)
INSN_VEC(xvmax_du,         xdxjxk)
INSN_VEC(xvmin_bu,         xdxjxk)
INSN_VEC(xvmin_hu,         xdxjxk)
INSN_VEC(xvmin_wu,         xdxjxk)
INSN_VEC(xvmin_du,         xdxjxk)
INSN_VEC(xvmul_b,          xdxjxk)
INSN_VEC(xvmul_h,          xdxjxk)
INSN_VEC(xvmul_w,          xdxjxk)
INSN_VEC(xvmul_d,          xdxjxk)
INSN_VEC(xvmuh_b,          xdxjxk)
INSN_VEC(xvmuh_h,          xdxjxk)
INSN_VEC(xvmuh_w,          xdxjxk)
INSN_VEC(xvmuh_d,          xdxjxk)
INSN_VEC(xvmuh_bu,         xdxjxk)
INSN_VEC(xvmuh_hu,         xdxjxk)
INSN_VEC(xvmuh_wu,         xdxjxk)
INSN_VEC(xvmuh_du,         xdxjxk)
INSN_VEC(xvmulwev_h_b,     xdxjxk)
INSN_VEC(xvmulwev_w_h,     xdxjxk)
INSN_VEC(xvmulwev_d_w,     xdxjxk)
INSN_VEC(xvmulwev_q_d,     xdxjxk)
INSN_VEC(xvmulwod_h_b,     xdxjxk)
INSN_VEC(xvmulwod_w_h,     xdxjxk)
INSN_VEC(xvmulwod_d_w,     xdxjxk)
INSN_VEC(xvmulwod_q_d,     xdxjxk)
INSN_VEC(xvmulwev_h_bu,    xdxjxk)
INSN_VEC(xvmulwev_w_hu,    xdxjxk)
INSN_VEC(xvmulwev_d_wu,    xdxjxk)
INSN_VEC(xvmulwev_q_du,    xdxjxk)
INSN_VEC(xvmulwod_h_bu,    xdxjxk)
INSN_VEC(xvmulwod_w_hu,    xdxjxk)
INSN_VEC(xvmulwod_d_wu,    xdxjxk)
INSN_VEC(xvmulwod_q_du,    xdxjxk)
INSN_VEC(xvmadd_b,         xdxjxk)
INSN_VEC(xvmadd_h,         xdxjxk)
INSN_VEC(xvmadd_w,         xdxjxk)
INSN_VEC(xvmadd_d,         xdxjxk)
INSN_VEC(xvmsub_b,         xdxjxk)
INSN_VEC(xvmsub_h,         xdxjxk)
INSN_VEC(xvmsub_w,         xdxjxk)
INSN_VEC(xvmsub_d,         xdxjxk)
INSN_VEC(xvmaddwev_h_b,    xdxjxk)
INSN_VEC(xvmaddwev_w_h,    xdxjxk)
INSN_VEC(xvmaddwev_d_w,    xdxjxk)
INSN_VEC(xvmaddwev_q_d,    xdxjxk)
INSN_VEC(xvmaddwod_h_b,    xdxjxk)
INSN_VEC(xvmaddwod_w_h,    xdxjxk)
INSN_VEC(xvmaddwod_d_w,    xdxjxk)
INSN_VEC(xvmaddwod_q_d,    xdxjxk)
INSN_VEC(xvmaddwev_h_bu,   xdxjxk)
INSN_VEC(xvmaddwev_w_hu,   xdxjxk)
INSN_VEC(xvmaddwev_d_wu,   xdxjxk)
INSN_VEC(xvmaddwev_q_du,   xdxjxk)
INSN_VEC(xvmaddwod_h_bu,   xdxjxk)
INSN_VEC(xvmaddwod_w_hu,   xdxjxk)
INSN_VEC(xvmaddwod_d_wu,   xdxjxk)
INSN_VEC(xvmaddwod_q_du,   xdxjxk)
INSN_VEC(xvmaddwev_h_bu_b, xdxjxk)
INSN_VEC(xvmaddwev_w_hu_h, xdxjxk)
INSN_VEC(xvmaddwev_d_wu_w, xdxjxk)
INSN_VEC(xvmaddwev_q_du_d, xdxjxk)
INSN_VEC(xvmaddwod_h_bu_b, xdxjxk)
INSN_VEC(xvmaddwod_w_hu_h, xdxjxk)
INSN_VEC(xvmaddwod_d_wu_w, xdxjxk)
INSN_VEC(xvmaddwod_q_du_d, xdxjxk)
INSN_VEC(xvdiv_b,          xdxjxk)
INSN_VEC(xvdiv_h,          xdxjxk)
INSN_VEC(xvdiv_w,          xdxjxk)
INSN_VEC(xvdiv_d,          xdxjxk)
INSN_VEC(xvmod_b,          xdxjxk)
INSN_VEC(xvmod_h,          xdxjxk)
INSN_VEC(xvmod_w,          xdxjxk)
INSN_VEC(xvmod_d,          xdxjxk)
INSN_VEC(xvdiv_bu,         xdxjxk)
INSN_VEC(xvdiv_hu,         xdxjxk)
INSN_VEC(xvdiv_wu,         xdxjxk)
INSN_VEC(xvdiv_du,         xdxjxk)
INSN_VEC(xvmod_bu,         xdxjxk)
INSN_VEC(xvmod_hu,         xdxjxk)
INSN_VEC(xvmod_wu,         xdxjxk)
INSN_VEC(xvmod_du,         xdxjxk)
INSN_VEC(xvsll_b,          xdxjxk)
INSN_VEC(xvsll_h,          xdxjxk)
INSN_VEC(xvsll_w,          xdxjxk)
INSN_VEC(xvsll_d,          xdxjxk)
INSN_VEC(xvsrl_b,          xdxjxk)
INSN_VEC(xvsrl_h,          xdxjxk)
INSN_VEC(xvsrl_w,          xdxjxk)
INSN_VEC(xvsrl_d,          xdxjxk)
INSN_VEC(xvsra_b,          xdxjxk)
INSN_VEC(xvsra_h,          xdxjxk)
INSN_VEC(xvsra_w,          xdxjxk)
INSN_VEC(xvsra_d,          xdxjxk)
INSN_VEC(xvrotr_b,         xdxjxk)
INSN_VEC(xvrotr_h,         xdxjxk)
INSN_VEC(xvrotr_w,         xdxjxk)
INSN_VEC(xvrotr_d,         xdxjxk)
INSN_VEC(xvsrlr_b,         xdxjxk)
INSN_VEC(xvsrlr_h,         xdxjxk)
INSN_VEC(xvsrlr_w,         xdxjxk)
INSN_VEC(xvsrlr_d,         xdxjxk)
INSN_VEC(xvsrar_b,         xdxjxk)
INSN_VEC(xvsrar_h,         xdxjxk)
INSN_VEC(xvsrar_w,         xdxjxk)
INSN_VEC(xvsrar_d,         xdxjxk)
INSN_VEC(xvsrln_b_h,       xdxjxk)
INSN_VEC(xvsrln_h_w,       xdxjxk)
INSN_VEC(xvsrln_w_d,       xdxjxk)
INSN_VEC(xvsran_b_h,       xdxjxk)
INSN_VEC(xvsran_h_w,       xdxjxk)
INSN_VEC(xvsran_w_d,       xdxjxk)
INSN_VEC(xvsrlrn_b_h,      xdxjxk)
INSN_VEC(xvsrlrn_h_w,      xdxjxk)
INSN_VEC(xvsrlrn_w_d,      xdxjxk)
INSN_VEC(xvsrarn_b_h,      xdxjxk)
INSN_VEC(xvsrarn_h_w,      xdxjxk)
INSN_VEC(xvsrarn_w_d,      xdxjxk)
INSN_VEC(xvssrln_b_h,      xdxjxk)
INSN_VEC(xvssrln_h_w,      xdxjxk)
INSN_VEC(xvssrln_w_d,      xdxjxk)
INSN_VEC(xvssran_b_h,      xdxjxk)
INSN_VEC(xvssran_h_w,      xdxjxk)
INSN_VEC(xvssran_w_d,      xdxjxk)
INSN_VEC(xvssrlrn_b_h,     xdxjxk)
INSN_VEC(xvssrlrn_h_w,     xdxjxk)
INSN_VEC(xvssrlrn_w_d,     xdxjxk)
INSN_VEC(xvssrarn_b_h,     xdxjxk)
INSN_VEC(xvssrarn_h_w,     xdxjxk)
INSN_VEC(xvssrarn_w_d,     xdxjxk)
INSN_VEC(xvssrln_bu_h,     xdxjxk)
INSN_VEC(xvssrln_hu_w,     xdxjxk)
INSN_VEC(xvssrln_wu_d,     xdxjxk)
INSN_VEC(xvssran_bu_h,     xdxjxk)
INSN_VEC(xvssran_hu_w,     xdxjxk)
INSN_VEC(xvssran_wu_d,     xdxjxk)
INSN_VEC(xvssrlrn_bu_h,    xdxjxk)
INSN_VEC(xvssrlrn_hu_w,    xdxjxk)
INSN_VEC(xvssrlrn_wu_d,    xdxjxk)
INSN_VEC(xvssrarn_bu_h,    xdxjxk)
INSN_VEC(xvssrarn_hu_w,    xdxjxk)
INSN_VEC(xvssrarn_wu_d,    xdxjxk)
INSN_VEC(xvbitclr_b,       xdxjxk)
INSN_VEC(xvbitclr_h,       xdxjxk)
INSN_VEC(xvbitclr_w,       xdxjxk)
INSN_VEC(xvbitclr_d,       xdxjxk)
INSN_VEC(xvbitset_b,       xdxjxk)
INSN_VEC(xvbitset_h,       xdxjxk)
INSN_VEC(xvbitset_w,       xdxjxk)
INSN_VEC(xvbitset_d,       xdxjxk)
INSN_VEC(xvbitrev_b,       xdxjxk)
INSN_VEC(xvbitrev_h,       xdxjxk)
INSN_VEC(xvbitrev_w,       xdxjxk)
INSN_VEC(xvbitrev_d,       xdxjxk)
INSN_VEC(xvpackev_b,       xdxjxk)
INSN_VEC(xvpackev_h,       xdxjxk)
INSN_VEC(xvpackev_w,       xdxjxk)
INSN_VEC(xvpackev_d,       xdxjxk)
INSN_VEC(xvpackod_b,       xdxjxk)
INSN_VEC(xvpackod_h,       xdxjxk)
INSN_VEC(xvpackod_w,       xdxjxk)
INSN_VEC(xvpackod_d,       xdxjxk)
INSN_VEC(xvilvl_b,         xdxjxk)
INSN_VEC(xvilvl_h,         xdxjxk)
INSN_VEC(xvilvl_w,         xdxjxk)
INSN_VEC(xvilvl_d,         xdxjxk)
INSN_VEC(xvilvh_b,         xdxjxk)
INSN_VEC(xvilvh_h,         xdxjxk)
INSN_VEC(xvilvh_w,         xdxjxk)
INSN_VEC(xvilvh_d,         xdxjxk)
INSN_VEC(xvpickev_b,       xdxjxk)
INSN_VEC(xvpickev_h,       xdxjxk)
INSN_VEC(xvpickev_w,       xdxjxk)
INSN_VEC(xvpickev_d,       xdxjxk)
INSN_VEC(xvpickod_b,       xdxjxk)
INSN_VEC(xvpickod_h,       xdxjxk)
INSN_VEC(xvpickod_w,       xdxjxk)
INSN_VEC(xvpickod_d,       xdxjxk)
INSN_VEC(xvreplve_b,       xdxjrk)
INSN_VEC(xvreplve_h,       xdxjrk)
INSN_VEC(xvreplve_w,       xdxjrk)
INSN_VEC(xvreplve_d,       xdxjrk)
INSN_VEC(xvand_v,          xdxjxk)
INSN_VEC(xvor_v,           xdxjxk)
INSN_VEC(xvxor_v,          xdxjxk)
INSN_VEC(xvnor_v,          xdxjxk)
INSN_VEC(xvandn_v,         xdxjxk)
INSN_VEC(xvorn_v,          xdxjxk)
INSN_VEC(xvfrstp_b,        xdxjxk)
INSN_VEC(xvfrstp_h,        xdxjxk)
INSN_VEC(xvadd_q,          xdxjxk)
INSN_VEC(xvsub_q,          xdxjxk)
INSN_VEC(xvsigncov_b,      xdxjxk)
INSN_VEC(xvsigncov_h,      xdxjxk)
INSN_VEC(xvsigncov_w,      xdxjxk)
INSN_VEC(xvsigncov_d,      xdxjxk)
INSN_VEC(xvfadd_s,         xdxjxk)
INSN_VEC(xvfadd_d,         xdxjxk)
INSN_VEC(xvfsub_s,         xdxjxk)
INSN_VEC(xvfsub_d,         xdxjxk)
INSN_VEC(xvfaddsub_s,      xdxjxk)
INSN_VEC(xvfaddsub_d,      xdxjxk)
INSN_VEC(xvfsubadd_s,      xdxjxk)
INSN_VEC(xvfsubadd_d,      xdxjxk)
INSN_VEC(xvfmul_s,         xdxjxk)
INSN_VEC(xvfmul_d,         xdxjxk)
INSN_VEC(xvfdiv_s,         xdxjxk)
INSN_VEC(xvfdiv_d,         xdxjxk)
INSN_VEC(xvfmax_s,         xdxjxk)
INSN_VEC(xvfmax_d,         xdxjxk)
INSN_VEC(xvfmin_s,         xdxjxk)
INSN_VEC(xvfmin_d,         xdxjxk)
INSN_VEC(xvfmaxa_s,        xdxjxk)
INSN_VEC(xvfmaxa_d,        xdxjxk)
INSN_VEC(xvfmina_s,        xdxjxk)
INSN_VEC(xvfmina_d,        xdxjxk)
INSN_VEC(xvfscaleb_s,      xdxjxk)
INSN_VEC(xvfscaleb_d,      xdxjxk)
INSN_VEC(xvfcvt_h_s,       xdxjxk)
INSN_VEC(xvfcvt_s_d,       xdxjxk)
INSN_VEC(xvffint_s_l,      xdxjxk)
INSN_VEC(xvftint_w_d,      xdxjxk)
INSN_VEC(xvftintrm_w_d,    xdxjxk)
INSN_VEC(xvftintrp_w_d,    xdxjxk)
INSN_VEC(xvftintrz_w_d,    xdxjxk)
INSN_VEC(xvftintrne_w_d,   xdxjxk)
INSN_VEC(xvhadd4_h_bu,     xdxjxk)
INSN_VEC(xvshuf4_w,        xdxjxk)
INSN_VEC(xvshuf2_d,        xdxjxk)
INSN_VEC(xvpmul_w,         xdxjxk)
INSN_VEC(xvpmul_d,         xdxjxk)
INSN_VEC(xvpmuh_w,         xdxjxk)
INSN_VEC(xvpmuh_d,         xdxjxk)
INSN_VEC(xvpmulacc_w,      xdxjxk)
INSN_VEC(xvpmulacc_d,      xdxjxk)
INSN_VEC(xvpmuhacc_w,      xdxjxk)
INSN_VEC(xvpmuhacc_d,      xdxjxk)
INSN_VEC(xvpmulwl_h_b,     xdxjxk)
INSN_VEC(xvpmulwl_w_h,     xdxjxk)
INSN_VEC(xvpmulwl_d_w,     xdxjxk)
INSN_VEC(xvpmulwl_q_d,     xdxjxk)
INSN_VEC(xvpmulwh_h_b,     xdxjxk)
INSN_VEC(xvpmulwh_w_h,     xdxjxk)
INSN_VEC(xvpmulwh_d_w,     xdxjxk)
INSN_VEC(xvpmulwh_q_d,     xdxjxk)
INSN_VEC(xvpmaddwl_h_b,    xdxjxk)
INSN_VEC(xvpmaddwl_w_h,    xdxjxk)
INSN_VEC(xvpmaddwl_d_w,    xdxjxk)
INSN_VEC(xvpmaddwl_q_d,    xdxjxk)
INSN_VEC(xvpmaddwh_h_b,    xdxjxk)
INSN_VEC(xvpmaddwh_w_h,    xdxjxk)
INSN_VEC(xvpmaddwh_d_w,    xdxjxk)
INSN_VEC(xvpmaddwh_q_d,    xdxjxk)
INSN_VEC(xvpdp2_q_d,       xdxjxk)
INSN_VEC(xvpdp2add_q_d,    xdxjxk)
INSN_VEC(xvcdp4_re_d_h,    xdxjxk)
INSN_VEC(xvcdp4_im_d_h,    xdxjxk)
INSN_VEC(xvcdp4add_re_d_h, xdxjxk)
INSN_VEC(xvcdp4add_im_d_h, xdxjxk)
INSN_VEC(xvcdp2_re_q_w,    xdxjxk)
INSN_VEC(xvcdp2_im_q_w,    xdxjxk)
INSN_VEC(xvcdp2add_re_q_w, xdxjxk)
INSN_VEC(xvcdp2add_im_q_w, xdxjxk)
INSN_VEC(xvsignsel_w,      xdxjxk)
INSN_VEC(xvsignsel_d,      xdxjxk)
INSN_VEC(xvshuf_h,         xdxjxk)
INSN_VEC(xvshuf_w,         xdxjxk)
INSN_VEC(xvshuf_d,         xdxjxk)
INSN_VEC(xvperm_w,         xdxjxk)
INSN_VEC(xvseqi_b,         xdxjsi5)
INSN_VEC(xvseqi_h,         xdxjsi5)
INSN_VEC(xvseqi_w,         xdxjsi5)
INSN_VEC(xvseqi_d,         xdxjsi5)
INSN_VEC(xvslei_b,         xdxjsi5)
INSN_VEC(xvslei_h,         xdxjsi5)
INSN_VEC(xvslei_w,         xdxjsi5)
INSN_VEC(xvslei_d,         xdxjsi5)
INSN_VEC(xvslei_bu,        xdxjui5)
INSN_VEC(xvslei_hu,        xdxjui5)
INSN_VEC(xvslei_wu,        xdxjui5)
INSN_VEC(xvslei_du,        xdxjui5)
INSN_VEC(xvslti_b,         xdxjsi5)
INSN_VEC(xvslti_h,         xdxjsi5)
INSN_VEC(xvslti_w,         xdxjsi5)
INSN_VEC(xvslti_d,         xdxjsi5)
INSN_VEC(xvslti_bu,        xdxjui5)
INSN_VEC(xvslti_hu,        xdxjui5)
INSN_VEC(xvslti_wu,        xdxjui5)
INSN_VEC(xvslti_du,        xdxjui5)
INSN_VEC(xvaddi_bu,        xdxjui5)
INSN_VEC(xvaddi_hu,        xdxjui5)
INSN_VEC(xvaddi_wu,        xdxjui5)
INSN_VEC(xvaddi_du,        xdxjui5)
INSN_VEC(xvsubi_bu,        xdxjui5)
INSN_VEC(xvsubi_hu,        xdxjui5)
INSN_VEC(xvsubi_wu,        xdxjui5)
INSN_VEC(xvsubi_du,        xdxjui5)
INSN_VEC(xvbsll_v,         xdxjui5)
INSN_VEC(xvbsrl_v,         xdxjui5)
INSN_VEC(xvmaxi_b,         xdxjsi5)
INSN_VEC(xvmaxi_h,         xdxjsi5)
INSN_VEC(xvmaxi_w,         xdxjsi5)
INSN_VEC(xvmaxi_d,         xdxjsi5)
INSN_VEC(xvmini_b,         xdxjsi5)
INSN_VEC(xvmini_h,         xdxjsi5)
INSN_VEC(xvmini_w,         xdxjsi5)
INSN_VEC(xvmini_d,         xdxjsi5)
INSN_VEC(xvmaxi_bu,        xdxjui5)
INSN_VEC(xvmaxi_hu,        xdxjui5)
INSN_VEC(xvmaxi_wu,        xdxjui5)
INSN_VEC(xvmaxi_du,        xdxjui5)
INSN_VEC(xvmini_bu,        xdxjui5)
INSN_VEC(xvmini_hu,        xdxjui5)
INSN_VEC(xvmini_wu,        xdxjui5)
INSN_VEC(xvmini_du,        xdxjui5)
INSN_VEC(xvrandsigni_b,    xdxjui5)
INSN_VEC(xvrandsigni_h,    xdxjui5)
INSN_VEC(xvrorsigni_b,     xdxjui5)
INSN_VEC(xvrorsigni_h,     xdxjui5)
INSN_VEC(xvfrstpi_b,       xdxjui5)
INSN_VEC(xvfrstpi_h,       xdxjui5)
INSN_VEC(xvclrstri_v,      xdxjui5)
INSN_VEC(xvmepatmsk_v,     xdmodeui5)
INSN_VEC(xvclo_b,          xdxj)
INSN_VEC(xvclo_h,          xdxj)
INSN_VEC(xvclo_w,          xdxj)
INSN_VEC(xvclo_d,          xdxj)
INSN_VEC(xvclz_b,          xdxj)
INSN_VEC(xvclz_h,          xdxj)
INSN_VEC(xvclz_w,          xdxj)
INSN_VEC(xvclz_d,          xdxj)
INSN_VEC(xvpcnt_b,         xdxj)
INSN_VEC(xvpcnt_h,         xdxj)
INSN_VEC(xvpcnt_w,         xdxj)
INSN_VEC(xvpcnt_d,         xdxj)
INSN_VEC(xvneg_b,          xdxj)
INSN_VEC(xvneg_h,          xdxj)
INSN_VEC(xvneg_w,          xdxj)
INSN_VEC(xvneg_d,          xdxj)
INSN_VEC(xvmskltz_b,       xdxj)
INSN_VEC(xvmskltz_h,       xdxj)
INSN_VEC(xvmskltz_w,       xdxj)
INSN_VEC(xvmskltz_d,       xdxj)
INSN_VEC(xvmskgez_b,       xdxj)
INSN_VEC(xvmsknz_b,        xdxj)
INSN_VEC(xvmskcopy_b,      xdxj)
INSN_VEC(xvmskfill_b,      xdxj)
INSN_VEC(xvfrstm_b,        xdxj)
INSN_VEC(xvfrstm_h,        xdxj)
INSN_VEC(xvseteqz_v,       cdxj)
INSN_VEC(xvsetnez_v,       cdxj)
INSN_VEC(xvsetanyeqz_b,    cdxj)
INSN_VEC(xvsetanyeqz_h,    cdxj)
INSN_VEC(xvsetanyeqz_w,    cdxj)
INSN_VEC(xvsetanyeqz_d,    cdxj)
INSN_VEC(xvsetallnez_b,    cdxj)
INSN_VEC(xvsetallnez_h,    cdxj)
INSN_VEC(xvsetallnez_w,    cdxj)
INSN_VEC(xvsetallnez_d,    cdxj)
INSN_VEC(xvflogb_s,        xdxj)
INSN_VEC(xvflogb_d,        xdxj)
INSN_VEC(xvfclass_s,       xdxj)
INSN_VEC(xvfclass_d,       xdxj)
INSN_VEC(xvfsqrt_s,        xdxj)
INSN_VEC(xvfsqrt_d,        xdxj)
INSN_VEC(xvfrecip_s,       xdxj)
INSN_VEC(xvfrecip_d,       xdxj)
INSN_VEC(xvfrsqrt_s,       xdxj)
INSN_VEC(xvfrsqrt_d,       xdxj)
INSN_VEC(xvfrint_s,        xdxj)
INSN_VEC(xvfrint_d,        xdxj)
INSN_VEC(xvfrintrm_s,      xdxj)
INSN_VEC(xvfrintrm_d,      xdxj)
INSN_VEC(xvfrintrp_s,      xdxj)
INSN_VEC(xvfrintrp_d,      xdxj)
INSN_VEC(xvfrintrz_s,      xdxj)
INSN_VEC(xvfrintrz_d,      xdxj)
INSN_VEC(xvfrintrne_s,     xdxj)
INSN_VEC(xvfrintrne_d,     xdxj)
INSN_VEC(xvextl_w_b,       xdxj)
INSN_VEC(xvextl_d_b,       xdxj)
INSN_VEC(xvextl_d_h,       xdxj)
INSN_VEC(xvextl_w_bu,      xdxj)
INSN_VEC(xvextl_d_bu,      xdxj)
INSN_VEC(xvextl_d_hu,      xdxj)
INSN_VEC(xvhadd8_d_bu,     xdxj)
INSN_VEC(xvhminpos_w_hu,   xdxj)
INSN_VEC(xvhminpos_d_hu,   xdxj)
INSN_VEC(xvhminpos_q_hu,   xdxj)
INSN_VEC(xvclrtail_b,      xdxj)
INSN_VEC(xvclrtail_h,      xdxj)
INSN_VEC(xvfcvtl_s_h,      xdxj)
INSN_VEC(xvfcvth_s_h,      xdxj)
INSN_VEC(xvfcvtl_d_s,      xdxj)
INSN_VEC(xvfcvth_d_s,      xdxj)
INSN_VEC(xvffint_s_w,      xdxj)
INSN_VEC(xvffint_s_wu,     xdxj)
INSN_VEC(xvffint_d_l,      xdxj)
INSN_VEC(xvffint_d_lu,     xdxj)
INSN_VEC(xvffintl_d_w,     xdxj)
INSN_VEC(xvffinth_d_w,     xdxj)
INSN_VEC(xvftint_w_s,      xdxj)
INSN_VEC(xvftint_l_d,      xdxj)
INSN_VEC(xvftintrm_w_s,    xdxj)
INSN_VEC(xvftintrm_l_d,    xdxj)
INSN_VEC(xvftintrp_w_s,    xdxj)
INSN_VEC(xvftintrp_l_d,    xdxj)
INSN_VEC(xvftintrz_w_s,    xdxj)
INSN_VEC(xvftintrz_l_d,    xdxj)
INSN_VEC(xvftintrne_w_s,   xdxj)
INSN_VEC(xvftintrne_l_d,   xdxj)
INSN_VEC(xvftint_wu_s,     xdxj)
INSN_VEC(xvftint_lu_d,     xdxj)
INSN_VEC(xvftintrz_wu_s,   xdxj)
INSN_VEC(xvftintrz_lu_d,   xdxj)
INSN_VEC(xvftintl_l_s,     xdxj)
INSN_VEC(xvftinth_l_s,     xdxj)
INSN_VEC(xvftintrml_l_s,   xdxj)
INSN_VEC(xvftintrmh_l_s,   xdxj)
INSN_VEC(xvftintrpl_l_s,   xdxj)
INSN_VEC(xvftintrph_l_s,   xdxj)
INSN_VEC(xvftintrzl_l_s,   xdxj)
INSN_VEC(xvftintrzh_l_s,   xdxj)
INSN_VEC(xvftintrnel_l_s,  xdxj)
INSN_VEC(xvftintrneh_l_s,  xdxj)
INSN_VEC(xvexth_h_b,       xdxj)
INSN_VEC(xvexth_w_h,       xdxj)
INSN_VEC(xvexth_d_w,       xdxj)
INSN_VEC(xvexth_q_d,       xdxj)
INSN_VEC(xvexth_hu_bu,     xdxj)
INSN_VEC(xvexth_wu_hu,     xdxj)
INSN_VEC(xvexth_du_wu,     xdxj)
INSN_VEC(xvexth_qu_du,     xdxj)
INSN_VEC(xvreplgr2vr_b,    xdrj)
INSN_VEC(xvreplgr2vr_h,    xdrj)
INSN_VEC(xvreplgr2vr_w,    xdrj)
INSN_VEC(xvreplgr2vr_d,    xdrj)
INSN_VEC(vext2xv_h_b,      xdxj)
INSN_VEC(vext2xv_w_b,      xdxj)
INSN_VEC(vext2xv_d_b,      xdxj)
INSN_VEC(vext2xv_w_h,      xdxj)
INSN_VEC(vext2xv_d_h,      xdxj)
INSN_VEC(vext2xv_d_w,      xdxj)
INSN_VEC(vext2xv_hu_bu,    xdxj)
INSN_VEC(vext2xv_wu_bu,    xdxj)
INSN_VEC(vext2xv_du_bu,    xdxj)
INSN_VEC(vext2xv_wu_hu,    xdxj)
INSN_VEC(vext2xv_du_hu,    xdxj)
INSN_VEC(vext2xv_du_wu,    xdxj)
INSN_VEC(xvhseli_d,        xdxjui5)
INSN_VEC(xvrotri_b,        xdxjui3)
INSN_VEC(xvrotri_h,        xdxjui4)
INSN_VEC(xvrotri_w,        xdxjui5)
INSN_VEC(xvrotri_d,        xdxjui6)
INSN_VEC(xvsrlri_b,        xdxjui3)
INSN_VEC(xvsrlri_h,        xdxjui4)
INSN_VEC(xvsrlri_w,        xdxjui5)
INSN_VEC(xvsrlri_d,        xdxjui6)
INSN_VEC(xvsrari_b,        xdxjui3)
INSN_VEC(xvsrari_h,        xdxjui4)
INSN_VEC(xvsrari_w,        xdxjui5)
INSN_VEC(xvsrari_d,        xdxjui6)
INSN_VEC(xvinsgr2vr_w,     xdrjui3)
INSN_VEC(xvinsgr2vr_d,     xdrjui2)
INSN_VEC(xvpickve2gr_w,    rdxjui3)
INSN_VEC(xvpickve2gr_d,    rdxjui2)
INSN_VEC(xvpickve2gr_wu,   rdxjui3)
INSN_VEC(xvpickve2gr_du,   rdxjui2)
INSN_VEC(xvrepl128vei_b,   xdxjui4)
INSN_VEC(xvrepl128vei_h,   xdxjui3)
INSN_VEC(xvrepl128vei_w,   xdxjui2)
INSN_VEC(xvrepl128vei_d,   xdxjui1)
INSN_VEC(xvextrcoli_b,     xdxjui4)
INSN_VEC(xvextrcoli_h,     xdxjui3)
INSN_VEC(xvextrcoli_w,     xdxjui2)
INSN_VEC(xvextrcoli_d,     xdxjui1)
INSN_VEC(xvinsve0_w,       xdxjui3)
INSN_VEC(xvinsve0_d,       xdxjui2)
INSN_VEC(xvpickve_w,       xdxjui3)
INSN_VEC(xvpickve_d,       xdxjui2)
INSN_VEC(xvreplve0_b,      xdxj)
INSN_VEC(xvreplve0_h,      xdxj)
INSN_VEC(xvreplve0_w,      xdxj)
INSN_VEC(xvreplve0_d,      xdxj)
INSN_VEC(xvreplve0_q,      xdxj)
INSN_VEC(xvsllwil_h_b,     xdxjui3)
INSN_VEC(xvsllwil_w_h,     xdxjui4)
INSN_VEC(xvsllwil_d_w,     xdxjui5)
INSN_VEC(xvextl_q_d,       xdxj)
INSN_VEC(xvsllwil_hu_bu,   xdxjui3)
INSN_VEC(xvsllwil_wu_hu,   xdxjui4)
INSN_VEC(xvsllwil_du_wu,   xdxjui5)
INSN_VEC(xvextl_qu_du,     xdxj)
INSN_VEC(xvbitclri_b,      xdxjui3)
INSN_VEC(xvbitclri_h,      xdxjui4)
INSN_VEC(xvbitclri_w,      xdxjui5)
INSN_VEC(xvbitclri_d,      xdxjui6)
INSN_VEC(xvbitseti_b,      xdxjui3)
INSN_VEC(xvbitseti_h,      xdxjui4)
INSN_VEC(xvbitseti_w,      xdxjui5)
INSN_VEC(xvbitseti_d,      xdxjui6)
INSN_VEC(xvbitrevi_b,      xdxjui3)
INSN_VEC(xvbitrevi_h,      xdxjui4)
INSN_VEC(xvbitrevi_w,      xdxjui5)
INSN_VEC(xvbitrevi_d,      xdxjui6)
INSN_VEC(xvbstrc12i_b,     xdxjui3)
INSN_VEC(xvbstrc12i_h,     xdxjui4)
INSN_VEC(xvbstrc12i_w,     xdxjui5)
INSN_VEC(xvbstrc12i_d,     xdxjui6)
INSN_VEC(xvbstrc21i_b,     xdxjui3)
INSN_VEC(xvbstrc21i_h,     xdxjui4)
INSN_VEC(xvbstrc21i_w,     xdxjui5)
INSN_VEC(xvbstrc21i_d,     xdxjui6)
INSN_VEC(xvsat_b,          xdxjui3)
INSN_VEC(xvsat_h,          xdxjui4)
INSN_VEC(xvsat_w,          xdxjui5)
INSN_VEC(xvsat_d,          xdxjui6)
INSN_VEC(xvsat_bu,         xdxjui3)
INSN_VEC(xvsat_hu,         xdxjui4)
INSN_VEC(xvsat_wu,         xdxjui5)
INSN_VEC(xvsat_du,         xdxjui6)
INSN_VEC(xvslli_b,         xdxjui3)
INSN_VEC(xvslli_h,         xdxjui4)
INSN_VEC(xvslli_w,         xdxjui5)
INSN_VEC(xvslli_d,         xdxjui6)
INSN_VEC(xvsrli_b,         xdxjui3)
INSN_VEC(xvsrli_h,         xdxjui4)
INSN_VEC(xvsrli_w,         xdxjui5)
INSN_VEC(xvsrli_d,         xdxjui6)
INSN_VEC(xvsrai_b,         xdxjui3)
INSN_VEC(xvsrai_h,         xdxjui4)
INSN_VEC(xvsrai_w,         xdxjui5)
INSN_VEC(xvsrai_d,         xdxjui6)
INSN_VEC(xvsrlrneni_b_h,   xdxjui4)
INSN_VEC(xvsrlrneni_h_w,   xdxjui5)
INSN_VEC(xvsrlrneni_w_d,   xdxjui6)
INSN_VEC(xvsrlrneni_d_q,   xdxjui7)
INSN_VEC(xvsrarneni_b_h,   xdxjui4)
INSN_VEC(xvsrarneni_h_w,   xdxjui5)
INSN_VEC(xvsrarneni_w_d,   xdxjui6)
INSN_VEC(xvsrarneni_d_q,   xdxjui7)
INSN_VEC(xvsrlni_b_h,      xdxjui4)
INSN_VEC(xvsrlni_h_w,      xdxjui5)
INSN_VEC(xvsrlni_w_d,      xdxjui6)
INSN_VEC(xvsrlni_d_q,      xdxjui7)
INSN_VEC(xvsrlrni_b_h,     xdxjui4)
INSN_VEC(xvsrlrni_h_w,     xdxjui5)
INSN_VEC(xvsrlrni_w_d,     xdxjui6)
INSN_VEC(xvsrlrni_d_q,     xdxjui7)
INSN_VEC(xvssrlni_b_h,     xdxjui4)
INSN_VEC(xvssrlni_h_w,     xdxjui5)
INSN_VEC(xvssrlni_w_d,     xdxjui6)
INSN_VEC(xvssrlni_d_q,     xdxjui7)
INSN_VEC(xvssrlni_bu_h,    xdxjui4)
INSN_VEC(xvssrlni_hu_w,    xdxjui5)
INSN_VEC(xvssrlni_wu_d,    xdxjui6)
INSN_VEC(xvssrlni_du_q,    xdxjui7)
INSN_VEC(xvssrlrni_b_h,    xdxjui4)
INSN_VEC(xvssrlrni_h_w,    xdxjui5)
INSN_VEC(xvssrlrni_w_d,    xdxjui6)
INSN_VEC(xvssrlrni_d_q,    xdxjui7)
INSN_VEC(xvssrlrni_bu_h,   xdxjui4)
INSN_VEC(xvssrlrni_hu_w,   xdxjui5)
INSN_VEC(xvssrlrni_wu_d,   xdxjui6)
INSN_VEC(xvssrlrni_du_q,   xdxjui7)
INSN_VEC(xvsrani_b_h,      xdxjui4)
INSN_VEC(xvsrani_h_w,      xdxjui5)
INSN_VEC(xvsrani_w_d,      xdxjui6)
INSN_VEC(xvsrani_d_q,      xdxjui7)
INSN_VEC(xvsrarni_b_h,     xdxjui4)
INSN_VEC(xvsrarni_h_w,     xdxjui5)
INSN_VEC(xvsrarni_w_d,     xdxjui6)
INSN_VEC(xvsrarni_d_q,     xdxjui7)
INSN_VEC(xvssrani_b_h,     xdxjui4)
INSN_VEC(xvssrani_h_w,     xdxjui5)
INSN_VEC(xvssrani_w_d,     xdxjui6)
INSN_VEC(xvssrani_d_q,     xdxjui7)
INSN_VEC(xvssrani_bu_h,    xdxjui4)
INSN_VEC(xvssrani_hu_w,    xdxjui5)
INSN_VEC(xvssrani_wu_d,    xdxjui6)
INSN_VEC(xvssrani_du_q,    xdxjui7)
INSN_VEC(xvssrarni_b_h,    xdxjui4)
INSN_VEC(xvssrarni_h_w,    xdxjui5)
INSN_VEC(xvssrarni_w_d,    xdxjui6)
INSN_VEC(xvssrarni_d_q,    xdxjui7)
INSN_VEC(xvssrarni_bu_h,   xdxjui4)
INSN_VEC(xvssrarni_hu_w,   xdxjui5)
INSN_VEC(xvssrarni_wu_d,   xdxjui6)
INSN_VEC(xvssrarni_du_q,   xdxjui7)
INSN_VEC(xvssrlrneni_b_h,  xdxjui4)
INSN_VEC(xvssrlrneni_h_w,  xdxjui5)
INSN_VEC(xvssrlrneni_w_d,  xdxjui6)
INSN_VEC(xvssrlrneni_d_q,  xdxjui7)
INSN_VEC(xvssrlrneni_bu_h, xdxjui4)
INSN_VEC(xvssrlrneni_hu_w, xdxjui5)
INSN_VEC(xvssrlrneni_wu_d, xdxjui6)
INSN_VEC(xvssrlrneni_du_q, xdxjui7)
INSN_VEC(xvssrarneni_b_h,  xdxjui4)
INSN_VEC(xvssrarneni_h_w,  xdxjui5)
INSN_VEC(xvssrarneni_w_d,  xdxjui6)
INSN_VEC(xvssrarneni_d_q,  xdxjui7)
INSN_VEC(xvssrarneni_bu_h, xdxjui4)
INSN_VEC(xvssrarneni_hu_w, xdxjui5)
INSN_VEC(xvssrarneni_wu_d, xdxjui6)
INSN_VEC(xvssrarneni_du_q, xdxjui7)
INSN_VEC(xvextrins_d,      xdxjui8)
INSN_VEC(xvextrins_w,      xdxjui8)
INSN_VEC(xvextrins_h,      xdxjui8)
INSN_VEC(xvextrins_b,      xdxjui8)
INSN_VEC(xvshuf4i_b,       xdxjui8)
INSN_VEC(xvshuf4i_h,       xdxjui8)
INSN_VEC(xvshuf4i_w,       xdxjui8)
INSN_VEC(xvshuf4i_d,       xdxjui8)
INSN_VEC(xvshufi1_b,       xdxjui8)
INSN_VEC(xvshufi2_b,       xdxjui8)
INSN_VEC(xvshufi3_b,       xdxjui8)
INSN_VEC(xvshufi4_b,       xdxjui8)
INSN_VEC(xvshufi1_h,       xdxjui8)
INSN_VEC(xvshufi2_h,       xdxjui8)
INSN_VEC(xvseli_h,         xdxjui8)
INSN_VEC(xvseli_w,         xdxjui8)
INSN_VEC(xvseli_d,         xdxjui8)
INSN_VEC(xvbitseli_b,      xdxjui8)
INSN_VEC(xvbitmvzi_b,      xdxjui8)
INSN_VEC(xvbitmvnzi_b,     xdxjui8)
INSN_VEC(xvandi_b,         xdxjui8)
INSN_VEC(xvori_b,          xdxjui8)
INSN_VEC(xvxori_b,         xdxjui8)
INSN_VEC(xvnori_b,         xdxjui8)
INSN_VEC(xvldi,            xdi13)
INSN_VEC(xvpermi_w,        xdxjui8)
INSN_VEC(xvpermi_d,        xdxjui8)
INSN_VEC(xvpermi_q,        xdxjui8)



#define output_fcmp(C, PREFIX, SUFFIX)                                         \
{                                                                              \
    (C)->info->fprintf_func((C)->info->stream, "%08x   %s%s\tfcc%d, f%d, f%d", \
                            (C)->insn, PREFIX, SUFFIX, a->cd,                  \
                            a->fj, a->fk);                                     \
}

static bool output_cff_fcond(DisasContext *ctx, arg_cff_fcond * a,
                               const char *suffix)
{
    bool ret = true;
    switch (a->fcond) {
    case 0x0:
        output_fcmp(ctx, "fcmp_caf_", suffix);
        break;
    case 0x1:
        output_fcmp(ctx, "fcmp_saf_", suffix);
        break;
    case 0x2:
        output_fcmp(ctx, "fcmp_clt_", suffix);
        break;
    case 0x3:
        output_fcmp(ctx, "fcmp_slt_", suffix);
        break;
    case 0x4:
        output_fcmp(ctx, "fcmp_ceq_", suffix);
        break;
    case 0x5:
        output_fcmp(ctx, "fcmp_seq_", suffix);
        break;
    case 0x6:
        output_fcmp(ctx, "fcmp_cle_", suffix);
        break;
    case 0x7:
        output_fcmp(ctx, "fcmp_sle_", suffix);
        break;
    case 0x8:
        output_fcmp(ctx, "fcmp_cun_", suffix);
        break;
    case 0x9:
        output_fcmp(ctx, "fcmp_sun_", suffix);
        break;
    case 0xA:
        output_fcmp(ctx, "fcmp_cult_", suffix);
        break;
    case 0xB:
        output_fcmp(ctx, "fcmp_sult_", suffix);
        break;
    case 0xC:
        output_fcmp(ctx, "fcmp_cueq_", suffix);
        break;
    case 0xD:
        output_fcmp(ctx, "fcmp_sueq_", suffix);
        break;
    case 0xE:
        output_fcmp(ctx, "fcmp_cule_", suffix);
        break;
    case 0xF:
        output_fcmp(ctx, "fcmp_sule_", suffix);
        break;
    case 0x10:
        output_fcmp(ctx, "fcmp_cne_", suffix);
        break;
    case 0x11:
        output_fcmp(ctx, "fcmp_sne_", suffix);
        break;
    case 0x14:
        output_fcmp(ctx, "fcmp_cor_", suffix);
        break;
    case 0x15:
        output_fcmp(ctx, "fcmp_sor_", suffix);
        break;
    case 0x18:
        output_fcmp(ctx, "fcmp_cune_", suffix);
        break;
    case 0x19:
        output_fcmp(ctx, "fcmp_sune_", suffix);
        break;
    default:
        ret = false;
    }
    return ret;
}

#define FCMP_INSN(suffix)                               \
static bool trans_fcmp_cond_##suffix(DisasContext *ctx, \
                                     arg_cff_fcond * a) \
{                                                       \
    return output_cff_fcond(ctx, a, #suffix);           \
}

FCMP_INSN(s)
FCMP_INSN(d)

#define VEC_FCMP_INSN(insn, suffix, type)                       \
static bool trans_##insn(DisasContext *ctx, arg_fmt_##type * a) \
{                                                               \
    output_##type(ctx, a, #suffix);                             \
    return true;                                                \
}

VEC_FCMP_INSN(vfcmp_cond_s, s, vdvjvkfcond)
VEC_FCMP_INSN(vfcmp_cond_d, d, vdvjvkfcond)
VEC_FCMP_INSN(xvfcmp_cond_s, s, xdxjxkfcond)
VEC_FCMP_INSN(xvfcmp_cond_d, d, xdxjxkfcond)
