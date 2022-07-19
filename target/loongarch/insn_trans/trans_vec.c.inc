static inline void check_lsx_enabled(DisasContext *ctx)
{
#ifndef CONFIG_USER_ONLY
    //check cpucfg or csr wether supported LSX
    //generate_exception(ctx, EXCCODE_SXD);
#endif
}

static inline void check_lasx_enabled(DisasContext *ctx)
{
#ifndef CONFIG_USER_ONLY
    //check cpucfg or csr wether supported LASX
    //generate_exception(ctx, EXCCODE_ASXD)
#endif
}

static bool trans_vseq_b(DisasContext *ctx, arg_vseq_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_ceq_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vseq_h(DisasContext *ctx, arg_vseq_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_ceq_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vseq_w(DisasContext *ctx, arg_vseq_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_ceq_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vseq_d(DisasContext *ctx, arg_vseq_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_ceq_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsle_b(DisasContext *ctx, arg_vsle_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_cle_s_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsle_h(DisasContext *ctx, arg_vsle_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_cle_s_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsle_w(DisasContext *ctx, arg_vsle_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_cle_s_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsle_d(DisasContext *ctx, arg_vsle_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_cle_s_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsle_bu(DisasContext *ctx, arg_vsle_bu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_cle_u_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsle_hu(DisasContext *ctx, arg_vsle_hu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_cle_u_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsle_wu(DisasContext *ctx, arg_vsle_wu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_cle_u_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsle_du(DisasContext *ctx, arg_vsle_du *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_cle_u_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vslt_b(DisasContext *ctx, arg_vslt_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_clt_s_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vslt_h(DisasContext *ctx, arg_vslt_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_clt_s_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vslt_w(DisasContext *ctx, arg_vslt_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_clt_s_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vslt_d(DisasContext *ctx, arg_vslt_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_clt_s_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vslt_bu(DisasContext *ctx, arg_vslt_bu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_clt_u_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vslt_hu(DisasContext *ctx, arg_vslt_hu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_clt_u_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vslt_wu(DisasContext *ctx, arg_vslt_wu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_clt_u_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vslt_du(DisasContext *ctx, arg_vslt_du *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_clt_u_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vadd_b(DisasContext *ctx, arg_vadd_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_addv_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vadd_h(DisasContext *ctx, arg_vadd_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_addv_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vadd_w(DisasContext *ctx, arg_vadd_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_addv_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vadd_d(DisasContext *ctx, arg_vadd_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_addv_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsub_b(DisasContext *ctx, arg_vsub_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);
    TCGv_i32 byte = tcg_const_i32(DF_BYTE);

    check_lsx_enabled(ctx);
    gen_helper_msa_subv_df(cpu_env, byte, twd, tws, twt);

    tcg_temp_free_i32(byte);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsub_h(DisasContext *ctx, arg_vsub_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);
    TCGv_i32 half = tcg_const_i32(DF_HALF);

    check_lsx_enabled(ctx);
    gen_helper_msa_subv_df(cpu_env, half, twd, tws, twt);

    tcg_temp_free_i32(half);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsub_w(DisasContext *ctx, arg_vsub_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);
    TCGv_i32 word = tcg_const_i32(DF_WORD);

    check_lsx_enabled(ctx);
    gen_helper_msa_subv_df(cpu_env, word, twd, tws, twt);

    tcg_temp_free_i32(word);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsub_d(DisasContext *ctx, arg_vsub_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);
    TCGv_i32 doub = tcg_const_i32(DF_DOUBLE);

    check_lsx_enabled(ctx);
    gen_helper_msa_subv_df(cpu_env, doub, twd, tws, twt);

    tcg_temp_free_i32(doub);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsadd_b(DisasContext *ctx, arg_vsadd_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_adds_s_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsadd_h(DisasContext *ctx, arg_vsadd_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_adds_s_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsadd_w(DisasContext *ctx, arg_vsadd_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_adds_s_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsadd_d(DisasContext *ctx, arg_vsadd_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_adds_s_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vssub_b(DisasContext *ctx, arg_vssub_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);
    TCGv_i32 byte = tcg_const_i32(DF_BYTE);

    check_lsx_enabled(ctx);
    gen_helper_msa_subs_s_df(cpu_env, byte, twd, tws, twt);

    tcg_temp_free_i32(byte);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vssub_h(DisasContext *ctx, arg_vssub_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);
    TCGv_i32 half = tcg_const_i32(DF_HALF);

    check_lsx_enabled(ctx);
    gen_helper_msa_subs_s_df(cpu_env, half, twd, tws, twt);

    tcg_temp_free_i32(half);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vssub_w(DisasContext *ctx, arg_vssub_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);
    TCGv_i32 word = tcg_const_i32(DF_WORD);

    check_lsx_enabled(ctx);
    gen_helper_msa_subs_s_df(cpu_env, word, twd, tws, twt);

    tcg_temp_free_i32(word);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vssub_d(DisasContext *ctx, arg_vssub_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);
    TCGv_i32 doub = tcg_const_i32(DF_DOUBLE);

    check_lsx_enabled(ctx);
    gen_helper_msa_subs_s_df(cpu_env, doub, twd, tws, twt);

    tcg_temp_free_i32(doub);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsadd_bu(DisasContext *ctx, arg_vsadd_bu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_adds_u_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsadd_hu(DisasContext *ctx, arg_vsadd_hu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_adds_u_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsadd_wu(DisasContext *ctx, arg_vsadd_wu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_adds_u_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsadd_du(DisasContext *ctx, arg_vsadd_du *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_adds_u_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vssub_bu(DisasContext *ctx, arg_vssub_bu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);
    TCGv_i32 byte = tcg_const_i32(DF_BYTE);

    check_lsx_enabled(ctx);
    gen_helper_msa_subs_u_df(cpu_env, byte, twd, tws, twt);

    tcg_temp_free_i32(byte);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vssub_hu(DisasContext *ctx, arg_vssub_hu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);
    TCGv_i32 half = tcg_const_i32(DF_HALF);

    check_lsx_enabled(ctx);
    gen_helper_msa_subs_u_df(cpu_env, half, twd, tws, twt);

    tcg_temp_free_i32(half);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vssub_wu(DisasContext *ctx, arg_vssub_wu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);
    TCGv_i32 word = tcg_const_i32(DF_WORD);

    check_lsx_enabled(ctx);
    gen_helper_msa_subs_u_df(cpu_env, word, twd, tws, twt);

    tcg_temp_free_i32(word);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vssub_du(DisasContext *ctx, arg_vssub_du *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);
    TCGv_i32 doub = tcg_const_i32(DF_DOUBLE);

    check_lsx_enabled(ctx);
    gen_helper_msa_subs_u_df(cpu_env, doub, twd, tws, twt);

    tcg_temp_free_i32(doub);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vhaddw_h_b(DisasContext *ctx, arg_vhaddw_h_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_hadd_s_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vhaddw_w_h(DisasContext *ctx, arg_vhaddw_w_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_hadd_s_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vhaddw_d_w(DisasContext *ctx, arg_vhaddw_d_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_hadd_s_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vhaddw_q_d(DisasContext *ctx, arg_vhaddw_q_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vhaddw_q_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vhsubw_h_b(DisasContext *ctx, arg_vhsubw_h_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_hsub_s_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vhsubw_w_h(DisasContext *ctx, arg_vhsubw_w_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_hsub_s_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vhsubw_d_w(DisasContext *ctx, arg_vhsubw_d_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_hsub_s_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vhsubw_q_d(DisasContext *ctx, arg_vhsubw_q_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vhsubw_q_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vhaddw_hu_bu(DisasContext *ctx, arg_vhaddw_hu_bu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_hadd_u_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vhaddw_wu_hu(DisasContext *ctx, arg_vhaddw_wu_hu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_hadd_u_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vhaddw_du_wu(DisasContext *ctx, arg_vhaddw_du_wu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_hadd_u_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vhaddw_qu_du(DisasContext *ctx, arg_vhaddw_qu_du *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vhaddw_qu_du(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vhsubw_hu_bu(DisasContext *ctx, arg_vhsubw_hu_bu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_hsub_u_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vhsubw_wu_hu(DisasContext *ctx, arg_vhsubw_wu_hu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_hsub_u_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vhsubw_du_wu(DisasContext *ctx, arg_vhsubw_du_wu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_hsub_u_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vhsubw_qu_du(DisasContext *ctx, arg_vhsubw_qu_du *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vhsubw_qu_du(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vadda_b(DisasContext *ctx, arg_vadda_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_add_a_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vadda_h(DisasContext *ctx, arg_vadda_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_add_a_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vadda_w(DisasContext *ctx, arg_vadda_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_add_a_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vadda_d(DisasContext *ctx, arg_vadda_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_add_a_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vabsd_b(DisasContext *ctx, arg_vabsd_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_asub_s_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vabsd_h(DisasContext *ctx, arg_vabsd_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_asub_s_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vabsd_w(DisasContext *ctx, arg_vabsd_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_asub_s_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vabsd_d(DisasContext *ctx, arg_vabsd_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_asub_s_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vabsd_bu(DisasContext *ctx, arg_vabsd_bu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_asub_u_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vabsd_hu(DisasContext *ctx, arg_vabsd_hu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_asub_u_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vabsd_wu(DisasContext *ctx, arg_vabsd_wu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_asub_u_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vabsd_du(DisasContext *ctx, arg_vabsd_du *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_asub_u_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vavg_b(DisasContext *ctx, arg_vavg_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_ave_s_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vavg_h(DisasContext *ctx, arg_vavg_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_ave_s_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vavg_w(DisasContext *ctx, arg_vavg_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_ave_s_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vavg_d(DisasContext *ctx, arg_vavg_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_ave_s_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vavg_bu(DisasContext *ctx, arg_vavg_bu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_ave_u_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vavg_hu(DisasContext *ctx, arg_vavg_hu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_ave_u_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vavg_wu(DisasContext *ctx, arg_vavg_wu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_ave_u_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vavg_du(DisasContext *ctx, arg_vavg_du *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_ave_u_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vavgr_b(DisasContext *ctx, arg_vavgr_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_aver_s_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vavgr_h(DisasContext *ctx, arg_vavgr_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_aver_s_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vavgr_w(DisasContext *ctx, arg_vavgr_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_aver_s_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vavgr_d(DisasContext *ctx, arg_vavgr_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_aver_s_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vavgr_bu(DisasContext *ctx, arg_vavgr_bu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_aver_u_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vavgr_hu(DisasContext *ctx, arg_vavgr_hu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_aver_u_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vavgr_wu(DisasContext *ctx, arg_vavgr_wu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_aver_u_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vavgr_du(DisasContext *ctx, arg_vavgr_du *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_aver_u_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vmax_b(DisasContext *ctx, arg_vmax_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_max_s_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vmax_h(DisasContext *ctx, arg_vmax_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_max_s_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vmax_w(DisasContext *ctx, arg_vmax_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_max_s_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vmax_d(DisasContext *ctx, arg_vmax_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_max_s_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vmin_b(DisasContext *ctx, arg_vmin_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_min_s_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vmin_h(DisasContext *ctx, arg_vmin_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_min_s_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vmin_w(DisasContext *ctx, arg_vmin_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_min_s_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vmin_d(DisasContext *ctx, arg_vmin_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_min_s_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vmax_bu(DisasContext *ctx, arg_vmax_bu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_max_u_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vmax_hu(DisasContext *ctx, arg_vmax_hu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_max_u_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vmax_wu(DisasContext *ctx, arg_vmax_wu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_max_u_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vmax_du(DisasContext *ctx, arg_vmax_du *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_max_u_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vmin_bu(DisasContext *ctx, arg_vmin_bu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_min_u_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vmin_hu(DisasContext *ctx, arg_vmin_hu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_min_u_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vmin_wu(DisasContext *ctx, arg_vmin_wu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_min_u_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vmin_du(DisasContext *ctx, arg_vmin_du *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_min_u_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vmul_b(DisasContext *ctx, arg_vmul_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);
    TCGv_i32 byte = tcg_const_i32(DF_BYTE);

    check_lsx_enabled(ctx);
    gen_helper_msa_mulv_df(cpu_env, byte, twd, tws, twt);

    tcg_temp_free_i32(byte);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vmul_h(DisasContext *ctx, arg_vmul_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);
    TCGv_i32 half = tcg_const_i32(DF_HALF);

    check_lsx_enabled(ctx);
    gen_helper_msa_mulv_df(cpu_env, half, twd, tws, twt);

    tcg_temp_free_i32(half);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vmul_w(DisasContext *ctx, arg_vmul_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);
    TCGv_i32 word = tcg_const_i32(DF_WORD);

    check_lsx_enabled(ctx);
    gen_helper_msa_mulv_df(cpu_env, word, twd, tws, twt);

    tcg_temp_free_i32(word);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vmul_d(DisasContext *ctx, arg_vmul_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);
    TCGv_i32 doub = tcg_const_i32(DF_DOUBLE);

    check_lsx_enabled(ctx);
    gen_helper_msa_mulv_df(cpu_env, doub, twd, tws, twt);

    tcg_temp_free_i32(doub);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vmadd_b(DisasContext *ctx, arg_vmadd_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);
    TCGv_i32 byte = tcg_const_i32(DF_BYTE);

    check_lsx_enabled(ctx);
    gen_helper_msa_maddv_df(cpu_env, byte, twd, tws, twt);

    tcg_temp_free_i32(byte);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vmadd_h(DisasContext *ctx, arg_vmadd_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);
    TCGv_i32 half = tcg_const_i32(DF_HALF);

    check_lsx_enabled(ctx);
    gen_helper_msa_maddv_df(cpu_env, half, twd, tws, twt);

    tcg_temp_free_i32(half);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vmadd_w(DisasContext *ctx, arg_vmadd_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);
    TCGv_i32 word = tcg_const_i32(DF_WORD);

    check_lsx_enabled(ctx);
    gen_helper_msa_maddv_df(cpu_env, word, twd, tws, twt);

    tcg_temp_free_i32(word);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vmadd_d(DisasContext *ctx, arg_vmadd_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);
    TCGv_i32 doub = tcg_const_i32(DF_DOUBLE);

    check_lsx_enabled(ctx);
    gen_helper_msa_maddv_df(cpu_env, doub, twd, tws, twt);

    tcg_temp_free_i32(doub);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vmsub_b(DisasContext *ctx, arg_vmsub_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);
    TCGv_i32 byte = tcg_const_i32(DF_BYTE);

    check_lsx_enabled(ctx);
    gen_helper_msa_msubv_df(cpu_env, byte, twd, tws, twt);

    tcg_temp_free_i32(byte);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vmsub_h(DisasContext *ctx, arg_vmsub_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);
    TCGv_i32 half = tcg_const_i32(DF_HALF);

    check_lsx_enabled(ctx);
    gen_helper_msa_msubv_df(cpu_env, half, twd, tws, twt);

    tcg_temp_free_i32(half);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vmsub_w(DisasContext *ctx, arg_vmsub_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);
    TCGv_i32 word = tcg_const_i32(DF_WORD);

    check_lsx_enabled(ctx);
    gen_helper_msa_msubv_df(cpu_env, word, twd, tws, twt);

    tcg_temp_free_i32(word);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vmsub_d(DisasContext *ctx, arg_vmsub_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);
    TCGv_i32 doub = tcg_const_i32(DF_DOUBLE);

    check_lsx_enabled(ctx);
    gen_helper_msa_msubv_df(cpu_env, doub, twd, tws, twt);

    tcg_temp_free_i32(doub);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vdiv_b(DisasContext *ctx, arg_vdiv_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_div_s_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vdiv_h(DisasContext *ctx, arg_vdiv_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_div_s_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vdiv_w(DisasContext *ctx, arg_vdiv_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_div_s_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vdiv_d(DisasContext *ctx, arg_vdiv_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_div_s_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vmod_b(DisasContext *ctx, arg_vmod_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_mod_s_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vmod_h(DisasContext *ctx, arg_vmod_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_mod_s_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vmod_w(DisasContext *ctx, arg_vmod_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_mod_s_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vmod_d(DisasContext *ctx, arg_vmod_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_mod_s_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vdiv_bu(DisasContext *ctx, arg_vdiv_bu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_div_u_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vdiv_hu(DisasContext *ctx, arg_vdiv_hu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_div_u_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vdiv_wu(DisasContext *ctx, arg_vdiv_wu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_div_u_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vdiv_du(DisasContext *ctx, arg_vdiv_du *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_div_u_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vmod_bu(DisasContext *ctx, arg_vmod_bu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_mod_u_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vmod_hu(DisasContext *ctx, arg_vmod_hu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_mod_u_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vmod_wu(DisasContext *ctx, arg_vmod_wu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_mod_u_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vmod_du(DisasContext *ctx, arg_vmod_du *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_mod_u_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsll_b(DisasContext *ctx, arg_vsll_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_sll_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vsll_h(DisasContext *ctx, arg_vsll_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_sll_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsll_w(DisasContext *ctx, arg_vsll_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_sll_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsll_d(DisasContext *ctx, arg_vsll_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_sll_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsrl_b(DisasContext *ctx, arg_vsrl_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_srl_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vsrl_h(DisasContext *ctx, arg_vsrl_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_srl_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsrl_w(DisasContext *ctx, arg_vsrl_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_srl_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsrl_d(DisasContext *ctx, arg_vsrl_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_srl_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsra_b(DisasContext *ctx, arg_vsra_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_sra_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsra_h(DisasContext *ctx, arg_vsra_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_sra_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsra_w(DisasContext *ctx, arg_vsra_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_sra_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsra_d(DisasContext *ctx, arg_vsra_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_sra_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vrotr_b(DisasContext *ctx, arg_vrotr_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vrotr_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vrotr_h(DisasContext *ctx, arg_vrotr_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vrotr_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vrotr_w(DisasContext *ctx, arg_vrotr_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vrotr_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vrotr_d(DisasContext *ctx, arg_vrotr_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vrotr_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsrlr_b(DisasContext *ctx, arg_vsrlr_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_srlr_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vsrlr_h(DisasContext *ctx, arg_vsrlr_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_srlr_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsrlr_w(DisasContext *ctx, arg_vsrlr_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_srlr_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsrlr_d(DisasContext *ctx, arg_vsrlr_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_srlr_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsrar_b(DisasContext *ctx, arg_vsrar_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_srar_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vsrar_h(DisasContext *ctx, arg_vsrar_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_srar_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsrar_w(DisasContext *ctx, arg_vsrar_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_srar_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsrar_d(DisasContext *ctx, arg_vsrar_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_srar_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vbitclr_b(DisasContext *ctx, arg_vbitclr_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_bclr_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vbitclr_h(DisasContext *ctx, arg_vbitclr_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_bclr_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vbitclr_w(DisasContext *ctx, arg_vbitclr_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_bclr_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vbitclr_d(DisasContext *ctx, arg_vbitclr_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_bclr_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vbitset_b(DisasContext *ctx, arg_vbitset_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_bset_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vbitset_h(DisasContext *ctx, arg_vbitset_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_bset_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vbitset_w(DisasContext *ctx, arg_vbitset_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_bset_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vbitset_d(DisasContext *ctx, arg_vbitset_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_bset_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vbitrev_b(DisasContext *ctx, arg_vbitrev_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_bneg_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vbitrev_h(DisasContext *ctx, arg_vbitrev_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_bneg_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vbitrev_w(DisasContext *ctx, arg_vbitrev_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_bneg_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vbitrev_d(DisasContext *ctx, arg_vbitrev_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_bneg_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vpackev_b(DisasContext *ctx, arg_vpackev_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_ilvev_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vpackev_h(DisasContext *ctx, arg_vpackev_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_ilvev_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vpackev_w(DisasContext *ctx, arg_vpackev_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_ilvev_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vpackev_d(DisasContext *ctx, arg_vpackev_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_ilvev_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vpackod_b(DisasContext *ctx, arg_vpackod_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_ilvod_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vpackod_h(DisasContext *ctx, arg_vpackod_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_ilvod_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vpackod_w(DisasContext *ctx, arg_vpackod_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_ilvod_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vpackod_d(DisasContext *ctx, arg_vpackod_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_ilvod_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vilvl_b(DisasContext *ctx, arg_vilvl_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_ilvr_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vilvl_h(DisasContext *ctx, arg_vilvl_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_ilvr_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vilvl_w(DisasContext *ctx, arg_vilvl_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_ilvr_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vilvl_d(DisasContext *ctx, arg_vilvl_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_ilvr_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vilvh_b(DisasContext *ctx, arg_vilvh_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_ilvl_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vilvh_h(DisasContext *ctx, arg_vilvh_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_ilvl_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vilvh_w(DisasContext *ctx, arg_vilvh_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_ilvl_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vilvh_d(DisasContext *ctx, arg_vilvh_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_ilvl_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vpickev_b(DisasContext *ctx, arg_vpickev_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_pckev_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vpickev_h(DisasContext *ctx, arg_vpickev_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_pckev_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vpickev_w(DisasContext *ctx, arg_vpickev_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_pckev_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vpickev_d(DisasContext *ctx, arg_vpickev_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_pckev_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vpickod_b(DisasContext *ctx, arg_vpickod_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_pckod_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vpickod_h(DisasContext *ctx, arg_vpickod_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_pckod_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vpickod_w(DisasContext *ctx, arg_vpickod_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_pckod_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vpickod_d(DisasContext *ctx, arg_vpickod_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_pckod_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vreplve_b(DisasContext *ctx, arg_vreplve_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 trt = tcg_const_i32(a->rk);
    TCGv_i32 byte = tcg_const_i32(DF_BYTE);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vreplve_df(cpu_env, byte, twd, tws, trt);

    tcg_temp_free_i32(byte);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(trt);
    return true;
}


static bool trans_vreplve_h(DisasContext *ctx, arg_vreplve_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 trt = tcg_const_i32(a->rk);
    TCGv_i32 half = tcg_const_i32(DF_HALF);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vreplve_df(cpu_env, half, twd, tws, trt);

    tcg_temp_free_i32(half);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(trt);
    return true;
}

static bool trans_vreplve_w(DisasContext *ctx, arg_vreplve_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 trt = tcg_const_i32(a->rk);
    TCGv_i32 word = tcg_const_i32(DF_WORD);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vreplve_df(cpu_env, word, twd, tws, trt);

    tcg_temp_free_i32(word);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(trt);
    return true;
}

static bool trans_vreplve_d(DisasContext *ctx, arg_vreplve_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 trt = tcg_const_i32(a->rk);
    TCGv_i32 doub = tcg_const_i32(DF_DOUBLE);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vreplve_df(cpu_env, doub, twd, tws, trt);

    tcg_temp_free_i32(doub);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(trt);
    return true;
}

static bool trans_vand_v(DisasContext *ctx, arg_vand_v *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_and_v(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vor_v(DisasContext *ctx, arg_vor_v *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_or_v(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vxor_v(DisasContext *ctx, arg_vxor_v *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_xor_v(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vnor_v(DisasContext *ctx, arg_vnor_v *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_msa_nor_v(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vandn_v(DisasContext *ctx, arg_vandn_v *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vandn_v(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vorn_v(DisasContext *ctx, arg_vorn_v *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vorn_v(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

#define TRANS_LSX_3XR_DF(name, op, df)                              \
static bool trans_ ## name(DisasContext *ctx, arg_ ## name *a)      \
{                                                                   \
    TCGv_i32 twd = tcg_const_i32(a->xd);                            \
    TCGv_i32 tws = tcg_const_i32(a->xj);                            \
    TCGv_i32 twt = tcg_const_i32(a->xk);                            \
    TCGv_i32 tdf = tcg_const_i32(df);                               \
    check_lasx_enabled(ctx);                                        \
    gen_helper_lsx_ ## op ## _df(cpu_env, tdf, twd, tws, twt);      \
    tcg_temp_free_i32(twd);                                         \
    tcg_temp_free_i32(tws);                                         \
    tcg_temp_free_i32(twt);                                         \
    tcg_temp_free_i32(tdf);                                         \
    return true;                                                    \
}

TRANS_LSX_3XR_DF(xvseq_b, xvseq_s, DF_BYTE)
TRANS_LSX_3XR_DF(xvseq_h, xvseq_s, DF_HALF)
TRANS_LSX_3XR_DF(xvseq_w, xvseq_s, DF_WORD)
TRANS_LSX_3XR_DF(xvseq_d, xvseq_s, DF_DOUBLE)
TRANS_LSX_3XR_DF(xvsle_b, xvsle_s, DF_BYTE)
TRANS_LSX_3XR_DF(xvsle_h, xvsle_s, DF_HALF)
TRANS_LSX_3XR_DF(xvsle_w, xvsle_s, DF_WORD)
TRANS_LSX_3XR_DF(xvsle_d, xvsle_s, DF_DOUBLE)
TRANS_LSX_3XR_DF(xvsle_bu, xvsle_u, DF_BYTE)
TRANS_LSX_3XR_DF(xvsle_hu, xvsle_u, DF_HALF)
TRANS_LSX_3XR_DF(xvsle_wu, xvsle_u, DF_WORD)
TRANS_LSX_3XR_DF(xvsle_du, xvsle_u, DF_DOUBLE)
TRANS_LSX_3XR_DF(xvslt_b, xvslt_s, DF_BYTE)
TRANS_LSX_3XR_DF(xvslt_h, xvslt_s, DF_HALF)
TRANS_LSX_3XR_DF(xvslt_w, xvslt_s, DF_WORD)
TRANS_LSX_3XR_DF(xvslt_d, xvslt_s, DF_DOUBLE)
TRANS_LSX_3XR_DF(xvslt_bu, xvslt_u, DF_BYTE)
TRANS_LSX_3XR_DF(xvslt_hu, xvslt_u, DF_HALF)
TRANS_LSX_3XR_DF(xvslt_wu, xvslt_u, DF_WORD)
TRANS_LSX_3XR_DF(xvslt_du, xvslt_u, DF_DOUBLE)
TRANS_LSX_3XR_DF(xvadd_b, xvadd, DF_BYTE)
TRANS_LSX_3XR_DF(xvadd_h, xvadd, DF_HALF)
TRANS_LSX_3XR_DF(xvadd_w, xvadd, DF_WORD)
TRANS_LSX_3XR_DF(xvadd_d, xvadd, DF_DOUBLE)
TRANS_LSX_3XR_DF(xvsub_b, xvsub, DF_BYTE)
TRANS_LSX_3XR_DF(xvsub_h, xvsub, DF_HALF)
TRANS_LSX_3XR_DF(xvsub_w, xvsub, DF_WORD)
TRANS_LSX_3XR_DF(xvsub_d, xvsub, DF_DOUBLE)

TRANS_LSX_3XR_DF(xvsadd_b, xvsadd_s, DF_BYTE)
TRANS_LSX_3XR_DF(xvsadd_h, xvsadd_s, DF_HALF)
TRANS_LSX_3XR_DF(xvsadd_w, xvsadd_s, DF_WORD)
TRANS_LSX_3XR_DF(xvsadd_d, xvsadd_s, DF_DOUBLE)
TRANS_LSX_3XR_DF(xvssub_b, xvssub_s, DF_BYTE)
TRANS_LSX_3XR_DF(xvssub_h, xvssub_s, DF_HALF)
TRANS_LSX_3XR_DF(xvssub_w, xvssub_s, DF_WORD)
TRANS_LSX_3XR_DF(xvssub_d, xvssub_s, DF_DOUBLE)
TRANS_LSX_3XR_DF(xvsadd_bu, xvsadd_u, DF_BYTE)
TRANS_LSX_3XR_DF(xvsadd_hu, xvsadd_u, DF_HALF)
TRANS_LSX_3XR_DF(xvsadd_wu, xvsadd_u, DF_WORD)
TRANS_LSX_3XR_DF(xvsadd_du, xvsadd_u, DF_DOUBLE)
TRANS_LSX_3XR_DF(xvssub_bu, xvssub_u, DF_BYTE)
TRANS_LSX_3XR_DF(xvssub_hu, xvssub_u, DF_HALF)
TRANS_LSX_3XR_DF(xvssub_wu, xvssub_u, DF_WORD)
TRANS_LSX_3XR_DF(xvssub_du, xvssub_u, DF_DOUBLE)
//TRANS_LSX_3XR_DF(xvssub_bu_bu_b, xvssub_u_u_s, DF_BYTE)
//TRANS_LSX_3XR_DF(xvssub_hu_hu_h, xvssub_u_u_s, DF_HALF)
//TRANS_LSX_3XR_DF(xvssub_wu_wu_w, xvssub_u_u_s, DF_WORD)
//TRANS_LSX_3XR_DF(xvssub_du_du_d, xvssub_u_u_s, DF_DOUBLE)
//TRANS_LSX_3XR_DF(xvssub_bu_b_bu, xvssub_u_s_u, DF_BYTE)
//TRANS_LSX_3XR_DF(xvssub_hu_h_hu, xvssub_u_s_u, DF_HALF)
//TRANS_LSX_3XR_DF(xvssub_wu_w_wu, xvssub_u_s_u, DF_WORD)
//TRANS_LSX_3XR_DF(xvssub_du_d_du, xvssub_u_s_u, DF_DOUBLE)
//TRANS_LSX_3XR_DF(xvssub_b_bu_bu, xvssub_s_u_u, DF_BYTE)
//TRANS_LSX_3XR_DF(xvssub_h_hu_hu, xvssub_s_u_u, DF_HALF)
//TRANS_LSX_3XR_DF(xvssub_w_wu_wu, xvssub_s_u_u, DF_WORD)
//TRANS_LSX_3XR_DF(xvssub_d_du_du, xvssub_s_u_u, DF_DOUBLE)
TRANS_LSX_3XR_DF(xvhaddw_h_b, xvhaddw_s_s, DF_HALF)
TRANS_LSX_3XR_DF(xvhaddw_w_h, xvhaddw_s_s, DF_WORD)
TRANS_LSX_3XR_DF(xvhaddw_d_w, xvhaddw_s_s, DF_DOUBLE)
TRANS_LSX_3XR_DF(xvhaddw_q_d, xvhaddw_s_s, DF_QUAD)
TRANS_LSX_3XR_DF(xvhsubw_h_b, xvhsubw_s_s, DF_HALF)
TRANS_LSX_3XR_DF(xvhsubw_w_h, xvhsubw_s_s, DF_WORD)
TRANS_LSX_3XR_DF(xvhsubw_d_w, xvhsubw_s_s, DF_DOUBLE)
TRANS_LSX_3XR_DF(xvhsubw_q_d, xvhsubw_s_s, DF_QUAD)
TRANS_LSX_3XR_DF(xvhaddw_hu_bu, xvhaddw_u_u, DF_HALF)
TRANS_LSX_3XR_DF(xvhaddw_wu_hu, xvhaddw_u_u, DF_WORD)
TRANS_LSX_3XR_DF(xvhaddw_du_wu, xvhaddw_u_u, DF_DOUBLE)
TRANS_LSX_3XR_DF(xvhaddw_qu_du, xvhaddw_u_u, DF_QUAD)
TRANS_LSX_3XR_DF(xvhsubw_hu_bu, xvhsubw_u_u, DF_HALF)
TRANS_LSX_3XR_DF(xvhsubw_wu_hu, xvhsubw_u_u, DF_WORD)
TRANS_LSX_3XR_DF(xvhsubw_du_wu, xvhsubw_u_u, DF_DOUBLE)
TRANS_LSX_3XR_DF(xvhsubw_qu_du, xvhsubw_u_u, DF_QUAD)
TRANS_LSX_3XR_DF(xvadda_b, xvadda, DF_BYTE)
TRANS_LSX_3XR_DF(xvadda_h, xvadda, DF_HALF)
TRANS_LSX_3XR_DF(xvadda_w, xvadda, DF_WORD)
TRANS_LSX_3XR_DF(xvadda_d, xvadda, DF_DOUBLE)
//TRANS_LSX_3XR_DF(xvsadda_b, xvsadda, DF_BYTE)
//TRANS_LSX_3XR_DF(xvsadda_h, xvsadda, DF_HALF)
//TRANS_LSX_3XR_DF(xvsadda_w, xvsadda, DF_WORD)
//TRANS_LSX_3XR_DF(xvsadda_d, xvsadda, DF_DOUBLE)
TRANS_LSX_3XR_DF(xvabsd_b, xvabsd_s, DF_BYTE)
TRANS_LSX_3XR_DF(xvabsd_h, xvabsd_s, DF_HALF)
TRANS_LSX_3XR_DF(xvabsd_w, xvabsd_s, DF_WORD)
TRANS_LSX_3XR_DF(xvabsd_d, xvabsd_s, DF_DOUBLE)
TRANS_LSX_3XR_DF(xvabsd_bu, xvabsd_u, DF_BYTE)
TRANS_LSX_3XR_DF(xvabsd_hu, xvabsd_u, DF_HALF)
TRANS_LSX_3XR_DF(xvabsd_wu, xvabsd_u, DF_WORD)
TRANS_LSX_3XR_DF(xvabsd_du, xvabsd_u, DF_DOUBLE)
TRANS_LSX_3XR_DF(xvavg_b, xvavg_s, DF_BYTE)
TRANS_LSX_3XR_DF(xvavg_h, xvavg_s, DF_HALF)
TRANS_LSX_3XR_DF(xvavg_w, xvavg_s, DF_WORD)
TRANS_LSX_3XR_DF(xvavg_d, xvavg_s, DF_DOUBLE)
TRANS_LSX_3XR_DF(xvavg_bu, xvavg_u, DF_BYTE)
TRANS_LSX_3XR_DF(xvavg_hu, xvavg_u, DF_HALF)
TRANS_LSX_3XR_DF(xvavg_wu, xvavg_u, DF_WORD)
TRANS_LSX_3XR_DF(xvavg_du, xvavg_u, DF_DOUBLE)
TRANS_LSX_3XR_DF(xvavgr_b, xvavgr_s, DF_BYTE)
TRANS_LSX_3XR_DF(xvavgr_h, xvavgr_s, DF_HALF)
TRANS_LSX_3XR_DF(xvavgr_w, xvavgr_s, DF_WORD)
TRANS_LSX_3XR_DF(xvavgr_d, xvavgr_s, DF_DOUBLE)
TRANS_LSX_3XR_DF(xvavgr_bu, xvavgr_u, DF_BYTE)
TRANS_LSX_3XR_DF(xvavgr_hu, xvavgr_u, DF_HALF)
TRANS_LSX_3XR_DF(xvavgr_wu, xvavgr_u, DF_WORD)
TRANS_LSX_3XR_DF(xvavgr_du, xvavgr_u, DF_DOUBLE)

TRANS_LSX_3XR_DF(xvmax_b, xvmax_s, DF_BYTE)
TRANS_LSX_3XR_DF(xvmax_h, xvmax_s, DF_HALF)
TRANS_LSX_3XR_DF(xvmax_w, xvmax_s, DF_WORD)
TRANS_LSX_3XR_DF(xvmax_d, xvmax_s, DF_DOUBLE)
TRANS_LSX_3XR_DF(xvmin_b, xvmin_s, DF_BYTE)
TRANS_LSX_3XR_DF(xvmin_h, xvmin_s, DF_HALF)
TRANS_LSX_3XR_DF(xvmin_w, xvmin_s, DF_WORD)
TRANS_LSX_3XR_DF(xvmin_d, xvmin_s, DF_DOUBLE)
TRANS_LSX_3XR_DF(xvmax_bu, xvmax_u, DF_BYTE)
TRANS_LSX_3XR_DF(xvmax_hu, xvmax_u, DF_HALF)
TRANS_LSX_3XR_DF(xvmax_wu, xvmax_u, DF_WORD)
TRANS_LSX_3XR_DF(xvmax_du, xvmax_u, DF_DOUBLE)
TRANS_LSX_3XR_DF(xvmin_bu, xvmin_u, DF_BYTE)
TRANS_LSX_3XR_DF(xvmin_hu, xvmin_u, DF_HALF)
TRANS_LSX_3XR_DF(xvmin_wu, xvmin_u, DF_WORD)
TRANS_LSX_3XR_DF(xvmin_du, xvmin_u, DF_DOUBLE)
//TRANS_LSX_3XR_DF(xvmaxa_b, xvmaxa, DF_BYTE)
//TRANS_LSX_3XR_DF(xvmaxa_h, xvmaxa, DF_HALF)
//TRANS_LSX_3XR_DF(xvmaxa_w, xvmaxa, DF_WORD)
//TRANS_LSX_3XR_DF(xvmaxa_d, xvmaxa, DF_DOUBLE)
//TRANS_LSX_3XR_DF(xvmina_b, xvmina, DF_BYTE)
//TRANS_LSX_3XR_DF(xvmina_h, xvmina, DF_HALF)
//TRANS_LSX_3XR_DF(xvmina_w, xvmina, DF_WORD)
//TRANS_LSX_3XR_DF(xvmina_d, xvmina, DF_DOUBLE)

TRANS_LSX_3XR_DF(xvmul_b, xvmul, DF_BYTE)
TRANS_LSX_3XR_DF(xvmul_h, xvmul, DF_HALF)
TRANS_LSX_3XR_DF(xvmul_w, xvmul, DF_WORD)
TRANS_LSX_3XR_DF(xvmul_d, xvmul, DF_DOUBLE)

TRANS_LSX_3XR_DF(xvmadd_b, xvmadd, DF_BYTE)
TRANS_LSX_3XR_DF(xvmadd_h, xvmadd, DF_HALF)
TRANS_LSX_3XR_DF(xvmadd_w, xvmadd, DF_WORD)
TRANS_LSX_3XR_DF(xvmadd_d, xvmadd, DF_DOUBLE)
TRANS_LSX_3XR_DF(xvmsub_b, xvmsub, DF_BYTE)
TRANS_LSX_3XR_DF(xvmsub_h, xvmsub, DF_HALF)
TRANS_LSX_3XR_DF(xvmsub_w, xvmsub, DF_WORD)
TRANS_LSX_3XR_DF(xvmsub_d, xvmsub, DF_DOUBLE)

//TRANS_LSX_3XR_DF(xvdp2_h_b, xvdp2_s_s, DF_HALF)
//TRANS_LSX_3XR_DF(xvdp2_w_h, xvdp2_s_s, DF_WORD)
//TRANS_LSX_3XR_DF(xvdp2_d_w, xvdp2_s_s, DF_DOUBLE)
//TRANS_LSX_3XR_DF(xvdp2_q_d, xvdp2_s_s, DF_QUAD)
//TRANS_LSX_3XR_DF(xvdp2_hu_bu, xvdp2_u_u, DF_HALF)
//TRANS_LSX_3XR_DF(xvdp2_wu_hu, xvdp2_u_u, DF_WORD)
//TRANS_LSX_3XR_DF(xvdp2_du_wu, xvdp2_u_u, DF_DOUBLE)
//TRANS_LSX_3XR_DF(xvdp2_qu_du, xvdp2_u_u, DF_QUAD)
//TRANS_LSX_3XR_DF(xvdp2_h_bu_b, xvdp2_s_u_s, DF_HALF)
//TRANS_LSX_3XR_DF(xvdp2_w_hu_h, xvdp2_s_u_s, DF_WORD)
//TRANS_LSX_3XR_DF(xvdp2_d_wu_w, xvdp2_s_u_s, DF_DOUBLE)
//TRANS_LSX_3XR_DF(xvdp2_q_du_d, xvdp2_s_u_s, DF_QUAD)
//TRANS_LSX_3XR_DF(xvdp2add_h_b, xvdp2add_s_s, DF_HALF)
//TRANS_LSX_3XR_DF(xvdp2add_w_h, xvdp2add_s_s, DF_WORD)
//TRANS_LSX_3XR_DF(xvdp2add_d_w, xvdp2add_s_s, DF_DOUBLE)
//TRANS_LSX_3XR_DF(xvdp2add_q_d, xvdp2add_s_s, DF_QUAD)
//TRANS_LSX_3XR_DF(xvdp2add_h_bu, xvdp2add_s_u, DF_HALF)
//TRANS_LSX_3XR_DF(xvdp2add_w_hu, xvdp2add_s_u, DF_WORD)
//TRANS_LSX_3XR_DF(xvdp2add_d_wu, xvdp2add_s_u, DF_DOUBLE)
//TRANS_LSX_3XR_DF(xvdp2add_q_du, xvdp2add_s_u, DF_QUAD)
//TRANS_LSX_3XR_DF(xvdp2add_h_bu_b, xvdp2add_s_u_s, DF_HALF)
//TRANS_LSX_3XR_DF(xvdp2add_w_hu_h, xvdp2add_s_u_s, DF_WORD)
//TRANS_LSX_3XR_DF(xvdp2add_d_wu_w, xvdp2add_s_u_s, DF_DOUBLE)
//TRANS_LSX_3XR_DF(xvdp2add_q_du_d, xvdp2add_s_u_s, DF_QUAD)
//TRANS_LSX_3XR_DF(xvdp2sub_h_b, xvdp2sub_s_s, DF_HALF)
//TRANS_LSX_3XR_DF(xvdp2sub_w_h, xvdp2sub_s_s, DF_WORD)
//TRANS_LSX_3XR_DF(xvdp2sub_d_w, xvdp2sub_s_s, DF_DOUBLE)
//TRANS_LSX_3XR_DF(xvdp2sub_q_d, xvdp2sub_s_s, DF_QUAD)
//TRANS_LSX_3XR_DF(xvdp2sub_h_bu, xvdp2sub_s_u, DF_HALF)
//TRANS_LSX_3XR_DF(xvdp2sub_w_hu, xvdp2sub_s_u, DF_WORD)
//TRANS_LSX_3XR_DF(xvdp2sub_d_wu, xvdp2sub_s_u, DF_DOUBLE)
//TRANS_LSX_3XR_DF(xvdp2sub_q_du, xvdp2sub_s_u, DF_QUAD)

TRANS_LSX_3XR_DF(xvdiv_b, xvdiv_s, DF_BYTE)
TRANS_LSX_3XR_DF(xvdiv_h, xvdiv_s, DF_HALF)
TRANS_LSX_3XR_DF(xvdiv_w, xvdiv_s, DF_WORD)
TRANS_LSX_3XR_DF(xvdiv_d, xvdiv_s, DF_DOUBLE)
TRANS_LSX_3XR_DF(xvmod_b, xvmod_s, DF_BYTE)
TRANS_LSX_3XR_DF(xvmod_h, xvmod_s, DF_HALF)
TRANS_LSX_3XR_DF(xvmod_w, xvmod_s, DF_WORD)
TRANS_LSX_3XR_DF(xvmod_d, xvmod_s, DF_DOUBLE)
TRANS_LSX_3XR_DF(xvdiv_bu, xvdiv_u, DF_BYTE)
TRANS_LSX_3XR_DF(xvdiv_hu, xvdiv_u, DF_HALF)
TRANS_LSX_3XR_DF(xvdiv_wu, xvdiv_u, DF_WORD)
TRANS_LSX_3XR_DF(xvdiv_du, xvdiv_u, DF_DOUBLE)
TRANS_LSX_3XR_DF(xvmod_bu, xvmod_u, DF_BYTE)
TRANS_LSX_3XR_DF(xvmod_hu, xvmod_u, DF_HALF)
TRANS_LSX_3XR_DF(xvmod_wu, xvmod_u, DF_WORD)
TRANS_LSX_3XR_DF(xvmod_du, xvmod_u, DF_DOUBLE)
TRANS_LSX_3XR_DF(xvsll_b, xvsll, DF_BYTE)
TRANS_LSX_3XR_DF(xvsll_h, xvsll, DF_HALF)
TRANS_LSX_3XR_DF(xvsll_w, xvsll, DF_WORD)
TRANS_LSX_3XR_DF(xvsll_d, xvsll, DF_DOUBLE)
TRANS_LSX_3XR_DF(xvsrl_b, xvsrl, DF_BYTE)
TRANS_LSX_3XR_DF(xvsrl_h, xvsrl, DF_HALF)
TRANS_LSX_3XR_DF(xvsrl_w, xvsrl, DF_WORD)
TRANS_LSX_3XR_DF(xvsrl_d, xvsrl, DF_DOUBLE)
TRANS_LSX_3XR_DF(xvsra_b, xvsra, DF_BYTE)
TRANS_LSX_3XR_DF(xvsra_h, xvsra, DF_HALF)
TRANS_LSX_3XR_DF(xvsra_w, xvsra, DF_WORD)
TRANS_LSX_3XR_DF(xvsra_d, xvsra, DF_DOUBLE)
TRANS_LSX_3XR_DF(xvrotr_b, xvrotr, DF_BYTE)
TRANS_LSX_3XR_DF(xvrotr_h, xvrotr, DF_HALF)
TRANS_LSX_3XR_DF(xvrotr_w, xvrotr, DF_WORD)
TRANS_LSX_3XR_DF(xvrotr_d, xvrotr, DF_DOUBLE)
TRANS_LSX_3XR_DF(xvsrlr_b, xvsrlr, DF_BYTE)
TRANS_LSX_3XR_DF(xvsrlr_h, xvsrlr, DF_HALF)
TRANS_LSX_3XR_DF(xvsrlr_w, xvsrlr, DF_WORD)
TRANS_LSX_3XR_DF(xvsrlr_d, xvsrlr, DF_DOUBLE)
TRANS_LSX_3XR_DF(xvsrar_b, xvsrar, DF_BYTE)
TRANS_LSX_3XR_DF(xvsrar_h, xvsrar, DF_HALF)
TRANS_LSX_3XR_DF(xvsrar_w, xvsrar, DF_WORD)
TRANS_LSX_3XR_DF(xvsrar_d, xvsrar, DF_DOUBLE)

TRANS_LSX_3XR_DF(xvbitclr_b, xvbitclr, DF_BYTE)
TRANS_LSX_3XR_DF(xvbitclr_h, xvbitclr, DF_HALF)
TRANS_LSX_3XR_DF(xvbitclr_w, xvbitclr, DF_WORD)
TRANS_LSX_3XR_DF(xvbitclr_d, xvbitclr, DF_DOUBLE)
TRANS_LSX_3XR_DF(xvbitset_b, xvbitset, DF_BYTE)
TRANS_LSX_3XR_DF(xvbitset_h, xvbitset, DF_HALF)
TRANS_LSX_3XR_DF(xvbitset_w, xvbitset, DF_WORD)
TRANS_LSX_3XR_DF(xvbitset_d, xvbitset, DF_DOUBLE)
TRANS_LSX_3XR_DF(xvbitrev_b, xvbitrev, DF_BYTE)
TRANS_LSX_3XR_DF(xvbitrev_h, xvbitrev, DF_HALF)
TRANS_LSX_3XR_DF(xvbitrev_w, xvbitrev, DF_WORD)
TRANS_LSX_3XR_DF(xvbitrev_d, xvbitrev, DF_DOUBLE)
//TRANS_LSX_3XR_DF(xvbstrc12_b, xvbstrc12, DF_BYTE)
//TRANS_LSX_3XR_DF(xvbstrc12_h, xvbstrc12, DF_HALF)
//TRANS_LSX_3XR_DF(xvbstrc12_w, xvbstrc12, DF_WORD)
//TRANS_LSX_3XR_DF(xvbstrc12_d, xvbstrc12, DF_DOUBLE)
//TRANS_LSX_3XR_DF(xvbstrc21_b, xvbstrc21, DF_BYTE)
//TRANS_LSX_3XR_DF(xvbstrc21_h, xvbstrc21, DF_HALF)
//TRANS_LSX_3XR_DF(xvbstrc21_w, xvbstrc21, DF_WORD)
//TRANS_LSX_3XR_DF(xvbstrc21_d, xvbstrc21, DF_DOUBLE)
TRANS_LSX_3XR_DF(xvpackev_b, xvpackev, DF_BYTE)
TRANS_LSX_3XR_DF(xvpackev_h, xvpackev, DF_HALF)
TRANS_LSX_3XR_DF(xvpackev_w, xvpackev, DF_WORD)
TRANS_LSX_3XR_DF(xvpackev_d, xvpackev, DF_DOUBLE)
TRANS_LSX_3XR_DF(xvpackod_b, xvpackod, DF_BYTE)
TRANS_LSX_3XR_DF(xvpackod_h, xvpackod, DF_HALF)
TRANS_LSX_3XR_DF(xvpackod_w, xvpackod, DF_WORD)
TRANS_LSX_3XR_DF(xvpackod_d, xvpackod, DF_DOUBLE)
TRANS_LSX_3XR_DF(xvilvl_b, xvilvl, DF_BYTE)
TRANS_LSX_3XR_DF(xvilvl_h, xvilvl, DF_HALF)
TRANS_LSX_3XR_DF(xvilvl_w, xvilvl, DF_WORD)
TRANS_LSX_3XR_DF(xvilvl_d, xvilvl, DF_DOUBLE)
TRANS_LSX_3XR_DF(xvilvh_b, xvilvh, DF_BYTE)
TRANS_LSX_3XR_DF(xvilvh_h, xvilvh, DF_HALF)
TRANS_LSX_3XR_DF(xvilvh_w, xvilvh, DF_WORD)
TRANS_LSX_3XR_DF(xvilvh_d, xvilvh, DF_DOUBLE)
TRANS_LSX_3XR_DF(xvpickev_b, xvpickev, DF_BYTE)
TRANS_LSX_3XR_DF(xvpickev_h, xvpickev, DF_HALF)
TRANS_LSX_3XR_DF(xvpickev_w, xvpickev, DF_WORD)
TRANS_LSX_3XR_DF(xvpickev_d, xvpickev, DF_DOUBLE)
TRANS_LSX_3XR_DF(xvpickod_b, xvpickod, DF_BYTE)
TRANS_LSX_3XR_DF(xvpickod_h, xvpickod, DF_HALF)
TRANS_LSX_3XR_DF(xvpickod_w, xvpickod, DF_WORD)
TRANS_LSX_3XR_DF(xvpickod_d, xvpickod, DF_DOUBLE)
#undef TRANS_LSX_3XR_DF



#define TRANS_LSX_XRK_DF(name, op, df)                              \
static bool trans_ ## name(DisasContext *ctx, arg_ ## name *a)      \
{                                                                   \
    TCGv_i32 twd = tcg_const_i32(a->xd);                            \
    TCGv_i32 tws = tcg_const_i32(a->xj);                            \
    TCGv_i32 trk = tcg_const_i32(a->rk);                            \
    TCGv_i32 tdf = tcg_const_i32(df);                               \
    gen_helper_lsx_ ## op ## _df(cpu_env, tdf, twd, tws, trk);      \
    tcg_temp_free_i32(twd);                                         \
    tcg_temp_free_i32(tws);                                         \
    tcg_temp_free_i32(trk);                                         \
    tcg_temp_free_i32(tdf);                                         \
    return true;                                                    \
}

TRANS_LSX_XRK_DF(xvreplve_b, xvreplve, DF_BYTE)
TRANS_LSX_XRK_DF(xvreplve_h, xvreplve, DF_HALF)
TRANS_LSX_XRK_DF(xvreplve_w, xvreplve, DF_WORD)
TRANS_LSX_XRK_DF(xvreplve_d, xvreplve, DF_DOUBLE)
//TRANS_LSX_XRK_DF(xvextrcol_b, xvextrcol, DF_BYTE)
//TRANS_LSX_XRK_DF(xvextrcol_h, xvextrcol, DF_HALF)
//TRANS_LSX_XRK_DF(xvextrcol_w, xvextrcol, DF_WORD)
//TRANS_LSX_XRK_DF(xvextrcol_d, xvextrcol, DF_DOUBLE)
#undef TRANS_LSX_XRK_DF



#define TRANS_LSX_3XR(name, op)                              \
static bool trans_ ## name(DisasContext *ctx, arg_ ## name *a)      \
{                                                                   \
    TCGv_i32 twd = tcg_const_i32(a->xd);                            \
    TCGv_i32 tws = tcg_const_i32(a->xj);                            \
    TCGv_i32 twt = tcg_const_i32(a->xk);                            \
    check_lasx_enabled(ctx);                                        \
    gen_helper_lsx_ ## op  (cpu_env, twd, tws, twt);                \
    tcg_temp_free_i32(twd);                                         \
    tcg_temp_free_i32(tws);                                         \
    tcg_temp_free_i32(twt);                                         \
    return true;                                                    \
}

TRANS_LSX_3XR(xvand_v, xvand_v)
TRANS_LSX_3XR(xvor_v, xvor_v)
TRANS_LSX_3XR(xvxor_v, xvxor_v)
TRANS_LSX_3XR(xvnor_v, xvnor_v)
TRANS_LSX_3XR(xvandn_v, xvandn_v)
TRANS_LSX_3XR(xvorn_v, xvorn_v)
#undef TRANS_LSX_3XR


static bool trans_vaddwev_h_b(DisasContext *ctx, arg_vaddwev_h_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vaddwev_h_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vaddwev_w_h(DisasContext *ctx, arg_vaddwev_w_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vaddwev_w_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vaddwev_d_w(DisasContext *ctx, arg_vaddwev_d_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vaddwev_d_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vaddwev_q_d(DisasContext *ctx, arg_vaddwev_q_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vaddwev_q_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsubwev_h_b(DisasContext *ctx, arg_vsubwev_h_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vsubwev_h_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsubwev_w_h(DisasContext *ctx, arg_vsubwev_w_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vsubwev_w_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vsubwev_d_w(DisasContext *ctx, arg_vsubwev_d_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vsubwev_d_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vsubwev_q_d(DisasContext *ctx, arg_vsubwev_q_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vsubwev_q_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vaddwod_h_b(DisasContext *ctx, arg_vaddwod_h_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vaddwod_h_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vaddwod_w_h(DisasContext *ctx, arg_vaddwod_w_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vaddwod_w_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vaddwod_d_w(DisasContext *ctx, arg_vaddwod_d_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vaddwod_d_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vaddwod_q_d(DisasContext *ctx, arg_vaddwod_q_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vaddwod_q_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsubwod_h_b(DisasContext *ctx, arg_vsubwod_h_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vsubwod_h_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsubwod_w_h(DisasContext *ctx, arg_vsubwod_w_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vsubwod_w_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vsubwod_d_w(DisasContext *ctx, arg_vsubwod_d_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vsubwod_d_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vsubwod_q_d(DisasContext *ctx, arg_vsubwod_q_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vsubwod_q_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvaddwev_h_b(DisasContext *ctx, arg_xvaddwev_h_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvaddwev_h_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvaddwev_w_h(DisasContext *ctx, arg_xvaddwev_w_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvaddwev_w_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_xvaddwev_d_w(DisasContext *ctx, arg_xvaddwev_d_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvaddwev_d_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_xvaddwev_q_d(DisasContext *ctx, arg_xvaddwev_q_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvaddwev_q_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvsubwev_h_b(DisasContext *ctx, arg_xvsubwev_h_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsubwev_h_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvsubwev_w_h(DisasContext *ctx, arg_xvsubwev_w_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsubwev_w_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_xvsubwev_d_w(DisasContext *ctx, arg_xvsubwev_d_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsubwev_d_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_xvsubwev_q_d(DisasContext *ctx, arg_xvsubwev_q_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsubwev_q_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_xvaddwod_h_b(DisasContext *ctx, arg_xvaddwod_h_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvaddwod_h_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvaddwod_w_h(DisasContext *ctx, arg_xvaddwod_w_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvaddwod_w_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_xvaddwod_d_w(DisasContext *ctx, arg_xvaddwod_d_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvaddwod_d_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_xvaddwod_q_d(DisasContext *ctx, arg_xvaddwod_q_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvaddwod_q_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvsubwod_h_b(DisasContext *ctx, arg_xvsubwod_h_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsubwod_h_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvsubwod_w_h(DisasContext *ctx, arg_xvsubwod_w_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsubwod_w_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_xvsubwod_d_w(DisasContext *ctx, arg_xvsubwod_d_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsubwod_d_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_xvsubwod_q_d(DisasContext *ctx, arg_xvsubwod_q_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsubwod_q_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vaddwev_h_bu(DisasContext *ctx, arg_vaddwev_h_bu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vaddwev_h_bu(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vaddwev_w_hu(DisasContext *ctx, arg_vaddwev_w_hu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vaddwev_w_hu(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vaddwev_d_wu(DisasContext *ctx, arg_vaddwev_d_wu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vaddwev_d_wu(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vaddwev_q_du(DisasContext *ctx, arg_vaddwev_q_du *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vaddwev_q_du(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsubwev_h_bu(DisasContext *ctx, arg_vsubwev_h_bu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vsubwev_h_bu(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsubwev_w_hu(DisasContext *ctx, arg_vsubwev_w_hu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vsubwev_w_hu(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vsubwev_d_wu(DisasContext *ctx, arg_vsubwev_d_wu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vsubwev_d_wu(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vsubwev_q_du(DisasContext *ctx, arg_vsubwev_q_du *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vsubwev_q_du(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vaddwod_h_bu(DisasContext *ctx, arg_vaddwod_h_bu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vaddwod_h_bu(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vaddwod_w_hu(DisasContext *ctx, arg_vaddwod_w_hu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vaddwod_w_hu(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vaddwod_d_wu(DisasContext *ctx, arg_vaddwod_d_wu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vaddwod_d_wu(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vaddwod_q_du(DisasContext *ctx, arg_vaddwod_q_du *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vaddwod_q_du(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsubwod_h_bu(DisasContext *ctx, arg_vsubwod_h_bu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vsubwod_h_bu(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsubwod_w_hu(DisasContext *ctx, arg_vsubwod_w_hu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vsubwod_w_hu(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vsubwod_d_wu(DisasContext *ctx, arg_vsubwod_d_wu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vsubwod_d_wu(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vsubwod_q_du(DisasContext *ctx, arg_vsubwod_q_du *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vsubwod_q_du(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvaddwev_h_bu(DisasContext *ctx, arg_xvaddwev_h_bu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvaddwev_h_bu(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvaddwev_w_hu(DisasContext *ctx, arg_xvaddwev_w_hu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvaddwev_w_hu(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_xvaddwev_d_wu(DisasContext *ctx, arg_xvaddwev_d_wu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvaddwev_d_wu(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_xvaddwev_q_du(DisasContext *ctx, arg_xvaddwev_q_du *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvaddwev_q_du(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvsubwev_h_bu(DisasContext *ctx, arg_xvsubwev_h_bu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsubwev_h_bu(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvsubwev_w_hu(DisasContext *ctx, arg_xvsubwev_w_hu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsubwev_w_hu(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_xvsubwev_d_wu(DisasContext *ctx, arg_xvsubwev_d_wu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsubwev_d_wu(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_xvsubwev_q_du(DisasContext *ctx, arg_xvsubwev_q_du *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsubwev_q_du(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_xvaddwod_h_bu(DisasContext *ctx, arg_xvaddwod_h_bu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvaddwod_h_bu(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvaddwod_w_hu(DisasContext *ctx, arg_xvaddwod_w_hu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvaddwod_w_hu(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_xvaddwod_d_wu(DisasContext *ctx, arg_xvaddwod_d_wu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvaddwod_d_wu(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_xvaddwod_q_du(DisasContext *ctx, arg_xvaddwod_q_du *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvaddwod_q_du(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvsubwod_h_bu(DisasContext *ctx, arg_xvsubwod_h_bu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsubwod_h_bu(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvsubwod_w_hu(DisasContext *ctx, arg_xvsubwod_w_hu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsubwod_w_hu(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_xvsubwod_d_wu(DisasContext *ctx, arg_xvsubwod_d_wu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsubwod_d_wu(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_xvsubwod_q_du(DisasContext *ctx, arg_xvsubwod_q_du *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsubwod_q_du(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vaddwev_h_bu_b(DisasContext *ctx, arg_vaddwev_h_bu_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vaddwev_h_bu_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vaddwev_w_hu_h(DisasContext *ctx, arg_vaddwev_w_hu_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vaddwev_w_hu_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vaddwev_d_wu_w(DisasContext *ctx, arg_vaddwev_d_wu_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vaddwev_d_wu_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vaddwev_q_du_d(DisasContext *ctx, arg_vaddwev_q_du_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vaddwev_q_du_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vaddwod_h_bu_b(DisasContext *ctx, arg_vaddwod_h_bu_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vaddwod_h_bu_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vaddwod_w_hu_h(DisasContext *ctx, arg_vaddwod_w_hu_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vaddwod_w_hu_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vaddwod_d_wu_w(DisasContext *ctx, arg_vaddwod_d_wu_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vaddwod_d_wu_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vaddwod_q_du_d(DisasContext *ctx, arg_vaddwod_q_du_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vaddwod_q_du_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvaddwev_h_bu_b(DisasContext *ctx, arg_xvaddwev_h_bu_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvaddwev_h_bu_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvaddwev_w_hu_h(DisasContext *ctx, arg_xvaddwev_w_hu_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvaddwev_w_hu_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_xvaddwev_d_wu_w(DisasContext *ctx, arg_xvaddwev_d_wu_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvaddwev_d_wu_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_xvaddwev_q_du_d(DisasContext *ctx, arg_xvaddwev_q_du_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvaddwev_q_du_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvaddwod_h_bu_b(DisasContext *ctx, arg_xvaddwod_h_bu_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvaddwod_h_bu_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvaddwod_w_hu_h(DisasContext *ctx, arg_xvaddwod_w_hu_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvaddwod_w_hu_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_xvaddwod_d_wu_w(DisasContext *ctx, arg_xvaddwod_d_wu_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvaddwod_d_wu_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_xvaddwod_q_du_d(DisasContext *ctx, arg_xvaddwod_q_du_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvaddwod_q_du_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsrln_b_h(DisasContext *ctx, arg_vsrln_b_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vsrln_b_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsrln_h_w(DisasContext *ctx, arg_vsrln_h_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vsrln_h_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vsrln_w_d(DisasContext *ctx, arg_vsrln_w_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vsrln_w_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsran_b_h(DisasContext *ctx, arg_vsran_b_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vsran_b_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsran_h_w(DisasContext *ctx, arg_vsran_h_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vsran_h_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vsran_w_d(DisasContext *ctx, arg_vsran_w_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vsran_w_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsrlrn_b_h(DisasContext *ctx, arg_vsrlrn_b_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vsrlrn_b_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}



static bool trans_vsrlrn_h_w(DisasContext *ctx, arg_vsrlrn_h_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vsrlrn_h_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vsrlrn_w_d(DisasContext *ctx, arg_vsrlrn_w_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vsrlrn_w_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsrarn_b_h(DisasContext *ctx, arg_vsrarn_b_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vsrarn_b_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsrarn_h_w(DisasContext *ctx, arg_vsrarn_h_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vsrarn_h_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vsrarn_w_d(DisasContext *ctx, arg_vsrarn_w_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vsrarn_w_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vssrln_b_h(DisasContext *ctx, arg_vssrln_b_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrln_b_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vssrln_h_w(DisasContext *ctx, arg_vssrln_h_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrln_h_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vssrln_w_d(DisasContext *ctx, arg_vssrln_w_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrln_w_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vssran_b_h(DisasContext *ctx, arg_vssran_b_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vssran_b_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vssran_h_w(DisasContext *ctx, arg_vssran_h_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vssran_h_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vssran_w_d(DisasContext *ctx, arg_vssran_w_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vssran_w_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vssrlrn_b_h(DisasContext *ctx, arg_vssrlrn_b_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrlrn_b_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vssrlrn_h_w(DisasContext *ctx, arg_vssrlrn_h_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrlrn_h_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vssrlrn_w_d(DisasContext *ctx, arg_vssrlrn_w_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrlrn_w_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vssrarn_b_h(DisasContext *ctx, arg_vssrarn_b_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrarn_b_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vssrarn_h_w(DisasContext *ctx, arg_vssrarn_h_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrarn_h_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vssrarn_w_d(DisasContext *ctx, arg_vssrarn_w_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrarn_w_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vssrln_bu_h(DisasContext *ctx, arg_vssrln_bu_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrln_bu_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vssrln_hu_w(DisasContext *ctx, arg_vssrln_hu_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrln_hu_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vssrln_wu_d(DisasContext *ctx, arg_vssrln_wu_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrln_wu_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vssran_bu_h(DisasContext *ctx, arg_vssran_bu_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vssran_bu_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vssran_hu_w(DisasContext *ctx, arg_vssran_hu_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vssran_hu_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vssran_wu_d(DisasContext *ctx, arg_vssran_wu_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vssran_wu_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vssrlrn_bu_h(DisasContext *ctx, arg_vssrlrn_bu_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrlrn_bu_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vssrlrn_hu_w(DisasContext *ctx, arg_vssrlrn_hu_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrlrn_hu_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vssrlrn_wu_d(DisasContext *ctx, arg_vssrlrn_wu_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrlrn_wu_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vssrarn_bu_h(DisasContext *ctx, arg_vssrarn_bu_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrarn_bu_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vssrarn_hu_w(DisasContext *ctx, arg_vssrarn_hu_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrarn_hu_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vssrarn_wu_d(DisasContext *ctx, arg_vssrarn_wu_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrarn_wu_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_xvsrln_b_h(DisasContext *ctx, arg_xvsrln_b_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsrln_b_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvsrln_h_w(DisasContext *ctx, arg_xvsrln_h_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsrln_h_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_xvsrln_w_d(DisasContext *ctx, arg_xvsrln_w_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsrln_w_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvsran_b_h(DisasContext *ctx, arg_xvsran_b_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsran_b_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvsran_h_w(DisasContext *ctx, arg_xvsran_h_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsran_h_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_xvsran_w_d(DisasContext *ctx, arg_xvsran_w_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsran_w_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvsrlrn_b_h(DisasContext *ctx, arg_xvsrlrn_b_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsrlrn_b_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}



static bool trans_xvsrlrn_h_w(DisasContext *ctx, arg_xvsrlrn_h_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsrlrn_h_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_xvsrlrn_w_d(DisasContext *ctx, arg_xvsrlrn_w_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsrlrn_w_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvsrarn_b_h(DisasContext *ctx, arg_xvsrarn_b_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsrarn_b_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvsrarn_h_w(DisasContext *ctx, arg_xvsrarn_h_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsrarn_h_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_xvsrarn_w_d(DisasContext *ctx, arg_xvsrarn_w_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsrarn_w_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvssrln_b_h(DisasContext *ctx, arg_xvssrln_b_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrln_b_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvssrln_h_w(DisasContext *ctx, arg_xvssrln_h_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrln_h_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_xvssrln_w_d(DisasContext *ctx, arg_xvssrln_w_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrln_w_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvssran_b_h(DisasContext *ctx, arg_xvssran_b_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssran_b_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvssran_h_w(DisasContext *ctx, arg_xvssran_h_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssran_h_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_xvssran_w_d(DisasContext *ctx, arg_xvssran_w_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssran_w_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvssrlrn_b_h(DisasContext *ctx, arg_xvssrlrn_b_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrlrn_b_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvssrlrn_h_w(DisasContext *ctx, arg_xvssrlrn_h_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrlrn_h_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_xvssrlrn_w_d(DisasContext *ctx, arg_xvssrlrn_w_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrlrn_w_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvssrarn_b_h(DisasContext *ctx, arg_xvssrarn_b_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrarn_b_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvssrarn_h_w(DisasContext *ctx, arg_xvssrarn_h_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrarn_h_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_xvssrarn_w_d(DisasContext *ctx, arg_xvssrarn_w_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrarn_w_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvssrln_bu_h(DisasContext *ctx, arg_xvssrln_bu_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrln_bu_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvssrln_hu_w(DisasContext *ctx, arg_xvssrln_hu_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrln_hu_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_xvssrln_wu_d(DisasContext *ctx, arg_xvssrln_wu_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrln_wu_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvssran_bu_h(DisasContext *ctx, arg_xvssran_bu_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssran_bu_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvssran_hu_w(DisasContext *ctx, arg_xvssran_hu_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssran_hu_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_xvssran_wu_d(DisasContext *ctx, arg_xvssran_wu_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssran_wu_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvssrlrn_bu_h(DisasContext *ctx, arg_xvssrlrn_bu_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrlrn_bu_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvssrlrn_hu_w(DisasContext *ctx, arg_xvssrlrn_hu_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrlrn_hu_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_xvssrlrn_wu_d(DisasContext *ctx, arg_xvssrlrn_wu_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrlrn_wu_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvssrarn_bu_h(DisasContext *ctx, arg_xvssrarn_bu_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrarn_bu_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvssrarn_hu_w(DisasContext *ctx, arg_xvssrarn_hu_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrarn_hu_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_xvssrarn_wu_d(DisasContext *ctx, arg_xvssrarn_wu_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrarn_wu_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}


static bool trans_vsllwil_h_b(DisasContext *ctx, arg_vsllwil_h_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui3);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vsllwil_h_b(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_vsllwil_w_h(DisasContext *ctx, arg_vsllwil_w_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui4);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vsllwil_w_h(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_vsllwil_d_w(DisasContext *ctx, arg_vsllwil_d_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui5);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vsllwil_d_w(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_vsllwil_hu_bu(DisasContext *ctx, arg_vsllwil_hu_bu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui3);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vsllwil_hu_bu(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_vsllwil_wu_hu(DisasContext *ctx, arg_vsllwil_wu_hu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui4);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vsllwil_wu_hu(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_vsllwil_du_wu(DisasContext *ctx, arg_vsllwil_du_wu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui5);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vsllwil_du_wu(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_vextl_q_d(DisasContext *ctx, arg_vextl_q_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vextl_q_d(cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_vextl_qu_du(DisasContext *ctx, arg_vextl_qu_du *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vextl_qu_du(cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_vexth_h_b(DisasContext *ctx, arg_vexth_h_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vexth_h_b(cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_vexth_w_h(DisasContext *ctx, arg_vexth_w_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vexth_w_h(cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_vexth_d_w(DisasContext *ctx, arg_vexth_d_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vexth_d_w(cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_vexth_q_d(DisasContext *ctx, arg_vexth_q_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vexth_q_d(cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_vexth_hu_bu(DisasContext *ctx, arg_vexth_hu_bu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vexth_hu_bu(cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_vexth_wu_hu(DisasContext *ctx, arg_vexth_wu_hu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vexth_wu_hu(cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_vexth_du_wu(DisasContext *ctx, arg_vexth_du_wu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vexth_du_wu(cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_vexth_qu_du(DisasContext *ctx, arg_vexth_qu_du *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vexth_qu_du(cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_xvsllwil_h_b(DisasContext *ctx, arg_xvsllwil_h_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui3);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsllwil_h_b(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_xvsllwil_w_h(DisasContext *ctx, arg_xvsllwil_w_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui4);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsllwil_w_h(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_xvsllwil_d_w(DisasContext *ctx, arg_xvsllwil_d_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui5);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsllwil_d_w(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_xvsllwil_hu_bu(DisasContext *ctx, arg_xvsllwil_hu_bu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui3);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsllwil_hu_bu(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_xvsllwil_wu_hu(DisasContext *ctx, arg_xvsllwil_wu_hu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui4);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsllwil_wu_hu(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_xvsllwil_du_wu(DisasContext *ctx, arg_xvsllwil_du_wu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui5);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsllwil_du_wu(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_xvextl_q_d(DisasContext *ctx, arg_xvextl_q_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvextl_q_d(cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_xvextl_qu_du(DisasContext *ctx, arg_xvextl_qu_du *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvextl_qu_du(cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_xvexth_h_b(DisasContext *ctx, arg_xvexth_h_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvexth_h_b(cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_xvexth_w_h(DisasContext *ctx, arg_xvexth_w_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvexth_w_h(cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_xvexth_d_w(DisasContext *ctx, arg_xvexth_d_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvexth_d_w(cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_xvexth_q_d(DisasContext *ctx, arg_xvexth_q_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvexth_q_d(cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_xvexth_hu_bu(DisasContext *ctx, arg_xvexth_hu_bu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvexth_hu_bu(cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_xvexth_wu_hu(DisasContext *ctx, arg_xvexth_wu_hu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvexth_wu_hu(cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_xvexth_du_wu(DisasContext *ctx, arg_xvexth_du_wu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvexth_du_wu(cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_xvexth_qu_du(DisasContext *ctx, arg_xvexth_qu_du *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvexth_qu_du(cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}


static bool trans_vsrlrneni_b_h(DisasContext *ctx, arg_vsrlrneni_b_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui4);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vsrlrneni_b_h(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_vsrlrneni_h_w(DisasContext *ctx, arg_vsrlrneni_h_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui5);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vsrlrneni_h_w(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_vsrlrneni_w_d(DisasContext *ctx, arg_vsrlrneni_w_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui6);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vsrlrneni_w_d(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_vsrlrneni_d_q(DisasContext *ctx, arg_vsrlrneni_d_q *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui7);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vsrlrneni_d_q(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_vsrarneni_b_h(DisasContext *ctx, arg_vsrarneni_b_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui4);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vsrarneni_b_h(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_vsrarneni_h_w(DisasContext *ctx, arg_vsrarneni_h_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui5);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vsrarneni_h_w(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_vsrarneni_w_d(DisasContext *ctx, arg_vsrarneni_w_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui6);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vsrarneni_w_d(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_vsrarneni_d_q(DisasContext *ctx, arg_vsrarneni_d_q *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui7);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vsrarneni_d_q(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_vsrlni_b_h(DisasContext *ctx, arg_vsrlni_b_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui4);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vsrlni_b_h(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_vsrlni_h_w(DisasContext *ctx, arg_vsrlni_h_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui5);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vsrlni_h_w(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_vsrlni_w_d(DisasContext *ctx, arg_vsrlni_w_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui6);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vsrlni_w_d(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_vsrlni_d_q(DisasContext *ctx, arg_vsrlni_d_q *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui7);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vsrlni_d_q(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_vsrlrni_b_h(DisasContext *ctx, arg_vsrlrni_b_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui4);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vsrlrni_b_h(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_vsrlrni_h_w(DisasContext *ctx, arg_vsrlrni_h_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui5);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vsrlrni_h_w(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_vsrlrni_w_d(DisasContext *ctx, arg_vsrlrni_w_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui6);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vsrlrni_w_d(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_vsrlrni_d_q(DisasContext *ctx, arg_vsrlrni_d_q *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui7);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vsrlrni_d_q(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_vssrlni_b_h(DisasContext *ctx, arg_vssrlni_b_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui4);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrlni_b_h(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_vssrlni_h_w(DisasContext *ctx, arg_vssrlni_h_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui5);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrlni_h_w(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_vssrlni_w_d(DisasContext *ctx, arg_vssrlni_w_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui6);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrlni_w_d(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_vssrlni_d_q(DisasContext *ctx, arg_vssrlni_d_q *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui7);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrlni_d_q(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_vssrlni_bu_h(DisasContext *ctx, arg_vssrlni_bu_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui4);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrlni_bu_h(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_vssrlni_hu_w(DisasContext *ctx, arg_vssrlni_hu_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui5);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrlni_hu_w(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_vssrlni_wu_d(DisasContext *ctx, arg_vssrlni_wu_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui6);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrlni_wu_d(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_vssrlni_du_q(DisasContext *ctx, arg_vssrlni_du_q *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui7);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrlni_du_q(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_vssrlrni_b_h(DisasContext *ctx, arg_vssrlrni_b_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui4);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrlrni_b_h(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_vssrlrni_h_w(DisasContext *ctx, arg_vssrlrni_h_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui5);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrlrni_h_w(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_vssrlrni_w_d(DisasContext *ctx, arg_vssrlrni_w_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui6);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrlrni_w_d(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_vssrlrni_d_q(DisasContext *ctx, arg_vssrlrni_d_q *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui7);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrlrni_d_q(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_vssrlrni_bu_h(DisasContext *ctx, arg_vssrlrni_bu_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui4);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrlrni_bu_h(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_vssrlrni_hu_w(DisasContext *ctx, arg_vssrlrni_hu_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui5);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrlrni_hu_w(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_vssrlrni_wu_d(DisasContext *ctx, arg_vssrlrni_wu_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui6);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrlrni_wu_d(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_vssrlrni_du_q(DisasContext *ctx, arg_vssrlrni_du_q *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui7);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrlrni_du_q(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_vsrani_b_h(DisasContext *ctx, arg_vsrani_b_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui4);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vsrani_b_h(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_vsrani_h_w(DisasContext *ctx, arg_vsrani_h_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui5);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vsrani_h_w(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_vsrani_w_d(DisasContext *ctx, arg_vsrani_w_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui6);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vsrani_w_d(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_vsrani_d_q(DisasContext *ctx, arg_vsrani_d_q *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui7);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vsrani_d_q(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_vsrarni_b_h(DisasContext *ctx, arg_vsrarni_b_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui4);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vsrarni_b_h(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_vsrarni_h_w(DisasContext *ctx, arg_vsrarni_h_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui5);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vsrarni_h_w(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_vsrarni_w_d(DisasContext *ctx, arg_vsrarni_w_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui6);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vsrarni_w_d(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_vsrarni_d_q(DisasContext *ctx, arg_vsrarni_d_q *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui7);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vsrarni_d_q(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_vssrani_b_h(DisasContext *ctx, arg_vssrani_b_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui4);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrani_b_h(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_vssrani_h_w(DisasContext *ctx, arg_vssrani_h_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui5);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrani_h_w(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_vssrani_w_d(DisasContext *ctx, arg_vssrani_w_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui6);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrani_w_d(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_vssrani_d_q(DisasContext *ctx, arg_vssrani_d_q *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui7);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrani_d_q(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_vssrani_bu_h(DisasContext *ctx, arg_vssrani_bu_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui4);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrani_bu_h(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_vssrani_hu_w(DisasContext *ctx, arg_vssrani_hu_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui5);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrani_hu_w(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_vssrani_wu_d(DisasContext *ctx, arg_vssrani_wu_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui6);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrani_wu_d(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_vssrani_du_q(DisasContext *ctx, arg_vssrani_du_q *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui7);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrani_du_q(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_vssrarni_b_h(DisasContext *ctx, arg_vssrarni_b_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui4);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrarni_b_h(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_vssrarni_h_w(DisasContext *ctx, arg_vssrarni_h_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui5);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrarni_h_w(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_vssrarni_w_d(DisasContext *ctx, arg_vssrarni_w_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui6);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrarni_w_d(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_vssrarni_d_q(DisasContext *ctx, arg_vssrarni_d_q *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui7);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrarni_d_q(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_vssrarni_bu_h(DisasContext *ctx, arg_vssrarni_bu_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui4);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrarni_bu_h(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_vssrarni_hu_w(DisasContext *ctx, arg_vssrarni_hu_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui5);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrarni_hu_w(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_vssrarni_wu_d(DisasContext *ctx, arg_vssrarni_wu_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui6);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrarni_wu_d(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_vssrarni_du_q(DisasContext *ctx, arg_vssrarni_du_q *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui7);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrarni_du_q(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_vssrlrneni_b_h(DisasContext *ctx, arg_vssrlrneni_b_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui4);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrlrneni_b_h(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_vssrlrneni_h_w(DisasContext *ctx, arg_vssrlrneni_h_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui5);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrlrneni_h_w(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_vssrlrneni_w_d(DisasContext *ctx, arg_vssrlrneni_w_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui6);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrlrneni_w_d(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_vssrlrneni_d_q(DisasContext *ctx, arg_vssrlrneni_d_q *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui7);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrlrneni_d_q(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_vssrlrneni_bu_h(DisasContext *ctx, arg_vssrlrneni_bu_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui4);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrlrneni_bu_h(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_vssrlrneni_hu_w(DisasContext *ctx, arg_vssrlrneni_hu_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui5);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrlrneni_hu_w(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_vssrlrneni_wu_d(DisasContext *ctx, arg_vssrlrneni_wu_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui6);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrlrneni_wu_d(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_vssrlrneni_du_q(DisasContext *ctx, arg_vssrlrneni_du_q *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui7);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrlrneni_du_q(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_vssrarneni_b_h(DisasContext *ctx, arg_vssrarneni_b_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui4);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrarneni_b_h(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_vssrarneni_h_w(DisasContext *ctx, arg_vssrarneni_h_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui5);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrarneni_h_w(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_vssrarneni_w_d(DisasContext *ctx, arg_vssrarneni_w_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui6);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrarneni_w_d(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_vssrarneni_d_q(DisasContext *ctx, arg_vssrarneni_d_q *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui7);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrarneni_d_q(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_vssrarneni_bu_h(DisasContext *ctx, arg_vssrarneni_bu_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui4);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrarneni_bu_h(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_vssrarneni_hu_w(DisasContext *ctx, arg_vssrarneni_hu_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui5);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrarneni_hu_w(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_vssrarneni_wu_d(DisasContext *ctx, arg_vssrarneni_wu_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui6);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrarneni_wu_d(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_vssrarneni_du_q(DisasContext *ctx, arg_vssrarneni_du_q *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui7);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vssrarneni_du_q(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

//----------------------xv---/
static bool trans_xvsrlrneni_b_h(DisasContext *ctx, arg_xvsrlrneni_b_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui4);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsrlrneni_b_h(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_xvsrlrneni_h_w(DisasContext *ctx, arg_xvsrlrneni_h_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui5);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsrlrneni_h_w(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_xvsrlrneni_w_d(DisasContext *ctx, arg_xvsrlrneni_w_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui6);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsrlrneni_w_d(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_xvsrlrneni_d_q(DisasContext *ctx, arg_xvsrlrneni_d_q *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui7);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsrlrneni_d_q(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_xvsrarneni_b_h(DisasContext *ctx, arg_xvsrarneni_b_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui4);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsrarneni_b_h(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_xvsrarneni_h_w(DisasContext *ctx, arg_xvsrarneni_h_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui5);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsrarneni_h_w(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_xvsrarneni_w_d(DisasContext *ctx, arg_xvsrarneni_w_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui6);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsrarneni_w_d(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_xvsrarneni_d_q(DisasContext *ctx, arg_xvsrarneni_d_q *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui7);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsrarneni_d_q(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_xvsrlni_b_h(DisasContext *ctx, arg_xvsrlni_b_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui4);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsrlni_b_h(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_xvsrlni_h_w(DisasContext *ctx, arg_xvsrlni_h_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui5);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsrlni_h_w(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_xvsrlni_w_d(DisasContext *ctx, arg_xvsrlni_w_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui6);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsrlni_w_d(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_xvsrlni_d_q(DisasContext *ctx, arg_xvsrlni_d_q *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui7);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsrlni_d_q(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_xvsrlrni_b_h(DisasContext *ctx, arg_xvsrlrni_b_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui4);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsrlrni_b_h(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_xvsrlrni_h_w(DisasContext *ctx, arg_xvsrlrni_h_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui5);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsrlrni_h_w(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_xvsrlrni_w_d(DisasContext *ctx, arg_xvsrlrni_w_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui6);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsrlrni_w_d(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_xvsrlrni_d_q(DisasContext *ctx, arg_xvsrlrni_d_q *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui7);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsrlrni_d_q(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_xvssrlni_b_h(DisasContext *ctx, arg_xvssrlni_b_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui4);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrlni_b_h(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_xvssrlni_h_w(DisasContext *ctx, arg_xvssrlni_h_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui5);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrlni_h_w(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_xvssrlni_w_d(DisasContext *ctx, arg_xvssrlni_w_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui6);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrlni_w_d(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_xvssrlni_d_q(DisasContext *ctx, arg_xvssrlni_d_q *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui7);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrlni_d_q(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_xvssrlni_bu_h(DisasContext *ctx, arg_xvssrlni_bu_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui4);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrlni_bu_h(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_xvssrlni_hu_w(DisasContext *ctx, arg_xvssrlni_hu_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui5);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrlni_hu_w(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_xvssrlni_wu_d(DisasContext *ctx, arg_xvssrlni_wu_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui6);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrlni_wu_d(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_xvssrlni_du_q(DisasContext *ctx, arg_xvssrlni_du_q *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui7);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrlni_du_q(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_xvssrlrni_b_h(DisasContext *ctx, arg_xvssrlrni_b_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui4);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrlrni_b_h(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_xvssrlrni_h_w(DisasContext *ctx, arg_xvssrlrni_h_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui5);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrlrni_h_w(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_xvssrlrni_w_d(DisasContext *ctx, arg_xvssrlrni_w_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui6);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrlrni_w_d(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_xvssrlrni_d_q(DisasContext *ctx, arg_xvssrlrni_d_q *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui7);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrlrni_d_q(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_xvssrlrni_bu_h(DisasContext *ctx, arg_xvssrlrni_bu_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui4);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrlrni_bu_h(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_xvssrlrni_hu_w(DisasContext *ctx, arg_xvssrlrni_hu_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui5);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrlrni_hu_w(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_xvssrlrni_wu_d(DisasContext *ctx, arg_xvssrlrni_wu_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui6);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrlrni_wu_d(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_xvssrlrni_du_q(DisasContext *ctx, arg_xvssrlrni_du_q *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui7);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrlrni_du_q(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_xvsrani_b_h(DisasContext *ctx, arg_xvsrani_b_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui4);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsrani_b_h(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_xvsrani_h_w(DisasContext *ctx, arg_xvsrani_h_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui5);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsrani_h_w(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_xvsrani_w_d(DisasContext *ctx, arg_xvsrani_w_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui6);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsrani_w_d(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_xvsrani_d_q(DisasContext *ctx, arg_xvsrani_d_q *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui7);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsrani_d_q(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_xvsrarni_b_h(DisasContext *ctx, arg_xvsrarni_b_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui4);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsrarni_b_h(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_xvsrarni_h_w(DisasContext *ctx, arg_xvsrarni_h_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui5);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsrarni_h_w(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_xvsrarni_w_d(DisasContext *ctx, arg_xvsrarni_w_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui6);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsrarni_w_d(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_xvsrarni_d_q(DisasContext *ctx, arg_xvsrarni_d_q *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui7);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsrarni_d_q(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_xvssrani_b_h(DisasContext *ctx, arg_xvssrani_b_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui4);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrani_b_h(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_xvssrani_h_w(DisasContext *ctx, arg_xvssrani_h_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui5);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrani_h_w(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_xvssrani_w_d(DisasContext *ctx, arg_xvssrani_w_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui6);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrani_w_d(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_xvssrani_d_q(DisasContext *ctx, arg_xvssrani_d_q *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui7);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrani_d_q(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_xvssrani_bu_h(DisasContext *ctx, arg_xvssrani_bu_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui4);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrani_bu_h(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_xvssrani_hu_w(DisasContext *ctx, arg_xvssrani_hu_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui5);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrani_hu_w(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_xvssrani_wu_d(DisasContext *ctx, arg_xvssrani_wu_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui6);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrani_wu_d(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_xvssrani_du_q(DisasContext *ctx, arg_xvssrani_du_q *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui7);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrani_du_q(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_xvssrarni_b_h(DisasContext *ctx, arg_xvssrarni_b_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui4);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrarni_b_h(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_xvssrarni_h_w(DisasContext *ctx, arg_xvssrarni_h_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui5);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrarni_h_w(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_xvssrarni_w_d(DisasContext *ctx, arg_xvssrarni_w_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui6);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrarni_w_d(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_xvssrarni_d_q(DisasContext *ctx, arg_xvssrarni_d_q *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui7);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrarni_d_q(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_xvssrarni_bu_h(DisasContext *ctx, arg_xvssrarni_bu_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui4);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrarni_bu_h(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_xvssrarni_hu_w(DisasContext *ctx, arg_xvssrarni_hu_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui5);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrarni_hu_w(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_xvssrarni_wu_d(DisasContext *ctx, arg_xvssrarni_wu_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui6);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrarni_wu_d(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_xvssrarni_du_q(DisasContext *ctx, arg_xvssrarni_du_q *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui7);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrarni_du_q(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_xvssrlrneni_b_h(DisasContext *ctx, arg_xvssrlrneni_b_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui4);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrlrneni_b_h(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_xvssrlrneni_h_w(DisasContext *ctx, arg_xvssrlrneni_h_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui5);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrlrneni_h_w(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_xvssrlrneni_w_d(DisasContext *ctx, arg_xvssrlrneni_w_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui6);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrlrneni_w_d(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_xvssrlrneni_d_q(DisasContext *ctx, arg_xvssrlrneni_d_q *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui7);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrlrneni_d_q(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_xvssrlrneni_bu_h(DisasContext *ctx, arg_xvssrlrneni_bu_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui4);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrlrneni_bu_h(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_xvssrlrneni_hu_w(DisasContext *ctx, arg_xvssrlrneni_hu_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui5);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrlrneni_hu_w(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_xvssrlrneni_wu_d(DisasContext *ctx, arg_xvssrlrneni_wu_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui6);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrlrneni_wu_d(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_xvssrlrneni_du_q(DisasContext *ctx, arg_xvssrlrneni_du_q *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui7);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrlrneni_du_q(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_xvssrarneni_b_h(DisasContext *ctx, arg_xvssrarneni_b_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui4);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrarneni_b_h(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_xvssrarneni_h_w(DisasContext *ctx, arg_xvssrarneni_h_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui5);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrarneni_h_w(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_xvssrarneni_w_d(DisasContext *ctx, arg_xvssrarneni_w_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui6);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrarneni_w_d(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_xvssrarneni_d_q(DisasContext *ctx, arg_xvssrarneni_d_q *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui7);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrarneni_d_q(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_xvssrarneni_bu_h(DisasContext *ctx, arg_xvssrarneni_bu_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui4);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrarneni_bu_h(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_xvssrarneni_hu_w(DisasContext *ctx, arg_xvssrarneni_hu_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui5);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrarneni_hu_w(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_xvssrarneni_wu_d(DisasContext *ctx, arg_xvssrarneni_wu_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui6);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrarneni_wu_d(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}


static bool trans_xvssrarneni_du_q(DisasContext *ctx, arg_xvssrarneni_du_q *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->ui7);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvssrarneni_du_q(cpu_env, twd, tws, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tui);
    return true;
}

















#define TRANS_MUL_3WR(name)                                         \
static bool trans_ ## name(DisasContext *ctx, arg_ ## name *a)      \
{                                                                   \
    TCGv_i32 twd = tcg_const_i32(a->vd);                            \
    TCGv_i32 tws = tcg_const_i32(a->vj);                            \
    TCGv_i32 twt = tcg_const_i32(a->vk);                            \
                                                                    \
    gen_helper_lsx_ ## name(cpu_env, twd, tws, twt);                \
                                                                    \
    tcg_temp_free_i32(twd);                                         \
    tcg_temp_free_i32(tws);                                         \
    tcg_temp_free_i32(twt);                                         \
    return true;                                                    \
}

#define TRANS_MUL_3XR(name)                                         \
static bool trans_ ## name(DisasContext *ctx, arg_ ## name *a)      \
{                                                                   \
    TCGv_i32 twd = tcg_const_i32(a->xd);                            \
    TCGv_i32 tws = tcg_const_i32(a->xj);                            \
    TCGv_i32 twt = tcg_const_i32(a->xk);                            \
                                                                    \
    gen_helper_lsx_ ## name(cpu_env, twd, tws, twt);                \
                                                                    \
    tcg_temp_free_i32(twd);                                         \
    tcg_temp_free_i32(tws);                                         \
    tcg_temp_free_i32(twt);                                         \
    return true;                                                    \
}

TRANS_MUL_3WR(vmuh_b)
TRANS_MUL_3WR(vmuh_h)
TRANS_MUL_3WR(vmuh_w)
TRANS_MUL_3WR(vmuh_d)
TRANS_MUL_3WR(vmuh_bu)
TRANS_MUL_3WR(vmuh_hu)
TRANS_MUL_3WR(vmuh_wu)
TRANS_MUL_3WR(vmuh_du)
//TRANS_MUL_3WR(vmuh_bu_b)
//TRANS_MUL_3WR(vmuh_hu_h)
//TRANS_MUL_3WR(vmuh_wu_w)
//TRANS_MUL_3WR(vmuh_du_d)
//TRANS_MUL_3WR(vmulxw_h_b)
//TRANS_MUL_3WR(vmulxw_w_h)
//TRANS_MUL_3WR(vmulxw_d_w)
//TRANS_MUL_3WR(vmulxw_h_bu)
//TRANS_MUL_3WR(vmulxw_w_hu)
//TRANS_MUL_3WR(vmulxw_d_wu)

TRANS_MUL_3XR(xvmuh_b)
TRANS_MUL_3XR(xvmuh_h)
TRANS_MUL_3XR(xvmuh_w)
TRANS_MUL_3XR(xvmuh_d)
TRANS_MUL_3XR(xvmuh_bu)
TRANS_MUL_3XR(xvmuh_hu)
TRANS_MUL_3XR(xvmuh_wu)
TRANS_MUL_3XR(xvmuh_du)
//TRANS_MUL_3XR(xvmuh_bu_b)
//TRANS_MUL_3XR(xvmuh_hu_h)
//TRANS_MUL_3XR(xvmuh_wu_w)
//TRANS_MUL_3XR(xvmuh_du_d)
//TRANS_MUL_3XR(xvmulxw_h_b)
//TRANS_MUL_3XR(xvmulxw_w_h)
//TRANS_MUL_3XR(xvmulxw_d_w)
//TRANS_MUL_3XR(xvmulxw_h_bu)
//TRANS_MUL_3XR(xvmulxw_w_hu)
//TRANS_MUL_3XR(xvmulxw_d_wu)

TRANS_MUL_3WR(vmulwev_h_b)
TRANS_MUL_3WR(vmulwev_w_h)
TRANS_MUL_3WR(vmulwev_d_w)
TRANS_MUL_3WR(vmulwev_q_d)
TRANS_MUL_3WR(vmulwod_h_b)
TRANS_MUL_3WR(vmulwod_w_h)
TRANS_MUL_3WR(vmulwod_d_w)
TRANS_MUL_3WR(vmulwod_q_d)
//TRANS_MUL_3WR(vmulwl_h_b)
//TRANS_MUL_3WR(vmulwl_w_h)
//TRANS_MUL_3WR(vmulwl_d_w)
//TRANS_MUL_3WR(vmulwl_q_d)
//TRANS_MUL_3WR(vmulwh_h_b)
//TRANS_MUL_3WR(vmulwh_w_h)
//TRANS_MUL_3WR(vmulwh_d_w)
//TRANS_MUL_3WR(vmulwh_q_d)
TRANS_MUL_3WR(vmulwev_h_bu)
TRANS_MUL_3WR(vmulwev_w_hu)
TRANS_MUL_3WR(vmulwev_d_wu)
TRANS_MUL_3WR(vmulwev_q_du)
TRANS_MUL_3WR(vmulwod_h_bu)
TRANS_MUL_3WR(vmulwod_w_hu)
TRANS_MUL_3WR(vmulwod_d_wu)
TRANS_MUL_3WR(vmulwod_q_du)
//TRANS_MUL_3WR(vmulwl_h_bu)
//TRANS_MUL_3WR(vmulwl_w_hu)
//TRANS_MUL_3WR(vmulwl_d_wu)
//TRANS_MUL_3WR(vmulwl_q_du)
//TRANS_MUL_3WR(vmulwh_h_bu)
//TRANS_MUL_3WR(vmulwh_w_hu)
//TRANS_MUL_3WR(vmulwh_d_wu)
//TRANS_MUL_3WR(vmulwh_q_du)
TRANS_MUL_3WR(vmulwev_h_bu_b)
TRANS_MUL_3WR(vmulwev_w_hu_h)
TRANS_MUL_3WR(vmulwev_d_wu_w)
TRANS_MUL_3WR(vmulwev_q_du_d)
TRANS_MUL_3WR(vmulwod_h_bu_b)
TRANS_MUL_3WR(vmulwod_w_hu_h)
TRANS_MUL_3WR(vmulwod_d_wu_w)
TRANS_MUL_3WR(vmulwod_q_du_d)
//TRANS_MUL_3WR(vmulwl_h_bu_b)
//TRANS_MUL_3WR(vmulwl_w_hu_h)
//TRANS_MUL_3WR(vmulwl_d_wu_w)
//TRANS_MUL_3WR(vmulwl_q_du_d)
//TRANS_MUL_3WR(vmulwh_h_bu_b)
//TRANS_MUL_3WR(vmulwh_w_hu_h)
//TRANS_MUL_3WR(vmulwh_d_wu_w)
//TRANS_MUL_3WR(vmulwh_q_du_d)

TRANS_MUL_3XR(xvmulwev_h_b)
TRANS_MUL_3XR(xvmulwev_w_h)
TRANS_MUL_3XR(xvmulwev_d_w)
TRANS_MUL_3XR(xvmulwev_q_d)
TRANS_MUL_3XR(xvmulwod_h_b)
TRANS_MUL_3XR(xvmulwod_w_h)
TRANS_MUL_3XR(xvmulwod_d_w)
TRANS_MUL_3XR(xvmulwod_q_d)
//TRANS_MUL_3XR(xvmulwl_h_b)
//TRANS_MUL_3XR(xvmulwl_w_h)
//TRANS_MUL_3XR(xvmulwl_d_w)
//TRANS_MUL_3XR(xvmulwl_q_d)
//TRANS_MUL_3XR(xvmulwh_h_b)
//TRANS_MUL_3XR(xvmulwh_w_h)
//TRANS_MUL_3XR(xvmulwh_d_w)
//TRANS_MUL_3XR(xvmulwh_q_d)
TRANS_MUL_3XR(xvmulwev_h_bu)
TRANS_MUL_3XR(xvmulwev_w_hu)
TRANS_MUL_3XR(xvmulwev_d_wu)
TRANS_MUL_3XR(xvmulwev_q_du)
TRANS_MUL_3XR(xvmulwod_h_bu)
TRANS_MUL_3XR(xvmulwod_w_hu)
TRANS_MUL_3XR(xvmulwod_d_wu)
TRANS_MUL_3XR(xvmulwod_q_du)
//TRANS_MUL_3XR(xvmulwl_h_bu)
//TRANS_MUL_3XR(xvmulwl_w_hu)
//TRANS_MUL_3XR(xvmulwl_d_wu)
//TRANS_MUL_3XR(xvmulwl_q_du)
//TRANS_MUL_3XR(xvmulwh_h_bu)
//TRANS_MUL_3XR(xvmulwh_w_hu)
//TRANS_MUL_3XR(xvmulwh_d_wu)
//TRANS_MUL_3XR(xvmulwh_q_du)
//TRANS_MUL_3XR(xvmulwev_h_bu_b)
//TRANS_MUL_3XR(xvmulwev_w_hu_h)
//TRANS_MUL_3XR(xvmulwev_d_wu_w)
//TRANS_MUL_3XR(xvmulwev_q_du_d)
//TRANS_MUL_3XR(xvmulwod_h_bu_b)
//TRANS_MUL_3XR(xvmulwod_w_hu_h)
//TRANS_MUL_3XR(xvmulwod_d_wu_w)
//TRANS_MUL_3XR(xvmulwod_q_du_d)
//TRANS_MUL_3XR(xvmulwl_h_bu_b)
//TRANS_MUL_3XR(xvmulwl_w_hu_h)
//TRANS_MUL_3XR(xvmulwl_d_wu_w)
//TRANS_MUL_3XR(xvmulwl_q_du_d)
//TRANS_MUL_3XR(xvmulwh_h_bu_b)
//TRANS_MUL_3XR(xvmulwh_w_hu_h)
//TRANS_MUL_3XR(xvmulwh_d_wu_w)
//TRANS_MUL_3XR(xvmulwh_q_du_d)

TRANS_MUL_3WR(vmaddwev_h_b)
TRANS_MUL_3WR(vmaddwev_w_h)
TRANS_MUL_3WR(vmaddwev_d_w)
TRANS_MUL_3WR(vmaddwev_q_d)
TRANS_MUL_3WR(vmaddwod_h_b)
TRANS_MUL_3WR(vmaddwod_w_h)
TRANS_MUL_3WR(vmaddwod_d_w)
TRANS_MUL_3WR(vmaddwod_q_d)
//TRANS_MUL_3WR(vmaddwl_h_b)
//TRANS_MUL_3WR(vmaddwl_w_h)
//TRANS_MUL_3WR(vmaddwl_d_w)
//TRANS_MUL_3WR(vmaddwl_q_d)
//TRANS_MUL_3WR(vmaddwh_h_b)
//TRANS_MUL_3WR(vmaddwh_w_h)
//TRANS_MUL_3WR(vmaddwh_d_w)
//TRANS_MUL_3WR(vmaddwh_q_d)
TRANS_MUL_3WR(vmaddwev_h_bu)
TRANS_MUL_3WR(vmaddwev_w_hu)
TRANS_MUL_3WR(vmaddwev_d_wu)
TRANS_MUL_3WR(vmaddwev_q_du)
TRANS_MUL_3WR(vmaddwod_h_bu)
TRANS_MUL_3WR(vmaddwod_w_hu)
TRANS_MUL_3WR(vmaddwod_d_wu)
TRANS_MUL_3WR(vmaddwod_q_du)
//TRANS_MUL_3WR(vmaddwl_h_bu)
//TRANS_MUL_3WR(vmaddwl_w_hu)
//TRANS_MUL_3WR(vmaddwl_d_wu)
//TRANS_MUL_3WR(vmaddwl_q_du)
//TRANS_MUL_3WR(vmaddwh_h_bu)
//TRANS_MUL_3WR(vmaddwh_w_hu)
//TRANS_MUL_3WR(vmaddwh_d_wu)
//TRANS_MUL_3WR(vmaddwh_q_du)
TRANS_MUL_3WR(vmaddwev_h_bu_b)
TRANS_MUL_3WR(vmaddwev_w_hu_h)
TRANS_MUL_3WR(vmaddwev_d_wu_w)
TRANS_MUL_3WR(vmaddwev_q_du_d)
TRANS_MUL_3WR(vmaddwod_h_bu_b)
TRANS_MUL_3WR(vmaddwod_w_hu_h)
TRANS_MUL_3WR(vmaddwod_d_wu_w)
TRANS_MUL_3WR(vmaddwod_q_du_d)
//TRANS_MUL_3WR(vmaddwl_h_bu_b)
//TRANS_MUL_3WR(vmaddwl_w_hu_h)
//TRANS_MUL_3WR(vmaddwl_d_wu_w)
//TRANS_MUL_3WR(vmaddwl_q_du_d)
//TRANS_MUL_3WR(vmaddwh_h_bu_b)
//TRANS_MUL_3WR(vmaddwh_w_hu_h)
//TRANS_MUL_3WR(vmaddwh_d_wu_w)
//TRANS_MUL_3WR(vmaddwh_q_du_d)

TRANS_MUL_3XR(xvmaddwev_h_b)
TRANS_MUL_3XR(xvmaddwev_w_h)
TRANS_MUL_3XR(xvmaddwev_d_w)
TRANS_MUL_3XR(xvmaddwev_q_d)
TRANS_MUL_3XR(xvmaddwod_h_b)
TRANS_MUL_3XR(xvmaddwod_w_h)
TRANS_MUL_3XR(xvmaddwod_d_w)
TRANS_MUL_3XR(xvmaddwod_q_d)
//TRANS_MUL_3XR(xvmaddwl_h_b)
//TRANS_MUL_3XR(xvmaddwl_w_h)
//TRANS_MUL_3XR(xvmaddwl_d_w)
//TRANS_MUL_3XR(xvmaddwl_q_d)
//TRANS_MUL_3XR(xvmaddwh_h_b)
//TRANS_MUL_3XR(xvmaddwh_w_h)
//TRANS_MUL_3XR(xvmaddwh_d_w)
//TRANS_MUL_3XR(xvmaddwh_q_d)
TRANS_MUL_3XR(xvmaddwev_h_bu)
TRANS_MUL_3XR(xvmaddwev_w_hu)
TRANS_MUL_3XR(xvmaddwev_d_wu)
TRANS_MUL_3XR(xvmaddwev_q_du)
TRANS_MUL_3XR(xvmaddwod_h_bu)
TRANS_MUL_3XR(xvmaddwod_w_hu)
TRANS_MUL_3XR(xvmaddwod_d_wu)
TRANS_MUL_3XR(xvmaddwod_q_du)
//TRANS_MUL_3XR(xvmaddwl_h_bu)
//TRANS_MUL_3XR(xvmaddwl_w_hu)
//TRANS_MUL_3XR(xvmaddwl_d_wu)
//TRANS_MUL_3XR(xvmaddwl_q_du)
//TRANS_MUL_3XR(xvmaddwh_h_bu)
//TRANS_MUL_3XR(xvmaddwh_w_hu)
//TRANS_MUL_3XR(xvmaddwh_d_wu)
//TRANS_MUL_3XR(xvmaddwh_q_du)
TRANS_MUL_3XR(xvmaddwev_h_bu_b)
TRANS_MUL_3XR(xvmaddwev_w_hu_h)
TRANS_MUL_3XR(xvmaddwev_d_wu_w)
TRANS_MUL_3XR(xvmaddwev_q_du_d)
TRANS_MUL_3XR(xvmaddwod_h_bu_b)
TRANS_MUL_3XR(xvmaddwod_w_hu_h)
TRANS_MUL_3XR(xvmaddwod_d_wu_w)
TRANS_MUL_3XR(xvmaddwod_q_du_d)
//TRANS_MUL_3XR(xvmaddwl_h_bu_b)
//TRANS_MUL_3XR(xvmaddwl_w_hu_h)
//TRANS_MUL_3XR(xvmaddwl_d_wu_w)
//TRANS_MUL_3XR(xvmaddwl_q_du_d)
//TRANS_MUL_3XR(xvmaddwh_h_bu_b)
//TRANS_MUL_3XR(xvmaddwh_w_hu_h)
//TRANS_MUL_3XR(xvmaddwh_d_wu_w)
//TRANS_MUL_3XR(xvmaddwh_q_du_d)

//TRANS_MUL_3WR(vdp4_w_b)
//TRANS_MUL_3WR(vdp4_d_h)
//TRANS_MUL_3WR(vdp4_q_w)
//TRANS_MUL_3WR(vdp4_w_bu)
//TRANS_MUL_3WR(vdp4_d_hu)
//TRANS_MUL_3WR(vdp4_q_wu)
//TRANS_MUL_3WR(vdp4_w_bu_b)
//TRANS_MUL_3WR(vdp4_d_hu_h)
//TRANS_MUL_3WR(vdp4_q_wu_w)
//TRANS_MUL_3WR(vdp4add_w_b)
//TRANS_MUL_3WR(vdp4add_d_h)
//TRANS_MUL_3WR(vdp4add_q_w)
//TRANS_MUL_3WR(vdp4add_w_bu)
//TRANS_MUL_3WR(vdp4add_d_hu)
//TRANS_MUL_3WR(vdp4add_q_wu)
//TRANS_MUL_3WR(vdp4add_w_bu_b)
//TRANS_MUL_3WR(vdp4add_d_hu_h)
//TRANS_MUL_3WR(vdp4add_q_wu_w)

//TRANS_MUL_3XR(xvdp4_w_b)
//TRANS_MUL_3XR(xvdp4_d_h)
//TRANS_MUL_3XR(xvdp4_q_w)
//TRANS_MUL_3XR(xvdp4_w_bu)
//TRANS_MUL_3XR(xvdp4_d_hu)
//TRANS_MUL_3XR(xvdp4_q_wu)
//TRANS_MUL_3XR(xvdp4_w_bu_b)
//TRANS_MUL_3XR(xvdp4_d_hu_h)
//TRANS_MUL_3XR(xvdp4_q_wu_w)
//TRANS_MUL_3XR(xvdp4add_w_b)
//TRANS_MUL_3XR(xvdp4add_d_h)
//TRANS_MUL_3XR(xvdp4add_q_w)
//TRANS_MUL_3XR(xvdp4add_w_bu)
//TRANS_MUL_3XR(xvdp4add_d_hu)
//TRANS_MUL_3XR(xvdp4add_q_wu)
//TRANS_MUL_3XR(xvdp4add_w_bu_b)
//TRANS_MUL_3XR(xvdp4add_d_hu_h)
//TRANS_MUL_3XR(xvdp4add_q_wu_w)

static bool trans_vldi(DisasContext *ctx, arg_vldi *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->i13);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vldi(cpu_env, twd, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tui);
    return true;
}

static bool trans_xvldi(DisasContext *ctx, arg_xvldi *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tui = tcg_temp_new_i32();

    tcg_gen_movi_i32(tui, a->i13);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvldi(cpu_env, twd, tui);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tui);
    return true;
}

#define TRANS_VFMA(name) \
static bool trans_ ## name (DisasContext *ctx, arg_ ## name *a) \
{ \
    TCGv_i32 fp0 = tcg_const_i32(a->vj); \
    TCGv_i32 fp1 = tcg_const_i32(a->vk); \
    TCGv_i32 fp2 = tcg_const_i32(a->va); \
    TCGv_i32 fpd = tcg_const_i32(a->vd); \
    \
    check_lsx_enabled(ctx);    \
    gen_helper_lsx_ ## name (cpu_env, fpd, fp0, fp1, fp2);\
    \
    tcg_temp_free_i32(fpd); \
    tcg_temp_free_i32(fp2); \
    tcg_temp_free_i32(fp1); \
    tcg_temp_free_i32(fp0); \
    return true; \
}
TRANS_VFMA(vfmadd_s)    // trans_vfmadd_s
TRANS_VFMA(vfmadd_d)    // trans_vfmadd_d
TRANS_VFMA(vfmsub_s)    // trans_vfmsub_s
TRANS_VFMA(vfmsub_d)    // trans_vfmsub_d
TRANS_VFMA(vfnmadd_s)   // trans_vfnmadd_s
TRANS_VFMA(vfnmadd_d)   // trans_vfnmadd_d
TRANS_VFMA(vfnmsub_s)   // trans_vfnmsub_s
TRANS_VFMA(vfnmsub_d)   // trans_vfnmsub_d
#undef TRANS_VFMA

#define TRANS_XVFMA(name) \
static bool trans_ ## name (DisasContext *ctx, arg_ ## name *a) \
{ \
    TCGv_i32 fp0 = tcg_const_i32(a->xj); \
    TCGv_i32 fp1 = tcg_const_i32(a->xk); \
    TCGv_i32 fp2 = tcg_const_i32(a->xa); \
    TCGv_i32 fpd = tcg_const_i32(a->xd); \
    \
    check_lasx_enabled(ctx);    \
    gen_helper_lsx_ ## name (cpu_env, fpd, fp0, fp1, fp2);\
    \
    tcg_temp_free_i32(fpd); \
    tcg_temp_free_i32(fp2); \
    tcg_temp_free_i32(fp1); \
    tcg_temp_free_i32(fp0); \
    return true; \
}
TRANS_XVFMA(xvfmadd_s)    // trans_xvfmadd_s
TRANS_XVFMA(xvfmadd_d)    // trans_xvfmadd_d
TRANS_XVFMA(xvfmsub_s)    // trans_xvfmsub_s
TRANS_XVFMA(xvfmsub_d)    // trans_xvfmsub_d
TRANS_XVFMA(xvfnmadd_s)   // trans_xvfnmadd_s
TRANS_XVFMA(xvfnmadd_d)   // trans_xvfnmadd_d
TRANS_XVFMA(xvfnmsub_s)   // trans_xvfnmsub_s
TRANS_XVFMA(xvfnmsub_d)   // trans_xvfnmsub_d
#undef TRANS_XVFMA

#define VOP_CONDNS(fmt, ifmt)                                           \
static inline void gen_vfcmp_ ## fmt(DisasContext *ctx, int n,          \
                                      int vk, int vj, int vd)           \
{                                                                       \
    TCGv_i32 twd = tcg_const_i32(vd);                                   \
    TCGv_i32 tws = tcg_const_i32(vj);                                   \
    TCGv_i32 twt = tcg_const_i32(vk);                                   \
    TCGv_i32 tdf = tcg_const_i32(DF_ ## ifmt);                          \
    switch (n) {                                                        \
    case  0:                                                            \
        gen_helper_msa_fcaf_df(cpu_env, tdf, twd, tws, twt);            \
        break;                                                          \
    case  1:                                                            \
        gen_helper_msa_fsaf_df(cpu_env, tdf, twd, tws, twt);            \
        break;                                                          \
    case  2:                                                            \
        gen_helper_msa_fclt_df(cpu_env, tdf, twd, tws, twt);            \
        break;                                                          \
    case  3:                                                            \
        gen_helper_msa_fslt_df(cpu_env, tdf, twd, tws, twt);            \
        break;                                                          \
    case  4:                                                            \
        gen_helper_msa_fceq_df(cpu_env, tdf, twd, tws, twt);            \
        break;                                                          \
    case  5:                                                            \
        gen_helper_msa_fseq_df(cpu_env, tdf, twd, tws, twt);            \
        break;                                                          \
    case  6:                                                            \
        gen_helper_msa_fcle_df(cpu_env, tdf, twd, tws, twt);            \
        break;                                                          \
    case  7:                                                            \
        gen_helper_msa_fsle_df(cpu_env, tdf, twd, tws, twt);            \
        break;                                                          \
    case  8:                                                            \
        gen_helper_msa_fcun_df(cpu_env, tdf, twd, tws, twt);            \
        break;                                                          \
    case  9:                                                            \
        gen_helper_msa_fsun_df(cpu_env, tdf, twd, tws, twt);            \
        break;                                                          \
    case 10:                                                            \
        gen_helper_msa_fcult_df(cpu_env, tdf, twd, tws, twt);           \
        break;                                                          \
    case 11:                                                            \
        gen_helper_msa_fsult_df(cpu_env, tdf, twd, tws, twt);           \
        break;                                                          \
    case 12:                                                            \
        gen_helper_msa_fcueq_df(cpu_env, tdf, twd, tws, twt);           \
        break;                                                          \
    case 13:                                                            \
        gen_helper_msa_fsueq_df(cpu_env, tdf, twd, tws, twt);           \
        break;                                                          \
    case 14:                                                            \
        gen_helper_msa_fcule_df(cpu_env, tdf, twd, tws, twt);           \
        break;                                                          \
    case 15:                                                            \
        gen_helper_msa_fsule_df(cpu_env, tdf, twd, tws, twt);           \
        break;                                                          \
    case 16:                                                            \
        gen_helper_msa_fcne_df(cpu_env, tdf, twd, tws, twt);            \
        break;                                                          \
    case 17:                                                            \
        gen_helper_msa_fsne_df(cpu_env, tdf, twd, tws, twt);            \
        break;                                                          \
    case 20:                                                            \
        gen_helper_msa_fcor_df(cpu_env, tdf, twd, tws, twt);            \
        break;                                                          \
    case 21:                                                            \
        gen_helper_msa_fsor_df(cpu_env, tdf, twd, tws, twt);            \
        break;                                                          \
    case 24:                                                            \
        gen_helper_msa_fcune_df(cpu_env, tdf, twd, tws, twt);           \
        break;                                                          \
    case 25:                                                            \
        gen_helper_msa_fsune_df(cpu_env, tdf, twd, tws, twt);           \
        break;                                                          \
    default:                                                            \
        abort();                                                        \
    }                                                                   \
    tcg_temp_free_i32(twd);                                             \
    tcg_temp_free_i32(tws);                                             \
    tcg_temp_free_i32(twt);                                             \
    tcg_temp_free_i32(tdf);                                             \
}

VOP_CONDNS(s, WORD)
VOP_CONDNS(d, DOUBLE)
#undef VOP_CONDNS


static bool trans_vfcmp_cond_s(DisasContext *ctx, arg_vfcmp_cond_s *a)
{
    check_lsx_enabled(ctx);
    gen_vfcmp_s(ctx, a->fcond, a->vk, a->vj, a->vd);
    return true;
}

static bool trans_vfcmp_cond_d(DisasContext *ctx, arg_vfcmp_cond_d *a)
{
    check_lsx_enabled(ctx);
    gen_vfcmp_d(ctx, a->fcond, a->vk, a->vj, a->vd);
    return true;
}

static bool trans_xvfcmp_cond_s(DisasContext *ctx, arg_xvfcmp_cond_s *a)
{
    check_lasx_enabled(ctx);
    gen_vfcmp_s(ctx, a->fcond, a->xk, a->xj, a->xd);
    return true;
}

static bool trans_xvfcmp_cond_d(DisasContext *ctx, arg_xvfcmp_cond_d *a)
{
    check_lasx_enabled(ctx);
    gen_vfcmp_d(ctx, a->fcond, a->xk, a->xj, a->xd);
    return true;
}

static bool trans_vbitsel_v(DisasContext *ctx, arg_vbitsel_v *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 twa = tcg_const_i32(a->va);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vbitsel_v(cpu_env, twa, twd, tws, twt);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(twa);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvbitsel_v(DisasContext *ctx, arg_xvbitsel_v *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 twa = tcg_const_i32(a->xa);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);
    check_lasx_enabled(ctx);
    gen_helper_lsx_vbitsel_v(cpu_env, twa, twd, tws, twt);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(twa);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vshuf_b(DisasContext *ctx, arg_vshuf_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 twa = tcg_const_i32(a->va);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vshuf_b(cpu_env, twa, twd, tws, twt);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(twa);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvshuf_b(DisasContext *ctx, arg_xvshuf_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 twa = tcg_const_i32(a->xa);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvshuf_b(cpu_env, twa, twd, tws, twt);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(twa);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vextr_v(DisasContext *ctx, arg_vextr_v *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 twa = tcg_const_i32(a->vui5);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vextr_v(cpu_env, twa, twd, tws, twt);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(twa);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvextr_v(DisasContext *ctx, arg_xvextr_v *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 twa = tcg_const_i32(a->vui5);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvextr_v(cpu_env, twa, twd, tws, twt);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(twa);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vfmaddsub_s(DisasContext *ctx, arg_vfmaddsub_s *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 twa = tcg_const_i32(a->va);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vfmaddsub_s(cpu_env, twd, tws, twt, twa);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(twa);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vfmaddsub_d(DisasContext *ctx, arg_vfmaddsub_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 twa = tcg_const_i32(a->va);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vfmaddsub_d(cpu_env, twd, tws, twt, twa);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(twa);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vfmsubadd_s(DisasContext *ctx, arg_vfmsubadd_s *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 twa = tcg_const_i32(a->va);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vfmsubadd_s(cpu_env, twd, tws, twt, twa);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(twa);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vfmsubadd_d(DisasContext *ctx, arg_vfmsubadd_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 twa = tcg_const_i32(a->va);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vfmsubadd_d(cpu_env, twd, tws, twt, twa);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(twa);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvfmaddsub_s(DisasContext *ctx, arg_xvfmaddsub_s *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 twa = tcg_const_i32(a->xa);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvfmaddsub_s(cpu_env, twd, tws, twt, twa);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(twa);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvfmaddsub_d(DisasContext *ctx, arg_xvfmaddsub_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 twa = tcg_const_i32(a->xa);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvfmaddsub_d(cpu_env, twd, tws, twt, twa);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(twa);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvfmsubadd_s(DisasContext *ctx, arg_xvfmsubadd_s *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 twa = tcg_const_i32(a->xa);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvfmsubadd_s(cpu_env, twd, tws, twt, twa);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(twa);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvfmsubadd_d(DisasContext *ctx, arg_xvfmsubadd_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 twa = tcg_const_i32(a->xa);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvfmsubadd_d(cpu_env, twd, tws, twt, twa);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(twa);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

#define TRANS_VSHUF(name, DF) \
static bool trans_ ## name (DisasContext *ctx, arg_ ## name *a) \
{ \
    TCGv_i32 twd = tcg_const_i32(a->vd); \
    TCGv_i32 tws = tcg_const_i32(a->vj); \
    TCGv_i32 twt = tcg_const_i32(a->vk); \
    TCGv_i32 tdf = tcg_const_i32(DF); \
    \
    check_lsx_enabled(ctx);    \
    gen_helper_lsx_vshuf_df(cpu_env, tdf, twd, tws, twt); \
    \
    tcg_temp_free_i32(twd); \
    tcg_temp_free_i32(tws); \
    tcg_temp_free_i32(twt); \
    tcg_temp_free_i32(tdf); \
    return true; \
}
TRANS_VSHUF(vshuf_h, DF_HALF)   // trans_vshuf_h
TRANS_VSHUF(vshuf_w, DF_WORD)   // trans_vshuf_w
TRANS_VSHUF(vshuf_d, DF_DOUBLE) // trans_vshuf_d
#undef TRANS_VSHUF

#define TRANS_VSEQI(name, DF) \
static bool trans_ ## name (DisasContext *ctx, arg_ ## name *a) \
{ \
    TCGv_i32 twd  = tcg_const_i32(a->vd); \
    TCGv_i32 tws  = tcg_const_i32(a->vj); \
    TCGv_i32 timm = tcg_const_i32(a->si5); \
    TCGv_i32 tdf  = tcg_const_i32(DF); \
    \
    check_lsx_enabled(ctx);    \
    gen_helper_lsx_vseqi_df(cpu_env, tdf, twd, tws, timm); \
    \
    tcg_temp_free_i32(twd); \
    tcg_temp_free_i32(tws); \
    tcg_temp_free_i32(timm); \
    tcg_temp_free_i32(tdf); \
    return true; \
}
TRANS_VSEQI(vseqi_b, DF_BYTE)   // trans_vseqi_b
TRANS_VSEQI(vseqi_h, DF_HALF)   // trans_vseqi_h
TRANS_VSEQI(vseqi_w, DF_WORD)   // trans_vseqi_w
TRANS_VSEQI(vseqi_d, DF_DOUBLE) // trans_vseqi_d
#undef TRANS_VSEQI

#define TRANS_VSLEI_S(name, DF) \
static bool trans_ ## name (DisasContext *ctx, arg_ ## name *a) \
{ \
    TCGv_i32 twd  = tcg_const_i32(a->vd); \
    TCGv_i32 tws  = tcg_const_i32(a->vj); \
    TCGv_i32 timm = tcg_const_i32(a->si5); \
    TCGv_i32 tdf  = tcg_const_i32(DF); \
    \
    check_lsx_enabled(ctx);    \
    gen_helper_lsx_vslei_s_df(cpu_env, tdf, twd, tws, timm); \
    \
    tcg_temp_free_i32(twd); \
    tcg_temp_free_i32(tws); \
    tcg_temp_free_i32(timm); \
    tcg_temp_free_i32(tdf); \
    return true; \
}
TRANS_VSLEI_S(vslei_b, DF_BYTE)     // trans_vslei_b
TRANS_VSLEI_S(vslei_h, DF_HALF)     // trans_vslei_h
TRANS_VSLEI_S(vslei_w, DF_WORD)     // trans_vslei_w
TRANS_VSLEI_S(vslei_d, DF_DOUBLE)   // trans_vslei_d
#undef TRANS_VSLEI_S

#define TRANS_VSLEI_U(name, DF) \
static bool trans_ ## name (DisasContext *ctx, arg_ ## name *a) \
{ \
    TCGv_i32 twd  = tcg_const_i32(a->vd); \
    TCGv_i32 tws  = tcg_const_i32(a->vj); \
    TCGv_i32 timm = tcg_const_i32(a->ui5); \
    TCGv_i32 tdf  = tcg_const_i32(DF); \
    \
    check_lsx_enabled(ctx);    \
    gen_helper_lsx_vslei_u_df(cpu_env, tdf, twd, tws, timm); \
    \
    tcg_temp_free_i32(twd); \
    tcg_temp_free_i32(tws); \
    tcg_temp_free_i32(timm); \
    tcg_temp_free_i32(tdf); \
    return true; \
}
TRANS_VSLEI_U(vslei_bu, DF_BYTE)    // trans_vslei_bu
TRANS_VSLEI_U(vslei_hu, DF_HALF)    // trans_vslei_hu
TRANS_VSLEI_U(vslei_wu, DF_WORD)    // trans_vslei_wu
TRANS_VSLEI_U(vslei_du, DF_DOUBLE)  // trans_vslei_du
#undef TRANS_VSLEI_U


static bool trans_vfrstp_b(DisasContext *ctx, arg_vfrstp_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vfrstp_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vfrstp_h(DisasContext *ctx, arg_vfrstp_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vfrstp_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvfrstp_b(DisasContext *ctx, arg_xvfrstp_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvfrstp_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvfrstp_h(DisasContext *ctx, arg_xvfrstp_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvfrstp_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vadd_q(DisasContext *ctx, arg_vadd_q *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vadd_q(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsub_q(DisasContext *ctx, arg_vsub_q *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vsub_q(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsigncov_b(DisasContext *ctx, arg_vsigncov_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vsigncov_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsigncov_h(DisasContext *ctx, arg_vsigncov_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vsigncov_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsigncov_w(DisasContext *ctx, arg_vsigncov_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vsigncov_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vsigncov_d(DisasContext *ctx, arg_vsigncov_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->vk);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vsigncov_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static inline void gen_load_gpr(TCGv t, int reg)
{
    if (reg == 0) {
        tcg_gen_movi_tl(t, 0);
    } else {
        tcg_gen_mov_tl(t, cpu_gpr[reg]);
    }
}

/* Addresses computation */
static inline void gen_op_addr_add(DisasContext *ctx,
                                   TCGv ret, TCGv arg0, TCGv arg1)
{
    tcg_gen_add_tl(ret, arg0, arg1);

//    if (ctx->hflags & LARCH_HFLAG_AWRAP) {
        tcg_gen_ext32s_i64(ret, ret);
//    }
}


static void gen_base_offset_addr(DisasContext *ctx, TCGv addr,
                                 int base, int offset)
{
    if (base == 0) {
        tcg_gen_movi_tl(addr, offset);
    } else if (offset == 0) {
        gen_load_gpr(addr, base);
    } else {
        tcg_gen_movi_tl(addr, offset);
        gen_op_addr_add(ctx, addr, cpu_gpr[base], addr);
    }
}



static bool trans_vld(DisasContext *ctx, arg_vld *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv taddr = tcg_temp_new();

    gen_base_offset_addr(ctx, taddr, a->rj, a->si12);
    check_lsx_enabled(ctx);
    gen_helper_msa_ld_b(cpu_env, twd, taddr);

    tcg_temp_free(taddr);
    tcg_temp_free_i32(twd);
    return true;
}

static bool trans_vst(DisasContext *ctx, arg_vst *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv taddr = tcg_temp_new();

    gen_base_offset_addr(ctx, taddr, a->rj, a->si12);
    check_lsx_enabled(ctx);
    gen_helper_msa_st_b(cpu_env, twd, taddr);

    tcg_temp_free(taddr);
    tcg_temp_free_i32(twd);
    return true;
}

static bool trans_xvld(DisasContext *ctx, arg_xvld *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv taddr = tcg_temp_new();

    gen_base_offset_addr(ctx, taddr, a->rj, a->si12);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvld(cpu_env, twd, taddr);

    tcg_temp_free(taddr);
    tcg_temp_free_i32(twd);
    return true;
}

static bool trans_xvst(DisasContext *ctx, arg_xvst *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv taddr = tcg_temp_new();

    gen_base_offset_addr(ctx, taddr, a->rj, a->si12);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvst(cpu_env, twd, taddr);

    tcg_temp_free(taddr);
    tcg_temp_free_i32(twd);
    return true;
}

static bool trans_vldrepl_d(DisasContext *ctx, arg_vldrepl_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv taddr = tcg_temp_new();

    gen_base_offset_addr(ctx, taddr, a->rj, a->si9 << 3);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vldrepl_d(cpu_env, twd, taddr);

    tcg_temp_free(taddr);
    tcg_temp_free_i32(twd);
    return true;
}

static bool trans_vldrepl_w(DisasContext *ctx, arg_vldrepl_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv taddr = tcg_temp_new();

    gen_base_offset_addr(ctx, taddr, a->rj, a->si10 << 2);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vldrepl_w(cpu_env, twd, taddr);

    tcg_temp_free(taddr);
    tcg_temp_free_i32(twd);
    return true;
}

static bool trans_vldrepl_h(DisasContext *ctx, arg_vldrepl_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv taddr = tcg_temp_new();

    gen_base_offset_addr(ctx, taddr, a->rj, a->si11 << 1);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vldrepl_h(cpu_env, twd, taddr);

    tcg_temp_free(taddr);
    tcg_temp_free_i32(twd);
    return true;
}

static bool trans_vldrepl_b(DisasContext *ctx, arg_vldrepl_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv taddr = tcg_temp_new();

    gen_base_offset_addr(ctx, taddr, a->rj, a->si12);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vldrepl_b(cpu_env, twd, taddr);

    tcg_temp_free(taddr);
    tcg_temp_free_i32(twd);
    return true;
}

static bool trans_vstelm_d(DisasContext *ctx, arg_vstelm_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tidx = tcg_const_i32(a->idx1);
    TCGv taddr = tcg_temp_new();

    gen_base_offset_addr(ctx, taddr, a->rj, a->si8 << 3);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vstelm_d(cpu_env, twd, taddr, tidx);

    tcg_temp_free(taddr);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tidx);
    return true;
}

static bool trans_vstelm_w(DisasContext *ctx, arg_vstelm_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tidx = tcg_const_i32(a->idx2);
    TCGv taddr = tcg_temp_new();

    gen_base_offset_addr(ctx, taddr, a->rj, a->si8 << 2);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vstelm_w(cpu_env, twd, taddr, tidx);

    tcg_temp_free(taddr);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tidx);
    return true;
}

static bool trans_vstelm_h(DisasContext *ctx, arg_vstelm_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tidx = tcg_const_i32(a->idx3);
    TCGv taddr = tcg_temp_new();

    gen_base_offset_addr(ctx, taddr, a->rj, a->si8 << 1);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vstelm_h(cpu_env, twd, taddr, tidx);

    tcg_temp_free(taddr);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tidx);
    return true;
}

static bool trans_vstelm_b(DisasContext *ctx, arg_vstelm_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tidx = tcg_const_i32(a->idx4);
    TCGv taddr = tcg_temp_new();

    gen_base_offset_addr(ctx, taddr, a->rj, a->si8);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vstelm_b(cpu_env, twd, taddr, tidx);

    tcg_temp_free(taddr);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tidx);
    return true;
}

static bool trans_xvldrepl_d(DisasContext *ctx, arg_xvldrepl_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv taddr = tcg_temp_new();

    gen_base_offset_addr(ctx, taddr, a->rj, a->si9 << 3);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvldrepl_d(cpu_env, twd, taddr);

    tcg_temp_free(taddr);
    tcg_temp_free_i32(twd);
    return true;
}

static bool trans_xvldrepl_w(DisasContext *ctx, arg_xvldrepl_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv taddr = tcg_temp_new();

    gen_base_offset_addr(ctx, taddr, a->rj, a->si10 << 2);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvldrepl_w(cpu_env, twd, taddr);

    tcg_temp_free(taddr);
    tcg_temp_free_i32(twd);
    return true;
}

static bool trans_xvldrepl_h(DisasContext *ctx, arg_xvldrepl_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv taddr = tcg_temp_new();

    gen_base_offset_addr(ctx, taddr, a->rj, a->si11 << 1);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvldrepl_h(cpu_env, twd, taddr);

    tcg_temp_free(taddr);
    tcg_temp_free_i32(twd);
    return true;
}

static bool trans_xvldrepl_b(DisasContext *ctx, arg_xvldrepl_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv taddr = tcg_temp_new();

    gen_base_offset_addr(ctx, taddr, a->rj, a->si12);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvldrepl_b(cpu_env, twd, taddr);

    tcg_temp_free(taddr);
    tcg_temp_free_i32(twd);
    return true;
}

static bool trans_xvstelm_d(DisasContext *ctx, arg_xvstelm_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tidx = tcg_const_i32(a->idx2);
    TCGv taddr = tcg_temp_new();

    gen_base_offset_addr(ctx, taddr, a->rj, a->si8 << 3);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvstelm_d(cpu_env, twd, taddr, tidx);

    tcg_temp_free(taddr);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tidx);
    return true;
}

static bool trans_xvstelm_w(DisasContext *ctx, arg_xvstelm_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tidx = tcg_const_i32(a->idx3);
    TCGv taddr = tcg_temp_new();

    gen_base_offset_addr(ctx, taddr, a->rj, a->si8 << 2);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvstelm_w(cpu_env, twd, taddr, tidx);

    tcg_temp_free(taddr);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tidx);
    return true;
}

static bool trans_xvstelm_h(DisasContext *ctx, arg_xvstelm_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tidx = tcg_const_i32(a->idx4);
    TCGv taddr = tcg_temp_new();

    gen_base_offset_addr(ctx, taddr, a->rj, a->si8 << 1);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvstelm_h(cpu_env, twd, taddr, tidx);

    tcg_temp_free(taddr);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tidx);
    return true;
}

static bool trans_xvstelm_b(DisasContext *ctx, arg_xvstelm_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tidx = tcg_const_i32(a->idx);
    TCGv taddr = tcg_temp_new();

    gen_base_offset_addr(ctx, taddr, a->rj, a->si8);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvstelm_b(cpu_env, twd, taddr, tidx);

    tcg_temp_free(taddr);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tidx);
    return true;
}

static bool trans_vldx(DisasContext *ctx, arg_vldx *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv taddr = tcg_temp_new();

    gen_op_addr_add(ctx, taddr, cpu_gpr[a->rj], cpu_gpr[a->rk]);
    check_lsx_enabled(ctx);
    gen_helper_msa_ld_b(cpu_env, twd, taddr);

    tcg_temp_free(taddr);
    tcg_temp_free_i32(twd);
    return true;
}

static bool trans_vstx(DisasContext *ctx, arg_vstx *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv taddr = tcg_temp_new();

    gen_op_addr_add(ctx, taddr, cpu_gpr[a->rj], cpu_gpr[a->rk]);
    check_lsx_enabled(ctx);
    gen_helper_msa_st_b(cpu_env, twd, taddr);

    tcg_temp_free(taddr);
    tcg_temp_free_i32(twd);
    return true;
}

static bool trans_xvldx(DisasContext *ctx, arg_xvldx *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv taddr = tcg_temp_new();

    gen_op_addr_add(ctx, taddr, cpu_gpr[a->rj], cpu_gpr[a->rk]);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvld(cpu_env, twd, taddr);

    tcg_temp_free(taddr);
    tcg_temp_free_i32(twd);
    return true;
}

static bool trans_xvstx(DisasContext *ctx, arg_xvstx *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv taddr = tcg_temp_new();

    gen_op_addr_add(ctx, taddr, cpu_gpr[a->rj], cpu_gpr[a->rk]);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvst(cpu_env, twd, taddr);

    tcg_temp_free(taddr);
    tcg_temp_free_i32(twd);
    return true;
}

#define TRANS_MSA_F3WR(name, op, DF) \
static bool trans_ ## name(DisasContext *ctx, arg_ ## name *a)   \
{                                                                \
    TCGv_i32 twd = tcg_const_i32(a->vd);                         \
    TCGv_i32 tws = tcg_const_i32(a->vj);                         \
    TCGv_i32 twt = tcg_const_i32(a->vk);                         \
    TCGv_i32 tdf  = tcg_const_i32(DF);                           \
    check_lsx_enabled(ctx);    \
    gen_helper_msa_ ## op ## _df(cpu_env, tdf, twd, tws, twt);   \
    tcg_temp_free_i32(twd);                                      \
    tcg_temp_free_i32(tws);                                      \
    tcg_temp_free_i32(twt);                                      \
    tcg_temp_free_i32(tdf);                                      \
    return true;                                                 \
}
TRANS_MSA_F3WR(vfadd_s,    fadd,   DF_WORD)        // trans_vfadd_s
TRANS_MSA_F3WR(vfadd_d,    fadd,   DF_DOUBLE)      // trans_vfadd_d
TRANS_MSA_F3WR(vfsub_s,    fsub,   DF_WORD)        // trans_vfsub_s
TRANS_MSA_F3WR(vfsub_d,    fsub,   DF_DOUBLE)      // trans_vfsub_d
TRANS_MSA_F3WR(vfmul_s,    fmul,   DF_WORD)        // trans_vfmul_s
TRANS_MSA_F3WR(vfmul_d,    fmul,   DF_DOUBLE)      // trans_vfmul_d
TRANS_MSA_F3WR(vfdiv_s,    fdiv,   DF_WORD)        // trans_vfdiv_s
TRANS_MSA_F3WR(vfdiv_d,    fdiv,   DF_DOUBLE)      // trans_vfdiv_d
TRANS_MSA_F3WR(vfmax_s,    fmax,   DF_WORD)        // trans_vfmax_s
TRANS_MSA_F3WR(vfmax_d,    fmax,   DF_DOUBLE)      // trans_vfmax_d
TRANS_MSA_F3WR(vfmin_s,    fmin,   DF_WORD)        // trans_vfmin_s
TRANS_MSA_F3WR(vfmin_d,    fmin,   DF_DOUBLE)      // trans_vfmin_d
TRANS_MSA_F3WR(vfmaxa_s,   fmax_a, DF_WORD)        // trans_vfmaxa_s
TRANS_MSA_F3WR(vfmaxa_d,   fmax_a, DF_DOUBLE)      // trans_vfmaxa_d
TRANS_MSA_F3WR(vfmina_s,   fmin_a, DF_WORD)        // trans_vfmina_s
TRANS_MSA_F3WR(vfmina_d,   fmin_a, DF_DOUBLE)      // trans_vfmina_d
TRANS_MSA_F3WR(vfscaleb_s, fexp2,  DF_WORD)        // trans_vfscaleb_s
TRANS_MSA_F3WR(vfscaleb_d, fexp2,  DF_DOUBLE)      // trans_vfscaleb_d
TRANS_MSA_F3WR(vfcvt_h_s,  fexdo,  DF_WORD)        // trans_vfcvt_h_s
TRANS_MSA_F3WR(vfcvt_s_d,  fexdo,  DF_DOUBLE)      // trans_vfcvt_s_d
#undef TRANS_MSA_F3WR

#define TRANS_MSA_F3XR(name, op, DF) \
static bool trans_ ## name(DisasContext *ctx, arg_ ## name *a)   \
{                                                                \
    TCGv_i32 twd = tcg_const_i32(a->xd);                         \
    TCGv_i32 tws = tcg_const_i32(a->xj);                         \
    TCGv_i32 twt = tcg_const_i32(a->xk);                         \
    TCGv_i32 tdf  = tcg_const_i32(DF);                           \
    check_lasx_enabled(ctx);    \
    gen_helper_msa_ ## op ## _df(cpu_env, tdf, twd, tws, twt);   \
    tcg_temp_free_i32(twd);                                      \
    tcg_temp_free_i32(tws);                                      \
    tcg_temp_free_i32(twt);                                      \
    tcg_temp_free_i32(tdf);                                      \
    return true;                                                 \
}
TRANS_MSA_F3XR(xvfadd_s,    fadd,   DF_WORD)        // trans_xvfadd_s
TRANS_MSA_F3XR(xvfadd_d,    fadd,   DF_DOUBLE)      // trans_xvfadd_d
TRANS_MSA_F3XR(xvfsub_s,    fsub,   DF_WORD)        // trans_xvfsub_s
TRANS_MSA_F3XR(xvfsub_d,    fsub,   DF_DOUBLE)      // trans_xvfsub_d
TRANS_MSA_F3XR(xvfmul_s,    fmul,   DF_WORD)        // trans_xvfmul_s
TRANS_MSA_F3XR(xvfmul_d,    fmul,   DF_DOUBLE)      // trans_xvfmul_d
TRANS_MSA_F3XR(xvfdiv_s,    fdiv,   DF_WORD)        // trans_xvfdiv_s
TRANS_MSA_F3XR(xvfdiv_d,    fdiv,   DF_DOUBLE)      // trans_xvfdiv_d
TRANS_MSA_F3XR(xvfmax_s,    fmax,   DF_WORD)        // trans_xvfmax_s
TRANS_MSA_F3XR(xvfmax_d,    fmax,   DF_DOUBLE)      // trans_xvfmax_d
TRANS_MSA_F3XR(xvfmin_s,    fmin,   DF_WORD)        // trans_xvfmin_s
TRANS_MSA_F3XR(xvfmin_d,    fmin,   DF_DOUBLE)      // trans_xvfmin_d
TRANS_MSA_F3XR(xvfmaxa_s,   fmax_a, DF_WORD)        // trans_xvfmaxa_s
TRANS_MSA_F3XR(xvfmaxa_d,   fmax_a, DF_DOUBLE)      // trans_xvfmaxa_d
TRANS_MSA_F3XR(xvfmina_s,   fmin_a, DF_WORD)        // trans_xvfmina_s
TRANS_MSA_F3XR(xvfmina_d,   fmin_a, DF_DOUBLE)      // trans_xvfmina_d
TRANS_MSA_F3XR(xvfscaleb_s, fexp2,  DF_WORD)        // trans_xvfscaleb_s
TRANS_MSA_F3XR(xvfscaleb_d, fexp2,  DF_DOUBLE)      // trans_xvfscaleb_d
TRANS_MSA_F3XR(xvfcvt_h_s,  fexdo,  DF_WORD)        // trans_xvfcvt_h_s
TRANS_MSA_F3XR(xvfcvt_s_d,  fexdo,  DF_DOUBLE)      // trans_xvfcvt_s_d
#undef TRANS_MSA_F3XR


#define TRANS_LSX_3WR(name)                                          \
static bool trans_ ## name(DisasContext *ctx, arg_ ## name *a)      \
{                                                                   \
    TCGv_i32 twd = tcg_const_i32(a->vd);                            \
    TCGv_i32 tws = tcg_const_i32(a->vj);                            \
    TCGv_i32 twt = tcg_const_i32(a->vk);                            \
    check_lsx_enabled(ctx);    \
    gen_helper_lsx_ ## name(cpu_env, twd, tws, twt);                \
    tcg_temp_free_i32(twd);                                         \
    tcg_temp_free_i32(tws);                                         \
    tcg_temp_free_i32(twt);                                         \
    return true;                                                    \
}
//TRANS_LSX_3WR(vfaddsub_s)    // trans_vfaddsub_s
//TRANS_LSX_3WR(vfaddsub_d)    // trans_vfaddsub_d
//TRANS_LSX_3WR(vfsubadd_s)    // trans_vfsubadd_s
//TRANS_LSX_3WR(vfsubadd_d)    // trans_vfsubadd_d
TRANS_LSX_3WR(vffint_s_l)    // trans_vffint_s_l
TRANS_LSX_3WR(vftint_w_d)    // trans_vftint_w_d
TRANS_LSX_3WR(vftintrm_w_d)  // trans_vftintrm_w_d
TRANS_LSX_3WR(vftintrp_w_d)  // trans_vftintrp_w_d
TRANS_LSX_3WR(vftintrz_w_d)  // trans_vftintrz_w_d
TRANS_LSX_3WR(vftintrne_w_d) // trans_vftintrne_w_d
#undef TRANS_LSX_3WR

#define TRANS_LASX_3XR(name)                                          \
static bool trans_ ## name(DisasContext *ctx, arg_ ## name *a)      \
{                                                                   \
    TCGv_i32 twd = tcg_const_i32(a->xd);                            \
    TCGv_i32 tws = tcg_const_i32(a->xj);                            \
    TCGv_i32 twt = tcg_const_i32(a->xk);                            \
    check_lasx_enabled(ctx);    \
    gen_helper_lsx_ ## name(cpu_env, twd, tws, twt);                \
    tcg_temp_free_i32(twd);                                         \
    tcg_temp_free_i32(tws);                                         \
    tcg_temp_free_i32(twt);                                         \
    return true;                                                    \
}
TRANS_LASX_3XR(xvfaddsub_s)    // trans_xvfaddsub_s
TRANS_LASX_3XR(xvfaddsub_d)    // trans_xvfaddsub_d
TRANS_LASX_3XR(xvfsubadd_s)    // trans_xvfsubadd_s
TRANS_LASX_3XR(xvfsubadd_d)    // trans_xvfsubadd_d
TRANS_LASX_3XR(xvffint_s_l)    // trans_xvffint_s_l
TRANS_LASX_3XR(xvftint_w_d)    // trans_xvftint_w_d
TRANS_LASX_3XR(xvftintrm_w_d)  // trans_xvftintrm_w_d
TRANS_LASX_3XR(xvftintrp_w_d)  // trans_xvftintrp_w_d
TRANS_LASX_3XR(xvftintrz_w_d)  // trans_xvftintrz_w_d
TRANS_LASX_3XR(xvftintrne_w_d) // trans_xvftintrne_w_d
#undef TRANS_LASX_3XR

#define TRANS_F2WR(type, name, op, DF) \
static bool trans_ ## name(DisasContext *ctx, arg_ ## name *a)    \
{                                                                 \
    TCGv_i32 twd = tcg_const_i32(a->vd);                          \
    TCGv_i32 tws = tcg_const_i32(a->vj);                          \
    TCGv_i32 tdf  = tcg_const_i32(DF);                            \
    check_lsx_enabled(ctx);    \
    gen_helper_ ## type ## _ ## op ## _df(cpu_env, tdf, twd, tws);\
    tcg_temp_free_i32(twd);                                       \
    tcg_temp_free_i32(tws);                                       \
    tcg_temp_free_i32(tdf);                                       \
    return true;                                                  \
}
TRANS_F2WR(msa, vflogb_s  , flog2 , DF_WORD)     // trans_vflogb_s
TRANS_F2WR(msa, vflogb_d  , flog2 , DF_DOUBLE)   // trans_vflogb_d
TRANS_F2WR(msa, vfclass_s , fclass, DF_WORD)     // trans_vfclass_s
TRANS_F2WR(msa, vfclass_d , fclass, DF_DOUBLE)   // trans_vfclass_d
TRANS_F2WR(msa, vfsqrt_s  , fsqrt , DF_WORD)     // trans_vfsqrt_s
TRANS_F2WR(msa, vfsqrt_d  , fsqrt , DF_DOUBLE)   // trans_vfsqrt_d
TRANS_F2WR(msa, vfrecip_s , frcp  , DF_WORD)     // trans_vfrecip_s
TRANS_F2WR(msa, vfrecip_d , frcp  , DF_DOUBLE)   // trans_vfrecip_d
TRANS_F2WR(msa, vfrsqrt_s , frsqrt, DF_WORD)     // trans_vfrsqrt_s
TRANS_F2WR(msa, vfrsqrt_d , frsqrt, DF_DOUBLE)   // trans_vfrsqrt_d
TRANS_F2WR(msa, vfrint_s  , frint , DF_WORD)     // trans_vfrint_s
TRANS_F2WR(msa, vfrint_d  , frint , DF_DOUBLE)   // trans_vfrint_d

TRANS_F2WR(lsx, vfrintrm_s , vfrintrm , DF_WORD)    // trans_vfrintrm_s
TRANS_F2WR(lsx, vfrintrm_d , vfrintrm , DF_DOUBLE)  // trans_vfrintrm_d
TRANS_F2WR(lsx, vfrintrp_s , vfrintrp , DF_WORD)    // trans_vfrintrp_s
TRANS_F2WR(lsx, vfrintrp_d , vfrintrp , DF_DOUBLE)  // trans_vfrintrp_d
TRANS_F2WR(lsx, vfrintrz_s , vfrintrz , DF_WORD)    // trans_vfrintrz_s
TRANS_F2WR(lsx, vfrintrz_d , vfrintrz , DF_DOUBLE)  // trans_vfrintrz_d
TRANS_F2WR(lsx, vfrintrne_s, vfrintrne, DF_WORD)    // trans_vfrintrne_s
TRANS_F2WR(lsx, vfrintrne_d, vfrintrne, DF_DOUBLE)  // trans_vfrintrne_d

TRANS_F2WR(msa, vfcvtl_s_h, fexupr, DF_WORD)      // trans_vfcvtl_s_h
TRANS_F2WR(msa, vfcvth_s_h, fexupl, DF_WORD)      // trans_vfcvth_s_h
TRANS_F2WR(msa, vfcvtl_d_s, fexupr, DF_DOUBLE)    // trans_vfcvtl_d_s
TRANS_F2WR(msa, vfcvth_d_s, fexupl, DF_DOUBLE)    // trans_vfcvth_d_s
TRANS_F2WR(msa, vffint_s_w, ffint_s, DF_WORD)     // trans_vffint_s_w
TRANS_F2WR(msa, vffint_s_wu,ffint_u, DF_WORD)     // trans_vffint_s_wu
TRANS_F2WR(msa, vffint_d_l, ffint_s, DF_DOUBLE)   // trans_vffint_d_l
TRANS_F2WR(msa, vffint_d_lu,ffint_u, DF_DOUBLE)   // trans_vffint_d_lu
TRANS_F2WR(msa, vftint_w_s, ftint_s, DF_WORD)     // trans_vftint_w_s
TRANS_F2WR(msa, vftint_l_d, ftint_s, DF_DOUBLE)   // trans_vftint_l_d
TRANS_F2WR(msa, vftint_wu_s, ftint_u, DF_WORD)    // trans_vftint_wu_s
TRANS_F2WR(msa, vftint_lu_d, ftint_u, DF_DOUBLE)  // trans_vftint_lu_d

TRANS_F2WR(lsx, vftintrm_w_s , vftintrm , DF_WORD  )  // trans_vftintrm_w_s
TRANS_F2WR(lsx, vftintrm_l_d , vftintrm , DF_DOUBLE)  // trans_vftintrm_l_d
TRANS_F2WR(lsx, vftintrp_w_s , vftintrp , DF_WORD  )  // trans_vftintrp_w_s
TRANS_F2WR(lsx, vftintrp_l_d , vftintrp , DF_DOUBLE)  // trans_vftintrp_l_d
TRANS_F2WR(lsx, vftintrz_w_s , vftintrz , DF_WORD  )  // trans_vftintrz_w_s
TRANS_F2WR(lsx, vftintrz_l_d , vftintrz , DF_DOUBLE)  // trans_vftintrz_l_d
TRANS_F2WR(lsx, vftintrne_w_s, vftintrne, DF_WORD  )  // trans_vftintrne_w_s
TRANS_F2WR(lsx, vftintrne_l_d, vftintrne, DF_DOUBLE)  // trans_vftintrne_l_d
TRANS_F2WR(msa, vftintrz_wu_s, ftrunc_u, DF_WORD  )   // trans_vftintrz_wu_s
TRANS_F2WR(msa, vftintrz_lu_d, ftrunc_u, DF_DOUBLE)   // trans_vftintrz_lu_d
#undef TRANS_F2WR

#define TRANS_F2XR(type, name, op, DF) \
static bool trans_ ## name(DisasContext *ctx, arg_ ## name *a)    \
{                                                                 \
    TCGv_i32 twd = tcg_const_i32(a->xd);                          \
    TCGv_i32 tws = tcg_const_i32(a->xj);                          \
    TCGv_i32 tdf  = tcg_const_i32(DF);                            \
    check_lasx_enabled(ctx);   \
    gen_helper_ ## type ## _ ## op ## _df(cpu_env, tdf, twd, tws);\
    tcg_temp_free_i32(twd);                                       \
    tcg_temp_free_i32(tws);                                       \
    tcg_temp_free_i32(tdf);                                       \
    return true;                                                  \
}
TRANS_F2XR(msa, xvflogb_s  , flog2 , DF_WORD)     // trans_xvflogb_s
TRANS_F2XR(msa, xvflogb_d  , flog2 , DF_DOUBLE)   // trans_xvflogb_d
TRANS_F2XR(msa, xvfclass_s , fclass, DF_WORD)     // trans_xvfclass_s
TRANS_F2XR(msa, xvfclass_d , fclass, DF_DOUBLE)   // trans_xvfclass_d
TRANS_F2XR(msa, xvfsqrt_s  , fsqrt , DF_WORD)     // trans_xvfsqrt_s
TRANS_F2XR(msa, xvfsqrt_d  , fsqrt , DF_DOUBLE)   // trans_xvfsqrt_d
TRANS_F2XR(msa, xvfrecip_s , frcp  , DF_WORD)     // trans_xvfrecip_s
TRANS_F2XR(msa, xvfrecip_d , frcp  , DF_DOUBLE)   // trans_xvfrecip_d
TRANS_F2XR(msa, xvfrsqrt_s , frsqrt, DF_WORD)     // trans_xvfrsqrt_s
TRANS_F2XR(msa, xvfrsqrt_d , frsqrt, DF_DOUBLE)   // trans_xvfrsqrt_d
TRANS_F2XR(msa, xvfrint_s  , frint , DF_WORD)     // trans_xvfrint_s
TRANS_F2XR(msa, xvfrint_d  , frint , DF_DOUBLE)   // trans_xvfrint_d

TRANS_F2XR(lsx, xvfrintrm_s , vfrintrm , DF_WORD)    // trans_xvfrintrm_s
TRANS_F2XR(lsx, xvfrintrm_d , vfrintrm , DF_DOUBLE)  // trans_xvfrintrm_d
TRANS_F2XR(lsx, xvfrintrp_s , vfrintrp , DF_WORD)    // trans_xvfrintrp_s
TRANS_F2XR(lsx, xvfrintrp_d , vfrintrp , DF_DOUBLE)  // trans_xvfrintrp_d
TRANS_F2XR(lsx, xvfrintrz_s , vfrintrz , DF_WORD)    // trans_xvfrintrz_s
TRANS_F2XR(lsx, xvfrintrz_d , vfrintrz , DF_DOUBLE)  // trans_xvfrintrz_d
TRANS_F2XR(lsx, xvfrintrne_s, vfrintrne, DF_WORD)    // trans_xvfrintrne_s
TRANS_F2XR(lsx, xvfrintrne_d, vfrintrne, DF_DOUBLE)  // trans_xvfrintrne_d

TRANS_F2XR(msa, xvfcvtl_s_h, fexupr, DF_WORD)      // trans_xvfcvtl_s_h
TRANS_F2XR(msa, xvfcvth_s_h, fexupl, DF_WORD)      // trans_xvfcvth_s_h
TRANS_F2XR(msa, xvfcvtl_d_s, fexupr, DF_DOUBLE)    // trans_xvfcvtl_d_s
TRANS_F2XR(msa, xvfcvth_d_s, fexupl, DF_DOUBLE)    // trans_xvfcvth_d_s
TRANS_F2XR(msa, xvffint_s_w, ffint_s, DF_WORD)     // trans_xvffint_s_w
TRANS_F2XR(msa, xvffint_s_wu,ffint_u, DF_WORD)     // trans_xvffint_s_wu
TRANS_F2XR(msa, xvffint_d_l, ffint_s, DF_DOUBLE)   // trans_xvffint_d_l
TRANS_F2XR(msa, xvffint_d_lu,ffint_u, DF_DOUBLE)   // trans_xvffint_d_lu
TRANS_F2XR(msa, xvftint_w_s, ftint_s, DF_WORD)     // trans_xvftint_w_s
TRANS_F2XR(msa, xvftint_l_d, ftint_s, DF_DOUBLE)   // trans_xvftint_l_d
TRANS_F2XR(msa, xvftint_wu_s, ftint_u, DF_WORD)    // trans_xvftint_wu_s
TRANS_F2XR(msa, xvftint_lu_d, ftint_u, DF_DOUBLE)  // trans_xvftint_lu_d

TRANS_F2XR(lsx, xvftintrm_w_s , vftintrm , DF_WORD  )  // trans_xvftintrm_w_s
TRANS_F2XR(lsx, xvftintrm_l_d , vftintrm , DF_DOUBLE)  // trans_xvftintrm_l_d
TRANS_F2XR(lsx, xvftintrp_w_s , vftintrp , DF_WORD  )  // trans_xvftintrp_w_s
TRANS_F2XR(lsx, xvftintrp_l_d , vftintrp , DF_DOUBLE)  // trans_xvftintrp_l_d
TRANS_F2XR(lsx, xvftintrz_w_s , vftintrz , DF_WORD  )  // trans_xvftintrz_w_s
TRANS_F2XR(lsx, xvftintrz_l_d , vftintrz , DF_DOUBLE)  // trans_xvftintrz_l_d
TRANS_F2XR(lsx, xvftintrne_w_s, vftintrne, DF_WORD  )  // trans_xvftintrne_w_s
TRANS_F2XR(lsx, xvftintrne_l_d, vftintrne, DF_DOUBLE)  // trans_xvftintrne_l_d
TRANS_F2XR(msa, xvftintrz_wu_s, ftrunc_u, DF_WORD  )   // trans_xvftintrz_wu_s
TRANS_F2XR(msa, xvftintrz_lu_d, ftrunc_u, DF_DOUBLE)   // trans_xvftintrz_lu_d

TRANS_F2XR(lsx, xvreplve0_b, xvreplve0, DF_BYTE)
TRANS_F2XR(lsx, xvreplve0_h, xvreplve0, DF_HALF)
TRANS_F2XR(lsx, xvreplve0_w, xvreplve0, DF_WORD)
TRANS_F2XR(lsx, xvreplve0_d, xvreplve0, DF_DOUBLE)
TRANS_F2XR(lsx, xvreplve0_q, xvreplve0, DF_QUAD)
#undef TRANS_F2XR

#define TRANS_LSX_2WR(name)                                         \
static bool trans_ ## name(DisasContext *ctx, arg_ ## name *a)      \
{                                                                   \
    TCGv_i32 twd = tcg_const_i32(a->vd);                            \
    TCGv_i32 tws = tcg_const_i32(a->vj);                            \
    check_lsx_enabled(ctx);    \
    gen_helper_lsx_ ## name(cpu_env, twd, tws);                     \
    tcg_temp_free_i32(twd);                                         \
    tcg_temp_free_i32(tws);                                         \
    return true;                                                    \
}
TRANS_LSX_2WR(vffintl_d_w)
TRANS_LSX_2WR(vffinth_d_w)
TRANS_LSX_2WR(vftintl_l_s)
TRANS_LSX_2WR(vftinth_l_s)
TRANS_LSX_2WR(vftintrml_l_s)
TRANS_LSX_2WR(vftintrmh_l_s)
TRANS_LSX_2WR(vftintrpl_l_s)
TRANS_LSX_2WR(vftintrph_l_s)
TRANS_LSX_2WR(vftintrzl_l_s)
TRANS_LSX_2WR(vftintrzh_l_s)
TRANS_LSX_2WR(vftintrnel_l_s)
TRANS_LSX_2WR(vftintrneh_l_s)
#undef TRANS_LSX_2WR

#define TRANS_LASX_2XR(name)                                         \
static bool trans_ ## name(DisasContext *ctx, arg_ ## name *a)      \
{                                                                   \
    TCGv_i32 twd = tcg_const_i32(a->xd);                            \
    TCGv_i32 tws = tcg_const_i32(a->xj);                            \
    check_lasx_enabled(ctx);    \
    gen_helper_lsx_ ## name(cpu_env, twd, tws);                     \
    tcg_temp_free_i32(twd);                                         \
    tcg_temp_free_i32(tws);                                         \
    return true;                                                    \
}
TRANS_LASX_2XR(xvffintl_d_w)
TRANS_LASX_2XR(xvffinth_d_w)
TRANS_LASX_2XR(xvftintl_l_s)
TRANS_LASX_2XR(xvftinth_l_s)
TRANS_LASX_2XR(xvftintrml_l_s)
TRANS_LASX_2XR(xvftintrmh_l_s)
TRANS_LASX_2XR(xvftintrpl_l_s)
TRANS_LASX_2XR(xvftintrph_l_s)
TRANS_LASX_2XR(xvftintrzl_l_s)
TRANS_LASX_2XR(xvftintrzh_l_s)
TRANS_LASX_2XR(xvftintrnel_l_s)
TRANS_LASX_2XR(xvftintrneh_l_s)

TRANS_LASX_2XR(vext2xv_h_b)
TRANS_LASX_2XR(vext2xv_w_b)
TRANS_LASX_2XR(vext2xv_d_b)
TRANS_LASX_2XR(vext2xv_w_h)
TRANS_LASX_2XR(vext2xv_d_h)
TRANS_LASX_2XR(vext2xv_d_w)
TRANS_LASX_2XR(vext2xv_hu_bu)
TRANS_LASX_2XR(vext2xv_wu_bu)
TRANS_LASX_2XR(vext2xv_du_bu)
TRANS_LASX_2XR(vext2xv_wu_hu)
TRANS_LASX_2XR(vext2xv_du_hu)
TRANS_LASX_2XR(vext2xv_du_wu)
#undef TRANS_LSX_2XR

#define TRANS_MSA_F1XD1RJ(name, op, DF)                             \
static bool trans_ ## name(DisasContext *ctx, arg_ ## name *a)      \
{                                                                   \
    TCGv_i32 twd = tcg_const_i32(a->xd);                            \
    TCGv_i32 tws = tcg_const_i32(a->rj);                            \
    TCGv_i32 tdf  = tcg_const_i32(DF);                              \
    check_lasx_enabled(ctx);   \
    gen_helper_msa_ ## op ## _df(cpu_env, tdf, twd, tws);           \
    tcg_temp_free_i32(twd);                                         \
    tcg_temp_free_i32(tws);                                         \
    tcg_temp_free_i32(tdf);                                         \
    return true;                                                    \
}
TRANS_MSA_F1XD1RJ(xvreplgr2vr_b, fill, DF_BYTE)
TRANS_MSA_F1XD1RJ(xvreplgr2vr_h, fill, DF_HALF)
TRANS_MSA_F1XD1RJ(xvreplgr2vr_w, fill, DF_WORD)
TRANS_MSA_F1XD1RJ(xvreplgr2vr_d, fill, DF_DOUBLE)
#undef TRANS_MSA_F1XD1RJ

static bool trans_xvhseli_d(DisasContext *ctx, arg_xvhseli_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 sel = tcg_const_i32(a->ui5);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvhseli_d(cpu_env, twd, tws, sel);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(sel);
    return true;
}

static bool trans_xvinsgr2vr_w(DisasContext *ctx, arg_xvinsgr2vr_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->rj);
    TCGv_i32 sel = tcg_const_i32(a->ui3);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvinsgr2vr_w(cpu_env, twd, tws, sel);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(sel);
    return true;
}

static bool trans_xvinsgr2vr_d(DisasContext *ctx, arg_xvinsgr2vr_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->rj);
    TCGv_i32 sel = tcg_const_i32(a->ui2);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvinsgr2vr_d(cpu_env, twd, tws, sel);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(sel);
    return true;
}

static bool trans_xvpickve2gr_w(DisasContext *ctx, arg_xvpickve2gr_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->rd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 sel = tcg_const_i32(a->ui3);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvpickve2gr_w(cpu_env, twd, tws, sel);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(sel);
    return true;
}

static bool trans_xvpickve2gr_d(DisasContext *ctx, arg_xvpickve2gr_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->rd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 sel = tcg_const_i32(a->ui2);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvpickve2gr_d(cpu_env, twd, tws, sel);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(sel);
    return true;
}

static bool trans_xvpickve2gr_wu(DisasContext *ctx, arg_xvpickve2gr_wu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->rd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 sel = tcg_const_i32(a->ui3);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvpickve2gr_wu(cpu_env, twd, tws, sel);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(sel);
    return true;
}

static bool trans_xvpickve2gr_du(DisasContext *ctx, arg_xvpickve2gr_du *a)
{
    TCGv_i32 twd = tcg_const_i32(a->rd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 sel = tcg_const_i32(a->ui2);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvpickve2gr_d(cpu_env, twd, tws, sel);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(sel);
    return true;
}

static bool trans_xvrepl128vei_b(DisasContext *ctx, arg_xvrepl128vei_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 sel = tcg_const_i32(a->ui4);
    TCGv_i32 tdf  = tcg_const_i32(DF_BYTE);
    check_lasx_enabled(ctx);
    gen_helper_msa_splati_df(cpu_env, tdf, twd, tws, sel);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(sel);
    tcg_temp_free_i32(tdf);
    return true;
}

static bool trans_xvrepl128vei_h(DisasContext *ctx, arg_xvrepl128vei_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 sel = tcg_const_i32(a->ui3);
    TCGv_i32 tdf  = tcg_const_i32(DF_HALF);
    check_lasx_enabled(ctx);
    gen_helper_msa_splati_df(cpu_env, tdf, twd, tws, sel);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(sel);
    tcg_temp_free_i32(tdf);
    return true;
}

static bool trans_xvrepl128vei_w(DisasContext *ctx, arg_xvrepl128vei_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 sel = tcg_const_i32(a->ui2);
    TCGv_i32 tdf  = tcg_const_i32(DF_WORD);
    check_lasx_enabled(ctx);
    gen_helper_msa_splati_df(cpu_env, tdf, twd, tws, sel);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(sel);
    tcg_temp_free_i32(tdf);
    return true;
}

static bool trans_xvrepl128vei_d(DisasContext *ctx, arg_xvrepl128vei_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 sel = tcg_const_i32(a->ui1);
    TCGv_i32 tdf  = tcg_const_i32(DF_DOUBLE);
    check_lasx_enabled(ctx);
    gen_helper_msa_splati_df(cpu_env, tdf, twd, tws, sel);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(sel);
    tcg_temp_free_i32(tdf);
    return true;
}

static bool trans_xvextrcoli_b(DisasContext *ctx, arg_xvextrcoli_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 sel = tcg_const_i32(a->ui4);
    TCGv_i32 tdf = tcg_const_i32(DF_BYTE);
    check_lasx_enabled(ctx);
    gen_helper_msa_sldi_df(cpu_env, tdf, twd, tws, sel);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tdf);
    tcg_temp_free_i32(sel);
    return true;
}

static bool trans_xvextrcoli_h(DisasContext *ctx, arg_xvextrcoli_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 sel = tcg_const_i32(a->ui3);
    TCGv_i32 tdf = tcg_const_i32(DF_HALF);
    check_lasx_enabled(ctx);
    gen_helper_msa_sldi_df(cpu_env, tdf, twd, tws, sel);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tdf);
    tcg_temp_free_i32(sel);
    return true;
}

static bool trans_xvextrcoli_w(DisasContext *ctx, arg_xvextrcoli_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 sel = tcg_const_i32(a->ui2);
    TCGv_i32 tdf = tcg_const_i32(DF_WORD);
    check_lasx_enabled(ctx);
    gen_helper_msa_sldi_df(cpu_env, tdf, twd, tws, sel);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tdf);
    tcg_temp_free_i32(sel);
    return true;
}

static bool trans_xvextrcoli_d(DisasContext *ctx, arg_xvextrcoli_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 sel = tcg_const_i32(a->ui1);
    TCGv_i32 tdf = tcg_const_i32(DF_DOUBLE);
    check_lasx_enabled(ctx);
    gen_helper_msa_sldi_df(cpu_env, tdf, twd, tws, sel);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(tdf);
    tcg_temp_free_i32(sel);
    return true;
}

static bool trans_xvinsve0_w(DisasContext *ctx, arg_xvinsve0_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 sel = tcg_const_i32(a->ui3);
    TCGv_i32 tdf = tcg_const_i32(DF_WORD);
    check_lasx_enabled(ctx);
    gen_helper_msa_insve_df(cpu_env, tdf, twd, tws, sel);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(sel);
    tcg_temp_free_i32(tdf);
    return true;
}

static bool trans_xvinsve0_d(DisasContext *ctx, arg_xvinsve0_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 sel = tcg_const_i32(a->ui2);
    TCGv_i32 tdf = tcg_const_i32(DF_DOUBLE);
    check_lasx_enabled(ctx);
    gen_helper_msa_insve_df(cpu_env, tdf, twd, tws, sel);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(sel);
    tcg_temp_free_i32(tdf);
    return true;
}

static bool trans_xvpickve_w(DisasContext *ctx, arg_xvpickve_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 sel = tcg_const_i32(a->ui3);
    TCGv_i32 tdf = tcg_const_i32(DF_WORD);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvpickve_df(cpu_env, tdf, twd, tws, sel);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(sel);
    tcg_temp_free_i32(tdf);
    return true;
}

static bool trans_xvpickve_d(DisasContext *ctx, arg_xvpickve_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 sel = tcg_const_i32(a->ui2);
    TCGv_i32 tdf = tcg_const_i32(DF_DOUBLE);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvpickve_df(cpu_env, tdf, twd, tws, sel);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(sel);
    tcg_temp_free_i32(tdf);
    return true;
}

static bool trans_xvseli_w(DisasContext *ctx, arg_xvseli_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 sel = tcg_const_i32(a->ui8);
    TCGv_i32 tdf = tcg_const_i32(DF_WORD);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvseli_df(cpu_env, tdf, twd, tws, sel);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(sel);
    tcg_temp_free_i32(tdf);
    return true;
}

static bool trans_xvseli_d(DisasContext *ctx, arg_xvseli_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 sel = tcg_const_i32(a->ui8);
    TCGv_i32 tdf = tcg_const_i32(DF_DOUBLE);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvseli_df(cpu_env, tdf, twd, tws, sel);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(sel);
    tcg_temp_free_i32(tdf);
    return true;
}

static bool trans_xvpermi_d(DisasContext *ctx, arg_xvpermi_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 sel = tcg_const_i32(a->ui8);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvpermi_d(cpu_env, twd, tws, sel);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(sel);
    return true;
}

static bool trans_xvpermi_q(DisasContext *ctx, arg_xvpermi_q *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 sel = tcg_const_i32(a->ui8);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvpermi_q(cpu_env, twd, tws, sel);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(sel);
    return true;
}

#define TRANS_VSLTI_S(name, DF) \
static bool trans_ ## name (DisasContext *ctx, arg_ ## name *a) \
{ \
    TCGv_i32 twd  = tcg_const_i32(a->vd); \
    TCGv_i32 tws  = tcg_const_i32(a->vj); \
    TCGv_i32 timm = tcg_const_i32(a->si5); \
    TCGv_i32 tdf  = tcg_const_i32(DF); \
    \
    check_lsx_enabled(ctx);    \
    gen_helper_lsx_vslti_s_df(cpu_env, tdf, twd, tws, timm); \
    \
    tcg_temp_free_i32(twd); \
    tcg_temp_free_i32(tws); \
    tcg_temp_free_i32(timm); \
    tcg_temp_free_i32(tdf); \
    return true; \
}
TRANS_VSLTI_S(vslti_b, DF_BYTE)    // trans_vslti_b
TRANS_VSLTI_S(vslti_h, DF_HALF)    // trans_vslti_h
TRANS_VSLTI_S(vslti_w, DF_WORD)    // trans_vslti_w
TRANS_VSLTI_S(vslti_d, DF_DOUBLE)  // trans_vslti_d
#undef TRANS_VSLTI_S

#define TRANS_VSLTI_U(name, DF) \
static bool trans_ ## name (DisasContext *ctx, arg_ ## name *a) \
{ \
    TCGv_i32 twd  = tcg_const_i32(a->vd); \
    TCGv_i32 tws  = tcg_const_i32(a->vj); \
    TCGv_i32 timm = tcg_const_i32(a->ui5); \
    TCGv_i32 tdf  = tcg_const_i32(DF); \
    \
    check_lsx_enabled(ctx);    \
    gen_helper_lsx_vslti_u_df(cpu_env, tdf, twd, tws, timm); \
    \
    tcg_temp_free_i32(twd); \
    tcg_temp_free_i32(tws); \
    tcg_temp_free_i32(timm); \
    tcg_temp_free_i32(tdf); \
    return true; \
}
TRANS_VSLTI_U(vslti_bu, DF_BYTE)    // trans_vslti_bu
TRANS_VSLTI_U(vslti_hu, DF_HALF)    // trans_vslti_hu
TRANS_VSLTI_U(vslti_wu, DF_WORD)    // trans_vslti_wu
TRANS_VSLTI_U(vslti_du, DF_DOUBLE)  // trans_vslti_du
#undef TRANS_VSLTI_U

#define TRANS_VADDI_U(name, DF) \
static bool trans_ ## name (DisasContext *ctx, arg_ ## name *a) \
{ \
    TCGv_i32 twd  = tcg_const_i32(a->vd); \
    TCGv_i32 tws  = tcg_const_i32(a->vj); \
    TCGv_i32 timm = tcg_const_i32(a->ui5); \
    TCGv_i32 tdf  = tcg_const_i32(DF); \
    \
    check_lsx_enabled(ctx);    \
    gen_helper_lsx_vaddi_u_df(cpu_env, tdf, twd, tws, timm); \
    \
    tcg_temp_free_i32(twd); \
    tcg_temp_free_i32(tws); \
    tcg_temp_free_i32(timm); \
    tcg_temp_free_i32(tdf); \
    return true; \
}
TRANS_VADDI_U(vaddi_bu, DF_BYTE)    // trans_vaddi_bu
TRANS_VADDI_U(vaddi_hu, DF_HALF)    // trans_vaddi_hu
TRANS_VADDI_U(vaddi_wu, DF_WORD)    // trans_vaddi_wu
TRANS_VADDI_U(vaddi_du, DF_DOUBLE)  // trans_vaddi_du
#undef TRANS_VADDI_U

#define TRANS_VSUBI_U(name, DF) \
static bool trans_ ## name (DisasContext *ctx, arg_ ## name *a) \
{ \
    TCGv_i32 twd  = tcg_const_i32(a->vd); \
    TCGv_i32 tws  = tcg_const_i32(a->vj); \
    TCGv_i32 timm = tcg_const_i32(a->ui5); \
    TCGv_i32 tdf  = tcg_const_i32(DF); \
    \
    check_lsx_enabled(ctx);    \
    gen_helper_lsx_vsubi_u_df(cpu_env, tdf, twd, tws, timm); \
    \
    tcg_temp_free_i32(twd); \
    tcg_temp_free_i32(tws); \
    tcg_temp_free_i32(timm); \
    tcg_temp_free_i32(tdf); \
    return true; \
}
TRANS_VSUBI_U(vsubi_bu, DF_BYTE)    // trans_vsubi_bu
TRANS_VSUBI_U(vsubi_hu, DF_HALF)    // trans_vsubi_hu
TRANS_VSUBI_U(vsubi_wu, DF_WORD)    // trans_vsubi_wu
TRANS_VSUBI_U(vsubi_du, DF_DOUBLE)  // trans_vsubi_du
#undef TRANS_VSUBI_U

static bool trans_vbsll_v(DisasContext *ctx, arg_vbsll_v *a)
{
    TCGv_i32 twd  = tcg_const_i32(a->vd);
    TCGv_i32 tws  = tcg_const_i32(a->vj);
    TCGv_i32 timm = tcg_const_i32(a->ui5 & 0xf);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vbsll_v(cpu_env, twd, tws, timm);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(timm);
    return true;
}

static bool trans_vbsrl_v(DisasContext *ctx, arg_vbsrl_v *a)
{
    TCGv_i32 twd  = tcg_const_i32(a->vd);
    TCGv_i32 tws  = tcg_const_i32(a->vj);
    TCGv_i32 timm = tcg_const_i32(a->ui5 & 0xf);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vbsrl_v(cpu_env, twd, tws, timm);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(timm);
    return true;
}

#define TRANS_VMAXI_S(name, DF) \
static bool trans_ ## name (DisasContext *ctx, arg_ ## name *a) \
{ \
    TCGv_i32 twd  = tcg_const_i32(a->vd); \
    TCGv_i32 tws  = tcg_const_i32(a->vj); \
    TCGv_i32 timm = tcg_const_i32(a->si5); \
    TCGv_i32 tdf  = tcg_const_i32(DF); \
    \
    check_lsx_enabled(ctx);    \
    gen_helper_lsx_vmaxi_s_df(cpu_env, tdf, twd, tws, timm); \
    \
    tcg_temp_free_i32(twd); \
    tcg_temp_free_i32(tws); \
    tcg_temp_free_i32(timm); \
    tcg_temp_free_i32(tdf); \
    return true; \
}
TRANS_VMAXI_S(vmaxi_b, DF_BYTE)    // trans_vmaxi_b
TRANS_VMAXI_S(vmaxi_h, DF_HALF)    // trans_vmaxi_h
TRANS_VMAXI_S(vmaxi_w, DF_WORD)    // trans_vmaxi_w
TRANS_VMAXI_S(vmaxi_d, DF_DOUBLE)  // trans_vmaxi_d
#undef TRANS_VMAXI_S

#define TRANS_VMINI_S(name, DF) \
static bool trans_ ## name (DisasContext *ctx, arg_ ## name *a) \
{ \
    TCGv_i32 twd  = tcg_const_i32(a->vd); \
    TCGv_i32 tws  = tcg_const_i32(a->vj); \
    TCGv_i32 timm = tcg_const_i32(a->si5); \
    TCGv_i32 tdf  = tcg_const_i32(DF); \
    \
    check_lsx_enabled(ctx);    \
    gen_helper_lsx_vmini_s_df(cpu_env, tdf, twd, tws, timm); \
    \
    tcg_temp_free_i32(twd); \
    tcg_temp_free_i32(tws); \
    tcg_temp_free_i32(timm); \
    tcg_temp_free_i32(tdf); \
    return true; \
}
TRANS_VMINI_S(vmini_b, DF_BYTE)    // trans_vmini_b
TRANS_VMINI_S(vmini_h, DF_HALF)    // trans_vmini_h
TRANS_VMINI_S(vmini_w, DF_WORD)    // trans_vmini_w
TRANS_VMINI_S(vmini_d, DF_DOUBLE)  // trans_vmini_d
#undef TRANS_VMINI_S

#define TRANS_VMAXI_U(name, DF) \
static bool trans_ ## name (DisasContext *ctx, arg_ ## name *a) \
{ \
    TCGv_i32 twd  = tcg_const_i32(a->vd); \
    TCGv_i32 tws  = tcg_const_i32(a->vj); \
    TCGv_i32 timm = tcg_const_i32(a->ui5); \
    TCGv_i32 tdf  = tcg_const_i32(DF); \
    \
    check_lsx_enabled(ctx);    \
    gen_helper_lsx_vmaxi_u_df(cpu_env, tdf, twd, tws, timm); \
    \
    tcg_temp_free_i32(twd); \
    tcg_temp_free_i32(tws); \
    tcg_temp_free_i32(timm); \
    tcg_temp_free_i32(tdf); \
    return true; \
}
TRANS_VMAXI_U(vmaxi_bu, DF_BYTE)    // trans_vmaxi_bu
TRANS_VMAXI_U(vmaxi_hu, DF_HALF)    // trans_vmaxi_hu
TRANS_VMAXI_U(vmaxi_wu, DF_WORD)    // trans_vmaxi_wu
TRANS_VMAXI_U(vmaxi_du, DF_DOUBLE)  // trans_vmaxi_du
#undef TRANS_VMAXI_U

#define TRANS_VMINI_U(name, DF) \
static bool trans_ ## name (DisasContext *ctx, arg_ ## name *a) \
{ \
    TCGv_i32 twd  = tcg_const_i32(a->vd); \
    TCGv_i32 tws  = tcg_const_i32(a->vj); \
    TCGv_i32 timm = tcg_const_i32(a->ui5); \
    TCGv_i32 tdf  = tcg_const_i32(DF); \
    \
    check_lsx_enabled(ctx);    \
    gen_helper_lsx_vmini_u_df(cpu_env, tdf, twd, tws, timm); \
    \
    tcg_temp_free_i32(twd); \
    tcg_temp_free_i32(tws); \
    tcg_temp_free_i32(timm); \
    tcg_temp_free_i32(tdf); \
    return true; \
}
TRANS_VMINI_U(vmini_bu, DF_BYTE)    // trans_vmini_bu
TRANS_VMINI_U(vmini_hu, DF_HALF)    // trans_vmini_hu
TRANS_VMINI_U(vmini_wu, DF_WORD)    // trans_vmini_wu
TRANS_VMINI_U(vmini_du, DF_DOUBLE)  // trans_vmini_du
#undef TRANS_VMINI_U

#define TRANS_VCLO(name) \
static bool trans_ ## name (DisasContext *ctx, arg_ ## name *a) \
{ \
    TCGv_i32 twd  = tcg_const_i32(a->vd); \
    TCGv_i32 tws  = tcg_const_i32(a->vj); \
    \
    check_lsx_enabled(ctx);    \
    gen_helper_lsx_ ## name (cpu_env, twd, tws); \
    \
    tcg_temp_free_i32(twd); \
    tcg_temp_free_i32(tws); \
    return true; \
}
TRANS_VCLO(vclo_b)  // trans_vclo_b
TRANS_VCLO(vclo_h)  // trans_vclo_h
TRANS_VCLO(vclo_w)  // trans_vclo_w
TRANS_VCLO(vclo_d)  // trans_vclo_d
#undef TRANS_VCLO

#define TRANS_VCLZ(name) \
static bool trans_ ## name (DisasContext *ctx, arg_ ## name *a) \
{ \
    TCGv_i32 twd  = tcg_const_i32(a->vd); \
    TCGv_i32 tws  = tcg_const_i32(a->vj); \
    \
    check_lsx_enabled(ctx);    \
    gen_helper_lsx_ ## name (cpu_env, twd, tws); \
    \
    tcg_temp_free_i32(twd); \
    tcg_temp_free_i32(tws); \
    return true; \
}
TRANS_VCLZ(vclz_b)  // trans_vclz_b
TRANS_VCLZ(vclz_h)  // trans_vclz_h
TRANS_VCLZ(vclz_w)  // trans_vclz_w
TRANS_VCLZ(vclz_d)  // trans_vclz_d
#undef TRANS_VCLZ

#define TRANS_VPCNT(name) \
static bool trans_ ## name (DisasContext *ctx, arg_ ## name *a) \
{ \
    TCGv_i32 twd  = tcg_const_i32(a->vd); \
    TCGv_i32 tws  = tcg_const_i32(a->vj); \
    \
    check_lsx_enabled(ctx);    \
    gen_helper_lsx_ ## name (cpu_env, twd, tws); \
    \
    tcg_temp_free_i32(twd); \
    tcg_temp_free_i32(tws); \
    return true; \
}
TRANS_VPCNT(vpcnt_b)  // trans_vpcnt_b
TRANS_VPCNT(vpcnt_h)  // trans_vpcnt_h
TRANS_VPCNT(vpcnt_w)  // trans_vpcnt_w
TRANS_VPCNT(vpcnt_d)  // trans_vpcnt_d
#undef TRANS_VPCNT

#define TRANS_VNEG(name) \
static bool trans_ ## name (DisasContext *ctx, arg_ ## name *a) \
{ \
    TCGv_i32 twd  = tcg_const_i32(a->vd); \
    TCGv_i32 tws  = tcg_const_i32(a->vj); \
    \
    check_lsx_enabled(ctx);    \
    gen_helper_lsx_ ## name (cpu_env, twd, tws); \
    \
    tcg_temp_free_i32(twd); \
    tcg_temp_free_i32(tws); \
    return true; \
}
TRANS_VNEG(vneg_b)  // trans_vneg_b
TRANS_VNEG(vneg_h)  // trans_vneg_h
TRANS_VNEG(vneg_w)  // trans_vneg_w
TRANS_VNEG(vneg_d)  // trans_vneg_d
#undef TRANS_VNEG

#define TRANS_VMSKLTZ(name) \
static bool trans_ ## name (DisasContext *ctx, arg_ ## name *a) \
{ \
    TCGv_i32 twd  = tcg_const_i32(a->vd); \
    TCGv_i32 tws  = tcg_const_i32(a->vj); \
    \
    check_lsx_enabled(ctx);    \
    gen_helper_lsx_ ## name (cpu_env, twd, tws); \
    \
    tcg_temp_free_i32(twd); \
    tcg_temp_free_i32(tws); \
    return true; \
}
TRANS_VMSKLTZ(vmskltz_b)  // trans_vmskltz_b
TRANS_VMSKLTZ(vmskltz_h)  // trans_vmskltz_h
TRANS_VMSKLTZ(vmskltz_w)  // trans_vmskltz_w
TRANS_VMSKLTZ(vmskltz_d)  // trans_vmskltz_d
#undef TRANS_VMSKLTZ

static bool trans_vmskgez_b (DisasContext *ctx, arg_vmskgez_b *a) \
{
    TCGv_i32 twd  = tcg_const_i32(a->vd);
    TCGv_i32 tws  = tcg_const_i32(a->vj);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vmskgez_b (cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_vmsknz_b (DisasContext *ctx, arg_vmsknz_b *a) \
{
    TCGv_i32 twd  = tcg_const_i32(a->vd);
    TCGv_i32 tws  = tcg_const_i32(a->vj);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vmsknz_b (cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_vmskcopy_b (DisasContext *ctx, arg_vmskcopy_b *a) \
{
    TCGv_i32 twd  = tcg_const_i32(a->vd);
    TCGv_i32 tws  = tcg_const_i32(a->vj);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vmskcopy_b (cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_vmskfill_b (DisasContext *ctx, arg_vmskfill_b *a) \
{
    TCGv_i32 twd  = tcg_const_i32(a->vd);
    TCGv_i32 tws  = tcg_const_i32(a->vj);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vmskfill_b (cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

#define TRANS_VREPLGR2VR(name, DF) \
static bool trans_ ## name (DisasContext *ctx, arg_ ## name *a) \
{ \
    TCGv_i32 twd  = tcg_const_i32(a->vd); \
    TCGv_i32 tws  = tcg_const_i32(a->rj); \
    TCGv_i32 tdf  = tcg_const_i32(DF); \
    \
    check_lsx_enabled(ctx);    \
    gen_helper_lsx_vreplgr2vr_df(cpu_env, tdf, twd, tws); \
    \
    tcg_temp_free_i32(twd); \
    tcg_temp_free_i32(tws); \
    tcg_temp_free_i32(tdf); \
    return true; \
}
TRANS_VREPLGR2VR(vreplgr2vr_b, DF_BYTE)  // trans_vreplgr2vr_b
TRANS_VREPLGR2VR(vreplgr2vr_h, DF_HALF)  // trans_vreplgr2vr_h
TRANS_VREPLGR2VR(vreplgr2vr_w, DF_WORD)  // trans_vreplgr2vr_w
TRANS_VREPLGR2VR(vreplgr2vr_d, DF_DOUBLE)  // trans_vreplgr2vr_d
#undef TRANS_VREPLGR2VR

static bool trans_vinsgr2vr_b (DisasContext *ctx, arg_vinsgr2vr_b *a)
{
    TCGv_i32 twd  = tcg_const_i32(a->vd);
    TCGv_i32 tws  = tcg_const_i32(a->rj);
    TCGv_i32 timm = tcg_const_i32(a->ui4);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vinsgr2vr_b (cpu_env, twd, tws, timm);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(timm);
    return true;
}

static bool trans_vinsgr2vr_h (DisasContext *ctx, arg_vinsgr2vr_h *a)
{
    TCGv_i32 twd  = tcg_const_i32(a->vd);
    TCGv_i32 tws  = tcg_const_i32(a->rj);
    TCGv_i32 timm = tcg_const_i32(a->ui3);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vinsgr2vr_h (cpu_env, twd, tws, timm);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(timm);
    return true;
}

static bool trans_vinsgr2vr_w (DisasContext *ctx, arg_vinsgr2vr_w *a)
{
    TCGv_i32 twd  = tcg_const_i32(a->vd);
    TCGv_i32 tws  = tcg_const_i32(a->rj);
    TCGv_i32 timm = tcg_const_i32(a->ui2);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vinsgr2vr_w (cpu_env, twd, tws, timm);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(timm);
    return true;
}

static bool trans_vinsgr2vr_d (DisasContext *ctx, arg_vinsgr2vr_d *a)
{
    TCGv_i32 twd  = tcg_const_i32(a->vd);
    TCGv_i32 tws  = tcg_const_i32(a->rj);
    TCGv_i32 timm = tcg_const_i32(a->ui1);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vinsgr2vr_d (cpu_env, twd, tws, timm);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(timm);
    return true;
}

static bool trans_vpickve2gr_b (DisasContext *ctx, arg_vpickve2gr_b *a)
{
    TCGv_i32 twd  = tcg_const_i32(a->rd);
    TCGv_i32 tws  = tcg_const_i32(a->vj);
    TCGv_i32 timm = tcg_const_i32(a->ui4);

    check_lsx_enabled(ctx);
    gen_helper_msa_copy_s_b (cpu_env, twd, tws, timm);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(timm);
    return true;
}

static bool trans_vpickve2gr_h (DisasContext *ctx, arg_vpickve2gr_h *a)
{
    TCGv_i32 twd  = tcg_const_i32(a->rd);
    TCGv_i32 tws  = tcg_const_i32(a->vj);
    TCGv_i32 timm = tcg_const_i32(a->ui3);

    check_lsx_enabled(ctx);
    gen_helper_msa_copy_s_h (cpu_env, twd, tws, timm);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(timm);
    return true;
}

static bool trans_vpickve2gr_w (DisasContext *ctx, arg_vpickve2gr_w *a)
{
    TCGv_i32 twd  = tcg_const_i32(a->rd);
    TCGv_i32 tws  = tcg_const_i32(a->vj);
    TCGv_i32 timm = tcg_const_i32(a->ui2);

    check_lsx_enabled(ctx);
    gen_helper_msa_copy_s_w (cpu_env, twd, tws, timm);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(timm);
    return true;
}

static bool trans_vpickve2gr_d (DisasContext *ctx, arg_vpickve2gr_d *a)
{
    TCGv_i32 twd  = tcg_const_i32(a->rd);
    TCGv_i32 tws  = tcg_const_i32(a->vj);
    TCGv_i32 timm = tcg_const_i32(a->ui1);

    check_lsx_enabled(ctx);
    gen_helper_msa_copy_s_d (cpu_env, twd, tws, timm);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(timm);
    return true;
}

static bool trans_vpickve2gr_bu (DisasContext *ctx, arg_vpickve2gr_bu *a) \
{
    TCGv_i32 twd  = tcg_const_i32(a->rd);
    TCGv_i32 tws  = tcg_const_i32(a->vj);
    TCGv_i32 timm = tcg_const_i32(a->ui4);

    check_lsx_enabled(ctx);
    gen_helper_msa_copy_u_b (cpu_env, twd, tws, timm);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(timm);
    return true;
}

static bool trans_vpickve2gr_hu (DisasContext *ctx, arg_vpickve2gr_hu *a) \
{
    TCGv_i32 twd  = tcg_const_i32(a->rd);
    TCGv_i32 tws  = tcg_const_i32(a->vj);
    TCGv_i32 timm = tcg_const_i32(a->ui3);

    check_lsx_enabled(ctx);
    gen_helper_msa_copy_u_h (cpu_env, twd, tws, timm);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(timm);
    return true;
}

static bool trans_vpickve2gr_wu (DisasContext *ctx, arg_vpickve2gr_wu *a) \
{
    TCGv_i32 twd  = tcg_const_i32(a->rd);
    TCGv_i32 tws  = tcg_const_i32(a->vj);
    TCGv_i32 timm = tcg_const_i32(a->ui2);

    check_lsx_enabled(ctx);
    gen_helper_msa_copy_u_w (cpu_env, twd, tws, timm);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(timm);
    return true;
}

static bool trans_vpickve2gr_du (DisasContext *ctx, arg_vpickve2gr_du *a) \
{
    TCGv_i32 twd  = tcg_const_i32(a->rd);
    TCGv_i32 tws  = tcg_const_i32(a->vj);
    TCGv_i32 timm = tcg_const_i32(a->ui1);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vpickve2gr_du (cpu_env, twd, tws, timm);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(timm);
    return true;
}

static bool trans_vreplvei_b (DisasContext *ctx, arg_vreplvei_b *a) \
{
    TCGv_i32 twd  = tcg_const_i32(a->vd);
    TCGv_i32 tws  = tcg_const_i32(a->vj);
    TCGv_i32 timm = tcg_const_i32(a->ui4);
    TCGv_i32 tdf  = tcg_const_i32(DF_BYTE);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vreplvei_df(cpu_env, tdf, twd, tws, timm);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(timm);
    tcg_temp_free_i32(tdf);
    return true;
}

static bool trans_vreplvei_h (DisasContext *ctx, arg_vreplvei_h *a) \
{
    TCGv_i32 twd  = tcg_const_i32(a->vd);
    TCGv_i32 tws  = tcg_const_i32(a->vj);
    TCGv_i32 timm = tcg_const_i32(a->ui3);
    TCGv_i32 tdf  = tcg_const_i32(DF_HALF);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vreplvei_df(cpu_env, tdf, twd, tws, timm);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(timm);
    tcg_temp_free_i32(tdf);
    return true;
}

static bool trans_vreplvei_w (DisasContext *ctx, arg_vreplvei_w *a) \
{
    TCGv_i32 twd  = tcg_const_i32(a->vd);
    TCGv_i32 tws  = tcg_const_i32(a->vj);
    TCGv_i32 timm = tcg_const_i32(a->ui2);
    TCGv_i32 tdf  = tcg_const_i32(DF_WORD);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vreplvei_df(cpu_env, tdf, twd, tws, timm);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(timm);
    tcg_temp_free_i32(tdf);
    return true;
}

static bool trans_vreplvei_d (DisasContext *ctx, arg_vreplvei_d *a) \
{
    TCGv_i32 twd  = tcg_const_i32(a->vd);
    TCGv_i32 tws  = tcg_const_i32(a->vj);
    TCGv_i32 timm = tcg_const_i32(a->ui1);
    TCGv_i32 tdf  = tcg_const_i32(DF_DOUBLE);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vreplvei_df(cpu_env, tdf, twd, tws, timm);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(timm);
    tcg_temp_free_i32(tdf);
    return true;
}

static bool trans_vextrcoli_b (DisasContext *ctx, arg_vextrcoli_b *a)
{
    TCGv_i32 twd  = tcg_const_i32(a->vd);
    TCGv_i32 tws  = tcg_const_i32(a->vj);
    TCGv_i32 timm = tcg_const_i32(a->ui4);
    TCGv_i32 tdf  = tcg_const_i32(DF_BYTE);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vextrcoli_df(cpu_env, tdf, twd, tws, timm);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(timm);
    tcg_temp_free_i32(tdf);
    return true;
}

static bool trans_vextrcoli_h (DisasContext *ctx, arg_vextrcoli_h *a)
{
    TCGv_i32 twd  = tcg_const_i32(a->vd);
    TCGv_i32 tws  = tcg_const_i32(a->vj);
    TCGv_i32 timm = tcg_const_i32(a->ui3);
    TCGv_i32 tdf  = tcg_const_i32(DF_HALF);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vextrcoli_df(cpu_env, tdf, twd, tws, timm);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(timm);
    tcg_temp_free_i32(tdf);
    return true;
}

static bool trans_vextrcoli_w (DisasContext *ctx, arg_vextrcoli_w *a)
{
    TCGv_i32 twd  = tcg_const_i32(a->vd);
    TCGv_i32 tws  = tcg_const_i32(a->vj);
    TCGv_i32 timm = tcg_const_i32(a->ui2);
    TCGv_i32 tdf  = tcg_const_i32(DF_WORD);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vextrcoli_df(cpu_env, tdf, twd, tws, timm);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(timm);
    tcg_temp_free_i32(tdf);
    return true;
}

static bool trans_vextrcoli_d (DisasContext *ctx, arg_vextrcoli_d *a)
{
    TCGv_i32 twd  = tcg_const_i32(a->vd);
    TCGv_i32 tws  = tcg_const_i32(a->vj);
    TCGv_i32 timm = tcg_const_i32(a->ui1);
    TCGv_i32 tdf  = tcg_const_i32(DF_DOUBLE);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vextrcoli_df(cpu_env, tdf, twd, tws, timm);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(timm);
    tcg_temp_free_i32(tdf);
    return true;
}

#define TRANS_VSET_V(name) \
static bool trans_ ## name (DisasContext *ctx, arg_ ## name *a) \
{ \
    TCGv_i32 tcd  = tcg_const_i32(a->cd); \
    TCGv_i32 tws  = tcg_const_i32(a->vj); \
    \
    check_lsx_enabled(ctx);    \
    gen_helper_lsx_ ## name (cpu_env, tcd, tws); \
    \
    tcg_temp_free_i32(tcd); \
    tcg_temp_free_i32(tws); \
    return true; \
}
TRANS_VSET_V(vseteqz_v)  // trans_vseteqz_v
TRANS_VSET_V(vsetnez_v)  // trans_vsetnez_v
#undef TRANS_VSET_V

#define TRANS_VSET_DF(name) \
static bool trans_ ## name (DisasContext *ctx, arg_ ## name *a) \
{ \
    TCGv_i32 tcd  = tcg_const_i32(a->cd); \
    TCGv_i32 tws  = tcg_const_i32(a->vj); \
    \
    check_lsx_enabled(ctx);    \
    gen_helper_lsx_ ## name (cpu_env, tcd, tws); \
    \
    tcg_temp_free_i32(tcd); \
    tcg_temp_free_i32(tws); \
    return true; \
}
TRANS_VSET_DF(vsetanyeqz_b)  // trans_vsetanyeqz_b
TRANS_VSET_DF(vsetanyeqz_h)  // trans_vsetanyeqz_h
TRANS_VSET_DF(vsetanyeqz_w)  // trans_vsetanyeqz_w
TRANS_VSET_DF(vsetanyeqz_d)  // trans_vsetanyeqz_d
TRANS_VSET_DF(vsetallnez_b)  // trans_vsetallnez_b
TRANS_VSET_DF(vsetallnez_h)  // trans_vsetallnez_h
TRANS_VSET_DF(vsetallnez_w)  // trans_vsetallnez_w
TRANS_VSET_DF(vsetallnez_d)  // trans_vsetallnez_d
#undef TRANS_VSET_DF

#define TRANS_VBIT_DF(name, op, DF, u_width) \
static bool trans_ ## name (DisasContext *ctx, arg_ ## name *a) \
{ \
    TCGv_i32 twd  = tcg_const_i32(a->vd); \
    TCGv_i32 tws  = tcg_const_i32(a->vj); \
    TCGv_i32 timm = tcg_const_i32(a->ui ## u_width ); \
    TCGv_i32 tdf  = tcg_const_i32(DF); \
    \
    check_lsx_enabled(ctx);    \
    gen_helper_lsx_ ## op ## _df(cpu_env, tdf, twd, tws, timm); \
    \
    tcg_temp_free_i32(twd); \
    tcg_temp_free_i32(tws); \
    tcg_temp_free_i32(timm); \
    tcg_temp_free_i32(tdf); \
    return true; \
}
TRANS_VBIT_DF(vbitclri_b, vbitclri, DF_BYTE, 3)   // trans_vbitclri_b
TRANS_VBIT_DF(vbitclri_h, vbitclri, DF_HALF, 4)   // trans_vbitclri_h
TRANS_VBIT_DF(vbitclri_w, vbitclri, DF_WORD, 5)   // trans_vbitclri_w
TRANS_VBIT_DF(vbitclri_d, vbitclri, DF_DOUBLE, 6) // trans_vbitclri_d
TRANS_VBIT_DF(vbitseti_b, vbitseti, DF_BYTE, 3)   // trans_vbitseti_b
TRANS_VBIT_DF(vbitseti_h, vbitseti, DF_HALF, 4)   // trans_vbitseti_h
TRANS_VBIT_DF(vbitseti_w, vbitseti, DF_WORD, 5)   // trans_vbitseti_w
TRANS_VBIT_DF(vbitseti_d, vbitseti, DF_DOUBLE, 6) // trans_vbitseti_d
TRANS_VBIT_DF(vbitrevi_b, vbitrevi, DF_BYTE, 3)   // trans_vbitrevi_b
TRANS_VBIT_DF(vbitrevi_h, vbitrevi, DF_HALF, 4)   // trans_vbitrevi_h
TRANS_VBIT_DF(vbitrevi_w, vbitrevi, DF_WORD, 5)   // trans_vbitrevi_w
TRANS_VBIT_DF(vbitrevi_d, vbitrevi, DF_DOUBLE, 6) // trans_vbitrevi_d
TRANS_VBIT_DF(vbstrc12i_b, vbstrc12i, DF_BYTE, 3)   // trans_vbstrc12i_b
TRANS_VBIT_DF(vbstrc12i_h, vbstrc12i, DF_HALF, 4)   // trans_vbstrc12i_h
TRANS_VBIT_DF(vbstrc12i_w, vbstrc12i, DF_WORD, 5)   // trans_vbstrc12i_w
TRANS_VBIT_DF(vbstrc12i_d, vbstrc12i, DF_DOUBLE, 6) // trans_vbstrc12i_d
TRANS_VBIT_DF(vbstrc21i_b, vbstrc21i, DF_BYTE, 3)   // trans_vbstrc21i_b
TRANS_VBIT_DF(vbstrc21i_h, vbstrc21i, DF_HALF, 4)   // trans_vbstrc21i_h
TRANS_VBIT_DF(vbstrc21i_w, vbstrc21i, DF_WORD, 5)   // trans_vbstrc21i_w
TRANS_VBIT_DF(vbstrc21i_d, vbstrc21i, DF_DOUBLE, 6) // trans_vbstrc21i_d
TRANS_VBIT_DF(vsat_b, vsat_s, DF_BYTE, 3)   // trans_vsat_b
TRANS_VBIT_DF(vsat_h, vsat_s, DF_HALF, 4)   // trans_vsat_h
TRANS_VBIT_DF(vsat_w, vsat_s, DF_WORD, 5)   // trans_vsat_w
TRANS_VBIT_DF(vsat_d, vsat_s, DF_DOUBLE, 6) // trans_vsat_d
TRANS_VBIT_DF(vsat_bu, vsat_u, DF_BYTE, 3)   // trans_vsat_bu
TRANS_VBIT_DF(vsat_hu, vsat_u, DF_HALF, 4)   // trans_vsat_hu
TRANS_VBIT_DF(vsat_wu, vsat_u, DF_WORD, 5)   // trans_vsat_wu
TRANS_VBIT_DF(vsat_du, vsat_u, DF_DOUBLE, 6) // trans_vsat_du
TRANS_VBIT_DF(vslli_b, vslli, DF_BYTE, 3)   // trans_vslli_b
TRANS_VBIT_DF(vslli_h, vslli, DF_HALF, 4)   // trans_vslli_h
TRANS_VBIT_DF(vslli_w, vslli, DF_WORD, 5)   // trans_vslli_w
TRANS_VBIT_DF(vslli_d, vslli, DF_DOUBLE, 6) // trans_vslli_d
TRANS_VBIT_DF(vsrli_b, vsrli, DF_BYTE, 3)   // trans_vsrli_b
TRANS_VBIT_DF(vsrli_h, vsrli, DF_HALF, 4)   // trans_vsrli_h
TRANS_VBIT_DF(vsrli_w, vsrli, DF_WORD, 5)   // trans_vsrli_w
TRANS_VBIT_DF(vsrli_d, vsrli, DF_DOUBLE, 6) // trans_vsrli_d
TRANS_VBIT_DF(vsrai_b, vsrai, DF_BYTE, 3)   // trans_vsrai_b
TRANS_VBIT_DF(vsrai_h, vsrai, DF_HALF, 4)   // trans_vsrai_h
TRANS_VBIT_DF(vsrai_w, vsrai, DF_WORD, 5)   // trans_vsrai_w
TRANS_VBIT_DF(vsrai_d, vsrai, DF_DOUBLE, 6) // trans_vsrai_d
TRANS_VBIT_DF(vrotri_b, vrotri, DF_BYTE, 3)   // trans_vrotri_b
TRANS_VBIT_DF(vrotri_h, vrotri, DF_HALF, 4)   // trans_vrotri_h
TRANS_VBIT_DF(vrotri_w, vrotri, DF_WORD, 5)   // trans_vrotri_w
TRANS_VBIT_DF(vrotri_d, vrotri, DF_DOUBLE, 6) // trans_vrotri_d
TRANS_VBIT_DF(vsrlri_b, vsrlri, DF_BYTE, 3)   // trans_vsrlri_b
TRANS_VBIT_DF(vsrlri_h, vsrlri, DF_HALF, 4)   // trans_vsrlri_h
TRANS_VBIT_DF(vsrlri_w, vsrlri, DF_WORD, 5)   // trans_vsrlri_w
TRANS_VBIT_DF(vsrlri_d, vsrlri, DF_DOUBLE, 6) // trans_vsrlri_d
TRANS_VBIT_DF(vsrari_b, vsrari, DF_BYTE, 3)   // trans_vsrari_b
TRANS_VBIT_DF(vsrari_h, vsrari, DF_HALF, 4)   // trans_vsrari_h
TRANS_VBIT_DF(vsrari_w, vsrari, DF_WORD, 5)   // trans_vsrari_w
TRANS_VBIT_DF(vsrari_d, vsrari, DF_DOUBLE, 6) // trans_vsrari_d
#undef TRANS_VBIT_DF

#define TRANS_XVBIT_DF(name, op, DF, u_width) \
static bool trans_ ## name (DisasContext *ctx, arg_ ## name *a) \
{ \
    TCGv_i32 twd  = tcg_const_i32(a->xd); \
    TCGv_i32 tws  = tcg_const_i32(a->xj); \
    TCGv_i32 timm = tcg_const_i32(a->ui ## u_width ); \
    TCGv_i32 tdf  = tcg_const_i32(DF); \
    \
    check_lasx_enabled(ctx);                                          \
    gen_helper_lsx_ ## op ## _df(cpu_env, tdf, twd, tws, timm); \
    \
    tcg_temp_free_i32(twd); \
    tcg_temp_free_i32(tws); \
    tcg_temp_free_i32(timm); \
    tcg_temp_free_i32(tdf); \
    return true; \
}
TRANS_XVBIT_DF(xvbitrevi_b, xvbitrevi, DF_BYTE, 3)   // trans_xvbitrevi_b
TRANS_XVBIT_DF(xvbitrevi_h, xvbitrevi, DF_HALF, 4)   // trans_xvbitrevi_h
TRANS_XVBIT_DF(xvbitrevi_w, xvbitrevi, DF_WORD, 5)   // trans_xvbitrevi_w
TRANS_XVBIT_DF(xvbitrevi_d, xvbitrevi, DF_DOUBLE, 6) // trans_xvbitrevi_d
#undef TRANS_XVBIT_DF

#define TRANS_VSHUF4I(name, DF)\
static bool trans_ ## name (DisasContext *ctx, arg_ ## name *a) \
{ \
    TCGv_i32 twd  = tcg_const_i32(a->vd); \
    TCGv_i32 tws  = tcg_const_i32(a->vj); \
    TCGv_i32 timm = tcg_const_i32(a->ui8); \
    TCGv_i32 tdf  = tcg_const_i32(DF); \
    \
    check_lsx_enabled(ctx);    \
    gen_helper_lsx_vshuf4i_df(cpu_env, tdf, twd, tws, timm); \
    \
    tcg_temp_free_i32(twd); \
    tcg_temp_free_i32(tws); \
    tcg_temp_free_i32(timm); \
    tcg_temp_free_i32(tdf); \
    return true; \
}
TRANS_VSHUF4I(vshuf4i_b, DF_BYTE)   // trans_vshuf4i_b
TRANS_VSHUF4I(vshuf4i_h, DF_HALF)   // trans_vshuf4i_h
TRANS_VSHUF4I(vshuf4i_w, DF_WORD)   // trans_vshuf4i_w
TRANS_VSHUF4I(vshuf4i_d, DF_DOUBLE) // trans_vshuf4i_d
#undef TRANS_VSHUF4I

#define TRANS_FN_IMM8(name)\
static bool trans_ ## name (DisasContext *ctx, arg_ ## name *a) \
{ \
    TCGv_i32 twd  = tcg_const_i32(a->vd); \
    TCGv_i32 tws  = tcg_const_i32(a->vj); \
    TCGv_i32 timm = tcg_const_i32(a->ui8); \
    \
    check_lsx_enabled(ctx);    \
    gen_helper_lsx_ ## name (cpu_env, twd, tws, timm); \
    \
    tcg_temp_free_i32(twd); \
    tcg_temp_free_i32(tws); \
    tcg_temp_free_i32(timm); \
    return true; \
}
TRANS_FN_IMM8(vbitseli_b)  // trans_vbitseli_b
TRANS_FN_IMM8(vbitmvzi_b)  // trans_vbitmvzi_b
TRANS_FN_IMM8(vbitmvnzi_b) // trans_vbitmvnzi_b
TRANS_FN_IMM8(vandi_b)     // trans_vandi_b
TRANS_FN_IMM8(vori_b)      // trans_vori_b
TRANS_FN_IMM8(vxori_b)     // trans_vxori_b
TRANS_FN_IMM8(vnori_b)     // trans_vnori_b
#undef TRANS_FN_IMM8

#define TRANS_LSX_DJK_DF(name, op, DF)                          \
static bool trans_ ## name (DisasContext *ctx, arg_ ## name *a) \
{                                                               \
    TCGv_i32 twd = tcg_const_i32(a->xd);                        \
    TCGv_i32 tws = tcg_const_i32(a->xj);                        \
    TCGv_i32 twt = tcg_const_i32(a->xk);                        \
    TCGv_i32 tdf = tcg_const_i32(DF);                           \
                                                                \
    check_lasx_enabled(ctx);   \
    gen_helper_lsx_ ## op ## _df(cpu_env, tdf, twd, tws, twt);  \
                                                                \
    tcg_temp_free_i32(twd);                                     \
    tcg_temp_free_i32(tws);                                     \
    tcg_temp_free_i32(twt);                                     \
    tcg_temp_free_i32(tdf);                                     \
    return true;                                                \
}
TRANS_LSX_DJK_DF(xvshuf_h, xvshuf, DF_HALF)   // trans_xvshuf_h
TRANS_LSX_DJK_DF(xvshuf_w, xvshuf, DF_WORD)   // trans_xvshuf_w
TRANS_LSX_DJK_DF(xvshuf_d, xvshuf, DF_DOUBLE) // trans_xvshuf_d
TRANS_LSX_DJK_DF(xvperm_w, xvperm, DF_WORD)   // trans_xvperm_w
#undef  TRANS_LSX_DJK_DF

#define TRANS_LSX_DJSI_DF(name, op, u_width, DF)                \
static bool trans_ ## name (DisasContext *ctx, arg_ ## name *a) \
{                                                               \
    TCGv_i32 twd  = tcg_const_i32(a->xd);                       \
    TCGv_i32 tws  = tcg_const_i32(a->xj);                       \
    TCGv_i32 timm = tcg_const_i32(a->si ## u_width);            \
    TCGv_i32 tdf  = tcg_const_i32(DF);                          \
                                                                \
    check_lasx_enabled(ctx);   \
    gen_helper_lsx_ ## op ## _df(cpu_env, tdf, twd, tws, timm); \
                                                                \
    tcg_temp_free_i32(twd);                                     \
    tcg_temp_free_i32(tws);                                     \
    tcg_temp_free_i32(timm);                                    \
    tcg_temp_free_i32(tdf);                                     \
    return true;                                                \
}
TRANS_LSX_DJSI_DF(xvseqi_b, xvseqi, 5, DF_BYTE)      // trans_xvseqi_b
TRANS_LSX_DJSI_DF(xvseqi_h, xvseqi, 5, DF_HALF)      // trans_xvseqi_h
TRANS_LSX_DJSI_DF(xvseqi_w, xvseqi, 5, DF_WORD)      // trans_xvseqi_w
TRANS_LSX_DJSI_DF(xvseqi_d, xvseqi, 5, DF_DOUBLE)    // trans_xvseqi_d
TRANS_LSX_DJSI_DF(xvslei_b, xvslei_s, 5, DF_BYTE)    // trans_xvslei_b
TRANS_LSX_DJSI_DF(xvslei_h, xvslei_s, 5, DF_HALF)    // trans_xvslei_h
TRANS_LSX_DJSI_DF(xvslei_w, xvslei_s, 5, DF_WORD)    // trans_xvslei_w
TRANS_LSX_DJSI_DF(xvslei_d, xvslei_s, 5, DF_DOUBLE)  // trans_xvslei_d
TRANS_LSX_DJSI_DF(xvslti_b, xvslti_s, 5, DF_BYTE)    // trans_xvslti_b
TRANS_LSX_DJSI_DF(xvslti_h, xvslti_s, 5, DF_HALF)    // trans_xvslti_h
TRANS_LSX_DJSI_DF(xvslti_w, xvslti_s, 5, DF_WORD)    // trans_xvslti_w
TRANS_LSX_DJSI_DF(xvslti_d, xvslti_s, 5, DF_DOUBLE)  // trans_xvslti_d
TRANS_LSX_DJSI_DF(xvmaxi_b, xvmaxi_s, 5, DF_BYTE)    // trans_xvmaxi_b
TRANS_LSX_DJSI_DF(xvmaxi_h, xvmaxi_s, 5, DF_HALF)    // trans_xvmaxi_h
TRANS_LSX_DJSI_DF(xvmaxi_w, xvmaxi_s, 5, DF_WORD)    // trans_xvmaxi_w
TRANS_LSX_DJSI_DF(xvmaxi_d, xvmaxi_s, 5, DF_DOUBLE)  // trans_xvmaxi_d
TRANS_LSX_DJSI_DF(xvmini_b, xvmini_s, 5, DF_BYTE)    // trans_xvmini_b
TRANS_LSX_DJSI_DF(xvmini_h, xvmini_s, 5, DF_HALF)    // trans_xvmini_h
TRANS_LSX_DJSI_DF(xvmini_w, xvmini_s, 5, DF_WORD)    // trans_xvmini_w
TRANS_LSX_DJSI_DF(xvmini_d, xvmini_s, 5, DF_DOUBLE)  // trans_xvmini_d
#undef TRANS_LSX_DJSI_DF

#define TRANS_LSX_DJUI_DF(name, op, u_width, DF)                \
static bool trans_ ## name (DisasContext *ctx, arg_ ## name *a) \
{                                                               \
    TCGv_i32 twd  = tcg_const_i32(a->xd);                       \
    TCGv_i32 tws  = tcg_const_i32(a->xj);                       \
    TCGv_i32 timm = tcg_const_i32(a->ui ## u_width);            \
    TCGv_i32 tdf  = tcg_const_i32(DF);                          \
                                                                \
    check_lasx_enabled(ctx);   \
    gen_helper_lsx_ ## op ## _df(cpu_env, tdf, twd, tws, timm); \
                                                                \
    tcg_temp_free_i32(twd);                                     \
    tcg_temp_free_i32(tws);                                     \
    tcg_temp_free_i32(timm);                                    \
    tcg_temp_free_i32(tdf);                                     \
    return true;                                                \
}
TRANS_LSX_DJUI_DF(xvslei_bu, xvslei_u, 5, DF_BYTE)   // trans_xvslei_bu
TRANS_LSX_DJUI_DF(xvslei_hu, xvslei_u, 5, DF_HALF)   // trans_xvslei_hu
TRANS_LSX_DJUI_DF(xvslei_wu, xvslei_u, 5, DF_WORD)   // trans_xvslei_wu
TRANS_LSX_DJUI_DF(xvslei_du, xvslei_u, 5, DF_DOUBLE) // trans_xvslei_du
TRANS_LSX_DJUI_DF(xvslti_bu, xvslti_u, 5, DF_BYTE)   // trans_xvslti_bu
TRANS_LSX_DJUI_DF(xvslti_hu, xvslti_u, 5, DF_HALF)   // trans_xvslti_hu
TRANS_LSX_DJUI_DF(xvslti_wu, xvslti_u, 5, DF_WORD)   // trans_xvslti_wu
TRANS_LSX_DJUI_DF(xvslti_du, xvslti_u, 5, DF_DOUBLE) // trans_xvslti_du
TRANS_LSX_DJUI_DF(xvaddi_bu, xvaddi_u, 5, DF_BYTE)   // trans_xvaddi_bu
TRANS_LSX_DJUI_DF(xvaddi_hu, xvaddi_u, 5, DF_HALF)   // trans_xvaddi_hu
TRANS_LSX_DJUI_DF(xvaddi_wu, xvaddi_u, 5, DF_WORD)   // trans_xvaddi_wu
TRANS_LSX_DJUI_DF(xvaddi_du, xvaddi_u, 5, DF_DOUBLE) // trans_xvaddi_du
TRANS_LSX_DJUI_DF(xvsubi_bu, xvsubi_u, 5, DF_BYTE)   // trans_xvsubi_bu
TRANS_LSX_DJUI_DF(xvsubi_hu, xvsubi_u, 5, DF_HALF)   // trans_xvsubi_hu
TRANS_LSX_DJUI_DF(xvsubi_wu, xvsubi_u, 5, DF_WORD)   // trans_xvsubi_wu
TRANS_LSX_DJUI_DF(xvsubi_du, xvsubi_u, 5, DF_DOUBLE) // trans_xvsubi_du
TRANS_LSX_DJUI_DF(xvmaxi_bu, xvmaxi_u, 5, DF_BYTE)   // trans_xvmaxi_bu
TRANS_LSX_DJUI_DF(xvmaxi_hu, xvmaxi_u, 5, DF_HALF)   // trans_xvmaxi_hu
TRANS_LSX_DJUI_DF(xvmaxi_wu, xvmaxi_u, 5, DF_WORD)   // trans_xvmaxi_wu
TRANS_LSX_DJUI_DF(xvmaxi_du, xvmaxi_u, 5, DF_DOUBLE) // trans_xvmaxi_du
TRANS_LSX_DJUI_DF(xvmini_bu, xvmini_u, 5, DF_BYTE)   // trans_xvmini_bu
TRANS_LSX_DJUI_DF(xvmini_hu, xvmini_u, 5, DF_HALF)   // trans_xvmini_hu
TRANS_LSX_DJUI_DF(xvmini_wu, xvmini_u, 5, DF_WORD)   // trans_xvmini_wu
TRANS_LSX_DJUI_DF(xvmini_du, xvmini_u, 5, DF_DOUBLE) // trans_xvmini_du
#undef TRANS_LSX_DJUI_DF

#define TRANS_LSX_XVBSHIFT(name, op)                                \
static bool trans_ ## name (DisasContext *ctx, arg_ ## name *a) \
{                                                               \
    TCGv_i32 twd  = tcg_const_i32(a->xd);                       \
    TCGv_i32 tws  = tcg_const_i32(a->xj);                       \
    TCGv_i32 timm = tcg_const_i32(a->ui5 & 0xf);                \
                                                                \
    check_lasx_enabled(ctx);  									\
    gen_helper_lsx_ ## op (cpu_env, twd, tws, timm);   	 	 	\
                                                                \
    tcg_temp_free_i32(twd);                                     \
    tcg_temp_free_i32(tws);                                     \
    tcg_temp_free_i32(timm);                                    \
    return true;                                                \
}
TRANS_LSX_XVBSHIFT(xvbsll_v, xvbsll_v)  // trans_xvbsll_v
TRANS_LSX_XVBSHIFT(xvbsrl_v, xvbsrl_v)  // trans_xvbsrl_v
#undef TRANS_LSX_XVBSHIFT

#define TRANS_LSX_DJ(name)                                      \
static bool trans_ ## name (DisasContext *ctx, arg_ ## name *a) \
{                                                               \
    TCGv_i32 twd  = tcg_const_i32(a->xd);                       \
    TCGv_i32 tws  = tcg_const_i32(a->xj);                       \
                                                                \
    check_lasx_enabled(ctx);   \
    gen_helper_lsx_ ## name (cpu_env, twd, tws);                \
                                                                \
    tcg_temp_free_i32(twd);                                     \
    tcg_temp_free_i32(tws);                                     \
    return true;                                                \
}
TRANS_LSX_DJ(xvclo_b)     // trans_xvclo_b
TRANS_LSX_DJ(xvclo_h)     // trans_xvclo_h
TRANS_LSX_DJ(xvclo_w)     // trans_xvclo_w
TRANS_LSX_DJ(xvclo_d)     // trans_xvclo_d
TRANS_LSX_DJ(xvclz_b)     // trans_xvclz_b
TRANS_LSX_DJ(xvclz_h)     // trans_xvclz_h
TRANS_LSX_DJ(xvclz_w)     // trans_xvclz_w
TRANS_LSX_DJ(xvclz_d)     // trans_xvclz_d
TRANS_LSX_DJ(xvpcnt_b)    // trans_xvpcnt_b
TRANS_LSX_DJ(xvpcnt_h)    // trans_xvpcnt_h
TRANS_LSX_DJ(xvpcnt_w)    // trans_xvpcnt_w
TRANS_LSX_DJ(xvpcnt_d)    // trans_xvpcnt_d
TRANS_LSX_DJ(xvneg_b)     // trans_xvneg_b
TRANS_LSX_DJ(xvneg_h)     // trans_xvneg_h
TRANS_LSX_DJ(xvneg_w)     // trans_xvneg_w
TRANS_LSX_DJ(xvneg_d)     // trans_xvneg_d
TRANS_LSX_DJ(xvmskltz_b)  // trans_xvmskltz_b
TRANS_LSX_DJ(xvmskltz_h)  // trans_xvmskltz_h
TRANS_LSX_DJ(xvmskltz_w)  // trans_xvmskltz_w
TRANS_LSX_DJ(xvmskltz_d)  // trans_xvmskltz_d
TRANS_LSX_DJ(xvmskgez_b)  // trans_xvmskgez_b
TRANS_LSX_DJ(xvmsknz_b)   // trans_xvmsknz_b
TRANS_LSX_DJ(xvmskcopy_b) // trans_xvmskcopy_b
TRANS_LSX_DJ(xvmskfill_b) // trans_xvmskfill_b
#undef TRANS_LSX_DJ

#define TRANS_LSX_XCJ(name)                                     \
static bool trans_ ## name (DisasContext *ctx, arg_ ## name *a) \
{                                                               \
    TCGv_i32 tcd  = tcg_const_i32(a->cd);                       \
    TCGv_i32 tws  = tcg_const_i32(a->xj);                       \
                                                                \
    check_lasx_enabled(ctx);   \
    gen_helper_lsx_ ## name (cpu_env, tcd, tws);                \
                                                                \
    tcg_temp_free_i32(tcd);                                     \
    tcg_temp_free_i32(tws);                                     \
    return true;                                                \
}
TRANS_LSX_XCJ(xvseteqz_v)     // trans_xvseteqz_v
TRANS_LSX_XCJ(xvsetnez_v)     // trans_xvsetnez_v
TRANS_LSX_XCJ(xvsetanyeqz_b)  // trans_xvsetanyeqz_b
TRANS_LSX_XCJ(xvsetanyeqz_h)  // trans_xvsetanyeqz_h
TRANS_LSX_XCJ(xvsetanyeqz_w)  // trans_xvsetanyeqz_w
TRANS_LSX_XCJ(xvsetanyeqz_d)  // trans_xvsetanyeqz_d
TRANS_LSX_XCJ(xvsetallnez_b)  // trans_xvsetallnez_b
TRANS_LSX_XCJ(xvsetallnez_h)  // trans_xvsetallnez_h
TRANS_LSX_XCJ(xvsetallnez_w)  // trans_xvsetallnez_w
TRANS_LSX_XCJ(xvsetallnez_d)  // trans_xvsetallnez_d
#undef TRANS_LSX_XCJ

#define TRANS_XVBIT_DF(name, op, DF, u_width)                   \
static bool trans_ ## name (DisasContext *ctx, arg_ ## name *a) \
{                                                               \
    TCGv_i32 twd  = tcg_const_i32(a->xd);                       \
    TCGv_i32 tws  = tcg_const_i32(a->xj);                       \
    TCGv_i32 timm = tcg_const_i32(a->ui ## u_width );           \
    TCGv_i32 tdf  = tcg_const_i32(DF);                          \
                                                                \
    check_lasx_enabled(ctx);   \
    gen_helper_lsx_ ## op ## _df(cpu_env, tdf, twd, tws, timm); \
                                                                \
    tcg_temp_free_i32(twd);                                     \
    tcg_temp_free_i32(tws);                                     \
    tcg_temp_free_i32(timm);                                    \
    tcg_temp_free_i32(tdf);                                     \
    return true;                                                \
}
TRANS_XVBIT_DF(xvbitclri_b, xvbitclri, DF_BYTE, 3)      // trans_xvbitclri_b
TRANS_XVBIT_DF(xvbitclri_h, xvbitclri, DF_HALF, 4)      // trans_xvbitclri_h
TRANS_XVBIT_DF(xvbitclri_w, xvbitclri, DF_WORD, 5)      // trans_xvbitclri_w
TRANS_XVBIT_DF(xvbitclri_d, xvbitclri, DF_DOUBLE, 6)    // trans_xvbitclri_d
TRANS_XVBIT_DF(xvbitseti_b, xvbitseti, DF_BYTE, 3)      // trans_xvbitseti_b
TRANS_XVBIT_DF(xvbitseti_h, xvbitseti, DF_HALF, 4)      // trans_xvbitseti_h
TRANS_XVBIT_DF(xvbitseti_w, xvbitseti, DF_WORD, 5)      // trans_xvbitseti_w
TRANS_XVBIT_DF(xvbitseti_d, xvbitseti, DF_DOUBLE, 6)    // trans_xvbitseti_d
TRANS_XVBIT_DF(xvbstrc12i_b, xvbstrc12i, DF_BYTE, 3)    // trans_xvbstrc12i_b
TRANS_XVBIT_DF(xvbstrc12i_h, xvbstrc12i, DF_HALF, 4)    // trans_xvbstrc12i_h
TRANS_XVBIT_DF(xvbstrc12i_w, xvbstrc12i, DF_WORD, 5)    // trans_xvbstrc12i_w
TRANS_XVBIT_DF(xvbstrc12i_d, xvbstrc12i, DF_DOUBLE, 6)  // trans_xvbstrc12i_d
TRANS_XVBIT_DF(xvbstrc21i_b, xvbstrc21i, DF_BYTE, 3)    // trans_xvbstrc21i_b
TRANS_XVBIT_DF(xvbstrc21i_h, xvbstrc21i, DF_HALF, 4)    // trans_xvbstrc21i_h
TRANS_XVBIT_DF(xvbstrc21i_w, xvbstrc21i, DF_WORD, 5)    // trans_xvbstrc21i_w
TRANS_XVBIT_DF(xvbstrc21i_d, xvbstrc21i, DF_DOUBLE, 6)  // trans_xvbstrc21i_d
TRANS_XVBIT_DF(xvsat_b, xvsat_s, DF_BYTE, 3)            // trans_xvsat_b
TRANS_XVBIT_DF(xvsat_h, xvsat_s, DF_HALF, 4)            // trans_xvsat_h
TRANS_XVBIT_DF(xvsat_w, xvsat_s, DF_WORD, 5)            // trans_xvsat_w
TRANS_XVBIT_DF(xvsat_d, xvsat_s, DF_DOUBLE, 6)          // trans_xvsat_d
TRANS_XVBIT_DF(xvsat_bu, xvsat_u, DF_BYTE, 3)           // trans_xvsat_bu
TRANS_XVBIT_DF(xvsat_hu, xvsat_u, DF_HALF, 4)           // trans_xvsat_hu
TRANS_XVBIT_DF(xvsat_wu, xvsat_u, DF_WORD, 5)           // trans_xvsat_wu
TRANS_XVBIT_DF(xvsat_du, xvsat_u, DF_DOUBLE, 6)         // trans_xvsat_du
TRANS_XVBIT_DF(xvslli_b, xvslli, DF_BYTE, 3)            // trans_xvslli_b
TRANS_XVBIT_DF(xvslli_h, xvslli, DF_HALF, 4)            // trans_xvslli_h
TRANS_XVBIT_DF(xvslli_w, xvslli, DF_WORD, 5)            // trans_xvslli_w
TRANS_XVBIT_DF(xvslli_d, xvslli, DF_DOUBLE, 6)          // trans_xvslli_d
TRANS_XVBIT_DF(xvsrli_b, xvsrli, DF_BYTE, 3)            // trans_xvsrli_b
TRANS_XVBIT_DF(xvsrli_h, xvsrli, DF_HALF, 4)            // trans_xvsrli_h
TRANS_XVBIT_DF(xvsrli_w, xvsrli, DF_WORD, 5)            // trans_xvsrli_w
TRANS_XVBIT_DF(xvsrli_d, xvsrli, DF_DOUBLE, 6)          // trans_xvsrli_d
TRANS_XVBIT_DF(xvsrai_b, xvsrai, DF_BYTE, 3)            // trans_xvsrai_b
TRANS_XVBIT_DF(xvsrai_h, xvsrai, DF_HALF, 4)            // trans_xvsrai_h
TRANS_XVBIT_DF(xvsrai_w, xvsrai, DF_WORD, 5)            // trans_xvsrai_w
TRANS_XVBIT_DF(xvsrai_d, xvsrai, DF_DOUBLE, 6)          // trans_xvsrai_d
TRANS_XVBIT_DF(xvrotri_b, xvrotri, DF_BYTE, 3)          // trans_xvrotri_b
TRANS_XVBIT_DF(xvrotri_h, xvrotri, DF_HALF, 4)          // trans_xvrotri_h
TRANS_XVBIT_DF(xvrotri_w, xvrotri, DF_WORD, 5)          // trans_xvrotri_w
TRANS_XVBIT_DF(xvrotri_d, xvrotri, DF_DOUBLE, 6)        // trans_xvrotri_d
TRANS_XVBIT_DF(xvsrlri_b, xvsrlri, DF_BYTE, 3)          // trans_xvsrlri_b
TRANS_XVBIT_DF(xvsrlri_h, xvsrlri, DF_HALF, 4)          // trans_xvsrlri_h
TRANS_XVBIT_DF(xvsrlri_w, xvsrlri, DF_WORD, 5)          // trans_xvsrlri_w
TRANS_XVBIT_DF(xvsrlri_d, xvsrlri, DF_DOUBLE, 6)        // trans_xvsrlri_d
TRANS_XVBIT_DF(xvsrari_b, xvsrari, DF_BYTE, 3)          // trans_xvsrari_b
TRANS_XVBIT_DF(xvsrari_h, xvsrari, DF_HALF, 4)          // trans_xvsrari_h
TRANS_XVBIT_DF(xvsrari_w, xvsrari, DF_WORD, 5)          // trans_xvsrari_w
TRANS_XVBIT_DF(xvsrari_d, xvsrari, DF_DOUBLE, 6)        // trans_xvsrari_d
TRANS_XVBIT_DF(xvshuf4i_b, xvshuf4i, DF_BYTE, 8)        // trans_xvshuf4i_b
TRANS_XVBIT_DF(xvshuf4i_h, xvshuf4i, DF_HALF, 8)        // trans_xvshuf4i_h
TRANS_XVBIT_DF(xvshuf4i_w, xvshuf4i, DF_WORD, 8)        // trans_xvshuf4i_w
TRANS_XVBIT_DF(xvshuf4i_d, xvshuf4i, DF_DOUBLE, 8)      // trans_xvshuf4i_d
#undef TRANS_XVBIT_DF

#define TRANS_XFN_IMM8(name)                                    \
static bool trans_ ## name (DisasContext *ctx, arg_ ## name *a) \
{                                                               \
    TCGv_i32 twd  = tcg_const_i32(a->xd);                       \
    TCGv_i32 tws  = tcg_const_i32(a->xj);                       \
    TCGv_i32 timm = tcg_const_i32(a->ui8);                      \
                                                                \
    check_lasx_enabled(ctx);   \
    gen_helper_lsx_ ## name (cpu_env, twd, tws, timm);          \
                                                                \
    tcg_temp_free_i32(twd);                                     \
    tcg_temp_free_i32(tws);                                     \
    tcg_temp_free_i32(timm);                                    \
    return true;                                                \
}
TRANS_XFN_IMM8(xvbitseli_b)  // trans_xvbitseli_b
TRANS_XFN_IMM8(xvbitmvzi_b)  // trans_xvbitmvzi_b
TRANS_XFN_IMM8(xvbitmvnzi_b) // trans_xvbitmvnzi_b
TRANS_XFN_IMM8(xvandi_b)     // trans_xvandi_b
TRANS_XFN_IMM8(xvori_b)      // trans_xvori_b
TRANS_XFN_IMM8(xvxori_b)     // trans_xvxori_b
TRANS_XFN_IMM8(xvnori_b)     // trans_xvnori_b
#undef TRANS_XFN_IMM8

static bool trans_vfrstpi_b(DisasContext *ctx, arg_vfrstpi_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->ui5);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vfrstpi_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vfrstpi_h(DisasContext *ctx, arg_vfrstpi_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->ui5);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vfrstpi_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vclrstri_v(DisasContext *ctx, arg_vclrstri_v *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 twt = tcg_const_i32(a->ui5);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vclrstri_v(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vmepatmsk_v(DisasContext *ctx, arg_vmepatmsk_v *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->mode);
    TCGv_i32 twt = tcg_const_i32(a->ui5);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vmepatmsk_v(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_vfrstm_b(DisasContext *ctx, arg_vfrstm_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vfrstm_b(cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_vfrstm_h(DisasContext *ctx, arg_vfrstm_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vfrstm_h(cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_vextl_w_b(DisasContext *ctx, arg_vextl_w_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vextl_w_b(cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_vextl_d_b(DisasContext *ctx, arg_vextl_d_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vextl_d_b(cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_vextl_d_h(DisasContext *ctx, arg_vextl_d_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vextl_d_h(cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_vextl_w_bu(DisasContext *ctx, arg_vextl_w_bu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vextl_w_bu(cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_vextl_d_bu(DisasContext *ctx, arg_vextl_d_bu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vextl_d_bu(cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_vextl_d_hu(DisasContext *ctx, arg_vextl_d_hu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vextl_d_hu(cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_vhadd8_d_bu(DisasContext *ctx, arg_vhadd8_d_bu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vhadd8_d_bu(cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_vhminpos_w_hu(DisasContext *ctx, arg_vhminpos_w_hu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vhminpos_w_hu(cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_vhminpos_d_hu(DisasContext *ctx, arg_vhminpos_d_hu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vhminpos_d_hu(cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_vhminpos_q_hu(DisasContext *ctx, arg_vhminpos_q_hu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vhminpos_q_hu(cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_vclrtail_b(DisasContext *ctx, arg_vclrtail_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vclrtail_b(cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_vclrtail_h(DisasContext *ctx, arg_vclrtail_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);

    check_lsx_enabled(ctx);
    gen_helper_lsx_vclrtail_h(cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_vextrins_d(DisasContext *ctx, arg_vextrins_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 ui8 = tcg_const_i32(a->ui8);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vextrins_d(cpu_env, twd, tws, ui8);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(ui8);
    return true;
}

static bool trans_vextrins_w(DisasContext *ctx, arg_vextrins_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 ui8 = tcg_const_i32(a->ui8);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vextrins_w(cpu_env, twd, tws, ui8);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(ui8);
    return true;
}

static bool trans_vextrins_h(DisasContext *ctx, arg_vextrins_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 ui8 = tcg_const_i32(a->ui8);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vextrins_h(cpu_env, twd, tws, ui8);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(ui8);
    return true;
}

static bool trans_vextrins_b(DisasContext *ctx, arg_vextrins_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 ui8 = tcg_const_i32(a->ui8);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vextrins_b(cpu_env, twd, tws, ui8);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(ui8);
    return true;
}

static bool trans_vshufi1_b(DisasContext *ctx, arg_vshufi1_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 ui8 = tcg_const_i32(a->ui8);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vshufi1_b(cpu_env, twd, tws, ui8);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(ui8);
    return true;
}

static bool trans_vshufi2_b(DisasContext *ctx, arg_vshufi2_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 ui8 = tcg_const_i32(a->ui8);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vshufi2_b(cpu_env, twd, tws, ui8);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(ui8);
    return true;
}

static bool trans_vshufi3_b(DisasContext *ctx, arg_vshufi3_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 ui8 = tcg_const_i32(a->ui8);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vshufi3_b(cpu_env, twd, tws, ui8);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(ui8);
    return true;
}

static bool trans_vshufi4_b(DisasContext *ctx, arg_vshufi4_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 ui8 = tcg_const_i32(a->ui8);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vshufi4_b(cpu_env, twd, tws, ui8);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(ui8);
    return true;
}

static bool trans_vshufi1_h(DisasContext *ctx, arg_vshufi1_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 ui8 = tcg_const_i32(a->ui8);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vshufi1_h(cpu_env, twd, tws, ui8);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(ui8);
    return true;
}

static bool trans_vshufi2_h(DisasContext *ctx, arg_vshufi2_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 ui8 = tcg_const_i32(a->ui8);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vshufi2_h(cpu_env, twd, tws, ui8);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(ui8);
    return true;
}

static bool trans_vseli_h(DisasContext *ctx, arg_vseli_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 ui8 = tcg_const_i32(a->ui8);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vseli_h(cpu_env, twd, tws, ui8);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(ui8);
    return true;
}

static bool trans_vseli_w(DisasContext *ctx, arg_vseli_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 ui8 = tcg_const_i32(a->ui8);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vseli_w(cpu_env, twd, tws, ui8);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(ui8);
    return true;
}

static bool trans_vseli_d(DisasContext *ctx, arg_vseli_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 ui8 = tcg_const_i32(a->ui8);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vseli_d(cpu_env, twd, tws, ui8);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(ui8);
    return true;
}

static bool trans_xvadd_q(DisasContext *ctx, arg_xvadd_q *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvadd_q(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvsub_q(DisasContext *ctx, arg_xvsub_q *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsub_q(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvsigncov_b(DisasContext *ctx, arg_xvsigncov_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsigncov_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvsigncov_h(DisasContext *ctx, arg_xvsigncov_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsigncov_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvsigncov_w(DisasContext *ctx, arg_xvsigncov_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsigncov_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvsigncov_d(DisasContext *ctx, arg_xvsigncov_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsigncov_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvhadd4_h_bu(DisasContext *ctx, arg_xvhadd4_h_bu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvhadd4_h_bu(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvshuf4_w(DisasContext *ctx, arg_xvshuf4_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvshuf4_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvshuf2_d(DisasContext *ctx, arg_xvshuf2_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvshuf2_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}
static bool trans_xvpmul_w(DisasContext *ctx, arg_xvpmul_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvpmul_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvpmul_d(DisasContext *ctx, arg_xvpmul_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvpmul_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvpmuh_w(DisasContext *ctx, arg_xvpmuh_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvpmuh_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvpmuh_d(DisasContext *ctx, arg_xvpmuh_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvpmuh_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvpmulacc_w(DisasContext *ctx, arg_xvpmulacc_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvpmulacc_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvpmulacc_d(DisasContext *ctx, arg_xvpmulacc_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvpmulacc_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvpmuhacc_w(DisasContext *ctx, arg_xvpmuhacc_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvpmuhacc_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvpmuhacc_d(DisasContext *ctx, arg_xvpmuhacc_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvpmuhacc_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvpmulwl_h_b(DisasContext *ctx, arg_xvpmulwl_h_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvpmulwl_h_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvpmulwl_w_h(DisasContext *ctx, arg_xvpmulwl_w_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvpmulwl_w_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvpmulwl_d_w(DisasContext *ctx, arg_xvpmulwl_d_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvpmulwl_d_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvpmulwl_q_d(DisasContext *ctx, arg_xvpmulwl_q_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvpmulwl_q_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvpmulwh_h_b(DisasContext *ctx, arg_xvpmulwh_h_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvpmulwh_h_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvpmulwh_w_h(DisasContext *ctx, arg_xvpmulwh_w_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvpmulwh_w_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvpmulwh_d_w(DisasContext *ctx, arg_xvpmulwh_d_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvpmulwh_d_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvpmulwh_q_d(DisasContext *ctx, arg_xvpmulwh_q_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvpmulwh_q_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvpmaddwl_h_b(DisasContext *ctx, arg_xvpmaddwl_h_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvpmaddwl_h_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvpmaddwl_w_h(DisasContext *ctx, arg_xvpmaddwl_w_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvpmaddwl_w_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvpmaddwl_d_w(DisasContext *ctx, arg_xvpmaddwl_d_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvpmaddwl_d_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvpmaddwl_q_d(DisasContext *ctx, arg_xvpmaddwl_q_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvpmaddwl_q_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvpmaddwh_h_b(DisasContext *ctx, arg_xvpmaddwh_h_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvpmaddwh_h_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvpmaddwh_w_h(DisasContext *ctx, arg_xvpmaddwh_w_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvpmaddwh_w_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvpmaddwh_d_w(DisasContext *ctx, arg_xvpmaddwh_d_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvpmaddwh_d_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvpmaddwh_q_d(DisasContext *ctx, arg_xvpmaddwh_q_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvpmaddwh_q_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvpdp2_q_d(DisasContext *ctx, arg_xvpdp2_q_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvpdp2_q_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvpdp2add_q_d(DisasContext *ctx, arg_xvpdp2add_q_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvpdp2add_q_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvcdp4_re_d_h(DisasContext *ctx, arg_xvcdp4_re_d_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvcdp4_re_d_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvcdp4_im_d_h(DisasContext *ctx, arg_xvcdp4_im_d_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvcdp4_im_d_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvcdp4add_re_d_h(DisasContext *ctx, arg_xvcdp4add_re_d_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvcdp4add_re_d_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvcdp4add_im_d_h(DisasContext *ctx, arg_xvcdp4add_im_d_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvcdp4add_im_d_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvcdp2_re_q_w(DisasContext *ctx, arg_xvcdp2_re_q_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvcdp2_re_q_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvcdp2_im_q_w(DisasContext *ctx, arg_xvcdp2_im_q_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvcdp2_im_q_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvcdp2add_re_q_w(DisasContext *ctx, arg_xvcdp2add_re_q_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvcdp2add_im_q_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvcdp2add_im_q_w(DisasContext *ctx, arg_xvcdp2add_im_q_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvcdp2add_im_q_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvsignsel_w(DisasContext *ctx, arg_xvsignsel_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsignsel_w(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvsignsel_d(DisasContext *ctx, arg_xvsignsel_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->xk);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvsignsel_d(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvrandsigni_b(DisasContext *ctx, arg_xvrandsigni_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->ui5);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvrandsigni_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvrandsigni_h(DisasContext *ctx, arg_xvrandsigni_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->ui5);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvrandsigni_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvrorsigni_b(DisasContext *ctx, arg_xvrorsigni_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->ui5);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvrorsigni_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvrorsigni_h(DisasContext *ctx, arg_xvrorsigni_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->ui5);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvrorsigni_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvfrstpi_b(DisasContext *ctx, arg_xvfrstpi_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->ui5);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvfrstpi_b(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvfrstpi_h(DisasContext *ctx, arg_xvfrstpi_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->ui5);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvfrstpi_h(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvclrstri_v(DisasContext *ctx, arg_xvclrstri_v *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 twt = tcg_const_i32(a->ui5);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvclrstri_v(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvmepatmsk_v(DisasContext *ctx, arg_xvmepatmsk_v *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->mode);
    TCGv_i32 twt = tcg_const_i32(a->ui5);

    check_lasx_enabled(ctx);
    gen_helper_lsx_vmepatmsk_v(cpu_env, twd, tws, twt);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(twt);
    return true;
}

static bool trans_xvfrstm_b(DisasContext *ctx, arg_xvfrstm_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvfrstm_b(cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_xvfrstm_h(DisasContext *ctx, arg_xvfrstm_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvfrstm_h(cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_xvextl_w_b(DisasContext *ctx, arg_xvextl_w_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvextl_w_b(cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_xvextl_d_b(DisasContext *ctx, arg_xvextl_d_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvextl_d_b(cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_xvextl_d_h(DisasContext *ctx, arg_xvextl_d_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvextl_d_h(cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_xvextl_w_bu(DisasContext *ctx, arg_xvextl_w_bu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvextl_w_bu(cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_xvextl_d_bu(DisasContext *ctx, arg_xvextl_d_bu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvextl_d_bu(cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_xvextl_d_hu(DisasContext *ctx, arg_xvextl_d_hu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvextl_d_hu(cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_xvhadd8_d_bu(DisasContext *ctx, arg_xvhadd8_d_bu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvhadd8_d_bu(cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_xvhminpos_w_hu(DisasContext *ctx, arg_xvhminpos_w_hu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvhminpos_w_hu(cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_xvhminpos_d_hu(DisasContext *ctx, arg_xvhminpos_d_hu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvhminpos_d_hu(cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_xvhminpos_q_hu(DisasContext *ctx, arg_xvhminpos_q_hu *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvhminpos_q_hu(cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_xvclrtail_b(DisasContext *ctx, arg_xvclrtail_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvclrtail_b(cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_xvclrtail_h(DisasContext *ctx, arg_xvclrtail_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);

    check_lasx_enabled(ctx);
    gen_helper_lsx_xvclrtail_h(cpu_env, twd, tws);

    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    return true;
}

static bool trans_xvextrins_d(DisasContext *ctx, arg_xvextrins_d *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 ui8 = tcg_const_i32(a->ui8);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvextrins_d(cpu_env, twd, tws, ui8);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(ui8);
    return true;
}

static bool trans_xvextrins_w(DisasContext *ctx, arg_xvextrins_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 ui8 = tcg_const_i32(a->ui8);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvextrins_w(cpu_env, twd, tws, ui8);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(ui8);
    return true;
}

static bool trans_xvextrins_h(DisasContext *ctx, arg_xvextrins_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 ui8 = tcg_const_i32(a->ui8);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvextrins_h(cpu_env, twd, tws, ui8);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(ui8);
    return true;
}

static bool trans_xvextrins_b(DisasContext *ctx, arg_xvextrins_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 ui8 = tcg_const_i32(a->ui8);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvextrins_b(cpu_env, twd, tws, ui8);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(ui8);
    return true;
}

static bool trans_xvshufi1_b(DisasContext *ctx, arg_xvshufi1_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 ui8 = tcg_const_i32(a->ui8);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvshufi1_b(cpu_env, twd, tws, ui8);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(ui8);
    return true;
}

static bool trans_xvshufi2_b(DisasContext *ctx, arg_xvshufi2_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 ui8 = tcg_const_i32(a->ui8);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvshufi2_b(cpu_env, twd, tws, ui8);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(ui8);
    return true;
}

static bool trans_xvshufi3_b(DisasContext *ctx, arg_xvshufi3_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 ui8 = tcg_const_i32(a->ui8);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvshufi3_b(cpu_env, twd, tws, ui8);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(ui8);
    return true;
}

static bool trans_xvshufi4_b(DisasContext *ctx, arg_xvshufi4_b *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 ui8 = tcg_const_i32(a->ui8);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvshufi4_b(cpu_env, twd, tws, ui8);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(ui8);
    return true;
}

static bool trans_xvshufi1_h(DisasContext *ctx, arg_xvshufi1_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 ui8 = tcg_const_i32(a->ui8);
    check_lasx_enabled(ctx);
    gen_helper_lsx_vshufi1_h(cpu_env, twd, tws, ui8);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(ui8);
    return true;
}

static bool trans_xvshufi2_h(DisasContext *ctx, arg_xvshufi2_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 ui8 = tcg_const_i32(a->ui8);
    check_lasx_enabled(ctx);
    gen_helper_lsx_vshufi2_h(cpu_env, twd, tws, ui8);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(ui8);
    return true;
}

static bool trans_xvseli_h(DisasContext *ctx, arg_xvseli_h *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 ui8 = tcg_const_i32(a->ui8);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvseli_h(cpu_env, twd, tws, ui8);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(ui8);
    return true;
}

static bool trans_vpermi_w(DisasContext *ctx, arg_vpermi_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->vd);
    TCGv_i32 tws = tcg_const_i32(a->vj);
    TCGv_i32 sel = tcg_const_i32(a->ui8);
    check_lsx_enabled(ctx);
    gen_helper_lsx_vpermi_w(cpu_env, twd, tws, sel);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(sel);
    return true;
}

static bool trans_xvpermi_w(DisasContext *ctx, arg_xvpermi_w *a)
{
    TCGv_i32 twd = tcg_const_i32(a->xd);
    TCGv_i32 tws = tcg_const_i32(a->xj);
    TCGv_i32 sel = tcg_const_i32(a->ui8);
    check_lasx_enabled(ctx);
    gen_helper_lsx_xvpermi_w(cpu_env, twd, tws, sel);
    tcg_temp_free_i32(twd);
    tcg_temp_free_i32(tws);
    tcg_temp_free_i32(sel);
    return true;
}
