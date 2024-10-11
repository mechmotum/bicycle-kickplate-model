"""Parameter data for the bicycle, rider, and linear tire model."""

from nonlin_sym import ps

# TODO : Define all symbols in a seperate Python file so the can be imported
# and used without constructing the symbolic equations of motion.
# NOTE : Constant parameters must be sored into alphabetical order! We rely on
# the index number in various places in the code when working with the
# parameter array.
(
    c_af,  # 0
    c_ar,  # 1
    c_f,  # 2
    c_maf,  # 3
    c_mar,  # 4
    c_mpf,  # 5
    c_mpr,  # 6
    c_pf,  # 7
    c_pr,  # 8
    c_r,  # 9
    d1,  # 10
    d2,  # 11
    d3,  # 12
    g,  # 13
    ic11,  # 14
    ic22,  # 15
    ic31,  # 16
    ic33,  # 17
    id11,  # 18
    id22,  # 19
    ie11,  # 20
    ie22,  # 21
    ie31,  # 22
    ie33,  # 23
    if11,  # 24
    if22,  # 25
    k_f,  # 26
    k_r,  # 27
    l1,  # 28
    l2,  # 29
    l3,  # 30
    l4,  # 31
    mc,  # 32
    md,  # 33
    me,  # 34
    mf,  # 35
    r_tf,  # 36
    r_tr,  # 37
    rf,  # 38
    rr,  # 39
    s_yf,  # 40
    s_yr,  # 41
    s_zf,  # 42
    s_zr,  # 43
) = ps

# Batavus Browser with Jason from BicycleParameters and tire parameters from
# Andrew and Gabriele or other references.
browser_jason_par = {
    c_af: 11.46,  # estimates from Andrew's dissertation (done by him)
    c_ar: 11.46,
    c_f: 4000.0,  # guess
    c_maf: 0.33,  # 0.33 is rough calc from Gabriele's data
    c_mar: 0.33,
    c_mpf: 0.0,  # need real numbers for this
    c_mpr: 0.0,  # need real numbers for this
    c_pf: 0.573,
    c_pr: 0.573,
    c_r: 4000.0,  # guess
    d1: 0.9631492634872098,
    d2: 0.4338396131640938,
    d3: 0.0705000000001252,
    g: 9.81,
    ic11: 11.519805885486146,
    ic22: 12.2177848012,
    ic31: 1.57915608541552,
    ic33: 2.959474124693854,
    id11: 0.0883819364527,
    id22: 0.152467620286,
    ie11: 0.2811355367159554,
    ie22: 0.246138810935,
    ie31: 0.0063377219110826045,
    ie33: 0.06782113764394461,
    if11: 0.0904106601579,
    if22: 0.149389340425,
    k_f: 120000.0,  # ~ twice the stiffness of a 1.25" tire from Rothhamel 2024
    k_r: 120000.0,  # ~ twice the stiffness of a 1.25" tire from Rothhamel 2024
    l1: 0.5384415640161426,
    l2: -0.531720230353059,
    l3: -0.07654646159268344,
    l4: -0.47166687226492093,
    mc: 81.86,
    md: 3.11,
    me: 3.22,
    mf: 2.02,
    r_tf: 0.01,
    r_tr: 0.01,
    rf: 0.34352982332,
    rr: 0.340958858855,
    s_yf: 0.175,  # Andrew's estimates from his dissertation data
    s_yr: 0.175,
    s_zf: 0.175,
    s_zr: 0.175,
}
