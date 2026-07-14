#################################################################################
#################################################################################

fig, ax = plt.subplots(3,1, figsize=(7,22))

ax1=ax[1]; ax2=ax[2]; ax3=ax[3];

fig.tight_layout(pad=25.0)

fsize = 26
lsize = 24

msize = 100.0

x1_all = vcat(kx_EKE_max_mu0p125[2:6], kx_EKE_max_mu0p125[8:end-1])
y1_all = vcat(kx_tau_wave_min_mu0p125[2:6], kx_tau_wave_min_mu0p125[8:end-1])

x2_all = vcat(kx_EKE_max_mu0p25[2:6], kx_EKE_max_mu0p25[8:end-1])
y2_all = vcat(kx_tau_wave_min_mu0p25[2:6], kx_tau_wave_min_mu0p25[8:end-1])

x3_all = vcat(kx_EKE_max_mu0p5[2:6], kx_EKE_max_mu0p5[8:end-1])
y3_all = vcat(kx_tau_wave_min_mu0p5[2:6], kx_tau_wave_min_mu0p5[8:end-1])

#################################################################################
#################################################################################
R2_mu0p125 = r2_1to1(x1_all, y1_all)
R2_mu0p25 = r2_1to1(x2_all, y2_all)
R2_mu0p5 = r2_1to1(x3_all, y3_all)

R2_all = r2_1to1(vcat(x1_all, x2_all, x3_all), vcat(y1_all, y2_all, y3_all))

#################################################################################
#################################################################################

ax1.scatter(x1_all, y1_all, color="red", marker="P", s=msize, label=L"$\kappa^* = 0.125 \mathrm{; \ } R^2= %$(round(R2_mu0p125, digits=2)) $")
ax1.scatter(x2_all, y2_all, color="green", marker="X", s=msize, label=L"$\kappa^* = 0.25 \mathrm{; \ \ } R^2= %$(round(R2_mu0p25, digits=2)) $")
ax1.scatter(x3_all, y3_all, color="blue", marker="o", s=msize, label=L"$\kappa^* = 0.5 \mathrm{; \ \ \ } R^2= %$(round(R2_mu0p5, digits=2)) $")

ax1.plot([0,1], [0,1], "r--")

ax1.set_xlim(0,1); ax1.set_ylim(0,1)

ax1.set_xlabel(L"k^\mathrm{diag.}_{\tau^{-1} \mathrm{ \ max.}}", fontsize=fsize, pad=-0.1)
ax1.set_ylabel(L"k^\mathrm{pred.}_{\tau^{-1} \mathrm{ \ min.}}", fontsize=fsize)

ax1.legend(loc="lower right", bbox_to_anchor=(1.175, -0.025), fontsize=lsize)

ax1.tick_params(axis="both", labelsize=lsize)

ax1.grid()
ax1.set_axisbelow(true)


#################################################################################
#################################################################################

#################################################################################
#################################################################################

x1_all = kx_arrest_mu0p125
y1_all = k_diag_max_EKE_mu0p125

x2_all = kx_arrest_mu0p25
y2_all = k_diag_max_EKE_mu0p25

x3_all = kx_arrest_mu0p5
y3_all = k_diag_max_EKE_mu0p5

#################################################################################
#################################################################################

R2_mu0p125 = r2_1to1(x1_all, y1_all)
R2_mu0p25 = r2_1to1(x2_all, y2_all)
R2_mu0p5 = r2_1to1(x3_all, y3_all)

R2_all = r2_1to1(vcat(x1_all, x2_all, x3_all), vcat(y1_all, y2_all, y3_all))

#################################################################################
#################################################################################

val1 = (round(R2_mu0p125, digits=2))

ax2.scatter(x1_all, y1_all, color="red", marker="P", s=msize, label=L"$\kappa^* = 0.125 \mathrm{; \ } R^2= %$val1 $")
ax2.scatter(x2_all, y2_all, color="green", marker="X", s=msize, label=L"$\kappa^* = 0.25 \mathrm{; \ \ } R^2= %$(round(R2_mu0p25, digits=2)) $")
ax2.scatter(x3_all, y3_all, color="blue", marker="o", s=msize, label=L"$\kappa^* = 0.5 \mathrm{; \ \ \ } R^2= %$(round(R2_mu0p5, digits=2)) $")

ax2.plot([0,1], [0,1], "r--")

ax2.set_xlim(0,1); ax2.set_ylim(0,1)

ax2.set_xlabel(L"k^\mathrm{diag.}_{\mathrm{arrest}}", fontsize=fsize)
ax2.set_ylabel(L"k^\mathrm{diag.}_{\mathrm{EKE \ max.}}", fontsize=fsize)

ax2.legend(loc="lower right", bbox_to_anchor=(1.175, -0.025), fontsize=lsize)

ax2.tick_params(axis="both", labelsize=lsize)

ax2.grid()
ax2.set_axisbelow(true)


#################################################################################
#################################################################################


ax3.plot(kxp, EKE_mu0p25_hm1p25[:,slice_ind] ./ ((2*pi)^2), color=h_color(h0s[2],hc_range,colors), linestyle="dashed")
ax3.plot(kxp, EKE_mu0p25_hm1p0[:,slice_ind] ./ ((2*pi)^2), color=h_color(h0s[3],hc_range,colors), linestyle="dashed")
ax3.plot(kxp, EKE_mu0p25_hm0p75[:,slice_ind] ./ ((2*pi)^2), color=h_color(h0s[4],hc_range,colors), linestyle="dashed")
ax3.plot(kxp, EKE_mu0p25_hm0p5[:,slice_ind] ./ ((2*pi)^2), color=h_color(h0s[5],hc_range,colors), linestyle="dashed")
ax3.plot(kxp, EKE_mu0p25_hm0p25[:,slice_ind] ./ ((2*pi)^2), color=h_color(h0s[6],hc_range,colors), linestyle="dashed")


ax3.plot(kxp, EKE_mu0p25_h1p25[:,slice_ind] ./ ((2*pi)^2), color=h_color(h0s[12],hc_range,colors))
ax3.plot(kxp, EKE_mu0p25_h1p0[:,slice_ind] ./ ((2*pi)^2), color=h_color(h0s[11],hc_range,colors))
ax3.plot(kxp, EKE_mu0p25_h0p75[:,slice_ind] ./ ((2*pi)^2), color=h_color(h0s[10],hc_range,colors))
ax3.plot(kxp, EKE_mu0p25_h0p5[:,slice_ind] ./ ((2*pi)^2), color=h_color(h0s[9],hc_range,colors))
ax3.plot(kxp, EKE_mu0p25_h0p25[:,slice_ind] ./ ((2*pi)^2), color=h_color(h0s[8],hc_range,colors))

cc = 0.01 / ((2*pi)^2)
ax3.plot(kxp, cc .* kxp.^(-4), "k--", label=L"k_{x}^{-4}")


# # U2_max_mu0p25 = cc .* (kx_EKE_max_mu0p25[2:end-1]).^(-4)


# # ax2.scatter(kx_EKE_max_mu0p25[2:end-1], U2_max_mu0p25, color="black", marker="P")

# kx_EKE_pred_mu0p25 = vcat(kx_tau_wave_min_mu0p25[2:6], kx_tau_wave_min_mu0p25[8:end-1])
# U2_max_mu0p25 = cc .* kx_EKE_pred_mu0p25.^(-4)

# for (i,k) in enumerate(vcat(range(2,6), range(8,12)))
#     # ax2.plot([kx_tau_wave_min_mu0p25[ii]], [U2_max_mu0p25[ii-1]], color=h_color(h0s[ii],hc_range,colors), marker="P", markersize=15.0)

#     # kx_EKE_pred_mu0p25[ii-1] = kx_EKE_max_mu0p25[ii]
#     if k<7
#         # ax2.plot([kx_EKE_pred_mu0p25[ii-1]], [U2_max_mu0p25[ii-1]], color=h_color(h0s[ii],hc_range,colors), marker="P", markersize=15.0)
#         ax3.plot([kx_EKE_pred_mu0p25[i]], [U2_max_mu0p25[i]], color=h_color(h0s[k],hc_range,colors), marker="P", markersize=15.0)
#     else
#         # ax2.plot([kx_EKE_pred_mu0p25[ii-1]], [U2_max_mu0p25[ii-1]], color=h_color(h0s[ii],hc_range,colors), marker=">", markersize=15.0)
#         ax3.plot([kx_EKE_pred_mu0p25[i]], [U2_max_mu0p25[i]], color=h_color(h0s[k],hc_range,colors), marker=">", markersize=15.0)
#     end
# end


ax3.set_xlim(1e-2, 5e0)
ax3.set_yscale("log")
ax3.set_xscale("log")

ax3.set_ylim(1e-2 / ((2*pi)^2), 5e2 / ((2*pi)^2))

ax3.set_xlabel(L"k_{x} \ \lambda", fontsize=fsize)
ax3.set_ylabel(L"\mathrm{EKE \ [m}^2 \mathrm{s}^{-2} \mathrm{]}", fontsize=fsize)

ax3.tick_params(axis="both", labelsize=lsize)

ax3.legend(fontsize=lsize)

ax3.grid()
ax3.set_axisbelow(true)

#################################################################################
#################################################################################


for (i,t) in enumerate(["(a)", "(b)", "(c)"])
    ax[i].text(0.05, 0.95, t, transform=ax[i].transAxes, fontsize=fsize,
            ha="left", va="top")
end


#################################################################################
#################################################################################


savefig("./JPO_rev2_figs/kpred_kdiag_EKE_panels.pdf",bbox_inches="tight")










