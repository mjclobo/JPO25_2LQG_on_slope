
function plot_timescales(ax, tau_wave_BC, tau_wave_BT, tau_EKE, kxp)

    norm = 3600 * 24 * 31

    ax.plot(kxp, tau_wave_BC .* norm, "k-")
    ax.plot(kxp, tau_wave_BT .* norm, "k-")

    ax.plot(kxp, tau_EKE .* (norm), "r-")

    ax.axvline(sqrt.(fftshift(grid.Ksq)[:,513])[argmax(tau_EKE)] * Ld, color="red", linestyle="dashed")
    ax.axvline(kxp[513+argmin(maximum(hcat(tau_wave_BC , tau_wave_BT), dims=2)[:][514:end])], color="black", linestyle="dashed")
    
end

fsize = 26
lsize = 20

slice_ind = 513

fig, ax = plt.subplots(2,3, figsize=(16,10))

plot_timescales(ax[6], tau_wave_BC[1,:,slice_ind], tau_wave_BT[1,:,slice_ind], tau_EKE_mu0p25_hm1p25[:,slice_ind], kxp)
plot_timescales(ax[4], tau_wave_BC[2,:,slice_ind], tau_wave_BT[2,:,slice_ind], tau_EKE_mu0p25_hm0p75[:,slice_ind], kxp)
plot_timescales(ax[2], tau_wave_BC[3,:,slice_ind], tau_wave_BT[3,:,slice_ind], tau_EKE_mu0p25_hm0p25[:,slice_ind], kxp)

plot_timescales(ax[5], tau_wave_BC[6,:,slice_ind], tau_wave_BT[6,:,slice_ind], tau_EKE_mu0p25_h1p25[:,slice_ind], kxp)
plot_timescales(ax[3], tau_wave_BC[5,:,slice_ind], tau_wave_BT[5,:,slice_ind], tau_EKE_mu0p25_h0p75[:,slice_ind], kxp)
plot_timescales(ax[1], tau_wave_BC[4,:,slice_ind], tau_wave_BT[4,:,slice_ind], tau_EKE_mu0p25_h0p25[:,slice_ind], kxp)


# ax[6].axvline(lpeak_mu0p25_hm1p25[slice_ind] * (2*pi*Ld), color="green", linestyle="dashed")
# ax[4].axvline(lpeak_mu0p25_hm0p75[slice_ind] * (2*pi*Ld), color="green", linestyle="dashed")
# ax[2].axvline(lpeak_mu0p25_hm0p25[slice_ind] * (2*pi*Ld), color="green", linestyle="dashed")

# ax[5].axvline(lpeak_mu0p25_h1p25[slice_ind] * (2*pi*Ld), color="green", linestyle="dashed")
# ax[3].axvline(lpeak_mu0p25_h0p75[slice_ind] * (2*pi*Ld), color="green", linestyle="dashed")
# ax[1].axvline(lpeak_mu0p25_h0p25[slice_ind] * (2*pi*Ld), color="green", linestyle="dashed")


# ax[5].plot(kxp, urms_mu0p25_h1p25 .* (3600*24*31) .* kxp ./(2*pi*Ld))
# ax[3].plot(kxp, urms_mu0p25_h0p75 .* (3600*24*31) .* kxp ./(2*pi*Ld))

for axn in ax
    axn.set_xlim(0,2)
    axn.set_ylim(0, 0.225 )

    axn.tick_params(axis="both", labelsize=lsize)
end

## labeling
ax[1].set_ylabel(L"\tau^{-1} \ \ [\mathrm{month}^{-1}]", fontsize=fsize)
ax[2].set_ylabel(L"\tau^{-1} \ \ [\mathrm{month}^{-1}]", fontsize=fsize)


ax[2].set_xlabel(L"k_{x} \, \lambda", fontsize=fsize)
ax[4].set_xlabel(L"k_{x} \, \lambda", fontsize=fsize)
ax[6].set_xlabel(L"k_{x} \, \lambda", fontsize=fsize)

ax[1].text(-0.35, 1.1, L"\mathrm{pos.}", transform=ax[1].transAxes, fontsize=fsize,
            ha="left", va="top")
ax[1].text(-0.375, 1.0, L"\mathrm{slopes}", transform=ax[1].transAxes, fontsize=fsize,
            ha="left", va="top")

ax[2].text(-0.35, 1.1, L"\mathrm{neg.}", transform=ax[2].transAxes, fontsize=fsize,
            ha="left", va="top")
ax[2].text(-0.375, 1.0, L"\mathrm{slopes}", transform=ax[2].transAxes, fontsize=fsize,
            ha="left", va="top")

ax[1].set_title(L"| \, \beta^{*}_\mathrm{t} \, | = 0.25", fontsize=fsize, pad=20)
ax[3].set_title(L"| \, \beta^{*}_\mathrm{t} \, | = 0.75", fontsize=fsize, pad=20)
ax[5].set_title(L"| \, \beta^{*}_\mathrm{t} \, | = 1.25", fontsize=fsize, pad=20)

####
for (i,t) in enumerate([L"(a)", L"(d)", L"(b)", L"(e)", L"(c)", L"(f)"])
    ax[i].text(0.05, 0.95, t, transform=ax[i].transAxes, fontsize=fsize,
            ha="left", va="top", bbox=Dict(
        "facecolor"=>"white",  # Background color
        "alpha"=>0.8,          # Transparency (0=transparent, 1=opaque)
        "edgecolor"=>"white",  # Border color (set to match facecolor to avoid a visible edge)
        "pad"=>5.0             # Padding around the text
    ))
end


savefig("./JPO_rev2_figs/timescale_kx.png",bbox_inches="tight")


