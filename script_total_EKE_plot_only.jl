# wavenumber-frequency spectra
# νstars = [10^-11, 10^-11, 10^-11, 10^-11, 10^-11, 10^-11, 10^-11 , 10^-11, 10^-11, 10^-11, 10^-11, 10^-11, 10^-11]
νstars = [10^-11, 10^-11, 10^-11, 10^-11, 10^-11, 10^-11, 10^-11 , 10^-13, 10^-13, 10^-13, 10^-13, 10^-13, 10^-13]

nus = νstars .* ((U[1]-U[end])/2) * (Lx/2/pi)^7
ν = nus[1]


function plot_precomputed_EKE(ax, i, k, data_dir)  # i: bottom slope index, k: linear drag index
    j=1  # quad drag index

    h0 = (h0s[i], 0.0)
    kappa_loc = kappas[j]
    mu_loc = mus[k]

    global model_params = redef_mu_kappa_topoPV_h0_nu(model_params, mu_loc, kappa_loc, (0., 0.), h0, nus[i])
    global topo_PV, eta = define_topo(model_params)
    global model_params = redef_mu_kappa_topoPV_h0_nu(model_params, mu_loc, kappa_loc, topo_PV, h0, nus[i])

    global model_params = redef_mu_kappa_beta(model_params, mu_loc, kappa_loc, 0.)

    a = load(data_dir*jld_name(model_params,89.0))

    dx = Lx/Nx
    dy = Ly/Ny

    kxp = collect(fftshift(fftfreq(Nx, 1/dx))) .* (2*pi*Ld)
    kyp = reshape(collect(fftshift(fftfreq(Ny, 1/dy))) .* (2*pi*Ld), 1, Ny)[:]

    EKE_2D = fftshift(reshape(sum(a["jld_data"]["kspace_modal_nrg_spectrum"], dims=3), Nx, Ny))

    vlim = maximum(abs.(EKE_2D))

    # pc = ax.pcolormesh(kxp, kyp, fftshift(a["jld_data"]["kspace_modal_nrg_spectrum"][:,:,1])', cmap=PyPlot.cm.Greys, norm=matplotlib.colors.LogNorm(vmin=1e-3))
    pc = ax.pcolormesh(kxp, kyp, EKE_2D', cmap=PyPlot.cm.Greys, norm=matplotlib.colors.LogNorm(vmin=1e-2*vlim, vmax=vlim))

    return pc, vlim
end

function def_levs(gr)
    m = maximum(gr)
    return [0.3*m, 0.6*m, 0.9*m]
end
data_dir_EKE_spec = "/scratch/cimes/ml1994/QG/julia/data/JPO_2LQG_on_slope_data/mu0p25_kspace/"

fsize = 26  # 20
lsize = 20  # 16

fig, ax = plt.subplots(2, 3, figsize=(18, 10))

pc_hm1p25, vlim_hm1p25 = plot_precomputed_EKE(ax[6], 2, 2, data_dir_EKE_spec)
pc_hm0p75, vlim_hm0p75 = plot_precomputed_EKE(ax[4], 4, 2, data_dir_EKE_spec)
pc_hm0p25, vlim_hm0p25 = plot_precomputed_EKE(ax[2], 6, 2, data_dir_EKE_spec)
pc_h0p25, vlim_h0p25  = plot_precomputed_EKE(ax[5], 12, 2, data_dir_EKE_spec)
pc_h0p75, vlim_h0p75  = plot_precomputed_EKE(ax[3], 10, 2, data_dir_EKE_spec)
pc_h1p25, vlim_h1p25  = plot_precomputed_EKE(ax[1], 8, 2, data_dir_EKE_spec)

ax[6].contour(kxp, kyp[:], growth[2, :, :]', colors=(43.0/255.0, 208.0/255.0, 0.0), levels=def_levs(growth[2, :, :]), alpha=0.75)
ax[4].contour(kxp, kyp[:], growth[4, :, :]', colors=(43.0/255.0, 208.0/255.0, 0.0), levels=def_levs(growth[4, :, :]), alpha=0.75)
ax[2].contour(kxp, kyp[:], growth[6, :, :]', colors=(43.0/255.0, 208.0/255.0, 0.0), levels=def_levs(growth[6, :, :]), alpha=0.75)

ax[5].contour(kxp, kyp[:], growth[12, :, :]', colors=(43.0/255.0, 208.0/255.0, 0.0), levels=def_levs(growth[12, :, :]), alpha=0.75)
ax[3].contour(kxp, kyp[:], growth[10, :, :]', colors=(43.0/255.0, 208.0/255.0, 0.0), levels=def_levs(growth[10, :, :]), alpha=0.75)
ax[1].contour(kxp, kyp[:], growth[8, :, :]', colors=(43.0/255.0, 208.0/255.0, 0.0), levels=def_levs(growth[8, :, :]), alpha=0.75)




for axn in ax
    axn.set_xlim(-1.25,1.25)
    axn.set_ylim(-1.25,1.25)

    axn.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])

    axn.tick_params(axis="both", labelsize=lsize)
end

## labeling
ax[1].set_ylabel(L"k_{y} \, \lambda", fontsize=fsize)
ax[2].set_ylabel(L"k_{y} \, \lambda", fontsize=fsize)

ax[2].set_xlabel(L"k_{x} \, \lambda", fontsize=fsize)
ax[4].set_xlabel(L"k_{x} \, \lambda", fontsize=fsize)
ax[6].set_xlabel(L"k_{x} \, \lambda", fontsize=fsize)

ax[1].text(-0.35, 0.9, L"\mathrm{pos.}", transform=ax[1].transAxes, fontsize=fsize,
            ha="left", va="top")
ax[1].text(-0.375, 0.8, L"\mathrm{slopes}", transform=ax[1].transAxes, fontsize=fsize,
            ha="left", va="top")

ax[2].text(-0.35, 0.9, L"\mathrm{neg.}", transform=ax[2].transAxes, fontsize=fsize,
            ha="left", va="top")
ax[2].text(-0.375, 0.8, L"\mathrm{slopes}", transform=ax[2].transAxes, fontsize=fsize,
            ha="left", va="top")

ax[1].set_title(L"| \, \beta^{*}_\mathrm{t} \, | = 0.25", fontsize=fsize, pad=20)
ax[3].set_title(L"| \, \beta^{*}_\mathrm{t} \, | = 0.75", fontsize=fsize, pad=20)
ax[5].set_title(L"| \, \beta^{*}_\mathrm{t} \, | = 1.25", fontsize=fsize, pad=20)

####
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.025, 0.7])
cbar = fig.colorbar(pc_h0p25, ticks=[1e-2*vlim_h0p25, 1e-1*vlim_h0p25, vlim_h0p25], cax=cbar_ax)
cbar.ax.tick_params(labelsize=lsize)
cbar.set_ticklabels([L"E_\mathrm{max} \times 10^{-2}", L"E_\mathrm{max} \times 10^{-1}", L"E_\mathrm{max}"])

for (i,t) in enumerate([L"(a)", L"(d)", L"(b)", L"(e)", L"(c)", L"(f)"])
    ax[i].text(0.05, 0.95, t, transform=ax[i].transAxes, fontsize=fsize,
            ha="left", va="top")
end


savefig("./JPO_rev2_figs/total_EKE_2D_rev2.png",bbox_inches="tight")




