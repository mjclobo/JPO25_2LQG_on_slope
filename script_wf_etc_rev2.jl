
# loading modules
using PyPlot, Printf, LinearAlgebra, FFTW, FileIO, JLD2

using CUDA

using Setfield

# include("../../LinStab/mjcl_stab.jl")
# using .LinStab

using FourierFlows: CPU, TwoDGrid, plan_flows_rfft

using GeophysicalFlows

using Glob

using Statistics

using Parameters

using StaticArrays, KernelAbstractions

using LoopVectorization, Base.Threads

using Dates

run_file_dir = "/scratch/cimes/ml1994/QG/julia/QG_topo/"
include(run_file_dir * "mjcl_gfjl_fcns.jl")




########################################################
matplotlib[:rc]("text", usetex= "true")
matplotlib[:rc]("text.latex", preamble= "\\usepackage{amsmath,amsthm}")

using PlotUtils




dev = GeophysicalFlows.CPU()

Ny = Nx = n = 1024                # 2D resolution = n²


###################################################################################
## Geometry and background fields
###################################################################################

Nz = nlayers = 2                 # number of layers
f0 = f₀= 8.3e-5            # Coriolis param [s^-1]
g = 9.81                    # gravity
H = [2000., 2000.]     # the rest depths of each layer;

Us = [0.01, 0.017634058790440166, 0.024]
rhos = 1025.75 - 0.5775

kt = 0.

rho0 = 1025.        # Boussinesq reference density

# always zero for now, model can't even take this in as a parameter...
V = zeros(nlayers)


# running model

rho = ρ = [0., 1025.75]
rho[1] = ρ[1] = rhos[1]

strat_str = "uni_strat"
shear_str = "uni_shear"

# change U here
U = [0.,0.]
U[1] = Us[1]

gprime = (g/rho0)*(ρ[2] - ρ[1])

Ld = sqrt(gprime * sum(H)) / 2 / f0

L = Lx = Ly = 25*2*pi*Ld   # domain size [m]

###################################################################################
## Planetary PV gradient
###################################################################################
betas = βs = collect(range(-1.5,1.5,13)) * (U[1]/2) * Ld^-2
β = 0. # 1e-11              # the y-gradient of planetary PV; beta = β = 0. * (U[1]/2) * Ld^-2

###################################################################################
## Topography
###################################################################################

# set topography w/ current options:
topo_type = "y_slope"

# Defining a bulk slope parameter...should probably change this!

# setting slope(s)
S32 = f0 * U[1] / gprime

dyetab_over_S32 = collect(range(-1.5,1.5,13))
h0s = dyetab_over_S32 .* S32
# h0s = h0s[7:end]

# tuple of (bottom slope, h_rms)
h0 = (h0s[1], 0.)    #

# central wavenumber for rough topo
kt = 0.

# defining GFJL topo params
interim_params = mod_params(topo_type=topo_type, h0=h0, kt=kt)

topographic_pv_gradient, eta = define_topo(interim_params)

###################################################################################
# Dissipation
###################################################################################

# biharmonic viscosity
nν = 4
νstar = 10^-11 # 10^-13
ν = νstar * ((U[1]-U[end])/2) * (Lx/2/pi)^7

dyn_nu = false

# Linear bottom drag
μstars = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
mus = @. μstars * ((U[1]-U[end]) / 2) / Ld

# Quadratic bottom drag
κstars = [0.]
kappas = [0.] # @. κstars / Ld

###################################################################################
## Time stepping and sample rate
###################################################################################

if ν==0.
    stepper = "FilteredETDRK4"
    # af = 0.
    # global filt_order=4
    # global innerK = 2/3
else
    stepper = "ETDRK4"
    # af = 1/3
    # global filt_order=4
    # innerK = 2/3
end

# should I set dt via CFL condition instead?
dt = 600.

###################################################################################
## Details of saving data
###################################################################################
global diags = diag_bools(two_layer_kspace_modal_nrg_budget_bool=true);

# where to save model output; frequency of output
data_dir = "/scratch/cimes/ml1994/QG/julia/data/sloping_LD_two_layer_data/"

alt_data_dir = "/scratch/cimes/ml1994/QG/julia/data/JFM_May2025_data/"

f_s = 5. * (Ld / ((U[1]-U[end])/2)) #  N eddy turnover periods

# sampling period in time steps (set above)
nsubs = round(Int64, f_s/dt)

###################################################################################
## Model run parameters
###################################################################################

# steady-state  parameters
ss_yr_max = ceil(100 * (Ld / ((U[1]-U[end])/2)) / 3600 / 24 / 365.25)  # number of eddy periods converted to years for model run
yr_increment = 1.0 # how often to write to a save file

restart_bool = false

pre_buoy_restart_file = true
data_dir_pre_buoy = "/scratch/cimes/ml1994/QG/julia/data/two_layer_drag_study/"
data_dir_2L = "/scratch/cimes/ml1994/QG/julia/data/sloping_2L_hi_output_WFS/"

###################################################################################
## Define params struct
###################################################################################
global model_params = mod_params(
data_dir = data_dir,
Nz = Nz, Nx = Nx, Ny = Ny, Lx = Lx, Ly = Ly, Ld = Ld,
H = H,
rho0 = rho0, rho = rho, strat_str = strat_str,
shear_str = shear_str, U = U,
μ = mus[1], κ = kappas[1], nν = nν, ν = ν, dyn_nu = dyn_nu,
eta = eta, topographic_pv_gradient = topographic_pv_gradient, topo_type = topo_type, h0 = h0,
f0=f0, β = β,
dt = dt,
stepper = stepper,
dev = dev,
restart_bool = restart_bool,
pre_buoy_restart_file = pre_buoy_restart_file,
data_dir_pre_buoy = data_dir_pre_buoy,
ss_yr_max = ss_yr_max,
yr_increment = yr_increment,
nsubs = nsubs)

###################################################################################
## Initialize model
###################################################################################

prob, prob_filt = initialize_model(model_params);

# prob = set_initial_conditions(prob, prob_filt, model_params)

sol, clock, params, vars, grid = prob.sol, prob.clock, prob.params, prob.vars, prob.grid;


dev = grid.device
T = eltype(grid)
A = device_array(dev)

rfftplan = plan_flows_rfft(A{T, 3}(undef, grid.nx, grid.ny, 1), [1, 2]; flags=FFTW.MEASURE);

function redef_mu_kappa(model_params, mu, kappa)
    @unpack_mod_params model_params

    mp_out = mod_params(
    data_dir = data_dir,
    Nz = Nz, Nx = Nx, Ny = Ny, Lx = Lx, Ly = Ly, Ld = Ld,
    H = H,
    rho0 = rho0, rho = rho, strat_str = strat_str,
    shear_str = shear_str, U = U,
    μ = mu , κ = kappa , nν = nν, ν = ν, dyn_nu=dyn_nu,
    eta = eta, topographic_pv_gradient = topographic_pv_gradient, topo_type = topo_type, h0 = h0,
    β = β,
    dt = dt,
    stepper = stepper,
    dev = dev,
    restart_bool = restart_bool,
    ss_yr_max = ss_yr_max,
    yr_increment = yr_increment,
    nsubs = nsubs);

    return mp_out
end



using PlotUtils

cm = cgrad(:coolwarm);
clrs = [cm[i] for i in range(0,1,100)]

hc_range = collect(range(h0s[1],h0s[end],100))

function h_color(h_in,hc_range,colors)
    c_ind = argmin(abs.(h_in .- (hc_range)))

    color_out = [red(clrs[c_ind]),green(clrs[c_ind]),blue(clrs[c_ind])]

    if h_in==0
        color_out = [0.,0.,0.]
    end
    return color_out
end

####################################################################
##
####################################################################

function plot_info_box(ax, var, x, y, plus_mode, fsize)
    textstr = var * "\n" * L"+ \rightarrow" * plus_mode

    # these are matplotlib.patch.Patch properties
    props = Dict("boxstyle"=>"round", "facecolor"=>"wheat", "alpha"=>0.5)

    # place a text box in upper left in axes coords
    ax.text(x, y, textstr, transform=ax.transAxes, fontsize=fsize,
            ha="left", va="top", bbox=props)

    return nothing
end


####################################################################
##
####################################################################


# tab20_range = collect(range(0,1,20))

function tab20_color(c_ind)

    cm2 = cgrad(:tab20);
    clrs2 = [cm2[i] for i in range(0,1,20)]

    color_out = [red(clrs2[c_ind]),green(clrs2[c_ind]),blue(clrs2[c_ind])]

    return color_out
end


function redef_mu_kappa_topoPV_h0(model_params, mu, kappa, topo_PV, h0_new)
    @unpack_mod_params model_params

    mp_out = mod_params(
    data_dir = data_dir,
    Nz = Nz, Nx = Nx, Ny = Ny, Lx = Lx, Ly = Ly, Ld = Ld,
    H = H,
    rho0 = rho0, rho = rho, strat_str = strat_str,
    rhotop = rhotop, rhobottom = rhobottom, rhoscaledepth = rhoscaledepth,
    shear_str = shear_str, U = U,
    Utop = Utop, Ubottom = Ubottom, Uscaledepth = Uscaledepth,
    μ = mu , κ = kappa , nν = nν, ν = ν, dyn_nu=dyn_nu, dyn_nu_coeff=dyn_nu_coeff,
    eta = eta, topographic_pv_gradient = topo_PV, topo_type = topo_type, h0 = h0_new, kt=kt,
    f0 = f0, β = β,
    dt = dt,
    stepper = stepper,
    dev = dev,
    restart_bool = restart_bool,
    restart_yr = restart_yr,
    pre_buoy_restart_file = pre_buoy_restart_file,
    data_dir_pre_buoy = data_dir_pre_buoy,
    pre_multilayer_restart_file = pre_multilayer_restart_file,
    pre_multilayer_dir = pre_multilayer_dir,
    ss_yr_max = ss_yr_max,
    yr_increment = yr_increment,
    nsubs = nsubs);

    return mp_out
end


# wavenumber-frequency spectra
# νstars = [10^-11, 10^-11, 10^-11, 10^-11, 10^-11, 10^-11, 10^-11, 10^-11, 10^-11, 10^-11, 10^-11, 10^-11, 10^-11]

νstars = [10^-11, 10^-11, 10^-11, 10^-11, 10^-11, 10^-11, 10^-11, 10^-13, 10^-13, 10^-13, 10^-13, 10^-13, 10^-13]
nus = νstars .* ((U[1]-U[end])/2) * (Lx/2/pi)^7
ν = nus[1]

i=10  # bottom slope index
j=1  # quad drag index
k=2  # linear drag index


h0 = (h0s[i], 0.0)
kappa_loc = kappas[j]
mu_loc = mus[k]

global model_params = redef_mu_kappa_topoPV_h0_nu(model_params, mu_loc, kappa_loc, (0., 0.), h0, nus[i])
global topo_PV, eta = define_topo(model_params)
global model_params = redef_mu_kappa_topoPV_h0_nu(model_params, mu_loc, kappa_loc, topo_PV, h0, nus[i])

global model_params = redef_mu_kappa_beta(model_params, mu_loc, kappa_loc, 0.)

# filename = "/layer2_Htot_4000.0_L_2pi4.449E+06_y_slope_h0_-2.253E-04_beta0.000E+00_uni_shear_uni_strat_mu1.765E-07_kappa0.000E+00_nu0.000E+00_res1024_yr62.95.jld"

# added 2pi added tot
# data_dir_hi_budget = "/scratch/cimes/ml1994/QG/julia/data/sloping_2L_2D_budget_hi_output/"
# a = load(data_dir_hi_budget*jld_name(model_params,84.95))

data_dir_psi_ot = "/scratch/cimes/ml1994/QG/julia/data/JPO_2LQG_on_slope_data/mu0p25_hov/"

a = load(data_dir_psi_ot*jld_name(model_params,89.5))

# a = load(data_dir*filename)




dd = load("./mom_heat_flux_m0p25_h1p25.jld")

v1zeta1 = dd["mom_heat_flux"]["v1zeta1"] .* ((1024*529)^2);
v1tau = -2*dd["mom_heat_flux"]["v1tau"] .* ((1024*529)^2);
c_vec = dd["mom_heat_flux"]["c_vec"]
k_vec = dd["mom_heat_flux"]["k_vec"]
ω_vec = dd["mom_heat_flux"]["ω_vec"]
ubar = dd["mom_heat_flux"]["ubar"];


ub2 = load("./ubar2_m0p25_hm1p25.jld")
ubar2 = ub2["ubar2_jld"]["ubar2"];

dd_l2 = load("./layer2_mom_heat_flux_m0p25_h1p25.jld")
v2zeta2 = dd_l2["l2_mom_heat_flux"]["v1zeta1"] .* ((1024*529)^2)
v2tau = -2*dd_l2["l2_mom_heat_flux"]["v1tau"] .* ((1024*529)^2);


using PyCall

rs = load("./psi1_Hov_"*"m0p25_h1p25"*".jld")

dt_nd = 864000 * ((0.01/2) / Ld)

t = 500 .+ collect(range(1, size(ubar)[2])) .* dt_nd
# fig, ax = plt.subplots(1,3, figsize=(6, 12))

fsize = 18
lsize = 12

fig = plt.figure(figsize=(10, 11))

fig.tight_layout(pad=1.5)

# Define the GridSpec: 3 rows and 3 columns (3 columns for the bottom panels)
gs = matplotlib.gridspec.GridSpec(3, 3, figure=fig, wspace=0.3, hspace=0.45)

# Add the top subplot, spanning all 3 columns in the first row
# ax1 = fig.add_subplot(gs[1, 1:end])
ax1 = fig.add_subplot(py"$(gs)[0, :]")

# Add the two bottom subplots in the second row
ax2 = fig.add_subplot(py"$(gs)[1, 0]")
ax3 = fig.add_subplot(py"$(gs)[1, 1]")
ax4 = fig.add_subplot(py"$(gs)[1, 2]")
ax5 = fig.add_subplot(py"$(gs)[2, 0]")
ax6 = fig.add_subplot(py"$(gs)[2, 1]")
ax7 = fig.add_subplot(py"$(gs)[2, 2]")

ax = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]

################################################################
y = collect(range(0,25*2*pi*Ld, Ny))
# ax[1].

ym = y ./ (2 * pi * Ld)
yp = y[256:2:512] ./ (2*pi*Ld)
# pc1 = ax[1].pcolormesh(fftshift(ω_vec ./ (2*pi)) ./ (2*pi*U[1]/Ld), y, fftshift(v1zeta1)')



################################################################
my_gradient = matplotlib.colors.LinearSegmentedColormap.from_list("my_gradient", (
                 (0.000, (26/255, 148.0/255.0, 49.0/255)),
                 # (0.250, (0.000, 0.145, 0.702)),
                 (0.500, (1.000, 1.000, 1.000)),
                 # (0.750, (0.780, 0.012, 0.051)),
                 (1.0, (15/255, 82/255, 186/255.0))))

my_gradient.set_bad("black")

vlim1 = maximum(abs.(ubar)) / 0.01
pc1 = ax1.pcolormesh(t, ym, ubar./ 0.01, vmin=-vlim1, vmax=vlim1, cmap=my_gradient)

fig.colorbar(pc1, ax=ax1)



################################################################


ax2.plot(mean(ubar .+ 0.01, dims=2) ./ 0.01, ym, "k--")
ax2.plot(mean(ubar2, dims=2) ./ 0.01, ym, "g--")

vlim2 = maximum(abs.(v1zeta1))
vlim3 = maximum(abs.((f0^2/(gprime * H[1])) * v1tau))

# vlim2 = vlim3 = maximum([vlim2, vlim3])

pc2 = ax2.pcolormesh(-c_vec ./ 0.01 , yp, v1zeta1', vmin=-vlim2, vmax=vlim2, cmap=PyPlot.cm.bwr)

################################################################

ax3.plot(mean(ubar .+ 0.01, dims=2) ./ 0.01, ym, "k--")
ax3.plot(mean(ubar2, dims=2) ./ 0.01, ym, "g--")



# pc1 = ax[1].pcolormesh(fftshift(ω_vec ./ (2*pi)) ./ (2*pi*U[1]/Ld), y, fftshift(v1tau)')
pc3 = ax3.pcolormesh(-c_vec ./ 0.01, yp, ((f0^2/(gprime * H[1])) * v1tau)', vmin=-vlim3, vmax=vlim3, cmap=PyPlot.cm.bwr)


################################################################
ax4.axvline(0.0, color="black", linewidth=0.5)

ax4.plot(sum(v1zeta1, dims=1)', yp, label=L"\overline{v_{1}^{\prime} \, \zeta_{1}^{\prime}}", color=(27/255, 82.0/255.0, 151.0/255))
ax4.plot(sum(((f0^2/(gprime * H[1])) * v1tau), dims=1)', yp, label = L"\frac{2 f_{0}}{H} \overline{v_{1}^{\prime} \, \eta_{3/2}^{\prime}}", color=(210/255, 46.0/255.0, 41.0/255))
ax4.plot(-(sum(((f0^2/(gprime * H[1])) * v1tau), dims=1) .+ sum(v1zeta1, dims=1))', yp, label=L"f_0 \, \overline{v}_{1}^{\dagger}", color=(27/255, 210/255.0, 60.0/255))

# ax4.plot(mean(((f0^2/(gprime * H[1])) * v1tau), dims=1)', yp, label = L"-\overline{v_{1} \, \eta_{3/2}}", color=(210/255, 46.0/255.0, 41.0/255))

# ax4.plot([NaN], [NaN], "k--", label=L"(\overline{u}_{1} + U_{1}) / U_{1}")


# ax4t = ax4.twiny()
# ax4t.plot(mean(ubar .+ 0.01, dims=2) ./ 0.01, ym, "k--", label=L"\overline{u_{1}} / U_{1}")

ax4.legend(fontsize=lsize, loc="lower right", bbox_to_anchor=(1.7,0.0))

# ax4.set_xlabel(L"PV flux terms", fontsize=fsize)
# ax4t.set_xlabel(L"\overline{u}_{1} / U_{1}", fontsize=fsize)

# ax4t.set_xticklabels([])
# ax4t.set_xlim(-1.5, 1.5)
# ax4.set_xlim(-6e-10, 6e-10)
################################################################


################################################################


ax5.plot(mean(ubar .+ 0.01, dims=2) ./ 0.01, ym, "k--", label=L"(\overline{u}_{1} + U_{1}) / U_{1}")
ax5.plot(mean(ubar2, dims=2) ./ 0.01, ym, "g--", label=L"\overline{u}_{2} / U_{1}")

vlim5 = maximum(abs.(v2zeta2))
vlim6 = maximum(abs.((f0^2/(gprime * H[1])) * v2tau))

# vlim5 = vlim6 = maximum([vlim5, vlim6])

pc5 = ax5.pcolormesh(-c_vec ./ 0.01 , yp, v2zeta2', vmin=-vlim5, vmax=vlim5, cmap=PyPlot.cm.bwr)

ax5.legend(loc="lower left", fontsize=lsize)
################################################################

ax6.plot(mean(ubar .+ 0.01, dims=2) ./ 0.01, ym, "k--")
ax6.plot(mean(ubar2, dims=2) ./ 0.01, ym, "g--")



# pc1 = ax[1].pcolormesh(fftshift(ω_vec ./ (2*pi)) ./ (2*pi*U[1]/Ld), y, fftshift(v1tau)')
pc6 = ax6.pcolormesh(-c_vec ./ 0.01, yp, -((f0^2/(gprime * H[1])) * v2tau)', vmin=-vlim6, vmax=vlim6, cmap=PyPlot.cm.bwr)


################################################################
ax7.axvline(0.0, color="black", linewidth=0.5)

kap_u2 = mus[2] * mean(ubar2, dims=2)

ax7.plot(sum(v2zeta2, dims=1)', yp, label=L"\overline{v_{2}^{\prime} \, \zeta_{2}^{\prime}}", color=(27/255, 82.0/255.0, 151.0/255))
ax7.plot(-sum(((f0^2/(gprime * H[1])) * v2tau), dims=1)', yp, label = L"- \frac{2 f_{0}}{H} \overline{v_{2}^{\prime} \, \eta_{3/2}^{\prime}}", color=(210/255, 46.0/255.0, 41.0/255))
ax7.plot(- kap_u2[256:2:512], yp, label=L"- \kappa \, \overline{u}_{2}", color=(227/255, 114/255.0, 34.0/255))
ax7.plot(-(-sum(((f0^2/(gprime * H[1])) * v2tau), dims=1) .+ sum(v2zeta2, dims=1) .- kap_u2[256:2:512]')', yp, label=L"f_0 \, \overline{v}_{2}^{\dagger}", color=(27/255, 210/255.0, 60.0/255))

# ax4.plot(mean(((f0^2/(gprime * H[1])) * v1tau), dims=1)', yp, label = L"-\overline{v_{1} \, \eta_{3/2}}", color=(210/255, 46.0/255.0, 41.0/255))

# ax7.plot([NaN], [NaN], "g--", label=L"\overline{u}_{2}")


# ax7t = ax7.twiny()
# ax7t.plot(mean(ubar2, dims=2) ./ 0.01, ym, "g--", label=L"\overline{u_{2}} / U_{1}")

ax7.legend(fontsize=lsize, loc="lower right", bbox_to_anchor=(1.75,0.0))

# ax7.set_xlabel(L"PV flux terms", fontsize=fsize)
# ax7t.set_xlabel(L"\overline{u}_{1} / U_{1}", fontsize=fsize)

# ax7t.set_xticklabels([])
# ax7t.set_xlim(-1.5, 1.5)
# ax7.set_xlim(-6e-10, 6e-10)
################################################################

for axn in ax[2:end]
    axn.set_ylim(yp[1], yp[end])
end


for axn in [ax[2], ax[3], ax[5], ax[6]]
    axn.set_xlim(-1.0, 2.25)
end

for axn in [ax4, ax7]
    axn.set_xlim(-6e-10, 6e-10)
end

ax1.set_ylabel(L"y / 2 \pi \lambda ", fontsize=fsize)
ax2.set_ylabel(L"y / 2 \pi \lambda ", fontsize=fsize)
ax5.set_ylabel(L"y / 2 \pi \lambda ", fontsize=fsize)


panel_labels = [L"(a)", L"(b)", L"(c)", L"(d)",  L"(e)", L"(f)", L"(g)"]
for (i,t) in enumerate(panel_labels)
    ax[i].text(0.05, 0.95, t, transform=ax[i].transAxes, fontsize=fsize,
            ha="left", va="top", bbox=Dict(
        "facecolor"=>"white",  # Background color
        "alpha"=>0.8,          # Transparency (0=transparent, 1=opaque)
        "edgecolor"=>"white",  # Border color (set to match facecolor to avoid a visible edge)
        "pad"=>5.0             # Padding around the text
    ))
end

for axn in [ax5, ax6]
    axn.set_xlabel(L"c / U_{1} ", fontsize=fsize)
end

ax[1].set_xlabel(L"t \ U \ / \ \lambda", fontsize=fsize)

ax[1].set_title(L"\overline{u}_{1} (y,t) / U_{1} ", fontsize=fsize, pad=10)
ax[2].set_title(L"\overline{v_{1}^{\prime} \zeta_{1}^{\prime}} ", fontsize=fsize, pad=10)
ax[3].set_title(L"\frac{2 \, f_{0}}{H} \overline{v_{1}^{\prime} \, \eta_{3/2}^{\prime}} ", fontsize=fsize, pad=10)

ax[4].set_title(L"\mathrm{TEM \ terms} \ \ \  [\mathrm{m} \ \mathrm{s}^{-2}]", fontsize=fsize, pad=10)


ax[5].set_title(L"\overline{v_{2}^{\prime} \zeta_{2}^{\prime}} ", fontsize=fsize, pad=10)
ax[6].set_title(L"- \frac{2 \, f_{0}}{H}  \, \overline{v_{2}^{\prime} \, \eta_{3/2}^{\prime}} ", fontsize=fsize, pad=10)

ax[7].set_title(L"\mathrm{TEM \ terms} \ \ \  [\mathrm{m} \ \mathrm{s}^{-2}]", fontsize=fsize, pad=10)


savefig("./JPO_rev2_figs/mom_heat_flux_m0p25.png",bbox_inches="tight")






















