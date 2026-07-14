
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
# νstars = [10^-11, 10^-11, 10^-11, 10^-11, 10^-11, 10^-11, 10^-11 , 10^-11, 10^-11, 10^-11, 10^-11, 10^-11, 10^-11]
νstars = [10^-11, 10^-11, 10^-11, 10^-11, 10^-11, 10^-11, 10^-11 , 10^-13, 10^-13, 10^-13, 10^-13, 10^-13, 10^-13]

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


# energies are: (1) BTEKE, (2) BCEKE, (3) EAPE; (4) CBC; (5) Tflat, (6) Ttopo; , (7) DBT, (8) DBC;
#               (9) NLBC2BT, (10) NLBTEKE, (11) NLBT2BC; (12) NLBVEKE (13) NLBCEAPE (14) resid

# layer2_Htot_4000.0_L_2pi4.449E+06_y_slope_h0_0.000E+00_beta3.116E-12_uni_shear_uni_strat_mu8.826E-08_kappa0.000E+00_nu0.000E+00_res1024_yr57.7.jld

# layer2_H_4000.0_L_4.449E+06_y_slope_h0_3.754E-05_beta0.000E+00_uni_shear1.000E+000.000E+001.000E+03
# _uni_strat1.025E+031.026E+031.000E+03__mu4.413E-08_kappa0.000E+00_nu4.464E+27_res1024_yr89.5.jld


# readdir(data_dir_psi_ot)


using Roots

function dispersion_relation_full_bottom(beta_topo, U1, mode_string, mu_loc; kx=0, ky=0, frequency=false)

    # if ky==0

    #     k2 = 1 * (2 * pi / L)^2   # pre-factor of 2 assumes isotropic length scale; prefactor of 1 assumes zonal wavenumber only, l=0

    #     A = k2 * (k2 + Ld^-2)
    #     B = (beta_topo + im * mu_loc * sqrt(k2)) * (k2 + (2*Ld^2)^-1) - U1 * k2 * (k2 + Ld^-2)
    #     C = U1 * k2 * (U1 * (2 * Ld^2)^-1 - beta_topo - im * mu_loc * sqrt(k2))

    #     # a = @. k2 * (k2 + Ld^-2)
    #     # b = @. - U[1] * k2 * (k2 + Ld^-2) + (k2 + 0.5 * Ld^-2) * ((f0/H[2]) * h0 + 1im * μ * sqrt(k2))
    #     # c = @. - U[1] * k2 * ((f0/H[2]) * h0 - 0.5 * Ld^-2 *U[1] + 1im * μ * sqrt(k2))

    # else
    if kx==0

        k2 = ky^2   # pre-factor of 2 assumes isotropic length scale; prefactor of 1 assumes zonal wavenumber only, l=0

        A = k2 * (k2 + Ld^-2)
        # B = (beta_topo) * (k2 + (2*Ld^2)^-1) - U1 * k2 * (k2 + Ld^-2)
        # C = U1 * k2 * (U1 * (2 * Ld^2)^-1 - beta_topo)

        B = (im * mu_loc) * (k2 + (2*Ld^2)^-1)
        C = 0

        kx_to_k2 = 0.
    else
        k2 = kx^2 + ky^2   # pre-factor of 2 assumes isotropic length scale; prefactor of 1 assumes zonal wavenumber only, l=0

        # code used for rev1 to find c
        # A = k2 * (k2 + Ld^-2)
        # B = (beta_topo + im * mu_loc * k2 / kx) * (k2 + (2*Ld^2)^-1) - U1 * k2 * (k2 + Ld^-2)
        # C = U1 * k2 * (U1 * (2 * Ld^2)^-1 - beta_topo - im * mu_loc * k2/kx)

        # kx_to_k2 = (kx/sqrt(k2))

        # this calculates omega directly from my derivation on 7 May 2026 (see tablet);
        # to get c instead of omega we divide B by sqrt(k2) and divide C by k2
        F = 1/(2 * Ld^2)
        A = (k2 + F)^2 - F^2
        B = (k2 + F) * (-kx * U1 * k2 + kx * (beta_topo - U1 * F) + im * mu_loc * k2) + kx * U1 * F^2
        C = - kx * U1 * k2 * (kx * (beta_topo - U1 * F) + im * mu_loc * k2)

        # converts omega to phase speed
        B = B / sqrt(k2)
        C = C / k2

        kx_to_k2 = 1.0 # (kx/sqrt(k2))
    end

    # if mode_string=="BT"
    #     dr = U1/2 - (beta_topo / 2) / k2
    # elseif mode_string=="BC"
    #     dr = U1/2 - (beta_topo / 2) / (k2 + Ld^-2)
    # end

    # this is full dispersion relation for bottom slope PV gradient in lower layer only
    dr_both = [- B / 2 / A + real(sqrt(Complex(B^2 - 4 * A * C))) / 2 / A; - B / 2 / A - real(sqrt(Complex(B^2 - 4 * A * C))) / 2 / A]
    dr_both = dr_both .* kx_to_k2
    if mode_string=="BT"
        indmax = 1 # argmax(abs.(dr_both))
        dr = real(dr_both[indmax])
    elseif mode_string=="BC"
        indmin = 2 #argmin(abs.(dr_both))
        dr = real(dr_both[indmin])
    end

    # if real(B^2 - 4 * A * C) < 0
    #     println(B^2 - 4 * A * C)
    # end

    dr_both_comp = sqrt(k2) .* [- B / 2 / A + sqrt(Complex(B^2 - 4 * A * C)) / 2 / A; - B / 2 / A - sqrt(Complex(B^2 - 4 * A * C)) / 2 / A]
    dr_both_comp = dr_both_comp .* kx_to_k2

    disc = imag.(dr_both_comp)
    if any(abs.(disc) .> 0)
        max_growth = maximum((disc))
    else
        max_growth = 0.
    end

    # if (B^2 - 4 * A * C) > 0
    #     println(sqrt(B^2 - 4 * A * C) / B)
    # end
    # dr = U1/2 - beta_topo * (2*k2 + Ld^-2) / (2*k2 * (k2 + Ld^-2)) + sf * real(sqrt(Complex(U1^2 * k2^2 * (k2^2 - 2 * Ld^-2) + 2 * Ld^-2 * beta_topo^2)) / (2 * k2 * ( k2 + Ld^-2)))


    return dr, max_growth

end

############################################################################

############################################################################

dx = Lx/Nx
dy = Ly/Ny

kxp = collect(fftshift(fftfreq(Nx, 1/dx))) .* (2*pi*Ld)     # this is now radians (initially per meter), normed by Ld
kyp = reshape(collect(fftshift(fftfreq(Ny, 1/dy))) .* (2*pi*Ld), 1, Ny)     # this is now radians (initially per meter), normed by Ld



# ks = collect(range(0.001, 2, Nell)) ./ Ld
# kx_all = collect(range(1/(3*Ld), 1/(0.1*Ld), 200))
kx_all = kxp ./ (2*pi*Ld) #    this is now in cycles per meter
ky_all = kyp[:] ./ (2*pi*Ld) #    this is now in cycles per meter

ells_x = kx_all.^-1 # this is in meters (no radians; not normed by Ld)
ells_y = ky_all.^-1 # this is in meters (no radians; not normed by Ld))
# ells = collect(range(-30 * 2 * pi * Ld, 30 * 2 * pi * Ld, 200))  # kx_all.^-1 #

# Nell=201
# ells = collect(range(0, 100*2*pi*Ld, Nell-1))

# ells = [ells[2:end];Inf]


kvec_str = "iso"

mu_loc = mus[1]

c_BT_full = zeros(length(h0s), length(ells_x), length(ells_y))
c_BC_full = zeros(length(h0s), length(ells_x), length(ells_y))
growth = zeros(length(h0s), length(ells_x), length(ells_y))

for h_ind in range(1, length(h0s))  # [2, 4, 6, 8, 10, 12]  #

    h0 = (h0s[h_ind], 0.0)

    alpha_s = (f0 * h0s[h_ind] / H[2])

    for (i,Lkx) in enumerate(ells_x)
        for (j, Lky) in enumerate(ells_y)
            # if h_ind != 7

            # if kvec_str=="zonal"
            #     kx=2*pi/L0
            #     ky=0
            # elseif kvec_str=="ky_fixed"
            #     kx = 2*pi/L0
            #     ky = 2*pi/Lx*5
            # elseif kvec_str=="k_iso"
                global kx = 2*pi/Lkx  # 1/Lkx #   # this is in radians per meter
                global ky = 2*pi/Lky  # 1/Lky #   # this is in radians per meter
            # end

            # these phase speeds are in m/s (no radians)
            c_BT_full[h_ind, i, j], nul = dispersion_relation_full_bottom(alpha_s, 0.01, "BT", 0.0; kx=kx, ky=ky)
            c_BC_full[h_ind, i, j], nul = dispersion_relation_full_bottom(alpha_s, 0.01, "BC", 0.0; kx=kx, ky=ky)
            nul, growth[h_ind, i, j] = dispersion_relation_full_bottom(alpha_s, 0.01, "BC", mu_loc; kx=kx, ky=ky)

        end
    end
end



nH = 13 # 6

tau_wave = zeros(nH, length(ells_x), length(ells_y))
tau_wave_BC = zeros(nH, length(ells_x), length(ells_y))
tau_wave_BT = zeros(nH, length(ells_x), length(ells_y))

h_ind_bank = collect(range(1, 13))
# h_ind_bank = [2, 4, 6, 8, 10, 12]

# ells_x, ells_y are in meters (no radians; not normed by Ld)
# for (h, h_ind) in enumerate(h_ind_bank)
    # for (i, lx) in enumerate(ells_x)
    #     for (j, ly) in enumerate(ells_y)
    #         l = sqrt(lx^-2 + ly^-2)^-1
    #         # tau_wave[h,i,j] = maximum(abs.([c_BC_full[h_ind, i, j], c_BT_full[h_ind, i, j]])) / l 

for (h, h_ind) in enumerate(h_ind_bank)

    for (i, kx) in enumerate(kxp ./ (2*pi*Ld))  # this correctly removes Ld and radian factors
        for (j, ky) in enumerate(kyp ./ (2*pi*Ld))
            l = sqrt(kx^2 + ky^2)^-1    # these are wavenumbers in inverse length (no radians)
            # with c in m/s (no radians), this is a time scale in Hz (i.e., no radians)
            tau_wave_BT[h,i,j] = abs(c_BT_full[h_ind, i, j]) / l
            tau_wave_BC[h,i,j] = abs(c_BC_full[h_ind, i, j]) / l 
            tau_wave[h,i,j] = maximum(abs.([c_BC_full[h_ind, i, j], c_BT_full[h_ind, i, j]])) / l 
        end
    end
end



function EKE_timescale(i, k, kxp, kyp, data_dir; yr_last=89.0, nus=zeros(13))
    j=1  # quad drag index    
    
    h0 = (h0s[i], 0.0)
    kappa_loc = kappas[j]
    mu_loc = mus[k]
    
    global model_params = redef_mu_kappa_topoPV_h0_nu(model_params, mu_loc, kappa_loc, (0., 0.), h0, nus[i])
    global topo_PV, eta = define_topo(model_params)
    global model_params = redef_mu_kappa_topoPV_h0_nu(model_params, mu_loc, kappa_loc, topo_PV, h0, nus[i])
    
    global model_params = redef_mu_kappa_beta(model_params, mu_loc, kappa_loc, 0.)
    
    a = load(data_dir*jld_name(model_params,yr_last))

    EKE_2D = reshape(sum(a["jld_data"]["kspace_modal_nrg_spectrum"], dims=3), Nx, Ny)
    
    tau_EKE = zeros(length(kxp), length(kyp))

    l_peak_EKE = zeros(length(kyp))

    sdf = Lx * Ly * ((2*pi)^-2)   # this is the spectral density factor; (dk_{x} * dk_{y})^-1
    tau_EKE = 0.5 .* sqrt.(fftshift(EKE_2D ./ (Nx*Ny))) .* sqrt.(fftshift(grid.Ksq .* ((2*pi)^2))) # ./ ((2 * pi)^2))) .* (2*pi)

    return tau_EKE, l_peak_EKE
end


data_dir_EKE_spec = "/scratch/cimes/ml1994/QG/julia/data/JPO_2LQG_on_slope_data/mu0p25_kspace/"

νstars = [10^-11, 10^-11, 10^-11, 10^-11, 10^-11, 10^-11, 10^-11 , 10^-13, 10^-13, 10^-13, 10^-13, 10^-13, 10^-13]
nus = νstars .* ((U[1]-U[end])/2) * (Lx/2/pi)^7

tau_EKE_mu0p25_hm1p25, lpeak_mu0p25_hm1p25 = EKE_timescale(2, 2, kxp, kyp, data_dir_EKE_spec; nus=nus);
tau_EKE_mu0p25_hm0p75, lpeak_mu0p25_hm0p75 = EKE_timescale(4, 2, kxp, kyp, data_dir_EKE_spec; nus=nus);
tau_EKE_mu0p25_hm0p25, lpeak_mu0p25_hm0p25 = EKE_timescale(6, 2, kxp, kyp, data_dir_EKE_spec; nus=nus);
tau_EKE_mu0p25_h0p25, lpeak_mu0p25_h0p25  = EKE_timescale(8, 2, kxp, kyp, data_dir_EKE_spec; nus=nus);
tau_EKE_mu0p25_h0p75, lpeak_mu0p25_h0p75  = EKE_timescale(10, 2, kxp, kyp, data_dir_EKE_spec; nus=nus);
tau_EKE_mu0p25_h1p25, lpeak_mu0p25_h1p25  = EKE_timescale(12, 2, kxp, kyp, data_dir_EKE_spec; nus=nus);


tau_EKE_mu0p25_hm1p0, lpeak_mu0p25_hm1p0 = EKE_timescale(3, 2, kxp, kyp, data_dir_EKE_spec; nus=nus);
tau_EKE_mu0p25_hm0p5, lpeak_mu0p25_hm0p5 = EKE_timescale(5, 2, kxp, kyp, data_dir_EKE_spec; nus=nus);
tau_EKE_mu0p25_h0p5, lpeak_mu0p25_h0p5  = EKE_timescale(9, 2, kxp, kyp, data_dir_EKE_spec; nus=nus);
tau_EKE_mu0p25_h1p0, lpeak_mu0p25_h1p0  = EKE_timescale(11, 2, kxp, kyp, data_dir_EKE_spec; nus=nus);


# for mu = 0.125
data_dir_EKE_spec_mu0p125 = "/scratch/cimes/ml1994/QG/julia/data/JPO_2LQG_on_slope_data/mu0p125_kspace/"

νstars = [10^-11, 10^-11, 10^-11, 10^-11, 10^-11, 10^-11, 10^-11 , 10^-13, 10^-13, 10^-13, 10^-13, 10^-13, 10^-13]
nus = νstars .* ((U[1]-U[end])/2) * (Lx/2/pi)^7

tau_EKE_mu0p125_hm1p25, lpeak_mu0p125_hm1p25 = EKE_timescale(2, 1, kxp, kyp, data_dir_EKE_spec_mu0p125; nus=nus);
tau_EKE_mu0p125_hm0p75, lpeak_mu0p125_hm0p75 = EKE_timescale(4, 1, kxp, kyp, data_dir_EKE_spec_mu0p125; nus=nus);
tau_EKE_mu0p125_hm0p25, lpeak_mu0p125_hm0p25 = EKE_timescale(6, 1, kxp, kyp, data_dir_EKE_spec_mu0p125; nus=nus);
tau_EKE_mu0p125_h0p25, lpeak_mu0p125_h0p25  = EKE_timescale(8, 1, kxp, kyp, data_dir_EKE_spec_mu0p125; nus=nus);
tau_EKE_mu0p125_h0p75, lpeak_mu0p125_h0p75  = EKE_timescale(10, 1, kxp, kyp, data_dir_EKE_spec_mu0p125; nus=nus);
tau_EKE_mu0p125_h1p25, lpeak_mu0p125_h1p25  = EKE_timescale(12, 1, kxp, kyp, data_dir_EKE_spec_mu0p125; nus=nus);


tau_EKE_mu0p125_hm1p0, lpeak_mu0p125_hm1p0 = EKE_timescale(3, 1, kxp, kyp, data_dir_EKE_spec_mu0p125; nus=nus);
tau_EKE_mu0p125_hm0p5, lpeak_mu0p125_hm0p5 = EKE_timescale(5, 1, kxp, kyp, data_dir_EKE_spec_mu0p125; nus=nus);
tau_EKE_mu0p125_h0p5, lpeak_mu0p125_h0p5  = EKE_timescale(9, 1, kxp, kyp, data_dir_EKE_spec_mu0p125; nus=nus);
tau_EKE_mu0p125_h1p0, lpeak_mu0p125_h1p0  = EKE_timescale(11, 1, kxp, kyp, data_dir_EKE_spec_mu0p125; nus=nus);


# for mu = 0.125
data_dir_EKE_spec_mu0p5 = "/scratch/cimes/ml1994/QG/julia/data/JPO_2LQG_on_slope_data/mu0p5_kspace/"

νstars = [10^-13, 10^-13, 10^-13, 10^-13, 10^-13, 10^-13, 10^-13 , 10^-13, 10^-13, 10^-13, 10^-13, 10^-13, 10^-13]
nus = νstars .* ((U[1]-U[end])/2) * (Lx/2/pi)^7

tau_EKE_mu0p5_hm1p25, lpeak_mu0p5_hm1p25 = EKE_timescale(2, 3, kxp, kyp, data_dir_EKE_spec_mu0p5; yr_last=179.0, nus=nus);
tau_EKE_mu0p5_hm0p75, lpeak_mu0p5_hm0p75 = EKE_timescale(4, 3, kxp, kyp, data_dir_EKE_spec_mu0p5; yr_last=179.0, nus=nus);
tau_EKE_mu0p5_hm0p25, lpeak_mu0p5_hm0p25 = EKE_timescale(6, 3, kxp, kyp, data_dir_EKE_spec_mu0p5; yr_last=179.0, nus=nus);
tau_EKE_mu0p5_h0p25, lpeak_mu0p5_h0p25  = EKE_timescale(8, 3, kxp, kyp, data_dir_EKE_spec_mu0p5; yr_last=179.0, nus=nus);
tau_EKE_mu0p5_h0p75, lpeak_mu0p5_h0p75  = EKE_timescale(10, 3, kxp, kyp, data_dir_EKE_spec_mu0p5; yr_last=179.0, nus=nus);
tau_EKE_mu0p5_h1p25, lpeak_mu0p5_h1p25  = EKE_timescale(12, 3, kxp, kyp, data_dir_EKE_spec_mu0p5; yr_last=179.0, nus=nus);


tau_EKE_mu0p5_hm1p0, lpeak_mu0p5_hm1p0 = EKE_timescale(3, 3, kxp, kyp, data_dir_EKE_spec_mu0p5; yr_last=179.0, nus=nus);
tau_EKE_mu0p5_hm0p5, lpeak_mu0p5_hm0p5 = EKE_timescale(5, 3, kxp, kyp, data_dir_EKE_spec_mu0p5; yr_last=179.0, nus=nus);
tau_EKE_mu0p5_h0p5, lpeak_mu0p5_h0p5  = EKE_timescale(9, 3, kxp, kyp, data_dir_EKE_spec_mu0p5; yr_last=179.0, nus=nus);
tau_EKE_mu0p5_h1p0, lpeak_mu0p5_h1p0  = EKE_timescale(11, 3, kxp, kyp, data_dir_EKE_spec_mu0p5; yr_last=179.0, nus=nus);

function plot_timescales(ax, tau_wave_BC, tau_wave_BT, tau_EKE, kxp)

    norm = 3600 * 24 * 31
    
    ax.plot(kxp, tau_wave_BC .* norm, "k-")
    ax.plot(kxp, tau_wave_BT .* norm, "k-")

    ax.plot(kxp, tau_EKE .* (norm), "r-")
    
end

fsize = 20
lsize = 16

slice_ind = 513

fig, ax = plt.subplots(2,3, figsize=(16,10))

plot_timescales(ax[6], tau_wave_BC[2,:,slice_ind], tau_wave_BT[2,:,slice_ind], tau_EKE_mu0p25_hm1p25[:,slice_ind], kxp)
plot_timescales(ax[4], tau_wave_BC[4,:,slice_ind], tau_wave_BT[4,:,slice_ind], tau_EKE_mu0p25_hm0p75[:,slice_ind], kxp)
plot_timescales(ax[2], tau_wave_BC[6,:,slice_ind], tau_wave_BT[6,:,slice_ind], tau_EKE_mu0p25_hm0p25[:,slice_ind], kxp)

plot_timescales(ax[5], tau_wave_BC[12,:,slice_ind], tau_wave_BT[12,:,slice_ind], tau_EKE_mu0p25_h1p25[:,slice_ind], kxp)
plot_timescales(ax[3], tau_wave_BC[10,:,slice_ind], tau_wave_BT[10,:,slice_ind], tau_EKE_mu0p25_h0p75[:,slice_ind], kxp)
plot_timescales(ax[1], tau_wave_BC[8,:,slice_ind], tau_wave_BT[8,:,slice_ind], tau_EKE_mu0p25_h0p25[:,slice_ind], kxp)


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

ax[1].set_title(L"| \, \beta^{*}_\mathrm{t} \, | = 0.25", fontsize=fsize)
ax[3].set_title(L"| \, \beta^{*}_\mathrm{t} \, | = 0.75", fontsize=fsize)
ax[5].set_title(L"| \, \beta^{*}_\mathrm{t} \, | = 1.25", fontsize=fsize)

####
for (i,t) in enumerate(["(a)", "(d)", "(b)", "(e)", "(c)", "(f)"])
    ax[i].text(0.05, 0.95, t, transform=ax[i].transAxes, fontsize=fsize,
            ha="left", va="top", bbox=Dict(
        "facecolor"=>"white",  # Background color
        "alpha"=>0.8,          # Transparency (0=transparent, 1=opaque)
        "edgecolor"=>"white",  # Border color (set to match facecolor to avoid a visible edge)
        "pad"=>5.0             # Padding around the text
    ))
end


savefig("./JPO_rev2_figs/timescale_kx.png",bbox_inches="tight")

##;#########################################################################
##;#########################################################################

kx_EKE_max_mu0p25 = zeros(13)
kx_tau_wave_min_mu0p25 = zeros(13)

max_tau_mu0p25 = zeros(13, 50)

slice_ind = 513
norm2 = 3600 * 24 * 31

for i in range(1,13)
    for j in range(1,50)
        max_tau_mu0p25[i,j] = maximum([tau_wave_BC[i,513+j,slice_ind], tau_wave_BT[i,513+j,slice_ind]]) * norm2
    end
    
    kx_tau_wave_min_mu0p25[i] = kxp[512+argmin(max_tau_mu0p25[i,:])]
    # println(argmin(max_tau))
end

# this is the wavenumber of the maximum EKE timescale; this is predicted by the minimum in the wave frequency
kx_EKE_max_mu0p25[2] = kxp[512+argmax(tau_EKE_mu0p25_hm1p25[513:550,slice_ind] .* norm2)] 
kx_EKE_max_mu0p25[3] = kxp[512+argmax(tau_EKE_mu0p25_hm1p0[513:550,slice_ind] .* norm2)]
kx_EKE_max_mu0p25[4] = kxp[512+argmax(tau_EKE_mu0p25_hm0p75[513:550,slice_ind] .* norm2)]
kx_EKE_max_mu0p25[5] = kxp[512+argmax(tau_EKE_mu0p25_hm0p5[513:550,slice_ind] .* norm2)]
kx_EKE_max_mu0p25[6] = kxp[512+argmax(tau_EKE_mu0p25_hm0p25[513:550,slice_ind] .* norm2)]

kx_EKE_max_mu0p25[12] = kxp[512+argmax(tau_EKE_mu0p25_h1p25[513:550,slice_ind] .* norm2)]
kx_EKE_max_mu0p25[11] = kxp[512+argmax(tau_EKE_mu0p25_h1p0[513:550,slice_ind] .* norm2)]
kx_EKE_max_mu0p25[10] = kxp[512+argmax(tau_EKE_mu0p25_h0p75[513:550,slice_ind] .* norm2)]
kx_EKE_max_mu0p25[9] = kxp[512+argmax(tau_EKE_mu0p25_h0p5[513:550,slice_ind] .* norm2)];
kx_EKE_max_mu0p25[8] = kxp[512+argmax(tau_EKE_mu0p25_h0p25[513:550,slice_ind] .* norm2)];

kx_EKE_max_mu0p125 = zeros(13)
kx_tau_wave_min_mu0p125 = zeros(13)

max_tau_mu0p125 = zeros(13, 50)

slice_ind = 513
norm2 = 3600 * 24 * 31

for i in range(1,13)
    max_tau = zeros(50)
    for j in range(1,50)
        max_tau_mu0p125[i,j] = maximum([tau_wave_BC[i,513+j,slice_ind], tau_wave_BT[i,513+j,slice_ind]]) * norm2
    end
    
    kx_tau_wave_min_mu0p125[i] = kxp[512+argmin(max_tau_mu0p125[i,:])]
    # println(argmin(max_tau))
end

kx_EKE_max_mu0p125[2] = kxp[512+argmax(tau_EKE_mu0p125_hm1p25[513:550,slice_ind] .* norm2)]
kx_EKE_max_mu0p125[3] = kxp[512+argmax(tau_EKE_mu0p125_hm1p0[513:550,slice_ind] .* norm2)]
kx_EKE_max_mu0p125[4] = kxp[512+argmax(tau_EKE_mu0p125_hm0p75[513:550,slice_ind] .* norm2)]
kx_EKE_max_mu0p125[5] = kxp[512+argmax(tau_EKE_mu0p125_hm0p5[513:550,slice_ind] .* norm2)]
kx_EKE_max_mu0p125[6] = kxp[512+argmax(tau_EKE_mu0p125_hm0p25[513:550,slice_ind] .* norm2)]

kx_EKE_max_mu0p125[12] = kxp[512+argmax(tau_EKE_mu0p125_h1p25[513:550,slice_ind] .* norm2)]
kx_EKE_max_mu0p125[11] = kxp[512+argmax(tau_EKE_mu0p125_h1p0[513:550,slice_ind] .* norm2)]
kx_EKE_max_mu0p125[10] = kxp[512+argmax(tau_EKE_mu0p125_h0p75[513:550,slice_ind] .* norm2)]
kx_EKE_max_mu0p125[9] = kxp[512+argmax(tau_EKE_mu0p125_h0p5[513:550,slice_ind] .* norm2)];
kx_EKE_max_mu0p125[8] = kxp[512+argmax(tau_EKE_mu0p125_h0p25[513:550,slice_ind] .* norm2)];

kx_EKE_max_mu0p5 = zeros(13)
kx_tau_wave_min_mu0p5 = zeros(13)

max_tau_mu0p5 = zeros(13, 50)

slice_ind = 513
norm2 = 3600 * 24 * 31

for i in range(1,13)
    max_tau = zeros(50)
    for j in range(1,50)
        max_tau_mu0p5[i,j] = maximum([tau_wave_BC[i,513+j,slice_ind], tau_wave_BT[i,513+j,slice_ind]]) * norm2
    end
    
    kx_tau_wave_min_mu0p5[i] = kxp[512+argmin(max_tau_mu0p5[i,:])]
    # println(argmin(max_tau))
end

ni=512

kx_EKE_max_mu0p5[2] = kxp[ni+argmax(tau_EKE_mu0p5_hm1p25[513:550,slice_ind] .* norm2)]
kx_EKE_max_mu0p5[3] = kxp[ni+argmax(tau_EKE_mu0p5_hm1p0[513:550,slice_ind] .* norm2)]
kx_EKE_max_mu0p5[4] = kxp[ni+argmax(tau_EKE_mu0p5_hm0p75[513:550,slice_ind] .* norm2)]
kx_EKE_max_mu0p5[5] = kxp[ni+argmax(tau_EKE_mu0p5_hm0p5[513:550,slice_ind] .* norm2)]
kx_EKE_max_mu0p5[6] = kxp[ni+argmax(tau_EKE_mu0p5_hm0p25[513:550,slice_ind] .* norm2)]

kx_EKE_max_mu0p5[12] = kxp[ni+argmax(tau_EKE_mu0p5_h1p25[513:550,slice_ind] .* norm2)]
kx_EKE_max_mu0p5[11] = kxp[ni+argmax(tau_EKE_mu0p5_h1p0[513:550,slice_ind] .* norm2)]
kx_EKE_max_mu0p5[10] = kxp[ni+argmax(tau_EKE_mu0p5_h0p75[513:550,slice_ind] .* norm2)]
kx_EKE_max_mu0p5[9] = kxp[ni+argmax(tau_EKE_mu0p5_h0p5[513:550,slice_ind] .* norm2)];
kx_EKE_max_mu0p5[8] = kxp[ni+argmax(tau_EKE_mu0p5_h0p25[513:550,slice_ind] .* norm2)];



function EKE_spec(i, k, kxp, kyp, data_dir; yr_last=89.0, nus=zeros(13))
    j=1  # quad drag index    
    
    h0 = (h0s[i], 0.0)
    kappa_loc = kappas[j]
    mu_loc = mus[k]
    
    global model_params = redef_mu_kappa_topoPV_h0_nu(model_params, mu_loc, kappa_loc, (0., 0.), h0, nus[i])
    global topo_PV, eta = define_topo(model_params)
    global model_params = redef_mu_kappa_topoPV_h0_nu(model_params, mu_loc, kappa_loc, topo_PV, h0, nus[i])
    
    global model_params = redef_mu_kappa_beta(model_params, mu_loc, kappa_loc, 0.)
    
    a = load(data_dir*jld_name(model_params,yr_last))

    # EKE_2D = reshape(sum(a["jld_data"]["kspace_modal_nrg_spectrum"], dims=3), Nx, Ny)   # summing BT and BC components of EKE
    # fft_EKE = fftshift(EKE_2D)  # above I do sqrt(eke diag) / ((2pi)^2)

    # prefactor of grid.Ksq turns from spectral energy density to energy spectrum
    EKE_2D = reshape(sum(a["jld_data"]["kspace_modal_nrg_spectrum"], dims=3), Nx, Ny)
    # sdf = Lx * Ly * ((2*pi)^-2)   # this is the spectral density factor; dk_{x} * dk_{y}
    # fft_EKE = ((grid.Ksq .* 0.5 .* fftshift(EKE_2D) ./ (Nx*Ny) .* sdf)) ./ ((2*pi)^2) 

    fft_EKE = (0.5 .* fftshift(EKE_2D) ./ (Nx*Ny)) #./ ((2*pi)^2)
    
    return fft_EKE
end


data_dir_EKE_spec = "/scratch/cimes/ml1994/QG/julia/data/JPO_2LQG_on_slope_data/mu0p25_kspace/"

νstars = [10^-11, 10^-11, 10^-11, 10^-11, 10^-11, 10^-11, 10^-11 , 10^-13, 10^-13, 10^-13, 10^-13, 10^-13, 10^-13]
nus = νstars .* ((U[1]-U[end])/2) * (Lx/2/pi)^7

EKE_mu0p25_hm1p25 = EKE_spec(2, 2, kxp, kyp, data_dir_EKE_spec; nus=nus);
EKE_mu0p25_hm0p75 = EKE_spec(4, 2, kxp, kyp, data_dir_EKE_spec; nus=nus);
EKE_mu0p25_hm0p25 = EKE_spec(6, 2, kxp, kyp, data_dir_EKE_spec; nus=nus);
EKE_mu0p25_h0p25  = EKE_spec(8, 2, kxp, kyp, data_dir_EKE_spec; nus=nus);
EKE_mu0p25_h0p75  = EKE_spec(10, 2, kxp, kyp, data_dir_EKE_spec; nus=nus);
EKE_mu0p25_h1p25  = EKE_spec(12, 2, kxp, kyp, data_dir_EKE_spec; nus=nus);


EKE_mu0p25_hm1p0 = EKE_spec(3, 2, kxp, kyp, data_dir_EKE_spec; nus=nus);
EKE_mu0p25_hm0p5 = EKE_spec(5, 2, kxp, kyp, data_dir_EKE_spec; nus=nus);
EKE_mu0p25_h0p5  = EKE_spec(9, 2, kxp, kyp, data_dir_EKE_spec; nus=nus);
EKE_mu0p25_h1p0  = EKE_spec(11, 2, kxp, kyp, data_dir_EKE_spec; nus=nus);

# for mu = 0.125
data_dir_EKE_spec_mu0p125 = "/scratch/cimes/ml1994/QG/julia/data/JPO_2LQG_on_slope_data/mu0p125_kspace/"

νstars = [10^-11, 10^-11, 10^-11, 10^-11, 10^-11, 10^-11, 10^-11 , 10^-13, 10^-13, 10^-13, 10^-13, 10^-13, 10^-13]
nus = νstars .* ((U[1]-U[end])/2) * (Lx/2/pi)^7

EKE_mu0p125_hm1p25 = EKE_spec(2, 1, kxp, kyp, data_dir_EKE_spec_mu0p125; nus=nus);
EKE_mu0p125_hm0p75 = EKE_spec(4, 1, kxp, kyp, data_dir_EKE_spec_mu0p125; nus=nus);
EKE_mu0p125_hm0p25 = EKE_spec(6, 1, kxp, kyp, data_dir_EKE_spec_mu0p125; nus=nus);
EKE_mu0p125_h0p25  = EKE_spec(8, 1, kxp, kyp, data_dir_EKE_spec_mu0p125; nus=nus);
EKE_mu0p125_h0p75  = EKE_spec(10, 1, kxp, kyp, data_dir_EKE_spec_mu0p125; nus=nus);
EKE_mu0p125_h1p25  = EKE_spec(12, 1, kxp, kyp, data_dir_EKE_spec_mu0p125; nus=nus);


EKE_mu0p125_hm1p0 = EKE_spec(3, 1, kxp, kyp, data_dir_EKE_spec_mu0p125; nus=nus);
EKE_mu0p125_hm0p5 = EKE_spec(5, 1, kxp, kyp, data_dir_EKE_spec_mu0p125; nus=nus);
EKE_mu0p125_h0p5  = EKE_spec(9, 1, kxp, kyp, data_dir_EKE_spec_mu0p125; nus=nus);
EKE_mu0p125_h1p0  = EKE_spec(11, 1, kxp, kyp, data_dir_EKE_spec_mu0p125; nus=nus);

# for mu = 0.125
data_dir_EKE_spec_mu0p5 = "/scratch/cimes/ml1994/QG/julia/data/JPO_2LQG_on_slope_data/mu0p5_kspace/"

νstars = [10^-13, 10^-13, 10^-13, 10^-13, 10^-13, 10^-13, 10^-13 , 10^-13, 10^-13, 10^-13, 10^-13, 10^-13, 10^-13]
nus = νstars .* ((U[1]-U[end])/2) * (Lx/2/pi)^7

EKE_mu0p5_hm1p25 = EKE_spec(2, 3, kxp, kyp, data_dir_EKE_spec_mu0p5; yr_last=179.0, nus=nus);
EKE_mu0p5_hm0p75 = EKE_spec(4, 3, kxp, kyp, data_dir_EKE_spec_mu0p5; yr_last=179.0, nus=nus);
EKE_mu0p5_hm0p25 = EKE_spec(6, 3, kxp, kyp, data_dir_EKE_spec_mu0p5; yr_last=179.0, nus=nus);
EKE_mu0p5_h0p25  = EKE_spec(8, 3, kxp, kyp, data_dir_EKE_spec_mu0p5; yr_last=179.0, nus=nus);
EKE_mu0p5_h0p75  = EKE_spec(10, 3, kxp, kyp, data_dir_EKE_spec_mu0p5; yr_last=179.0, nus=nus);
EKE_mu0p5_h1p25  = EKE_spec(12, 3, kxp, kyp, data_dir_EKE_spec_mu0p5; yr_last=179.0, nus=nus);


EKE_mu0p5_hm1p0 = EKE_spec(3, 3, kxp, kyp, data_dir_EKE_spec_mu0p5; yr_last=179.0, nus=nus);
EKE_mu0p5_hm0p5 = EKE_spec(5, 3, kxp, kyp, data_dir_EKE_spec_mu0p5; yr_last=179.0, nus=nus);
EKE_mu0p5_h0p5  = EKE_spec(9, 3, kxp, kyp, data_dir_EKE_spec_mu0p5; yr_last=179.0, nus=nus);
EKE_mu0p5_h1p0  = EKE_spec(11, 3, kxp, kyp, data_dir_EKE_spec_mu0p5; yr_last=179.0, nus=nus);


n_end = 550
n_end2 = n_end-513

kx_arrest_mu0p125 = zeros(10)


h_ind = 2
kx_arrest_mu0p125[1] = kxp[513 + argmin(abs.(tau_EKE_mu0p125_hm1p25[514:n_end,slice_ind] * norm2 .- max_tau_mu0p125[h_ind,1:n_end2]))]

h_ind = 3
kx_arrest_mu0p125[2] = kxp[513 + argmin(abs.(tau_EKE_mu0p125_hm1p0[514:n_end,slice_ind] * norm2 .- max_tau_mu0p125[h_ind,1:n_end2]))]

n_end = 528
n_end2 = n_end-513


h_ind = 4
kx_arrest_mu0p125[3] = kxp[513 + argmin(abs.(tau_EKE_mu0p125_hm0p75[514:n_end,slice_ind] * norm2 .- max_tau_mu0p125[h_ind,1:n_end2]))]

h_ind = 5
kx_arrest_mu0p125[4] = kxp[513 + argmin(abs.(tau_EKE_mu0p125_hm0p5[514:n_end,slice_ind] * norm2 .- max_tau_mu0p125[h_ind,1:n_end2]))]

h_ind = 6
kx_arrest_mu0p125[5] = kxp[513 + argmin(abs.(tau_EKE_mu0p125_hm0p25[514:n_end,slice_ind] * norm2 .- max_tau_mu0p125[h_ind,1:n_end2]))]

n_end = 528
n_end2 = n_end-513

h_ind = 12
kx_arrest_mu0p125[10] = kxp[513 + argmin(abs.(tau_EKE_mu0p125_h1p25[514:n_end,slice_ind] * norm2 .- max_tau_mu0p125[h_ind,1:n_end2]))]

h_ind = 11
kx_arrest_mu0p125[9] = kxp[513 + argmin(abs.(tau_EKE_mu0p125_h1p0[514:n_end,slice_ind] * norm2 .- max_tau_mu0p125[h_ind,1:n_end2]))]

h_ind = 10
kx_arrest_mu0p125[8] = kxp[513 + argmin(abs.(tau_EKE_mu0p125_h0p75[514:n_end,slice_ind] * norm2 .- max_tau_mu0p125[h_ind,1:n_end2]))]

n_end = 517
n_end2 = n_end-513

h_ind = 9
kx_arrest_mu0p125[7] = kxp[513 + argmin(abs.(tau_EKE_mu0p125_h0p5[514:n_end,slice_ind] * norm2 .- max_tau_mu0p125[h_ind,1:n_end2]))]

h_ind = 8
kx_arrest_mu0p125[6] = kxp[513 + argmin(abs.(tau_EKE_mu0p125_h0p25[514:n_end,slice_ind] * norm2 .- max_tau_mu0p125[h_ind,1:n_end2]))]



kx_arrest_mu0p25 = zeros(10)

n_end = 545
n_end2 = n_end-513

h_ind = 2
kx_arrest_mu0p25[1] = kxp[513 + argmin(abs.(tau_EKE_mu0p25_hm1p25[514:n_end,slice_ind] * norm2 .- max_tau_mu0p25[h_ind,1:n_end2]))]

h_ind = 3
kx_arrest_mu0p25[2] = kxp[513 + argmin(abs.(tau_EKE_mu0p25_hm1p0[514:n_end,slice_ind] * norm2 .- max_tau_mu0p25[h_ind,1:n_end2]))]


##################################
n_end = 528
n_end2 = n_end-513

h_ind = 4
kx_arrest_mu0p25[3] = kxp[513 + argmin(abs.(tau_EKE_mu0p25_hm0p75[514:n_end,slice_ind] * norm2 .- max_tau_mu0p25[h_ind,1:n_end2]))]

h_ind = 5
kx_arrest_mu0p25[4] = kxp[513 + argmin(abs.(tau_EKE_mu0p25_hm0p5[514:n_end,slice_ind] * norm2 .- max_tau_mu0p25[h_ind,1:n_end2]))]

h_ind = 6
kx_arrest_mu0p25[5] = kxp[513 + argmin(abs.(tau_EKE_mu0p25_hm0p25[514:n_end,slice_ind] * norm2 .- max_tau_mu0p25[h_ind,1:n_end2]))]


h_ind = 12
kx_arrest_mu0p25[10] = kxp[513 + argmin(abs.(tau_EKE_mu0p25_h1p25[514:n_end,slice_ind] * norm2 .- max_tau_mu0p25[h_ind,1:n_end2]))]

h_ind = 11
kx_arrest_mu0p25[9] = kxp[513 + argmin(abs.(tau_EKE_mu0p25_h1p0[514:n_end,slice_ind] * norm2 .- max_tau_mu0p25[h_ind,1:n_end2]))]

h_ind = 10
kx_arrest_mu0p25[8] = kxp[513 + argmin(abs.(tau_EKE_mu0p25_h0p75[514:n_end,slice_ind] * norm2 .- max_tau_mu0p25[h_ind,1:n_end2]))]

h_ind = 9
kx_arrest_mu0p25[7] = kxp[513 + argmin(abs.(tau_EKE_mu0p25_h0p5[514:n_end,slice_ind] * norm2 .- max_tau_mu0p25[h_ind,1:n_end2]))]

###############################################
n_end = 523
n_end2 = n_end-513

h_ind = 8
kx_arrest_mu0p25[6] = kxp[513 + argmin(abs.(tau_EKE_mu0p25_h0p25[514:n_end,slice_ind] * norm2 .- max_tau_mu0p25[h_ind,1:n_end2]))]


kx_arrest_mu0p5 = zeros(10)

n_end = 545
n_end2 = n_end-513

h_ind = 2
kx_arrest_mu0p5[1] = kxp[513 + argmin(abs.(tau_EKE_mu0p5_hm1p25[514:n_end,slice_ind] * norm2 .- max_tau_mu0p5[h_ind,1:n_end2]))]

h_ind = 3
kx_arrest_mu0p5[2] = kxp[513 + argmin(abs.(tau_EKE_mu0p5_hm1p0[514:n_end,slice_ind] * norm2 .- max_tau_mu0p5[h_ind,1:n_end2]))]


###################################
n_end = 528
n_end2 = n_end-513


h_ind = 4
kx_arrest_mu0p5[3] = kxp[513 + argmin(abs.(tau_EKE_mu0p5_hm0p75[514:n_end,slice_ind] * norm2 .- max_tau_mu0p5[h_ind,1:n_end2]))]

h_ind = 5
kx_arrest_mu0p5[4] = kxp[513 + argmin(abs.(tau_EKE_mu0p5_hm0p5[514:n_end,slice_ind] * norm2 .- max_tau_mu0p5[h_ind,1:n_end2]))]

h_ind = 6
kx_arrest_mu0p5[5] = kxp[513 + argmin(abs.(tau_EKE_mu0p5_hm0p25[514:n_end,slice_ind] * norm2 .- max_tau_mu0p5[h_ind,1:n_end2]))]


h_ind = 12
kx_arrest_mu0p5[10] = kxp[513 + argmin(abs.(tau_EKE_mu0p5_h1p25[514:n_end,slice_ind] * norm2 .- max_tau_mu0p5[h_ind,1:n_end2]))]

h_ind = 11
kx_arrest_mu0p5[9] = kxp[513 + argmin(abs.(tau_EKE_mu0p5_h1p0[514:n_end,slice_ind] * norm2 .- max_tau_mu0p5[h_ind,1:n_end2]))]

h_ind = 10
kx_arrest_mu0p5[8] = kxp[513 + argmin(abs.(tau_EKE_mu0p5_h0p75[514:n_end,slice_ind] * norm2 .- max_tau_mu0p5[h_ind,1:n_end2]))]

h_ind = 9
kx_arrest_mu0p5[7] = kxp[513 + argmin(abs.(tau_EKE_mu0p5_h0p5[514:n_end,slice_ind] * norm2 .- max_tau_mu0p5[h_ind,1:n_end2]))]

h_ind = 8
kx_arrest_mu0p5[6] = kxp[513 + argmin(abs.(tau_EKE_mu0p5_h0p25[514:n_end,slice_ind] * norm2 .- max_tau_mu0p5[h_ind,1:n_end2]))]




# wavenumbers of maximum EKE diagnosed and predicted by finding kx where friction time scale is equal to wave frequency 
norm2 = 3600 * 24 * 31

k_diag_max_EKE_mu0p125 = zeros(10)
k_pred_max_EKE_mu0p125 = zeros(10)

k_diag_max_EKE_mu0p125[1] = kxp[512+argmax(EKE_mu0p125_hm1p25[513:550,slice_ind])]
k_diag_max_EKE_mu0p125[2] = kxp[512+argmax(EKE_mu0p125_hm1p0[513:550,slice_ind])]
k_diag_max_EKE_mu0p125[3] = kxp[512+argmax(EKE_mu0p125_hm0p75[513:550,slice_ind])]
k_diag_max_EKE_mu0p125[4] = kxp[512+argmax(EKE_mu0p125_hm0p5[513:550,slice_ind])]
k_diag_max_EKE_mu0p125[5] = kxp[512+argmax(EKE_mu0p125_hm0p25[513:550,slice_ind])]

k_diag_max_EKE_mu0p125[6] = kxp[512+argmax(EKE_mu0p125_h0p25[513:550,slice_ind])]
k_diag_max_EKE_mu0p125[7] = kxp[512+argmax(EKE_mu0p125_h0p5[513:550,slice_ind])]
k_diag_max_EKE_mu0p125[8] = kxp[512+argmax(EKE_mu0p125_h0p75[513:550,slice_ind])]
k_diag_max_EKE_mu0p125[9] = kxp[512+argmax(EKE_mu0p125_h1p0[513:550,slice_ind])]
k_diag_max_EKE_mu0p125[10] = kxp[512+argmax(EKE_mu0p125_h1p25[513:550,slice_ind])]


for (i,k) in enumerate(vcat(range(2,6), range(8, 12)))
    k_pred_max_EKE_mu0p125[i] = kxp[513 + argmin(abs.(max_tau_mu0p25[k,1:20] .- mus[1] * norm2))]
end

# wavenumbers of maximum EKE diagnosed and predicted by finding kx where friction time scale is equal to wave frequency 

k_diag_max_EKE_mu0p25 = zeros(10)
k_pred_max_EKE_mu0p25 = zeros(10)

k_diag_max_EKE_mu0p25[1] = kxp[512+argmax(EKE_mu0p25_hm1p25[513:550,slice_ind])]
k_diag_max_EKE_mu0p25[2] = kxp[512+argmax(EKE_mu0p25_hm1p0[513:550,slice_ind])]
k_diag_max_EKE_mu0p25[3] = kxp[512+argmax(EKE_mu0p25_hm0p75[513:550,slice_ind])]
k_diag_max_EKE_mu0p25[4] = kxp[512+argmax(EKE_mu0p25_hm0p5[513:550,slice_ind])]
k_diag_max_EKE_mu0p25[5] = kxp[512+argmax(EKE_mu0p25_hm0p25[513:550,slice_ind])]

k_diag_max_EKE_mu0p25[6] = kxp[512+argmax(EKE_mu0p25_h0p25[513:550,slice_ind])]
k_diag_max_EKE_mu0p25[7] = kxp[512+argmax(EKE_mu0p25_h0p5[513:550,slice_ind])]
k_diag_max_EKE_mu0p25[8] = kxp[512+argmax(EKE_mu0p25_h0p75[513:550,slice_ind])]
k_diag_max_EKE_mu0p25[9] = kxp[512+argmax(EKE_mu0p25_h1p0[513:550,slice_ind])]
k_diag_max_EKE_mu0p25[10] = kxp[512+argmax(EKE_mu0p25_h1p25[513:550,slice_ind])]


for (i,k) in enumerate(vcat(range(2,6), range(8, 12)))
    k_pred_max_EKE_mu0p25[i] = kxp[513 + argmin(abs.(max_tau_mu0p25[k,1:20] .- mus[2] * norm2))]
end

# wavenumbers of maximum EKE diagnosed and predicted by finding kx where friction time scale is equal to wave frequency 

k_diag_max_EKE_mu0p5 = zeros(10)
k_pred_max_EKE_mu0p5 = zeros(10)

k_diag_max_EKE_mu0p5[1] = kxp[512+argmax(EKE_mu0p5_hm1p25[513:550,slice_ind])]
k_diag_max_EKE_mu0p5[2] = kxp[512+argmax(EKE_mu0p5_hm1p0[513:550,slice_ind])]
k_diag_max_EKE_mu0p5[3] = kxp[512+argmax(EKE_mu0p5_hm0p75[513:550,slice_ind])]
k_diag_max_EKE_mu0p5[4] = kxp[512+argmax(EKE_mu0p5_hm0p5[513:550,slice_ind])]
k_diag_max_EKE_mu0p5[5] = kxp[512+argmax(EKE_mu0p5_hm0p25[513:550,slice_ind])]

k_diag_max_EKE_mu0p5[6] = kxp[512+argmax(EKE_mu0p5_h0p25[513:550,slice_ind])]
k_diag_max_EKE_mu0p5[7] = kxp[512+argmax(EKE_mu0p5_h0p5[513:550,slice_ind])]
k_diag_max_EKE_mu0p5[8] = kxp[512+argmax(EKE_mu0p5_h0p75[513:550,slice_ind])]
k_diag_max_EKE_mu0p5[9] = kxp[512+argmax(EKE_mu0p5_h1p0[513:550,slice_ind])]
k_diag_max_EKE_mu0p5[10] = kxp[512+argmax(EKE_mu0p5_h1p25[513:550,slice_ind])]


for (i,k) in enumerate(vcat(range(2,6), range(8, 12)))
    k_pred_max_EKE_mu0p5[i] = kxp[513 + argmin(abs.(max_tau_mu0p5[k,1:20] .- mus[3] * norm2))]
end


function EKE_raw_spec(i, k, kxp, kyp, data_dir; yr_last=89.0, nus=zeros(13))
    j=1  # quad drag index    
    
    h0 = (h0s[i], 0.0)
    kappa_loc = kappas[j]
    mu_loc = mus[k]
    
    global model_params = redef_mu_kappa_topoPV_h0_nu(model_params, mu_loc, kappa_loc, (0., 0.), h0, nus[i])
    global topo_PV, eta = define_topo(model_params)
    global model_params = redef_mu_kappa_topoPV_h0_nu(model_params, mu_loc, kappa_loc, topo_PV, h0, nus[i])
    
    global model_params = redef_mu_kappa_beta(model_params, mu_loc, kappa_loc, 0.)
    
    a = load(data_dir*jld_name(model_params,yr_last))

    EKE_2D = reshape(sum(a["jld_data"]["kspace_modal_nrg_spectrum"], dims=3), Nx, Ny)   # summing BT and BC components of EKE
    # fft_EKE = fftshift(EKE_2D) ./ (Nx*Ny)  # above I do sqrt(eke diag) / ((2pi)^2)

    # # # prefactor of grid.Ksq turns from spectral energy density to energy spectrum
    # EKE_2D = reshape(sum(a["jld_data"]["kspace_modal_nrg_spectrum"], dims=3), Nx, Ny)
    # sdf = Lx * Ly * ((2*pi)^-2)   # this is the spectral density factor; dk_{x} * dk_{y}
    # fft_EKE = ((fftshift(grid.Ksq) .* 0.5 .* fftshift(EKE_2D) ./ (Nx*Ny) .* sdf)) ./ ((2*pi)^2)

    fft_EKE = ((0.5 .* fftshift(EKE_2D) ./ (Nx*Ny))) # ./ ((2*pi)^2)
    
    return fft_EKE # .* fftshift(grid.Ksq) .* sdf  # fft_EKE
end


data_dir_EKE_spec = "/scratch/cimes/ml1994/QG/julia/data/JPO_2LQG_on_slope_data/mu0p25_kspace/"

νstars = [10^-11, 10^-11, 10^-11, 10^-11, 10^-11, 10^-11, 10^-11 , 10^-13, 10^-13, 10^-13, 10^-13, 10^-13, 10^-13]
nus = νstars .* ((U[1]-U[end])/2) * (Lx/2/pi)^7

EKE_raw_mu0p25_hm1p25 = EKE_raw_spec(2, 2, kxp, kyp, data_dir_EKE_spec; nus=nus);
EKE_raw_mu0p25_hm0p75 = EKE_raw_spec(4, 2, kxp, kyp, data_dir_EKE_spec; nus=nus);
EKE_raw_mu0p25_hm0p25 = EKE_raw_spec(6, 2, kxp, kyp, data_dir_EKE_spec; nus=nus);
EKE_raw_mu0p25_h0p25  = EKE_raw_spec(8, 2, kxp, kyp, data_dir_EKE_spec; nus=nus);
EKE_raw_mu0p25_h0p75  = EKE_raw_spec(10, 2, kxp, kyp, data_dir_EKE_spec; nus=nus);
EKE_raw_mu0p25_h1p25  = EKE_raw_spec(12, 2, kxp, kyp, data_dir_EKE_spec; nus=nus);


EKE_raw_mu0p25_hm1p0 = EKE_raw_spec(3, 2, kxp, kyp, data_dir_EKE_spec; nus=nus);
EKE_raw_mu0p25_hm0p5 = EKE_raw_spec(5, 2, kxp, kyp, data_dir_EKE_spec; nus=nus);
EKE_raw_mu0p25_h0p5  = EKE_raw_spec(9, 2, kxp, kyp, data_dir_EKE_spec; nus=nus);
EKE_raw_mu0p25_h1p0  = EKE_raw_spec(11, 2, kxp, kyp, data_dir_EKE_spec; nus=nus);


# for mu = 0.125
data_dir_EKE_spec_mu0p125 = "/scratch/cimes/ml1994/QG/julia/data/JPO_2LQG_on_slope_data/mu0p125_kspace/"

νstars = [10^-11, 10^-11, 10^-11, 10^-11, 10^-11, 10^-11, 10^-11 , 10^-13, 10^-13, 10^-13, 10^-13, 10^-13, 10^-13]
nus = νstars .* ((U[1]-U[end])/2) * (Lx/2/pi)^7

EKE_raw_mu0p125_hm1p25 = EKE_raw_spec(2, 1, kxp, kyp, data_dir_EKE_spec_mu0p125; nus=nus);
EKE_raw_mu0p125_hm0p75 = EKE_raw_spec(4, 1, kxp, kyp, data_dir_EKE_spec_mu0p125; nus=nus);
EKE_raw_mu0p125_hm0p25 = EKE_raw_spec(6, 1, kxp, kyp, data_dir_EKE_spec_mu0p125; nus=nus);
EKE_raw_mu0p125_h0p25  = EKE_raw_spec(8, 1, kxp, kyp, data_dir_EKE_spec_mu0p125; nus=nus);
EKE_raw_mu0p125_h0p75  = EKE_raw_spec(10, 1, kxp, kyp, data_dir_EKE_spec_mu0p125; nus=nus);
EKE_raw_mu0p125_h1p25  = EKE_raw_spec(12, 1, kxp, kyp, data_dir_EKE_spec_mu0p125; nus=nus);


EKE_raw_mu0p125_hm1p0 = EKE_raw_spec(3, 1, kxp, kyp, data_dir_EKE_spec_mu0p125; nus=nus);
EKE_raw_mu0p125_hm0p5 = EKE_raw_spec(5, 1, kxp, kyp, data_dir_EKE_spec_mu0p125; nus=nus);
EKE_raw_mu0p125_h0p5  = EKE_raw_spec(9, 1, kxp, kyp, data_dir_EKE_spec_mu0p125; nus=nus);
EKE_raw_mu0p125_h1p0  = EKE_raw_spec(11, 1, kxp, kyp, data_dir_EKE_spec_mu0p125; nus=nus);


# for mu = 0.125
data_dir_EKE_spec_mu0p5 = "/scratch/cimes/ml1994/QG/julia/data/JPO_2LQG_on_slope_data/mu0p5_kspace/"

νstars = [10^-13, 10^-13, 10^-13, 10^-13, 10^-13, 10^-13, 10^-13 , 10^-13, 10^-13, 10^-13, 10^-13, 10^-13, 10^-13]
nus = νstars .* ((U[1]-U[end])/2) * (Lx/2/pi)^7

EKE_raw_mu0p5_hm1p25 = EKE_raw_spec(2, 3, kxp, kyp, data_dir_EKE_spec_mu0p5; yr_last=179.0, nus=nus);
EKE_raw_mu0p5_hm0p75 = EKE_raw_spec(4, 3, kxp, kyp, data_dir_EKE_spec_mu0p5; yr_last=179.0, nus=nus);
EKE_raw_mu0p5_hm0p25 = EKE_raw_spec(6, 3, kxp, kyp, data_dir_EKE_spec_mu0p5; yr_last=179.0, nus=nus);
EKE_raw_mu0p5_h0p25  = EKE_raw_spec(8, 3, kxp, kyp, data_dir_EKE_spec_mu0p5; yr_last=179.0, nus=nus);
EKE_raw_mu0p5_h0p75  = EKE_raw_spec(10, 3, kxp, kyp, data_dir_EKE_spec_mu0p5; yr_last=179.0, nus=nus);
EKE_raw_mu0p5_h1p25  = EKE_raw_spec(12, 3, kxp, kyp, data_dir_EKE_spec_mu0p5; yr_last=179.0, nus=nus);


EKE_raw_mu0p5_hm1p0 = EKE_raw_spec(3, 3, kxp, kyp, data_dir_EKE_spec_mu0p5; yr_last=179.0, nus=nus);
EKE_raw_mu0p5_hm0p5 = EKE_raw_spec(5, 3, kxp, kyp, data_dir_EKE_spec_mu0p5; yr_last=179.0, nus=nus);
EKE_raw_mu0p5_h0p5  = EKE_raw_spec(9, 3, kxp, kyp, data_dir_EKE_spec_mu0p5; yr_last=179.0, nus=nus);
EKE_raw_mu0p5_h1p0  = EKE_raw_spec(11, 3, kxp, kyp, data_dir_EKE_spec_mu0p5; yr_last=179.0, nus=nus);

using Statistics, Printf

function r2_1to1(observed, predicted)
    # Residual Sum of Squares: deviation from the 1:1 line (y = x)
    ss_res = sum((observed .- predicted).^2)
    
    # Total Sum of Squares: deviation from the mean of observed data
    ss_tot = sum((observed .- mean(observed)).^2)
    
    # R² calculation
    return 1 - (ss_res / ss_tot)
end

#################################################################################
#################################################################################

fig, ax = plt.subplots(3,1, figsize=(7,20))

ax1=ax[1]; ax2=ax[2]; ax3=ax[3];

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

ax1.set_xlabel(L"k^\mathrm{diag.}_{\tau \mathrm{ \ max.}}", fontsize=fsize)
ax1.set_ylabel(L"k^\mathrm{pred.}_{\tau \mathrm{ \ max.}}", fontsize=fsize)

ax1.legend(loc="lower right", bbox_to_anchor=(1.175, -0.025), fontsize=lsize, edgecolor="black", fancybox=false, framealpha=1, shadow=true)

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

ax2.legend(loc="lower right", bbox_to_anchor=(1.175, -0.025), fontsize=lsize, edgecolor="black", fancybox=false, framealpha=1, shadow=true)

ax2.tick_params(axis="both", labelsize=lsize) 

ax2.grid()
ax2.set_axisbelow(true)


#################################################################################
#################################################################################


ax3.plot(kxp, EKE_raw_mu0p25_hm1p25[:,slice_ind], color=h_color(h0s[2],hc_range,colors), linestyle="dashed")
ax3.plot(kxp, EKE_raw_mu0p25_hm1p0[:,slice_ind], color=h_color(h0s[3],hc_range,colors), linestyle="dashed")
ax3.plot(kxp, EKE_raw_mu0p25_hm0p75[:,slice_ind], color=h_color(h0s[4],hc_range,colors), linestyle="dashed")
ax3.plot(kxp, EKE_raw_mu0p25_hm0p5[:,slice_ind], color=h_color(h0s[5],hc_range,colors), linestyle="dashed")
ax3.plot(kxp, EKE_raw_mu0p25_hm0p25[:,slice_ind], color=h_color(h0s[6],hc_range,colors), linestyle="dashed")


ax3.plot(kxp, EKE_raw_mu0p25_h1p25[:,slice_ind], color=h_color(h0s[12],hc_range,colors))
ax3.plot(kxp, EKE_raw_mu0p25_h1p0[:,slice_ind], color=h_color(h0s[11],hc_range,colors))
ax3.plot(kxp, EKE_raw_mu0p25_h0p75[:,slice_ind], color=h_color(h0s[10],hc_range,colors))
ax3.plot(kxp, EKE_raw_mu0p25_h0p5[:,slice_ind], color=h_color(h0s[9],hc_range,colors))
ax3.plot(kxp, EKE_raw_mu0p25_h0p25[:,slice_ind], color=h_color(h0s[8],hc_range,colors))

cc = 8.0e-5
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

ax3.set_ylim(1e-4, 1e1)

ax3.set_xlabel(L"k_{x} \ \lambda", fontsize=fsize)
ax3.set_ylabel(L"\mathrm{EKE \ [m}^2 \mathrm{s}^{-2} \mathrm{]}", fontsize=fsize)

ax3.tick_params(axis="both", labelsize=lsize) 

ax3.legend(fontsize=lsize, edgecolor="black", fancybox=false, framealpha=1, shadow=true)

ax3.grid()
ax3.set_axisbelow(true)

#################################################################################
#################################################################################


for (i,t) in enumerate([L"(a)", L"(b)", L"(c)"])
    ax[i].text(0.05, 0.95, t, transform=ax[i].transAxes, fontsize=fsize,
            ha="left", va="top")
end


#################################################################################
#################################################################################

savefig("./JPO_rev2_figs/kpred_kdiag_EKE_panels.pdf",bbox_inches="tight")






