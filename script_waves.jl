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

i=6  # bottom slope index
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

# a = load(data_dir_psi_ot*jld_name(model_params,89.5))

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
        
        A = k2 * (k2 + Ld^-2)
        B = (beta_topo + im * mu_loc * k2 / kx) * (k2 + (2*Ld^2)^-1) - U1 * k2 * (k2 + Ld^-2)
        C = U1 * k2 * (U1 * (2 * Ld^2)^-1 - beta_topo - im * mu_loc * k2/kx)

        kx_to_k2 = (kx/sqrt(k2))
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

mu_loc = mus[2]

c_BT_full = zeros(length(h0s), length(ells_x), length(ells_y))
c_BC_full = zeros(length(h0s), length(ells_x), length(ells_y))
growth = zeros(length(h0s), length(ells_x), length(ells_y))

for h_ind in [2, 4, 6, 8, 10, 12]  # range(1, length(h0s))

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
                global kx = 2*pi/Lkx    # this is in radians per meter
                global ky = 2*pi/Lky    # this is in radians per meter
            # end

            # these phase speeds are in m/s (no radians)
            c_BT_full[h_ind, i, j], nul = dispersion_relation_full_bottom(alpha_s, 0.01, "BT", 0.0; kx=kx, ky=ky) 
            c_BC_full[h_ind, i, j], nul = dispersion_relation_full_bottom(alpha_s, 0.01, "BC", 0.0; kx=kx, ky=ky)
            nul, growth[h_ind, i, j] = dispersion_relation_full_bottom(alpha_s, 0.01, "BC", mu_loc; kx=kx, ky=ky)
            
        end
    end
end



tau_wave = zeros(6, length(ells_x), length(ells_y))
tau_wave_BC = zeros(6, length(ells_x), length(ells_y))
tau_wave_BT = zeros(6, length(ells_x), length(ells_y))

h_ind_bank = [2, 4, 6, 8, 10, 12]

# ells_x, ells_y are in meters (no radians; not normed by Ld)
for (h, h_ind) in enumerate(h_ind_bank)
    for (i, lx) in enumerate(ells_x)
        for (j, ly) in enumerate(ells_y)
            l = sqrt(lx^-2 + ly^-2)^-1
            # tau_wave[h,i,j] = maximum(abs.([c_BC_full[h_ind, i, j], c_BT_full[h_ind, i, j]])) / l

            # with c in m/s (no radians), this is a time scale in Hz (i.e., no radians)
            tau_wave_BT[h,i,j] = abs(c_BT_full[h_ind, i, j]) / l
            tau_wave_BC[h,i,j] = abs(c_BC_full[h_ind, i, j]) / l
            tau_wave[h,i,j] = maximum(abs.([c_BC_full[h_ind, i, j], c_BT_full[h_ind, i, j]])) / l
        end
    end
end



function EKE_timescale(i, k, kxp, kyp, data_dir)
    j=1  # quad drag index

    h0 = (h0s[i], 0.0)
    kappa_loc = kappas[j]
    mu_loc = mus[k]

    global model_params = redef_mu_kappa_topoPV_h0_nu(model_params, mu_loc, kappa_loc, (0., 0.), h0, nus[i])
    global topo_PV, eta = define_topo(model_params)
    global model_params = redef_mu_kappa_topoPV_h0_nu(model_params, mu_loc, kappa_loc, topo_PV, h0, nus[i])

    global model_params = redef_mu_kappa_beta(model_params, mu_loc, kappa_loc, 0.)

    a = load(data_dir*jld_name(model_params,89.0))

    EKE_2D = reshape(sum(a["jld_data"]["kspace_modal_nrg_spectrum"], dims=3), Nx, Ny)

    tau_EKE = zeros(length(kxp), length(kyp))

    l_peak_EKE = zeros(length(kyp))

    fft_EKE = sqrt.(fftshift(EKE_2D))

    # kxp, kyp are in radians per meter times Ld (so, radians overall)
    for (i, kx) in enumerate(kxp ./ (2*pi*Ld))  # this correctly removes Ld and radian factors
        for (j, ky) in enumerate(kyp ./ (2*pi*Ld))
            l = sqrt(kx^2 + ky^2)^-1    # these are wavenumbers in inverse length (no radians)

            # factor of 2pi^2 req'd because grid.Ksq has factor of 2pi^2
            tau_EKE[i,j] = fft_EKE[i, j] / l / ((2*pi)^2)    # this is now in Hz, no normalization needed for radians, Ld, etc.
        end
    end

    for (j, ky) in enumerate(kyp ./ (2*pi*Ld))
        l_peak_EKE[j] = kxp[argmax(fft_EKE[:, j])] / (2 * pi * Ld)
        # l_peak_EKE[j] = sum(fft_EKE[513:end, j] .* (kxp[513:end] ./ (2 * pi * Ld))) / sum(fft_EKE[513:end, j])
    end

    return tau_EKE, l_peak_EKE
end

data_dir_EKE_spec = "/scratch/cimes/ml1994/QG/julia/data/JPO_2LQG_on_slope_data/mu0p25_kspace/"


tau_EKE_mu0p25_hm1p25, lpeak_mu0p25_hm1p25 = EKE_timescale(2, 2, kxp, kyp, data_dir_EKE_spec);
tau_EKE_mu0p25_hm0p75, lpeak_mu0p25_hm0p75 = EKE_timescale(4, 2, kxp, kyp, data_dir_EKE_spec);
tau_EKE_mu0p25_hm0p25, lpeak_mu0p25_hm0p25 = EKE_timescale(6, 2, kxp, kyp, data_dir_EKE_spec);
tau_EKE_mu0p25_h0p25, lpeak_mu0p25_h0p25  = EKE_timescale(8, 2, kxp, kyp, data_dir_EKE_spec);
tau_EKE_mu0p25_h0p75, lpeak_mu0p25_h0p75  = EKE_timescale(10, 2, kxp, kyp, data_dir_EKE_spec);
tau_EKE_mu0p25_h1p25, lpeak_mu0p25_h1p25  = EKE_timescale(12, 2, kxp, kyp, data_dir_EKE_spec);













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

for (i,t) in enumerate(["(a)", "(d)", "(b)", "(e)", "(c)", "(f)"])
    ax[i].text(0.05, 0.95, t, transform=ax[i].transAxes, fontsize=fsize,
            ha="left", va="top")
end


savefig("./JPO_rev1_figs/total_EKE_2D.png",bbox_inches="tight")







function plot_timescales(ax, tau_wave_BC, tau_wave_BT, tau_EKE, kxp)

    norm = 3600 * 24 * 31
    
    ax.plot(kxp, tau_wave_BC .* norm, "k-")
    ax.plot(kxp, tau_wave_BT .* norm, "k-")
    
    ax.plot(kxp, tau_EKE .* norm, "r-")
    
end

fsize = 26   # 20
lsize = 20   # 16

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
    axn.set_ylim(0, 0.225)

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
for (i,t) in enumerate(["(a)", "(d)", "(b)", "(e)", "(c)", "(f)"])
    ax[i].text(0.05, 0.95, t, transform=ax[i].transAxes, fontsize=fsize,
            ha="left", va="top", bbox=Dict(
        "facecolor"=>"white",  # Background color
        "alpha"=>0.8,          # Transparency (0=transparent, 1=opaque)
        "edgecolor"=>"white",  # Border color (set to match facecolor to avoid a visible edge)
        "pad"=>5.0             # Padding around the text
    ))
end


savefig("./JPO_rev1_figs/timescale_kx.png",bbox_inches="tight")



my_gradient = matplotlib.colors.LinearSegmentedColormap.from_list("my_gradient", (
                 (0.000, (0.0, 142.0/255.0, 1.0)),
                 # (0.250, (0.000, 0.145, 0.702)),
                 (0.500, (1.000, 1.000, 1.000)),
                 # (0.750, (0.780, 0.012, 0.051)),
                 (1.0, (1.0, 0.0, 60.0/255.0))))

my_gradient.set_bad("black")


#
cmap = PyPlot.cm.get_cmap("bwr")
cmap.set_bad("black")

fig, ax = plt.subplots(2,3, figsize=(16,10))

ax[6].pcolormesh(kxp, kyp[:], (tau_EKE_mu0p25_hm1p25 ./ (tau_wave[1,:,:]))',norm=matplotlib.colors.LogNorm(vmin=1e-2, vmax=100), cmap=my_gradient)
ax[4].pcolormesh(kxp, kyp[:], (tau_EKE_mu0p25_hm0p75 ./ (tau_wave[2,:,:]))',norm=matplotlib.colors.LogNorm(vmin=1e-2, vmax=100), cmap=my_gradient)
ax[2].pcolormesh(kxp, kyp[:], (tau_EKE_mu0p25_hm0p25 ./ (tau_wave[3,:,:]))',norm=matplotlib.colors.LogNorm(vmin=1e-2, vmax=100), cmap=my_gradient)

ax[5].pcolormesh(kxp, kyp[:], (tau_EKE_mu0p25_h1p25 ./ (tau_wave[6,:,:]))',norm=matplotlib.colors.LogNorm(vmin=1e-2, vmax=100), cmap=my_gradient)
ax[3].pcolormesh(kxp, kyp[:], (tau_EKE_mu0p25_h0p75 ./ (tau_wave[5,:,:]))',norm=matplotlib.colors.LogNorm(vmin=1e-2, vmax=100), cmap=my_gradient)
pc_h0p25 = ax[1].pcolormesh(kxp, kyp[:], (tau_EKE_mu0p25_h0p25 ./ (tau_wave[4,:,:]))',norm=matplotlib.colors.LogNorm(vmin=1e-2, vmax=100), cmap=my_gradient)


ax[6].contour(kxp, kyp[:], growth[2, :, :]', colors=(43.0/255.0, 208.0/255.0, 0.0), levels=def_levs(growth[2, :, :]), alpha=1.0)
ax[4].contour(kxp, kyp[:], growth[4, :, :]', colors=(43.0/255.0, 208.0/255.0, 0.0), levels=def_levs(growth[4, :, :]), alpha=1.0)
ax[2].contour(kxp, kyp[:], growth[6, :, :]', colors=(43.0/255.0, 208.0/255.0, 0.0), levels=def_levs(growth[6, :, :]), alpha=1.0)

ax[5].contour(kxp, kyp[:], growth[12, :, :]', colors=(43.0/255.0, 208.0/255.0, 0.0), levels=def_levs(growth[12, :, :]), alpha=1.0)
ax[3].contour(kxp, kyp[:], growth[10, :, :]', colors=(43.0/255.0, 208.0/255.0, 0.0), levels=def_levs(growth[10, :, :]), alpha=1.0)
ax[1].contour(kxp, kyp[:], growth[8, :, :]', colors=(43.0/255.0, 208.0/255.0, 0.0), levels=def_levs(growth[8, :, :]), alpha=1.0)


for axn in ax
    axn.set_xlim(-1.25, 1.25)
    axn.set_ylim(-1.25, 1.25)

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
cbar = fig.colorbar(pc_h0p25, ticks=[1e-2, 1, 1e2], cax=cbar_ax)
cbar.ax.tick_params(labelsize=lsize)
cbar.ax.set_title(L"\tau_{w} / \tau_{E}", fontsize=fsize, pad=20)
# cbar.set_ticklabels([L"E_\mathrm{max} \times 10^{-2}", L"E_\mathrm{max} \times 10^{-1}", L"E_\mathrm{max}"])

for (i,t) in enumerate(["(a)", "(d)", "(b)", "(e)", "(c)", "(f)"])
    ax[i].text(0.05, 0.95, t, transform=ax[i].transAxes, fontsize=fsize,
            ha="left", va="top")
end


savefig("./JPO_rev1_figs/timescale_ratio.png",bbox_inches="tight")

























