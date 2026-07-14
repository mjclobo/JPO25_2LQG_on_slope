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


# energies are: (1) BTEKE, (2) BCEKE, (3) EAPE; (4) CBC; (5) Tflat, (6) Ttopo; , (7) DBT, (8) DBC;
#               (9) NLBC2BT, (10) NLBTEKE, (11) NLBT2BC; (12) NLBVEKE (13) NLBCEAPE (14) resid

# layer2_Htot_4000.0_L_2pi4.449E+06_y_slope_h0_0.000E+00_beta3.116E-12_uni_shear_uni_strat_mu8.826E-08_kappa0.000E+00_nu0.000E+00_res1024_yr57.7.jld

# layer2_H_4000.0_L_4.449E+06_y_slope_h0_3.754E-05_beta0.000E+00_uni_shear1.000E+000.000E+001.000E+03
# _uni_strat1.025E+031.026E+031.000E+03__mu4.413E-08_kappa0.000E+00_nu4.464E+27_res1024_yr89.5.jld


# readdir(data_dir_psi_ot)




νstars = [10^-11, 10^-11, 10^-11, 10^-11, 10^-11, 10^-11, 10^-11 , 10^-13, 10^-13, 10^-13, 10^-13, 10^-13, 10^-13]

nus = νstars .* ((U[1]-U[end])/2) * (Lx/2/pi)^7
ν = nus[1]

function calc_wf(z_ind_vec, h_ind, mu_ind, data_dir, model_params)

    # one for loop to rule them all (psi_ot edition)
    i=h_ind  # bottom slope index
    j=1  # quad drag index
    k=mu_ind  # linear drag index


    h0 = (h0s[i], 0.0)
    kappa_loc = kappas[j]
    mu_loc = mus[k]

    model_params = redef_mu_kappa_topoPV_h0_nu(model_params, mu_loc, kappa_loc, (0., 0.), h0, nus[i])
    topo_PV, eta = define_topo(model_params)
    model_params = redef_mu_kappa_topoPV_h0_nu(model_params, mu_loc, kappa_loc, topo_PV, h0, nus[i])

    model_params = redef_mu_kappa_beta(model_params, mu_loc, kappa_loc, 0.)

    wf = nothing # zeros(Int(size(a["jld_data"]["psi_ot_slice"])[1]/2+1), size(a["jld_data"]["psi_ot_slice"][:,:,:,1:10:end])[4])
    cnt = 0

    ##################################
    first_yr = 90.5 # 45.5
    last_yr = 157.5   # 89.5

    n_window = 8

    years = collect(range(first_yr, last_yr, step=0.5))

    # 2 means 50% overlap
    l_window = 2*Int(floor((length(years)-1)/2/(n_window+1)))  # length of window in years
    fwd_step = l_window  # length of half window in INDEX OF years vector

    nt_per_year =  size(load(data_dir*jld_name(model_params, years[1]))["jld_data"]["psi_ot"], 4)
    nt_total = Int((last_yr - first_yr) * nt_per_year / 2)

    for z_ind in z_ind_vec
        println("zonal index: "*string(z_ind))
        println(now())
        # z_ind % 10 == 0 && @info "zonal index $z_ind"

        for i in range(1, Int(n_window))

            # psi_slice = zeros(1024,0)
            psi_slice = zeros(Nx, nt_total)
	    psi2_slice = zeros(Nx, nt_total)
            is = 1
            nt_loc = 0

            loc_yr_start = years[(i-1)*fwd_step + 1]
            loc_yr_end   = years[(i-1)*fwd_step + 1] + l_window

            # println("year start: "*string(loc_yr_start)*", year end: "*string(loc_yr_end))
            # psi_BC_xyt, v1_xyt, zeta1_xyt, dt = calc_Hovs(h_ind, mu_ind, data_dir, model_params, years[(i-1)*fwd_step + 1], years[(i-1)*fwd_step + 1] + l_window) # years[i*fwd_step])

            for yr in collect(range(loc_yr_start, loc_yr_end, step=0.5))
                a = load(data_dir*jld_name(model_params,yr))

                layer_no = 1

                nt_loc = size(a["jld_data"]["psi_ot"], 4)

                psi_slice[:,is:(is+nt_loc-1)] = a["jld_data"]["psi_ot"][:,z_ind,layer_no, :]
                psi2_slice[:,is:(is+nt_loc-1)] = a["jld_data"]["psi_ot"][:,z_ind,2, :]


                is+= nt_loc
            end

            psi_slice = psi_slice[:,1:is-1]

	    psi2_slice = psi2_slice[:,1:is-1]

            if isnothing(wf)

                # # mid_ind makes this a two-window average in time (zero overlap)
                # mid_ind = floor(Int, length(psi_slice[1,:])/2)
                # wf = 0.5 * (abs.(rfft(psi_slice[:, 1:mid_ind])) .+ abs.(rfft(psi_slice[:, mid_ind+1:Int(2*mid_ind)])))

                # no longer do two windows here, because we are doing 8 overlapping windows already
                wf = abs.(rfft(psi_slice))

		rat = abs.(rfft(psi_slice)) ./ abs.(rfft(psi2_slice))

            else

                # mid_ind = floor(Int, length(psi_slice[1,:])/2)
                # wf .+= 0.5 * (abs.(rfft(psi_slice[:, 1:mid_ind])) .+ abs.(rfft(psi_slice[:, mid_ind+1:Int(2*mid_ind)])))

                # no longer do two windows here, because we are doing 8 overlapping windows already
                wf .+= abs.(rfft(psi_slice))

		rat .+= abs.(rfft(psi_slice)) ./ abs.(rfft(psi2_slice))
            end

            cnt+=1

        end

    end

    wf = wf ./ cnt;

    rat = rat ./ cnt;

    return wf, rat
end


data_dir_psi_ot = "/scratch/cimes/ml1994/QG/julia/data/JPO_2LQG_on_slope_data/mu0p25_hov/"

zonal_ind = Int.(floor.(collect(range(1, Nx, 128))))

##

wf_m0p25_hm0p75, rat_m0p25_hm0p75 = calc_wf(zonal_ind, 4, 2, data_dir_psi_ot, model_params);  # zonal slice index, h_ind, mu_ind

nt = size(wf_m0p25_hm0p75)[2]
dt = a["jld_data"]["t"][6] - a["jld_data"]["t"][5]

# c = fftfreq(nt, 1/dt) ./ (grid.kr ./ (2 * pi))
fr = collect(fftshift(fftfreq(nt, 1/dt))) * 3600 * 24 ;



wf_out_m0p25_hm0p75 = Dict("w_f_spectrum" => wf_m0p25_hm0p75, "w_f_ratio" => rat_m0p25_hm0p75, "dt" => dt, "fr"=> fr)

jldsave("./wf_m0p25_hm0p75_total_EKE.jld"; wf_out_m0p25_hm0p75)




