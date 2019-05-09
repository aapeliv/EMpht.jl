using HCubature
using Dates
using DelimitedFiles: readdlm, writedlm
using JSON
using LinearAlgebra
using OrdinaryDiffEq
using Random
using Statistics

include("phasetype.jl");

#BLAS.set_num_threads(1)

# Definition of a sample which we fit the phase-type distribution to.
struct Sample
    obs::Vector{Float64}
    obsweight::Vector{Float64}
    cens::Vector{Float64}
    censweight::Vector{Float64}
    int::Matrix{Float64}
    intweight::Vector{Float64}

    function Sample(obs::Vector{Float64}, obsweight::Vector{Float64},
            cens::Vector{Float64}, censweight::Vector{Float64},
            int::Matrix{Float64}, intweight::Vector{Float64})
        cond = all(obs .>= 0) && all(obsweight .> 0) && all(cens .>= 0) &&
                all(censweight .> 0) && all(int .>= 0) && all(intweight .> 0)
        if ~cond
            error("Require non-negativity of observations and positivity of weight")
        end
        new(obs, obsweight, cens, censweight, int, intweight)
    end
end

@inbounds @views function ode_observations!(du::Array{Float64}, u::Array{Float64}, fit::PhaseType, t::Float64)
    # dc = T * C + t * a
    du[:] = vec(fit.T * reshape(u, fit.p, fit.p) + fit.t * (fit.π' * exp(fit.T * t)))
end

function ode_censored!(du::AbstractArray{Float64}, u::AbstractArray{Float64}, fit::PhaseType, t::Float64)
    # dc = T * C + 1 * a
    a = fit.π' * exp(fit.T * t)
    du[:] = vec(fit.T * reshape(u, fit.p, fit.p) + ones(fit.p) * a)
end

function loglikelihoodcensored(s::Sample, fit::PhaseType)
    ll = 0.0

    for k = 1:length(s.obs)
        ll += s.obsweight[k] * log(pdf(fit, s.obs[k]))
    end

    for k = 1:length(s.cens)
        ll += s.censweight[k] * log(1 - cdf(fit, s.cens[k]))
    end

    for k = 1:size(s.int, 1)
        ll_k = log( fit.π' * (exp(fit.T * s.int[k,1]) - exp(fit.T * s.int[k,2]) ) * ones(fit.p) )
        ll += s.intweight[k] * ll_k
    end

    ll
end

function parse_settings(settings_filename::String)
    # Check the file exists.
    if ~isfile(settings_filename)
        error("Settings file $settings_filename not found.")
    end

    # Check the input file is a json file.
    if length(settings_filename) < 6 || settings_filename[end-4:end] != ".json"
        error("Require a settings file as 'filename.json'.")
    end

    # Read in the properties of this fit (e.g. number of phases, PH structure)
    println("Reading settings from $settings_filename")
    settings = JSON.parsefile(settings_filename, use_mmap=false)

    name = get(settings, "Name", basename(settings_filename)[1:end-5])
    p = get(settings, "NumberPhases", 15)
    ph_structure = get(settings, "Structure", p < 20 ? "General" : "Coxian")
    continueFit = get(settings, "ContinuePreviousFit", true)
    num_iter = get(settings, "NumberIterations", 1_000)
    timeout = get(settings, "TimeOut", 30)

    # Set the seed for the random number generation if requested.
    if haskey(settings, "RandomSeed")
    Random.seed!(settings["RandomSeed"])
    else
    Random.seed!(1)
    end

    # Fill in the default values for the sample.
    s = settings["Sample"]

    obs = haskey(s, "Uncensored") ? Vector{Float64}(s["Uncensored"]["Observations"]) : Vector{Float64}()
    cens = haskey(s, "RightCensored") ? Vector{Float64}(s["RightCensored"]["Cutoffs"]) : Vector{Float64}()
    int = haskey(s, "IntervalCensored") ? Matrix{Float64}(transpose(hcat(s["IntervalCensored"]["Intervals"]...))) : Matrix{Float64}(undef, 0, 0)

    # Set the weight to 1 if not specified.
    obsweight = length(obs) > 0 && haskey(s["Uncensored"], "Weights") ? Vector{Float64}(s["Uncensored"]["Weights"]) : ones(length(obs))
    censweight = length(cens) > 0 && haskey(s["RightCensored"], "Weights") ? Vector{Float64}(s["RightCensored"]["Weights"]) : ones(length(cens))
    intweight = length(int) > 0 && haskey(s["IntervalCensored"], "Weights") ? Vector{Float64}(s["IntervalCensored"]["Weights"]) : ones(length(int))

    s = Sample(obs, obsweight, cens, censweight, int, intweight)

    (name, p, ph_structure, continueFit, num_iter, timeout, s)
end

function initial_phasetype(name::String, p::Int, ph_structure::String, continueFit::Bool, s::Sample)
    # If there is a <Name>_phases.csv then read the data from there.
    phases_filename = string(name, "_fit.csv")

    if continueFit && isfile(phases_filename)
        println("Continuing fit in $phases_filename")
        phases = readdlm(phases_filename)
        π = phases[1:end, 1]
        T = phases[1:end, 2:end]
        if length(π) != p || size(T) != (p, p)
            error("Error reading $phases_filename, expecting $p phases")
        end
        t = -T * ones(p)

    else # Otherwise, make a random start for the matrix.
        println("Using a random starting value")
        if ph_structure == "General"
            π_legal = trues(p)
            T_legal = trues(p, p)
        elseif ph_structure == "Coxian"
            π_legal = 1:p .== 1
            T_legal = diagm(1 => ones(p-1)) .> 0
        elseif ph_structure == "GeneralisedCoxian"
            π_legal = trues(p)
            T_legal = diagm(1 => ones(p-1)) .> 0
        else
            error("Nothing implemented for phase-type structure $ph_structure")
        end

        # Create a structure using [0.1, 1] uniforms.
        t = (0.9 * rand(p) .+ 0.1)

        π = (0.9 * rand(p) .+ 0.1)
        π[.~π_legal] .= 0
        π /= sum(π)

        T = (0.9 * rand(p, p) .+ 0.1)
        T[.~T_legal] .= 0
        T -= diagm(0 => T*ones(p) + t)

        # Rescale t and T using the same scaling as in the EMPHT.c program.
        if length(s.obs) > min(length(s.cens), size(s.int, 1))
            scalefactor = median(s.obs)
        elseif size(s.int, 1) > length(s.cens)
            scalefactor = median(s.int[:,2])
        else
            scalefactor = median(s.cens)
        end

        t *= p / scalefactor / 10
        T *= p / scalefactor / 10
    end

    PhaseType(π, T)
end

function save_progress(name::String, s::Sample, fit::PhaseType, iter::Integer, start::DateTime, seed::Integer)
    ll = loglikelihoodcensored(s, fit)

    open(string(name, "_$(seed)_loglikelihood.csv"), "a") do f
        mins = (now() - start).value / 1000 / 60
        write(f, "$iter $ll $(round(mins; digits=4))\n")
    end

    writedlm(string(name, "_$(seed)_fit.csv"), [fit.π fit.T])

    return ll
end

function create_c_integrand(fit, y)
    return function (u)
        first = fit.π' * exp(fit.T * u)
        second = exp(fit.T * (y-u)) * fit.t
        # Construct the matrix of integrands using outer product
        # then reshape it to a vector.
        C = second * first
        return vec(C)
    end
end

function d_integrand(u, fit, y)
    # Compute the two vector terms in the integrand
    first = fit.π' * exp(fit.T * u)
    second = exp(fit.T * (y-u)) * ones(fit.p)

    # Construct the matrix of integrands using outer product
    # then reshape it to a vector.
    D = second * first
    vec(D)
end

chunk(xs, n) = collect(Iterators.partition(xs, ceil(Int, length(xs)/n)))

@inbounds @views function ode_exp!(du::Array{Float64}, u::Array{Float64}, fit::PhaseType, t::Float64)
    # dc = T * C + t * a
    du[:] = vec(fit.T * reshape(u, fit.p, fit.p))
end

@inbounds @views function B(weight::Float64, π::Vector{Float64}, b::Vector{Float64}, denom::Float64)
    return weight * (π .* b) / denom
end

@inbounds @views function Z(weight::Float64, C::Matrix{Float64}, denom::Float64)
    return weight * diag(C) / denom
end

@inbounds @views function N1(weight::Float64, T::Matrix{Float64}, C::Matrix{Float64}, p::Int64, denom::Float64)
    return weight * (T .* transpose(C) .* (1 .- Matrix{Float64}(I, p, p))) / denom
end

@inbounds @views function N2(weight::Float64, t::Vector{Float64}, a::Vector{Float64}, denom::Float64)
    return weight * (t .* a) / denom
end

@inbounds @views function inner_loop!(Bs::Vector{Float64}, Zs::Vector{Float64}, Ns::Matrix{Float64}, p::Int64, weight::Float64, C::Matrix{Float64}, fit::PhaseType, a::Vector{Float64}, b::Vector{Float64})
    denom = fit.π' * b :: Vector{Float64}
    Bs[:] .+= B(weight, fit.π, b, denom)
    Zs[:] .+= Z(weight, C, denom)
    Ns[:,1:p] .+= N1(weight, fit.T, C, p, denom)
    Ns[:,p+1] .+= N2(weight, fit.t, a, denom)
end

function conditional_on_obs!(fit::PhaseType, s::Sample, workers::Int64, Bs_w::Array{Vector{Float64}}, Zs_w::Array{Vector{Float64}}, Ns_w::Array{Matrix{Float64}})
    p = fit.p

    # Setup initial conditions.
    u0 = zeros(p*p)

    # Run the ODE solver.
    prob = ODEProblem(ode_observations!, u0, (0.0, maximum(s.obs)), fit)
    sol = solve(prob, Tsit5())

    exp_prob = ODEProblem(ode_exp!, Matrix{Float64}(I, p, p), (0.0, maximum(s.obs)), fit)
    exp_sol = solve(exp_prob, Tsit5())

    print(", chunking away...")

    probs = Threads.Atomic{Int64}(0)

    print(" done")

    cc = chunk(1:length(s.obs), workers)
    Threads.@threads for worker = 1:workers
    #for worker = 1:workers

        fill!(Bs_w[worker], 0.0)
        fill!(Zs_w[worker], 0.0)
        fill!(Ns_w[worker], 0.0)

        for k in cc[worker]
            weight = s.obsweight[k]

            #expTy = exp(fit.T * s.obs[k])
            expTy = exp_sol(s.obs[k])::Matrix{Float64}

            a = transpose(fit.π' * expTy)
            b = expTy * fit.t

            C = reshape(sol(s.obs[k]), p, p)


            if minimum(C) < 0
                Threads.atomic_add!(probs, 1)
                #println("C is less than 0... ", s.obs[k], ", ", maximum(C))
                (C,err) = hquadrature(create_c_integrand(fit, s.obs[k]), 0, s.obs[k], atol=1e-3, maxevals=500)
                C = reshape(C, p, p)
            end

            if sum(b) == 0.0
                #println("Ignoring observation with b = 0")
            else
                inner_loop!(Bs_w[worker], Zs_w[worker], Ns_w[worker], p, weight, C, fit, a, b)
            end
        end
    end

    return probs[]
end

function conv_int_unif(r, P, fit, t, beta, alpha)
    p = fit.p

    ϵ = 1e-3
    R = quantile(Poisson(r * t), 1-ϵ)

    if R > 50
        println("R is getting big: $R")
    end

    betas = Array{Float64}(undef, p, R+1)
    betas[:,1] = beta

    for u = 1:R
        betas[:,u+1] = P * betas[:,u]
    end

    poissPDFs = pdf.(Poisson(r*t), 1:R+1)

    alphas = Array{Float64}(undef, p, R+1)
    alphas[:, R+1] = poissPDFs[R+1] .* alpha'

    for u = (R-1):-1:0
        alphas[:, u+1] = alphas[:, u+2]' * P + poissPDFs[u+1] .* alpha'
    end

    Υ = zeros(p, p)
    for u = 0:R
        Υ += betas[:, u+1] * alphas[:, u+1]' ./ r
    end

    Υ
end

function conditional_on_cens!(s::Sample, fit::PhaseType, Bs::AbstractArray{Float64}, Zs::AbstractArray{Float64}, Ns::AbstractArray{Float64})
    p = fit.p
    K = size(s.int, 1)
    deltaTs = s.int[:,2] - s.int[:,1]

    barfs = zeros(p, K+1)
    tildefs = zeros(p, K)
    barbs = zeros(p, K+1)
    tildebs = zeros(p, K)
    ws = zeros(K+1)
    N = sum(s.intweight)
    U = 0

    barfs[:,1] = fit.π' * inv(-fit.T)
    barbs[:,1] = ones(p)

    for k = 1:K
        expTDeltaT = exp(fit.T * deltaTs[k])

        barfs[:,k+1] = barfs[:,k]' * expTDeltaT
        tildefs[:,k] = barfs[:,k] - barfs[:,k+1]
        barbs[:,k+1] = expTDeltaT * barbs[:,k]
        tildebs[:,k] = barbs[:,k] - barbs[:,k+1]

        U += fit.π' * tildebs[:,k]

        ws[k] = s.intweight[k] / (fit.π' * tildebs[:,k])
    end
    U += fit.π' * barbs[:,K+1]
    ws[K+1] = 0

    cs = zeros(p, K)
    cs[:,K] = (ws[K+1] - ws[K]) .* fit.π'
    for k = (K-1):-1:1
        cs[:,k] = cs[:,k+1]' * exp(fit.T * deltaTs[k+1]) + (ws[k+1] - ws[k]) .* fit.π'
    end

    H = zeros(p, p)
    r = 1.01 * maximum(abs.(diag(fit.T)))
    P = I + (fit.T ./ r)

    for k = 1:K
        H += ws[k] .* ones(p) * tildefs[:,k]' + conv_int_unif(r, P, fit, deltaTs[k], barbs[:,k], cs[:,k])
    end
    H += ws[K+1] .* ones(p) * barfs[:,K+1]'


    # Step 4
    for k = 1:K
        Bs[:] = Bs[:] + s.intweight[k] .* (fit.π .* tildebs[:,k]) ./ (fit.π' * tildebs[:,k])
        Ns[:,end] = Ns[:,end] + s.intweight[k] .* (tildefs[:,k] .* fit.t) ./ (tildefs[:,k]' * fit.t)
    end

    Zs[:] = Zs[:] + diag(H)
    Ns[:,1:p] = Ns[:,1:p] + fit.T .* (H') .* (1 .- Matrix{Float64}(I, p, p))
end


function em_iterate(name, s, fit, num_iter, timeout, test_run, seed)
    p = fit.p

    # Count the total of all weight.
    sumOfWeights = sum(s.obsweight) + sum(s.censweight) + sum(s.intweight)

    start = now()

    save_progress(name, s, fit, 0, start, seed)
    last_save = now()

    ll = 0

    workers = Threads.nthreads()

    # preallocate
    Bs_w = Array{Vector{Float64}}(undef, workers);
    Zs_w = Array{Vector{Float64}}(undef, workers);
    Ns_w = Array{Matrix{Float64}}(undef, workers);

    for worker in 1:workers
        Bs_w[worker] = zeros(p)
        Zs_w[worker] = zeros(p)
        Ns_w[worker] = zeros(p, p+1)
    end

    Bs = zeros(p)
    Zs = zeros(p)
    Ns = zeros(p, p+1)

    for iter = 1:num_iter
        fill!(Bs, 0.0)
        fill!(Zs, 0.0)
        fill!(Ns, 0.0)

        print("iteration ", iter)

        iter_start = now()

        ##  The expectation step!
        if length(s.obs) > 0
            probs = conditional_on_obs!(fit, s, workers, Bs_w, Zs_w, Ns_w)

            print(", copying...")

            for worker in 1:workers
                Bs += Bs_w[worker]
                Zs += Zs_w[worker]
                Ns += Ns_w[worker]
            end

            print(" done")

            print(", problems: ", probs, " (out of ", length(s.obs), ")")
        end

        if length(s.cens) > 0 || length(s.int) > 0
            conditional_on_cens!(s, fit, Bs, Zs, Ns)
        end


        ## The maximisation step!
        π_next = max.(Bs ./ sumOfWeights, 0)
        t_next = max.(Ns[:,end] ./ Zs, 0)
        t_next[isnan.(t_next)] .= 0

        T_next = zeros(p,p)::Matrix{Float64}
        for i=1:p
            T_next[i,:] = max.(Ns[i,1:end-1] ./ Zs[i], 0)
            T_next[i,isnan.(T_next[i,:])] .= 0
            T_next[i,i] = -(t_next[i] + sum(T_next[i,:]))
        end

        # Remove any numerical instabilities.
        π_next = max.(π_next, 0)
        π_next /= sum(π_next)

        fit = PhaseType(π_next, T_next, t_next)

        print(", took ", now() - iter_start, ", (total ", now() - start, ", average ", div(now() - start, iter), ")")

        if now() - last_save > Dates.Second(60)
            ll = save_progress(name, s, fit, iter, start, seed)
            last_save = now()
            println(", ll ", ll)
        else
            println()
        end

    end

    save_progress(name, s, fit, -1, start, seed)
end

function em(name, p, ph_structure, continueFit, num_iter, timeout, s, seed)
    println("name, p, ph_structure, continueFit, num_iter, timeout, seed = $((name, p, ph_structure, continueFit, num_iter, timeout, seed))")

    # Check we don't just have right-censored obs, since this blows things up.
    if length(s.obs) == 0 && length(s.cens) > 0 && length(s.int) == 0
        error("Can't just have right-censored observations!")
    end

    # If not continuing previous fit, remove any left-over output files.
    if ~continueFit
        rm(string(name, "_$(seed)_loglikelihood.csv"), force=true)
        rm(string(name, "_$(seed)_fit.csv"), force=true)
    end

    fit = initial_phasetype(name, p, ph_structure, continueFit, s)

    if p <= 10
        println("first pi is $(fit.π), first T is $(fit.T)\n")
    end

    em_iterate(name, s, fit, num_iter, timeout, false, seed)
end

function em(settings_filename::String)
    # Read in details for the fit from the settings file.
    name, p, ph_structure, continueFit, num_iter, timeout, s = parse_settings(settings_filename)

    for seed in 1:5
        Random.seed!(seed)
        em(name, p, ph_structure, continueFit, num_iter, timeout, s, seed)
    end
end