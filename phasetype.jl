## Definition of a phase-type distribution, and related functions.
using Distributions
using Statistics: cor, cov, median, std, quantile

import Base: minimum, maximum
import Distributions: cdf, insupport, logpdf, pdf
import Statistics: mean

macro check_args(D, cond)
    quote
        if !($(esc(cond)))
            throw(ArgumentError(string(
                $(string(D)), ": the condition ", $(string(cond)), " is not satisfied.")))
        end
    end
end

struct PhaseType <: ContinuousUnivariateDistribution
    # Defining properties
    π # entry probabilities
    T # transition probabilities

    # Derived properties
    t # exit probabilities
    p # number of phases

    function PhaseType(π, T, t, p)
        @check_args(PhaseType, all(π .>= zero(π[1])))
        @check_args(PhaseType, isapprox(sum(π), 1.0, atol=1e-4))
        @check_args(PhaseType, p == length(π) && p == length(t) && all(p .== size(T)))
        @check_args(PhaseType, all(t .>= zero(t[1])))
        @check_args(PhaseType, all(isapprox(t, -T*ones(p))))
        new(π, T, t, p)
    end
end

PhaseType(π, T) = PhaseType(π, T, -T*ones(length(π)), length(π))
PhaseType(π, T, t) = PhaseType(π, T, t, length(π))

minimum(d::PhaseType) = 0
maximum(d::PhaseType) = Inf
insupport(d::PhaseType, x::Real) = x > 0 && x < Inf

pdf(d::PhaseType, x::Real) = transpose(d.π) * exp(d.T * x) * d.t
logpdf(d::PhaseType, x::Real) = log(pdf(d, x))
cdf(d::PhaseType, x::Real) = 1 - transpose(d.π) * exp(d.T * x) * ones(d.p)

mean(d::PhaseType) = -transpose(d.π) * inv(d.T) * ones(d.p)

iscoxian(d::PhaseType) = all(isapprox.(d.T[diagm(0 => ones(d.p), 1 => ones(d.p-1)) .< 1], 0))
