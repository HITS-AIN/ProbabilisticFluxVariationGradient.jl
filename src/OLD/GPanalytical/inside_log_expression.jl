using Distributed

@everywhere using Distributions, Random, LinearAlgebra, Statistics

#-------------------------------
# Notation for random variables
#-------------------------------
#
# f is a vector of length N
# K is the covariance matrix of size N×N
# α and β are vectors of length F, i.e. number of filters
# A is a N×N diagonal matrix whose elements take values in α
# b is N length vector whose elements take vales in β
# S is a N×N diagonal matrix that holds the squared standard errors of the measurements
# y are the observations collected from all filters, has length N

#-------------------------------
# Notation for posteriors
#-------------------------------
#
# q(α) = 𝓝(α|μα,Σα)
# q(β) = 𝓝(β|μβ,Σβ)
# q(f) = 𝓝(f|μf,Σf)


# In this file we code and numerically verify the expectations of the following terms:
#
# log𝓝(y|Af + b, S) + log𝓝(f|0,K) + log𝓝(a|0,v²ₐI) + log𝓝(b|0,v²ᵦ)
#        (1)               (2)             (3)             (4)
#


function check_functions_of_terms(seed=0)

    # set dummy dimensions and instantiate dummy values for the random variables
    N   = 6
    F   = 2
    rg  = MersenneTwister(seed)

    idx = ceil.(Int, rand(rg, N)*F)

    v²ₐ = 0.3
    μα = rand(rg, F)
    Σα = Diagonal(rand(rg, F))*0.2
    α  = rand(rg, MvNormal(μα, Σα))
    A  = Diagonal(α[idx])

    v²ᵦ = 0.4
    μβ = rand(rg, F)
    Σβ = Diagonal(rand(rg, F))*0.2
    β  = rand(rg, MvNormal(μβ, Σβ))
    b  = β[idx]

    K  = randn(rg,N,N)*0.2 + 0.1*I; K=K*K'
    μf = rand(rg,N)
    Σf = randn(rg,N,N)*0.2 + 0.1*I; Σf = Σf*Σf'
    f  = rand(rg, MvNormal(μf, Σf))

    S = Diagonal(rand(rg,N) .+ 0.1)
    y = rand(rg, MvNormal(f, S))


    #--------------------------
    # define terms
    #--------------------------

    t1 = logpdf(MvNormal(A*f + b, S), y)          # (1)

    t2 = logpdf(MvNormal(zeros(N), K), f)         # (2)

    t3 = logpdf(MvNormal(zeros(F), sqrt(v²ₐ)), α) # (3)

    t4 = logpdf(MvNormal(zeros(F), sqrt(v²ᵦ)), β) # (4)

    @assert(t1 ≈ term1(y, A, f, b, S) ≈ term1_broken_down(y, A, f, b, S))
    @assert(t2 ≈ term2(f, K))
    @assert(t3 ≈ term3(α, v²ₐ))
    @assert(t4 ≈ term4(β, v²ᵦ))

    return true

end



function numerical_expectation_inside_log_of_lower_bound(; y=y, idx=idx, S=S, μα=μα, Σα=Σα, v²ₐ=v²ₐ, μβ=μβ, Σβ=Σβ, v²ᵦ=v²ᵦ, μf=μf, Σf=Σf, K=K, numsamples=numsamples)

    qα = MvNormal(μα, Σα)
    qβ = MvNormal(μβ, Σβ)
    qf = MvNormal(μf, Σf)

    @distributed (+) for i=1:numsamples

        local α = rand(qα)
        local β = rand(qβ)
        local f = rand(qf)
        local A = Diagonal(α[idx])
        local b = β[idx]

        (term1(y, A, f, b, S) + term2(f, K) + term3(α, v²ₐ) + term4(β, v²ᵦ)) / numsamples

    end


end



function test_expectations(numsamples=1000; seed=0)

    # set dummy dimensions and instantiate dummy values for the random variables
    N   = 6
    F   = 2
    rg  = MersenneTwister(seed)

    idx = ceil.(Int, rand(rg, N)*F)

    v²ₐ = 0.3
    μα = rand(rg, F)
    Σα = Diagonal(rand(rg, F))*0.2
    α  = rand(rg, MvNormal(μα, Σα))
    A  = Diagonal(α[idx])

    v²ᵦ = 0.4
    μβ = rand(rg, F)
    Σβ = Diagonal(rand(rg, F))*0.2
    β  = rand(rg, MvNormal(μβ, Σβ))
    b  = β[idx]

    K  = randn(rg,N,N)*0.2 + 0.0*I; K=K*K'
    μf = rand(rg,N)
    Σf = randn(rg,N,N)*0.2 + 0.0*I; Σf = Σf*Σf'
    f  = rand(rg, MvNormal(μf, Σf))

    S = Diagonal(rand(rg,N) .+ 0.1)
    y = rand(rg, MvNormal(f, S))

    # #--------------------------
    # # Define posteriors
    # #--------------------------
    #
    # qα = MvNormal(μα, Σα)
    # qβ = MvNormal(μβ, Σβ)
    # qf = MvNormal(μf, Σf)

        exact_expectation_inside_log_of_lower_bound(; y=y, idx=idx, S=S, μα=μα, Σα=Σα, v²ₐ=v²ₐ, μβ=μβ, Σβ=Σβ, v²ᵦ=v²ᵦ, μf=μf, Σf=Σf, K=K),
    numerical_expectation_inside_log_of_lower_bound(; y=y, idx=idx, S=S, μα=μα, Σα=Σα, v²ₐ=v²ₐ, μβ=μβ, Σβ=Σβ, v²ᵦ=v²ᵦ, μf=μf, Σf=Σf, K=K, numsamples=numsamples)

end


function inside_log_of_lower_bound(numsamples=1000; seed=0)

    # set dummy dimensions and instantiate dummy values for the random variables
    N   = 6
    F   = 2
    rg  = MersenneTwister(seed)

    idx = ceil.(Int, rand(rg, N)*F)

    v²ₐ = 0.3
    μα = rand(rg, F)
    Σα = Diagonal(rand(rg, F))*0.2
    α  = rand(rg, MvNormal(μα, Σα))
    A  = Diagonal(α[idx])

    v²ᵦ = 0.4
    μβ = rand(rg, F)
    Σβ = Diagonal(rand(rg, F))*0.2
    β  = rand(rg, MvNormal(μβ, Σβ))
    b  = β[idx]

    K  = randn(rg,N,N)*0.2 + 0.0*I; K=K*K'
    μf = rand(rg,N)
    Σf = randn(rg,N,N)*0.2 + 0.0*I; Σf = Σf*Σf'
    f  = rand(rg, MvNormal(μf, Σf))

    S = Diagonal(rand(rg,N) .+ 0.1)
    y = rand(rg, MvNormal(f, S))

    #--------------------------
    # Define posteriors
    #--------------------------

    qα = MvNormal(μα, Σα)
    qβ = MvNormal(μβ, Σβ)
    qf = MvNormal(μf, Σf)

    #--------------------------------
    # verify expectation of term (1)
    #--------------------------------

    numexpectationterm1 = @distributed (+) for i=1:numsamples

        local α = rand(qα)
        local β = rand(qβ)
        local f = rand(qf)
        local A = Diagonal(α[idx])
        local b = β[idx]

        term1(y, A, f, b, S) / numsamples
    end

    @show expectationterm1(;y=y, μα=μα, Σα=Σα, μβ=μβ, Σβ=Σβ, μf=μf, Σf=Σf, S=S, idx=idx), numexpectationterm1 # ✅


    #--------------------------------
    # verify expectation of term (2)
    #--------------------------------

    numexpectationterm2 = @distributed (+) for i=1:numsamples

        term2(rand(qf), K) / numsamples

    end

    @show expectationterm2(μf=μf, Σf=Σf, K=K), numexpectationterm2 # ✅


    #--------------------------------
    # verify expectation of term (3)
    #--------------------------------

    numexpectationterm3 = @distributed (+) for i=1:numsamples

        term3(rand(qα), v²ₐ) / numsamples

    end

    @show expectationterm3(μα=μα, Σα=Σα, v²ₐ=v²ₐ), numexpectationterm3  # ✅



    #--------------------------------
    # verify expectation of term (4)
    #--------------------------------

    numexpectationterm4 = @distributed (+) for i=1:numsamples

        term4(rand(qβ), v²ᵦ) / numsamples

    end

    @show expectationterm4(μβ=μβ, Σβ=Σβ, v²ᵦ=v²ᵦ), numexpectationterm4



end




#------------------ Individual terms and their expectations ------------------#

##########
# TERM 1 #
##########

@everywhere function term1(y, A, f, b, S)

    N = length(y)

    -0.5*logdet(S) - 0.5*N*log(2π) - 0.5*(y - A*f - b)'*(S\(y - A*f - b))

end

@everywhere function term1_broken_down(y, A, f, b, S)

    N = length(y)

    subterm0 = -0.5*logdet(S) - 0.5*N*log(2π)

    subterm1 = -0.5*dot(y, S\y)

    subterm2 = -0.5*dot(y, -S\A*f) * 2

    subterm3 = -0.5*dot(y, -S\b) * 2

    subterm4 =  -0.5*dot(A*f,S\A*f)

    subterm5 = -0.5*dot(b, S\A*f) * 2

    subterm6 = -0.5*dot(b, S\b)

    (subterm0 + subterm1 + subterm2 + subterm3 + subterm4 + subterm5 + subterm6)

end


##########
# TERM 2 #
##########

@everywhere function term2(f, K)
    N = length(f)
    -0.5*N*log(2π) -0.5*logdet(K) - 0.5*f'*(K\f)
end


##########
# TERM 3 #
##########

@everywhere function term3(α, v²ₐ)
    F = length(α)
    - 0.5*F*log(2π) - 0.5*F*log(v²ₐ) - 0.5*dot(α,α)/v²ₐ
end


##########
# TERM 4 #
##########

@everywhere function term4(β, v²ᵦ)
    F = length(β)
    - 0.5*F*log(2π) - 0.5*F*log(v²ᵦ) - 0.5*dot(β,β)/v²ᵦ
end
