using Distributions, LinearAlgebra, Random, ProgressMeter

function test_expectation_muASAmu(D=3, seed=1, numsamples=10_000)

    rg = MersenneTwister(seed)

    μα = randn(rg, D)

    Σα = rand(rg, D, D)*0.2; Σα = Σα*Σα' .+ 0.1

    Apdf = MvNormal(μα, Σα)

    S = Diagonal(rand(rg, D) .+ 2.0) +1.0I

    μf = randn(rg, D)


    f(A) = μf'*A'*inv(S)*A*μf

    g(A) = f(Diagonal(A))[1]



    #————————————————————————————————————————————————
    # This is the result we are after
    #————————————————————————————————————————————————
    # analytical expectation
    l = 0.0
    for i in 1:length(μα)
        l += (μα[i]*μα[i] + Σα[i,i]) * μf[i]^2 / S[i,i]
    end
    #————————————————————————————————————————————————


    # numerical expectation vs analytical
    mean(@showprogress map(g, [rand(Apdf) for _ in 1:numsamples])), l


end
