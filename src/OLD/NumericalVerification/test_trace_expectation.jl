using Distributions, LinearAlgebra, Random, ProgressMeter


function test_trace_expectation(D=3, seed=1, numsamples=10_000)

    rg = MersenneTwister(seed)

    μα = randn(rg, D)

    Σα = rand(rg, D, D)*0.2; Σα = Σα*Σα' .+ 0.1

    Apdf = MvNormal(μα, Σα)

    S = Diagonal(rand(rg, D) .+ 2.0) +1.0I

    K = randn(rg, D, D); K = K*K'


    #----------------------------------------

    f(A) = tr(A*inv(S)*A*K)

    g(A) = f(Diagonal(A))

    # verify first that expression below is equiv to trace
    Asample = Diagonal(rand(Apdf))
    @show f(Asample)
    @show sum((Asample).^2 .* diag(inv(S)) .* diag(K))
    @show tr(K*Asample*inv(S)*Asample) # switching order should not matter
    @show tr(Asample.^2*inv(S)*K) # switching order should not matter


    #————————————————————————————————————————————————
    # This is the result we are after
    #————————————————————————————————————————————————
    # analytical expectation
    l = 0.0
    for i in 1:length(μα)
        l += (μα[i]*μα[i] + Σα[i,i]) * K[i,i] / S[i,i]
    end
    #————————————————————————————————————————————————


    # alternative analytical expectation in matrix form
    l2 = tr(Diagonal(μα.^2 .+ diag(Σα))*inv(S)*K)

    
    # numerical expectation vs analytical
    mean(@showprogress map(g, [rand(Apdf) for _ in 1:numsamples])), l, l2

end
