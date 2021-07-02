########################################################################
function ppca_given_b(X, b; Q=1)
########################################################################

    D, N = size(X)

    @assert(length(b) == D)

    JITTER = 1e-8

    numparam = D*Q        +   1     +  1
    #    linear mapping    scale    noise

    @printf("Running PPCA_given_b for %d number of data items of dimension %d, projected to %d components\n", N, D, Q)


    #---------------------------------------------
    function marginalLogLikelihood(W, μ, σ)
    #---------------------------------------------

        sum(logpdf(MvNormal(μ, W*W' + σ*σ*I + JITTER*I), X))

    end


    #---------------------------------------------
    function unpack(param)
    #---------------------------------------------

        local W = reshape(param[1:end-2], D, Q)

        local s = exp(param[end-1])

        local σ = exp(param[end])

        return W, s, σ

    end


    #---------------------------------------------
    function objective(param)
    #---------------------------------------------

        @assert(length(param) == numparam)

        local W, s, σ = unpack(param)

        return -1.0 * marginalLogLikelihood(W, s * b, σ)

    end


    #---------------------------------------------
    # Run optimiser
    #---------------------------------------------

    opt    = Optim.Options(show_trace = true, iterations = 10_000)

    result = optimize(objective, [randn(D*Q); randn()*3; randn()*3], LBFGS(), opt, autodiff=:forward)

    W, s, σ = unpack(result.minimizer)


    #---------------------------------------------
    # Return results
    #---------------------------------------------

    return W, s, σ

end
