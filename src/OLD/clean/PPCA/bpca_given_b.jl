########################################################################
function bpca_given_b(X, Σ, b; maxiter=1, S=1)
########################################################################

    #---------------------------------------------
    # Set constants, check dimensions and report
    #---------------------------------------------

    Q = 1

    JITTER = 1e-6

    D, N = size(X)

    @assert(all(size(X) .== size(Σ)))

    @assert(length(b) == D)

    numparam = D*Q + 1

    Σdiag = Diagonal(Σ)

    @printf("BPCA_given_b: %d number of data items of dimension %d, projected to %d components\n", N, D, Q)


    #---------------------------------------------
    function logPrior(W, μ)
    #---------------------------------------------

        -1e-10*(sum(W.^2) + sum(μ.^2))

    end


    #---------------------------------------------
    function marginalLogLikelihood(W, μ)
    #---------------------------------------------

        # local ℓ = zero(eltype(W))
        #
        # @inbounds for n in 1:N
        #
        #     local C = W*W' + Matrix(Diagonal(Σ[:,n])) + JITTER*I
        #
        #     ℓ +=  -0.5*logdet(C)-0.5*sum((X[:,n].-μ)'*(C\(X[:,n].-μ))) -0.5*D*log(2π)
        #
        # end

        # return ℓ

        sum(logpdf(MvNormal(μ, W*W' + Σdiag*I + JITTER*I), X))


    end


    #---------------------------------------------
    function unpack(param)
    #---------------------------------------------

        local MARK = 0

        local W = reshape(param[MARK+1:MARK+D*Q], D, Q)

        MARK += D*Q

        local s = exp(param[MARK+1])

        MARK += 1

        @assert(MARK == length(param))

        return W, s

    end


    #---------------------------------------------
    function objective(param)
    #---------------------------------------------

        @assert(length(param) == numparam)

        local W, s = unpack(param)

        return marginalLogLikelihood(W, s*b) + logPrior(W, s*b)

    end


    #---------------------------------------------
    # Initialise with ppca estimates
    #---------------------------------------------

    W, s, _σ = ppca_given_b(X, b)


    #---------------------------------------------
    # Run optimiser
    #---------------------------------------------

    opt = Optim.Options(show_trace = true, show_every = 10, iterations = maxiter, g_tol=1e-6)

    mu, Sigma, neglogev = VI(objective, [[vec(W); log(s)] .+ s*0.001*randn(numparam) for s in 1:30], S = S, optimiser=LBFGS(), optimoptions = opt)


    #---------------------------------------------
    # Return posterior and log evidence
    #---------------------------------------------

    return MvNormal(mu, Sigma), -1*neglogev


end
