########################################################################
function bpca(X, Σ; maxiter=1, S=1)
########################################################################

    #---------------------------------------------
    # Set constants and report message
    #---------------------------------------------

    Q = 1

    JITTER = 1e-8

    D, N = size(X)

    Σdiag = Diagonal(Σ)

    @printf("BPCA: %d number of data items of dimension %d, projected to %d components\n", N, D, Q)


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
        #     ℓ +=  -0.5*logdet(C)-0.5*sum((X[:,n].-μ)'*(C\(X[:,n].-μ)))
        #
        # end
        #
        # return ℓ

        sum(logpdf(MvNormal(μ, W*W' + Σdiag*I + JITTER*I), X))

    end


    #---------------------------------------------
    function unpack(param)
    #---------------------------------------------

        local MARK = 0

        local W = reshape(param[MARK+1:MARK+D*Q], D, Q)

        MARK += D*Q

        local μ = reshape(param[MARK+1:MARK+D], D)

        MARK += D

        @assert(MARK == length(param))

        return W, μ

    end


    #---------------------------------------------
    function objective(param)
    #---------------------------------------------

        @assert(length(param) == D*Q + D)

        local W, μ = unpack(param)

        return marginalLogLikelihood(W, μ) + logPrior(W, μ)

    end

    #---------------------------------------------
    # Initialise with ppca estimates
    #---------------------------------------------

    W, μ, = ppca(X)


    #---------------------------------------------
    # Run optimiser and return posterior
    #---------------------------------------------

    posterior, logevidence = VI(objective, [vec(W); vec(μ)], S = S, Stest = 0*S, optimiser=:lbfgs, iterations = maxiter, show_every = 3)

    return posterior

end
