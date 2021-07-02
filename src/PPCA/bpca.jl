"""
    posterior = bpca(X, Σ; [maxiter=1000], [S=300])

`X` is a D×N matrix of flux observations.

`σ` is a D×N matrix of error measurements.

Arguments in brackets are optional.
Default number of iterations is 1200.
Default number of samples for automatic variational inference is 300.

Returns posterior distribution of PCA parameters returned as a ```Distributions.MvNormal``` type.
"""
function bpca(X, σ; maxiter=1000, S=300)

    Σ = σ.^2

    #---------------------------------------------
    # Set constants and report message
    #---------------------------------------------

    Q = 1

    JITTER = 1e-8

    D, N = size(X)


    @printf("BPCA: %d number of data items of dimension %d, projected to %d components\n", N, D, Q)


    #---------------------------------------------
    function logPrior(W, μ)
    #---------------------------------------------

        -1e-10*(sum(W.^2) + sum(μ.^2))

    end


    #---------------------------------------------
    function marginalLogLikelihood(W, μ)
    #---------------------------------------------

       local ℓ = zero(eltype(W))

       local WWT = W*W'

       @inbounds for n in 1:N

           # check eqs. (12.36), (12.43) in Bishop
           local C = WWT + Diagonal(Σ[:,n]) + JITTER*I

           ℓ +=  - 0.5*logdet(C) - 0.5*sum((X[:,n].-μ)'*(C\(X[:,n].-μ)))

       end

       return ℓ

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

    function initsol()
        local indices = map(x -> ceil(Int,x), rand(N)*N)
        local W, μ, = @suppress ppca(X[:,indices])
        [vec(W);μ]
    end


    #---------------------------------------------
    # Run optimiser and return posterior
    #---------------------------------------------

    posterior, logevidence = VI(objective, [initsol() for i=1:5], S = S, Stest = 0*S, optimiser=LBFGS(), inititerations=10, iterations = maxiter, show_every = 10)

    return posterior

end
