using DistributionsAD, AdvancedVI

########################################################################
function bpca(X, Σ; maxiter=1, S=1)
########################################################################

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

        @inbounds for n in 1:N

            local C = W*W' + Matrix(Diagonal(Σ[:,n])) + JITTER*I

            ℓ +=  -0.5*logdet(C)-0.5*sum((X[:,n].-μ)'*(C\(X[:,n].-μ)))

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

    W, μ, = ppca(X)


    #---------------------------------------------
    # Run optimiser
    #---------------------------------------------

    # options for initialisation
    opt0 = Optim.Options(show_trace = false, show_every = 5, iterations = 1)

    # options for optimisations
    opt  = Optim.Options(show_trace = false, show_every = 5, iterations = maxiter, g_tol=1e-6)

    mu, C, elbof = VIoverfitting(objective, [vec(W); vec(μ)], S = S, optimiser=LBFGS(), optimoptions = opt, initoptions = opt0)

    # @show elbof(mu, C, [randn(2D) for s=1:20_000])



    makeC(x) = (local C=reshape(x,2D,2D); cholesky(C*C'))
    getq(θ)  = TuringDenseMvNormal(θ[1:2D], makeC(θ[2D+1:end]))
    advi     = ADVI(30, 10000)
    qadvi    = vi(objective, advi, getq, randn(2D + 2D*2D))

    @show AdvancedVI.elbo(advi, qadvi, objective, 20_000)

    @show elbof(mu, C, [randn(2D) for s=1:20_000])

    return MvNormal(mu, C*C'),  elbof#, historyparameter

end
