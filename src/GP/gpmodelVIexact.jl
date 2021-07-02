####################################################################
function gpmodelVI(x, flux, σ2, idx; maxiter = 10, seed = 1, show_trace = true)
####################################################################


    #--------------------------------------------------------
    # Auxiliary
    #--------------------------------------------------------

    rg = MersenneTwister(seed)

    JITTER = 1e-8

    Σdiag = Diagonal(σ2)

    numF  = length(unique(idx))

    @assert(all(sort(unique(idx)) .== collect(1:numF)))

    N = length(x)

    @assert(N == length(flux) == length(idx) == length(σ2))

    @printf("Running gpmodelVIexact with %d and %d filters\n", N, numF)


    #--------------------------------------------------------
    # Initialise GP hyperparameters with ML estimation
    #--------------------------------------------------------

    a, b, logs = gpmodel(x, flux, σ2, idx; maxiter = 10_000, seed = 1, maxrandom=1000,show_trace = show_trace)


    #--------------------------------------------------------
    # Calculate kernel matrix and keep fixed!
    #--------------------------------------------------------

    dist2 = pairwise(SqEuclidean(), reshape(x, 1, N), dims=2)

    K1 = zeros(N, N)

    calculatekernelmatrix!(K1, dist2, rbf, [0.0; logs])

    K1 = K1 + JITTER * I

    K = PDMat((K1 + K1') / 2) # This helps us speed calculations


    #--------------------------------------------------------------------
    # Define functions for calling lower bound.
    # Actual calculation takes place in function "complete_lower_bound"
    #--------------------------------------------------------------------


    # Auxiliary function that calls the complete lower bound

    function lb(μα, Σα, μβ, Σβ, μf, Σf)

        local v²ₐ = v²ᵦ = 1000.0

        complete_lower_bound(; y=flux, idx=idx, S=Σdiag, μα=μα, Σα=Σα, v²ₐ=v²ₐ, μβ=μβ, Σβ=Σβ, v²ᵦ=v²ᵦ, μf=μf, Σf=Σf, K=K)

    end


    # Unpacks vectorised parameters

    function unpack(param)

        @assert(length(param) == numF*4)

        local μα =               param[1+0*numF:1*numF]

        local Σα = Diagonal(exp.(param[1+1*numF:2*numF]))

        local μβ =               param[1+2*numF:3*numF]

        local Σβ = Diagonal(exp.(param[1+3*numF:4*numF]))

        return μα, Σα, μβ, Σβ

    end


    # Constructs necessary matrices

    function negative_lb_helper(param)

        local μα, Σα, μβ, Σβ = unpack(param)

        local Ā  = Diagonal(μα[idx])

        local Σa = Diagonal(Σα.diag[idx])

        local b̄  = μβ[idx]

        # Calculate optimal posterior q(f)

        # local Σf = ((Ā*inv(Σdiag)*Ā + inv(Σdiag)*Σa) + inv(K))\I

        local invDiagPart = (Ā*(Σdiag\Ā) + Σdiag\Σa)\I #
                                                       # This is faster/more stable
        local Σf = invDiagPart*((invDiagPart + K)\K)   #

        local μf = Σf * Ā * (Σdiag \ (flux - b̄))

        -1.0 * lb(μα, Σα, μβ, Σβ, μf, Σf)

    end


    # Initial solution given by point estimates

    initsol = [a; log.(1.0*ones(numF)); b; log.(1.0*ones(numF))]


    # Use the LBFGS optimiser

    # opt = Optim.Options(show_trace = show_trace, iterations = maxiter, show_every = 1)
    #
    # result = Optim.optimize(negative_lb_helper, initsol, LBFGS(), opt, autodiff=:forward)

    # result = bboptimize(negative_lb_helper; SearchRange=(-10.0,10.0), NumDimensions = numF*4, Method=:xnes)


    opt = NLopt.Opt(:LN_BOBYQA, numF*4)

    opt.xtol_rel = 1e-8

    COUNTER, BESTF = 0, Inf

    function myfunc(x, dummy)

        COUNTER +=1

        F = negative_lb_helper(x)

        BESTF = min(F, BESTF)

        mod(COUNTER, 100)==0 ? @printf("Iter %d f=%e\n", COUNTER, BESTF) : nothing

        return F

    end

    opt.min_objective = myfunc

    (_minf, minx, _ret) = NLopt.optimize(opt, initsol)

    # Retrieve result and return posterior distribution of type Distributions.MvNormal

    μα, Σα, μβ, Σβ = unpack(minx)#result.minimizer)

    return MvNormal([μα; μβ], Diagonal([Σα.diag; Σβ.diag]))


end
