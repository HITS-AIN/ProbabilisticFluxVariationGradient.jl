####################################################################
function gpmodel(x, flux, σ2, idx; maxiter = 10, maxrandom = 100, seed = 1, show_trace = true)
####################################################################

    #--------------------------------------------------------
    # Auxiliary
    #--------------------------------------------------------

    rg = MersenneTwister(seed)

    JITTER = 1e-6

    N = length(x) # number of observations

    numF  = length(unique(idx)) # number of filters

    @assert(all(sort(unique(idx)) .== collect(1:numF)))

    @assert(N == length(flux) == length(idx) == length(σ2))

    Σdiag = Diagonal(σ2)

    @printf("Running gpmodel with %d observations and %d filters\n", N, numF)


    #--------------------------------------------------------
    # Pre-calculate distances
    #--------------------------------------------------------

    dist2 = pairwise(SqEuclidean(), reshape(x, 1, N), dims=2)

    K     = zeros(N, N)


    #--------------------------------------------------------
    function unpack(param)
    #--------------------------------------------------------

        @assert(length(param) == 2*numF + 1)

        local a     = param[0*numF+1:1*numF]
        local b     = param[1*numF+1:2*numF]
        local logs  = param[2*numF + 1]

        return a, b, logs

    end

    #--------------------------------------------------------
    function objective(param)
    #--------------------------------------------------------

        local a, b, logs = unpack(param)

        -1.0 * logl(a = a, logs = logs, b = b)

    end


    #--------------------------------------------------------
    function blockvector(idx, b)
    #--------------------------------------------------------

        local bv = zeros(eltype(b), N)

        for i in 1:numF
            bv[findall(idx .== i)] .= b[i] # this should be changes, too slow
        end

        return bv

    end


    #--------------------------------------------------------
    function logl(; a = a, b = b, logs = logs)
    #--------------------------------------------------------

        calculatekernelmatrix!(K, dist2, rbf, [0.0; logs])

        local A = Diagonal(blockvector(idx, a))

        local C = (A' * K * A) + Σdiag

        local C = (C.+C')*0.5 + JITTER*I

        local L = try

            cholesky(C).L

        catch err

            if isa(err, PosDefException)

                @warn("Cholesky failed, matrix not posdef. Return -Inf.")

                return -Inf

            else

                throw(err)

            end

        end


        local bvec    = blockvector(idx, b)

        local logdetC = 2*sum(log.(diag(L)))

        local yTCinvy = dot( (flux .- bvec), L'\(L\(flux .- bvec)) ) # this can be improved

        return - 0.5*logdetC - 0.5*yTCinvy - 0.5*N*log(2.0π)

    end


    #--------------------------------------------------------
    # Initialise GP hyperparameters with random search
    # and few iterations
    #--------------------------------------------------------

    initsolutions       = [randn(rg, 2*numF+1)*5 for i=1:maxrandom]
    initialfitness      = @showprogress "\t init random search " map(objective, initsolutions)
    bestinitialsolution = initsolutions[argmin(initialfitness)]

    #--------------------------------------------------------
    # Call optimiser
    #--------------------------------------------------------

    opt    = Optim.Options(iterations = maxiter, show_trace = show_trace, show_every = 100)
    result = Optim.optimize(objective, bestinitialsolution, NelderMead(), opt)

    # opt = NLopt.Opt(:LN_BOBYQA, length(bestinitialsolution))
    #
    # opt.xtol_rel = 1e-8
    #
    # COUNTER, BESTF = 0, Inf
    #
    # function myfunc(x, dummy)
    #
    #     COUNTER +=1
    #
    #     F = objective(x)
    #
    #     BESTF = min(F, BESTF)
    #
    #     mod(COUNTER, 100)==0 ? @printf("Iter %d f=%e\n", COUNTER, BESTF) : nothing
    #
    #     return F
    #
    # end
    #
    # opt.min_objective = myfunc
    #
    # (_minf, minx, _ret) = NLopt.optimize(opt, bestinitialsolution)


    #--------------------------------------------------------
    # Retrieve optimised parameters
    #--------------------------------------------------------

    a, b, logs = unpack(result.minimizer)




    #--------------------------------------------------------
    # Reconstruct curves
    #--------------------------------------------------------
    calculatekernelmatrix!(K, dist2, rbf, [0.0; logs])

    K = K + JITTER*I

    bvec = blockvector(idx, b)
    A    = Diagonal(blockvector(idx, a))
    L    = Σdiag\I
    Λ    = K\I

    Σ = (Λ + A'*L*A)\I

    μ = Σ * (A' * L * (flux .- bvec) )


    figure(101)

    cla()

    clr = ["b","g","r","k","y","m"]

    for (i,l) in enumerate(unique(idx))

        aux = findall(idx .== l)

        plot(t[aux], flux[aux], clr[i]*".", alpha=0.5)

        plot(t, b[i] .+ a[i]*μ, clr[i]*".")

    end





    # xtest = collect(LinRange(minimum(x), maximum(x), 200))
    #
    # # dimensions: N × Ntest
    # k = calculatekernelmatrix(x, xtest, rbf, [0.0;logs])

    # Ntest × 1
    # c = calculatekernelmatrix(xtest, xtest, rbf, uθ)

    # figure(102)
    #
    # cla()
    #
    # μ = k' * ((A'*K*A + σ2*I) \ (flux .- bvec))
    #
    # for (i,l) in enumerate(unique(idx))
    #
    #     plot(xtest, a[i]*μ .+ b[i], clr[i]*"-")
    #
    #     aux = findall(idx .== l)
    #
    #     plot(t[aux], flux[aux], clr[i]*".", alpha=0.5)
    #
    # end

    @show a, b, logs

end
