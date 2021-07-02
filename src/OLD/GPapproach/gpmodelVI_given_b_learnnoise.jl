####################################################################
function gpmodelVI(x=x, flux=flux, idx=idx, b=b, maxiterml = 10, maxitervi = 10, maxrandom=10, seed = 1, show_trace = true, S=S)
####################################################################

    #--------------------------------------------------------
    # Auxiliary
    #--------------------------------------------------------

    rg = MersenneTwister(seed)

    JITTER = 1e-8

    numF = length(unique(idx))

    numparam = numF  +  numF  +  1    +   1
    #           a      logσ2    logl    scaling factor logs

    @assert(all(unique(idx) .== collect(1:numF)))

    N = length(x)

    @assert(N == length(flux) == length(idx))



    @printf("Running gpmodelVI with %d and %d filters\n", N, numF)
    @printf("\tNumber of free parameters is %d\n", numparam)

    #--------------------------------------------------------
    # Pre-allocate arrays and pre-calculate distances
    #--------------------------------------------------------

    Σdiag = Diagonal(zeros(N))

    A     = Diagonal(zeros(N))

    bvec = blockvector!(bvec, idx, b)

    K    = zeros(N, N)

    C    = zeros(N, N)

    dist2 = pairwise(SqEuclidean(), reshape(x, 1, N), dims=2)


    #--------------------------------------------------------
    function unpack(param)
    #--------------------------------------------------------

        @assert(length(param) == numparam)

        local a     = param[1:numF]
        local logσ2 = param[1 + numF]
        local logl  = param[2 + numF]
        local logs  = param[3 + numF]

        a, logσ2, logl, logs

    end


    #--------------------------------------------------------
    function objective(param)
    #--------------------------------------------------------

        local a, logσ2, logl, logs = unpack(param)

        -1.0 * logp(a = a, logσ2 = logσ2, logl = logl, logs = logs)

    end


    #--------------------------------------------------------
    function blockvector!(bvec, idx, b)
    #--------------------------------------------------------

        for i in 1:N
            bvec[i] = b[idx[i]]
        end

    end


    #--------------------------------------------------------
    function diagonalmatrix!(A, idx, a)
    #--------------------------------------------------------

        for i in 1:N
            A.diag[i] = a[idx[i]]
        end

    end


    #--------------------------------------------------------
    function logp(; a = a, logσ2 = logσ2, logl = logl, logs = logs)
    #--------------------------------------------------------

        calculatekernelmatrix!(K, dist2, rbf, [0.0; logl])

        # blockvector!(bvec, idx, b)

        diagonalmatrix!(A, idx, a)

        diagonalmatrix!(Σdiag, idx, exp.(logσ2))

        C .= (A * K * A) .+ Σdiag

        C .= (C .+ C') .* 0.5

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


        local logdetC = 2.0*sum(log.(diag(L)))

        bvec .= L\(flux .- exp(logs)*bvec)

        local yTCinvy = sum(bvec.^2)

        return - 0.5*logdetC - 0.5*yTCinvy - 0.5*N*log(2.0π)

    end

    #--------------------------------------------------------
    function logp0(; a = a, logs = logs)
    #--------------------------------------------------------


        # local bvec = b[idx]
        local A    = Diagonal(a[idx])

        local C = (A * K * A) .+ Σdiag

        local C = (C .+ C') .* 0.5

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


        local logdetC = 2.0*sum(log.(diag(L)))

        local bvec = L\(flux .- exp(logs)*bvec)

        local yTCinvy = sum(bvec.^2)

        return - 0.5*logdetC - 0.5*yTCinvy - 0.5*N*log(2.0π)

    end


    # a, b, logl = randn(numF), randn(numF), 0.0
    # @show  logp(a=a, b=b, logl=logl)
    # calculatekernelmatrix!(K, dist2, rbf, [0.0; logl])
    # @show logp0(a=a, b=b)

    #--------------------------------------------------------
    # Initialise GP hyperparameters with multiple random
    # restarts and few iterations
    #--------------------------------------------------------

    opt = Optim.Options(iterations = 1000, show_trace = false)

    initoptimise(x) = Optim.optimize(objective, x, NelderMead(), opt).minimum

    initsolutions       = [randn(rg, numparam) for i=1:maxrandom]
    initialfitness      = @showprogress "Initial search with random start " map(initoptimise, initsolutions)
    bestinitialsolution = initsolutions[argmin(initialfitness)]


    #--------------------------------------------------------
    # Call optimiser
    #--------------------------------------------------------

    opt    = Optim.Options(iterations = maxiterml, show_trace = show_trace, show_every=100)
    result = Optim.optimize(objective, bestinitialsolution, NelderMead(), opt)


    #--------------------------------------------------------
    # Retrieve optimised parameters
    #--------------------------------------------------------

    a, logσ2, logl, logs = unpack(result.minimizer)

    @show exp.(logσ2)

    #--------------------------------------------------------
    # Call VI but fix length scale parameter and noise params
    #--------------------------------------------------------

    calculatekernelmatrix!(K, dist2, rbf, [0.0; logl])

    diagonalmatrix!(Σdiag, idx, exp.(logσ2))

    newobjective(x) = logp0(a = x[1:numF], logs = x[numF+1])


    opt = Optim.Options(show_trace = true, show_every = 10, iterations = maxitervi, g_tol=1e-6)

    mu, Sigma, nev = VI(newobjective, [a;logs], S=S, maxiter=maxitervi, optimiser = NelderMead(), optimoptions = opt)




    #--------------------------------------------------------
    # Reconstruct curves on latent points
    #--------------------------------------------------------

    calculatekernelmatrix!(K, dist2, rbf, [0.0; logl])

    K = (K+K')*0.5 + JITTER*I # numerical stability

    # blockvector!(bvec, idx, b)

    diagonalmatrix!(A, idx, a)

    L    = Σdiag\I

    Λ    = K\I

    Σ = (Λ + A'*L*A)\I

    μ = Σ * (A' * L * (flux .- bvec) )


    figure(101)

    cla()

    clr = ["b","g","r","m","k","c"]

    for (i,l) in enumerate(unique(idx))

        aux = findall(idx .== l)

        plot(t[aux], flux[aux], clr[i]*".", alpha=0.5)

        plot(t, b[i] .+ a[i]*μ, clr[i]*".")

    end

    plot(t, μ, "k.", alpha=0.5)
    @show maximum(μ)
    @show argmax(μ)
    @show t[argmax(μ)]
    plot(t[argmax(μ)], maximum(μ), "ko")

    # #--------------------------------------------------------
    # # Reconstruct mean curves on test points
    # #--------------------------------------------------------
    #
    # xtest = collect(LinRange(minimum(x), maximum(x), 200))
    #
    # # dimensions: N × Ntest
    # k = calculatekernelmatrix(x, xtest, rbf, [0.0; logl])
    #
    # # Ntest × 1
    # c = calculatekernelmatrix(xtest, xtest, rbf, [0.0; logl])
    #
    # figure(102)
    #
    # cla()
    #
    # # predicted mean
    #
    # μ = k' * ((A'*K*A + Σdiag) \ (flux .- bvec))
    #
    # for (i,l) in enumerate(unique(idx))
    #
    #     plot(xtest, a[i]*μ .+ b[i], clr[i]*".-")
    #
    #     aux = findall(idx .== l)
    #
    #     plot(x[aux], flux[aux], clr[i]*".", alpha=0.5)
    #
    # end

    # determine highest point
    # peak = abs.(a) .* maximum(μ) .+ b

    return MvNormal(mu, 0.5*(Sigma + Sigma')), exp.(logσ2), exp(logs)

end
