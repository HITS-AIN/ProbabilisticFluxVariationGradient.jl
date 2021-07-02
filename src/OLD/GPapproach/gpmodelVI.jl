####################################################################
function gpmodelVI(x, flux, σ2, idx; maxiterml = 10, maxitervi = 10, maxrandom=10, seed = 1, show_trace = true, S=S,Stest=0)
####################################################################

    #--------------------------------------------------------
    # Auxiliary
    #--------------------------------------------------------

    rg = MersenneTwister(seed)

    JITTER = 1e-8

    Σdiag = Diagonal(σ2)

    numF = length(unique(idx))

    numparam = 2*numF + 1

    @assert(all(sort(unique(idx)) .== collect(1:numF)))

    N = length(x)

    @assert(N == length(flux) == length(idx) == length(σ2))



    @printf("Running gpmodelVI with %d and %d filters\n", N, numF)
    @printf("\tNumber of free parameters is %d\n", numparam)

    #--------------------------------------------------------
    # Pre-allocate arrays and pre-calculate distances
    #--------------------------------------------------------

    A    = Diagonal(zeros(N))

    bvec = zeros(N)

    K    = zeros(N, N)

    C    = zeros(N, N)

    dist2 = pairwise(SqEuclidean(), reshape(x, 1, N), dims=2)


    #--------------------------------------------------------
    function unpack(param)
    #--------------------------------------------------------

        @assert(length(param) == numparam)

        local a     = param[1:numF]
        local b     = param[numF+1:2*numF]
        local logs  = param[end]

        a, b, logs

    end


    #--------------------------------------------------------
    function objective(param)
    #--------------------------------------------------------

        local a, b, logs = unpack(param)

        -1.0 * logl(a = a, logs = logs, b = b)

    end


    #--------------------------------------------------------
    function blockvector!(bvec, idx, b)
    #--------------------------------------------------------

        @inbounds for i in 1:N
            bvec[i] = b[idx[i]]
        end

    end


    #--------------------------------------------------------
    function diagonalmatrix!(A, idx, a)
    #--------------------------------------------------------

        @inbounds for i in 1:N
            A.diag[i] = a[idx[i]]
        end

    end


    #--------------------------------------------------------
    function logl(; a = a, b = b, logs = logs)
    #--------------------------------------------------------

        calculatekernelmatrix!(K, dist2, rbf, [0.0; logs])

        blockvector!(bvec, idx, b)

        diagonalmatrix!(A, idx, a)

        C .= (A * K * A) .+ Σdiag

        C .= (C .+ C') .* 0.5 + JITTER*I

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

        bvec .= L\(flux .- bvec)

        local yTCinvy = sum(bvec.^2)

        return - 0.5*logdetC - 0.5*yTCinvy - 0.5*N*log(2.0π)

    end

    #--------------------------------------------------------
    function logl0(; a = a, b = b)
    #--------------------------------------------------------


        @inbounds local  bvec = b[idx]

        @inbounds local  A    = Diagonal(a[idx])

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

        local bvec = L\(flux .- bvec)

        local yTCinvy = sum(bvec.^2)

        return - 0.5*logdetC - 0.5*yTCinvy - 0.5*N*log(2.0π)

    end


    # a, b, logs = randn(numF), randn(numF), 0.0
    # @show  logl(a=a, b=b, logs=logs)
    # calculatekernelmatrix!(K, dist2, rbf, [0.0; logs])
    # @show logl0(a=a, b=b)

    #--------------------------------------------------------
    # Initialise GP hyperparameters with multiple random
    # restarts and few iterations
    #--------------------------------------------------------

    # opt = Optim.Options(iterations = 10, show_trace = false)
    #
    # initoptimise(x) = Optim.optimize(objective, x, NelderMead(), opt).minimum
    #
    # initsolutions       = [randn(rg, numparam)*10 for i=1:maxrandom]
    #
    # initialfitness      = @showprogress "Initial search with random start " map(initoptimise, initsolutions)
    #
    # bestinitialsolution = initsolutions[argmin(initialfitness)]

    initsolutions       = [randn(rg, numparam)*3 for i=1:1000]
    initialfitness      = map(objective, initsolutions)
    bestinitialsolution = initsolutions[argmin(initialfitness)]

    #--------------------------------------------------------
    # Call optimiser
    #--------------------------------------------------------

    opt    = Optim.Options(iterations = maxiterml, show_trace = show_trace, show_every=50)

    result = Optim.optimize(objective, bestinitialsolution, NelderMead(), opt)


    #--------------------------------------------------------
    # Retrieve optimised parameters
    #--------------------------------------------------------

    a, b, logs = unpack(result.minimizer)
    @show logs

    #--------------------------------------------------------
    # Call VI but fix length scale parameter
    #--------------------------------------------------------

    calculatekernelmatrix!(K, dist2, rbf, [0.0; logs])

    newobjective(x) = logl0(a = x[1:numF], b = x[numF+1:end])

    posterior, logevidence = VI(newobjective, [a;b], S=S, Stest=Stest, show_every=1, iterations=maxitervi, optimiser = LBFGS())

    # mu, Sigma, = VIdiag(newobjective, [a;b], S=S,  maxiter=maxitervi)
    # posterior = MvNormal(mu, Sigma)


    #--------------------------------------------------------
    # Reconstruct curves on latent points
    #--------------------------------------------------------

    calculatekernelmatrix!(K, dist2, rbf, [0.0; logs])

    K = (K+K')*0.5 + JITTER*I # numerical stability

    blockvector!(bvec, idx, b)

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

        PyPlot.plot(t[aux], flux[aux], clr[i]*".", alpha=0.5)

        PyPlot.plot(t, b[i] .+ a[i]*μ, clr[i]*".")

    end

    PyPlot.plot(t, μ, "k.", alpha=0.5)


    # #--------------------------------------------------------
    # # Reconstruct mean curves on test points
    # #--------------------------------------------------------
    #
    # xtest = collect(LinRange(minimum(x), maximum(x), 200))
    #
    # # dimensions: N × Ntest
    # k = calculatekernelmatrix(x, xtest, rbf, [0.0; logs])
    #
    # # Ntest × 1
    # c = calculatekernelmatrix(xtest, xtest, rbf, [0.0; logs])
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
    #peak = abs.(a) .* maximum(μ) .+ b

    return posterior

end
