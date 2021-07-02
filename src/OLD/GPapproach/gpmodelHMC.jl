####################################################################
function gpmodelHMC(x, flux, σ2, idx; maxiterml = 10000, n_samples=2_000, n_adapts =1_000, maxrandom=500, seed = 1, show_trace = true)
####################################################################


    #--------------------------------------------------------
    # Auxiliary
    #--------------------------------------------------------


    rg = MersenneTwister(seed)

    JITTER = 1e-8

    Σdiag = Diagonal(σ2) + JITTER*I

    prvlogs = NaN

    numF = length(unique(idx))

    # @assert(numF == length(galaxyline(3000.0)))

    numparam = 2*numF + 1

    @assert(all(unique(idx) .== collect(1:numF)))

    N = length(x)

    @assert(N == length(flux) == length(idx) == length(σ2))



    @printf("Running gpmodelHMC with %d and %d filters\n", N, numF)

    @printf("\tNumber of free parameters is %d\n", numparam)

    #--------------------------------------------------------
    # Pre-allocate arrays and pre-calculate distances
    #--------------------------------------------------------

    A    = Diagonal(zeros(N))

    bvec = zeros(N)

    K    = zeros(N, N)

    C    = zeros(N, N)

    prvlogs = NaN

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
    function logl(; a = a, b = b, logs = logs)
    #--------------------------------------------------------

        local K = rbf(dist2, [0.0; logs])
        local A = Diagonal(zeros(eltype(a), N))
        local bvec = zeros(eltype(a), N)


        for i in 1:N
                bvec[i] = b[idx[i]]

                A.diag[i] = a[idx[i]]
        end

        C = (A * K * A) .+ Σdiag ; C .= (C .+ C').*0.5

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

    # a = randn(rg, 6).+1; b = randn(rg, 6).+1
    # @show logl(a=a,b=b,logs=0.0) logl0(a=a,b=b,logs=0.0)
    # # @btime  $logl(a=$a,b=$b,logs=0.0)
    # # @btime $logl0(a=$a,b=$b,logs=0.0)
    # return 0,0

    #--------------------------------------------------------
    # Initialise GP hyperparameters with random search
    # and few iterations
    #--------------------------------------------------------

    initsolutions       = [randn(rg, numparam) for i=1:maxrandom]
    initialfitness      = @showprogress "Initial random search " map(objective, initsolutions)
    bestinitialsolution = initsolutions[argmin(initialfitness)]


    #--------------------------------------------------------
    # Call optimiser
    #--------------------------------------------------------

    opt    = Optim.Options(iterations = maxiterml, show_trace = show_trace, show_every=100)
    result = Optim.optimize(objective, bestinitialsolution, NelderMead(), opt)


    #--------------------------------------------------------
    # Retrieve optimised parameters
    #--------------------------------------------------------

    a, b, logs = unpack(result.minimizer)


    #--------------------------------------------------------
    # Inference via HMC
    #--------------------------------------------------------

    newobjective(x) = -1.0* objective([x; logs])

    # Choose parameter dimensionality and initial parameter value
    initial_θ = [a;b]; D = length(initial_θ)

    # Define the target distribution
    ℓπ(θ) = newobjective(θ)

    # Define a Hamiltonian system
    metric = DiagEuclideanMetric(D)
    hamiltonian = Hamiltonian(metric, ℓπ, ForwardDiff)

    # Define a leapfrog solver, with initial step size chosen heuristically
    initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
    integrator = Leapfrog(initial_ϵ)

    # Define an HMC sampler, with the following components
    #   - multinomial sampling scheme,
    #   - generalised No-U-Turn criteria, and
    #   - windowed adaption for step-size and diagonal mass matrix
    proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

    # Run the sampler to draw samples from the specified Gaussian, where
    #   - `samples` will store the samples
    #   - `stats` will store diagnostic statistics for each sample
    return samples, stats = sample(hamiltonian, proposal, initial_θ, n_samples, adaptor, n_adapts; progress=true)

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

        plot(t[aux], flux[aux], clr[i]*".", alpha=0.5)

        plot(t, b[i] .+ a[i]*μ, clr[i]*".")

    end


    # #--------------------------------------------------------
    # # Reconstruct curves on test points
    # #--------------------------------------------------------
    #
    #
    #
    #     xtest = collect(LinRange(minimum(x), maximum(x), 200))
    #
    #     # dimensions: N × Ntest
    #     k = calculatekernelmatrix(x, xtest, rbf, [0.0;logs])
    #
    #     # Ntest × 1
    #     c = calculatekernelmatrix(xtest, xtest, rbf, [0.0;logs])
    #
    #     figure(102)
    #
    #     cla()
    #
    #     # predicted mean
    #
    #     μ = k' * ((A'*K*A + Σdiag) \ (flux .- bvec))
    #
    #     # predicted covariance
    #
    #     #C
    #
    #     for (i,l) in enumerate(unique(idx))
    #
    #         plot(xtest, a[i]*μ .+ b[i], clr[i]*".-")
    #
    #         aux = findall(idx .== l)
    #
    #         plot(x[aux], flux[aux], clr[i]*".", alpha=0.5)
    #
    #     end


    return a, b

end
