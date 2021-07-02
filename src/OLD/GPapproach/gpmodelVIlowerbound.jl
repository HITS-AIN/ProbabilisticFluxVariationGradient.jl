####################################################################
function gpmodelVI(x, flux, σ2, idx; maxiterml = 10, maxitervi = 10, maxrandom=10, seed = 1, show_trace = true, S=S)
####################################################################

    #--------------------------------------------------------
    # Auxiliary
    #--------------------------------------------------------

    rg = MersenneTwister(seed)

    JITTER = 1e-8

    Σdiag = Diagonal(σ2) + JITTER*I

    numF = length(unique(idx))

    numparam = 2*numF + 1

    @assert(all(unique(idx) .== collect(1:numF)))

    N = length(x)

    @assert(N == length(flux) == length(idx) == length(σ2))



    @printf("Running gpmodelVIlowerbound with %d and %d filters\n", N, numF)
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

        bvec .= L\(flux .- bvec)

        local yTCinvy = sum(bvec.^2)

        return - 0.5*logdetC - 0.5*yTCinvy - 0.5*N*log(2.0π)

    end


    #--------------------------------------------------------
    function updateposteriorf(; a = a, b = b)
    #--------------------------------------------------------

        local bvec = b[idx]

        local A = Diagonal(a[idx])

        local ỹ = (flux .- bvec)

        # local Σf = (A'*(Σdiag\A) + K\I) \ I

        local Σf = (K*A)/(A*K*A + Σdiag)

        local μf = Σf * A' * (Σdiag \ ỹ)

        return μf, Σf

    end

    #--------------------------------------------------------
    function logl0(; a = a, b = b, μf = μf, Σf = Σf)
    #--------------------------------------------------------

        local bvec = b[idx]

        local A = Diagonal(a[idx])

        local ỹ = (flux .- bvec)

        local v = ỹ - A*μf

        - 0.5 * dot(v, Σdiag\v) - 0.5 * tr(A'*(Σdiag\A)*Σf)


        #
        # Keep for debugging purposes
        #
        # - 0.5 * (ỹ - A*μf)'*(Σdiag\(ỹ - A*μf)) - 0.5 * tr(A'*(Σdiag\A)*Σf)
        #

    end

    #--------------------------------------------------------
    function gradlogl0(; a = a, b = b, μf = μf, Σf = Σf)
    #--------------------------------------------------------

        local agrad = zeros(numF)
        local bgrad = zeros(numF)

        @inbounds for n in 1:N
            bgrad[idx[n]] += (flux[n] - b[idx[n]] - a[idx[n]]*μf[n])/Σdiag[n,n]
        end

        @inbounds for n in 1:N
            agrad[idx[n]] += (flux[n]*μf[n] - b[idx[n]]*μf[n] - a[idx[n]]*μf[n]*μf[n])/Σdiag[n,n]  - a[idx[n]]/Σdiag[n,n]*Σf[n,n]
        end

        return [agrad;bgrad]

    end

    # a, b, logs = randn(numF), randn(numF), 0.0
    # @show  logl(a=a, b=b, logs=logs)
    # calculatekernelmatrix!(K, dist2, rbf, [0.0; logs])
    # @show logl0(a=a, b=b)

    #--------------------------------------------------------
    # Initialise GP hyperparameters with multiple random
    # restarts and few iterations
    #--------------------------------------------------------

    opt = Optim.Options(iterations = 100, show_trace = false)

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

    a, b, logs = unpack(result.minimizer)


    #--------------------------------------------------------
    # Call VI but fix length scale parameter
    #--------------------------------------------------------

    calculatekernelmatrix!(K, dist2, rbf, [0.0; logs])

    posterior = MvNormal(zeros(2*numF), 1.0)

    logevidence = NaN

    for inner=1:15

        μf, Σf = updateposteriorf(; a = a, b = b)


        # display(ForwardDiff.gradient(a->logl0(; a = a, b = b, μf = μf, Σf = Σf), a))
        # display(ForwardDiff.gradient(b->logl0(; a = a, b = b, μf = μf, Σf = Σf), b))
        # display(gradlogl0(a = a, b = b, μf = μf, Σf = Σf))
        #
        # @assert(2 == 1)


        newobjective(x)     =     logl0(a = x[1:numF], b = x[numF+1:end], μf=μf, Σf=Σf)
        newobjectivegrad(x) = gradlogl0(a = x[1:numF], b = x[numF+1:end], μf=μf, Σf=Σf)

        posterior, logevidence = VI(newobjective,newobjectivegrad, mean(posterior), show_every=2, S=S, iterations=maxitervi, optimiser = LBFGS())

    end


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
