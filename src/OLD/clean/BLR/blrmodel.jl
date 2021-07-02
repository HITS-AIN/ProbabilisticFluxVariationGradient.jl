####################################################################
function blrmodel(x, flux, σ2, idx; maxiterml = 10, maxitervi = 10, maxrandom=10, seed = 1, show_trace = true, S=S, Stest=0)
####################################################################

    M = numcentres = 30

    JITTER = 1e-8

    #--------------------------------------------------------
    # Auxiliary
    #--------------------------------------------------------

    rg = MersenneTwister(seed)

    Σdiag = Diagonal(σ2)

    numF = length(unique(idx))

    @assert(numF == length(galaxyvector(3000.0)))

    numparam = 2*numF + 2

    @assert(all(unique(idx) .== collect(1:numF)))

    @show N = length(x)

    @assert(N == length(flux) == length(idx) == length(σ2))


    @printf("Running blrmodel with %d and %d filters\n", N, numF)
    @printf("\tNumber of free parameters is %d\n", numparam)
    @printf("\tVI has S=%d samples and runs for %d iterations\n", S, maxitervi)


    #--------------------------------------------------------
    function unpack(param)
    #--------------------------------------------------------

        @assert(length(param) == numparam)

        local a       = param[1:numF]
        local b       = param[numF+1:2*numF]
        local logrinv = param[2*numF+1]
        local logα    = param[2*numF+2]

        a, b, logrinv, logα

    end


    #--------------------------------------------------------
    function objective(param)
    #--------------------------------------------------------

        local a, b, logrinv, logα = unpack(param)

        -1.0 * logl(a = a, logrinv = logrinv, b = b, logα = logα)

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
    function logl(; a = a, b = b, logrinv = logrinv, logα = logα)
    #--------------------------------------------------------

        local Φ = PHI(x, centresongrid(x; numcentres=numcentres), exp(logrinv))

        local A = Diagonal(zeros(eltype(a), N))

        local bvec = zeros(eltype(a),N)

        blockvector!(bvec, idx, b)

        diagonalmatrix!(A, idx, a)

        local C = Σdiag + (A*Φ*Φ'*A) / exp(logα)

        logpdf(MvNormal(bvec, 0.5*(C+C')), flux)

    end


    #--------------------------------------------------------
    function loglΦ(; a = a, b = b, logα = logα, Φ = Φ)
    #--------------------------------------------------------

        local A = Diagonal(zeros(eltype(a), N))

        local bvec = zeros(eltype(a),N)

        blockvector!(bvec, idx, b)

        diagonalmatrix!(A, idx, a)

        local U = A*Φ / sqrt(exp(logα))

        local logd = logdet(Σdiag) + logdet(I + U'*(Σdiag\U))

        local Σinv = Σdiag\I

        local Cinv = Σinv - (Σinv*U)*((I + U'*Σinv*U)\(U'*Σinv))

         -0.5*logd -0.5*N*log(2π) -0.5*(bvec - flux)'*(Cinv*(bvec - flux))

    end

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

    a, b, logrinv, logα = unpack(result.minimizer)

    @show exp(logα) 1/exp(logrinv)

    #--------------------------------------------------------
    # Call VI but fix length scale parameter, logα and basis Φ
    #--------------------------------------------------------

    Φ = PHI(x, centresongrid(x; numcentres=numcentres), exp(logrinv))

    A = Diagonal(a[idx])

    bvec = b[idx]


    # @show  logl(; a = a, b = b, logrinv = logrinv, logα = logα)
    # @show loglΦ(; a = a, b = b, logα = logα, Φ = Φ)
    # return 0,0

    function newobjective(θ)
        @assert(length(θ) == numF*2)
        local a = @view θ[1:numF]
        local b = @view θ[numF+1:2*numF]
        loglΦ(; a = a, b = b, logα = logα, Φ = Φ)
    end

    posterior, logevidence = VI(newobjective, [a;b], S=S, Stest=Stest, show_every=10, iterations=maxitervi, optimiser = :lbfgs)

    return posterior

    #--------------------------------------------------------
    # Reconstruct curves on test points
    #--------------------------------------------------------

    xtest  = collect(LinRange(minimum(x), maximum(x), 200))

    Φ      = PHI(x, centresongrid(x; numcentres=numcentres), exp(logrinv))

    Φxtest = PHI(xtest, centresongrid(x; numcentres=numcentres), exp(logrinv))

    Sninv  = exp(logα)*I + Φ'*A*inv(Σdiag)*A*Φ     # 3.81

    Mn     = Sninv \ (Φ'*A*inv(Σdiag)*(flux .- bvec)) # 3.84


    # predicted mean

    μ = vec(Φxtest*Mn)

    figure(101)

    cla()

    clr = ["b","g","r","m","k","c"]


    for (i,l) in enumerate(unique(idx))

        plot(xtest, a[i]*μ .+ b[i], clr[i]*"-")

        aux = findall(idx .== l)

        plot(x[aux], flux[aux], clr[i]*".")

    end


    return mu, Sigma

end
