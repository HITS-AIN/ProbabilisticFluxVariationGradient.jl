##################################################################
function multigp(x, y, σ2, idx; kernel=OU, maxiter = 10, maxrandom = 100, seed = 1, show_trace = true)
##################################################################

    #-----------------------------------------------------------------
    # Preliminary stuff, check dimensions, set random seed...
    #-----------------------------------------------------------------

    # check input parameters: there should be as many inputs x as outputs y as
    # many squared errors σ2 as many labels in idx
    @assert(length(x) == length(y) == length(σ2) == length(idx))

    # make sure labels run continuously from 1 to maximum number of labels
    @assert(all(unique(idx) .== collect(1:length(unique(idx)))))

    # instantiate random number generator
    rg = MersenneTwister(seed)

    # How much pertubation for numerical stability
    JITTER = 1e-8


    #-----------------------------------------------------------------
    # Split into seperate datasets according to identifier in `idx`
    #-----------------------------------------------------------------

    X =  [x[findall(idx .== i)] for i in unique(idx)]

    Y =  [y[findall(idx .== i)] for i in unique(idx)]

    Σ =  [Diagonal(σ2[findall(idx .== i)]) for i in unique(idx)]

    D = length(Y)

    #-----------------------------------------------------------------
    # pre-calculate distances and pre-allocate kernel matrices
    #-----------------------------------------------------------------

    SqDist = [pairwise(SqEuclidean(), reshape(x, 1, length(x))) for x in X]

    K = [zeros(length(x), length(x)) for x in X]


    #-----------------------------------------------------------------
    # Various tests for checking dimensions
    #-----------------------------------------------------------------

    for d in 1:D
        @assert(size(SqDist[d], 1) == size(SqDist[d], 2) == length(X[d]))
    end

    @assert(all(map(length, X) .== map(length, Y) .== map(s->size(s, 1), Σ) .== map(s->size(s, 2), Σ)))

    @assert(D == length(SqDist) == length(X) == length(Y) == length(Σ) == length(K))


    #-----------------------------------------------------------------
    # Report
    #-----------------------------------------------------------------

    @printf("Running multigp\n")

    @printf("\t There is a total of %d observations\n", length(x))

    @printf("\t There are a %d dimensions\n", D)



    #----------------------------------------------------
    # Auxiliary for sorting out parameters for GP
    #----------------------------------------------------

    function unpack(hyper)
        local logα = hyper[1:D]
        local logs = hyper[D+1]
        return logα, logs
    end


    #----------------------------------------------------
    # Define objective as marginal log-likelihood
    #----------------------------------------------------

    function objective(hyper)

        local logα, logs = unpack(hyper)

        for i in 1:length(K)
            calculatekernelmatrix!(K[i], SqDist[i], kernel, [logα[i]; logs])
            K[i] += Σ[i] + JITTER*I
        end


        f = zero(eltype(hyper))

        try

            # below is Eq. 2.30 in RW for D dimensions
            for i in 1:D
                C = cholesky(K[i]).L
                #f += -0.5*(Y[i]'*(K[i]\Y[i])) - 0.5*logdet(K[i])
                f += -0.5*sum(abs2.(C\Y[i])) - 0.5*2*sum(log.(diag(C)))
            end


        catch err

            if isa(err, PosDefException)

                @warn("Cholesky failed, matrix not posdef. Return -Inf.")

                return Inf

            else

                throw(err)

            end

        end

        -1.0 * f # we are minimising

    end



    #--------------------------------------------------------
    # Initialise GP hyperparameters with random search
    # and few iterations
    #--------------------------------------------------------

    initsolutions       = [randn(rg, 2*D+1)*7.5 for i=1:maxrandom]

    initoptimise(x)     = Optim.optimize(objective, x, NelderMead(),
                                Optim.Options(iterations=200)).minimum

    initialfitness      = @showprogress "\t init random search " map(initoptimise, initsolutions)

    bestinitialsolution = initsolutions[argmin(initialfitness)]


    #----------------------------------------------------
    # Optimise hyperparameters
    #----------------------------------------------------

    result = Optim.optimize(objective, bestinitialsolution, NelderMead(),
        Optim.Options(show_every=100, show_trace=show_trace, iterations=maxiter))


    # instantiate learnt matrix and observed variance parameter
    logα, logs = unpack(result.minimizer)

    for i in 1:length(K)

        calculatekernelmatrix!(K[i], SqDist[i], kernel, [logα[i]; logs])

        K[i] += Σ[i] + JITTER*I

    end

    #----------------------------------------------------
    # return prediction function
    #----------------------------------------------------

    function predictTest(xtest, d)

        # dimensions: N × Ntest
        k = calculatekernelmatrix(X[d], xtest, kernel,  [logα[d]; logs])

        # Ntest × 1
        c = calculatekernelmatrix(xtest, xtest, kernel, [logα[d]; logs])

        # predictive covariance and mean
        Σpred = c-k'*(K[d]\k)  + I*JITTER

        Σpred = 0.5*(Σpred+Σpred')

        # D × Ntest
        μpred = k'*(K[d]\Y[d])

        return μpred, Σpred

    end


    return logs, predictTest


end
