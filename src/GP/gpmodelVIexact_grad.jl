####################################################################
function gpmodelVI(x, flux, σ2, idx; logs=logs, maxiter = 10, seed = 1, show_trace = true)
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

    @printf("Running gpmodelVIexact_grad with %d and %d filters\n", N, numF)



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

    v²ₐ = v²ᵦ = 1000.0

    # Convenient calls to lower bound

    function lb(μα, Σα, μβ, Σβ, μf, Σf)

        complete_lower_bound(; y=flux, idx=idx, S=Σdiag, μα=μα, Σα=Σα, v²ₐ=v²ₐ, μβ=μβ, Σβ=Σβ, v²ᵦ=v²ᵦ, μf=μf, Σf=Σf, K=K)

    end


    function lb_grad(μα, Σα, μβ, Σβ, μf, Σf)

        complete_lower_bound_grad(; y=flux, idx=idx, S=Σdiag, μα=μα, Σα=Σα, v²ₐ=v²ₐ, μβ=μβ, Σβ=Σβ, v²ᵦ=v²ᵦ, μf=μf, Σf=Σf, K=K)

    end


    # Unpacks vectorised parameters

    function unpack(param)

        @assert(length(param) == numF*4)

        local μα =               param[1+0*numF:1*numF]

        # the parametrisation used here has implications on the calculation
        # of the gradient, see complete_lower_bound.jl and exact_expectations_grad.jl
        local Σα = Diagonal(exp.(param[1+1*numF:2*numF]))

        local μβ =               param[1+2*numF:3*numF]

        # same comment as above applies here
        local Σβ = Diagonal(exp.(param[1+3*numF:4*numF]))

        return μα, Σα, μβ, Σβ

    end


    # Constructs necessary matrices

    function construct_matrices(μα, Σα, μβ, Σβ)

        local Ā  = Diagonal(μα[idx])

        local Σa = Diagonal(Σα.diag[idx])

        local b̄  = μβ[idx]

        return Ā, Σa, b̄

    end


    # Calculate optimal posterior q(f)

    function optimal_qf(Ā, Σa, b̄)

        # local Σf = ((Ā*inv(Σdiag)*Ā + inv(Σdiag)*Σa) + inv(K))\I

        local invDiagPart = (Ā*(Σdiag\Ā) + Σdiag\Σa)\I #
                                                       # This is faster/more stable
        local Σf = invDiagPart*((invDiagPart + K)\K)   #

        local μf = Σf * Ā * (Σdiag \ (flux - b̄))

        return μf, Σf

    end



    function fg!(F,G, param)

      # do common computations here

      local μα, Σα, μβ, Σβ = unpack(param)

      local Ā, Σa, b̄ = construct_matrices(μα, Σα, μβ, Σβ)

      local μf, Σf = optimal_qf(Ā, Σa, b̄)


      if G != nothing
        # code to compute gradient here
        # writing the result to the vector G
        local gradα, gradΣα, gradβ, gradΣβ = lb_grad(μα, Σα, μβ, Σβ, μf, Σf)

        copyto!(G, -1.0 * [gradα; gradΣα; gradβ; gradΣβ])

      end

      if F != nothing
        # value = ... code to compute objective function
        local value = -1.0 * lb(μα, Σα, μβ, Σβ, μf, Σf)
        return value
      end

    end



    #--------------------------------------------------------
    # Initialise GP hyperparameters with random search
    # and few iterations
    #--------------------------------------------------------

    a0 = [std(flux[findall(i.==idx)])  for i in unique(idx)]

    b0 = [mean(flux[findall(i.==idx)]) for i in unique(idx)]

    initsolutions   = [[a0.+3*randn(numF); log.(3*rand()*ones(numF)); b0.+3*randn(numF); log.(3*rand()*ones(numF))] for i=1:100]

    initialfitness  = @showprogress "\t init random search " map(x->fg!(0.0, nothing, x), initsolutions)

    solution = initsolutions[argmin(initialfitness)]

    # Use the LBFGS optimiser

    opt = Optim.Options(show_trace = show_trace, iterations = maxiter, show_every = 50, f_tol=0.0, g_tol=1e-8)

    result = Optim.optimize(Optim.only_fg!(fg!), solution, LBFGS(), opt)


    # retrieve result and return posterior of parameters describing
    # fitted line as object of type Distributions.MvNormal

    μα, Σα, μβ, Σβ = unpack(result.minimizer)

    return MvNormal([μα; μβ], Diagonal([Σα.diag; Σβ.diag]))

end
