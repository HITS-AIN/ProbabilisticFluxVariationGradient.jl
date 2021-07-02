#######################################################################
function infertemperature(randpcaline, Spca, Sgal, maxiter)
#######################################################################

    #----------------------------------------------------
    # Sample PCA parameters that describe total flux
    #----------------------------------------------------

    param = [randpcaline() for s=1:Spca]

    W = [p[1] for p in param]

    μ = [p[2] for p in param]


    #----------------------------------------------------
    # Define function for VI
    #----------------------------------------------------

    function logp(T, scale)

        @assert(T >= mintemperature())

        @assert(T <= maxtemperature())

        local ℓ = zeros(eltype(T), Spca)

        local σ = 0.2

        local y = vec(scale * observedunitgalaxyvector(T))

        for s in 1:Spca

            local M = W[s]'*W[s] # + Matrix(Diagonal(σ.*σ))

            local proj(x) = vec(M \ (W[s]'*(x - μ[s])))[1]

            local fwd(z)  = vec(W[s]*z + μ[s])

            ℓ[s]    = logpdf(MvNormal(fwd(proj(y)), σ), y)

        end

        return logsumexp(ℓ)

    end


    function auxiliary(param)

        local T = converttemperature(param[1])

        local scale = convertscale(param[2])

        logp(T, scale)

    end


    #----------------------------------------------------
    # Initial mean for VI
    #----------------------------------------------------

    T0     = convertunctemperature(mintemperature() + rand()*(maxtemperature()-mintemperature()))

    scale0 = log(rand()*100.0)



    #--------------------------------------------------------
    # Call optimiser
    #--------------------------------------------------------

    opt    = Optim.Options(iterations = 10000, show_trace = true, show_every=2)
    result = Optim.optimize(x->-auxiliary(x), [T0;scale0], NelderMead(), opt)

    paramOpt = result.minimizer
    TOpt     = paramOpt[1]
    scaleOpt = paramOpt[2]


    @show converttemperature(TOpt)

    # #----------------------------------------------------
    # # Run VI
    # #----------------------------------------------------
    #
    # mu, Sigma, neglogev = VI(auxiliary, [TOpt; scaleOpt], S = Sgal, maxiter = maxiter, optimiser=NelderMead())
    #
    #
    # #----------------------------------------------------
    # # Output sampling function
    # #----------------------------------------------------
    #
    # jointPDFunc = MvNormal(mu, Sigma)
    #
    # function randobservedgalaxyvector()
    #
    #     local p = rand(jointPDFunc)
    #
    #     converttemperature(p[1]), convertscale(p[2])
    #
    # end
    #
    # return randobservedgalaxyvector

end
