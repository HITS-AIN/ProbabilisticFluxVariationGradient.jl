#######################################################################
function infertemperaturegrid(randpcaline, S)
#######################################################################

    #----------------------------------------------------
    # Sample PCA parameters that describe total flux
    #----------------------------------------------------

    param = [randpcaline() for s=1:S]

    W = [p[1] for p in param]

    μ = [p[2] for p in param]


    #----------------------------------------------------
    # Define function for VI
    #----------------------------------------------------

    function logp(T, scale)

        @assert(T >= mintemperature())

        @assert(T <= maxtemperature())

        local ℓ = zeros(eltype(T), S)

        local σ = 0.2

        local y = vec(scale * observedgalaxyvector(T))

        for s in 1:S

            local M = W[s]'*W[s] # + Matrix(Diagonal(σ.*σ))

            local proj(x) = vec(M \ (W[s]'*(x - μ[s])))[1]

            local fwd(z)  = vec(W[s]*z + μ[s])

            ℓ[s]    = logpdf(MvNormal(fwd(proj(y)), σ), y)

        end

        return logsumexp(ℓ)

    end





    #----------------------------------------------------
    # Initial mean for VI
    #----------------------------------------------------
    Trange = LinRange(mintemperature(), maxtemperature(), 500)
    Srange = LinRange(0.01, 50, 1000)

    gridparam = [[T;scale] for T in Trange, scale in Srange]

    G = @showprogress map(x->exp(logp(x[1],x[2])), gridparam)

    return Trange, vec(sum(G,dims=2))

end
