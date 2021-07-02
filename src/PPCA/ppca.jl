########################################################################
function ppca(X; Q=1)
########################################################################

    D, N = size(X)

    μ = vec(mean(X, dims=2))

    JITTER = 1e-8

    @printf("Running PPCA for %d number of data items of dimension %d, projected to %d components\n", N, D, Q)

    @printf("\t Fixing jitter to %f\n", JITTER)


    #---------------------------------------------
    function marginalLogLikelihood(W, σ)
    #---------------------------------------------

        sum(logpdf(MvNormal(μ, W*W' + σ*σ*I + JITTER*I), X))

    end


    #---------------------------------------------
    function unpack(param)
    #---------------------------------------------

        local W = reshape(param[1:end-1], D, Q)

        local σ = exp(param[end]) + 1e-8

        return W, σ

    end


    #---------------------------------------------
    function objective(param)
    #---------------------------------------------

        @assert(length(param) == D*Q + 1)

        local W, σ = unpack(param)

        try

            return -1.0 * marginalLogLikelihood(W, σ)

        catch err

            if isa(err, PosDefException)

                @warn("covariance is not positive definite, returning Inf")

                return Inf

            else

                throw(err)

            end

        end

    end

    #---------------------------------------------
    # Run optimiser
    #---------------------------------------------

    opt    = Optim.Options(show_trace = true, iterations = 100_000)

    result = optimize(objective, [randn(D*Q); randn()*3], NelderMead())# LBFGS(), opt, autodiff=:forward)

    W, σ   = unpack(result.minimizer)

    # @show W, μ, σ
    #
    # #---------------------------------------------
    # # Define projections
    # #---------------------------------------------
    #
    # M = W'*W  + σ*σ*I # eq. (12.41) in Bishop
    #
    # proj(x::Array{T,1} where T<:Real) = vec(M \ (W'*(x - μ)))[1]
    #
    # fwd(z::Real) = vec(W*z + μ)

    return W, μ, σ#, proj, fwd

end
