#######################################################################
function estimatetemperature1(W, μ, T)
#######################################################################

    g = unitgalaxyvector(T)

    function loss(x)

        norm(x - μ - proj(W, x - μ))^2 + norm(x - proj(g, x))^2

    end

    opt = Optim.Options(show_trace = false, iterations = 1_000_000)

    result = optimize(loss, randn(3)*10, LBFGS(), opt, autodiff=:forward)

    result.minimizer, result.minimum

end


#######################################################################
function estimatetemperature2(W, μ, T)
#######################################################################

    g = unitgalaxyvector(T)

    function loss(s)

        norm(s*g - μ - proj(W, s*g - μ))^2

    end

    opt = Optim.Options(show_trace = true, iterations = 1_000_000)

    result = optimize(loss, 0.001, 1000.0)

    result.minimizer, result.minimum

end


#######################################################################
function estimatetemperature3(posterior, σ2, T; S=100, seed=1)
#######################################################################

    # dimension of problem is D

    D = round(Int, length(posterior)/2)

    # generate samples

    rg = MersenneTwister(seed)

    aux(p) = (W = p[1:D], μ = p[D+1:2*D])

    P = [aux(rand(rg, posterior)) for _ in 1:S]

    # create galaxy vector

    g = unitgalaxyvector(T)

    # find "intersection" by optimising scale parameter

    ℓ = zeros(S) # pre-allocate

    function loss(s)

        for (i, (W, μ)) in enumerate(P)

            mu = proj(W, s*g - μ) + μ

            ℓ[i] = logpdf(MvNormal(mu, Diagonal(σ2)), s*g)

        end

        # we are minimising!
        -logsumexp(ℓ)

    end


    # set options and call optimiser

    opt = Optim.Options(show_trace = true, iterations = 1_000_000)

    result = optimize(loss, 0.001, 1000.0)

    # retrieve result and return


    result.minimizer, result.minimum

end




#######################################################################
function estimatetemperature4(posterior, σ2, T)
#######################################################################

    # dimension of problem is D

    linemeansigma = marginaline(posterior, σ2)

    # create galaxy vector

    g = unitgalaxyvector(T)

    function loss(p)

        local s, t = exp(p[1]), p[2]

        local μ, Σ = linemeansigma(t)

        -1.0 * logpdf(MvNormal(μ, Σ), s*g)

    end

    # set options and call optimiser
    bestfitness, bestsolution = Inf, randn(2)

    opt = Optim.Options(show_trace = false, iterations = 1_000_000)

    for repeat in 1:100

        result = optimize(loss, randn(2)*3, NelderMead(), opt)

        if result.minimum < bestfitness
            bestfitness, bestsolution  = result.minimum, result.minimizer
        end

    end

    # retrieve result and return

    bestsolution, bestfitness

end






#######################################################################
function marginaline(posterior, σ2)
#######################################################################

    # dimension of problem is D

    D = round(Int, length(posterior)/2)


    # marginals, conditionals

    μ, Σ = mean(posterior), cov(posterior)

    a_idx, b_idx = collect(1:D), collect(D+1:2D)

    μ_a = μ[a_idx]

    μ_b = μ[b_idx]

    Σ_aa = Σ[a_idx, a_idx]

    Σ_ab = Σ[a_idx, b_idx]

    Σ_ba = Σ[b_idx, a_idx]

    Σ_bb = Σ[b_idx, b_idx]

    Σ_a_given_b = Σ_aa - Σ_ab*(Σ_bb\Σ_ba)

    σ2Diag = Diagonal(σ2)

    function linemeansigma(t)

        local mu = μ_a * t + μ_b

        local K = t*(Σ_ab/Σ_bb) + I

        local Sigma = σ2Diag + t * Σ_a_given_b * t  +  K*Σ_bb*K'

        Sigma = (Sigma + Sigma')/2

        mu, Sigma

    end


end
