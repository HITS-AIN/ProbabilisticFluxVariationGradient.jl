#
# Calculates numerically:∫ N(t⋅x | μ, Σ) dt
# (see also function analyticalintegraloft)
#
function numericalintegraloft(x, μ, Σ)

    g = MvNormal(μ, Σ)

    hquadrature(t -> pdf(g, t*x), -150, 150)[1]

end


#
# Calculates the integral: ∫ N(t⋅x | μ, Σ) dt
# (see also function loganalyticalintegraloft)
#
function analyticalintegraloft(x, μ, Σ)

    # see https://en.wikipedia.org/wiki/Gaussian_integral#Generalizations

    D = length(μ) ; @assert(size(Σ, 1) == size(Σ, 2) == D == length(x))

    a =  0.5*x'*(Σ\x)

    b =  x'*(Σ\μ)

    c = -0.5*μ'*(Σ\μ)

    return 1/sqrt((2π)^D*det(Σ)) * sqrt(π/a) * exp(b^2/(4*a)+c)


end


#
# Calculates: log ∫ N(t⋅x | μ, Σ) dt
# (see also function analyticalintegraloft)
#
function loganalyticalintegraloft(x, μ, Σ)

    # see https://en.wikipedia.org/wiki/Gaussian_integral#Generalizations

    D = length(μ) ; @assert(size(Σ, 1) == size(Σ, 2) == D == length(x))

    a =  0.5*x'*(Σ\x)

    b =  x'*(Σ\μ)

    c = -0.5*μ'*(Σ\μ)

    return - 0.5*D*log(2π) - 0.5*logdet(Σ) + log(sqrt(π/a)) + (b^2/(4*a)+c)

end


function difficultintegral4expectation(post, x, S)

    rg = MersenneTwister(1)

    bcandidates = [rand(rg, post)[4:6] for _ in 1:S]

    μ_a_given_b, Σ_a_given_b = gaussconditional(mean(post), cov(post), [1;2;3], [4;5;6])

    sub(b) = analyticalintegraloft(x-b, μ_a_given_b(b), Σ_a_given_b)

    return mean(map(sub, bcandidates))

end



function difficultintegral5(post, numsamples, Trange, Srange)

    # Fix random seed
    rg = MersenneTwister(1)

    # Draw samples for b
    bcandidates = [rand(rg, post)[4:6] for _ in 1:numsamples]

    # Calculate conditional Gaussian
    μ_a_given_b, Σ_a_given_b = gaussconditional(mean(post), cov(post), [1;2;3], [4;5;6])

    aux(x, b) = analyticalintegraloft(x-b, μ_a_given_b(b), Σ_a_given_b)

    # Store here results
    TS = zeros(length(Trange), length(Srange))

    # Report progress
    pr = Progress(length(Trange))


    # Loop over temperatures
    for i in 1:length(Trange)

        # generate galaxy vector
        obsgal = observedunitgalaxyvector(Trange[i])

        for j in 1:length(Srange)

            # scale galaxy vector
            x = obsgal * Srange[j]

            # calculate density
            TS[i, j] = mean(map(b -> aux(x, b), bcandidates))

        end


        # Update progress bar

        bestTsofar = Trange[argmax(vec(sum(TS,dims=2)))]

        ProgressMeter.next!(pr; showvalues = [(:Current_Temperature, Trange[i]), (:Best_so_far, bestTsofar)])

    end

    return TS

end


function difficultintegral6(post, numsamples, Trange)

    # Fix random seed
    rg = MersenneTwister(1)

    # Draw samples for b
    bcandidates = [rand(rg, post)[4:6] for _ in 1:numsamples]

    # Calculate conditional Gaussian
    μ_a_given_b, Σ_a_given_b = gaussconditional(mean(post), cov(post), [1;2;3], [4;5;6])

    aux(x, b) = loganalyticalintegraloft(x-b, μ_a_given_b(b), Σ_a_given_b)

    # Store here results
    TS = zeros(length(Trange))

    # Report progress
    pr = Progress(length(Trange))

    # Loop over temperatures
    Threads.@threads for i in 1:length(Trange)

        # generate galaxy vector
        obsgal = observedunitgalaxyvector(Trange[i])

        # define objective, remember we are minimising
        f = s -> -1.0 * logsumexp(map(b -> aux(obsgal * s, b), bcandidates))

        # approximate integral with value at mode
        TS[i] = exp(-1.0 * optimize(f, 0.001, 200).minimum)

        # Update progress bar
        bestTsofar = Trange[argmax(vec(sum(TS, dims=2)))]

        ProgressMeter.next!(pr; showvalues = [(:Best_so_far, bestTsofar)])

    end

    return TS

end


function difficultintegralVI(x, posterior, S, maxiter)

    function logp(a, b)
         logpdf(posterior, [a;b])
     end

    aux(param) = logp(param[1:3], param[4:end])

    μVI, ΣVI, negVI = VI(aux, randn(6), optimiser=LBFGS(), S=S, maxiter=maxiter)

    return μVI, ΣVI, negVI

end
