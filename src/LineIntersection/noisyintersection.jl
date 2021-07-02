#######################################################################
function noisyintersectionML(; posterior = posterior, g = g, σ2 = 1e-6*ones(length(g)))
#######################################################################

    # work with normalised vector

    gunit = g / norm(g)

    linemeansigma = marginaline(posterior, σ2)

    function loss(p)

        local μ, Σ = linemeansigma(p[2])

        -1.0 * logpdf(MvNormal(μ, Σ), p[1]*gunit)

    end

    # get parameters of mean line

    D = length(g)

    a = mean(posterior)[1:D]

    b = mean(posterior)[D+1:2*D]

    # find point on g that minimises distance to mean line

    aux1(s) = norm(s*gunit - b - proj(a, s*gunit - b))^2

    s0 = optimize(aux1, -1000.0, 1000.0).minimizer

    # find point on mean line that minimises distance to point on g we just found

    aux2(t) = norm(s0*gunit - (a*t + b))^2

    t0 = optimize(aux2, -1000.0, 1000.0).minimizer

    # find s, t starting from s0, t0 that minimise above log-likelihood

    opt    = Optim.Options(show_trace = false, iterations = 1_000_000)

    result = Optim.optimize(loss, [s0; t0], NelderMead(), opt).minimizer

end



"""
    randx = noisyintersectionvi(; posterior = posterior, g = g, [σ2 = 1e-6*ones(length(g))])

`posterior` is the posterior distribution returned by bpca, of type ```Distributions.MvNormal```.

`g` ∈ ℜⁿ₊ is the candidate direction vector that is being tested, of type ```Array{Float64, 1}```.

The function returns a function `randx()` that draws a sample intersection point.

"""
function noisyintersectionvi(; posterior = posterior, g = g, σ2 = 1e-6*ones(length(g)))

    @info("Scale parameter re-parametrised to enforce positivity")

    # check dimensions of arguments

    @assert(length(g) == round(Int, length(posterior)/2))

    # check that components of vector are positive

    @assert(all(g .> 0.0))

    # work with normalised vector

    gunit = g / norm(g)

    # instantiate posterior of lines

    linemeansigma = marginaline(posterior, σ2)

    # auxiliary functions for making positive parameters

    makepos(x)    = x^2

    invmakepos(x) = sqrt(x)

    # Initialise with ML estimate

    bestsolutionML = noisyintersectionML(; posterior = posterior, g = gunit, σ2 = σ2)

    # log-likleihood function for VI

    function logp(p)

        local μ, Σ = linemeansigma(p[2])

        logpdf(MvNormal(μ, Σ), makepos(p[1])*gunit)

    end

    # Get ML estimate as initial solution

    @printf("ML solution for scale is %f\n", bestsolutionML[1])

    bestsolutionML[1] = invmakepos(max(1e-3, bestsolutionML[1]))

    # Call variational inference

    postp, logevidence = VI(x -> logp(x), [bestsolutionML.+(i/10)*(randn(length(bestsolutionML))) for i=1:30], optimiser = NelderMead(), S = 500, inititerations = 50, iterations = 10_000, show_every = 25)

    # Retrieve results from VI and instantiate sampling function

    pdfscale = Normal(mean(postp)[1], sqrt(cov(postp)[1,1]))

    randscale = () -> makepos(rand(pdfscale))

    # xavg = mean([randscale() for i=1:10_000]) * gunit

    randx = () -> randscale() * gunit

    @printf("Log-evidence is %f (higher is better)\n", logevidence)

    return randx

end
