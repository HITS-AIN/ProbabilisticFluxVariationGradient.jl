"""
    x, randx, logev = noisyintersectionvi(; posterior = posterior, g = g, [σ2 = 1e-8*ones(length(g))])

`posterior` is the posterior distribution returned by bpca, of type ```Distributions.MvNormal```.

`g` ∈ ℜⁿ₊ is the candidate direction vector that is being tested, of type ```Array{Float64, 1}```.


The function returns:

(1) the average intersection point `x`.

(2) a function ```randx()``` that draws a sample intersection point.

(3) the ```logevidence```, a measure of how good an intersection the candidate direction vector is.
"""
function noisyintersectionvi(; posterior = posterior, g = g, σ2 = 1e-6*ones(length(g)))

    # check dimensions of arguments

    @assert(length(g) == round(Int, length(posterior)/2))

    # check that components of vector are positive

    @assert(all(g .> 0.0))

    # work with normalised vector

    gunit = g / norm(g)

    linemeansigma = marginaline(posterior, σ2)

    function logp(p)

        local μ, Σ = linemeansigma(p[2])

        logpdf(MvNormal(μ, Σ), p[1]*gunit)

    end


    # Initialise with ML estimate

    bestsolution = noisyintersectionML(; posterior = posterior, g = gunit, σ2 = σ2)

    # Call variational inference

    postp, logevidence = VI(x->logp(x), [bestsolution.+(i/10)*(randn(length(bestsolution))) for i=1:30], optimiser = NelderMead(), S = 750, inititerations = 50, iterations = 1000000, show_every = 25)

    pdfscale = Normal(mean(postp)[1], sqrt(cov(postp)[1,1]))

    randx = ()->rand(pdfscale)*gunit

    return mean(pdfscale)*gunit, randx, logevidence

end
