using Random, HCubature, QuadGK, Distributions, Statistics, LinearAlgebra



function test_t_integral_1D_gaussian()

    rg = MersenneTwister(1)

    # easy case: standard univeriate Gaussian

    σ = 3.1

    μ = 0.1

    g = Normal(μ, σ)

    x = randn(rg)

    f(t) = pdf(g, t*x)

    # numerical integral

    num1 = hquadrature(f, -50, 50)[1]

    num2 = quadgk(f, -50, 50)[1]

    # analytical solution

    ana = 1/sqrt(2π*σ*σ) * sqrt(π) / sqrt(x*x/(2*σ*σ))

    num1, num2, ana, abs(num1-ana)

end



function test_t_integral_2D_diagonal_gaussian(seed)

    rg = MersenneTwister(seed)

    D = 3

    Σ = Diagonal(rand(rg, D)*2)

    U = svd(Σ).U

    μ = randn(rg, D)

    x = randn(rg, D)

    num1 = numericalintegraloft(x, μ, Σ)

    ana = analyticalintegraloft(x, μ, Σ)

    num1, ana, abs(num1-ana)

end





function test_t_integral_2D_full_gaussian(seed=1)

    rg = MersenneTwister(seed)

    D = 3

    Σ = randn(rg,D, D)*0.5; Σ = Σ*Σ' + 0.5*I

    U = svd(Σ).U

    μ = randn(rg, D)

    x = randn(rg, D)

    num1 = numericalintegraloft(x, μ, Σ)

    ana = analyticalintegraloft(x, μ, Σ)

    num1, ana, abs(num1-ana)

end


function numericalintegraloft(x, μ, Σ)

    g = MvNormal(μ, Σ)

    hquadrature(t -> pdf(g, t*x), -50, 50)[1]

end

function analyticalintegraloft(x, μ, Σ)

    D = length(μ) ; @assert(size(Σ, 1) == size(Σ, 2) == D == length(x))

    # see https://en.wikipedia.org/wiki/Gaussian_integral#Generalizations
    a =  0.5*x'*(Σ\x)
    b =  x'*(Σ\μ)
    c = -0.5*μ'*(Σ\μ)

    1/sqrt((2π)^D*det(Σ)) * sqrt(π/a) * exp(b^2/(4*a)+c)

end
