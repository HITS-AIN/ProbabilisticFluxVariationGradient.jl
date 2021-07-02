using Distributions, HCubature, Printf, LinearAlgebra, Random, ProgressMeter

function test_inside_logarithm()

    rg = MersenneTwister(1)

    N = 5 # number of observations

    y = randn(rg, N)
    f = randn(rg, N)
    A = Diagonal(rand(rg, N))
    b = rand(rg, N)
    S = Diagonal(rand(rg, N)) + 0.2*I
    K = 0.1*randn(rg, N, N) ; K= K*K' + 1.1*I

    # Exact
    a1 = logpdf(MvNormal(A*f + b, S), y)
    a2 = logpdf(MvNormal(zeros(N), K), f)

    # My implementation

    b1 = -0.5*N*log(2π) - 0.5*logdet(S)

    c1 = -0.5*(y .- A*f .- b)'*(S\( y.- A*f .- b))

    b2 = -0.5*N*log(2π)

    c2 = -0.5*logdet(K) - 0.5*f'*(K\f)

    # a1, b1+c1, a2, b2+c
    a1+a2, b1+c1+b2+c2
end
