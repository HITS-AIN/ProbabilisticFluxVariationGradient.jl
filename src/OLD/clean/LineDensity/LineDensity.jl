using Random, LinearAlgebra, Distributions, PDMats, PyPlot, ProgressMeter, Optim, StatsFuns
using HCubature, QuadGK


struct LineDensity


         joint_a_b::AbstractMvNormal
        marginal_b::AbstractMvNormal
    cond_a_given_b::Function
          bsamples::Array{Array{T,1},1} where T<:Real
        numsamples::Int
                 D::Int

end


function LineDensity(; posterior::MvNormal, numsamples=250)

    joint_a_b = posterior # just another name

    D = round(Int, length(joint_a_b)/2)

    marginal_b = MvNormalPartition(joint_a_b, collect(D+1:2*D))

    cond_a_given_b = MvNormalConditional(joint_a_b, collect(1:D),collect(D+1:2D))

    bsamples = [rand(marginal_b) for i=1:numsamples]

    LineDensity(joint_a_b, marginal_b, cond_a_given_b, bsamples, numsamples, D)

end



function Base.show(io::IO, a::LineDensity)

    print(io, "Line density.",
              "\n\t Number of samples ", a.numsamples,
              "\n\t Dimension is  ", a.D)
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


function (linepdf::LineDensity)(x)

    sum(map(b -> integralconddensityatx(x, b, linepdf.cond_a_given_b), linepdf.bsamples))

end


function logl(linepdf::LineDensity, x)

    logsumexp(map(b -> logintegralconddensityatx(x, b, linepdf.cond_a_given_b), linepdf.bsamples))

end

function numintegralconddensityatx(x, b, cond_a_b)

    # This serves as numerical verification for function integralconddensityatx
    aux = (x-b)/norm(x-b)

    f(t) = pdf(cond_a_b(b), t * aux)

    quadgk(f, 0.01, 100)[1]

end


function integralconddensityatx(x, b, cond_a_b)

    aux = (x-b)/norm(x-b)

    g = cond_a_b(b)

    analyticalintegraloft(aux, mean(g), cov(g))

end

function logintegralconddensityatx(x, b, cond_a_b)

    aux = (x-b)/norm(x-b)

    g = cond_a_b(b)

    loganalyticalintegraloft(aux, mean(g), cov(g))

end

function example(lpdf::LineDensity)

    if lpdf.D > 3
        @printf("Visualisation only available for D=2 or D=3\n")
        return nothing
    end

    figure()

    c = zeros(lpdf.D, lpdf.numsamples)

    for (index, b) in enumerate(lpdf.bsamples)

        r =  rand()*9 + 1

        a = rand(lpdf.cond_a_given_b(b))

        c[:,index] = a * r + b

        plotvector(b)

        if lpdf.D == 3
            plot([b[1];b[1]+r*a[1]], [b[2];b[2]+r*a[2]], [b[3];b[3]+r*a[3]], "--g")
        else
            plot([b[1];b[1]+r*a[1]], [b[2];b[2]+r*a[2]], "--g")
        end

    end

    if lpdf.D == 3
        plot3D(c[1,:], c[2,:], c[3,:], "ro", alpha=0.9)
    else
        plot(c[1,:], c[2,:], "ro", alpha=0.9)
    end

end
