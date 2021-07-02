#############################################################
function mylogpdf_Cov(x, μ, Σ)
#############################################################

    @assert(length(x) == length(μ) == size(Σ,1) == size(Σ,2))

    d = length(μ)

    L = chol(Hermitian(Σ))'

    TERM1 = -0.5 * 2.0*sum(log.(diag(L))) # consult eq. A.18 in RW
    TERM2 = -0.5 * d * log(2π)
    TERM3 = -0.5 * sum(( (L\(x-μ)).^2 ))

    return TERM1 + TERM2 + TERM3

end

#############################################################
function mylogpdf_scalCov(x, μ, σ2)
#############################################################

    @assert(length(x) == length(μ))

    d = length(μ)

    TERM1 = -0.5 * d * log(σ2)
    TERM2 = -0.5 * d * log(2π)
    TERM3 = -0.5 * sum((x-μ).^2)/σ2

    return TERM1 + TERM2 + TERM3

end
