#######################################################################
function marginaline(posterior::FullNormal, σ2)
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


#######################################################################
function marginaline(posterior::DiagNormal, σ2)
#######################################################################

    # dimension of problem is D

    D = round(Int, length(posterior)/2)


    # marginals, conditionals

    μ, Σ = mean(posterior), cov(posterior)

    a_idx, b_idx = collect(1:D), collect(D+1:2D)

    μ_a = μ[a_idx]

    μ_b = μ[b_idx]

    Σ_aa = Diagonal(Σ[a_idx, a_idx])

    # Σ_ab = Σ[a_idx, b_idx]      # These are zero because of diagonal
    #                             # covariance matrix
    # Σ_ba = Σ[b_idx, a_idx]

    Σ_bb = Diagonal(Σ[b_idx, b_idx])

    Σ_a_given_b = Σ_aa # - Σ_ab*(Σ_bb\Σ_ba) # The last term is zero, same reason as above

    σ2Diag = Diagonal(σ2)

    function linemeansigma(t)

        local mu = μ_a * t + μ_b

        # local K = t*(Σ_ab/Σ_bb) + I # The first term is zero, hence K is idenity

        local Sigma = σ2Diag + t * Σ_a_given_b * t  +  Σ_bb# K*Σ_bb*K'

        Sigma = (Sigma + Sigma')/2

        mu, Sigma

    end


end
