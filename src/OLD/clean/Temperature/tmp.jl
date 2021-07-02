function tmp(posterior, t, numsamples)

    μ, Σ = mean(posterior), cov(posterior)

    a_idx, b_idx = [1;2], [3;4]

    μ_a = μ[a_idx]
    μ_b = μ[b_idx]


    Σ_aa = Σ[a_idx, a_idx]

    Σ_ab = Σ[a_idx, b_idx]

    Σ_ba = Σ[b_idx, a_idx]

    Σ_bb = Σ[b_idx, b_idx]





    samples = [rand(posterior) for i in 1:numsamples]

    aux(x) = x[a_idx]*t + x[b_idx]

    # mean(map(aux, samples)), μ_a*t + μ_b

    Σ_a_given_b = Σ_aa - Σ_ab*(Σ_bb\Σ_ba)

    K = t*Σ_ab/Σ_bb + I

    cov(map(aux, samples)), t*Σ_a_given_b*t + K*Σ_bb*K'

end
