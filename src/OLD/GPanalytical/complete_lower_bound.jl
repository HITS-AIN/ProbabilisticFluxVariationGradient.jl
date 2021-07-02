
function complete_lower_bound(; y=y, idx=idx, S=S, μα=μα, Σα=Σα, v²ₐ=v²ₐ, μβ=μβ, Σβ=Σβ, v²ᵦ=v²ᵦ, μf=μf, Σf=Σf, K=K)

    #
    # ∫ q(α) q(β) q(f) log𝓝(y| Af + b, S) + log𝓝(f|0,K) + log𝓝(a|0,v²ₐI) + log𝓝(b|0,v²ᵦ) dα dβ df
    #                           (1)               (2)             (3)             (4)


    Elogl = exact_expectation_inside_log_of_lower_bound(; y=y, idx=idx, S=S, μα=μα, Σα=Σα, v²ₐ=v²ₐ, μβ=μβ, Σβ=Σβ, v²ᵦ=v²ᵦ, μf=μf, Σf=Σf, K=K)

    #
    # H[q(α)]  +  H[q(β)]  +  H[q(f)]
    #

    gaussianentropy(Σ) = 0.5*logdet(2.0 * π * ℯ * Σ)

    return Elogl + gaussianentropy(Σα) + gaussianentropy(Σβ) + gaussianentropy(Σf)

end
