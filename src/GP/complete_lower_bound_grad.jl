function complete_lower_bound_grad(; y=y, idx=idx, S=S, μα=μα, Σα=Σα, v²ₐ=v²ₐ, μβ=μβ, Σβ=Σβ, v²ᵦ=v²ᵦ, μf=μf, Σf=Σf, K=K)

    #
    # ∫ q(α) q(β) q(f) log𝓝(y| Af + b, S) + log𝓝(f|0,K) + log𝓝(a|0,v²ₐI) + log𝓝(b|0,v²ᵦ) dα dβ df
    #                           (1)               (2)             (3)             (4)

    gradα, gradΣα, gradβ, gradΣβ = exact_expectation_inside_log_of_lower_bound_grad(; y=y, idx=idx, S=S, μα=μα, Σα=Σα, v²ₐ=v²ₐ, μβ=μβ, Σβ=Σβ, v²ᵦ=v²ᵦ, μf=μf, Σf=Σf, K=K)


    #
    # H[q(α)]  +  H[q(β)]  +  H[q(f)]
    #

    # remember we parametrise the n-th entry of the diagonal covariance
    # Σa with elements exp(pₙ), i.e. gradient calculated below is dependent
    # on this parametrisation (see "unpack" function)
    gradΣα += 0.5 * ones(size(Σα, 1))

    # same comment as above applies here
    gradΣβ += 0.5 * ones(size(Σβ, 1))

    return gradα, gradΣα, gradβ, gradΣβ

end
