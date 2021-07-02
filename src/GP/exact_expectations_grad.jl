
function exact_expectation_inside_log_of_lower_bound_grad(; y=y, idx=idx, S=S, μα=μα, Σα=Σα, v²ₐ=v²ₐ, μβ=μβ, Σβ=Σβ, v²ᵦ=v²ᵦ, μf=μf, Σf=Σf, K=K)


    gradα, gradΣα, gradβ, gradΣβ = expectationterm1_grad(;y=y, μα=μα, Σα=Σα, μβ=μβ, Σβ=Σβ, μf=μf, Σf=Σf, S=S, idx=idx)


    auxα, auxΣα  = expectationterm3_grad(; μα=μα, Σα=Σα, v²ₐ=v²ₐ)

    gradα  += auxα

    gradΣα += auxΣα



    auxβ, auxΣβ = expectationterm4_grad(; μβ=μβ, Σβ=Σβ, v²ᵦ=v²ᵦ)

    gradβ  += auxβ

    gradΣβ += auxΣβ


    return gradα, gradΣα, gradβ, gradΣβ

end


##########
# TERM 1 #
##########

function expectationterm1_grad(;y=y, μα=μα, Σα=Σα, μβ=μβ, Σβ=Σβ, μf=μf, Σf=Σf, S=S, idx=idx)

    N = length(y)

    ## look out for the diagonal and the diff between Σα and Σa
    Ā = Diagonal(μα[idx]) ; Σa = Diagonal(diag(Σα)[idx])

    ## look out for the diagonal and the diff between Σβ and Σb
    b̄ = μβ[idx] ; Σb = Diagonal(diag(Σβ)[idx])

    # subterm0 = -0.5*logdet(S) - 0.5*N*log(2π)
    #
    # subterm1 = -0.5*dot(y, S\y)
    #
    # subterm2 = -0.5*dot(y, -S\Ā*μf) * 2
    #
    # subterm3 = -0.5*dot(y, -S\b̄) * 2
    #
    # # subterm4 = -0.5*sum((μf.^2 .+ diag(Σf)) .* (Ā.diag.^2 .+ Σa.diag) ./ S.diag)
    # subterm4 = -0.5*(μf'*(S\(Ā.^2 + Σa))*μf)  -0.5*tr(Σf*(S\(Ā.^2 + Σa)))
    #
    # subterm5 = -0.5*dot(b̄, S\Ā*μf) * 2
    #
    # subterm6 = -0.5*dot(b̄, S\b̄) -0.5*tr(S\Σb)
    #
    # (subterm0 + subterm1 + subterm2 + subterm3 + subterm4 + subterm5 + subterm6)


    #########
    # gradα #
    #########

    gradα = zeros(length(μα))

    for n = 1:N

        # from subterm (2)
        gradα[idx[n]] +=   y[n] * μf[n] / S[n,n]

        # from subterm (4)
        gradα[idx[n]] += - (μf[n]^2 +  Σf[n,n]) / S[n,n] * Ā[n,n]

        # from subterm (5)
        gradα[idx[n]] += - b̄[n] * μf[n] / S[n,n]

    end


    ##########
    # gradΣα #
    ##########

    gradΣα = zeros(length(μα)); @assert(length(μα) == size(Σα,1))

    for n = 1:N

        # subterm (4)
        # remember we parametrise the n-th entry of the diagonal covariance
        # Σa with elements exp(pₙ), i.e. gradient calculated below is dependent
        # on this parametrisation (see "unpack" function)
        gradΣα[idx[n]] += - 0.5*(μf[n]^2 +  Σf[n,n]) / S[n,n] *  Σa[n,n]

    end


    #########
    # gradβ #
    #########

    gradβ = zeros(length(μβ))

    for n = 1:N

        # from subterm (3)
        gradβ[idx[n]] +=   y[n] / S[n,n]

        # from subterm (5)
        gradβ[idx[n]] += - Ā[n,n] * μf[n] / S[n,n]

        # from subterm (6)
        gradβ[idx[n]] += - b̄[n] / S[n,n]

    end


    ##########
    # gradΣβ #
    ##########

    gradΣβ = zeros(length(μβ)); @assert(length(μβ) == size(Σβ,1))

    for n = 1:N

        # from subterm 6
        gradΣβ[idx[n]] += - 0.5 * Σb[n, n] / S[n,n]

    end


    return gradα, gradΣα, gradβ, gradΣβ

end


##########
# TERM 2 #
##########

# function expectationterm2_grad(;μf=μf, Σf=Σf, K=K)
#     N = length(μf)
#     -0.5*N*log(2π) -0.5*logdet(K) - 0.5*μf'*(K\μf) - 0.5*tr(K\Σf)
# end


##########
# TERM 3 #
##########

function expectationterm3_grad(;μα=μα, Σα=Σα, v²ₐ=v²ₐ)
    # F = length(μα)
    # - 0.5*F*log(2π) - 0.5*F*log(v²ₐ) - 0.5*dot(μα, μα)/v²ₐ - 0.5*tr(Σα)/v²ₐ

    gradα = - μα / v²ₐ

    gradΣα = -0.5*diag(Σα)/v²ₐ

    return gradα, gradΣα

end


##########
# TERM 4 #
##########

function expectationterm4_grad(;μβ=μβ, Σβ=Σβ, v²ᵦ=v²ᵦ)
    # F = length(μβ)
    # - 0.5*F*log(2π) - 0.5*F*log(v²ᵦ) - 0.5*dot(μβ,μβ)/v²ᵦ - 0.5*tr(Σβ)/v²ᵦ

    gradβ = - μβ / v²ᵦ

    gradΣβ = -0.5*diag(Σβ)/v²ᵦ

    return gradβ, gradΣβ

end
