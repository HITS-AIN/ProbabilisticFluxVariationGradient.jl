
function exact_expectation_inside_log_of_lower_bound(; y=y, idx=idx, S=S, μα=μα, Σα=Σα, v²ₐ=v²ₐ, μβ=μβ, Σβ=Σβ, v²ᵦ=v²ᵦ, μf=μf, Σf=Σf, K=K)

    expectationterm1(;y=y, μα=μα, Σα=Σα, μβ=μβ, Σβ=Σβ, μf=μf, Σf=Σf, S=S, idx=idx) +
    expectationterm2(;μf=μf, Σf=Σf, K=K)     +
    expectationterm3(;μα=μα, Σα=Σα, v²ₐ=v²ₐ) +
    expectationterm4(;μβ=μβ, Σβ=Σβ, v²ᵦ=v²ᵦ)

end


##########
# TERM 1 #
##########

function expectationterm1(;y=y, μα=μα, Σα=Σα, μβ=μβ, Σβ=Σβ, μf=μf, Σf=Σf, S=S, idx=idx)

    N = length(y)

    ## look out for the diagonal and the diff between Σα and Σa
    Ā = Diagonal(μα[idx]) ; Σa = Diagonal(diag(Σα)[idx])

    ## look out for the diagonal and the diff between Σβ and Σb
    b̄ = μβ[idx] ; Σb = Diagonal(diag(Σβ)[idx])

    subterm0 = -0.5*logdet(S) - 0.5*N*log(2π)

    subterm1 = -0.5*dot(y, S\y)

    subterm2 = -0.5*dot(y, -S\Ā*μf) * 2

    subterm3 = -0.5*dot(y, -S\b̄) * 2

    # subterm4 = -0.5*sum((μf.^2 .+ diag(Σf)) .* (Ā.diag.^2 .+ Σa.diag) ./ S.diag)
    subterm4 = -0.5*(μf'*(S\(Ā.^2 + Σa))*μf)  -0.5*tr(Σf*(S\(Ā.^2 + Σa)))

    subterm5 = -0.5*dot(b̄, S\Ā*μf) * 2

    subterm6 = -0.5*dot(b̄, S\b̄) -0.5*tr(S\Σb)

    (subterm0 + subterm1 + subterm2 + subterm3 + subterm4 + subterm5 + subterm6)

end


##########
# TERM 2 #
##########

function expectationterm2(;μf=μf, Σf=Σf, K=K)
    N = length(μf)
    -0.5*N*log(2π) -0.5*logdet(K) - 0.5*μf'*(K\μf) - 0.5*tr(K\Σf)
end


##########
# TERM 3 #
##########

function expectationterm3(;μα=μα, Σα=Σα, v²ₐ=v²ₐ)
    F = length(μα)
    - 0.5*F*log(2π) - 0.5*F*log(v²ₐ) - 0.5*dot(μα, μα)/v²ₐ - 0.5*tr(Σα)/v²ₐ
end


##########
# TERM 4 #
##########

function expectationterm4(;μβ=μβ, Σβ=Σβ, v²ᵦ=v²ᵦ)
    F = length(μβ)
    - 0.5*F*log(2π) - 0.5*F*log(v²ᵦ) - 0.5*dot(μβ,μβ)/v²ᵦ - 0.5*tr(Σβ)/v²ᵦ
end
