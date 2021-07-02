function gaussconditional(μ::Array{Float64,1}, Σ::Array{Float64,2}, a_idx::Array{Int64,1}, b_idx::Array{Int64,1})

    # make sure the given indices do not overlap
    @assert(isempty(intersect(a_idx, b_idx)))

    Σ_aa = Σ[a_idx, a_idx]

    Σ_ab = Σ[a_idx, b_idx]

    Σ_ba = Σ[b_idx, a_idx]

    Σ_bb = Σ[b_idx, b_idx]

    μ_a  = @view μ[a_idx]

    μ_b  = @view μ[a_idx]

    # L = cholesky(Σ_bb).L
    #
    # Q = L \ Σ_ba
    #
    # μ_a_given_b(x) = μ_a + Σ_ab*(L' \ (L \ (x .- μ_b)))
    #
    # Σ_a_given_b    = PDMat(Σ_aa - Q'*Q)

    μ_a_given_b(x) = μ_a + Σ_ab*(Σ_bb \ (x .- μ_b))

     Σ_a_given_b    = PDMat(Σ_aa - Σ_ab*(Σ_bb\Σ_ba))

    return μ_a_given_b, Σ_a_given_b

end


function MvNormalConditional(g::AbstractMvNormal, a_idx::Array{Int64,1}, b_idx::Array{Int64,1})

    μcond, Σcond = gaussconditional(g.μ, g.Σ.mat, a_idx, b_idx)

    x -> MvNormal(μcond(x), Σcond)

end


function MvNormalPartition(g::AbstractMvNormal, keep_idx::Array{Int64,1})

    μnew = g.μ[keep_idx]

    Σnew = PDMat(g.Σ.mat[keep_idx, keep_idx])

    MvNormal(μnew, Σnew)

end


function testMvNormalConditional()

    # let's do a comparison on an example calculated in two different ways



end
