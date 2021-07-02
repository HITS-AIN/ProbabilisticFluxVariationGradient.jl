function PHI(x::Array{T,1} where T<:Real, c::Array{T,1} where T<:Real, rinv::Real)

    dist2 = pairwise(SqEuclidean(), reshape(x, 1, length(x)), reshape(c, 1, length(c)), dims=2)

    Φ = exp.(-0.5*dist2*rinv*rinv)

    @assert(size(Φ,1) == length(x)) # N

    @assert(size(Φ,2) == length(c)) # M

    return Φ
    
end


function centresongrid(x; numcentres=100)

    collect(LinRange(minimum(x), maximum(x), numcentres))

end
