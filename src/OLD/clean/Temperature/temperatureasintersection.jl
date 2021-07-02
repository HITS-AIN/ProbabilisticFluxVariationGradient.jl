
function temperatureasintersection(a::Array{Float64,1}, b::Array{Float64,1}, σ2::Array{Float64,1}, T::Float64, s::Float64)

    x = s*unitgalaxyvector(T)

    p = dot(a, (x-b)) / dot(a,a)

    logpdf(MvNormal(a*p + b, Diagonal(σ2)), x)

end


function temperatureasintersection(a::Array{Float64,1}, b::Array{Float64,1}, σ2::Array{Float64,1}, Trange::AbstractRange{Float64}, Srange::AbstractRange{Float64})

    f(T, s) = temperatureasintersection(a,b,σ2,T,s)

    @showprogress map(P -> f(P[1], P[2]), Iterators.product(Trange, Srange))


end
