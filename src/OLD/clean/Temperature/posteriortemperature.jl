function posteriortemperature(lpdf::LineDensity, Trange::AbstractRange{Float64}, Srange::AbstractRange{Float64})

    @showprogress map(P -> logl(lpdf, P[2] * observedunitgalaxyvector(P[1])), Iterators.product(Trange, Srange))

end
