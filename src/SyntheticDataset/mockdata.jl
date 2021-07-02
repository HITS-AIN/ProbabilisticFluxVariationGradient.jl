# Convenience function that calls the more complicated simulatedata
# function with fixed arguments.
"""
    t, fluxes, errors, truegalaxyvector = mockdata()

Returns 50 regular (i.e. co-occuring/simultaneous) observations in three filters
and the true galaxy vector.
`fluxes` and `errors` have both dimensions 3 × 50.
`truegalaxyvector` is a 3 dimensional vector.
"""
function mockdata()

    PyPlot.ioff()

    Temp = 5000.0
    Sc   =   10.0

    _, _, _, _, regflux, regσ, _, obstimeco = @suppress simulatedata(N=50, SN=50.0, Temperature=Temp, offset=5.0, scale=Sc, logs=-6.0);
    close("all")
    PyPlot.ion()

    return obstimeco, Matrix(regflux[:,1:3]'), Matrix(regσ[:,1:3]'), Sc * vec(unitgalaxyvector(Temp)[1:3])

end
