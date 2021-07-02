"""

    simulatedata(; N=50, SN=100.0, Temperature=5000.0, offset=5.0, scale=11.0, seed=1)

Simulates observations from total flux, i.e. AGN + Galaxy. The observations are simulated at the LSST wavelengths. Briefly, the data are simulated as follows:

- The AGN is simulated using the `lsststd` function.

- The galaxy is simulated as follows: given temperature, we use a planck model to simulate emission at LSST wavelengths. These emissions form a vector which is then normalised and then scaled by the input argument `scale`.

The function returns two types of observations: **co-occuring** (or simultaneous) in time and **non co-occuring**.

When called, it will produce the following plots:

- One plot ...
- The other plot ...

The function is useful for verifiying the code and is not meant to necessarily present a physically realistic scenario.


# Arguments

- `N`           : integer number of observations to be simulated per filter
- `SN`          : Signal to noise
- `Temperature` : temperature of the galaxy simulated as a black body between 1000 and 12000 Kelvin
- `offset`      : positive offset from minimum observed AGN activity
- `scale`       : positive scalar mulitplied with the unit galaxy vector (see above)
- `seed`        : controls the random number generator, useful for reproducing data


# Returns

- `obstime`   : vector of observation times
- `obsflux`   : vector of non co-occuring total flux observations
- `σobs`      : observed errors of measurements
- `filteridx` : vector of integer entries that express filter membership
- `obsfluxco` : matrix of dimensions N×(number of filters) that holds co-occuring total flux observations
- `σobsco`    : matrix of dimensions N×(number of filters) that holds errors of co-occuring observations
- `lineeq`    : true total flux line
- `obstimeco` : times of co-occuring flux observations

# Example

```julia-repl
obstime, obsflux, σobs, filteridx, obsfluxco, σobsco = simulatedata(N=50, σ=0.1, Temperature=3000, offset=15.5, scale=10.0)
```
"""
function simulatedata(; N=50, SN=50.0, logs=-8.0, Temperature = 5000.0, offset=2.0, scale=20.0, seed=1, cooccur=false)

    @assert(scale > 0.0)
    @assert(Temperature > 1000.0)
    @assert(Temperature < 15_000.0)
    @assert(offset >= 0.0)
    @assert(SN > 0.0)

    rg = MersenneTwister(seed)

    numwaves = length(lsstwaves())


    # specify parameters for data generation

    a = [lsststd(zpwave = λ, z = 0.5, lum = 5e43) for λ in lsstwaves()].*5
    b = unitgalaxyvector(Temperature) * scale
    θ = [0.0; logs]


    # These are the filter memberships of the data

    filteridx = kron(collect(1:numwaves), ones(Int, N))


    # Initialise arrays to store data

    obstime   = Array{Array{Float64,1},1}(undef, numwaves)
    agnflux   = Array{Array{Float64,1},1}(undef, numwaves)
    obsflux   = Array{Array{Float64,1},1}(undef, numwaves)
    σobs      = Array{Array{Float64,1},1}(undef, numwaves)

    agnfluxco = zeros(N, numwaves)
    obsfluxco = zeros(N, numwaves)
    σobsco    = zeros(N, numwaves)

    # draw latent function from Gaussian process

    T         = 2000
    timegrid  = collect(LinRange(0.0, 1000, T))
    K         = calculatekernelmatrix(timegrid',timegrid', rbf, θ)
    z         = rand(rg, MvNormal(zeros(T), K))

    @printf("Statistics of latent signal\n")
    @show mean(z) std(z)


    sametime = sort(randperm(rg, length(timegrid))[1:N])

    for f = 1:numwaves

        # Non co-occuring observations

        timeidx    = cooccur ? sametime : sort(randperm(rg, length(timegrid))[1:N])

        obstime[f] = timegrid[timeidx] .+ offset

        agnflux[f] = a[f]*(z[timeidx] .+ offset)

        flux       = a[f]*(z[timeidx] .+ offset) .+ b[f]

        σobs[f]    = flux / SN

        obsflux[f] = flux .+ σobs[f] .* randn(rg, N)

        # Co-occuring observations

        agnfluxco[:, f] = a[f]*(z[sametime] .+ offset)

        fluxco          = a[f]*(z[sametime] .+ offset) .+ b[f]

        σobsco[:, f]    = fluxco / SN

        obsfluxco[:, f] = fluxco .+ σobsco[:, f].*randn(rg, N)

    end


    figure(1)
    cla()
    title("latent signal simulated as gp draw")
    plot(timegrid, z, "k")


    figure(2)
    cla()
    title("plotting irregular total flux time series")
    for f = 1:numwaves
        PyPlot.plot(obstime[f], obsflux[f], "o", label=@sprintf("%.2f", lsstwaves()[f]))
    end

    PyPlot.legend()

    figure(3)
    cla()
    title("plotting regular total flux time series")
    for f = 1:numwaves
        PyPlot.plot(timegrid[sametime].+offset, obsfluxco[:,f], "o", label=@sprintf("%.2f", lsstwaves()[f]))
    end

    PyPlot.legend()


    figure(4)
    cla()
    title("plotting first three filters")
    PyPlot.plot3D(obsfluxco[:, 1], obsfluxco[:, 2], obsfluxco[:, 3], "bo", alpha=0.2, label="obsmix")
    PyPlot.plot3D(agnfluxco[:, 1], agnfluxco[:, 2], agnfluxco[:, 3], "k.", alpha=1.0, label="agn")
    PyPlot.plot3D([0;b[1]], [0;b[2]], [0;b[3]], "c-", alpha=1.0, label="galaxy")

    PyPlot.legend()


    lineeq(t) = a*(t + offset) + b

    obstimeco = timegrid[sametime] .+ offset

    return reduce(vcat, obstime), reduce(vcat, obsflux), reduce(vcat, σobs), filteridx, obsfluxco, σobsco, lineeq, obstimeco

end
