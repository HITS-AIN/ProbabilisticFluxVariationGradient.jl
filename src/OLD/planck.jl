function B(; T=T, λ=λ)

    k_B = 1.380649e-23
    h   = 6.62607015e-34
    c   = 2.99792458e8

    aux1 = (2.0 * h * c^2) / λ^5

    aux2 = (h * c) / (λ * k_B * T)

    aux1 / (exp(aux2) - 1.0)

end

function scaledB(; s=s, T=T, λ=λ)

    s * B(T=T, λ=λ)

end


function fitgalaxy()

    wave = [433,550,790] * 10^-9

    Y = [4.18, 8.60, 20.83]

    # wave = [0.5;4;6] * 1000.0 * 1e-9
    # Y = 1e-8*[B(T=5555.5,λ = 500*1e-9); B(T=5555.5,λ = 4000*1e-9);B(T=5555.5,λ = 6000*1e-9)]

    function loss(T)

        local V = map( w -> B(T=T,λ=w), wave)

        local s = dot(V,Y) / dot(V,V)

        local l = 0.0

        for i in 1:length(Y)
            l += (Y[i] - s*V[i])^2
        end

        return l

    end


    result = optimize(loss, 10, 10_000)

    @show T = result.minimizer

    V = map( w -> B(T=T, λ=w), wave)

    s = dot(V,Y) / dot(V,V)

    λ = collect(LinRange(100, 10_000, 100)) * 1e-9;

    plot(λ, map(λ -> s*B(T=T,λ=λ), λ), "g.")

    plot(wave, Y, "ko")

end



function galaxyposterior()

    Tmin = 1000.0
    Tmax = 10_000.0
    σ(x) = 1.0 / (exp(-x) + 1)

    wave = [433,550,790] * 10^-9

    Y = [4.18, 8.60, 20.83]

    # wave = [0.5;4;6] * 1000.0 * 1e-9
    # Y = 1e-8*[B(T=5555.5,λ = 500*1e-9); B(T=5555.5,λ = 4000*1e-9);B(T=5555.5,λ = 6000*1e-9)]

    function loss(T)

        local V = map( w -> B(T=T,λ=w), wave)

        local s = dot(V,Y) / dot(V,V)

        local l = 0.0

        for i in 1:length(Y)
            l += (Y[i] - s*V[i])^2
        end

        return l

    end


    Trange = 1000:20:8_000

    negloss = map(x -> -loss(x), Trange)

    plot(Trange, negloss, ".")

    Trange[argmax(negloss)], maximum(negloss)

end

# λ = collect(LinRange(10,10_000,1000)) * 1e-9;
# plot(λ, map(λ->B(T=4000.0,λ=λ), λ), "g.")
# plot(λ, map(λ->B(T=5000.0,λ=λ), λ), "b.")
