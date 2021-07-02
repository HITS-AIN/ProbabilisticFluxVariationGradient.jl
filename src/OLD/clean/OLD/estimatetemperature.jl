#######################################################################
function estimatetemperature1(W, μ, T)
#######################################################################

    g = observedunitgalaxyvector(T)

    function loss(x)

        norm(x - μ - proj(W, x - μ))^2 + norm(x - proj(g, x))^2

    end

    opt = Optim.Options(show_trace = false, iterations = 1_000_000)

    result = optimize(loss, randn(3)*10, LBFGS(), opt, autodiff=:forward)

    result.minimizer, result.minimum

end


#######################################################################
function estimatetemperature2(W, μ, T)
#######################################################################

    g = observedunitgalaxyvector(T)

    function loss(s)

        norm(s*g - μ - proj(W, s*g - μ))^2

    end

    opt = Optim.Options(show_trace = true, iterations = 1_000_000)

    result = optimize(loss, 0.001, 1000.0)

    result.minimizer, result.minimum

end
