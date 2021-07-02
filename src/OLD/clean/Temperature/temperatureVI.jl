#---------------------------------------------------------------
function temperatureVI(posterior, σ2, T; maxiter = 1, S = 10)
#---------------------------------------------------------------

    D = round(Int, length(posterior)/2)

    gT = observedunitgalaxyvector(T)

    function unpack(param)

        @assert(length(param) == 2D+2)

        local a     = param[1:D]
        local b     = param[1+D:2*D]
        local t     = param[end-1]
        local scale = exp(param[end])

        return a, b, t, scale

    end

    function logp(a, b, t, scale)

        local x = scale * gT

        logpdf(MvNormal(a*t + b, Diagonal(σ2)), x) + logpdf(posterior,[a;b])

    end

    objective(param) = logp(unpack(param)...)

    optimoptions = Optim.Options(show_trace = false, iterations = maxiter, g_tol=1e-6)

    VI(objective, [5*randn(2D+2) for i=1:100]; maxiter = maxiter, S = S, optimiser = LBFGS(), optimoptions = optimoptions)

end



#---------------------------------------------------------------
function temperatureVI2(posterior, σ2, T; maxiter = 1, S = 10)
#---------------------------------------------------------------

    D = round(Int, length(posterior)/2)

    gT = observedunitgalaxyvector(T)


    function unpack(param)

        @assert(length(param) == 2D+1)

        local a     = param[1:D]
        local b     = param[1+D:2*D]
        local scale = exp(param[end])

        return a, b, scale

    end


    function logp(a, b, scale)

        local x = scale * gT

        logpdf(MvNormal(a*(dot(a,x-b)) + b, Diagonal(σ2)), x) + logpdf(posterior,[a;b])

    end


    objective(param) = logp(unpack(param)...)

    optimoptions = Optim.Options(show_trace = false, iterations = maxiter, g_tol=1e-6)

    VI(objective, [5*randn(2D+1) for i=1:100]; maxiter = maxiter, S = S, optimiser = LBFGS(), optimoptions = optimoptions)

end





#---------------------------------------------------------------
function temperatureVI3(posterior, σ2, T; S = S, maxiter = maxiter)
#---------------------------------------------------------------

    # dimension of problem is D

    D = round(Int, length(posterior)/2)

    # create galaxy vector

    g = observedunitgalaxyvector(T)

    # marginals, conditionals

    linemeansigma = marginaline(posterior, σ2)


    function logp(p)

        local s, t = exp(p[1]), p[2]

        local mu, Sigma = linemeansigma(t)

        # local logprior = logpdf(Normal(0.0, 1000.0), s) + logpdf(Normal(0.0, 1000.0), p[2])

        logpdf(MvNormal(mu, Sigma), s*g)# + logprior

    end


    initoptions  = Optim.Options(iterations = 5, g_tol = 1e-6)

    optimoptions = Optim.Options(show_trace = false, show_every = 1, iterations = maxiter, g_tol = 1e-6)

    initsol      = estimatetemperature4(posterior, σ2, T)[1]

    VI(logp, push!([randn(2) for i=1:5], initsol); S = S, optimiser = LBFGS(), optimoptions = optimoptions, initoptions = initoptions)

end
