#######################################################################
function test_run_bpca(σ, seed, S, maxiter)
#######################################################################

    x, Σ = simulatedatapca(σ, seed)#
    #x, Σ = realresampleddataforpca()#

    posterior = bpca(x', Σ'; maxiter=maxiter, S=S)

    figure(1)

    cla()

    plot3D(x[:,1], x[:,2], x[:,3], "bo", alpha=0.3)


    for i in 1:30

        D = size(x, 2)

        param = rand(posterior)

        W, μ = param[1:D], param[D+1:2*D]

        project(x) = ((W'*W) \ (W'*(x - μ)))[1]

        fwd(z::Real) = vec(W*z + μ)

        z = sort([project(x[i,:]) for i=1:size(x,1)])

        zgrid = collect(LinRange(minimum(z), maximum(z), 100))

        X = reduce(hcat, fwd.(zgrid))'

        plot3D(X[:,1], X[:,2], X[:,3], "-")

    end

    return posterior

end
