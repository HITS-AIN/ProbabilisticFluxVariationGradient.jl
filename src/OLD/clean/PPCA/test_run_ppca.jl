#######################################################################
function test_run_ppca(σ, seed)
#######################################################################

    x, = simulate3Ddata(σ, seed)

    W, μ, σ, proj, fwd = ppca(x')

    figure(1)

    cla()

    plot3D(x[:,1], x[:,2], x[:,3], "bo", alpha=0.3)

    z = sort([proj(x[i,:]) for i=1:size(x,1)])

    zgrid = collect(LinRange(minimum(z), maximum(z), 100))

    X = reduce(hcat, fwd.(zgrid))'

    plot3D(X[:,1], X[:,2], X[:,3], "-")

    return W, μ

end
