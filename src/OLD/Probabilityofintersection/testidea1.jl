function testidea1()

    rg = MersenneTwister(1)

    # define parameter distributions

    mu    = [3.57; 9.95; 3.87; 9.95; 28.22; 14.07]

    Sigma = [0.0148; 0.0706; 0.0187; 0.0266; 0.1006; 0.0238]

    # Since Sigma is a diagonal covariance matrix
    # we can sample parameters a and b independently
    # (in the future these parameters will be correlated)

    pdf_a = MvNormal(mu[1:3], Diagonal(Sigma[1:3]))

    pdf_b = MvNormal(mu[4:6], Diagonal(Sigma[4:6]))


    # Distribution of normed vectors (this needs to be refined)
    A = reduce(hcat, [(a=rand(rg,pdf_a); a/norm(a)) for i=1:10_000])'

    pdf_a_norm = MvNormal(vec(mean(A,dims=1)), cov(A))

    # draw samples for b
    B = [rand(pdf_b) for i=1:200]

    figure(1)
    cla()
    xrange = collect(LinRange(-2,2,100))

    for i=1:5

        a, b = rand(rg, pdf_a), rand(rg, pdf_b)

        for x in xrange
               local p = b + a*x
               plot3D(p[1], p[2], p[3], "r.", alpha=0.3)
        end
    end


    function test(x)

        l = zeros(length(B))

        for (i,b) in enumerate(B)

            local v = x - b

            l[i] = logpdf(pdf_a_norm, v / norm(v))

        end

        return logsumexp(l) - log(length(B))

    end

end
