function testidea2(posterior)

    rg = MersenneTwister(1)

    pdf_without_scale = MvNormalPartition(posterior, [2;3;4;5;6])

    pdf_b = MvNormalPartition(pdf_without_scale, [3;4;5])

    pdf_a_given_b = MvNormalConditional(pdf_without_scale, [1;2], [3;4;5])


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
