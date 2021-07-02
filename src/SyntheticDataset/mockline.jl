function mockline(; a=a, b=b, t_range=t_range, σ=0.1)

    observations = zeros(3, length(t_range))

    for (index, t) in enumerate(t_range)

        for i in 1:3

            observations[i, index] = a[i]*t + b[i] + randn()*σ

        end

    end

    @printf("Intersection point is at\n") ; display(b) ; @printf("\n")

    figure()

    plot3D(observations[1, :], observations[2, :], observations[3, :], "bo", label="observations")

    plotvector(b, "k-", "vector b")

    legend()

    return observations, σ*ones(size(observations))

end
