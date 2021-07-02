
function double_expectation(numsamples=1000;seed=1)

    rg = MersenneTwister(seed)

    N = 5
    F = 2
    idx = ceil.(Int, rand(rg, N)*F)

    μf = rand(rg,N)
    Σf = randn(rg,N,N) ; Σf = Σf*Σf'
    qf = MvNormal(μf, Σf)
    f  = rand(rg, qf)

    μα = randn(rg, F); Σα = Diagonal(rand(rg, F))
    qα = MvNormal(μα, Σα)

    S = Diagonal(rand(rg,N) .+ 0.1);

    # first verify that this holds for diagonal matrices
    A = Diagonal(rand(rg,N))
    f'*A'*inv(S)*A*f, sum(f.^2 .* diag(A).^2 ./ diag(S))  # ✅

    # test inner expectation
    innerexpectation = μf'*A'*inv(S)*A*μf + tr(A*inv(S)*A*Σf)

    # numinnerexpectation = @distributed (+) for i=1:numsamples
    #
    #     local f = rand(qf)
    #
    #     f'*A'*inv(S)*A*f / numsamples
    #
    # end
    #
    # innerexpectation, numinnerexpectation # ✅



    # test double expectation
    μa = μα[idx]
    Σa = Diagonal(Σα.diag[idx])

    doubleexpectation  = sum(μf.^2 .* (μa.^2 .+ Σa.diag) ./ S.diag) + sum((μa.^2 .+ Σa.diag).*diag(Σf) ./S.diag)

    shorterexpectation = sum((μf.^2 .+ diag(Σf)) .* (μa.^2 .+ Σa.diag) ./ S.diag)


    numdoubleexpectation = 0#@distributed (+) for i=1:numsamples
    #
    #     local A = Diagonal(rand(qa))
    #
    #     (μf'*A'*(S\A*μf) + tr(A*inv(S)*A*Σf)) / numsamples
    #
    # end

    altnumdoubleexpectation = @distributed (+) for i=1:numsamples

        local A = Diagonal(rand(qα)[idx])
        local f = rand(qf)

        f'*A'*(S\A*f) / numsamples

    end

    shorterexpectation, doubleexpectation, numdoubleexpectation,altnumdoubleexpectation # ✅


end
