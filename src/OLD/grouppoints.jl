using Printf

function grouppoints(Xcopy)

    X = deepcopy(Xcopy)

    K = length(X)

    #------------------------------------------------
    # find overall earliest and latest point in time
    #------------------------------------------------

    earliestK = 0
    earliestT = Inf

    latestK = 0
    latestT = -Inf


    for k in 1:K

        # make sure it is sorted
        @assert(issorted(vec(X[k][:,1])))

        if X[k][1,1] < earliestT
            earliestT = X[k][1,1]
            earliestK = k
        end

        if X[k][end,1] > latestT
            latestT = X[k][end,1]
            latestK = k
        end

    end

    @printf("Earliest time is %5.3f\n", earliestT)
    @printf("Latest   time is %5.3f\n", latestT)

    #------------------------------------------------
    # group points
    #------------------------------------------------

    gridpoints = ceil(Int, (latestT - earliestT) * 10)

    tol = 1.0

    Grouped = zeros(0, K)
    T       = zeros(0)

    for t in LinRange(earliestT, latestT, gridpoints)

        closestintime = [minimum(abs.(X[k][:,1] .- t)) for k in 1:K]

        if all(closestintime .< tol)


            idx = [argmin(abs.(X[k][:,1] .- t)) for k in 1:K]

            @show t idx

            Grouped = [Grouped; reshape([X[k][idx[k], 2] for k in 1:K], 1, K)]

            T = [T;t]

            # delete used elements from respective arrays

            for k in 1:K
                X[k] = X[k][setdiff(1:size(X[k],1), idx[k]), :]

                if isempty(X[k])
                    return T, Grouped
                end

            end



        end

    end

    return T, Grouped

end
