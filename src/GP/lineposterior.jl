
function lineposterior(; time=time, flux=flux, σ2=σ2, idx=idx, maxiter = 10, maxinit = 10, maxrandom = 10, seed = 1, show_trace = true, usercheck = true)


    logs, predictTest = multigp(time, flux, σ2, idx; maxiter = maxinit,
                                                   maxrandom = maxrandom,
                                                        seed = seed,
                                                  show_trace = show_trace)

    if usercheck
        for i in unique(idx)

            figure()
            cla()
            title(@sprintf("filter index %d", i))
            xtest = collect(LinRange(minimum(time), maximum(time), 200))
            μpred, Σpred = predictTest(xtest, i)
            PyPlot.plot(time[findall(idx.==i)], flux[findall(idx.==i)], "ko")
            PyPlot.plot(xtest, μpred, "r-")
            PyPlot.fill_between(xtest, μpred - sqrt.(diag(Σpred)), μpred + sqrt.(diag(Σpred)), color="r", alpha=0.1)

        end
    end

    @show logs

    # gpmodelVI(time, flux, σ2, idx; logs=logs, maxiterml=15_000, maxitervi=maxiter, S=200, seed=seed)

    gpmodelVI(time, flux, σ2, idx; logs=logs, maxiter = maxiter, seed = seed, show_trace = show_trace)

end
