function plotmarginaline(fluxobs, posterior, σ2)

  linemeansigma = marginaline(posterior, σ2)

  # project flux obs on line to find minimum, maximum t for plotting purposes

  D = round(Int, length(posterior)/2)

  ā, b̄ = mean(posterior)[1:D], mean(posterior)[D+1:2*D]

  proj(x) = dot(ā, x-b̄) / dot(ā, ā)

  tproj = [proj(fluxobs[:,i]) for i=1:size(fluxobs, 2)]

  tmin, tmax = minimum(tproj), maximum(tproj)

  tmin, tmax = sign(tmin) * abs(tmin * 1.2), sign(tmax) * abs(tmax * 1.2)

  trange = collect(LinRange(tmin, tmax, 100))

  predictions     = map(linemeansigma, trange)
  meanpredictions = reduce(hcat, [p[1] for p in predictions])
  covpredictions  = reduce(hcat, [diag(p[2]) for p in predictions])

  figure()

  # verification
  # plot3D(ā[1].*trange .+ b̄[1], ā[2].*trange .+ b̄[2], ā[3].*trange .+ b̄[3], "c-", label="mean fit")
  # map(x -> plot3D(x...,"k."), meanpredictions)

  plot3D(fluxobs[1,:], fluxobs[2,:], fluxobs[3,:], "bo", label="observed total flux")

  plot3D(vec(meanpredictions[1,:]), vec(meanpredictions[2,:]), vec(meanpredictions[3,:]), "k-", label="mean fit")


  plot3D(vec(meanpredictions[1,:]) .+ 2*vec(sqrt.(covpredictions[1,:])),
    vec(meanpredictions[2,:]),
    vec(meanpredictions[3,:]), "k--", label="std2")

  plot3D(vec(meanpredictions[1,:]),
    vec(meanpredictions[2,:] .+ 2*vec(sqrt.(covpredictions[2,:]))),
    vec(meanpredictions[3,:]), "k--", label="std2")

  plot3D(vec(meanpredictions[1,:]),
    vec(meanpredictions[2,:]),
    vec(meanpredictions[3,:]) .+ 2*vec(sqrt.(covpredictions[3,:])), "k--", label="std2")

end
