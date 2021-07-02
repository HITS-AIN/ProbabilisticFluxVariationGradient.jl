"""
    plotresults(;flux=flux, posterior=posterior, [σ2=1e-4*ones(size(flux, 1))])

"""
function plotresults(; flux=flux, posterior=posterior, g=g, randx = randx, σ2=1e-4*ones(size(flux, 1)), showstd=false)

  linemeansigma = marginaline(posterior, σ2)

  sampleinter   = [randx() for i=1:20]


  # project flux obs on line to find minimum, maximum t for plotting purposes

  D = round(Int, length(posterior)/2)

  ā, b̄ = mean(posterior)[1:D], mean(posterior)[D+1:2*D]

  proj(x) = dot(ā, x-b̄) / dot(ā, ā)


  tproj = [proj(flux[:,i]) for i=1:size(flux, 2)]

  tinter = proj(mean(sampleinter))

  tmin, tmax = min(tinter,minimum(tproj)), max(tinter, maximum(tproj))

  tmin, tmax = sign(tmin) * abs(tmin * 1.1), sign(tmax) * abs(tmax * 1.1) # make them a bit longer

  trange = collect(LinRange(tmin, tmax, 400))


  predictions     = map(linemeansigma, trange)

  meanpredictions = reduce(hcat, [p[1] for p in predictions])

  covpredictions  = reduce(hcat, [diag(p[2]) for p in predictions])


  figure()

  plot3D(flux[1,:], flux[2,:], flux[3,:], "mo", label="observed total flux", alpha=0.75)

  # plot3D(vec(meanpredictions[1,:]), vec(meanpredictions[2,:]), vec(meanpredictions[3,:]), "m-", label="mean fit")

  tmp = reduce(hcat, sampleinter)
  plot3D(tmp[1,:],tmp[2,:],tmp[3,:], "yo", label = "intersection samples", markersize=6)

  plotvector(g/norm(g) * norm(ā*tinter .+ b̄)*1.2, "g", "galaxy vector")

  if showstd

      circle = [cos.(LinRange(0.0,2π, 100)) sin.(LinRange(0.0,2π, 100))]'

      for t in 1:length(trange)

            μ = predictions[t][1]
            c = predictions[t][2]

            # principal components
            E = eigen(c)

            vaxis  = [E.vectors[:,1]*(E.values[1])*20 E.vectors[:,2]*(E.values[2])*20]
            v     = eigen(c).vectors[:, 3]


            pnts = vaxis*circle
            plot3D(pnts[1,:].+μ[1],pnts[2,:].+μ[2],pnts[3,:].+μ[3],"-c", alpha=0.1)

      end


      # plot3D(vec(meanpredictions[1,:]) .+ 2*vec(sqrt.(covpredictions[1,:])),
      #   vec(meanpredictions[2,:]),
      #   vec(meanpredictions[3,:]), "m-", label="2std", alpha=0.5)
      #
      # plot3D(vec(meanpredictions[1,:]),
      #   vec(meanpredictions[2,:] .+ 2*vec(sqrt.(covpredictions[2,:]))),
      #   vec(meanpredictions[3,:]), "m-", label="2std", alpha=0.5)
      #
      # plot3D(vec(meanpredictions[1,:]),
      #   vec(meanpredictions[2,:]),
      #   vec(meanpredictions[3,:]) .+ 2*vec(sqrt.(covpredictions[3,:])), "m-", label="2std", alpha=0.5)

  end


  # plot sample lines


  legend()

    xlim(0.0, (xlim()[2]))
    ylim(0.0, (ylim()[2]))
    zlim(0.0, (zlim()[2]))

    xlabel("band at 3670")
    ylabel("band at 4826")
    zlabel("band at 6223")


  nothing

end
