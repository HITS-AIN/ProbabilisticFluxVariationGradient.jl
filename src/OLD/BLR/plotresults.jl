#######################################################################
function plotresults(flux, idx, randpcaline, S)
#######################################################################

    figure(2222)

    cla()

    #-------------------------------------------------
    # Plot data
    #-------------------------------------------------
    F = [flux[i .== idx] for i in 1:3]

    plot3D(F[1], F[2], F[3], "bo", label = "observed data in u,g,r")

    x = reduce(hcat, F)

    minT, maxT = Inf, -Inf

    proj(v, x) = dot(v, x) / dot(v, v)

    for _  in 1:S

        # T, s = randgalaxyline()
        #
        # minT, maxT = min(minT, T), max(maxT, T)

        w, μ = randpcaline()

        z = [proj(w, x[i,:] - μ) for i=1:size(x,1)]

        z = sort(z)

        rangeline = maximum(z) - minimum(z)

        zgrid = collect(LinRange(minimum(z)-rangeline*0.0, maximum(z)+rangeline*0.0, 100))


        fwd(z::Real) = w*z + μ


        PCAline = reduce(hcat, fwd.(zgrid))

        @show size(PCAline)

        plot3D(PCAline[1,:], PCAline[2,:], PCAline[3,:], "-")

        # plot3D(gline[1,:], gline[2,:], gline[3,:], "r-")

        # x0 = s * G

        # plot3D([x0[1]],[x0[2]],[x0[3]], "co")

    end

    # title(@sprintf("temperatures range from %f to %f", minT, maxT))

    xlabel("u"); ylabel("g"); zlabel("r")

end
