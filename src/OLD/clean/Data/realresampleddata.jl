########################################################
function realresampleddataforpca()
########################################################


    A = h5read("Data/AGNs.hdf5","simulations/Mrk509/AGNGALresamp")

    x = Matrix(A[1:3, 2, :]')  # select filters out of available six

    Σ =  Matrix(A[1:3, 3, :]')

    return x, Σ

end


########################################################
function realresampleddataforblr()
########################################################


    A = h5read("Data/AGNs.hdf5","simulations/Mrk509/AGNGALresamp")

    x = Matrix(A[1:6, 2, :]')  # select filters out of available six

    srt = sortperm(vec(A[1,1,:])) # sort according to time,
                                  # not necessary, just convenient for plotting

    idx  = [ones(Int, 110)   ; ones(Int, 110)*2 ; ones(Int, 110)*3 ; ones(Int, 110)*4 ; ones(Int, 110)*5 ; ones(Int, 110)*6]
    flux = [vec(x[srt,1])    ; vec(x[srt,2])    ; vec(x[srt,3])    ; vec(x[srt,4])    ; vec(x[srt,5])    ; vec(x[srt,6])]
    Σ    = [vec(A[1, 3, srt]); vec(A[2, 3, srt]); vec(A[3, 3, srt]); vec(A[4, 3, srt]); vec(A[5, 3, srt]); vec(A[6, 3, srt])];

    t    = A[1,1,srt] .- minimum(A[1,1,srt]) # no need to subtract,
                                             # just convenient that times starts at 0

    idx1 = findall(idx .== 1)
    idx2 = findall(idx .== 2)
    idx3 = findall(idx .== 3)
    idx4 = findall(idx .== 4)
    idx5 = findall(idx .== 5)
    idx6 = findall(idx .== 6)

    figure(1)
    cla()
    title("plotting three first LSST filters")
    plot3D(flux[idx1], flux[idx2], flux[idx3], "bo", alpha=0.4)


    figure(2)
    cla()
    plot(t, flux[idx1], ".", label = "1 - u")
    plot(t, flux[idx2], ".", label = "2 - g")
    plot(t, flux[idx3], ".", label = "3 - r")
    plot(t, flux[idx4], ".", label = "4 - i")
    plot(t, flux[idx5], ".", label = "5 - y")
    plot(t, flux[idx6], ".", label = "6 - z")

    legend()

    # return [t;t;t; t;t;t], flux, Σ, idx
    return [t;t;t], flux[1:330], Σ[1:330], idx[1:330]

end
