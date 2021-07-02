########################################################
function lsststd(; zpwave=3670.7, z=0.5, lum=5e43) # Angstrom, unitless, erg/s
########################################################

    #Define QSO variability-luminosity-redshift relation
    #Morganson et al. (2014) ApJ 784:92
    #input luminosity in erg/s
    #See also Roberto Cid Fernandes, Jr, et al. (1996)
    angstonm = 0.1
    zpwave   = zpwave*angstonm
    sigmavar = 0.079*(1+z)^0.15 * (lum/1e46)^-0.2 * (zpwave/1000.)^-0.44
    return sigmavar
end



########################################################
function setlsstwavelengths(λ = [3670.7, 4826.9, 6223.2, 7546.0, 8690.0, 9710.3]) # Angstrom
########################################################

    lsstwaves() =  λ

end


########################################################
function unitgalaxyvector(T)
########################################################


    G = galaxyvector(T)

    G / norm(G)

end


########################################################
function galaxyvector(T)
########################################################

    G = map(λ -> B(; T=T, λ=λ), lsstwaves())

end



########################################################
function B(; λ=λ, T=T)   # λ is in Angstrom, T in Kelvin
########################################################

    h = 6.6261e-27      # Planck constant    [cgs]
    c = 2.9979245800e10 # Speed of light     [cm/s]
    k = 1.380648813e-16 # Boltzmann constant [cgs]

    wavcm = λ*1e-8      # convert Ang to cm
    nu    = c/wavcm     # convert wavelength to frequency [Hz]

    #--frequency units--
    # emittence [erg/sec/cm^2/Hz]

    In   = 2.0*pi*h*nu^3.0/c^2 * (1.0/(exp(h*nu/(k*T)) - 1.0))

    ImJy = In / 1.0e-26

    return ImJy # Flux in absolute units of milli jansky
                # Doesn't not contain observational effects
end
