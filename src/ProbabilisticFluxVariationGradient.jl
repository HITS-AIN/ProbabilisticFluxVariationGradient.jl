module ProbabilisticFluxVariationGradient

    using PyPlot

    using Suppressor, Printf, ProgressMeter

    using LinearAlgebra, Distributions, Statistics, StatsFuns, Random

    using Distances

    using Optim

    using Interpolations

    using PDMats

    using ApproximateVI

    using ForwardDiff

    export bpca, noisyintersectionvi, mockline, bpcaAIS

    # Utilities

    include("UTIL/astroutil.jl")
    include("UTIL/gputil.jl")
    include("UTIL/smallutil.jl")
    include("UTIL/plotresults.jl")
    lsstwaves = setlsstwavelengths()

    # Synthetic data generation

    include("SyntheticDataset/mockline.jl")
    include("SyntheticDataset/simulatedata.jl")

    # GP approach

    # include("GP/complete_lower_bound.jl")
    # include("GP/exact_expectations.jl")
    # include("GP/complete_lower_bound_grad.jl")
    # include("GP/exact_expectations_grad.jl")
    # include("GP/gpmodel.jl")
    # include("GP/gpmodelVIexact.jl")

    # Line intersection

    include("LineIntersection/noisyintersection.jl")
    include("LineIntersection/linemarginal.jl")

    # Probabilistic principal component algorithms

    include("PPCA/bpcaAIS.jl")
    include("PPCA/bpca.jl")
    include("PPCA/ppca.jl")

end
