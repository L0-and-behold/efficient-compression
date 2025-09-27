# this script can be run as a batch job on a cluster to instantiate the environment with the appropriate hardware (some packages like CUDA.jl require installation in presence of appropriate hardware and drivers to be installed correctly)

using Pkg
Pkg.activate(".")
Pkg.resolve()
Pkg.instantiate()

using CUDA

println("Active device: ", CUDA.device())

a = CUDA.fill(1.0f0, 3)
b = CUDA.fill(2.0f0, 3)
c = a .+ b

println("Result: ", c)

println("Done.")

using Lux, Random

model = Dense(10 => 5)

rng = Random.default_rng()
Random.seed!(rng, 0)
ps, st = Lux.setup(rng, model)

@show st

println("loaded Lux too")