### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ 3da9f1a8-f65a-11ea-2a34-c7eb4db8cd54
using Plots, Distributions, LinearAlgebra,Distributions, ProgressMeter

# ╔═╡ 316f1058-f65a-11ea-33d8-997377fba2b0
md"Preliminary: load packages, setup latex macro"

# ╔═╡ b37d84e0-f659-11ea-0e0e-1d14d3d5aba5
md"# Calculating the ELBO for PFVG
Here we detail the various calculation we need."

# ╔═╡ 5cabfa9e-f65a-11ea-0695-c73c26cf3342
md"## Model
$$\int \mathcal{N}(\mathbf{y}|\mathbf{A}\mathbf{f} - \mathbf{b}, \mathbf{S}) \mathcal{N}(\mathbf{f}|\mathbf{0},\mathbf{K}) \ \mathbf{df} \ \mathbf{dA}\ \mathbf{db}$$
where the matrix $$\mathbf{A}(\mathbf{\alpha})$$ and vector $$\mathbf{b}(\mathbf{\beta})$$ are function of the respective parameters.
"

# ╔═╡ d24a45be-f65b-11ea-2c07-d19988003b56
md"## Lower bound
$$\int q(\mathbf{f}) q(\mathbf{\alpha}) q(\mathbf{\beta}) \log\mathcal{N}(\mathbf{y}|\mathbf{A}\mathbf{f} - \mathbf{b}, \mathbf{S}) \mathcal{N}(\mathbf{f}|\mathbf{0},\mathbf{K}) \ \mathbf{df} \ \mathbf{dA}\ \mathbf{db} + \mathcal{H}[q_f] + \mathcal{H}[q_\alpha] + \mathcal{H}[q_\beta]$$
"

# ╔═╡ 3e54363e-f65c-11ea-0150-7d09f848a86e
md"## Inside the log
$$-\frac{1}{2}(\mathbf{y} - \mathbf{Af}-\mathbf{b})^T\mathbf{S}^{-1} (\mathbf{y} - \mathbf{Af}-\mathbf{b}) - \frac{1}{2}\log|\mathbf{K}| -\frac{1}{2}\mathbf{f}^T\mathbf{K}^{-1}\mathbf{f}$$
"

# ╔═╡ ab41e982-f65c-11ea-26f9-01f9de7053a3
md"## Work out posterior $$q(\mathbf{f})$$ of GP latent variables

Keep only relevant terms:

$$-\frac{1}{2}(-2\mathbf{y}^T\mathbf{S}^{-1}\mathbf{Af} + 2\mathbf{b}^T\mathbf{S}^{-1}\mathbf{Af} + \mathbf{f^T A^T S^{-1} A f})
-\frac{1}{2} \mathbf{f}^T\mathbf{K}^{-1}\mathbf{f}$$


We recognise the quadratic which leads to a Gaussian posterior:

$$-\frac{1}{2}(-2(\mathbf{-y+b})^T\mathbf{S}^{-1}\mathbf{Af}  + \mathbf{f^T A^T S^{-1} A f})
-\frac{1}{2} \mathbf{f}^T\mathbf{K}^{-1}\mathbf{f}$$

$$(\mathbf{y-b})^T\mathbf{S}^{-1}\mathbf{Af}  - \frac{1}{2} \mathbf{f^T (A^T S^{-1} A +K^{-1}) f}$$

The posterior reads:

$$q(\mathbf{f}) = \mathcal{N}(\mathbf{f} |  \mathbf{\Sigma_f} \ \mathbf{A^T S^{-1}(y-b)}, \ \underbrace{\mathbf{A^T S^{-1} A +K^{-1}}}_{\mathbf{\Sigma_f}} )$$

"

# ╔═╡ fe789bd2-f66e-11ea-3400-61f55bbf640b


# ╔═╡ Cell order:
# ╟─316f1058-f65a-11ea-33d8-997377fba2b0
# ╠═3da9f1a8-f65a-11ea-2a34-c7eb4db8cd54
# ╟─b37d84e0-f659-11ea-0e0e-1d14d3d5aba5
# ╟─5cabfa9e-f65a-11ea-0695-c73c26cf3342
# ╟─d24a45be-f65b-11ea-2c07-d19988003b56
# ╟─3e54363e-f65c-11ea-0150-7d09f848a86e
# ╟─ab41e982-f65c-11ea-26f9-01f9de7053a3
# ╠═fe789bd2-f66e-11ea-3400-61f55bbf640b
