
plotvector(x, str="k-",label="", alpha=1.0) = plot3D([0;x[1]],[0;x[2]],[0;x[3]], str,alpha=alpha,label=label)


proj(v, x) = vec( dot(v, x) / dot(v, v) * v )


unitvector(x) = x / norm(x)
