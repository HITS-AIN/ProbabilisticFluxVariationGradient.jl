function numerical_gradient(feval, x, dx=0.00001)

  g = zeros(size(x))

  for ii = 1:length(x)

    x[ii] = x[ii] + dx

    D1 = feval(x)

    x[ii] = x[ii] - 2*dx

    D2 = feval(x)

    x[ii] = x[ii] + dx

    g[ii] = (D1-D2)/(2*dx)

  end

  return g

end
