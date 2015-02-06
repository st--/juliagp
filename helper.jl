# functions that transparently work over both vectors and 2D arrays

# returns number of input points
inputlength(xs::Vector) = length(xs)
inputlength{T}(xs::Array{T,2}) = size(xs,1)

# returns i'th input point
inputindex(xs::Vector, i::Integer) = xs[i]
inputindex{T}(xs::Array{T,2}, i::Integer) = vec(xs[i,:])

# returns identity matrix corresponding to number of input points
neye(xs) = eye(inputlength(xs))

function logdet{N<:Number}(A::Matrix{N})
    L = chol(A)
    return 2sum(log(diag(L)))
end
