abstract Kernel

typealias GPInput{T<:Number} Union(Vector{T}, Matrix{T})
typealias GPOutput{T<:Number} Vector{T}

# for the naive implementation
type GaussianProcess{N<:Number}
    kernel::Kernel
    noise::N # standard deviation of noise
    xs::GPInput{N} # training input points
    ys::GPOutput{N} # training function values
    kxxI::Matrix{N} # inverse of (training, training) covariance matrix
end


# for the implementation with Cholesky decomposition
type GaussianProcess2{N<:Number}
    kernel::Kernel
    noise::N # standard deviation of noise
    xs::GPInput{N} # training input points
    ys::GPOutput{N} # training function values
    L::Matrix{N} # Cholesky factor of (training, training) covariance matrix
    α::Vector{N} # coefficients
end
