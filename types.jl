abstract Kernel # covariance kernel
# concrete subtypes should implement the following methods:
# length(k::<Kernel>) = the number of hyperparameters, needed for derivatives
# cov(k::<Kernel>, x1, x2) = the covariance function of x1 and x2
# covderiv(k::<Kernel>, i::Integer, x1, x2) = derivative of the covariance w.r.t. its i'th hyperparameter
# cov_d(k::<Kernel>, x1, x2) = derivative of the covariance w.r.t. x1
# covd_(k::<Kernel>, x1, x2) = derivative of the covariance w.r.t. x2
# covdd(k::<Kernel>, x1, x2) = second (mixed) derivative of the covariance w.r.t. x1 and x2


typealias GPInput{T<:Number} Union(Vector{T}, Matrix{T})
typealias GPOutput{T<:Number} Vector{T}

# for the naive implementation
type GaussianProcess{N<:Number}
    kernel::Kernel
    noise::N # standard deviation of noise
    xs::GPInput{N} # training input points
    ys::GPOutput{N} # training function values
    kxxI::Matrix{N} # inverse of (training, training) covariance matrix
    α::Vector{N} # coefficients
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
