abstract CombinedKernel <: Kernel

immutable KernelSum <: CombinedKernel
    a::Kernel
    b::Kernel
end
length(k::KernelSum) = length(k.a) + length(k.b)

+(a::Kernel, b::Kernel) = KernelSum(a, b)

cov{T}(k::KernelSum, x1::T, x2::T) = cov(k.a, x1, x2) + cov(k.b, x1, x2)

covderiv{T}(k::KernelSum, i::Integer, x1::T, x2::T) = (
    i <= length(k.a) ?
        covderiv(k.a, i, x1, x2)
    :
        covderiv(k.b, i - length(k.a), x1, x2)
)


immutable KernelProduct <: CombinedKernel
    a::Kernel
    b::Kernel
end
length(k::KernelProduct) = length(k.a) + length(k.b)

*(a::Kernel, b::Kernel) = KernelProduct(a, b)

cov{T}(k::KernelProduct, x1::T, x2::T) = cov(k.a, x1, x2) * cov(k.b, x1, x2)

covderiv{T}(k::KernelProduct, i::Integer, x1::T, x2::T) = (
    i <= length(k.a) ?
        covderiv(k.a, i, x1, x2) * cov(k.b, x1, x2)
    :
        cov(k.a, x1, x2) * covderiv(k.b, i - length(k.a), x1, x2)
)

