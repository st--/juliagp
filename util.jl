# compares numerical derivative (f(p+ϵ)-f(p-ϵ))/2ϵ with symbolic derivative to ensure correctness
function checkderiv(f, fprime, p, i; eps=1e-6)
    pplus = copy(p); pplus[i] += eps
    pminus = copy(p); pminus[i] -= eps
    return fprime(p...,i) - (f(pplus...)-f(pminus...))/2eps
end

checkderiv(f, fprime, p) = [checkderiv(f, fprime, p, i) for i=1:length(p)]


# returns function that returns return value of fprime as second Vector argument instead of actual return value
function make_storage(fprime)
    function grad!(x::Vector, storage::Vector)
        grad = fprime(x)
        for i=1:length(grad)
            storage[i] = grad[i]
        end
    end
    return grad!
end
