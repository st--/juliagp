module GP

using Distributions # for drawing from a multivariate Gaussian

# import builtins we want to extend without overwriting
import Base.length
import Base.cov

const jitterdefault = 1e-6

include("types.jl")
include("covcombine.jl")
include("covariances.jl")
include("helper.jl")
include("gpbase.jl")
include("gp.jl")
include("gp2.jl")

end
