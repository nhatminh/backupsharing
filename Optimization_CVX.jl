using Convex
using ECOS
using Gurobi
using GLPK
using SCS

x= Variable(2)

p = maximize(log(5 - ( 1/(1-x[1]) + 1/(1-x[2])) ) )
p.constraints += [x <= 0.9, x >= 0.5]



#     solve!(p)
solve!(p, GurobiSolver())
# solve!(p, GLPKSolverMIP())
# solve!(p, ECOSSolver())
# solve!(p, SCSSolver())

println(x.value)
println(p.optval)
