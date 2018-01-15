using ECOS
using Gurobi
using GLPK
using SCS
using Ipopt
using JuMP

# m = Model(solver=GurobiSolver())
# m = Model(solver=ECOSSolver())
# m = Model(solver=SCSSolver())
m = Model(solver=IpoptSolver())
# m = Model()


@variable(m, 0.5 <= x[1:2] <= 0.9)

# @NLobjective(m, Max, log(5 - ( 1/(1-x[1]) + 1/(1-x[2])) ) )
@NLobjective(m, Max, log(5 - sum{1/(1-x[i]), i=1:2} ))
# @objective(m,maximize(log(5 - ( 1/(1-x[1]) + 1/(1-x[2])) ) )

@constraint(m, 2*x .<= 1.5)

status = solve(m)


println("Objective value: ", getobjectivevalue(m))
println("x = ", getvalue(x))
