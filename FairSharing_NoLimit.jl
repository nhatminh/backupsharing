using ECOS
using Gurobi
using SCS
using Ipopt
using Distributions
using HDF5
using JuMP
# using Convex
#using CoinOptServices

include("Setting.jl")
include("Common.jl")
include("Plot_figs.jl")


function disagreement_point1(op::MOperator, alpha = 2)
  prob = Model(solver=IpoptSolver(tol=5e-6, max_iter=10000, print_level =2))
  # prob = Model(solver=BonminNLSolver())
  # prob = Model(solver=OsilBonminSolver())

  @variable(prob, 0<= p[1:dim_x*dim_y, 1:Numb_BS] <= 1)
  @variable(prob, 0<= rho[1:Numb_BS] <= 0.99)
  @variable(prob, Psi[1:Numb_BS] )

  if alpha == 2
    @NLobjective(prob, Min,  sum(1/(1-rho[j]) for j=1:Numb_BS) )
  elseif alpha == 1
    @NLobjective(prob, Min,  sum(log(1-rho[j]) for j=1:Numb_BS) )
  elseif alpha == 0
    @NLobjective(prob, Min,  sum(1-rho[j] for j=1:Numb_BS) )
  end

  for x=1:dim_x*dim_y
    @constraint(prob, sum(p[x,j] for j =1:Numb_BS) == 1 )
  end

  for j =1:Numb_BS
      @constraint(prob, rho[j] == dot(op.system_density[:,j] , p[:,j] ) )
      @constraint(prob, Psi[j] == (1 - m) * rho[j] * Q +  m * Q )
      if CENTRALIZED_BS & (j == 3)
        @constraint(prob, Psi[j] <= CEN_B)
      else
        @constraint(prob, Psi[j] <= B)
      end

  end

  status = solve(prob)
  println("Dis Solve: ",status)
  check_solutions(op,getvalue(p),1)
end

function primal_update_ADMM(op::MOperator, dis, dual_vars, k)
  prob = Model(solver=IpoptSolver(tol=5e-6, max_iter=10000, print_level =1))

  @variable(prob, 0<= p[1:dim_x*dim_y, 1:Numb_BS] <= 1)
  @variable(prob, 0<= rho[1:Numb_BS] <= 0.99)
  @variable(prob, Psi[1:Numb_BS] )

  sum_psi = zeros(Numb_BS)
  for j=1:Numb_BS
      for i=1:Numb_Operators
        if(op.idx!=i)
          sum_psi[j] += Psi_arr[i,j,k]
        end
      end
  end

  @NLobjective( prob, Max,  log(dis - sum(1/(1-rho[j]) for j=1:Numb_BS)) -
                sum(rho1/2*(Psi[j]+sum_psi[j] + b_arr[j,k] - Numb_Operators* B - dual_vars[j]/rho1)^2  +
                rho2/2*(Psi[j] - Psi_arr[op.idx,j,k])^2 for j=1:Numb_BS) )

  for x=1:dim_x*dim_y
    @constraint(prob, sum(p[x,j] for j =1:Numb_BS) == 1 )
  end

  for j =1:Numb_BS
      @constraint(prob, rho[j] == dot(op.system_density[:,j] , p[:,j] ) )
      @constraint(prob, Psi[j] == (1 - m) * rho[j] * Q +  m * Q )
  end

  status = solve(prob)
  # println("Primal Solve: ",status)

  Psi_arr[op.idx,:,k+1] = getvalue(Psi)
  check_solutions(op,getvalue(p),2)
end

function b_update_ADMM(dual_vars, k)
  prob = Model(solver=IpoptSolver(tol=5e-6, max_iter=10000, print_level =2))

  @variable(prob, b[1:Numb_BS] >= 0)

  @NLobjective(prob, Min, sum(rho1/2*(sum(Psi_arr[i,j,k+1] for i=1:Numb_Operators) + b[j] - Numb_Operators* B - dual_vars[j]/rho1)^2 +
  rho2/2*(b[j] - b_arr[j,k])^2 for j=1:Numb_BS) )

  status = solve(prob)
  # println("Primal Solve: ",status)

  return getvalue(b)
end

function b_update_ADMM1(dual_vars, k)
  b = zeros(Numb_BS)
  for j = 1:Numb_BS
    b[j] = max(0,(rho1*( Numb_Operators* B - sum(Psi_arr[i,j,k+1] for i=1:Numb_Operators) ) + dual_vars[j] + rho2*b_arr[j,k] )/(rho1+rho2))
  end

  return b
end

function Nash_Bargaining_Centralized(ops, dis, alpha = 2)
  println("----- NASH-Bargaining ----")
  Numb_Participants = size(ops)[1]
  prob = Model(solver=IpoptSolver(tol=1e-9, max_iter=50000, print_level =2))

  @variable(prob, 0<= p[1:Numb_Participants,1:dim_x*dim_y, 1:Numb_BS] <= 1)
  @variable(prob, 0<= rho[1:Numb_Participants,1:Numb_BS] <= 0.99)

  if alpha == 2
    @NLobjective(prob, Max, sum( log(dis[i] - sum(1/(1-rho[i,j]) for j=1:Numb_BS)) for i=1:Numb_Participants ) )
  elseif alpha == 1
    @NLobjective(prob, Max, sum( log(dis[i] - sum(log(1-rho[i,j]) for j=1:Numb_BS)) for i=1:Numb_Participants ) )
  elseif alpha == 0
    @NLobjective(prob, Max, sum( log(dis[i] - sum(1-rho[i,j] for j=1:Numb_BS)) for i=1:Numb_Participants ) )
  end

  for i = 1:Numb_Participants
    for x=1:dim_x*dim_y
      @constraint(prob, sum(p[i,x,j] for j =1:Numb_BS) == 1 )
    end
    for j =1:Numb_BS
        @constraint(prob, rho[i,j] == dot(ops[i].system_density[:,j] , p[i,:,j] ) )
    end
  end

  for j =1:Numb_BS
    if CENTRALIZED_BS & (j == 3)
      @constraint(prob, sum((1 - m) * rho[i,j] * Q +  m * Q for i =1:Numb_Participants) <= (Numb_Participants * CEN_B))
    else
      @constraint(prob, sum((1 - m) * rho[i,j] * Q +  m * Q for i =1:Numb_Participants) <= (Numb_Participants * B ))
    end
  end

  status = solve(prob)
  println("Nash Centralize Solve: ",status)

  Costs    = zeros(Numb_Participants)
  Primals   = zeros(Numb_Participants, Numb_BS)
  psi       = zeros(Numb_Participants, Numb_BS)

  for i = 1:Numb_Participants
    Costs[i],Primals[i,:] = check_solutions(ops[i],getvalue(p)[i,:,:],2)
  end

  for j = 1:Numb_BS
      psi[:,j] = (1 - m) * Q * Primals[:,j] + m * Q
      delta_power = 0
      if  CENTRALIZED_BS & (j == 3)
        delta_power = Numb_Operators * CEN_B - sum(psi[:,j])
      else
        delta_power = Numb_Operators * B - sum(psi[:,j])
      end

      if delta_power < -1e-4
          println("Fail: Overload Power BS ",j ," :", delta_power)
      end
  end
  return Costs, Primals, getvalue(p)
end

Max_Iters = 90
b_arr= zeros(Numb_BS, Max_Iters+1)
Psi_arr= B*ones(Numb_Operators, Numb_BS, Max_Iters+1)

rho1 = 0.0002
Jacobian_step = 1.
delta = 0.00002
rho2 = (rho1*(Numb_Operators/(2-Jacobian_step)-1) + delta)

function Nash_Bargaining_Distributed(Operators, dis, alp = 2)
  println("----- NASH-Bargaining Distributed ----")
  # alpha = 1.15e-4 #Step_Size
  alpha = 1.e-4 #Step_Size
  # alpha = 5e-5 #Step_Size 4 Ops

  eps1= 5e-6 #72 iters
  # eps1= 2e-6

  Primals   = zeros(Numb_Operators, Numb_BS, Max_Iters)
  probs     = zeros(Numb_Operators,dim_x*dim_y, Numb_BS)
  Costs    = zeros(Numb_Operators, Max_Iters)
  sum_Psi       = zeros(Numb_BS)
  Dual_gradient = zeros(Numb_BS)
  Dual_vars  = zeros(Max_Iters+1, Numb_BS)
  Dual_vars[1,:] =  0.2 * ones(Numb_BS)
  Compute_Time = zeros(Numb_Operators,Max_Iters)

  for k = 1:Max_Iters
      # println("- Iteration ",k, " -")

      for i = 1:Numb_Operators
        tic()
        Costs[i,k], Primals[i,:,k], probs[i,:,:] = primal_update_ADMM(Operators[i],dis[i], Dual_vars[k,:],k)
        Compute_Time[i,k] = toq()
      end
      if DEBUG > 1
        println("Costs: ",Costs)
        println("Total Costs: ", sum(Costs))
      end

      b_arr[:,k+1] = b_update_ADMM1(Dual_vars[k,:], k)

      for j = 1:Numb_BS
          sum_Psi[j] = (1 - m) * Q * sum(Primals[:,j,k]) + Numb_Operators * m * Q
          Dual_gradient[j] = Numb_Operators * B - sum_Psi[j] - b_arr[j,k+1]

          if Dual_gradient[j] < -1e-4
              # println("Fail: Overload Power BS ",j ," :", Dual_gradient[j])
          end

          Dual_vars[k+1,j] = Dual_vars[k,j] + Jacobian_step * rho1 * Dual_gradient[j]
      end

      # println("Dual Vars:", Dual_vars[k+1,:])

      if (norm(Dual_vars[k+1,:] - Dual_vars[k,:])< eps1)
        println("Converge at ", k+1)
        println("MEAN: ", mean(Compute_Time))
        println("MEDIAN: ", median(Compute_Time))

        return Costs[:,1:k], Primals[:,:,1:k],probs, k, OK_CODE ;
        exit();
      end
  end

  figure(19)
  for j=1:Numb_BS
    plot(Dual_vars[:,j],label="BS_$j")
  end
  legend(loc="best")
  title("introduced variable")
  return Costs, Primals, probs, Max_Iters, NOT_CONVERGENCE ;
end

function read_arrival()
  total_arrival = zeros(Numb_Operators)
  arrival_pattern = zeros(Numb_Operators,dim_x,dim_y)
  for i = 1:Numb_Operators
    arrv_filename = string("Arrival_matrix_",i,".h5")
    h5open(arrv_filename, "r") do file
      arrival_pattern[i,:,:] = read(file, "arrvial")
      total_arrival[i] = read(file, "total_arrival")

    end
  end
  return total_arrival, arrival_pattern
end

function compute_avg_Delay(rhos)
  avg_Delay = zeros(Numb_Operators)
  for i = 1:Numb_Operators
    avg_numb_flows = 0
    for j = 1:Numb_BS
      avg_numb_flows += rhos[i,j]/(1-rhos[i,j])
    end
    avg_Delay[i] = avg_numb_flows /Arrivals[i]
  end
  return avg_Delay
end

function main()
  println("START")

  global Capacity = compute_capacity()
  total_arrival, Arrivals_Pattern = read_arrival()
  global Arrivals = total_arrival
  # plt_spatial_arrival(Arrivals_Pattern)

  # println(compute_Shannon_capcity([100., 200.]))

  Operators = Array(MOperator,Numb_Operators)
  disagreement_2 = zeros(Numb_Operators)
  disagreement_1 = zeros(Numb_Operators)
  disagreement_0 = zeros(Numb_Operators)
  dis_rhos2 = zeros(Numb_Operators,Numb_BS)
  dis_rhos1 = zeros(Numb_Operators,Numb_BS)
  dis_rhos0 = zeros(Numb_Operators,Numb_BS)
  dis_probs2 = zeros(Numb_Operators,dim_x*dim_y,Numb_BS)

  for i = 1:Numb_Operators
    operator = MOperator(lambs[i],i,[])
    system_load(operator)
    # println(operator.system_density)
    Operators[i] = operator
  end

  Numb_Change = 1 ### 7

  Powers_Change = Array([388, 400, 420, 440, 460, 480, 500])

  Total_Cost2 = zeros(3,Numb_Change)
  Total_Cost0 = zeros(3)
  Total_Cost1 = zeros(3)

  for c = 1:Numb_Change
    global B = Powers_Change[c]

    for i = 1:Numb_Operators
      disagreement_2[i], dis_rhos2[i,:], dis_probs2[i,:,:] = disagreement_point1(Operators[i], 2)
      # disagreement_1[i], dis_rhos1[i,:] , _ = disagreement_point1(Operators[i],1)
      # disagreement_0[i], dis_rhos0[i,:] , _ = disagreement_point1(Operators[i],0)
    end

    if c == 1
      if CENTRALIZED_BS == false
        Costs2, Primals2, _, k2, status2 = Nash_Bargaining_Distributed(Operators, disagreement_2, 2)

        println("** Distributed Costs: ",Costs2[:,end])
        Total_Cost2[1,c] = sum(Costs2[:,end])
        println("** Distributed Total: ", Total_Cost2[1,c])
        if status2 != OK_CODE
          println("Ditributed NASH: NOT CONVERGENCE")
        end
      end

      # Costs0_cen, Primals0 = Nash_Bargaining_Centralized(Operators, disagreement_0, 0)
      # println("** Nash Centralized Costs: ",Costs0_cen)
      # Total_Cost0[2] = sum(Costs0_cen)
      # println("** Nash Centralized Total 0: ", Total_Cost0[2])
      #
      # Costs1_cen, Primals1 = Nash_Bargaining_Centralized(Operators, disagreement_1, 0)
      # println("** Nash Centralized Costs: ",Costs1_cen)
      # Total_Cost1[2] = sum(Costs1_cen)
      # println("** Nash Centralized Total 1: ", Total_Cost1[2] )
    end

    println("** Disagreement: ", disagreement_2)
    Total_Cost2[1,c] = sum(disagreement_2)
    println("** Total Disagreement: ", Total_Cost2[2,c])

    Costs2_cen, _, probs2 = Nash_Bargaining_Centralized(Operators, disagreement_2, 2)
    println("** Nash Centralized Costs: ",Costs2_cen)
    Total_Cost2[2,c] = sum(Costs2_cen)
    println("** Nash Centralized Total 2: ", sum(Costs2_cen))

    # Costs2 = Join_Centralized(Operators, disagreement)
    # println("** Join Centralized Costs: ",Costs2)
    # Total_Cost[3,c] = sum(Costs2)
    # println("** Join Centralized Total: ", Total_Cost[3,c])

    if c == 1
      if CENTRALIZED_BS == false
        h5open("decentralized_results6.h5", "w") do file
          write(file, "dis_cost",disagreement_2)
          write(file, "dis_rhos",dis_rhos2)
          write(file, "rho_decentralized",Primals2)
          write(file, "Costs2",Costs2)
          write(file, "total_cost",sum(Costs2,1))
          write(file, "total_cost_cen",Costs2_cen)
        end

        # plt_convergence1(k2, Costs2, Primals2, disagreement_2, dis_rhos2, Costs2_cen)
        # plt_comparison(Primals2[:,:,end],Costs2[:,end],disagreement_2,dis_rhos2)
        # plt_delay_comparison(compute_avg_Delay(Primals2[:,:,end]),compute_avg_Delay(dis_rhos2))

      end
      # plt_prob_comparison(probs2, dis_probs2)

      # plt_obj_comparison(Costs1_cen, Primals1, disagreement_1, dis_rhos1)
      # plt_obj_comparison(Costs0_cen, Primals0, disagreement_0, dis_rhos0)
    end
  end
  figure(20)
  for j=1:Numb_BS
    plot(b_arr[j,:],label="BS_$j")
  end
  legend(loc="best")
  title("introduced variable")

  # plt_total_cost_limited(Powers_Change, Total_Cost2)
end

main()
