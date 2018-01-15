global system_density_actual = zeros(Numb_BS,T_req,dim_x*dim_y,Numb_BS)

function disagreement_point(op::MPC_MOperator)
  p   = Convex.Variable(dim_x*dim_y, Numb_BS)
  rho = Convex.Variable(Numb_BS)
  d   = Convex.Variable(Numb_BS)

  prob = minimize(sum(1./d))
  prob.constraints  += [0 <= p, p <= 1, rho >= 0, rho <= 0.99, d >= 0.01, d <= 1 ]

  for x=1:(dim_x*dim_y)
      prob.constraints += [sum(p[x,:]) == 1 ]
  end

  for j =1:Numb_BS
      prob.constraints += [rho[j] == vecdot(op.system_density[1,:,j], p[:,j]) ]
      prob.constraints += [(1 - m) * rho[j] * Q +  m * Q <= B ]
      prob.constraints += [d[j] == 1 - rho[j]]
  end

  # solve!(prob, GurobiSolver(),verbose=false)
  # solve!(prob, MosekSolver(),verbose=false)
  solve!(prob, ECOSSolver(verbose=false, max_iters = 1000),verbose=false)
  # solve!(prob, SCSSolver(verbose=false),verbose=false)
  println("Greedy Solve: ", prob.status)

  Cost = 0
  psi = zeros(Numb_BS)
  if prob.status == :Optimal
    Cost, psi = MPC_check_solutions(op,p.value,m*Q*ones(Numb_BS),1,1)
    return Cost, psi, OK_CODE
  else
    return Cost, psi, ERROR_CODE
  end
end

function MPC_check_solutions(op::MPC_MOperator,p,Psi_inf, mode,t)
  # println("Psi Vector",Psi_inf)
  for x=1:(dim_x*dim_y)
    if abs(sum(p[x,:]) - 1) > 5e-4
       if (op.system_density[t,x,1] > 0)
          println("Fail: Association at ",t," with ", x, " : ", sum(p[x,:]) )
       end
    end
  end

  rho = zeros(Numb_BS)
  # rho_actual = zeros(Numb_BS)
  psi = zeros(Numb_BS)
  total_Cost = 0

  for j =1:Numb_BS
      rho[j] = vecdot(op.system_density[t,:,j], p[:,j])
      # rho_actual[j] = vecdot(system_density_actual[op.idx,t,:,j], p[:,j])
      psi[j] = (1 - m) * rho[j] * Q +  Psi_inf[j]
      # total_Cost += 1./(1-rho_actual[j])
      total_Cost += 1./(1-rho[j])
  end

  if DEBUG > 1   println("rho: ", rho);  end

  if mode == 1
      return total_Cost, psi
  else
      return total_Cost, rho
  end
end

function MPC_traffic_density_region(op::MPC_MOperator)
  traffic_matrix = zeros(dim_x,dim_y)

  # for x = 1:dim_x
  #     for y = 1:dim_y
  #         d = abs(y-x)
  #
  #         if op.idx == 2
  #           traffic_matrix[x,y]  = traffic_density_unitarea(op.init_lamb)
  #
  #         # if op.idx < 3
  #         elseif op.idx < 2
  #           if d > 7
  #               traffic_matrix[x,y] = traffic_density_unitarea(op.init_lamb + 8 * increased_arr_rate)
  #           elseif d > 4
  #               traffic_matrix[x,y] = traffic_density_unitarea(op.init_lamb + 5 * increased_arr_rate)
  #           elseif d > 1
  #               traffic_matrix[x,y] = traffic_density_unitarea(op.init_lamb + increased_arr_rate)
  #           else
  #              traffic_matrix[x,y] = traffic_density_unitarea(op.init_lamb)
  #           end
  #
  #         else op.idx > 2
  #           if d < 2
  #               traffic_matrix[x,y] = traffic_density_unitarea(op.init_lamb + 6 * increased_arr_rate)
  #           elseif d < 5
  #               traffic_matrix[x,y] = traffic_density_unitarea(op.init_lamb + 3 * increased_arr_rate)
  #           elseif d < 8
  #               traffic_matrix[x,y] = traffic_density_unitarea(op.init_lamb + increased_arr_rate)
  #           else
  #              traffic_matrix[x,y] = traffic_density_unitarea(op.init_lamb)
  #           end
  #         end
  #     end
  # end
  arrival_matrix = zeros(dim_x,dim_y)
  for x = 1:dim_x
      for y = 1:dim_y
          d = abs(y-x)
          if op.idx == 2
            if ((x>=4) & (y>=4) & (x<=7) & (y<=7))
              arrival_matrix[x,y]  = op.init_lamb + 7.
            else
              arrival_matrix[x,y]  = op.init_lamb - 7.
            end

          # if op.idx < 3
          elseif op.idx < 2
            if d > 7
                arrival_matrix[x,y] = op.init_lamb + 9 * increased_arr_rate
            elseif d > 4
                arrival_matrix[x,y] = op.init_lamb + 4 * increased_arr_rate
            elseif d > 1
                arrival_matrix[x,y] = op.init_lamb + increased_arr_rate
            else
                arrival_matrix[x,y] = op.init_lamb
            end

          else op.idx > 2
            if d < 2
                arrival_matrix[x,y] = op.init_lamb + 6 * increased_arr_rate
            elseif d < 5
                arrival_matrix[x,y] = op.init_lamb + 3 * increased_arr_rate
            elseif d < 8
                arrival_matrix[x,y] = op.init_lamb + increased_arr_rate
            else
                arrival_matrix[x,y] = op.init_lamb
            end

          end

          traffic_matrix[x,y] = traffic_density_unitarea(arrival_matrix[x,y])
      end
  end
  return traffic_matrix
end

function MPC_traffic_density(op::MPC_MOperator)
  filename = string("MPC_Traffic_pattern_",op.idx,".h5")

  if MPC_REUSED_TRAFFIC
    h5open(filename, "r") do file
      op.traffics = read(file, "traffic")  # alternatively, say "@write file A"
    end
  else

    traffics = MPC_traffic_density_region(op)
    full_traffics = zeros(T_req,dim_x,dim_y)

    #Should be randomly
    reduction_seq = ones(T_req)
    if TRAFFIC_CONTROL == 2
      reduction_seq = 0*reduction_seq
    elseif TRAFFIC_CONTROL == 1
      seq = Array([0.,1.,2.,3.])
      reduction_seq = append!(seq,3.5*ones(T_req - 4))
    else
      reduction_seq = 3.5*reduction_seq
    end

    reduction = 10000 * reduction_seq

    # reduction_seq = ones(T_req)
    # if op.idx == 2
    #   reduction_seq = 0*reduction_seq
    # elseif op.idx == 3
    #   seq = Array([0.,1.,2.,3.])
    #   reduction_seq = append!(seq,3.5*ones(T_req - 4))
    # elseif op.idx == 1
    #   seq = Array([0.,-1.,-2.,-3.])
    #   reduction_seq = append!(seq,-3.5*ones(T_req - 4))
    # end
    #
    # reduction = 10000 * reduction_seq


    for t =1:T_req
      for x = 1:dim_x
        for y = 1:dim_y
          full_traffics[t,x,y] = max(traffics[x,y] - reduction[t], 0.)
        end
      end
    end
    op.traffics = full_traffics
    # println(op.traffics)

    h5open(filename, "w") do file
      write(file, "traffic", full_traffics)  # alternatively, say "@write file A"
    end
  end
end

function MPC_traffic_prediction(op::MPC_MOperator,curr_time, T_pred)
  pred_traffics = zeros(T_pred,dim_x,dim_y)
  pred_traffics[1,:,:] = op.traffics[curr_time,:,:]

  filename = string("MPC_Traffic_pattern_",op.idx,".h5")
  if(MPC_REUSED_TRAFFIC)
    h5open(filename, "r") do file
      pred_traffics = read(file, "traffic$curr_time")
    end
    return pred_traffics
  end

  for t =2:T_pred
    # pred_traffics[t,:,:] = op.traffics[curr_time+t-1,:,:] + rand(Normal(0,0.1*(1+log2(t))))
    error  = op.traffics[curr_time+t-1,:,:]*0.1*rand(Normal(0,0.1*log(t)))

    for x = 1:dim_x
      for y = 1:dim_y
          # pred_traffics[t,x,y] = op.traffics[curr_time+t-1,x,y] + 0
          pred_traffics[t,x,y] = max(op.traffics[curr_time+t-1,x,y] + error[x,y], 0.)
      end
    end
  end

  h5open(filename, "r+") do file
    write(file, "traffic$curr_time",pred_traffics)
  end

  return pred_traffics
end

function MPC_system_load(op::MPC_MOperator,curr_time, T_pred, actual_mode = false)
  traffics = zeros(T_pred,dim_x,dim_y)
  system_density = zeros(T_pred,dim_x*dim_y, Numb_BS)

  if actual_mode
    traffics = op.traffics
  else
    traffics = MPC_traffic_prediction(op,curr_time, T_pred)
  end

  for t = 1:T_pred
    for x = 1:dim_x
        for y = 1:dim_y
            system_density[t,(x-1)*dim_x + y,:] = traffics[t,x,y] ./ Capacity[(x-1)*dim_x + y,:]
        end
    end
  end

  if actual_mode
    system_density_actual[op.idx,:,:,:] = system_density
  end
  op.system_density = system_density
end


function primal_update(op::MPC_MOperator,dis, dual_vars)
  # https://github.com/roboptim/roboptim-core-plugin-ipopt/wiki
  # https://projects.coin-or.org/Ipopt/wiki/HintsAndTricks
  #"nlp_scaling_method=none mu_init=1e-2 max_iter=500"
  # prob = Model(solver=IpoptSolver(tol=1e-8, max_iter=100000, print_level =1,nlp_scaling_method ="gradient-based",
  #             nlp_scaling_min_value=1e-9))
  prob = Model(solver=IpoptSolver(tol=1e-7, max_iter=10000, print_level =1))
  # prob = Model(solver=BonminNLSolver())
  # prob = Model(solver=OsilBonminSolver())

  @variable(prob, 0<= p[1:dim_x*dim_y, 1:Numb_BS] <= 1)
  @variable(prob, 0<= rho[1:Numb_BS] <= 0.99)
  # @variable(prob, 0.01<= d[1:Numb_BS] <= 1 )
  @variable(prob, Psi[1:Numb_BS] )

  @NLobjective(prob, Max,  log(dis - sum(1/(1-rho[j]) for j=1:Numb_BS)) -
              sum(dual_vars[j] * (Psi[j] - B) for j=1:Numb_BS) )
              # sum{dual_vars[j] * Psi[j], j=1:Numb_BS} )

  for x=1:dim_x*dim_y
    @constraint(prob, sum(p[x,j] for j =1:Numb_BS) == 1 )
  end

  for j =1:Numb_BS
      @constraint(prob, rho[j] == dot(op.system_density[1,:,j] , p[:,j] ) )
      # @constraint(prob, rho[j] == sum{op.system_density[x,j] * p[x,j], x=1:dim_x*dim_y} )
      @constraint(prob, Psi[j] == (1 - m) * rho[j] * Q +  m * Q )
      # @constraint(prob, d[j] == 1 - rho[j] )
  end

  status = solve(prob)
  println("Primal Solve: ",status)

  return MPC_check_solutions(op,getvalue(p),2,1)
end

# function Nash_Bargaining
function Nash_Bargaining_Distributed(Operators, dis)
  println("----- NASH-Bargaining Distributed ----")
  Numb_Participants = size(Operators)[1]

  # alpha = 0.0004 #Step_Size
  alpha = 5e-5
  eps1= 1e-6
  Max_Iters = 50
  Primals   = zeros(Numb_Participants, Numb_BS)
  Costs    = zeros(Numb_Participants)
  sum_Psi   = zeros(Numb_BS)
  psi       = zeros(Numb_Participants,Numb_BS)
  Dual_gradient = zeros(Numb_BS)
  Dual_vars  = zeros(Max_Iters+1, Numb_BS)
  Dual_vars[1,:] =  0.05 * ones(Numb_BS)

  for k = 1:Max_Iters
      println("- Iteration ",k, " -")

      for i = 1:Numb_Operators
          Costs[i], Primals[i,:] = primal_update(Operators[i],dis[i], Dual_vars[k,:])
      end
      if DEBUG > 1
        println("Costs: ",Costs)
        println("Total Costs: ", sum(Costs))
      end

      for j = 1:Numb_BS
          psi[:,j] = (1 - m) * Q * Primals[:,j] +  m * Q

          # sum_Psi[j] = (1 - m) * Q * sum(Primals[:,j]) + Numb_Operators * m * Q
          sum_Psi[j] = sum(psi[:,j])
          Dual_gradient[j] = Numb_Participants * B - sum_Psi[j]

          if Dual_gradient[j] < -1e-4
              println("Fail: Overload Power BS ",j ," :", Dual_gradient[j])
          end
          Dual_vars[k+1,j] = max(Dual_vars[k,j] - alpha * Dual_gradient[j], 0)
  #            Dual_vars[k+1,j] = max(Dual_vars[k,j] - alpha/(k+1) * Dual_gradient[j], 0)
      end

      println("Dual Vars:", Dual_vars[k+1,:])

      if (norm(Dual_vars[k+1,:] - Dual_vars[k,:])< eps1)
        return Costs, psi, OK_CODE ;
        exit();
      end
  end
  return Costs, psi, NOT_CONVERGENCE ;
end

function Nash_Bargaining_Centralized(ops, dis)
  println("----- NASH-Bargaining ----")
  Numb_Participants = size(ops)[1]
  prob = Model(solver=IpoptSolver(tol=1e-9, max_iter=50000, print_level =1))

  @variable(prob, 0<= p[1:Numb_Participants,1:dim_x*dim_y, 1:Numb_BS] <= 1)
  @variable(prob, 0<= rho[1:Numb_Participants,1:Numb_BS] <= 0.99)

  @NLobjective(prob, Max, sum( log(dis[i] - sum(1/(1-rho[i,j])for j=1:Numb_BS)) for i=1:Numb_Participants ) )

  for i = 1:Numb_Participants
    for x=1:dim_x*dim_y
      @constraint(prob, sum(p[i,x,j] for j =1:Numb_BS) == 1 )
    end
    for j =1:Numb_BS
        @constraint(prob, rho[i,j] == dot(ops[i].system_density[1,:,j] , p[i,:,j] ) )
    end
  end

  for j =1:Numb_BS
    @constraint(prob, sum((1 - m) * rho[i,j] * Q +  m * Q for i =1:Numb_Participants) <= (Numb_Participants * B ) )
  end

  status = solve(prob)
  println("Greedy Bargaining Solve: ",status)

  Costs    = zeros(Numb_Participants)
  Primals   = zeros(Numb_Participants, Numb_BS)
  psi       = zeros(Numb_Participants, Numb_BS)

  for i = 1:Numb_Participants
    Costs[i],Primals[i,:] = MPC_check_solutions(ops[i],getvalue(p)[i,:,:],2,1)
  end

  if status == :Optimal
    for j = 1:Numb_BS
        psi[:,j] = (1 - m) * Q * Primals[:,j] + m * Q
        delta_power = Numb_Operators * B - sum(psi[:,j])
        if delta_power < -2e-4
            println("Fail: Overload Power BS ",j ," :", delta_power)
        end
    end
    return Costs, psi, OK_CODE
  else
    return Costs, psi, ERROR_CODE
  end
end
