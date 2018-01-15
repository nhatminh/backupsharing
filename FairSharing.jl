using ECOS
using Gurobi
using GLPK
using SCS
using Ipopt
using Distributions
using HDF5
using JuMP
using Convex
# using Mosek

include("Setting.jl")
include("Common.jl")
include("MPC_Utils.jl")
include("Plot_figs.jl")
# global E = (B-5.)*T_req*dt # Total Energy
# global E = (B-10.)*T_req*dt # Total Energy
global E = B*T_req*dt # Total Energy

function MPC_disagreement_point(op::MPC_MOperator, T, Et_cut, Et_inf, Bi)
  # println("Bi: ",Bi)
  prob = Model(solver=IpoptSolver(tol=1e-8, max_iter=15000, print_level =0))

  @variable(prob,0 <= rho[1:T,1:Numb_BS] <= 0.99)
  # @variable(prob, Psi[1:T,1:Numb_BS] >= 0)
  @variable(prob, Psi_inf[1:T,1:Numb_BS] >= 0)
  @variable(prob, 0<= p[1:T,1:dim_x*dim_y,1:Numb_BS] <= 1)

  @NLobjective(prob, Min, sum(sum( 1/(1-rho[t,j]) for j=1:Numb_BS) for t=1:T) )

  for t = 1:T
    for x = 1:(dim_x*dim_y)
      #only check load of BS 1 is enough for load = 0
      if (op.system_density[t,x,1] > 0)
        @constraint(prob, sum( p[t,x,j] for j=1:Numb_BS) == 1 )
      else
        @constraint(prob, p[t,x,:] .== 0 )
      end
    end

    for j = 1:Numb_BS
        # @constraint(prob, Psi[t,j] == (1 - m) * rho[t,j] * Q +  Psi_inf[t,j])
        @constraint(prob, rho[t,j] == dot(op.system_density[t,:,j], p[t,:,j]) )
        # @constraint(prob, Psi[t,j] <= B )
        @constraint(prob, ((1 - m) * rho[t,j] * Q +  Psi_inf[t,j]) <= Bi[t,j] )

        if DEFERRABLE_MODE == false
          @constraint(prob, Psi_inf[t,j] == 173. )
        end
    end
  end

  for j =1:Numb_BS
    if DEFERRABLE_MODE
      @constraint(prob, sum(Psi_inf[t,j] for t=1:T) == (Et_inf[j]/dt) )
    end

    # @constraint(prob, sum{Psi[t,j],t=1:T} <= (Et_cut[j]/dt) )
    @constraint(prob, sum( (1 - m) * rho[t,j] * Q for t=1:T) <= (Et_cut[j] - Et_inf[j])/dt )
  end

  status = solve(prob)
  println("Dis: ",status)

  dis = zeros(T)
  energy = zeros(T,Numb_BS)
  inf_energy = zeros(T,Numb_BS)

  if (status == :Optimal) | (status == :UserLimit)
    inf_energy = getvalue(Psi_inf)
    for t = 1:T
      dis[t],energy[t,:]= MPC_check_solutions(op,getvalue(p)[t,:,:],inf_energy[t,:], 1,t)
    end

    if DEFERRABLE_MODE
      # println(energy[:,:])
      # println(inf_energy[:,:])
    end
    return dis, energy[1,:],inf_energy[1,:], OK_CODE
  else
    return dis, energy[1,:],inf_energy[1,:], ERROR_CODE
  end
end

function MPC_primal_update_ADMM(op::MPC_MOperator, dis, T, E_tilde, Et_inf, dual_vars, e_arr,Zeta_arr)
    Costs = zeros(T)
    primals = zeros(T,Numb_BS)
    inf_energy = 173*ones(T,Numb_BS)   #Non-Deferrable
    inf_status = OK_CODE
    prob = Model(solver=IpoptSolver(tol=5e-6, max_iter=1000, print_level =1))

    @variable(prob, 0<= rho[1:T,1:Numb_BS]<= 0.99)
    # @variable(prob, Zeta[1:Numb_BS] >= 0)
    # @variable(prob, 0<= p[1:T,1:dim_x*dim_y,1:Numb_BS] <= 1)
    @variable(prob, p[1:T,1:dim_x*dim_y,1:Numb_BS] >= 0)

    sum_Zeta = zeros(Numb_BS)
    for j=1:Numb_BS
        for i=1:Numb_Operators
          if(op.idx!=i)
            sum_Zeta[j] += Zeta_arr[i,j]
          end
        end
    end

    @NLobjective(prob, Min, -log(dis - sum(sum( 1/(1-rho[t,j]) for j=1:Numb_BS) for t=1:T)) +
                # sum(rho1/2*(Zeta[j]+sum_Zeta[j] + e_arr[j] - E_tilde[j] - dual_vars[j]/rho1)^2 + rho2/2*(Zeta[j] - Zeta_arr[op.idx,j])^2 for j=1:Numb_BS) )
                sum(rho1/2*( (1 - m) *Q*sum( rho[t,j] for t=1:T)+ 173*T +sum_Zeta[j] + e_arr[j] - E_tilde[j] - dual_vars[j]/rho1)^2 +
                rho2/2*( (1 - m)*Q*sum( rho[t,j] for t=1:T)+ 173*T + - Zeta_arr[op.idx,j])^2 for j=1:Numb_BS) )

    for t = 1:T
      for x = 1:(dim_x*dim_y)
        #only check load of BS 1 is enough for load = 0
        if (op.system_density[t,x,1] > 0)
          @constraint(prob, sum( p[t,x,j] for j=1:Numb_BS) == 1 )
        else
          @constraint(prob, p[t,x,:] .== 0 )
        end
      end

      for j = 1:Numb_BS
          @constraint(prob, rho[t,j] == dot(op.system_density[t,:,j], p[t,:,j]) )
      end
    end

    # for j = 1:Numb_BS
    #     @constraint(prob, Zeta[j] == sum((1 - m) * rho[t,j] * Q + 173 for t=1:T) )
    # end
    @NLconstraint(prob, sum(sum( 1/(1-rho[t,j]) for j=1:Numb_BS) for t=1:T) <= 0.999*dis )
    # tic()
    status = solve(prob)
    # println("TOC1:",toc())
    println("ADMM Primal Solve ",op.idx," : ",status)

    Costs = zeros(T)
    primals = zeros(T,Numb_BS)
    return_zeta = zeros(Numb_BS)
    # if (status == :Optimal) | (status == :UserLimit)
    if (status == :Optimal)
      for t = 1:T
        Costs[t],primals[t,:]= MPC_check_solutions(op,getvalue(p)[t,:,:],inf_energy[t,:],2,t)
      end
      for j =1:Numb_BS
        return_zeta[j] = (1 - m) *Q*sum(primals[:,j]) + 173*T
      end
      return Costs, primals, inf_energy, return_zeta, OK_CODE
    else
      return Costs, primals,inf_energy, Zeta_arr[op.idx,:], ERROR_CODE
    end
end

function e_update_ADMM(dual_vars,e_arr, Zeta_arr,E_tilde)
  prob = Model(solver=IpoptSolver(tol=1e-7, max_iter=1000, print_level =2))

  @variable(prob, e[1:Numb_BS] >= 0)

  @NLobjective(prob, Min, sum(rho1/2*(sum(Zeta_arr[i,j] for i=1:Numb_Operators) + e[j] - E_tilde[j] - dual_vars[j]/rho1)^2 +
              rho2/2*(e[j] - e_arr[j])^2 for j=1:Numb_BS) )

  status = solve(prob)
  # println("Primal Solve: ",status)

  return getvalue(e)
end

function e_update_ADMM1(dual_vars,e_arr, Zeta_arr,E_tilde)
  e = zeros(Numb_BS)
  for j = 1:Numb_BS
    e[j] = max(0,(rho1*( E_tilde[j] - sum(Zeta_arr[i,j]  for i=1:Numb_Operators) ) + dual_vars[j] + rho2*e_arr[j] )/(rho1+rho2))
  end

  return e
end

# function Nash_Bargaining
function MPC_Nash_Bargaining_Distributed(Participants, dis, T, Et_cut, Et_inf, Bt, Max_Iters=100)
  println("----- MPC NASH-Bargaining Distributed ----")
  Numb_Participants = size(Participants)[1]
  println("Number of Participants: ", Numb_Participants)
  for i =1:Numb_Participants
      print(Participants[i].idx)
  end
  println()
  # alpha1 = 3.4e-5*(1 + (T_req - T)/1.4) #Step_Size     #6 slots
  # alpha3 = 1.e-4 #Step_Size

  eps1= 5e-7

  global rho1 = 1e-6 *(1 + (T_req - T)/1.)
  Jacobian_step = 1.
  delta = 1e-7
  global rho2 = (rho1*(Numb_Operators/(2-Jacobian_step)-1) + delta)

  Primals   = zeros(Numb_Participants, T, Numb_BS)
  Costs     = zeros(Numb_Participants, T)
  sum_Psi   = zeros(T,Numb_BS)
  sum_Psi_usg   = zeros(T,Numb_BS)
  sum_Psi_inf   = zeros(T,Numb_BS)
  sum_Bt        = zeros(T,Numb_BS)
  Bt_rs         = zeros(Numb_Participants,T,Numb_BS)
  psi         = zeros(Numb_Participants,Numb_BS)  # store power for return
  psi_inf_rs  = zeros(Numb_Participants,Numb_BS) # store power for return
  psi_inf     = zeros(Numb_Participants,T,Numb_BS)
  # et    = zeros(Numb_Participants,Numb_BS)
  # et_inf    = zeros(Numb_Participants,Numb_BS)
  Dual_gradient = zeros(Numb_BS)
  Dual_vars  = zeros(Max_Iters+1, Numb_BS)
  e_arr= zeros(Numb_BS, Max_Iters+1)
  Zeta_arr= B*T*ones(Numb_Operators, Numb_BS, Max_Iters+1)

  for k = 1:Max_Iters
      if DEBUG > 0 println("- Iteration ",k, " -"); end
      E_tilde = sum(Et_cut,1)/dt

      for i = 1:Numb_Participants
          for t = 1:T
            sum_Bt[t,:] += Bt[i,t,:]
          end

           Costs[i,:], Primals[i,:,:], psi_inf[i,:,:], Zeta_arr[i,:,k+1],err = MPC_primal_update_ADMM(Participants[i], dis[i], T,
                                                              E_tilde, Et_inf[i,:], Dual_vars[k,:], e_arr[:,k],Zeta_arr[:,:,k])

          Bt_rs[i,:,:] = (1 - m) * Q * Primals[i,:,:] + psi_inf[i,:,:]

          if err == ERROR_CODE
            return Costs, psi, psi_inf_rs,Bt, err
            exit()
          end
      end

      e_arr[:,k+1] = e_update_ADMM1(Dual_vars[k,:], e_arr[:,k],Zeta_arr[:,:,k+1],E_tilde)

      # println("DIFF:",norm(e_arr[:,k+1]-old))

      for j = 1:Numb_BS
          psi_inf_rs[:,j] = psi_inf[:,1,j]
          psi[:,j] = (1 - m) * Q * Primals[:,1,j] + psi_inf_rs[:,j]

          # println("Zeta1:",Zeta_arr[:,j,k+1])
          # println("T2:",Et_cut[:,j] )
          # println("T3:",Et_inf[:,j])
          # println("T4:",e_arr[j,k+1] )

          Dual_gradient[j] = sum(Zeta_arr[:,j,k+1]) + e_arr[j,k+1] - E_tilde[j]
          Dual_vars[k+1,j] = Dual_vars[k,j] - Jacobian_step * rho1 * Dual_gradient[j]

          # Dual_gradient2[j] = sum(Et_inf[:,j]) - sum(sum_Psi_inf)*dt
          # Dual_vars2[k+1,j] = Dual_vars2[k,j] - alpha2 * Dual_gradient2[j]

          if Dual_gradient[j] > -1e-3
            println("Fail: Overload Energy BS ",j ," :", Dual_gradient[j])
          end
          # if Dual_gradient2[j] < -1e-3
          #   println("Fail: Overload Energy INF BS ",j ," :", Dual_gradient1[j])
          # end
      end

      # println("Difference:",Dual_gradient)

      if (DEBUG > 0)
        println("Dual Vars:", Dual_vars[k+1,:,:])
      end

      if (norm(Dual_vars[k+1,:] - Dual_vars[k,:])< eps1)
          return Costs, psi, psi_inf_rs,Bt_rs, OK_CODE ;  exit();
      end
  end
  # println("COSTS:",sum(Costs,2))
  # figure(21)
  # for j=1:Numb_BS
  #   plot(Dual_vars[:,j])
  # end
  #
  # figure(22)
  # for j=1:Numb_BS
  #   plot(e_arr[j,:])
  # end

  println("NOT_CONVERGENCE")
  return Costs, psi, psi_inf_rs,Bt, NOT_CONVERGENCE
end

function main()
  T_pred = T_req
  # Numb_shifts = T_req - T_pred
  Numb_shifts = T_req-1
  # Numb_shifts =0
  println("START")
  Operators = Array(MPC_MOperator,Numb_Operators)
  Opt_disagreement = zeros(Numb_Operators,T_req)
  Optimal_plan = zeros(Numb_Operators,T_req)
  Greedy_energy  = zeros(Numb_Operators,T_req+1,Numb_BS)
  Greedy_control_rs = zeros(Numb_Operators,T_req)
  Greedy_bargaining_energy  = zeros(Numb_Operators,T_req+1,Numb_BS)
  Greedy_bargaining_control_rs = zeros(Numb_Operators,T_req)
  MPC_unshared_rs          = zeros(Numb_Operators,T_req)
  MPC_unshared_energy          = zeros(Numb_Operators,T_req+1,Numb_BS)
  MPC_unshared_energy_inf      = zeros(Numb_Operators,T_req+1,Numb_BS)
  MPC_bargaining_rs            = zeros(Numb_Operators,T_req)
  MPC_bargaining_energy        = zeros(Numb_Operators,T_req+1,Numb_BS)
  MPC_bargaining_energy_inf    = zeros(Numb_Operators,T_req+1,Numb_BS)

  Total_Cost_opt_dis   = zeros(Numb_Operators)
  Total_Cost_opt_plan  = zeros(Numb_Operators)
  Total_Cost_greedy    = zeros(Numb_Operators)
  Total_Cost_g_bargaining  = zeros(Numb_Operators)
  Total_Cost_MPC_dis       = zeros(Numb_Operators)
  Total_Cost_MPC_dis1      = zeros(Numb_Operators)
  Total_Cost_d_bargaining  = zeros(Numb_Operators)

  Power_plan = B*ones(Numb_Operators, T_req, Numb_BS)

  curr_time = 1
  global Capacity = compute_capacity()

  for i = 1:Numb_Operators
    operator = MPC_MOperator(lambs[i],i,[],[])
    Operators[i] = operator
    MPC_traffic_density(operator)
    Greedy_energy[i,1,:]              = E*ones(Numb_BS)
    Greedy_bargaining_energy[i,1,:]   = E*ones(Numb_BS)
    MPC_unshared_energy[i,1,:]        = E*T_pred/T_req*ones(Numb_BS)
    MPC_bargaining_energy[i,1,:]      = E*T_pred/T_req*ones(Numb_BS)
    MPC_unshared_energy_inf[i,1,:]    = m*Q*dt*T_req*ones(Numb_BS)
    MPC_bargaining_energy_inf[i,1,:]  = m*Q*dt*T_req*ones(Numb_BS)

    actual_mode = true
    MPC_system_load(operator, curr_time, T_pred, actual_mode)
  end

  ##### 1. MPC OPTIMAL CONTROL PART
  if RUNNING_MODE > 1
    println("@@@ PART 1: MPC OPTIMAL CONTROL")
    for i = 1:Numb_Operators
      Opt_disagreement[i,:], _ , status = MPC_disagreement_point(Operators[i], T_req, E*ones(Numb_BS),
                                                                m*Q*dt*T_req*ones(Numb_BS),Power_plan[i,:,:])

      Total_Cost_opt_dis[i] = sum(Opt_disagreement[i,:])
      # println(Opt_disagreement[i,:])
      # println(Total_Cost_opt_dis[i])
    end

    Optimal_plan, _ , _, _, status = MPC_Nash_Bargaining_Distributed(Operators, Total_Cost_opt_dis, T_req, E*ones(Numb_Operators,Numb_BS),
                                                                m*Q*dt*T_req*ones(Numb_Operators,Numb_BS),Power_plan,100)
    if status != OK_CODE
      Optimal_plan = Opt_disagreement
    end
  end


  psi_dis         = zeros(Numb_Operators,Numb_BS)
  psi_dis_inf     = zeros(Numb_Operators,Numb_BS)
  psi_dis_greedy  = zeros(Numb_Operators,Numb_BS)

  G_Participants = MPC_MOperator[]
  G_Disagremments = Float64[]
  Numb_G_Participants = 0
  MPC_Participants = MPC_MOperator[]
  Numb_MPC_Participants = 0
  dis_solution = zeros(T_req)

  #### SHIFT WINDOWS T_pred
  for k = 0:(Numb_shifts)
    println("===== Current Time: ",curr_time, " =====")
    # println(Power_plan[:,:,:])
    # psi_dis         = zeros(Numb_Operators,Numb_BS)
    # psi_dis_inf     = zeros(Numb_Operators,Numb_BS)
    # psi_dis_greedy  = zeros(Numb_Operators,Numb_BS)
    psi_g_bargaining  = zeros(Numb_Operators,Numb_BS)
    psi_d_bargaining  = zeros(Numb_Operators,Numb_BS)
    psi_d_bargaining_inf  = zeros(Numb_Operators,Numb_BS)

    # G_Participants = MPC_MOperator[]
    # G_Disagremments = Float64[]
    # Numb_G_Participants = 0
    # MPC_Participants = MPC_MOperator[]
    MPC_Disagremments = Float64[]
    #
    # Numb_MPC_Participants = 0

    for i = 1:Numb_Operators
      MPC_system_load(Operators[i], curr_time, T_pred)

      # println(Operators[i].Et_cut)
      # println(operator.system_density)

      ##### 2. GREEDY CONTROL PART
      if RUNNING_MODE > 2
        println("@@@ PART 2: GREEDY CONTROL")
        if minimum(Greedy_energy[i,curr_time,:]) > 0
          dis_solution_greedy, psi_dis_greedy[i,:], status  = disagreement_point(Operators[i])
          remain_Energy = Greedy_energy[i,curr_time,:] - psi_dis_greedy[i,:]*dt
          # println(remain_Energy)

          if (minimum(remain_Energy) < 0) | (status != OK_CODE)
            println("Greedy: Out of resource for Operator ", i)
            Greedy_control_rs[i,curr_time] = 0
            Greedy_energy[i,curr_time+1,:] = Greedy_energy[i,curr_time,:]
          else
            Greedy_control_rs[i,curr_time] = dis_solution_greedy
            Greedy_energy[i,curr_time+1,:] = remain_Energy
            push!(G_Participants, Operators[i])
            push!(G_Disagremments, sum(dis_solution_greedy))
            Numb_G_Participants += 1
          end
        else
          println("Greedy: Out of resource for Operator ", i)
          Greedy_control_rs[i,curr_time] = 0
        end
      end

      ##### 4. MPC DISAGREEMENT PART
      println("@@@ PART 4: MPC DISAGREEMENT CONTROL")
      # println("Remaining Energy: ", MPC_bargaining_energy[i,curr_time,:])
      # println("Remaining Energy INF: ", MPC_bargaining_energy_inf[i,curr_time,:])

      # if sum(MPC_bargaining_energy[i,curr_time,:]) > 0  #[This logic may not true, can use other BS energy]
      if k == 0
        dis_solution, psi_dis[i,:], psi_dis_inf[i,:] , status = MPC_disagreement_point(Operators[i],T_pred,
                                              MPC_bargaining_energy[i,curr_time,:],MPC_bargaining_energy_inf[i,curr_time,:],Power_plan[i,curr_time:end,:])

        if status == ERROR_CODE
          MPC_unshared_rs[i,curr_time] = 0
          # dis_solution_greedy, psi_dis_greedy[i,:], status1  = disagreement_point(Operators[i])  # CANNOT PLAN and JOIN BAIRGAINING GAME
          # if status1 == ERROR_CODE
          #     println("Dis: Out of resource for Operator ", i)                             # Out of resource
          # else
          #     MPC_unshared_rs[i,curr_time] = dis_solution_greedy
          #     MPC_bargaining_energy[i,curr_time+1,:]     = MPC_bargaining_energy[i,curr_time,:]  - psi_dis_greedy[i,:]
          #     MPC_bargaining_energy_inf[i,curr_time+1,:] = MPC_bargaining_energy_inf[i,curr_time,:]  - m*Q*dt
          # end
        else
          push!(MPC_Participants, Operators[i])
          push!(MPC_Disagremments, sum(dis_solution[:]))
          Numb_MPC_Participants += 1
          MPC_unshared_rs[i,curr_time] = dis_solution[1]
        end
      else
        push!(MPC_Disagremments, sum(dis_solution[curr_time:end]))
      end

      dis_solution1, psi_dis1, psi_dis_inf1, status = MPC_disagreement_point(Operators[i],T_pred,
                                            MPC_unshared_energy[i,curr_time,:], MPC_unshared_energy_inf[i,curr_time,:],Power_plan[i,curr_time:end,:])

      remain_Energy = MPC_unshared_energy[i,curr_time,:] - psi_dis1[:]*dt
      remain_Energy_inf = MPC_unshared_energy_inf[i,curr_time,:] - psi_dis_inf1[:]*dt

      if (sum(remain_Energy) < 0) | (status != OK_CODE)
          MPC_unshared_rs[i,curr_time] = 0
          MPC_unshared_energy[i,curr_time+1,:] =   MPC_unshared_energy[i,curr_time,:]
          MPC_unshared_energy_inf[i,curr_time+1,:] = MPC_unshared_energy_inf[i,curr_time,:]
      else[i,curr_time,:]
          MPC_unshared_energy[i,curr_time+1,:] = remain_Energy
          MPC_unshared_energy_inf[i,curr_time+1,:] = remain_Energy_inf
          MPC_unshared_rs[i,curr_time] = dis_solution1[1]
      end
      # else
      #   MPC_unshared_rs[i,curr_time] = 0
      # end
    end

    if RUNNING_MODE > 2
    #### 3. GREEDY NASH-Bargaining CONTROL PART
      println("PART 3:  GREEDY NASH-Bargaining CONTROL")
      # G_bargaining, psi_g_bargaining, status  = Nash_Bargaining_Distributed(G_Participants, G_Disagremments)
      G_bargaining, psi_g_bargaining, status = Nash_Bargaining_Centralized(G_Participants, G_Disagremments)

      if status == OK_CODE
        for i = 1:Numb_G_Participants
          remain_Energy = Greedy_bargaining_energy[i,curr_time,:] -  psi_g_bargaining[i,:]*dt

          if minimum(remain_Energy) < 0
            println("Greedy-Bargaining: Out of resource for Operator ", i)
            Greedy_bargaining_control_rs[i,curr_time] = 0
            Greedy_bargaining_energy[i,curr_time+1,:] = Greedy_bargaining_energy[i,curr_time,:]
          else
            Greedy_bargaining_control_rs[i,curr_time] = G_bargaining[i]
            Greedy_bargaining_energy[i,curr_time+1,:] = remain_Energy
          end
        end
      else
        println("Error Here")
        Greedy_bargaining_control_rs[:,curr_time] = Greedy_control_rs[:,curr_time]
        for i = 1:Numb_Operators
          Greedy_bargaining_energy[i,curr_time+1,:] = Greedy_bargaining_energy[i,curr_time,:] - psi_dis_greedy[i,:]*dt
        end
      end
    end

    #### 5. MPC BARGAINING PART
    println("@@@ Part 5: MPC BARGAINING")
    MPC_bargaining_rs[:,curr_time] = MPC_unshared_rs[:,curr_time]
    status = ERROR_CODE

    if (Numb_MPC_Participants > 1)
      MPC_Et = zeros(Numb_MPC_Participants,Numb_BS)
      MPC_Et_inf = zeros(Numb_MPC_Participants,Numb_BS)
      MPC_Power_plan = zeros(Numb_MPC_Participants,T_pred,Numb_BS)

      for i = 1:Numb_MPC_Participants
        MPC_Et[i,:]     = MPC_bargaining_energy[MPC_Participants[i].idx,curr_time,:]
        MPC_Et_inf[i,:] = MPC_bargaining_energy_inf[MPC_Participants[i].idx,curr_time,:]
        MPC_Power_plan[i,:,:] = Power_plan[MPC_Participants[i].idx,curr_time:end,:]
      end

      MPC_bargaining, psi_d_bargaining, psi_d_bargaining_inf, Bt, status = MPC_Nash_Bargaining_Distributed(MPC_Participants, MPC_Disagremments, T_pred,
                                                              MPC_Et, MPC_Et_inf, MPC_Power_plan,200)

      if status == OK_CODE
        for i = 1:Numb_MPC_Participants
          idx = MPC_Participants[i].idx
          # Power_plan[idx,curr_time:end,:] = Bt[idx,:,:]  #### DYNAMIC POWER PLAN
          MPC_bargaining_rs[idx,curr_time]  = MPC_bargaining[i,1]
          MPC_bargaining_energy[idx,curr_time+1,:]     = MPC_bargaining_energy[idx,curr_time,:]  - psi_d_bargaining[i,:]*dt
          MPC_bargaining_energy_inf[idx,curr_time+1,:] = MPC_bargaining_energy_inf[idx,curr_time,:]  - psi_d_bargaining_inf[i,:]*dt
        end
      end
    end

    if status == ERROR_CODE
      println("MPC BARGAINING Error")
      for i = 1:Numb_MPC_Participants
        idx = MPC_Participants[i].idx
        MPC_bargaining_energy[idx,curr_time+1,:]     = MPC_bargaining_energy[idx,curr_time,:]  - psi_dis[idx,:]*dt
        MPC_bargaining_energy_inf[idx,curr_time+1,:] = MPC_bargaining_energy_inf[idx,curr_time,:]  - psi_dis_inf[idx,:]*dt
      end
    end

    curr_time += 1
    T_pred -= 1
  end

  println("******  RESULTS  ******")
  for i = 1:Numb_Operators
    if RUNNING_MODE > 1
      println("*** 1.Optimal ",i,": ", Optimal_plan[i,:])
    end
    if RUNNING_MODE > 2
      println("*** 2.Greedy ",i,": ", Greedy_control_rs[i,:])
      println("*** 3.G-Bargaining ",i,": ", Greedy_bargaining_control_rs[i,:])
    end
    println("*** 4.No Cooperation ",i,": ", MPC_unshared_rs[i,:])
    println("*** 5.D-Bargaining ",i,": ", MPC_bargaining_rs[i,:])

    Total_Cost_opt_plan[i] = sum(Optimal_plan[i,:])
    Total_Cost_greedy[i] = sum(Greedy_control_rs[i,:])
    Total_Cost_g_bargaining[i] = sum(Greedy_bargaining_control_rs[i,:])
    Total_Cost_MPC_dis[i] = sum(MPC_unshared_rs[i,:])
    Total_Cost_d_bargaining[i] = sum(MPC_bargaining_rs[i,:])
  end

  if RUNNING_MODE > 1
    println("+++ Total Cost MPC Optimal: ",Total_Cost_opt_plan)
  end

  if RUNNING_MODE > 2
    println("+++ Total Cost Greedy: ",Total_Cost_greedy)
    println("+++ Total Cost G-Bargaining: ",Total_Cost_g_bargaining)
  end
  println("+++ Total Cost Disaggreement: ",Total_Cost_MPC_dis)
  println("+++ Total Cost D-Bargaining: ",Total_Cost_d_bargaining)

  filename= "MPC_results.h5"
  h5open(filename, "w") do file
    write(file, "Total_Cost_MPC_dis",Total_Cost_MPC_dis)
    write(file, "MPC_unshared_rs",MPC_unshared_rs)
    write(file, "Total_Cost_d_bargaining",Total_Cost_d_bargaining )
    write(file, "MPC_bargaining_rs",MPC_bargaining_rs)
    write(file, "MPC_bargaining_energy",MPC_bargaining_energy)
    write(file, "MPC_unshared_energy",MPC_unshared_energy)
  end
  plt_MPC_cost_comparison(Total_Cost_MPC_dis, MPC_unshared_rs, Total_Cost_d_bargaining,MPC_bargaining_rs)
  plt_MPC_remaning_enery(MPC_bargaining_energy, MPC_unshared_energy)
end

function MPC_read_results()
  filename= "MPC_results.h5"
  h5open(filename, "r") do file
    Total_Cost_MPC_dis      = read(file, "Total_Cost_MPC_dis")
    MPC_unshared_rs         = read(file, "MPC_unshared_rs")
    Total_Cost_d_bargaining = read(file, "Total_Cost_d_bargaining")
    MPC_bargaining_rs       = read(file, "MPC_bargaining_rs")
    MPC_bargaining_energy   = read(file, "MPC_bargaining_energy")
    MPC_unshared_energy     = read(file, "MPC_unshared_energy")
    plt_MPC_cost_comparison(Total_Cost_MPC_dis, MPC_unshared_rs, Total_Cost_d_bargaining,MPC_bargaining_rs)
    plt_MPC_remaning_enery(MPC_bargaining_energy, MPC_unshared_energy)
  end
end

if(MPC_REUSED_READ_ONLY)
  MPC_read_results()
else
  main()
end
