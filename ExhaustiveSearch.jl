# Let us first make the Convex and MultiConvex modules available
using Convex
using ECOS
using Gurobi
using SCS
using Ipopt
using PyPlot
using HDF5

include("Setting.jl")
include("Common.jl")
include("Plot_figs.jl")

global READ_DATA = false

function cal_cost(u1,u2,u3)
  costs = zeros(Numb_Operators)

  costs[1]= sum(1/(1-u1))
  costs[2]= sum(1/(1-u2))
  costs[3]= sum(1/(1-u3))
  return costs
end

using Ipopt
using PyPlot
using JuMP

interval=0.1
numb_slices = convert(Int,1/interval)
a_g_cost = zeros(numb_slices,numb_slices)
fig_size = (7.1,4.8)
label_fontsize = 16

function Nash_Bargaining_Centralized(ops, c1, c2)
  println("----- NASH-Bargaining ----")
  println("DIS:",dis_cost)
  println("DIS1:",c1)
  println("DIS2:",c2)
  Numb_Participants = size(ops)[1]
  prob = Model(solver=IpoptSolver(tol=1e-9, max_iter=10000, print_level =2))

  @variable(prob, 0<= p[1:Numb_Participants,1:dim_x*dim_y, 1:Numb_BS] <= 1)
  @variable(prob, 0<= rho[1:Numb_Participants,1:Numb_BS] <= 0.99)

  @NLobjective(prob, Max, log(dis_cost[3] - sum(1/(1-rho[3,j]) for j=1:Numb_BS)) )

  for i = 1:Numb_Participants
    for x=1:dim_x*dim_y
      @constraint(prob, sum(p[i,x,j] for j =1:Numb_BS) == 1 )
    end
    for j =1:Numb_BS
        @constraint(prob, rho[i,j] == dot(ops[i].system_density[:,j] , p[i,:,j] ) )
    end
  end

  for j =1:Numb_BS
    @constraint(prob, sum((1 - m) * rho[i,j] * Q +  m * Q for i =1:Numb_Participants) <= (Numb_Participants * B ))
  end

  @NLconstraint(prob, sum(1/(1-rho[1,j]) for j=1:Numb_BS) == c1 )
  @NLconstraint(prob, sum(1/(1-rho[2,j]) for j=1:Numb_BS) == c2 )
  @NLconstraint(prob, sum(1/(1-rho[3,j]) for j=1:Numb_BS) <= 0.999*dis_cost[3] )

  status = solve(prob)
  println("Nash Centralize Solve: ",status)
  Costs    = 8*3
  if(status==:Optimal)
    # Costs = (dis_cost[1] - c1) * (dis_cost[2] - c2) * (dis_cost[3] -sum(1/(1-getvalue(rho)[3])))
    return c1 + c2+ sum(1/(1-getvalue(rho)[3,j]) for j=1:Numb_BS)
    # return sum(1/(1-getvalue(rho)[3,j]) for j=1:Numb_BS)
  else
    return Costs
  end
end

function Nash_Bargaining_Centralized1(ops, c1, c2, c3)
  # println("----- NASH-Bargaining ----")
  # println("DIS:",dis_cost)
  # println("DIS1:",c1)
  # println("DIS2:",c2)
  Numb_Participants = size(ops)[1]
  prob = Model(solver=IpoptSolver(tol=1e-9, max_iter=10000, print_level =2))

  @variable(prob, 0<= p[1:Numb_Participants,1:dim_x*dim_y, 1:Numb_BS] <= 1)
  @variable(prob, 0<= rho[1:Numb_Participants,1:Numb_BS] <= 0.99)

  @objective(prob, Max, 1)

  for i = 1:Numb_Participants
    for x=1:dim_x*dim_y
      @constraint(prob, sum(p[i,x,j] for j =1:Numb_BS) == 1 )
    end
    for j =1:Numb_BS
        @constraint(prob, rho[i,j] == dot(ops[i].system_density[:,j] , p[i,:,j] ) )
    end
  end

  for j =1:Numb_BS
    @constraint(prob, sum((1 - m) * rho[i,j] * Q +  m * Q for i =1:Numb_Participants) <= (Numb_Participants * B ))
  end

  @NLconstraint(prob, sum(1/(1-rho[1,j]) for j=1:Numb_BS) == c1 )
  @NLconstraint(prob, sum(1/(1-rho[2,j]) for j=1:Numb_BS) == c2 )
  @NLconstraint(prob, sum(1/(1-rho[3,j]) for j=1:Numb_BS) == c3 )

  for i = 1:Numb_Participants
    @NLconstraint(prob, sum(1/(1-rho[i,j]) for j=1:Numb_BS) <= 0.9995*dis_cost[i] )
  end


  status = solve(prob)
  println("Nash Centralize Solve: ",status)
  # Costs    = -12
  Costs    = 0
  if(status==:Optimal)
    return (max1-c1) *(max2-c2) *(dis_cost[3]-c3)
    # return log(max1-c1) + log(max2-c2) + log(dis_cost[3]-c3)
  else
    return Costs
  end
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

function save_cost(NBS_cost)
  h5open("cost_map.h5", "w") do file
      write(file, "NBS_cost", NBS_cost)
  end
end

function read_cost()
  h5open("cost_map.h5", "r") do file
      global NBS_cost = read(file, "NBS_cost")
  end
end

function read_utilization_decentralized()
  h5open("decentralized_results.h5", "r") do file
      global dis_cost = read(file, "dis_cost")
      global dis_rhos = read(file, "dis_rhos")
      global rho_decentralized = read(file, "rho_decentralized")
      global Costs2 = read(file, "Costs2")
  end

  h5open("cost_convergence.h5", "r") do file
    global Costs_dual = read(file, "Costs")  # alternatively, say "@write file A"
  end

  # alg_interval = 5
  # global converge_k = 1
  # for i=1:size(alphas[1,:])[1]
  #   if((abs(alphas[1,i] - alphas[1,i+1]) + abs(alphas[1,i]- alphas[1,i+1])) <1e-4 )
  #     break
  #   end
  #   converge_k += 1
  # end
  # println(converge_k)
end

interval=0.005
fig_size = (7.1,4.8)
label_fontsize = 16


function plot_utilization_map()
  read_utilization_decentralized()
  min1 = sum(1/(1-rho_decentralized[1,j,end]) for j=1:Numb_BS)
  min11 = sum(1/(1-rho_decentralized[1,j,end]) for j=1:Numb_BS)
  global max1 = dis_cost[1]
  println("Max1:",max1)
  println("Min1:",min1)
  min2 = sum(1/(1-rho_decentralized[2,j,end]) for j=1:Numb_BS)
  min22 = sum(1/(1-rho_decentralized[2,j,end]) for j=1:Numb_BS)
  global max2 = dis_cost[2]
  min3 = sum(1/(1-rho_decentralized[3,j,end]) for j=1:Numb_BS)
  println("Max2:",max2)
  println("Min2:",min2)
  println("num_slice1:",round(Int,(max1-min1)/interval))
  println("num_slice2:",round(Int,(max2-min2)/interval))
  numb_slices1 = round(Int,(max1-min1)/interval)
  numb_slices2 = round(Int,(max2-min2)/interval)
  # if (numb_slices1 < numb_slices2)
  #   numb_slices1 = numb_slices2
  #   max1 = min1+ numb_slices2*interval
  # else
  #   numb_slices2 = numb_slices1
  #   max2 = min2+ numb_slices1*interval
  # end
  min1 = min1 - interval
  min2 = min2 - interval
  numb_slices1 = numb_slices1 + 2
  numb_slices2 = numb_slices2 + 2
  max_dimension = max(numb_slices1,numb_slices2)
  global NBS_cost = -12*ones(max_dimension, max_dimension)
  # global NBS_cost = zeros(max_dimension, max_dimension)

  println("START")

  global Capacity = compute_capacity()
  total_arrival, Arrivals_Pattern = read_arrival()
  global Arrivals = total_arrival

  Operators = Array(MOperator,Numb_Operators)

  for i = 1:Numb_Operators
    operator = MOperator(lambs[i],i,[])
    system_load(operator)
    # println(operator.system_density)
    Operators[i] = operator
  end

  if(READ_DATA)
    read_cost()
  else
    for i = 0:max_dimension-1
      c1= min1+interval*i
      # println("steps:",interval*i)
      # println("C1:",c1)
      for j = 0:max_dimension-1
        c2= min2+interval*j
        # NBS_cost[i,j] = Nash_Bargaining_Centralized(Operators,c1,c2)
        # NBS_cost[i,j] = c1 + c2 + min3
        # NBS_cost[i,j] = log(max1-c1) + log(max2-c2) + log(dis_cost[3]-min3)
        # NBS_cost[i,j] = (max1-c1) *(max2-c2) *(dis_cost[3]-min3)
        # if(c1>min11) &  (c2>min22) & (c1<max1) & (c2<max2)
          NBS_cost[i+1,j+1] =(max1-c1) *(max2-c2) *(dis_cost[3]-min3)
        # else
          # NBS_cost[i+1,j+1]  = Nash_Bargaining_Centralized1(Operators,c1,c2,min3)
        # end
      end
    end

    save_cost(NBS_cost)
  end
  println(min1+interval*numb_slices1,":",min2 + numb_slices2*interval)
  println(dis_cost[1],":",dis_cost[2])
  println(min1+max_dimension*interval,":",min2 + max_dimension*interval)
  # println("Costs:",NBS_cost)

  # x_start = gammas[1,1]
  # y_start = alphas[1,1]
  # x_end = gammas[1,converge_k]
  # y_end = alphas[1,converge_k]

  BS1 = linspace(min1, min1+max_dimension*interval, max_dimension)
  BS2 = linspace(min2, min2+max_dimension*interval, max_dimension)

  xgrid = repmat(BS1',max_dimension,1)
  ygrid = repmat(BS2,1,max_dimension)


  figure("pyplot_surfaceplot1",figsize=fig_size)

  plot_surface(xgrid, ygrid, NBS_cost, rstride=3,edgecolors="k", cstride=3, cmap=ColorMap("coolwarm"), alpha=0.8, linewidth=0.25)
  xlabel("Operator 2",fontsize=label_fontsize)
  ylabel("Operator 1",fontsize=label_fontsize)
  # xlim(min2,dis_cost[2])
  # ylim(min1,dis_cost[1])
  tight_layout()
  savefig("util_map_3D.pdf")

  fig = figure("pyplot_surfaceplot2",fig_size)
  cp = contour(xgrid, ygrid, NBS_cost,cmap="coolwarm")
  clabel(cp, inline=1, fontsize=10)
  plot(Costs_dual[1,1:2:end],Costs_dual[2,1:2:end],linestyle="--",marker="^",markersize=7,color="blueviolet",alpha=0.5,label="Dual Decomposition")
  plot(Costs2[1,1:2:end],Costs2[2,1:2:end],linestyle="-",marker="o",markersize=7, color="darkgreen",alpha=0.6,label="JP-ADMM")

  annotate("Disagreement",
          	xy=[dis_cost[1];dis_cost[2]],
          	xytext=[dis_cost[1]-0.035;dis_cost[2]-0.01],
            xycoords="data",
            size=14,
            arrowprops=Dict("arrowstyle"=>"fancy",
                "facecolor"=>"black",
                "connectionstyle"=>"angle3,angleA=0,angleB=-90"))

  annotate("NBS",
          	xy=[min11;min22],
          	xytext=[min11+0.004;min22+0.014],
            xycoords="data",
            size=14,
            arrowprops=Dict("arrowstyle"=>"fancy",
                "facecolor"=>"black",
                "edgecolor"=>"black",
                "connectionstyle"=>"angle3,angleA=0,angleB=-85"))


  xlim(min1,dis_cost[1])
  ylim(min2,dis_cost[2])
  xlabel("Operator 2 (\$\\phi_2\$)",fontsize=label_fontsize)
  ylabel("Operator 1 (\$\\phi_1\$)",fontsize=label_fontsize)
  legend(loc=2,fontsize=label_fontsize-2)
  tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
  savefig("contour_pareto.pdf")

  figure("pyplot_surfaceplot5",fig_size)
  pcolor(xgrid, ygrid, NBS_cost, cmap=ColorMap("coolwarm"))
  plot(Costs_dual[1,1:2:end],Costs_dual[2,1:2:end],linestyle="--",marker="^",markersize=7,color="blueviolet",alpha=0.7,label="Dual Decomposition")
  plot(Costs2[1,1:2:end],Costs2[2,1:2:end],linestyle="-",marker="o",markersize=7, color="lime",alpha=0.7,label="JP-ADMM")


  annotate("Disagreement",
          	xy=[dis_cost[1];dis_cost[2]],
          	xytext=[dis_cost[1]-0.039;dis_cost[2]-0.01],
            xycoords="data",
            size=14,
            arrowprops=Dict("arrowstyle"=>"fancy",
                "facecolor"=>"black",
                "connectionstyle"=>"angle3,angleA=0,angleB=-100"))

  annotate("NBS",
          	xy=[min11;min22],
          	xytext=[min11+0.005;min22+0.014],
            xycoords="data",
            size=14,
            arrowprops=Dict("arrowstyle"=>"fancy",
                "facecolor"=>"black",
                "edgecolor"=>"black",
                "connectionstyle"=>"angle3,angleA=0,angleB=-90"))


  # xlim(min1,dis_cost[1])
  # ylim(min2,dis_cost[2])
  xlabel("Operator 2 (\$\\phi_2\$)",fontsize=label_fontsize)
  ylabel("Operator 1 (\$\\phi_1\$)",fontsize=label_fontsize)
  legend(loc=2,fontsize=label_fontsize-2)
  # tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
  tight_layout()
  savefig("color_pareto.pdf" )
end

plot_utilization_map()
