using PyPlot

fig_size = (8,6) #For 3 power
fig_size1 = (7.1,4.4)
label_fontsize = 14
legend_fontsize = label_fontsize-1
label_operators = ["Operator 1", "Operator 2", "Operator 3"]
label_BSs  = ["BS1","BS2","BS3","BS4","BS5","BS6"]
patterns =["","."]
colors1=["dodgerblue","mediumseagreen","coral"]
colors2=["lightskyblue","mediumaquamarine","sandybrown"]
colors3=["dodgerblue","green","orangered"]
markers = ["o","x", "s","v","d","^","<",">","+","*","h", "p"]

function plt_convergence(k,Costs,Primals, disagreement, dis_rhos, centralized)
  figure(1,figsize=fig_size1)
  colors=["b","r","k"]

  plot(1:k,centralized[1]*ones(k),color=colors[2],linestyle="--",label="Centralized")
  plot(1:k,Costs[1,:],color=colors[1],marker="o", markersize=6,markevery=7,label="Decentralized")

  for i = 2:Numb_Operators
    plot(1:k,centralized[i]*ones(k),color=colors[2],linestyle="--")
    plot(1:k,Costs[i,:],color=colors[1],marker="o", markersize=6,markevery=7)
    # plot(disagreement[i]*ones(k),color=colors[3],linestyle="--")
  end
  xlim(0,k)
  xlabel("Iterations",fontsize=label_fontsize)
  ylabel("Flow-level cost (\$\\phi_i\$)",fontsize=label_fontsize)
  # title("Sharing Solution Convergence")
  legend(loc="best",fontsize=label_fontsize-1)
  tight_layout()
  savefig("Case_1_Cost_Convergence.pdf")

  figure(2,figsize=fig_size1)
  for i = 1:Numb_Operators
    for j = 1:Numb_BS
      plot(1:k,(1-m) *Q * Primals[i,j,:] + m*Q)

    end
  end

  plot(1:k,B*ones(k),color="r",linestyle=":",label="Limited BK Power")
  xlim(0,k)
  legend(loc="best",fontsize=legend_fontsize)
  xlabel("Iterations")
  ylabel("Power (W)")
  title("Power Convergence")
  tight_layout()
  savefig("Case_1_Power_Convergence.pdf")
end

function plt_convergence1(k,Costs,Primals, disagreement, dis_rhos, centralized)
  Costs_dual = zeros(Numb_Operators,Max_Iters)
  filename= "cost_convergence.h5"
  h5open(filename, "r") do file
    Costs_dual = read(file, "Costs")  # alternatively, say "@write file A"
  end
  figure(1,figsize=fig_size1)
  colors=["dodgerblue","red","sandybrown"]
  marker_size=6
  stride = 9

  # plot(1:k,Costs_dual[1,:],color=colors1[3],marker="o", markersize=marker_size,markevery=stride,linewidth=1.5,label="Dual Decomposition")
  plot(1:k,Costs[1,:],color=colors[1],marker="d", markersize=marker_size,markevery=stride,linewidth=1.5,label="JP-ADMM")
  plot(1:k,centralized[1]*ones(k),color=colors[2],linestyle="--",label="Centralized")


  for i = 2:Numb_Operators
    # plot(1:k,Costs_dual[i,:],color=colors1[3],marker="o", markersize=marker_size,markevery=stride,linewidth=1.5)
    plot(1:k,Costs[i,:],color=colors[1],marker="d", markersize=marker_size,markevery=stride,linewidth=1.5)
    plot(1:k,centralized[i]*ones(k),color=colors[2],linestyle="--")
    # plot(disagreement[i]*ones(k),color=colors[3],linestyle="--")
  end
  xlim(0,k)
  xlabel("Iterations",fontsize=label_fontsize)
  ylabel("Flow-level cost (\$\\phi_i\$)",fontsize=label_fontsize)
  # title("Sharing Solution Convergence")
  legend(loc="best",fontsize=legend_fontsize)
  tight_layout()
  savefig("Case_1_Cost_Convergence.pdf")
end


function plt_comparison(Primals, Costs, disagreement, dis_rhos)
  figure(3,figsize=fig_size1)
  idx = [1:Numb_Operators;]
  width = 0.3
  # subplot()
  bar(idx-width/2-0.005, disagreement-Numb_BS, width, color=colors1,label="No Sharing",alpha=.9,hatch=patterns[1])
  bar(idx+width/2, Costs-Numb_BS, width, color=colors2, label="Sharing",alpha=0.7,hatch=patterns[2])

  axis("tight")
  legend(loc="best",fontsize=legend_fontsize)
  xticks(idx,label_operators,fontsize=label_fontsize)
  xlim(0.6,3.5)
  ylim(2.,3.)
  # ylim(6.2,7.5)
  # set_xticklabels(("Operator1", "Operator2", "Operator3"))
  ylabel("Expected number of traffic flows (\$L_i\$)",fontsize=label_fontsize)
  # title("Flows Delay comparison")
  tight_layout()
  savefig("Case_1_Total_Cost_comparison.pdf")
  println("Power Sharing Flows")
  println(disagreement-Numb_BS)
  println(Costs-Numb_BS)

  Total_Power_dis = zeros(Numb_Operators)
  Total_Power_bar = zeros(Numb_Operators)
  for i = 1:Numb_Operators
    Total_Power_dis[i] = sum((1-m)*Q*dis_rhos[i,:] + m*Q)
    Total_Power_bar[i] = sum((1-m)*Q*Primals[i,:] + m*Q)
  end

  figure(4,figsize=fig_size1)
  idx = [1:Numb_Operators;]
  width = 0.3
  # subplot()
  bar(idx-width/2-0.005, Total_Power_dis, width, color=colors1,label="No Sharing",alpha=.9,hatch=patterns[1])
  bar(idx+width/2, Total_Power_bar, width, color=colors2, label="Sharing",alpha=0.7,hatch=patterns[2])

  plot([0,3.5],[1940, 1940], color = "r", linestyle=":")
  axis("tight")
  legend(loc="best",fontsize=legend_fontsize)
  xticks(idx,label_operators,fontsize=label_fontsize)
  xlim(0.6,3.5)
  ylim(2100,2500)
  # set_xticklabels(("Operator1", "Operator2", "Operator3"))
  ylabel("Power (W)",fontsize=label_fontsize)
  # title("Total Power comparison")
  tight_layout()
  savefig("Case_1_Total_Power_comparison.pdf")
  println(" **** ",Total_Power_bar)
  println(" **** ",Total_Power_dis)

  figure(5,figsize=fig_size)
  for i = 1:Numb_Operators
    # figure(3+i)
    subplot(3,1,i)
    idx = [1:Numb_BS;]
    width = 0.3
    bar(idx-width/2-0.005, (1-m)*Q*dis_rhos[i,:] + m*Q, width, color=colors1[i],label="No Sharing",alpha=.9,hatch=patterns[1])
    bar(idx+width/2, (1-m)*Q*Primals[i,:] + m*Q, width, color=colors2[i], label="Sharing",alpha=0.7,hatch=patterns[2])
    # axis("tight")

    if i == 1
      tight_layout(pad=0.4, w_pad=0.5, h_pad=2.)
      legend(bbox_to_anchor=(0.2, 0.72, 0.6, .1), loc=3,
             ncol=2, mode="expand", borderaxespad=0.)
    else
      tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    end

    xticks(idx,label_BSs,fontsize=label_fontsize)
    title(string("Operator ",i),y=1.02, fontsize=label_fontsize)
    xlim(0.6,6.5)
    ylim(200,550)
    # set_xticklabels(("BS1", "BS2", "BS3"))
    # xlabel("Base Station index")
    ylabel("Power (W)",fontsize=label_fontsize)
  end
  # suptitle("Power Comparison",y=1.02)
  savefig("Case_1_Total_Power_BS_all.pdf")
end

function plt_delay_comparison(shared, no_shared)
  figure(6,figsize=fig_size1)
  idx = [1:Numb_Operators;]
  width = 0.3
  # subplot()
  bar(idx-width/2-0.005, no_shared*1000, width,color=colors1,label="No Sharing",alpha=0.9,hatch=patterns[1])
  bar(idx+width/2, shared*1000, width, color=colors2, label="Sharing",alpha=0.7,hatch=patterns[2])
  axis("tight")
  legend(loc="best",fontsize=legend_fontsize)
  xticks(idx,label_operators,fontsize=label_fontsize)
  xlim(0.6,3.5)
  ylim(1.8,2.4)
  # set_xticklabels(("Operator1", "Operator2", "Operator3"))
  ylabel("Average Delay (ms)",fontsize=label_fontsize)
  # title("Flows Delay comparison")
  tight_layout()
  savefig("Case_1_Avg_Delay_comparison.pdf")
  println(" ++++ ",shared)
  println(" ++++ ",no_shared)
end

function plt_prob_comparison(probs, dis_probs)
  for j= 1:Numb_BS
    probs_plt = zeros(dim_x,dim_y)
    figure(6+j,figsize=(12, 4.2))
    subplot(1,2,1)
    for i = 1:1
      for x = 1:dim_x
        for y = 1:dim_y
          # probs_plt[x,y] = maximum(dis_probs[i,(x-1)*dim_x+y,:])
          probs_plt[x,y] = dis_probs[i,(x-1)*dim_x+y,j]
          # savefig(string("Case_1_Mat_Power_BS_",i,".pdf"))
        end
      end

      pcolor(probs_plt,cmap="YlOrRd",vmin=0.0, vmax=1.001)
      colorbar()
      title("No Sharing", fontsize = 17)
    end

    subplot(1,2,2)
    for i = 1:1
      for x = 1:dim_x
        for y = 1:dim_y
          # probs_plt[x,y] = maximum(probs[i,(x-1)*dim_x+y,:])
          probs_plt[x,y] = probs[i,(x-1)*dim_x+y,j]
        end
      end

      pcolor(probs_plt,cmap="YlOrRd",vmin=0.0, vmax=1.001)
      colorbar()
      title("Sharing", fontsize = 17)
    end
    savefig(string("Case_1_Nash_Mat_Power_BS_",j,".pdf"))
  end
end

function plt_spatial_arrival(arrivals)
  figure(15,figsize=(12, 4.2))
  subplot(1,2,1)
  println("**** ",size(arrivals))
  pcolor(arrivals[1,:,:],cmap="YlOrRd",vmin=0.0, vmax=25.)
  colorbar()
  title("Operator 1", fontsize = 17)

  subplot(1,2,2)

  pcolor(arrivals[3,:,:],cmap="YlOrRd",vmin=0.0, vmax=25.)
  colorbar()
  title("Operator 3", fontsize = 17)
  tight_layout()
  savefig(string("Case_Spatial_Arrival.pdf"))

end

function plt_obj_comparison(Costs, Primals, disagreement, dis_rhos)
  # figure(11,figsize=fig_size1)
  # idx = [1:3;]
  # width = 0.3
  # # subplot()
  # bar(idx-width, disagreement, width, color="b",label="No Sharing",alpha=0.6)
  # bar(idx, Delay, width, color ="r", label="Sharing",alpha=0.6)
  # axis("tight")
  # legend(loc="best")
  # xticks(idx,labels)
  # xlim(0.6,3.5)
  # ylim(6,7.5)
  # # set_xticklabels(("Operator1", "Operator2", "Operator3"))
  # ylabel("Average delay (sec)")
  # title("Operators Flows Delay comparison")
  # savefig("Case_1_Total_Delay_Alpha0.pdf")
  #
  Total_Power_dis = zeros(Numb_Operators)
  Total_Power_bar = zeros(Numb_Operators)
  for i = 1:Numb_Operators
    Total_Power_dis[i] = sum((1-m)*Q*dis_rhos[i,:] + m*Q)
    Total_Power_bar[i] = sum((1-m)*Q*Primals[i,:] + m*Q)
  end

  figure(12,figsize=fig_size1)
  idx = [1:Numb_Operators;]
  width = 0.3
  # subplot()
  bar(idx-width/2-0.005, Total_Power_dis, width,color=colors1,label="No Sharing",alpha=0.9,hatch=patterns[1])
  bar(idx+width/2, Total_Power_bar, width, color=colors2, label="Sharing",alpha=0.7,hatch=patterns[2])
  axis("tight")
  legend(loc="best",fontsize=legend_fontsize)
  xticks(idx,label_operators)
  xlim(0.6,3.5)
  ylim(1500,2000)
  ylabel("Power (W)",fontsize=label_fontsize)
  # title("Total Power comparison")
  tight_layout()
  savefig("Case_1_Total_Power_Alpha0.pdf")
  println(Total_Power_bar)
  println(Total_Power_dis)
end

function plt_total_cost_limited(Power_Change, total_cost)
  figure(13,figsize=fig_size1)
  colors=["b","r","g"]

  plot(Power_Change, total_cost[1,:],color=colors[1],linestyle="-",marker="s", markersize=7,label="No Sharing")
  plot(Power_Change, total_cost[2,:],color=colors[2],linestyle="-",marker="o", markersize=7,label="Sharing")
  # plot(Power_Change, total_cost[3,:],color=colors[3],linestyle="--",marker="*", markersize=7,label="Join_Resource")
  legend(loc="best",fontsize=legend_fontsize)
  xlabel("Backup Power Capacity (W)",fontsize=label_fontsize)
  ylabel("Flow-level cost (\$\\phi_i\$)",fontsize=label_fontsize)
  # title("Total cost Comparison")
  tight_layout()
  grid("on")
  savefig("Case_1_Limit_Power_Comparison.pdf")
  println("MPC COST SEQ")
  println(total_cost[1,:])
  println(total_cost[2,:])

end

function plt_MPC_cost_comparison( individuals, seq_individuals, sharings, seq_sharing)
  figure(1,figsize=fig_size1)
  # colors=["b","r","g"]
  T = 1:T_req

  # plot(T,seq_individuals[1,:],color=colors[1],linestyle="--",marker="s", markersize=7,label="MPC-Unsharing_Op1")
  # plot(T,seq_sharing[1,:],color=colors[2],linestyle="--",marker="o", markersize=7,label="MPC-Sharing_Op2")

  for i = 1:Numb_Operators
    plot(T,seq_individuals[i,:],color=colors3[i],linestyle="-",marker=markers[i], markersize=7,label=string("No Sharing: Op",i))
  end
  for i = 1:Numb_Operators
    plot(T,seq_sharing[i,:],color=colors3[i],linestyle="-",marker=markers[3+i], markersize=7,label=string("Sharing: Op",i))
  end

  legend(loc="best",fontsize=legend_fontsize)
  xlabel("Time slot (10 minutes)",fontsize=label_fontsize)
  ylabel("Flow-level cost (\$\\phi_i\$)",fontsize=label_fontsize)
  # title("Cost Sequence Comparison")
  tight_layout()
  grid("on")
  savefig("Case_2_Cost_Sequence_Comparison.pdf")

  figure(2,figsize=fig_size1)
  idx = [1:Numb_Operators;]
  width = 0.3
  # subplot()
  bar(idx-width/2-0.005, individuals, width,color=colors1,label="No Sharing",alpha=0.9,hatch=patterns[1])
  bar(idx+width/2, sharings, width, color=colors2, label="Sharing",alpha=0.7,hatch=patterns[2])
  axis("tight")
  legend(loc="best",fontsize=legend_fontsize)
  xticks(idx,label_operators)
  xlim(0.6,3.5)
  ylim(40,44)
  # set_xticklabels(("Operator1", "Operator2", "Operator3"))
  ylabel("Flow-level cost (\$\\phi_i\$)",fontsize=label_fontsize)
  # title("Flows Cost of each operator")
  tight_layout()
  savefig("Case_2_Total_Cost_comparison.pdf")
  println("MPC COST")
  println(individuals)
  println(sharings)
end

function plt_MPC_remaning_enery( bargaining_energy, dis_energy)
  Total_sharing_energy  = zeros(Numb_Operators)
  Total_unshared_energy = zeros(Numb_Operators)
  for i = 1:Numb_Operators
    Total_sharing_energy[i] = sum(bargaining_energy[i,end,:])
    Total_unshared_energy[i] = sum(dis_energy[i,end,:])
  end

  figure(3,figsize=fig_size1)
  idx = [1:Numb_Operators;]
  width = 0.3
  # subplot()
  bar(idx-width/2 -0.005, Total_unshared_energy, width,color=colors1,label="No Sharing",alpha=0.9,hatch=patterns[1])
  bar(idx+width/2, Total_sharing_energy, width, color=colors2, label="Sharing",alpha=0.7,hatch=patterns[2])
  axis("tight")
  legend(loc="best",fontsize=legend_fontsize)
  xticks(idx,label_operators)
  xlim(0.6,3.5)
  ylim(0,200)
  ylabel("Energy (Wh)",fontsize=label_fontsize)
  # title("Total Remaining Energy of each operator")
  tight_layout()
  savefig("Case_2_Total_Remaining_Energy.pdf")
  println("MPC REMAINING ENERGY")
  println(Total_unshared_energy)
  println(Total_sharing_energy)
end
