#### CLASS OPERATOR ####
type MOperator
  init_lamb
  idx
  system_density
end

type MPC_MOperator
  init_lamb
  idx
  # Et        #Remaining Energy
  # Et_cut    #Cut off Energy
  # Et_inf    #infrastructure Energy
  system_density
  traffics
end

# function PoissonPP(rt)
#   N = rand(Poisson( rt*Dx*Dy ))
#   x = rand(DiscreteUniform(0,Dx),N)
#   y = rand(DiscreteUniform(0,Dy),N)
#   P = hstack([x y])
# return P
# end

function traffic_density_unitarea(rt)
  mu_size = 1.
  sigma = 1.

  N = rand(Poisson( rt*Dx*Dy ))
  traffic = rand(LogNormal(mu_size,sigma),N)
  # traffic = mu_size*ones(N)
  density = sum(traffic)
  #    print traffic
  #    print density
  return density
end

function traffic_density_region(op::MOperator)
  traff_filename = string("Traffic_pattern_",op.idx,".h5")
  arrv_filename = string("Arrival_matrix_",op.idx,".h5")

  if REUSED_TRAFFIC
    h5open(traff_filename, "r") do file
      return read(file, "traffic")  # alternatively, say "@write file A"
    end
  else
    traffic_matrix = zeros(dim_x,dim_y)
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
    # println(traffic_matrix)

    h5open(traff_filename, "w") do file
      write(file, "traffic", traffic_matrix)  # alternatively, say "@write file A"
    end
    h5open(arrv_filename, "w") do file
      write(file, "arrvial", arrival_matrix)
      write(file, "total_arrival", sum(arrival_matrix))  # alternatively, say "@write file A"
      # println(arrival_matrix)
    end
    return traffic_matrix
  end
end

function system_load(op::MOperator)
  traffics = traffic_density_region(op)
  system_density = zeros(dim_x*dim_y, Numb_BS)

  for x = 1:dim_x
      for y = 1:dim_y
          system_density[(x-1)*dim_x + y,:] = traffics[x,y] ./ Capacity[(x-1)*dim_x + y,:]
      end
  end
  op.system_density = system_density
end

function check_solutions(op::MOperator,p,mode)

  for x=1:(dim_x*dim_y)
    if abs(sum(p[x,:]) - 1) > 1e-5
       println("Fail: Association with ", x)
      #  println(sum(p[x,:]))
    end
  end

  rho = zeros(Numb_BS)
  total_costs = 0

  for j =1:Numb_BS
      rho[j] = vecdot(op.system_density[:,j], p[:,j])
      delta_power = B - (1 - m) * rho[j] * Q -  m * Q
      if(delta_power < -1e-4) && (mode == 1)
          println("Fail: Overload Power BS ",j ," :",delta_power)
      end
      total_costs += 1./(1-rho[j])
  end

  # if DEBUG >0 println("rho: ", rho); end
  #            print (1 - m) * rho[j] * Q +  m * Q
  if mode == 1
  #            print "Delay: ", total_delay
      return total_costs, rho, p
  elseif mode == 2
      return total_costs, rho, p
  else mode == 3
      return total_costs, (1 - m) * rho[:] * Q +  m * Q
  end
end

#### MAIN FUNCTIONS ####

function compute_gain(d)
  f = 2.5 #GHz

  path_loss = 35.2 +35*log10(d) + 26*log10(f/2.)   #dB
  shadowing_loss = 8                          #dB
  return - (path_loss + shadowing_loss)
end

function compute_Shannon_capcity(x)
  Tx_Power = 10^( 43. / 10) / 1000 #43 dBm -> Watts
  Band = 10e6 # 10 MHz
  Power_Noise = 10^( -174. * Band/ 10.)/1000.   #-174dBm/Hz
  # Power_Noise = 0

  gain = zeros(Numb_BS)
  SINR = zeros(Numb_BS)
  Capacity = zeros(Numb_BS)

  for i = 1:Numb_BS
      gain[i]= 10^( compute_gain(norm(BS[i,:] - x)) /10. )    #convert from dB to power ratios
  end
  # println(gain)
  for i = 1:Numb_BS
  #        interference = pow(10, -50. / 10)/1000   #-5dBm
      interference = 5e-13
      # interference = 1e-12
  #
  #        for j in range(numb_BS):
  #            if(j!=i):
  #                interference += Tx_Power*gain[j]

      SINR[i] = Tx_Power * gain[i] / (Power_Noise + interference)
      Capacity[i] = Band * log2(1 + SINR[i])                               #Shannon capacity bps
  end
  # println(SINR)
  return Capacity
end

function compute_capacity()
  capacity_list = zeros(dim_x * dim_y, Numb_BS)

  for x = 0:(dim_x-1)
      for y = 0:(dim_y-1)
          loc_area = [x*Dx,y*Dx]
          capacity_list[x*dim_x+y+1,:] = copy(compute_Shannon_capcity(loc_area))
          # println(compute_Shannon_capcity(loc_area))
      end
  end
  return capacity_list
end
