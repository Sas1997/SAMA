function [Cbuy]= calcTouRate(onPrice, midPrice, offPrice, onHours, midHours, offHours, Month, Day, holidays)
          
    Cbuy = zeros(1,8760);
    for m=2:13 
        
        t_start = floor(24 * sum(Day(1:m - 2)))+1;
        t_end = floor(24 * sum(Day(1:m-1)));
        t_index = t_start:t_end;
        nt = length(t_index);

        if Month(m - 1) == 2  % for summer
            
            tp =  (onHours(2,:));
            tm =  midHours{2};
            toff =  offHours{2};
            P_peak = onPrice(2);
            P_mid = midPrice(2);
            P_offpeak = offPrice(2);
        else  % for winter
            tp =  onHours(1,:);
            tm =  midHours{1};
            toff =  offHours{1};
            P_peak = onPrice(1);
            P_mid = midPrice(1);
            P_offpeak = offPrice(1);
        end
        Cbuy(t_index) = P_offpeak ; % set all hours to offpeak by default
    
        for d=2:Day(m - 1) +1
            idx0 = t_index(tp) + 24 * (d - 2) +1;
            Cbuy(idx0) = P_peak;
            idx1 = t_index(tm) + 24 * (d - 2)+1;
            Cbuy(idx1) = P_mid;
        end
    end
    for d=1:365
        if rem((d - 1),7) >= 5
            st = 24 * (d - 1)+2 ;
            ed = 24 * d ;
            Cbuy(st: ed) = P_offpeak;

            
        end
    end
    for d=1:365
      if ismember(d,holidays)
            st = 24 * (d - 1) + 2;
            ed = 24 * d ;
            Cbuy(st: ed) = P_offpeak;

      end
    end
