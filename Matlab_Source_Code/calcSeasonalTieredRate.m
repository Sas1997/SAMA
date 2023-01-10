function [Cbuy,Base_charge]= calcSeasonalTieredRate(Base_charge_tier_1,Base_charge_tier_2,Base_charge_tier_3,prices, tierMax, load, months)
    daysInMonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    Cbuy = zeros(1,8760);
    Base_charge = zeros(1,12);

    hCount = 1;
    for m=1:12
        monthlyLoad = 0;
        for h=1:(24 * daysInMonth(m))
            monthlyLoad= monthlyLoad + load(hCount);
        hCount = hCount + 1;
        end
        totalmonthlyload(m,1)=monthlyLoad;
    end

    hCount = 1;
    for m=1:12
        for h=1:(24 * daysInMonth(m))  
            if totalmonthlyload(m,1) < tierMax(months(m),1)
                Cbuy(hCount) = prices{months(m)}(1);
            elseif totalmonthlyload(m,1) < tierMax(months(m),2)
                Cbuy(hCount) = prices{months(m)}(2);
            else
                Cbuy(hCount) = prices{months(m)}(3);
            end
            hCount = hCount + 1;      
        end
        if totalmonthlyload(m,1) < tierMax(months(m),1)
             Base_charge(m) = Base_charge_tier_1;
         elseif totalmonthlyload(m,1) < tierMax(months(m),2)
             Base_charge(m) = Base_charge_tier_3;
         else 
             Base_charge(m) = Base_charge_tier_3;
         end 
    end
 end
    