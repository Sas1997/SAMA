function [Cbuy,Base_charge]= calcTieredRate(Base_charge_tier_1,Base_charge_tier_2,Base_charge_tier_3,prices, tierMax, load, daysInMonth)
    Cbuy = zeros(1,8760);
    Base_charge = zeros(1,12);
    hCount = 1;
    for m=1:12
        monthlyLoad = 0;
        for h=1:(24 * daysInMonth(m))
            monthlyLoad = monthlyLoad +load(hCount);

            if monthlyLoad < tierMax(1)
                
                Cbuy(hCount) = prices(1);
            elseif monthlyLoad < tierMax(2)
                 
                Cbuy(hCount) = prices(2);
            else
                Cbuy(hCount) = prices(3);
            end
            hCount = hCount+1;
         end
         if monthlyLoad < tierMax(1)
             Base_charge(m) = Base_charge_tier_1;
         elseif monthlyLoad < tierMax(2)
             Base_charge(m) = Base_charge_tier_2;
         else 
             Base_charge(m) = Base_charge_tier_3;
         end 
    end
end