function [Cbuy]= calcMonthlyTieredRate(prices, tierMax, load)
    Cbuy = zeros(1,8760);
    daysInMonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    hCount = 1;
    for m=1:12
        monthlyLoad = 0;
        for h=1:(24 * daysInMonth(m))
            monthlyLoad = monthlyLoad + load(hCount);

            if monthlyLoad < tierMax(m,1)
                Cbuy(hCount) = prices(m,1);
            elseif monthlyLoad < tierMax(m,2)
                Cbuy(hCount) = prices(m,2);
            else
                Cbuy(hCount) = prices(m,3);
            end
            hCount = hCount + 1;
  
        end
    end
end