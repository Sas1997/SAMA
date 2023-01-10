function [Cbuy]= calcMonthlyRate(prices, daysInMonth)
    Cbuy = zeros(1,8760);
    hCount = 1;
    for m=1:12
        for h=1:(24 * daysInMonth(m))
            Cbuy(hCount) = prices{1}(m);
            hCount = hCount +1;

        end
    end
end