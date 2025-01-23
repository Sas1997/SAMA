function [Cbuy]=calcSeasonalRate(prices, months, daysInMonth)
    Cbuy = zeros(1,8760);
    hCount = 1;

    for m=1:12
        hoursStart = hCount;
        hoursEnd = hoursStart + (24 * daysInMonth(m));
        hoursRange = hoursStart:hoursEnd-1;
        Cbuy(hoursRange) = prices{1}(months(m));
        hCount = hoursEnd ; 

    end
end