function [Cbuy] = calcFlatRate(price)
    Cbuy = zeros(1,8760);
    for h=1:8760
        Cbuy(h) = price;
    end
end