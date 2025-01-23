function [Cbuy]= calcTouRate(year,onPrice, midPrice, offPrice, onHours, midHours, offHours, Month, Day, holidays)
          
startDate = datetime(year, 1, 1);
firstDayOfWeek = weekday(startDate); % Get the day of the week for the first day
Cbuy = zeros(1, 8760);

for m = 1:12
    
    t_start = 24 * sum(Day(1:m - 1)) + 1;
    t_end = 24 * sum(Day(1:m));
    t_index = t_start:t_end;
    nt = length(t_index);
    
    if Month(m) == 1 % for summer
        
        tp = onHours(1, :);
        tm = midHours{1};
        toff = offHours{1};
        P_peak = onPrice(1);
        P_mid = midPrice(1);
        P_offpeak = offPrice(1);
        
    else % for winter
        tp = onHours(2, :);
        tm = midHours{2};
        toff = offHours{2};
        P_peak = onPrice(2);
        P_mid = midPrice(2);
        P_offpeak = offPrice(2);
    end
    Cbuy(t_index) = P_offpeak; % set all hours to offpeak by default
    
    for d = 1:Day(m)
        idx0 = t_index(tp) + 24 * (d - 1);
        Cbuy(idx0) = P_peak;
        idx1 = t_index(tm) + 24 * (d - 1);
        Cbuy(idx1) = P_mid;
    end
end

for d = 1:365
    currentDate = startDate + days(d - 1);
    currentDayOfWeek = weekday(currentDate);
    
    if currentDayOfWeek == 1 || currentDayOfWeek == 7 % Saturday or Sunday
        st = 24 * (d - 1) + 1;
        ed = 24 * d;
        Cbuy(st:ed) = P_offpeak;
    end
    
    if ismember(d, holidays)
        st = 24 * (d - 1) + 1;
        ed = 24 * d;
        Cbuy(st:ed) = P_offpeak;
    end
end