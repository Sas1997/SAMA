Input_Data;
if Grid==1
%% Utility
writematrix(Cbuy,'Cbuy.csv','Delimiter',',');
%%
p_c = zeros(12,31);
index = 1;
for m=1:12
        index1 = index;
        for d=1:daysInMonth(m)
            cost =mean(Cbuy(index1:index1+23));
            p_c(m,d) = cost;
            index1 = index1 + 24;
        end
        index = (24 * daysInMonth(m)) + index;
end
%%
L_c = zeros(12,31);
index = 1;
for m=1:12
        index1 = index;
        for d=1:daysInMonth(m)
            Total_daily_load =sum(Eload(index1:index1+23));
            L_c(m,d) = Total_daily_load;
            index1 = index1 + 24;
        end
        index = (24 * daysInMonth(m)) + index;
end

E_c=round(p_c.*L_c,2);
%%
figure(7)
p_c(p_c==0) = nan;
heatmap(round(p_c,2), 'Colormap', jet);
ax = gca;
ax.YData =["January" "February" "March" "April" "May" "June" "July" "August" "September" "October" "November" "December"]

figure(8)
E_c(E_c==0) = nan;
heatmap(E_c, 'Colormap', jet);
ax = gca;
ax.YData =["January" "February" "March" "April" "May" "June" "July" "August" "September" "October" "November" "December"]

figure(9)
imagesc(Cbuy)
colormap jet
colorbar('southoutside')

if sum(Psell)>0.1
S_c = zeros(12,31);
index = 1;
for m=1:12
        index1 = index;
        for d=1:daysInMonth(m)
            Total_daily_sell =sum(Psell(index1:index1+23));
            S_c(m,d) = Total_daily_sell;
            index1 = index1 + 24;
        end
        index = (24 * daysInMonth(m)) + index;
end
Ss_c=round(Csell.*S_c,2);

figure(10)
Ss_c(Ss_c==0) = nan;
heatmap(Ss_c, 'Colormap', jet);
ax = gca;
ax.YData =["January" "February" "March" "April" "May" "June" "July" "August" "September" "October" "November" "December"]

end
end