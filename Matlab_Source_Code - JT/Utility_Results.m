%% JT
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
figure(6)
p_c(p_c==0) = nan;
heatmap(p_c, 'Colormap', jet);
ax = gca;
ax.YData =["January" "February" "March" "April" "May" "June" "July" "August" "September" "October" "November" "December"]

figure(7)
imagesc(Cbuy)
colormap jet
colorbar('southoutside')
end