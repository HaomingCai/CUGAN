count_1 = 0;
count_2 = 0;
count_3 = 0;
count_4 = 0;
count_5 = 0;
for i = 0:1000000
    disp(i)
    noise_label = round(betarnd(0.5, 1.0)*40);
    if noise_label <= 10
        count_1 = count_1 + 1;
    end
end
disp(count_1)