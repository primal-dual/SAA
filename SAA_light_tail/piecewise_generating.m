%generating v and s 
clear
clc
rng(114514);                
m=10;
v = randn(1,m);
s = randn(1,m);
save('piecewise_parameter')