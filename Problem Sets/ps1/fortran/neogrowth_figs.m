%--------------------------------------------------------------------------
% File Name: neogrowth_figs.m
% Author: Philip Coyle
% Date Created: 09/16/2020
% cd
% /Users/philipcoyle/Documents/School/University_of_Wisconsin/SecondYear/Fall_2020/Computation/ProblemSets/PS2
%--------------------------------------------------------------------------

clear all
close all
clc

%% Load Data
dat = dlmread('pfs_neogrowth.dat');

col.grid_Z = 1;
col.grid_k = 2;
col.pf_c =  3;
col.pf_k =  4;
col.pf_v =  5;

%% Sort Data
Z_grid = [1.25, 0.2];
k_grid = linspace(0.01, 45, size(dat,1)/2);

% Allocate space for PFs
%pf_c = zeros(length(k_grid), length(Z_grid));
%pf_k = zeros(length(k_grid), length(Z_grid));
%pf_v = zeros(length(k_grid), length(Z_grid));
X = zeros(length(k_grid), length(Z_grid), 3);

for i_Z = 1:length(Z_grid)
    z_inx = find(dat(:,col.grid_Z) == Z_grid(i_Z));
    
    X(:, i_Z, 1) = dat(z_inx, col.pf_v);
    X(:, i_Z, 2) = dat(z_inx, col.pf_k);
    X(:, i_Z, 3) = dat(z_inx, col.pf_k) - dat(z_inx, col.grid_k);
end
        
%% Plotting
color = {'k','b'};
head = {'Value','Capital Inv.','Change in Capital Stock'};

fig(1) = figure(1);
for i = 1:size(X,3)
    for j = 1:length(Z_grid)
        subplot(2,3,i)
        box on
        grid on
        hold on
        if i == 1
            h(j) = plot(k_grid, X(:,j,i), 'LineWidth', 2, 'color', color{j});
        else
            plot(k_grid, X(:,j,i), 'LineWidth', 2, 'color',color{j});
        end
        title(head{i},'FontSize',15,'FontWeight','normal')
        set(gca,'FontSize',15,'XLim',[0 45])
        xlabel('Capital Stock','FontSize',15)        
    end
    if i == 1
        L = legend([h(1) h(2)],'Z = $\overline{Z}$','Z = $\underline{Z}$','interpreter','Latex');
        set(L,'Location','SouthEast','FontSize',15);
    end
end

        
        
set(fig(1),'PaperOrientation','Landscape');
set(fig(1),'PaperPosition',[0 0 11 8.5]);
print(fig(1),'-depsc','pfs_neogrowth.eps');

