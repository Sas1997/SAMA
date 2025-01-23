
clc;
clear;
warning off;
close all

%% Loading Data
global Eload Eload_Previous G T Vw
% load('Data')
Data = csvread('Data.csv');
Eload=Data(:,1)';
G=Data(:,2)';
T=Data(:,3)';
Vw=Data(:,4)';
Eload_Previous=Data(:,1)';

Input_Data;

%% Problem Definition
CostFunction=@(x) fitness(x);        % Cost Function

% Variable: PV number, WT number, Battery number, number of DG, Rated Power Inverter
VarMin = [0   0   0   0  0];   % Lower Bound of Variables
VarMax = [200 200 400 10 20];  % Upper Bound of Variables

lb=VarMin.*[PV WT Bat DG 1];
ub=VarMax.*[PV WT Bat DG 1];

dim=numel(lb);
fobj=@(x) fitness(x);

%% GWO Parameters
Max_iter=100;          % Maximum Number of Iterations
SearchAgents_no=50;    % Population Size (Swarm Size)

Run_Time=1;

empty_particle.BestCost=[];
empty_particle.BestSol=[];
empty_particle.CostCurve=[];
Sol=repmat(empty_particle,Run_Time,1);

for tt=1:Run_Time
    [Alpha_score,Alpha_pos,Convergence_curve,Mean_Cost]=GWO(SearchAgents_no,Max_iter,lb,ub,dim,fobj,tt);
    
    Sol(tt).BestCost=Alpha_score;
    Sol(tt).BestSol=Alpha_pos;
    Sol(tt).CostCurve=Convergence_curve;
    Sol(tt).MeanCurve=Mean_Cost;
end

Best=[Sol.BestCost];
[~,index]=min(Best);
X=Sol(index).BestSol;

%% Results
figure(1)
plot(Sol(index).CostCurve,'-.','LineWidth',2);
hold on
plot(Sol(index).MeanCurve,'-.','LineWidth',2);
legend('Best','Mean')
xlabel('iteration');
ylabel('Cost of Best Solution ')
title('Converage Curve')
hold on

Results
Utility_Results

% Grey Wolf Optimizer
function [Alpha_score,Alpha_pos,Convergence_curve,Mean_Cost]=GWO(SearchAgents_no,Max_iter,lb,ub,dim,fobj,tt)

% initialize alpha, beta, and delta_pos
Alpha_pos=zeros(1,dim);
Alpha_score=inf; %change this to -inf for maximization problems

Beta_pos=zeros(1,dim);
Beta_score=inf; %change this to -inf for maximization problems

Delta_pos=zeros(1,dim);
Delta_score=inf; %change this to -inf for maximization problems

%Initialize the positions of search agents
Positions=zeros(SearchAgents_no,dim);
for i=1:SearchAgents_no
    Positions(i,:)=lb+rand(1,dim).*(ub-lb);
end
Convergence_curve=zeros(1,Max_iter);
Mean_Cost=zeros(1,Max_iter);
fitness=zeros(1,SearchAgents_no);
l=0;% Loop counter

% Main loop
while l<Max_iter
    for i=1:size(Positions,1)  
        
       % Return back the search agents that go beyond the boundaries of the search space
        Flag4ub=Positions(i,:)>ub;
        Flag4lb=Positions(i,:)<lb;
        Positions(i,:)=(Positions(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;               
        
        % Calculate objective function for each search agent
        fitness(i)=fobj(Positions(i,:));
        
        % Update Alpha, Beta, and Delta
        if fitness(i)<Alpha_score 
            Alpha_score=fitness(i); % Update alpha
            Alpha_pos=Positions(i,:);
        end
        
        if fitness(i)>Alpha_score && fitness(i)<Beta_score 
            Beta_score=fitness(i); % Update beta
            Beta_pos=Positions(i,:);
        end
        
        if fitness(i)>Alpha_score && fitness(i)>Beta_score && fitness(i)<Delta_score 
            Delta_score=fitness(i); % Update delta
            Delta_pos=Positions(i,:);
        end
    end
        
    a=2-l*((2)/Max_iter); % a decreases linearly fron 2 to 0
    
    % Update the Position of search agents including omegas
    for i=1:size(Positions,1)
        for j=1:size(Positions,2)     
                       
            r1=rand(); % r1 is a random number in [0,1]
            r2=rand(); % r2 is a random number in [0,1]
            
            A1=2*a*r1-a; % Equation (3.3)
            C1=2*r2; % Equation (3.4)
            
            D_alpha=abs(C1*Alpha_pos(j)-Positions(i,j)); % Equation (3.5)-part 1
            X1=Alpha_pos(j)-A1*D_alpha; % Equation (3.6)-part 1
                       
            r1=rand();
            r2=rand();
            
            A2=2*a*r1-a; % Equation (3.3)
            C2=2*r2; % Equation (3.4)
            
            D_beta=abs(C2*Beta_pos(j)-Positions(i,j)); % Equation (3.5)-part 2
            X2=Beta_pos(j)-A2*D_beta; % Equation (3.6)-part 2       
            
            r1=rand();
            r2=rand(); 
            
            A3=2*a*r1-a; % Equation (3.3)
            C3=2*r2; % Equation (3.4)
            
            D_delta=abs(C3*Delta_pos(j)-Positions(i,j)); % Equation (3.5)-part 3
            X3=Delta_pos(j)-A3*D_delta; % Equation (3.5)-part 3             
            
            Positions(i,j)=(X1+X2+X3)/3;% Equation (3.7)
            
        end
    end
    
    l=l+1;    
    Convergence_curve(l)=Alpha_score;
    Mean_Cost(l)=mean(fitness);
    disp(['Run = ' num2str(tt) '   Iteration = ' num2str(l)  ',    Best = ' num2str(Alpha_score)])
end

end
