clc;
clear;
warning off;

%% Loading Data
global Eload G T Vw
% load('Data')
Data = csvread('Data.csv');
Eload=Data(:,1)';
G=Data(:,2)';
T=Data(:,3)';
Vw=Data(:,4)';

Input_Data;

%% Problem Definition
CostFunction=@(x) fitness(x);        % Cost Function

nVar=5;                % Number of Decision Variables
VarSize=[1 nVar];      % Size of Decision Variables Matrix

% Variable: PV number, WT number, Battery number, number of DG, Rated Power Inverter
VarMin = [0   0   0  0  0];   % Lower Bound of Variables
VarMax = [100 100 60 10 20];  % Upper Bound of Variables

VarMin=VarMin.*[PV WT Bat DG 1];
VarMax=VarMax.*[PV WT Bat DG 1];

%% PSO Parameters
MaxIt=100;      % Maximum Number of Iterations
nPop=50;        % Population Size (Swarm Size)
w=1;            % Inertia Weight
wdamp=0.99;     % Inertia Weight Damping Ratio
c1=2;           % Personal Learning Coefficient
c2=2;           % Global Learning Coefficient

% Velocity Limits
VelMax=0.3*(VarMax-VarMin);
VelMin=-VelMax;

Run_Time=1;

empty_particle.BestCost=[];
empty_particle.BestSol=[];
empty_particle.CostCurve=[];
Sol=repmat(empty_particle,Run_Time,1);

for tt=1:Run_Time
    
 w=1;            % Inertia Weight

%% Initialization
empty_particle.Position=[];
empty_particle.Cost=[];
empty_particle.Velocity=[];
empty_particle.Best.Position=[];
empty_particle.Best.Cost=[];
particle=repmat(empty_particle,nPop,1);

GlobalBest.Cost=inf;

for i=1:nPop

    % Initialize Position
    particle(i).Position=unifrnd(VarMin,VarMax,VarSize);
        
    % Initialize Velocity
    particle(i).Velocity=zeros(VarSize);
    
    % Evaluation
    particle(i).Cost= CostFunction(particle(i).Position);
    
    % Update Personal Best
    particle(i).Best.Position=particle(i).Position;
    particle(i).Best.Cost=particle(i).Cost;
        
    % Update Global Best
    if particle(i).Best.Cost<GlobalBest.Cost
        GlobalBest=particle(i).Best;   
    end   
end

BestCost=zeros(MaxIt,1);
MeanCost=zeros(MaxIt,1);

%% PSO Main Loop
for it=1:MaxIt
    
    for i=1:nPop

        % Update Velocity
        particle(i).Velocity = w*particle(i).Velocity+c1*rand(VarSize).*(particle(i).Best.Position-particle(i).Position)+c2*rand(VarSize).*(GlobalBest.Position-particle(i).Position);
         
        % Apply Velocity Limits
        particle(i).Velocity = max(particle(i).Velocity,VelMin);
        particle(i).Velocity = min(particle(i).Velocity,VelMax);
        
        % Update Position
        particle(i).Position = particle(i).Position + particle(i).Velocity;
        
        % Velocity Mirror Effect
        IsOutside=(particle(i).Position<VarMin | particle(i).Position>VarMax);
        particle(i).Velocity(IsOutside)=-particle(i).Velocity(IsOutside);
        
        % Apply Position Limits
        particle(i).Position = max(particle(i).Position,VarMin);
        particle(i).Position = min(particle(i).Position,VarMax);
        
        % Evaluation
        particle(i).Cost= CostFunction(particle(i).Position);
 
        % Update Personal Best
        if particle(i).Cost<particle(i).Best.Cost
            
            particle(i).Best.Position=particle(i).Position;
            particle(i).Best.Cost=particle(i).Cost;
            
            % Update Global Best
            if particle(i).Best.Cost<GlobalBest.Cost       
                GlobalBest=particle(i).Best;
            end
            
        end
        
    end
    
    BestCost(it)=GlobalBest.Cost;
    temp=0;
    for j=1:nPop
    temp=temp+particle(j).Best.Cost;
    end
    MeanCost(it)=temp/nPop;
    
    disp(['Run time = ' num2str(tt) ' , Iteration = ' num2str(it),', Best Cost = ' num2str(BestCost(it)),', Mean Cost = ' num2str(MeanCost(it))]);
    
    w=w*wdamp;
    
end

Sol(tt).BestCost=GlobalBest.Cost;
Sol(tt).BestSol=GlobalBest.Position;
Sol(tt).CostCurve=BestCost;

end

Best=[Sol.BestCost];
[~,index]=min(Best);
X=Sol(index).BestSol;

%% Results
figure(1)
plot(Sol(index).CostCurve,'-.','LineWidth',2);
xlabel('iteration');
ylabel('Cost of Best Solution ')
title('Converage Curve')
hold on

Results
