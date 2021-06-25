%% I. Empty environment variables
clear all
close all
clc
 
 %% II. Training set/test set generation
%%
 % 1. Import data
load iris.data
 
%%
 % 2 randomly generates training sets and test sets
P_train = [];
T_train = [];
P_test = [];
T_test = [];
for i = 1:3
    temp_input = features((i-1)*50+1:i*50,:);
    temp_output = classes((i-1)*50+1:i*50,:);
    n = randperm(50);
         % training set - 120 samples
    P_train = [P_train temp_input(n(1:40),:)'];
    T_train = [T_train temp_output(n(1:40),:)'];
         % test set - 30 samples
    P_test = [P_test temp_input(n(41:50),:)'];
    T_test = [T_test temp_output(n(41:50),:)'];
end
 
 %% III. Model building 
result_grnn = [];
result_pnn = [];
time_grnn = [];
time_pnn = [];
for i = 1:4
    for j = i:4
        p_train = P_train(i:j,:);
        p_test = P_test(i:j,:);
       %% 
                 % 1. GRNN creation and simulation test
        t = cputime;
                 % Create network
        net_grnn = newgrnn(p_train,T_train);
                 % simulation test
        t_sim_grnn = sim(net_grnn,p_test);
        T_sim_grnn = round(t_sim_grnn);
        t = cputime - t;
        time_grnn = [time_grnn t];
        result_grnn = [result_grnn T_sim_grnn'];
       %%
                 % 2. PNN creation and simulation test
        t = cputime;
        Tc_train = ind2vec(T_train);
                 % Create network
        net_pnn = newpnn(p_train,Tc_train);
                 % simulation test
        Tc_test = ind2vec(T_test);
        t_sim_pnn = sim(net_pnn,p_test);
        T_sim_pnn = vec2ind(t_sim_pnn);
        t = cputime - t;
        time_pnn = [time_pnn t];
        result_pnn = [result_pnn T_sim_pnn'];
    end
end
 
 %% IV. Performance Evaluation
%%
 % correct rate accuracy
accuracy_grnn = [];
accuracy_pnn = [];
time = [];
for i = 1:10
    accuracy_1 = length(find(result_grnn(:,i) == T_test'))/length(T_test);
    accuracy_2 = length(find(result_pnn(:,i) == T_test'))/length(T_test);
    accuracy_grnn = [accuracy_grnn accuracy_1];
    accuracy_pnn = [accuracy_pnn accuracy_2];
end
 
%%
 % 2. Comparison of results
result = [T_test' result_grnn result_pnn]
accuracy = [accuracy_grnn;accuracy_pnn]
time = [time_grnn;time_pnn]
 %% V. drawing
figure(1)
plot(1:30,T_test,'bo',1:30,result_grnn(:,4),'r-*',1:30,result_pnn(:,4),'k:^')
grid on
 Xlabel('test set sample number')
 Ylabel('test set sample category')
 String = {'test set prediction result comparison (GRNN vs PNN)';['correct rate: ' num2str(accuracy_grnn(4)*100) '%(GRNN) vs ' num2str(accuracy_pnn(4)*100) '%( PNN)']};
title(string)
 Legend('true value', 'GRNN predicted value', 'PNN predicted value')
figure(2)
plot(1:10,accuracy(1,:),'r-*',1:10,accuracy(2,:),'b:o')
grid on
 Xlabel('model number')
 Ylabel('test set correct rate')
 Title ('10 model test set correct rate comparison (GRNN vs PNN)')
legend('GRNN','PNN')
figure(3)
plot(1:10,time(1,:),'r-*',1:10,time(2,:),'b:o')
grid on
 Xlabel('model number')
 Ylabel('Runtime(s)')
 Title ('Runtime Comparison of 10 Models (GRNN vs PNN)')
legend('GRNN','PNN')