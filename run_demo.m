clear;
clc
folder_now = pwd;
addpath(['E:\科研\Dr\1.科研进展(clustering)\6. Plan F 张量+双曲正切+OPP\dataset']);
addpath(['E:\科研\Dr\1.科研进展(clustering)\6. Plan F 张量+双曲正切+OPP\funs']);
addpath(['E:\科研\Dr\1.科研进展(clustering)\6. Plan F 张量+双曲正切+OPP\AAAI\function']);
dataname=["CiteSeer"];
%% ==================== Load Datatset and Normalization ===================
for it_name = 1:length(dataname)
    load(strcat('dataset/',dataname(it_name),'.mat'));
    %Y=y0(:,1);
    cls_num=length(unique(Y));
    gt = Y;
    nV = length(X);
    for v=1:nV
        [X{v}]=NormalizeData(X{v}');
    end
    % for v=1:nV
    %     [X{v}]=X{v}';
    % end
    %% ========================== Parameters Setting ==========================
    result=[];
    num = 0;
    max_val=0;
    record_num = 0;
    ii=0;
    %anc =[5 10 20 50 100];
    anc =[cls_num,2*cls_num,3*cls_num,4*cls_num,5*cls_num,6*cls_num,7*cls_num,8*cls_num];
    %% ============================ Optimization ==============================
    %[-1,-3,-4,2][1,-1,-1,4]
    for i = -6:1:0
        for jj = -6:1:0
            for j = -6:1:0
                for k = 1:8                             
                    alpha = 10^(i);
                    gamma = 10^(jj);
                    delta=  10^(j);
                    anchor = anc(k);
                    ii=ii+1;
                    tic;
                    %%这个是nips的
                    %[Sbar,y,E,S,A,Z,converge_Z,converge_Z_G] = Train_new2(X, cls_num, anchor,alpha,gamma,delta);
                   %[Sbar,y,U,Z,converge_Z,converge_Z_G] = Train_problem(X, cls_num, anchor,alpha,gamma,delta);
                    %[Sbar,y,U,Z,converge_Z,converge_Z_G] = Train_baseline(X, cls_num, anchor,alpha,gamma,delta);
                    %%下面的是AAAI  注意修改秩函数
                    [ZZ,Z,E,M,A,y,converge_Z,converge_Z_G] = Train_E(X, cls_num, anchor,alpha,gamma,delta); 
                    time = toc;
                    %[ii,jj,hh,kk]=findnumber(1600,-6,1,-6,1,-6,1,1,7)%%查找具体某个数据对应的参数
                    [result(ii,:)]=  Clustering8Measure(gt, y);
                    fprintf('\n alpha:%.6f gamma:%.6f delta:%.6f anchor:%.1f\n ACC: %.4f NMI: %.4f ARI: %.4f Time: %.4f \n',[alpha gamma delta anchor result(ii,1) result(ii,2) result(ii,7) time]);                       

%                      res=[];
%                      temp=result;
%                      res=[res;temp];
                   % [result(ii,:),time]
                    if result(ii,1) > max_val
                        max_val = result(ii,1);
                        record = [i,jj,j,k,time];%[i,jj,j,k,time];
                        record_result = result(ii,:);
                        %record_c ={converge_Z,converge_Z_G};
                        record_time = time;
                    end
                end
            end
        end
    end
%save('result_final'+dataname(it_name),'result','record','max_val','record_result','time')
end

