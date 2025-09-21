function [ZZ,Z,E,M,A,labels,converge_Z,converge_Z_G] = Train_E(X, cls_num, anchor,alpha,gamma,delta)
% X is a cell data, each cell is a matrix in size of d_v *N,each column is a sample;
% cls_num is the clustering number 
% anchor is the anchor number
% alpha,gamma and delta are the parameters
%%加上E的2,1范数,加融合模块
nV = length(X);
N = size(X{1},2);
t=anchor;
nC=cls_num;

%% ============================ Initialization ============================
for k=1:nV
    Z{k} = zeros(t,N); 
    W{k} = zeros(t,N);
    J{k} = zeros(t,N);
    M{k} = zeros(t,t);
    A{k} = zeros(size(X{k},1),t);
    E{k} = zeros(size(X{k},1),N);
    Y{k} = zeros(size(X{k},1),N); %Y{2} = zeros(size(X{k},1),N);
end
ZZ = zeros(t,N);
w = zeros(t*N*nV,1);
j = zeros(t*N*nV,1);
sX = [t, N, nV];

Isconverg = 0;epson = 1e-7;
iter = 0;
mu = 10e-5; max_mu = 10e10; pho_mu = 2;
rho = 0.0001; max_rho = 10e12; pho_rho = 2;
%0.0001
converge_Z=[];
converge_Z_G=[];

while(Isconverg == 0)
%% ============================== Update Z^k=============================
%temp_E =[];
for k =1:nV
    tmp = mu*eye(t,t)+rho*eye(t,t)+2*gamma*eye(t,t);
    Z_temp = mu*A{k}'*X{k}-mu*A{k}'*E{k}+A{k}'*Y{k}+rho*J{k}-W{k}+2*gamma*M{k}*ZZ;
    Z{k} = inv(tmp)*Z_temp;
%     for ii = 1:size(X{k},2)
%         Z{k}(:,ii) = EProjSimplex(Z{k}(:,ii));
%     end
    %temp_E=[temp_E;(X{k}-A{k}*Z{k}+Y{k}/mu)];
end


  %% =========================== Update E^k, Y^k ===========================
  for i=1:nV
    P{i}=(X{i} - A{i}*Z{i}+ Y{i}/mu);
    E{i} = prox_l1(P{i},alpha/mu);

% for i=1:nV
%   [E{i}] = solve_l1l2(X{i}-A{i}*Z{i}+Y{i}/mu,alpha/mu);
% end
  % ro_b =0;
  % E{1} =  Econcat(1:size(X{1},1),:);
  % Y{1} = Y{1} + mu*(X{1}-A{1}*Z{1}-E{1});
  % ro_end = size(X{1},1);
  % for i=2:nV
  %     ro_b = ro_b + size(X{i-1},1);
  %     ro_end = ro_end + size(X{i},1);
  %     E{i} =  Econcat(ro_b+1:ro_end,:);
  %     Y{i} = Y{i} + mu*(X{i}-A{i}*Z{i}-E{i});
  % end   
%% ============================= Update J^k ==============================
                Z_tensor = cat(3, Z{:,:});%%把所有的矩阵堆叠为张量 n*n*V
                W_tensor = cat(3, W{:,:});
                z = Z_tensor(:);
                w = W_tensor(:);
                J_tensor = solve_G(Z_tensor + 1/rho*W_tensor,rho,sX,delta);
                j = J_tensor(:);
                %TNN
                % [j,objV] = wshrinkObj(Z_tensor + 1/rho*W_tensor,1/rho,sX,0,3);
                % J_tensor=reshape(j, sX);
%% ============================== Update W ===============================
        w = w + rho*(z - j);
        W_tensor = reshape(w, sX);
    for k=1:nV
        W{k} = W_tensor(:,:,k);
    end
        %% ============================== Update A{v} ===============================
   G={};
for i = 1 :nV
    G{i}=(Y{i}+mu*X{i}-mu*E{i})*Z{i}';
    [Au,ss,Av] = svd(G{i},'econ');
    A{i}=Au*Av';
end

%% ============================== Update M{v} ===============================
   MM={};
for i = 1 :nV
    MM{i}=Z{i}*ZZ';
    [Au,ss,Av] = svd(MM{i},'econ');
    M{i}=Au*Av';
end

%% ============================== Update ZZ ===============================
temp_ZZ=0;
for i = 1 :nV
    temp_ZZ=temp_ZZ+M{i}'*Z{i};
end
ZZ=temp_ZZ;
% for ii = 1:size(X{k},2)
%         ZZ(:,ii) = EProjSimplex(temp_ZZ(:,ii));
% end
% 
for i=1:nV
    Y{i} = Y{i} + mu*(X{i}-A{i}*Z{i}-E{i});
end
%% ====================== Checking Coverge Condition ======================
    max_Z=0;
    max_Z_G=0;
    Isconverg = 1;
    for k=1:nV
        if (norm(X{k}-A{k}*Z{k}-E{k},inf)>epson)
            history.norm_Z = norm(X{k}-A{k}*Z{k}-E{k},inf);
            Isconverg = 0;
            max_Z=max(max_Z,history.norm_Z );
        end
        
        J{k} = J_tensor(:,:,k);
        W_tensor = reshape(w, sX);
        W{k} = W_tensor(:,:,k);
        if (norm(Z{k}-J{k},inf)>epson)
            history.norm_Z_G = norm(Z{k}-J{k},inf);
            Isconverg = 0;
            max_Z_G=max(max_Z_G, history.norm_Z_G);
        end
    end
    converge_Z=[converge_Z max_Z];
    converge_Z_G=[converge_Z_G max_Z_G];
   
    
    if (iter>30)
        Isconverg  = 1;
    end
    iter = iter + 1;
    mu = min(mu*pho_mu, max_mu);
    rho = min(rho*pho_rho, max_rho);
end

% Sbar=[];
% for i = 1:nV
%     Sbar=cat(1,Sbar,1/sqrt(nV)*Z{i});
% end
% [U,Sig,V] = mySVD(Sbar',nC); 


% Sbar=0;
% for i = 1:nV
%     Sbar=Sbar+Z{i};
% end
% [U,Sig,V] = mySVD(Sbar',nC); 


[U,Sig,V] = mySVD(ZZ',nC); 

rand('twister',5489)
labels=litekmeans(U, nC, 'MaxIter', 100,'Replicates',10);%kmeans(U, c, 'emptyaction', 'singleton', 'replicates', 100, 'display', 'off');
end
