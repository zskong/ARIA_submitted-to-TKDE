function y = rank_fun_derivative(x,delta)

 %y  = delta*exp(delta^2)./((x+delta).^2+eps);
%y = delta./(delta+x).^2;
%  yyy=delta*exp(-x.^2);
%   yy = yyy.*(1+2*delta.*x+2*(x.^2));
%  y =yy./((x+delta).^2+eps);


%yyy= (4*delta.*x).*(exp(-delta*x.^2));
%yy = ((exp(-delta*x.^2) + 1).^2+eps);

%%AAAI
yyy=(2*(delta).*exp(-(delta).*x));
yy=(exp(-(delta).*x) + 1).^2;
y=yyy./yy;


%NIPS
% yyy=delta*exp(-x.^2);
%    yy = yyy.*(1+2*delta.*x+2*(x.^2));
%   y =yy./((x+delta).^2+eps);
end
