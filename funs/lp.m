function y=lp(x,p)
  y=2.*x*(1-p)^(1/(2-p))+x.*p.*(2.*x.*(1-p)^((p-1)/(2-p)));
end