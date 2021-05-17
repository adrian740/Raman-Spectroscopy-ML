function [maxval,minval] = ramanrange(mattype)
%
if mattype == 'PEI'
    maxval=1010;
    minval=1000;

elseif mattype == 'EPO'
    maxval=990;
    minval=980;
end

