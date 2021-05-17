%% master.m // Version 27.1.2017 Clemens Dransfeld
% creates a line plot with relative concentrations
% assumes Raman spectrum are at equal spacing along a line
% requires importfile.m natsortfiles.m ramanrange.m
% only txt files with Raman spectra should be in directory
clear all
%
FF = dir('*.txt'); % reads all txt files into a struct array
FF = struct2cell(FF)% convert to cell array
FF = FF(1,1:end) % only keep first row with filenames
FF = natsortfiles(FF) % naturally solve filename order
%
x_start = -70  % start corrdinate from first file (microns)
x_inc = 0.4       % increment (microns)
for ii = 1:length(FF)
filename = FF(1,ii) % call filename cell
filestr = filename{1} % convert cell to str
[shift,amp] = importfile(filestr); % import file Ramanshift vs amplitude
x_coord(ii) = (x_start - x_inc)+ (ii*x_inc); % adds coordinate entry to array
[maxval,minval] =ramanrange ('EPO');  % 'PEI' or 'EPO' reads in shift range for desired peak
s_shift= shift(shift<maxval & shift>minval); % select peak range data
s_amp= amp(shift<maxval & shift>minval); % select peak range data
amp_epo(ii) = max (s_amp); % write max epo peakvalue to array
[maxval,minval] =ramanrange ('PEI');  % 'PEI' or 'EPO'
s_shift= shift(shift<maxval & shift>minval); % select peak range data 
s_amp= amp(shift<maxval & shift>minval); % select peak range data
amp_pei(ii) = max (s_amp); % write max pei peak value to array
end;

% remove data between xcoord 33 to 36 due to sample contamination
amp_pei(x_coord>49 & x_coord<56) = [];
amp_epo(x_coord>49 & x_coord<56) = [];
x_coord(x_coord>49 & x_coord<56) = [];


%refined max min values in PEI range by averaging in platau range
max_amp_pei = mean(amp_pei(x_coord>-70 & x_coord<-57))
min_amp_epo = mean(amp_epo(x_coord>-70 & x_coord<-67.6))

min_amp_pei = mean(amp_pei(x_coord>60 & x_coord<70))
max_amp_epo = mean(amp_epo(x_coord>60 & x_coord<70))

relamp_pei = (amp_pei - min_amp_pei)./(max_amp_pei - min_amp_pei);
relamp_epo = (amp_epo - min_amp_epo)./(max_amp_epo - min_amp_epo);

%default procedure
% relamp_epo = (amp_epo-min(amp_epo))./(max(amp_epo)-min(amp_epo));
% relamp_pei = (amp_pei-min(amp_pei))./(max(amp_pei)-min(amp_pei));

% Smooth the data using the rloess method with a specific span in % (0.1 = 10%)
span = 0.03;
relamp_epo_smooth = smooth(x_coord,relamp_epo,span,'rloess');
relamp_pei_smooth = smooth(x_coord,relamp_pei,span,'rloess');

% Plot the gradient of the Epoxy curve
dCdx = gradient(relamp_epo_smooth(:));
% ./gradient(x_coord(:));

figure (1);
grid off;
title ('Isothermal 120°C');
xlabel ('distance (microns)');
ylabel ('peak amplitude normalized concentration');
hold on;
plot (x_coord, relamp_pei);
legend ('% Epoxy','% PEI','location','best');
print('Isothermal 120°C','-dpng','-r400');

figure (2);
grid off;
title ('Isothermal 120°C');
xlabel ('distance (microns)');
ylabel ('peak amplitude normalized concentration');
hold on;
plot (x_coord, relamp_pei_smooth);
legend ('% Epoxy','% PEI','location','best');
print('Isothermal 120°C - Smooth curves','-dpng','-r400');

figure (3);
title ('Isothermal 120°C - Gradient for Epoxy curve');
xlabel ('distance (microns)');
ylabel ('derivative of normalized concentration dC/dx');
hold on;
legend ('% Epoxy','location','best');
print('Isothermal 120°C - Gradient concentration','-dpng','-r400');
