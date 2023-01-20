clf
TT = readtimetable('timetabledata.xlsx');

% candle(TT(1:end,:),'b');
% title('Candlestick chart for TTF front month')

M = [TT.open, TT.high, TT.low, TT.close];
dates = TT.dealDate;


closeprice = M(:,4);

TF = isoutlier(closeprice,'movmean',120,'Thresholdfactor',5);
plot(dates,closeprice)
hold on
plot(dates(TF),closeprice(TF),"x")
hold off
legend("Original Data","Outlier Data")
ylabel('Close price')
xlabel('Time')

idx = find(TF);
for i = 1:length(idx)
    M(idx(i),:) = nan(1,4);
end

% DATA CLEANING
interpolated = fillmissing(M,'linear');

% open-to-close returns
% ocreturns = log(interpolated(:,4)./interpolated(:,1));
% 
% ocreturns_clean = ocreturns;
% ocreturns_clean(1:120:end,:) = [];
% 
% count=0;
% for m=1:length(ocreturns_clean)
%     if ocreturns_clean(m) == 0
%         count=count+1;
%     end
% end

% close-to-close returns
current = interpolated(2:end,:);
last = interpolated(1:end-1,:);
ccreturns = log(current(:,4)./last(:,4)); 

ccreturns_clean = ccreturns;
ccreturns_clean(120:120:end,:) = [];

dates_clean = dates;
dates_clean(1) = [];
dates_clean(120:120:end,:) = [];

ccreturns_clean(27133:27370) = [];
dates_clean(27133:27370) = [];

numberOfZeros = nnz(~ccreturns_clean);

% PLOT STABLE RESULTS
ccreturns_clean = 100*ccreturns_clean;
figure(2)
plot(dates_clean, ccreturns_clean)
ylabel('Returns')
xlabel('Time')

[acf,lags] = autocorr(ccreturns_clean);
[h,pValue,stat,cValue] = lbqtest(ccreturns_clean);



