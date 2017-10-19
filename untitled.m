clear
M = importdata('Census_Microdata.csv');
odata = M.data;

for i=1:21
    odata(odata(:,i)==-9,:)=[];
end


odata(:,1)=[];
odata=cat(2,odata,odata(:,6));
odata=cat(2,odata,odata(:,15));
odata(:,6)=[];
odata(:,15)=[];
odata=cat(2,odata,odata(:,19));
odata(:,19)=[];


csvwrite('data_one.csv',odata);