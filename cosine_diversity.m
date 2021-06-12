clc
clear

%Classifiers predictions
%SetI
% disp('Set I');
% c1=[1 1 2 1];
% c2=[1 2 2 1];
% c3=[2 1 1 2];
% c4=[1 1 1 2];
% c5=[1 2 2 2];

%SetII
% disp('Set II');
% c1=[1 1 2 2];
% c2=[2 1 1 2];
% c3=[1 1 1 2];
% c4=[2 1 2 2];
% c5=[1 2 2 2];

%SetIII
% disp('Set III');
% c1=[2 1 1 1];
% c2=[2 2 2 1];
% c3=[1 2 2 2];
% c4=[2 2 2 1];
% c5=[1 1 2 1];

%SetIV
disp('Set IV');
c1=[1 1 1 1];
c2=[2 1 2 1];
c3=[2 1 2 1];
c4=[2 2 1 1];
c5=[1 2 1 1];

%SetV
% disp('Set V');
% c1=[1 2 1 1];
% c2=[1 2 2 1];
% c3=[2 2 1 1];
% c4=[2 1 2 1];
% c5=[2 1 1 1];

%Actual Class Labels
A=[1 1 2 1];

disp('Unique classes (binary)');
uA=unique(A)  %unique classes (binary)
nA=length(uA); 
% sA=[];
% for i=1:nA
%    sA(i)=length(find(A==uA(i)));
% end
%actual numbers of decision classes
%sA

C=[c1' c2' c3' c4' c5'];

disp('.....C1   C2    C3    C4    C5.....A');
[C A']
%%
%voting
predict1=C;
predict1=(mean(predict1,2)>1.5)+1;
[c,order]=confusionmat(A', predict1);
acc1=sum(diag(c)/sum(c(:)));
fprintf('Accuracy by normal means (ensemble):%.2f\n\n',acc1);
%%
[m,n]=size(C);
cM=zeros(n,n);

for i=1:n
for j=1:n
cM(i,j)=abs(1-cosDist(C(:,i),C(:,j)));     %cosine gives similarity i.e (-1)
end
end

disp('Cosine Diversity Matrix (symmetric)');
cM

disp('Mean of Cosine Diversity');
avg_cosDiv=0;
for i=1:n
   for j=1:n
      if i > j 
      avg_cosDiv=avg_cosDiv+cM(i,j);
      end
   end
end
val_enty=((n*n-n)/2);
avg_cosDiv=avg_cosDiv/val_enty
%%
nClf=int16(n/2);  % new reduced number of classifiers in ensemble
disp('Top 3 cosine diversity values');

sortM=[];
k=1;
for i=1:n
   for j=1:n
      sortM(k)=cM(i,j);
      k=k+1;
   end
end
top=sort(unique(sortM),'descend') %top 3 cosine diversity values
top=top(1:nClf)
sM=zeros(1,n);

for i=1:n
   for j=1:n
   for l=1:nClf    
    if cM(i,j) == top(l)
    sM(i)=sM(i)+cM(i,j);
    end
   end
   end
end

%Sum of diversities for all classifiers
disp('Sum of diversities for all classifiers');
sM=sM'
%%
[~,idx]=sort(sM,'descend');
disp('New set of classifiers in ensemble after prruning (Top Classifiers)');
tIdx=idx(1:nClf)  %new set of classifiers in ensemble after prruning

newC=[];
for i=1:nClf
newC=[newC C(:,tIdx(i))]; 
end

fprintf('Predictions of %d best Classifiers',nClf);

[newC A']
%%
%voting after pruning
%voting
predict2=newC;
predict2=(mean(predict2,2)>1.5)+1;
[c,order]=confusionmat(A', predict2);
acc2=sum(diag(c)/sum(c(:)));
fprintf('Accuracy after pruning:%.2f\n',acc2);

%improvement in accuracy by cosine_diversity
disp('Improvement over normal means');
accImprov=acc2-acc1
%%
%only for binary classes
if acc2 <= acc1
    for i=1:m
      if nA==2
        if all(newC(i,:)==uA(1)) 
        predict2(i)=uA(end);
        elseif all(newC(i,:)==uA(end)) 
        predict2(i)=uA(1);
        end
      end
    end

%%
predict3=predict2;   %after mutation
[c,order]=confusionmat(A', predict3);
acc3=sum(diag(c)/sum(c(:)));
fprintf('Accuracy after pruning and mutation:%.2f\n',acc3);

%improvement in accuracy by cosine_diversity
disp('Improvement over pruning/normal');
accImprov=acc3-acc2
end