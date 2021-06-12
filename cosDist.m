function [t]=cosDist(x,y)

len=length(x);
s1=0;
s2=0;
s3=0;
for k=1:len
        s1=s1+(x(k)*y(k));
        s2=s2+x(k)^2;
        s3=s3+y(k)^2;
                              
end

t=s1/(sqrt(s2)*sqrt(s3));  