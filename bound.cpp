#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include "backbone.cpp"
int d,dr;
int nx,ny,na,nb;
long double P[maxd][maxd][maxd][maxd];
long double corr_functional[maxd][maxd][maxd][maxd];
int seq1[maxd],seq2[maxd];
long double calculate_bound_A()
{
	memset(seq1,0,sizeof(seq1));
	long double ans=1.0/0.0;
	int i;
	while(1)
	{
		seq1[dr]=seq1[0];
		long double sum=0.0;
		memset(seq2,0,sizeof(seq2));
		while(1)
		{
			seq2[dr]=seq2[0];
			long double prod=1.0;
			for(int k=0;k<dr;k++)
			{
				long double minterm=1.0/0.0;
				for(i=0;i<ny;i++)
				{
					long double tmpsum=0.0;
					for(int j=0;j<nb;j++)
						tmpsum+=sqrtl(P[seq1[k]][i][seq2[k]][j]*P[seq1[k+1]][i][seq2[k+1]][j]);
					minterm=std::min(minterm,tmpsum);
				}
				prod*=minterm;
			}
			sum+=prod;
			seq2[0]++;
			for(i=0;i<dr&&seq2[i]==na;i++) seq2[i]=0,seq2[i+1]++;
			if(i==dr)
				break;
		}
		ans=std::min(ans,sum);
		seq1[0]++;
		for(i=0;i<dr&&seq1[i]==nx;i++) seq1[i]=0,seq1[i+1]++;
		if(i==dr)
			break;
	}
	return ans;
}
long double calculate_bound_B()
{
	memset(seq1,0,sizeof(seq1));
	long double ans=1.0/0.0;
	int i;
	while(1)
	{
		seq1[dr]=seq1[0];
		long double sum=0.0;
		memset(seq2,0,sizeof(seq2));
		while(1)
		{
			seq2[dr]=seq2[0];
			long double prod=1.0;
			for(int k=0;k<dr;k++)
			{
				long double minterm=1.0/0.0;
				for(i=0;i<nx;i++)
				{
					long double tmpsum=0.0;
					for(int j=0;j<na;j++)
						tmpsum+=sqrtl(P[i][seq1[k]][j][seq2[k]]*P[i][seq1[k+1]][j][seq2[k+1]]);
					minterm=std::min(minterm,tmpsum);
				}
				prod*=minterm;
			}
			sum+=prod;
			seq2[0]++;
			for(i=0;i<dr&&seq2[i]==nb;i++) seq2[i]=0,seq2[i+1]++;
			if(i==dr)
				break;
		}
		ans=std::min(ans,sum);
		seq1[0]++;
		for(i=0;i<dr&&seq1[i]==ny;i++) seq1[i]=0,seq1[i+1]++;
		if(i==dr)
			break;
	}
	return ans;
}
int main()
{
	dr=2;
	d=na=nb=3;
	nx=ny=2;
	
	FILE* fP=fopen("corr","r");
    FILE* fcorr_functional=fopen("corr_functional","r");
	FILE* fResult1=fopen("entropy_bound_plot","a");

    for(int iA=0;iA<nx;iA++)
        for(int iB=0;iB<ny;iB++)
            for(int jA=0;jA<na;jA++)
                for(int jB=0;jB<nb;jB++)
                    fscanf(fP,"%Lf",&P[iA][iB][jA][jB]);
    
    for(int iA=0;iA<nx;iA++)
        for(int iB=0;iB<ny;iB++)
            for(int jA=0;jA<na;jA++)
                for(int jB=0;jB<nb;jB++)
                    fscanf(fcorr_functional,"%Lf",&corr_functional[iA][iB][jA][jB]);

	long double Cd1=3.304951405170892;
	long double Cd2=6.2071067811865475;
	long double I=0.0L;

    for(int iA=0;iA<nx;iA++)
        for(int iB=0;iB<ny;iB++)
            for(int jA=0;jA<na;jA++)
                for(int jB=0;jB<nb;jB++)
                    I+=P[iA][iB][jA][jB]*corr_functional[iA][iB][jA][jB];
	
	long double eps1=std::max(logeps,Cd1-I),eps2=Cd1-(Cd2-I);
	long double epsratio=eps1/eps2;
	long double QBA=calculate_bound_A(),QAB=calculate_bound_B();
	long double Qmin=std::min(QBA,QAB);
	fprintf(stderr,"Q_BA: %.10Lf\n",QBA);
	fprintf(stderr,"Q_AB: %.10Lf\n",QAB);
	fprintf(stderr,"Qmin: %.10Lf\n",Qmin);
	fprintf(stderr,"Epsratio: %.10Lf\n",epsratio);
	long double entbound=purity_to_entropy_min(Qmin,0)-purity_to_entropy_max(pc_to_purity(1.0L-epsratio,d*d),d*d);
	fprintf(stderr,"Coherent Information Bound: %.10Lf\n",entbound);
	fprintf(stderr,"Bell Value: %.10Lf\n",I);
	fprintf(stderr,"========== Debug Info ==========\n");
	fprintf(stderr,"C(d,1): %.10Lf\n",Cd1);
	fprintf(stderr,"C(d,2): %.10Lf\n",Cd2);
	fprintf(stderr,"Log 2: %.10Lf\n",log2l(2.0L));
	fprintf(stderr,"Log d: %.10Lf\n",log2l(1.0L*d));
	fprintf(fResult1,"%.15Lf,%.15Lf\n",Cd1-I,entbound);
	fcloseall();
	return 0;
}

