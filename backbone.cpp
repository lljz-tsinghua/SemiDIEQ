#include <cmath>
#include <random>
const int maxd=10;
const long double logeps=1e-13;
struct comp{long double x,y;};
inline comp operator+(const comp &lhs,const comp &rhs){return (comp){lhs.x+rhs.x,lhs.y+rhs.y};}
inline comp operator-(const comp &lhs,const comp &rhs){return (comp){lhs.x-rhs.x,lhs.y-rhs.y};}
inline comp operator*(const comp &lhs,const comp &rhs){return (comp){lhs.x*rhs.x-lhs.y*rhs.y,lhs.x*rhs.y+lhs.y*rhs.x};}
inline comp operator*(long double lhs,const comp &rhs){return (comp){lhs*rhs.x,lhs*rhs.y};}
inline comp operator*(const comp &lhs,long double rhs){return (comp){lhs.x*rhs,lhs.y*rhs};}
inline comp conjugate(const comp &z){return (comp){z.x,-z.y};}
inline long double norm2(const comp &z){return z.x*z.x+z.y*z.y;}
inline long double norm(const comp &z){return sqrtl(norm(z));}
inline comp operator/(const comp &lhs,long double rhs){return (comp){lhs.x/rhs,lhs.y/rhs};}
inline comp operator/(const comp &lhs,const comp &rhs){return lhs*conjugate(rhs)/norm2(rhs);}
struct mat
{
	int m,n;
	comp a[maxd][maxd];
};
mat operator+(mat lhs,mat rhs)
{
	mat ret;
	ret.m=lhs.m,ret.n=lhs.n;
	for(int i=0;i<ret.m;i++)
		for(int j=0;j<ret.n;j++)
			ret.a[i][j]=lhs.a[i][j]+rhs.a[i][j];
	return ret;
}
mat operator-(mat lhs,mat rhs)
{
	mat ret;
	ret.m=lhs.m,ret.n=lhs.n;
	for(int i=0;i<ret.m;i++)
		for(int j=0;j<ret.n;j++)
			ret.a[i][j]=lhs.a[i][j]-rhs.a[i][j];
	return ret;
}
mat operator*(long double lhs,mat rhs)
{
	mat ret;
	ret.m=rhs.m,ret.n=rhs.n;
	for(int i=0;i<ret.m;i++)
		for(int j=0;j<ret.n;j++)
			ret.a[i][j]=rhs.a[i][j]*lhs;
	return ret;
}
mat operator*(comp lhs,mat rhs)
{
	mat ret;
	ret.m=rhs.m,ret.n=rhs.n;
	for(int i=0;i<ret.m;i++)
		for(int j=0;j<ret.n;j++)
			ret.a[i][j]=rhs.a[i][j]*lhs;
	return ret;
}
mat operator*(mat lhs,mat rhs)
{
	mat ret;
	ret.m=lhs.m,ret.n=rhs.n;
	for(int i=0;i<ret.m;i++)
		for(int j=0;j<ret.n;j++)
			ret.a[i][j]=(comp){0,0};
	for(int i=0;i<ret.m;i++)
		for(int k=0;k<lhs.n;k++)
			for(int j=0;j<ret.n;j++)
				ret.a[i][j]=ret.a[i][j]+lhs.a[i][k]*rhs.a[k][j];
	return ret;
}
mat tensor(mat lhs,mat rhs)
{
	mat ret;
	ret.m=lhs.m*rhs.m,ret.n=lhs.n*rhs.n;
	for(int i1=0;i1<lhs.m;i1++)
		for(int j1=0;j1<lhs.n;j1++)
			for(int i2=0;i2<rhs.m;i2++)
				for(int j2=0;j2<rhs.n;j2++)
					ret.a[i1*rhs.m+i2][j1*rhs.n+j2]=lhs.a[i1][j1]*rhs.a[i2][j2];
	return ret;
}
mat transpose(mat x)
{
	mat ret;
	ret.m=x.n,ret.n=x.m;
	for(int i=0;i<ret.m;i++)
		for(int j=0;j<ret.n;j++)
			ret.a[i][j]=x.a[j][i];
	return ret;
}
mat adjoint(mat x)
{
	mat ret;
	ret.m=x.n,ret.n=x.m;
	for(int i=0;i<ret.m;i++)
		for(int j=0;j<ret.n;j++)
			ret.a[i][j]=conjugate(x.a[j][i]);
	return ret;
}
comp trace(mat x)
{
	comp ret=(comp){0,0};
	for(int i=0;i<x.m;i++)
		ret=ret+x.a[i][i];
	return ret;
}
mat random_matrix(mat x)
{
	mat ret;
	ret.m=x.m,ret.n=x.n;
	for(int i=0;i<ret.m;i++)
		for(int j=0;j<ret.n;j++)
			ret.a[i][j]=(comp){2.0*rand()/RAND_MAX-1.0,2.0*rand()/RAND_MAX-1.0};
	return ret;
}
mat cholesky_factorization(mat x)
{
	//Only works for positive matrices
	mat ret,res;
	ret.m=ret.n=res.m=res.n=x.m;
	for(int i=0;i<x.m;i++)
		for(int j=0;j<x.m;j++)
			res.a[i][j]=x.a[i][j],ret.a[i][j]=(comp){0,0};
	for(int i=0;i<x.m;i++)
	{
		ret.a[i][i]=(comp){sqrtl(res.a[i][i].x),0};
		for(int j=i+1;j<x.m;j++)
			ret.a[i][j]=res.a[i][j]/ret.a[i][i];
		for(int i2=i;i2<x.m;i2++)
			for(int j2=i;j2<x.m;j2++)
				res.a[i2][j2]=res.a[i2][j2]-ret.a[i][j2]*conjugate(ret.a[i][i2]);
	}
	return ret;
}
mat inverse_upper_triangular(mat x)
{
	mat ret,res;
	ret.m=ret.n=res.m=res.n=x.m;
	for(int i=0;i<x.m;i++)
		for(int j=0;j<x.m;j++)
			res.a[i][j]=x.a[i][j],ret.a[i][j]=(comp){1.0L*(i==j),0};
	for(int i=0;i<x.m;i++)
	{
		comp r=res.a[i][i];
		ret.a[i][i]=ret.a[i][i]/r;
		for(int j=i;j<x.m;j++)
			res.a[i][j]=res.a[i][j]/r;
	}
	for(int i=x.m-1;~i;i--)
	{
		for(int j=i+1;j<x.m;j++)
		{
			comp r=res.a[i][j];
			//ret.a[i][j]=(-1)*r;
			for(int k=j;k<x.m;k++)
			{
				res.a[i][k]=res.a[i][k]-r*res.a[j][k];
				ret.a[i][k]=ret.a[i][k]-r*ret.a[j][k];
			}
		}
	}
	return ret;
}
mat mat_gen_zero(int m,int n)
{
	mat ret;
	ret.m=m,ret.n=n;
	for(int i=0;i<m;i++)
		for(int j=0;j<n;j++)
			ret.a[i][j]=(comp){0,0};
	return ret;
}
mat mat_gen_id(int m)
{
	mat ret;
	ret.m=ret.n=m;
	for(int i=0;i<m;i++)
		for(int j=0;j<m;j++)
			ret.a[i][j]=(comp){1.0L*(i==j),0};
	return ret;
}
long double matrix_norm2(mat x)
{
	long double ret=0.0L;
	for(int i=0;i<x.m;i++)
		for(int j=0;j<x.n;j++)
			ret+=norm2(x.a[i][j]);
	return ret;
}
void copy(mat &lhs,mat rhs)
{
	lhs.m=rhs.m,lhs.n=rhs.n;
	memcpy(lhs.a,rhs.a,sizeof(rhs.a));
}
void dump_matrix(mat x)
{
	for(int i=0;i<x.m;i++)
		for(int j=0;j<x.n;j++)
			printf("(%.10Lf,%.10Lf)%c",x.a[i][j].x,x.a[i][j].y," \n"[j==x.n-1]);
}
long double H_entropy(long double x)
{
	long double l1,l2;
	l1=(fabsl(x)<logeps)?0.0:log2l(x);
	l2=(fabsl(x-1.0L)<logeps)?0.0:log2l(1.0L-x);
	return -x*l1-(1.0L-x)*l2;
}
long double purity_to_entropy_min(long double purity,long double d)
{
	if(fabsl(purity-1.0L)<logeps)
		return 0.0L;
	long double k=ceill(1.0L/purity);
	long double a=1.0L/k-sqrtl((1.0L-1.0L/k)*(purity-1.0L/k));
	return -(1.0L-a)*log2l((1.0L-a)/(k-1.0L))-a*log2l(a);
}
long double purity_to_entropy_max(long double purity,long double d)
{
	long double x1=1.0L/d+sqrtl((d-1.0L)/d*(purity-1.0L/d));
	return -(1.0L-x1)*log2l((1.0L-x1)/(d-1.0L))-x1*log2l(x1);
}
long double purity_to_entropy_max_incorrect(long double purity,long double d)
{
	long double x1=1.0L/d+sqrtl((d-1.0L)/d*(purity-1.0L/d));
	return -(d-1.0L)*(1.0L-x1)/d*log2l((1.0L-x1)/d)-x1*log2l(x1);
}
long double pc_to_purity(long double pc,long double d)
{
	if(pc<1.0L/d)
		return 1.0L/d;
	else
		return pc*pc+(1.0L-pc)*(1.0L-pc)/(d-1.0L);
}

