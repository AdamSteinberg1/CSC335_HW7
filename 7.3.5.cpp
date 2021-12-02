#include <iostream>
#include <armadillo>
using namespace std;
using namespace arma;

#define MAX_ITERATIONS 10000

//solves Ax=b
vec jacobi(mat A, vec b, double tol)
{
  int n = A.n_rows;
  vec x0(n, fill::zeros);
  for (int k=0; k<MAX_ITERATIONS; k++)
  {
    vec x(n);
    for(int i=0; i<n; i++)
    {
      double sum = 0;
      for(int j=0; j<n; j++)
      {
        if(j==i)
          continue;
        sum += A(i,j)*x0(j);
      }
      x(i) = (-sum + b(i)) / A(i,i);
    }
    if(norm(x-x0, "inf") < tol)
      return x;
    x0 = std::move(x);
  }
  cerr << MAX_ITERATIONS << " iterations exceeded" << endl;
  return vec();
}

int main()
{
  mat A = {{3,-1, 1},
           {3, 6, 2},
           {3, 3, 7}};
  vec b = {1, 0, 4};
  double tol = 1e-3;
  vec x = jacobi(A, b, tol);
  cout << "x = (";
  for(auto& num : x)
    cout << num << " ";
  cout << "\b)" << endl;
  return 0;
}
