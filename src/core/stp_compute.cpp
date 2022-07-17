/* also: Advanced Logic Synthesis and Optimization tool
 * Copyright (C) 2019- Ningbo University, Ningbo, China */

#include "stp_compute.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <vector>
using Eigen::MatrixXd;

namespace also
{

/******************************************************************************
 * Types                                                                      *
 ******************************************************************************/

/******************************************************************************
 * Private functions                                                          *
 ******************************************************************************/

/******************************************************************************
 * Public functions                                                           *
 ******************************************************************************/

Eigen::MatrixXi stp_kron_product( Eigen::MatrixXi matrix_f, Eigen::MatrixXi matrix_b )
{
  int m = matrix_f.rows();
  int n = matrix_f.cols();
  int p = matrix_b.rows();
  int q = matrix_b.cols();
  Eigen::MatrixXi dynamic_matrix( m * p, n * q );
  for ( int i = 0; i < m * p; i++ )
  {
    for ( int j = 0; j < n * q; j++ )
    {
      dynamic_matrix( i, j ) = matrix_f( i / p, j / q ) * matrix_b( i % p, j % q );
    }
  }
  return dynamic_matrix;
}

Eigen::MatrixXi stpm_basic_product( Eigen::MatrixXi matrix_f, Eigen::MatrixXi matrix_b )
{
  int z = 0;
  Eigen::MatrixXi result_matrix;
  Eigen::MatrixXi matrix_i1;
  Eigen::MatrixXi matrix_i2;
  int n_col = matrix_f.cols();
  int p_row = matrix_b.rows();
  if ( n_col % p_row == 0 )
  {
    z = n_col / p_row;
    matrix_i1 = Eigen::MatrixXi::Identity( z, z );
    result_matrix = matrix_f * stp_kron_product( matrix_b, matrix_i1 );
  }
  else if ( p_row % n_col == 0 )
  {
    z = p_row / n_col;
    matrix_i2 = Eigen::MatrixXi::Identity( z, z );
    result_matrix = stp_kron_product( matrix_f, matrix_i2 ) * matrix_b;
  }
  else
  {
    int c;
    int m = n_col, n = p_row;
    while ( p_row != 0 ) /* 余数不为0，继续相除，直到余数为0 */
    {
      c = n_col % p_row;
      n_col = p_row;
      p_row = c;
    }
    z = m * n / n_col;
    matrix_i1 = Eigen::MatrixXi::Identity( z / m, z / m );
    matrix_i2 = Eigen::MatrixXi::Identity( z / n, z / n );
    result_matrix = stp_kron_product( matrix_f, matrix_i1 ) * stp_kron_product( matrix_b, matrix_i2 );
  }
  return result_matrix;
}

Eigen::MatrixXi stpm_basic_product_boost( Eigen::MatrixXi matrix_f, Eigen::MatrixXi matrix_b )
{
  int mf_row = matrix_f.rows(); //m
  int mf_col = matrix_f.cols(); //n
  int mb_row = matrix_b.rows(); //p
  int mb_col = matrix_b.cols(); //q
  int row, col;
  Eigen::MatrixXi result_matrix;
  if ( mf_col % mb_row == 0 )
  {
    int times = mf_col / mb_row;
    row = mf_row;
    col = mf_col * mb_col / mb_row;
    Eigen::MatrixXi result_matrix( row, col );
    int num;
    for ( int i = 0; i < mf_row; i++ )
    {
      for ( int j = 0; j < mb_col; j++ )
      {
        Eigen::MatrixXi matrixm( 1, times );
        matrixm = Eigen::MatrixXi::Zero( 1, times );
        for ( int t = 0; t < mb_row; t++ )
        {
          Eigen::MatrixXi matrix1( 1, times );
          int v = 0;
          for ( v = 0; v < times; v++ )
          {
            matrix1( 0, v ) = matrix_f( i, v + t * times );
          }
          matrixm = matrixm + matrix_b( t, j ) * matrix1;
        }
        for ( int t = 0; t < times; t++ )
        {
          result_matrix( i, j * times + t ) = matrixm( 0, t );
        }
      }
    }
    return result_matrix;
  }
  else if ( mb_row % mf_col == 0 )
  {
    int times = mb_row / mf_col;
    row = mf_row * mb_row / mf_col;
    col = mb_col;
    Eigen::MatrixXi result_matrix( row, col );
    for ( int i = 0; i < mf_row; i++ )
    {
      for ( int j = 0; j < mb_col; j++ )
      {
        Eigen::MatrixXi matrixm( times, 1 );
        matrixm = Eigen::MatrixXi::Zero( times, 1 );
        for ( int t = 0; t < mf_col; t++ )
        {
          Eigen::MatrixXi matrix1( times, 1 );
          int v = 0;
          for ( v = 0; v < times; v++ )
          {
            matrix1( v, 0 ) = matrix_b( v + t * times, j );
          }
          matrixm = matrixm + matrix_f( i, t ) * matrix1;
        }
        for ( int t = 0; t < times; t++ )
        {
          result_matrix( i * times + t, j ) = matrixm( t, 0 );
        }
      }
    }
    return result_matrix;
  }
  return result_matrix;
}

Eigen::MatrixXi matrix_chain_multiplication( std::vector<Eigen::MatrixXi> matrix_chain )
{
  Eigen::MatrixXi result_matrix;
  if ( matrix_chain.size() == 1 )
  {
    return matrix_chain[0];
  }
  result_matrix = stpm_basic_product_boost( matrix_chain[0], matrix_chain[1] );
  for ( int i = 2; i < matrix_chain.size(); i++ )
  {
    result_matrix = stpm_basic_product_boost( result_matrix, matrix_chain[i] );
  }
  return result_matrix;
}

Eigen::MatrixXi swap_matrix( int m, int n )
{
  Eigen::MatrixXi swap_matrixXi( m * n, m * n );
  swap_matrixXi = Eigen::MatrixXi::Zero( m * n, m * n );
  int p, q;
  for ( int i = 0; i < m * n / 2 + 1; i++ )
  {
    p = i / m;
    q = i % m;
    int j = q * n + p;
    swap_matrixXi( i, j ) = 1;
    swap_matrixXi( m * n - 1 - i, m * n - 1 - j ) = 1;
  }
  return swap_matrixXi;
}

Eigen::MatrixXi normalize_matrix( std::vector<Eigen::MatrixXi> matrix_chain )
{
  Eigen::MatrixXi Mr( 4, 2 ); //Reduced power matrix
  Mr << 1, 0, 0, 0,
      0, 0, 0, 1;
  Eigen::MatrixXi I2( 2, 2 );
  I2 << 1, 0, 0, 1;
  Eigen::MatrixXi normal_matrix;
  int p_variable;
  int p;
  int max = 0;
  for ( int i = 0; i < matrix_chain.size(); i++ ) //the max is the number of variable
  {
    if ( matrix_chain[i]( 0, 0 ) > max )
      max = matrix_chain[i]( 0, 0 );
  }
  std::vector<int> idx( max + 1 ); //id0 is the max of idx
  p_variable = matrix_chain.size() - 1;
  int flag;
  while ( p_variable >= 0 )
  {
    int flag = 0;
    if ( matrix_chain[p_variable].rows() == 2 && matrix_chain[p_variable].cols() == 1 ) //1:find a variable
    {
      if ( idx[matrix_chain[p_variable]( 0, 0 )] == 0 )
      {
        idx[matrix_chain[p_variable]( 0, 0 )] = idx[0] + 1;
        idx[0]++;
        if ( p_variable == matrix_chain.size() - 1 ) //!
        {
          matrix_chain.pop_back();
          p_variable--;
          continue;
        }
      }
      else //idx[matrix_chain[p_variable](0,0)]!=0  false
      {
        if ( idx[matrix_chain[p_variable]( 0, 0 )] == idx[0] )
        {
          flag = 1;
        }
        else
        {
          flag = 1;
          matrix_chain.push_back( swap_matrix( 2, pow( 2, idx[0] - idx[matrix_chain[p_variable]( 0, 0 )] ) ) );
          for ( int i = 1; i <= max; i++ )
          {
            if ( idx[i] != 0 && idx[i] > idx[matrix_chain[p_variable]( 0, 0 )] )
              idx[i]--;
          }
          idx[matrix_chain[p_variable]( 0, 0 )] = idx[0];
        }
      }
      std::vector<Eigen::MatrixXi> matrix_chain1; //?
      matrix_chain1.clear();
      for ( p = p_variable + 1; p < matrix_chain.size(); p++ )
      {
        matrix_chain1.push_back( matrix_chain[p] ); //have no matrix
      }
      while ( p > p_variable + 1 )
      {
        matrix_chain.pop_back();
        p--;
      } //?
      if ( matrix_chain1.size() > 0 )
      {
        matrix_chain.push_back( matrix_chain_multiplication( matrix_chain1 ) );
      }
      if ( p_variable != matrix_chain.size() - 1 )
      {
        matrix_chain[p_variable] = stp_kron_product( I2, matrix_chain[p_variable + 1] );
        matrix_chain.pop_back();
      }
      if ( flag )
        matrix_chain.push_back( Mr );
      continue;
    }
    else
    {
      p_variable--;
    }
  }
  normal_matrix = matrix_chain_multiplication( matrix_chain );
  matrix_chain.clear();
  matrix_chain.push_back( normal_matrix );
  for ( int i = max; i > 0; i-- ) //!
  {
    matrix_chain.push_back( swap_matrix( 2, pow( 2, idx[0] - idx[i] ) ) );
    for ( int j = 1; j <= max; j++ )
    {
      if ( idx[j] != 0 && idx[j] > idx[i] )
        idx[j]--;
    }
    idx[i] = max;
  }
  normal_matrix = matrix_chain_multiplication( matrix_chain );
  return normal_matrix;
}

void test_stp_kron_product()
{
  Eigen::MatrixXi m( 2, 2 ); 
  m( 0, 0 ) = 1;            
  m( 0, 1 ) = 2;
  m( 1, 0 ) = 3;
  m( 1, 1 ) = 1;
  std::cout << "stp_kron_product front matrix:" << std::endl
            << std::endl;
  std::cout << m << std::endl
            << std::endl;
  Eigen::MatrixXi x;
  Eigen::MatrixXi n( 2, 2 ); 
  n( 0, 0 ) = 0;             
  n( 0, 1 ) = 3;
  n( 1, 0 ) = 2;
  n( 1, 1 ) = 1;
  std::cout << "stp_kron_product back matrix:" << std::endl
            << std::endl;
  std::cout << n << std::endl
            << std::endl;
  x = stp_kron_product( m, n );
  std::cout << "stp_kron_product:" << std::endl
            << std::endl;
  std::cout << x << std::endl
            << std::endl;
}

void test_stpm_basic_product() //p35 example
{
  Eigen::MatrixXi m1( 3, 6 );
  Eigen::MatrixXi m2( 2, 2 );
  Eigen::MatrixXi m3( 6, 3 );
  m1 << 1, 2, -1,
      2, 0, 1,
      2, 3, 3,
      3, 1, 1,
      1, 2, 3,
      4, 5, 6;
  m2 << 1, 2, -1, 2;
  m3 << 1, 2, -1, 2, 0, 1,
      2, 3, 3, 3, 1, 1,
      1, 2, 3, 4, 5, 6;
  std::cout << "The (left) semi-tensor product of m1 and m2:" << std::endl
            << std::endl;
  std::cout << "m1:" << std::endl
            << m1 << std::endl
            << std::endl;
  std::cout << "m2:" << std::endl
            << m2 << std::endl
            << std::endl;

  std::cout << "use stpm_basic_product:" << std::endl
            << std::endl;
  std::cout << stpm_basic_product( m1, m2 ) << std::endl
            << std::endl;
  std::cout << "use stpm_basic_product_boost:" << std::endl
            << std::endl;
  std::cout << stpm_basic_product_boost( m1, m2 ) << std::endl
            << std::endl;

  std::cout << "The (left) semi-tensor product of m2 and m3:" << std::endl
            << std::endl;
  std::cout << "m2:" << std::endl
            << m2 << std::endl
            << std::endl;
  std::cout << "m3:" << std::endl
            << m3 << std::endl
            << std::endl;
  std::cout << "use stpm_basic_product:" << std::endl
            << std::endl;
  std::cout << stpm_basic_product( m2, m3 ) << std::endl
            << std::endl;
  std::cout << "use stpm_basic_product_boost:" << std::endl
            << std::endl;
  std::cout << stpm_basic_product_boost( m2, m3 ) << std::endl
            << std::endl;
}

void test_swap_matrix()
{
  std::cout << "swap_matrix( 2, 4 ):" << std::endl
            << std::endl;
  std::cout << swap_matrix( 2, 4 ) << std::endl
            << std::endl;
  std::cout << "swap_matrix( 3, 2 ):" << std::endl
            << std::endl;
  std::cout << swap_matrix( 3, 2 ) << std::endl
            << std::endl;
  std::cout << "swap_matrix( 3, 3 ):" << std::endl
            << std::endl;
  std::cout << swap_matrix( 3, 3 ) << std::endl
            << std::endl;
}

void test_matrix_chain_multiplication()
{
  Eigen::MatrixXi A( 2, 1 );
  A << 0, 1;
  Eigen::MatrixXi B( 2, 1 );
  B << 1, 0;
  Eigen::MatrixXi C( 2, 1 );
  C << 0, 1;
  Eigen::MatrixXi Mc( 2, 4 );
  Mc << 1, 0, 0, 0, 0, 1, 1, 1;
  Eigen::MatrixXi Me( 2, 4 );
  Me << 1, 0, 0, 1, 0, 1, 1, 0;
  Eigen::MatrixXi Mn( 2, 2 );
  Mn << 0, 1, 1, 0;
  std::vector<Eigen::MatrixXi> matrix_chain;
  matrix_chain.push_back( Mc );
  matrix_chain.push_back( Mc );
  matrix_chain.push_back( Me );
  matrix_chain.push_back( A );
  matrix_chain.push_back( Mn );
  matrix_chain.push_back( B );
  matrix_chain.push_back( Me );
  matrix_chain.push_back( B );
  matrix_chain.push_back( Mn );
  matrix_chain.push_back( C );
  matrix_chain.push_back( Me );
  matrix_chain.push_back( C );
  matrix_chain.push_back( Mc );
  matrix_chain.push_back( Mn );
  matrix_chain.push_back( A );
  matrix_chain.push_back( Mn );
  matrix_chain.push_back( B );
  std::cout << "the matrix_chain from (a ↔ ¬b) ∧ (b ↔ ¬c) ∧ (c ↔ ¬a ∧ ¬b) , when a=0,b=1 and c=0 it is ture.(it is a case in book)" << std::endl
            << std::endl;
  std::cout << matrix_chain_multiplication( matrix_chain ) << std::endl;
}

void test1_normalize_matrix()
{
  Eigen::MatrixXi t( 2, 1 );
  t << 1, 0;
  Eigen::MatrixXi f( 2, 1 );
  f << 0, 1;
  Eigen::MatrixXi A( 2, 1 );
  A << 1, 1;
  Eigen::MatrixXi B( 2, 1 );
  B << 2, 2;
  Eigen::MatrixXi C( 2, 1 );
  C << 3, 3;
  Eigen::MatrixXi D( 2, 1 );
  D << 4, 4;
  Eigen::MatrixXi E( 2, 1 );
  E << 5, 5;
  Eigen::MatrixXi F( 2, 1 );
  F << 6, 6;
  Eigen::MatrixXi Mc( 2, 4 );
  Mc << 1, 0, 0, 0, 0, 1, 1, 1;
  Eigen::MatrixXi Me( 2, 4 );
  Me << 1, 0, 0, 1, 0, 1, 1, 0;
  Eigen::MatrixXi Mn( 2, 2 );
  Mn << 0, 1, 1, 0;
  std::vector<Eigen::MatrixXi> matrix_chain;
  matrix_chain.push_back( Mc );
  matrix_chain.push_back( Mc );
  matrix_chain.push_back( Me );
  matrix_chain.push_back( A );
  matrix_chain.push_back( Mn );
  matrix_chain.push_back( B );
  matrix_chain.push_back( Me );
  matrix_chain.push_back( B );
  matrix_chain.push_back( Mn );
  matrix_chain.push_back( C );
  matrix_chain.push_back( Me );
  matrix_chain.push_back( C );
  matrix_chain.push_back( Mc );
  matrix_chain.push_back( Mn );
  matrix_chain.push_back( A );
  matrix_chain.push_back( Mn );
  matrix_chain.push_back( B );
  std::cout << "the matrix_chain from (a ↔ ¬b) ∧ (b ↔ ¬c) ∧ (c ↔ ¬a ∧ ¬b)" << std::endl
            << std::endl;
  Eigen::MatrixXi matrix = normalize_matrix( matrix_chain );
  std::cout << matrix << std::endl
            << std::endl;
  matrix_chain.clear();
  matrix_chain.push_back( matrix );
  matrix_chain.push_back( f );
  matrix_chain.push_back( t );
  matrix_chain.push_back( f );
  std::cout << "a=0,b=1,c=0" << std::endl
            << std::endl;
  std::cout << matrix_chain_multiplication( matrix_chain ) << std::endl;
}

void test2_normalize_matrix()
{
  Eigen::MatrixXi t( 2, 1 );
  t << 1, 0;
  Eigen::MatrixXi f( 2, 1 );
  f << 0, 1;
  Eigen::MatrixXi A( 2, 1 );
  A << 1, 1;
  Eigen::MatrixXi B( 2, 1 );
  B << 2, 2;
  Eigen::MatrixXi C( 2, 1 );
  C << 3, 3;
  Eigen::MatrixXi D( 2, 1 );
  D << 4, 4;
  Eigen::MatrixXi E( 2, 1 );
  E << 5, 5;
  Eigen::MatrixXi F( 2, 1 );
  F << 6, 6;
  Eigen::MatrixXi Mc( 2, 4 );
  Mc << 1, 0, 0, 0, 0, 1, 1, 1;
  Eigen::MatrixXi Md( 2, 4 );
  Md << 1, 1, 1, 0, 0, 0, 0, 1;
  Eigen::MatrixXi Me( 2, 4 );
  Me << 1, 0, 0, 1, 0, 1, 1, 0;
  Eigen::MatrixXi Mn( 2, 2 );
  Mn << 0, 1, 1, 0;
  std::vector<Eigen::MatrixXi> matrix_chain;
  matrix_chain.push_back( Mc );
  matrix_chain.push_back( Mc );
  matrix_chain.push_back( Mc );
  matrix_chain.push_back( Mc );
  matrix_chain.push_back( Md );
  matrix_chain.push_back( Md );
  matrix_chain.push_back( Mn );
  matrix_chain.push_back( A );
  matrix_chain.push_back( F );
  matrix_chain.push_back( C );
  matrix_chain.push_back( Md );
  matrix_chain.push_back( B );
  matrix_chain.push_back( Mn );
  matrix_chain.push_back( C );
  matrix_chain.push_back( Md );
  matrix_chain.push_back( Mn );
  matrix_chain.push_back( D );
  matrix_chain.push_back( Mn );
  matrix_chain.push_back( E );
  matrix_chain.push_back( B );
  matrix_chain.push_back( D );
  std::cout << "the matrix_chain from a cnf file:" << std::endl
            << std::endl;
  std::cout << "p cnf 6 5" << std::endl;
  std::cout << "-1 6 3 0" << std::endl;
  std::cout << "2 -3 0" << std::endl;
  std::cout << "-4 -5 0" << std::endl;
  std::cout << "2 0" << std::endl;
  std::cout << "4 0" << std::endl
            << std::endl;
  std::cout << "There are seven solutions(we have used minisat to verify):" << std::endl;
  std::cout << "010100  010101  011100  011101  110101  111100  111101" << std::endl;
  std::cout << "we insert a wrong solution(111001) to verify UNSAT" << std::endl;
  std::vector<std::string> solutions;
  solutions.push_back( "010100" );
  solutions.push_back( "010101" );
  solutions.push_back( "011100" );
  solutions.push_back( "011101" );
  solutions.push_back( "110101" );
  solutions.push_back( "111100" );
  solutions.push_back( "111101" );
  solutions.push_back( "111001" );
  Eigen::MatrixXi matrix = normalize_matrix( matrix_chain );
  std::cout << std::endl
            << "normalize matrix:" << std::endl
            << matrix << std::endl
            << std::endl;
  std::cout << std::endl
            << "Then we assign values to all variables by some solution." << std::endl
            << std::endl;
  for ( int i = 0; i < solutions.size(); i++ )
  {
    matrix_chain.clear();
    matrix_chain.push_back( matrix );
    std::cout << ( i + 1 ) << "th solution:  ";
    for ( int j = 0; j < solutions[i].size(); j++ )
    {
      std::cout << solutions[i][j];
      if ( solutions[i][j] == '1' )
        matrix_chain.push_back( t );
      else
        matrix_chain.push_back( f );
    }
    if ( i != solutions.size() - 1 )
      std::cout << std::endl
                << "SAT verify:" << std::endl
                << matrix_chain_multiplication( matrix_chain ) << std::endl
                << std::endl;
    else
      std::cout << std::endl
                << "UNSAT verify:" << std::endl
                << matrix_chain_multiplication( matrix_chain ) << std::endl
                << std::endl;
  }
}

void test3_normalize_matrix()
{
  Eigen::MatrixXi t( 2, 1 );
  t << 1, 0;
  Eigen::MatrixXi f( 2, 1 );
  f << 0, 1;
  Eigen::MatrixXi A( 2, 1 );
  A << 1, 1;
  Eigen::MatrixXi B( 2, 1 );
  B << 2, 2;
  Eigen::MatrixXi Mc( 2, 4 );
  Mc << 1, 0, 0, 0, 0, 1, 1, 1;
  Eigen::MatrixXi Md( 2, 4 );
  Md << 1, 1, 1, 0, 0, 0, 0, 1;
  Eigen::MatrixXi Mn( 2, 2 );
  Mn << 0, 1, 1, 0;
  std::vector<Eigen::MatrixXi> matrix_chain;
  matrix_chain.push_back( Mc );
  matrix_chain.push_back( Mc );
  matrix_chain.push_back( Mc );

  matrix_chain.push_back( Md );
  matrix_chain.push_back( Mn );
  matrix_chain.push_back( A );
  matrix_chain.push_back( B );

  matrix_chain.push_back( Md );
  matrix_chain.push_back( A );
  matrix_chain.push_back( B );

  matrix_chain.push_back( Md );
  matrix_chain.push_back( Mn );
  matrix_chain.push_back( A );
  matrix_chain.push_back( Mn );
  matrix_chain.push_back( B );

  matrix_chain.push_back( Md );
  matrix_chain.push_back( A );
  matrix_chain.push_back( Mn );
  matrix_chain.push_back( B );
  std::cout << "the matrix_chain from a cnf file:" << std::endl
            << std::endl;
  std::cout << "p cnf 2 4" << std::endl;
  std::cout << "-1 2 0" << std::endl;
  std::cout << "1 2 0" << std::endl;
  std::cout << "-1 -2 0" << std::endl;
  std::cout << "1 -2 0" << std::endl
            << std::endl;
  std::cout << "There are no solutions(we have used minisat to verify):" << std::endl;
  Eigen::MatrixXi matrix = normalize_matrix( matrix_chain );
  std::cout << std::endl
            << "normalize matrix:" << std::endl
            << matrix << std::endl
            << std::endl;
}

void stp_demo()
{
  std::cout << "**********************************************test_stp_kron_product()**********************************************************" << std::endl;
  test_stp_kron_product();
  std::cout << "*********************************************test_stpm_basic_product()*********************************************************" << std::endl;
  test_stpm_basic_product();
  std::cout << "************************************************test_swap_matrix()*************************************************************" << std::endl;
  test_swap_matrix();
  std::cout << "*****************************************test_matrix_chain_multiplication()****************************************************" << std::endl;
  test_matrix_chain_multiplication();
  std::cout << "*********************************************test1_normalize_matrix()**********************************************************" << std::endl;
  test1_normalize_matrix();
  std::cout << "*********************************************test2_normalize_matrix()**********************************************************" << std::endl;
  test2_normalize_matrix();
  std::cout << "*********************************************test3_normalize_matrix()**********************************************************" << std::endl;
  test3_normalize_matrix();
}

} // namespace also
