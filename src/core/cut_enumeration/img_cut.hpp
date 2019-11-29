/* also: Advanced Logic Synthesis and Optimization tool
 * Copyright (C) 2019- Ningbo University, Ningbo, China 
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

/*!
  \file  img_cut.hpp
  \brief Cut enumeration using the exact_img as the cost function 

  \author Zhufei Chu
*/

#pragma once

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>

#include <mockturtle/mockturtle.hpp>

namespace mockturtle
{

//std::string npn3_s = "";

struct cut_enumeration_img_cut
{
  uint32_t delay{0};
  float flow{0};
  float cost{0};
};

template<bool ComputeTruth>
bool operator<( cut_type<ComputeTruth, cut_enumeration_img_cut> const& c1, cut_type<ComputeTruth, cut_enumeration_img_cut> const& c2 )
{
  constexpr auto eps{0.005f};
  if ( c1->data.flow < c2->data.flow - eps )
    return true;
  if ( c1->data.flow > c2->data.flow + eps )
    return false;
  if ( c1->data.delay < c2->data.delay )
    return true;
  if ( c1->data.delay > c2->data.delay )
    return false;
  return c1.size() < c2.size();
}

template<>
struct cut_enumeration_update_cut<cut_enumeration_img_cut>
{
  template<typename Cut, typename NetworkCuts, typename Ntk>
  static void apply( Cut& cut, NetworkCuts const& cuts, Ntk const& ntk, node<Ntk> const& n )
  {
    /******************************************************************
     * Zhufei Chu
     ******************************************************************/
    //std::cout << "node: " << n << " ";
    //std::cout << kitty::to_hex( cuts.truth_table( cut ) ) << std::endl;
     
    /******************************************************************/

    uint32_t delay{0};
    float flow = cut->data.cost = cut.size() < 2 ? 0.0f : 1.0f;

    for ( auto leaf : cut )
    {
      const auto& best_leaf_cut = cuts.cuts( leaf )[0];
      delay = std::max( delay, best_leaf_cut->data.delay );
      flow += best_leaf_cut->data.flow;
    }

    cut->data.delay = 1 + delay;
    cut->data.flow = flow / ntk.fanout_size( n );

    //std::cout << "delay: " << cut->data.delay << " area_flow: " << cut->data.flow << std::endl;
  }
};

template<int MaxLeaves>
std::ostream& operator<<( std::ostream& os, cut<MaxLeaves, cut_data<false, cut_enumeration_img_cut>> const& c )
{
  os << "{ ";
  std::copy( c.begin(), c.end(), std::ostream_iterator<uint32_t>( os, " " ) );
  os << "}, D = " << std::setw( 3 ) << c->data.delay << " A = " << c->data.flow;
  return os;
}

} // namespace 
