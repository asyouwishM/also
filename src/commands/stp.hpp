/* also: Advanced Logic Synthesis and Optimization tool
 * Copyright (C) 2019- Ningbo University, Ningbo, China */

/**
 * @file stp.hpp
 *
 * @brief Semi-tensor product (stp) command
 *
 * @author Zhufei
 * @since  0.1
 */

#ifndef STP_HPP
#define STP_HPP

#include "../core/stp_compute.hpp"

namespace alice
{

  class stp_command : public command
  {
    public:
      stp_command( const environment::ptr& env ) : command( env, "Semi-tensor product" )
    {
    }

    protected:
      void execute() override
      {
        also::stp_demo();
      }
  };

  ALICE_ADD_COMMAND( stp, "Various" );
}

#endif
