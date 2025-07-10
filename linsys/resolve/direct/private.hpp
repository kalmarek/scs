#ifndef PRIV_H_GUARD
#define PRIV_H_GUARD

#include "csparse.h"
#include "linsys.h"
#include <resolve/matrix/Csr.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/SystemSolver.hpp>

#ifdef __cplusplus
extern "C"
{
#endif

  struct SCS_LIN_SYS_WORK
  {
    // Host:
    ScsMatrix *kkt; /* Upper triangular KKT matrix (in CSR format) */
    scs_int n;      /* number of QP variables */
    scs_int m;      /* number of QP constraints */

    /* These are required for matrix updates */
    scs_int *diag_r_idxs; /* indices where R appears */
    scs_float *diag_p;    /* Diagonal values of P */

    ReSolve::LinAlgWorkspaceHIP *workspace_hip; /* ReSolve workspace */
    ReSolve::SystemSolver *solver; /* ReSolve system solver */
    ReSolve::matrix::Csr *mat_A; /* ReSolve matrix A */
    ReSolve::vector::Vector *vec_x; /* ReSolve vector x */
    ReSolve::vector::Vector *vec_rhs; /* ReSolve vector rhs */
  };

#ifdef __cplusplus
}
#endif

#endif
