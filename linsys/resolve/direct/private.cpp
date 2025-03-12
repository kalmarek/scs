#include "private.hpp"

#include <iostream>

#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/Csc.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/LinSolverDirectKLU.hpp>
#include <resolve/LinSolverDirectKLU.hpp>
#include <resolve/LinSolverDirectRocSolverRf.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>

#ifdef __cplusplus
extern "C"
{
#endif

  const char *scs_get_lin_sys_method()
  {
    return "hip-ReSolve-direct";
  }

  void scs_free_lin_sys_work(ScsLinSysWork *p)
  {
    if (p == NULL)
      return;

    // Free memory owned by ReSolve
    delete p->mat_A;
    delete p->Rf;
    delete p->vec_x;
    delete p->vec_rhs;

    // Free the matrix kkt data
    if (p->kkt)
      SCS(cs_spfree)
    (p->kkt);

    // Free host-side arrays used for updates
    if (p->diag_r_idxs)
      scs_free(p->diag_r_idxs);
    if (p->diag_p)
      scs_free(p->diag_p);

    // Finally, free the work struct itself
    scs_free(p);
  }

  scs_int __initialize_work(ScsLinSysWork *work)
  {
    // initialize ReSolve members of work:

    int nnz = work->kkt->p[work->kkt->n]; // The last element of A->p gives the number of non-zeros
    work->mat_A = new ReSolve::matrix::Csr(work->kkt->n, work->kkt->m, nnz);
    work->mat_A->setMatrixData(work->kkt->p, work->kkt->i, work->kkt->x, ReSolve::memory::HOST);

    // mat_A is CSC, symmetric, non-expanded
    work->mat_A->setSymmetric(true);
    work->mat_A->setExpanded(false);
    work->mat_A->syncData(ReSolve::memory::DEVICE);

    work->vec_x = new ReSolve::vector::Vector(work->mat_A->getNumRows());
    work->vec_x->setToConst(1.0, ReSolve::memory::HOST);
    work->vec_rhs = new ReSolve::vector::Vector(work->mat_A->getNumRows());
    work->vec_rhs->setToConst(1.0, ReSolve::memory::HOST);

    ReSolve::LinAlgWorkspaceHIP *workspace_HIP = new ReSolve::LinAlgWorkspaceHIP;
    workspace_HIP->initializeHandles();
    work->Rf = new ReSolve::LinSolverDirectRocSolverRf(workspace_HIP);

    // we start with KLU to setup work->Rf in a moment
    ReSolve::LinSolverDirectKLU *KLU = new ReSolve::LinSolverDirectKLU;
    scs_int status;
    status = KLU->setup(work->mat_A);
    std::cout << "KLU analysis status: " << status << std::endl;
    status = KLU->factorize();
    std::cout << "KLU factorization status: " << status << std::endl;
    status = KLU->solve(work->vec_rhs, work->vec_x);
    std::cout << "KLU solve status: " << status << std::endl;

    ReSolve::matrix::Csc *L = (ReSolve::matrix::Csc *)KLU->getLFactor();
    ReSolve::matrix::Csc *U = (ReSolve::matrix::Csc *)KLU->getUFactor();
    ReSolve::index_type *P = KLU->getPOrdering();
    ReSolve::index_type *Q = KLU->getQOrdering();

    // finally let's setup work->Rf and prepare the factorisation!
    status = work->Rf->setup(work->mat_A, L, U, P, Q, work->vec_rhs);
    std::cout << "rocsolver rf refactorization status: " << status << std::endl;
    status = work->Rf->refactorize();
    return status;
  }

  ScsLinSysWork *scs_init_lin_sys_work(const ScsMatrix *A, const ScsMatrix *P,
                                       const scs_float *diag_r)
  {
    // ScsLinSysWork *p = scs_calloc(1, sizeof(ScsLinSysWork));
    ScsLinSysWork *p = new ScsLinSysWork();

    p->n = A->n;
    p->m = A->m;
    scs_int n_plus_m = p->n + p->m;

    p->diag_r_idxs = (scs_int *)scs_calloc(n_plus_m, sizeof(scs_int));
    p->diag_p = (scs_float *)scs_calloc(p->n, sizeof(scs_float));

    // p->kkt is CSC in lower triangular form; this is equivalen to upper CSR
    p->kkt = SCS(form_kkt)(A, P, p->diag_p, diag_r, p->diag_r_idxs, 0);
    if (!(p->kkt))
    {
      scs_printf("Error in forming KKT matrix");
      scs_free_lin_sys_work(p);
      return SCS_NULL;
    }

    int status;
    status = __initialize_work(p);

    if (status == 0)
    {
      return p;
    }
    else
    {
      scs_printf("error in factorisation: %d", (int)status);
      scs_free_lin_sys_work(p);
      return SCS_NULL;
    }
  }

  /* Returns solution to linear system Ax = b with solution stored in b */
  scs_int scs_solve_lin_sys(ScsLinSysWork *p, scs_float *b, const scs_float *ws,
                            scs_float tol)
  {
    // TODO: tol is ignored for now
    if (p == NULL || b == NULL || ws == NULL)
    {
      return -1; // Error: invalid input
    }

    // copies data to device
    p->vec_rhs->update(b, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
    p->vec_x->update(const_cast<ReSolve::real_type *>(ws), ReSolve::memory::HOST, ReSolve::memory::DEVICE);

    int status;
    status = p->Rf->solve(p->vec_rhs, p->vec_x);
    std::cout << "rocsolver rf solve status: " << status << std::endl;

    // Copy the solution back to the host
    p->vec_x->deepCopyVectorData(b, ReSolve::memory::HOST);

    return (scs_int)status;
  }

  /* Update factorization when R changes */
  void scs_update_lin_sys_diag_r(ScsLinSysWork *p, const scs_float *diag_r)
  {
    scs_int i;

    for (i = 0; i < p->n; ++i)
    {
      /* top left is R_x + P, bottom right is -R_y */
      p->kkt->x[p->diag_r_idxs[i]] = p->diag_p[i] + diag_r[i];
    }
    for (i = p->n; i < p->n + p->m; ++i)
    {
      /* top left is R_x + P, bottom right is -R_y */
      p->kkt->x[p->diag_r_idxs[i]] = -diag_r[i];
    }

    p->mat_A->updateValues(p->kkt->x, ReSolve::memory::HOST, ReSolve::memory::DEVICE);

    int status;
    status = p->Rf->refactorize();
    if (status != 0)
    {
      scs_printf("Error in Re-factorization when updating: %d.\n", (int)status);
      scs_free_lin_sys_work(p);
    }
  }

#ifdef __cplusplus
}
#endif
