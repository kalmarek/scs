#include "private.hpp"

#include <iostream>

#include <resolve/workspace/LinAlgWorkspace.hpp>

#ifdef __cplusplus
extern "C"
{
#endif

  const char *scs_get_lin_sys_method()
  {
    return "hip-ReSolve-KLU-rocsolverrf";
  }

  void scs_free_lin_sys_work(ScsLinSysWork *work)
  {
    if (work == NULL)
      return;

    // Free memory owned by ReSolve
    delete work->mat_A;
    delete work->workspace_hip;
    delete work->vec_x;
    delete work->vec_rhs;

    // Free the matrix kkt data
    if (work->kkt)
      SCS(cs_spfree)(work->kkt);

    // Free host-side arrays used for updates
    if (work->diag_r_idxs)
      scs_free(work->diag_r_idxs);
    if (work->diag_p)
      scs_free(work->diag_p);

    // Finally, free the work struct itself
    scs_free(work);
  }

  scs_int __initialize_work(ScsLinSysWork *work)
  {
    work->workspace_hip = new ReSolve::LinAlgWorkspaceHIP;
    work->workspace_hip->initializeHandles();

    work->solver = new ReSolve::SystemSolver(work->workspace_hip,
                                  "klu", // factorization
                                  "rocsolverrf", // refactorization
                                  "rocsolverrf", // triangular solve
                                  "none", // preconditioner (always 'none' here)
                                  "none"); // iterative refinement

    // solver.setRefinementMethod("fgmres", "cgs2");
    // solver.getIterativeSolver().setCliParam("restart", "100");
    // solver.getIterativeSolver().setMaxit(200);

    // initialize ReSolve members of work:
    int nnz = work->kkt->p[work->kkt->n]; // The last element of A->work gives the number of non-zeros
    // The work->KKT matrix is CSC (lower triangular is stored only)
    // by symmetry this is CSR (upper-triangular is stored only)
    // symmetric and not expanded
    work->mat_A = new ReSolve::matrix::Csr(
      work->kkt->n, work->kkt->m, nnz, true, false);
    // work->mat_A->setDataPointers(
    //   work->kkt->p, work->kkt->i, work->kkt->x, ReSolve::memory::HOST);

    work->mat_A->allocateMatrixData(ReSolve::memory::HOST);
    work->mat_A->copyDataFrom(
      work->kkt->p, work->kkt->i, work->kkt->x, ReSolve::memory::HOST,
      ReSolve::memory::HOST);
    work->mat_A->allocateMatrixData(ReSolve::memory::DEVICE);
    work->mat_A->syncData(ReSolve::memory::DEVICE);

    work->mat_A->print(std::cout, 1);

    work->vec_rhs = new ReSolve::vector::Vector(work->mat_A->getNumRows());
    work->vec_rhs->allocate(ReSolve::memory::DEVICE);
    work->vec_rhs->allocate(ReSolve::memory::HOST);
    work->vec_rhs->setToConst(1.0, ReSolve::memory::DEVICE);
    work->vec_rhs->syncData(ReSolve::memory::HOST);

    work->vec_x = new ReSolve::vector::Vector(work->mat_A->getNumColumns());
    work->vec_x->allocate(ReSolve::memory::DEVICE);
    work->vec_x->allocate(ReSolve::memory::HOST);
    work->vec_x->setToConst(1.0, ReSolve::memory::DEVICE);
    work->vec_x->syncData(ReSolve::memory::HOST);

    scs_int status;
    status = work->solver->setMatrix(work->mat_A);
    if (status != 0)
    {
      scs_printf("Error in ReSolve solver setMatrix: %d\n", (int)status);
      return status;
    }

    status = work->solver->analyze();
    if (status != 0)
    {
      scs_printf("Error in ReSolve solver analyze: %d\n", (int)status);
      return status;
    }
    status = work->solver->factorize();
    if (status != 0)
    {
      scs_printf("Error in ReSolve solver factorize: %d\n", (int)status);
      return status;
    }
    // Now prepare the Rf solver
    status = work->solver->refactorizationSetup();
    status = work->solver->refactorize();
    if (status != 0)
    {
      scs_printf("Error in ReSolve Re-factorization: %d\n", (int)status);
      return status;
    }
    scs_printf("ReSolve solver initialized successfully.\n");

    return status;
  }

  ScsLinSysWork *scs_init_lin_sys_work(const ScsMatrix *A, const ScsMatrix *P,
                                       const scs_float *diag_r)
  {
    // ScsLinSysWork *work = scs_calloc(1, sizeof(ScsLinSysWork));
    ScsLinSysWork *work = new ScsLinSysWork();

    work->n = A->n;
    work->m = A->m;
    scs_int n_plus_m = work->n + work->m;

    work->diag_r_idxs = (scs_int *)scs_calloc(n_plus_m, sizeof(scs_int));
    work->diag_p = (scs_float *)scs_calloc(work->n, sizeof(scs_float));

    // p->kkt is CSC in lower triangular form; this is equivalen to upper CSR
    work->kkt = SCS(form_kkt)(A, P, work->diag_p, diag_r, work->diag_r_idxs, 0);
    if (!(work->kkt))
    {
      scs_printf("Error in forming KKT matrix");
      scs_free_lin_sys_work(work);
      return SCS_NULL;
    }

    int status;
    status = __initialize_work(work);

    if (status == 0)
    {
      return work;
    }
    else
    {
      scs_printf("error in initialization: %d", (int)status);
      scs_free_lin_sys_work(work);
      return SCS_NULL;
    }
  }

  /* Returns solution to linear system Ax = b with solution stored in b */
  scs_int scs_solve_lin_sys(ScsLinSysWork *p, scs_float *b, const scs_float *ws,
                            scs_float tol)
  {
    // TODO: tol is ignored for now

    // copies data to device
    // std::cout << "scs_solve_lin_sys: copying data to device" << std::endl;
    p->vec_rhs->copyDataFrom(b, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
    // std::cout << "rhs_size: " << p->vec_rhs->getSize() << std::endl;

    // p->vec_rhs->syncData(ReSolve::memory::DEVICE);
    // std::cout << "scs_solve_lin_sys: vec_rhs copied" << std::endl;
    // std::cout << "vec_x size: " << p->vec_x->getSize() << std::endl;
    // p->vec_x->copyDataFrom(b, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
    // p->vec_x->syncData(ReSolve::memory::DEVICE);
    // std::cout << "scs_solve_lin_sys: vec_x copied" << std::endl;

    int status = p->solver->solve(p->vec_rhs, p->vec_x);
    if (status != 0) {
      scs_printf("Error in ReSolve solve: %d\n", (int)status);
      return status;
    }

    // Copy the solution back to the host
    // std::cout << "scs_solve_lin_sys: copying solution back to host" << std::endl;
    p->vec_x->copyDataTo(b, ReSolve::memory::DEVICE);
    // std::cout << "scs_solve_lin_sys: solution copied back to host" << std::endl;

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

    p->mat_A->copyValues(p->kkt->x, ReSolve::memory::HOST, ReSolve::memory::HOST);
    p->mat_A->syncData(ReSolve::memory::DEVICE);

    int status = p->solver->refactorize();
    if (status != 0)
    {
      scs_printf("Error in Re-factorization when updating: %d.\n", (int)status);
      scs_free_lin_sys_work(p);
    }
  }

#ifdef __cplusplus
}
#endif
