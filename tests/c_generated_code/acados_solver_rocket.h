/*
 * Copyright (c) The acados authors.
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */

#ifndef ACADOS_SOLVER_rocket_H_
#define ACADOS_SOLVER_rocket_H_

#include "acados/utils/types.h"

#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

#define ROCKET_NX     14
#define ROCKET_NZ     0
#define ROCKET_NU     3
#define ROCKET_NP     0
#define ROCKET_NP_GLOBAL     0
#define ROCKET_NBX    0
#define ROCKET_NBX0   14
#define ROCKET_NBU    0
#define ROCKET_NSBX   0
#define ROCKET_NSBU   0
#define ROCKET_NSH    0
#define ROCKET_NSH0   0
#define ROCKET_NSG    0
#define ROCKET_NSPHI  0
#define ROCKET_NSHN   0
#define ROCKET_NSGN   0
#define ROCKET_NSPHIN 0
#define ROCKET_NSPHI0 0
#define ROCKET_NSBXN  0
#define ROCKET_NS     0
#define ROCKET_NS0    0
#define ROCKET_NSN    0
#define ROCKET_NG     0
#define ROCKET_NBXN   0
#define ROCKET_NGN    0
#define ROCKET_NY0    17
#define ROCKET_NY     17
#define ROCKET_NYN    14
#define ROCKET_N      100
#define ROCKET_NH     5
#define ROCKET_NHN    0
#define ROCKET_NH0    0
#define ROCKET_NPHI0  0
#define ROCKET_NPHI   0
#define ROCKET_NPHIN  0
#define ROCKET_NR     0

#ifdef __cplusplus
extern "C" {
#endif


// ** capsule for solver data **
typedef struct rocket_solver_capsule
{
    // acados objects
    ocp_nlp_in *nlp_in;
    ocp_nlp_out *nlp_out;
    ocp_nlp_out *sens_out;
    ocp_nlp_solver *nlp_solver;
    void *nlp_opts;
    ocp_nlp_plan_t *nlp_solver_plan;
    ocp_nlp_config *nlp_config;
    ocp_nlp_dims *nlp_dims;

    // number of expected runtime parameters
    unsigned int nlp_np;

    /* external functions */

    // dynamics

    external_function_external_param_casadi *expl_vde_forw;
    external_function_external_param_casadi *expl_ode_fun;
    external_function_external_param_casadi *expl_vde_adj;

    external_function_external_param_casadi *expl_ode_hess;



    // cost






    // constraints
    external_function_external_param_casadi *nl_constr_h_fun_jac;
    external_function_external_param_casadi *nl_constr_h_fun;
    external_function_external_param_casadi *nl_constr_h_fun_jac_hess;









} rocket_solver_capsule;

ACADOS_SYMBOL_EXPORT rocket_solver_capsule * rocket_acados_create_capsule(void);
ACADOS_SYMBOL_EXPORT int rocket_acados_free_capsule(rocket_solver_capsule *capsule);

ACADOS_SYMBOL_EXPORT int rocket_acados_create(rocket_solver_capsule * capsule);

ACADOS_SYMBOL_EXPORT int rocket_acados_reset(rocket_solver_capsule* capsule, int reset_qp_solver_mem);

/**
 * Generic version of rocket_acados_create which allows to use a different number of shooting intervals than
 * the number used for code generation. If new_time_steps=NULL and n_time_steps matches the number used for code
 * generation, the time-steps from code generation is used.
 */
ACADOS_SYMBOL_EXPORT int rocket_acados_create_with_discretization(rocket_solver_capsule * capsule, int n_time_steps, double* new_time_steps);
/**
 * Update the time step vector. Number N must be identical to the currently set number of shooting nodes in the
 * nlp_solver_plan. Returns 0 if no error occurred and a otherwise a value other than 0.
 */
ACADOS_SYMBOL_EXPORT int rocket_acados_update_time_steps(rocket_solver_capsule * capsule, int N, double* new_time_steps);
/**
 * This function is used for updating an already initialized solver with a different number of qp_cond_N.
 */
ACADOS_SYMBOL_EXPORT int rocket_acados_update_qp_solver_cond_N(rocket_solver_capsule * capsule, int qp_solver_cond_N);
ACADOS_SYMBOL_EXPORT int rocket_acados_update_params(rocket_solver_capsule * capsule, int stage, double *value, int np);
ACADOS_SYMBOL_EXPORT int rocket_acados_update_params_sparse(rocket_solver_capsule * capsule, int stage, int *idx, double *p, int n_update);
ACADOS_SYMBOL_EXPORT int rocket_acados_set_p_global_and_precompute_dependencies(rocket_solver_capsule* capsule, double* data, int data_len);

ACADOS_SYMBOL_EXPORT int rocket_acados_solve(rocket_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT int rocket_acados_setup_qp_matrices_and_factorize(rocket_solver_capsule* capsule);



ACADOS_SYMBOL_EXPORT int rocket_acados_free(rocket_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void rocket_acados_print_stats(rocket_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT int rocket_acados_custom_update(rocket_solver_capsule* capsule, double* data, int data_len);


ACADOS_SYMBOL_EXPORT ocp_nlp_in *rocket_acados_get_nlp_in(rocket_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_out *rocket_acados_get_nlp_out(rocket_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_out *rocket_acados_get_sens_out(rocket_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_solver *rocket_acados_get_nlp_solver(rocket_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_config *rocket_acados_get_nlp_config(rocket_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void *rocket_acados_get_nlp_opts(rocket_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_dims *rocket_acados_get_nlp_dims(rocket_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_plan_t *rocket_acados_get_nlp_plan(rocket_solver_capsule * capsule);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_SOLVER_rocket_H_
