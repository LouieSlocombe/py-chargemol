! Copyright (c) 2025 by Thomas A. Manz. Rights reserved.

 MODULE module_QDR_point_charges

 USE precise
 USE common_variable_declarations
 USE module_atomic_number_to_symbol
 USE module_total_multipole_moment_analysis
!$ USE omp_lib

 IMPLICIT NONE
 
 CONTAINS
 
 SUBROUTINE generate_QDR_point_charges()
 !--------------------------------------------------------------

 REAL(kind=dp) :: dot_prod,V1(3),V2(3),V3(3),delta_charge,bond_direction(3),weighted_bond_dir(3),temp_sum
 CHARACTER(34) :: filename
 INTEGER :: charge_fid,CG_iter,EV,SV,max_CG_iter,local_clock_count_1,local_clock_count_2,max_QDR_CG_iter
 REAL(kind=dp) :: CG_alpha,CG_beta,resid_L2_norm,PMP,TT_atom_old,squared_distance,temp_real,max_resid_L2_norm
 REAL(kind=dp),ALLOCATABLE,DIMENSION(:) :: Y_vector,T_vector,MP_vector,P_vector,TT_atom,S_vector,V_vector


 WRITE(output_FID,*)' '
 WRITE(output_FID,*)'======================================================'
 WRITE(output_FID,*)'Solving the QDR linear equation systems ...'
 CALL system_clock(local_clock_count_1,clock_count_rate,clock_count_max)
 WRITE(output_FID,*)' '
 FLUSH(output_FID)


 ALLOCATE(QDR_point_charge(natoms))
 ALLOCATE(QDR_atomic_dipole(3,natoms))

 !Initialize to the starting net atomic charges and atomic dipoles
 QDR_point_charge = net_atomic_charge
 QDR_atomic_dipole = starting_atomic_dipole

 IF (num_QDR_pairs > 0) THEN
    !Array allocation
    ALLOCATE(Y_vector(num_QDR_pairs))
    ALLOCATE(T_vector(num_QDR_pairs))
    ALLOCATE(MP_vector(num_QDR_pairs))
    ALLOCATE(P_vector(num_QDR_pairs))
    ALLOCATE(TT_atom(natoms))
    T_vector = 0.0_dp
    max_QDR_CG_iter = 0
    max_CG_iter = 0
    max_resid_L2_norm = 0.0_dp
    !
    !====================================
    ! Begin first quadrupole resorption step
    Y_vector = 0.0_dp
!$omp parallel do default(none) &
!$omp private(atom1,SV,EV,j,weighted_bond_dir,CG_iter,PMP,CG_alpha,TT_atom_old,CG_beta,temp_sum,temp_real, &
!$omp squared_distance,delta_charge,bond_direction) &
!$omp shared(natoms,start_value_for_atoms,overlap_array_int_variables,overlap_array_real_variables,T_vector, &
!$omp TT_atom,P_vector,MP_vector,Y_vector,QDR_atomic_dipole,atomic_quadrupole_tensor) &
!$omp reduction(max:max_CG_iter) &
!$omp schedule(static)
    DO atom1=1,natoms
       SV=start_value_for_atoms(atom1)
       EV=start_value_for_atoms(atom1+1)-1
       !Initialize the T_vector (the residual vector)
       DO j=SV,EV
          weighted_bond_dir = overlap_array_real_variables(6:8,j)
          bond_direction = overlap_array_real_variables(2:4,j)
          T_vector(j)= -weighted_bond_dir(1)*(atomic_quadrupole_tensor(1,atom1)*bond_direction(1) + &
          2.0_dp*atomic_quadrupole_tensor(2,atom1)*bond_direction(2) + 2.0_dp*atomic_quadrupole_tensor(3,atom1)*bond_direction(3))
          T_vector(j)= T_vector(j) - (atomic_quadrupole_tensor(4,atom1)*weighted_bond_dir(2)*bond_direction(2) + &
          2.0_dp*atomic_quadrupole_tensor(5,atom1)*weighted_bond_dir(2)*bond_direction(3) + &
          atomic_quadrupole_tensor(6,atom1)*weighted_bond_dir(3)*bond_direction(3))
       END DO
       !Initialize the TT
       TT_atom(atom1) = DOT_PRODUCT(T_vector(SV:EV),T_vector(SV:EV))
       !Initialize the P_vector (the CG search vector)
       P_vector(SV:EV) = T_vector(SV:EV)
       DO CG_iter = 1,10
          IF (SQRT(TT_atom(atom1)) .LE. 1.0E-12_dp) EXIT ! Convergence reached for atom
          MP_vector(SV:EV) = P_vector(SV:EV) ! Part of the diagonal component product
          ! (-1/3) term
          temp_sum=0.0_dp
          DO j=SV,EV
             temp_sum = temp_sum + overlap_array_real_variables(1,j)*P_vector(j)
          END DO
          temp_sum = -onethird*temp_sum
          MP_vector(SV:EV) = MP_vector(SV:EV) + temp_sum*overlap_array_real_variables(1,SV:EV)
          ! (X_A*X_b)(X_A*X_b) term
          temp_sum=0.0_dp
          DO j=SV,EV
             temp_sum = temp_sum + overlap_array_real_variables(2,j)*overlap_array_real_variables(6,j)*P_vector(j)
          END DO
          MP_vector(SV:EV) = MP_vector(SV:EV) + temp_sum*overlap_array_real_variables(2,SV:EV)*overlap_array_real_variables(6,SV:EV)
          ! 2(X_A*X_b)(Y_A*Y_b) term
          temp_sum=0.0_dp
          DO j=SV,EV
             temp_sum = temp_sum + overlap_array_real_variables(2,j)*overlap_array_real_variables(7,j)*P_vector(j)
          END DO
          temp_sum = 2.0_dp*temp_sum
          MP_vector(SV:EV) = MP_vector(SV:EV) + temp_sum*overlap_array_real_variables(2,SV:EV)*overlap_array_real_variables(7,SV:EV)
          ! 2(X_A*X_b)(Z_A*Z_b) term
          temp_sum=0.0_dp
          DO j=SV,EV
             temp_sum = temp_sum + overlap_array_real_variables(2,j)*overlap_array_real_variables(8,j)*P_vector(j)
          END DO
          temp_sum = 2.0_dp*temp_sum
          MP_vector(SV:EV) = MP_vector(SV:EV) + temp_sum*overlap_array_real_variables(2,SV:EV)*overlap_array_real_variables(8,SV:EV)
          ! (Y_A*Y_b)(Y_A*Y_b) term
          temp_sum=0.0_dp
          DO j=SV,EV
             temp_sum = temp_sum + overlap_array_real_variables(3,j)*overlap_array_real_variables(7,j)*P_vector(j)
          END DO
          MP_vector(SV:EV) = MP_vector(SV:EV) + temp_sum*overlap_array_real_variables(3,SV:EV)*overlap_array_real_variables(7,SV:EV)
          ! 2(Y_A*Y_b)(Z_A*Z_b) term
          temp_sum=0.0_dp
          DO j=SV,EV
             temp_sum = temp_sum + overlap_array_real_variables(3,j)*overlap_array_real_variables(8,j)*P_vector(j)
          END DO
          temp_sum = 2.0_dp*temp_sum
          MP_vector(SV:EV) = MP_vector(SV:EV) + temp_sum*overlap_array_real_variables(3,SV:EV)*overlap_array_real_variables(8,SV:EV)
          ! (Z_A*Z_b)(Z_A*Z_b) term
          temp_sum=0.0_dp
          DO j=SV,EV
             temp_sum = temp_sum + overlap_array_real_variables(4,j)*overlap_array_real_variables(8,j)*P_vector(j)
          END DO
          MP_vector(SV:EV) = MP_vector(SV:EV) + temp_sum*overlap_array_real_variables(4,SV:EV)*overlap_array_real_variables(8,SV:EV)
          !
          PMP = DOT_PRODUCT(P_vector(SV:EV),MP_vector(SV:EV))
          CG_alpha = TT_atom(atom1)/PMP
          Y_vector(SV:EV) = Y_vector(SV:EV) + CG_alpha*P_vector(SV:EV)
          TT_atom_old = TT_atom(atom1)
          T_vector(SV:EV) = T_vector(SV:EV) - CG_alpha*MP_vector(SV:EV) 
          TT_atom(atom1) = DOT_PRODUCT(T_vector(SV:EV),T_vector(SV:EV))       
          CG_beta = TT_atom(atom1)/TT_atom_old
          P_vector(SV:EV) = T_vector(SV:EV) + CG_beta*P_vector(SV:EV)
       END DO
       max_CG_iter=max((CG_iter-1),max_CG_iter)
       ! Update the atomic dipoles and atomic quadrupoles
       DO j=SV,EV
          temp_real=Y_vector(j)*overlap_array_real_variables(1,j)
          QDR_atomic_dipole(1:3,atom1) = QDR_atomic_dipole(1:3,atom1) &
          + (temp_real/overlap_array_real_variables(5,j))*overlap_array_real_variables(2:4,j)
          atomic_quadrupole_tensor(1,atom1) = atomic_quadrupole_tensor(1,atom1) + temp_real*&
          (overlap_array_real_variables(2,j)*overlap_array_real_variables(2,j) - onethird)
          atomic_quadrupole_tensor(2,atom1) = atomic_quadrupole_tensor(2,atom1) + temp_real*&
          overlap_array_real_variables(2,j)*overlap_array_real_variables(3,j)             
          atomic_quadrupole_tensor(3,atom1) = atomic_quadrupole_tensor(3,atom1) + temp_real*& 
          overlap_array_real_variables(2,j)*overlap_array_real_variables(4,j)
          atomic_quadrupole_tensor(4,atom1) = atomic_quadrupole_tensor(4,atom1) + temp_real*& 
          (overlap_array_real_variables(3,j)*overlap_array_real_variables(3,j) - onethird)
          atomic_quadrupole_tensor(5,atom1) = atomic_quadrupole_tensor(5,atom1) + temp_real*& 
          overlap_array_real_variables(3,j)*overlap_array_real_variables(4,j)
          atomic_quadrupole_tensor(6,atom1) = atomic_quadrupole_tensor(6,atom1) + temp_real*& 
          (overlap_array_real_variables(4,j)*overlap_array_real_variables(4,j) - onethird)
       END DO 
    END DO
!$omp end parallel do

    resid_L2_norm = SQRT(SUM(TT_atom))
    max_resid_L2_norm = max(max_resid_L2_norm,resid_L2_norm)
    max_QDR_CG_iter = max(max_QDR_CG_iter,max_CG_iter)

    ! Update the atomic charges
    DO j=1,num_QDR_pairs
       delta_charge=Y_vector(j)*overlap_array_real_variables(1,j)/&
       (overlap_array_real_variables(5,j)*overlap_array_real_variables(5,j))
       atom1 = overlap_array_int_variables(1,j)
       atom2 = overlap_array_int_variables(2,j)
       QDR_point_charge(atom1) = QDR_point_charge(atom1) + delta_charge
       QDR_point_charge(atom2) = QDR_point_charge(atom2) - delta_charge
    END DO 
    ! End first quadrupole resorption step
    !====================================
    !
    !====================================
    ! Begin first dipole resorption step
    Y_vector = 0.0_dp
!$omp parallel do default(none) &
!$omp private(atom1,SV,EV,j,weighted_bond_dir,CG_iter,PMP,CG_alpha,TT_atom_old,CG_beta,temp_sum) &
!$omp shared(natoms,start_value_for_atoms,overlap_array_int_variables,overlap_array_real_variables,T_vector, &
!$omp TT_atom,P_vector,MP_vector,Y_vector,QDR_atomic_dipole) &
!$omp reduction(max:max_CG_iter) &
!$omp schedule(static)
    DO atom1=1,natoms
       SV=start_value_for_atoms(atom1)
       EV=start_value_for_atoms(atom1+1)-1
       !Initialize the T_vector (the residual vector)
       DO j=SV,EV
          weighted_bond_dir = overlap_array_real_variables(6:8,j)
          T_vector(j)= -(QDR_atomic_dipole(1,atom1)*weighted_bond_dir(1) + QDR_atomic_dipole(2,atom1)*weighted_bond_dir(2) &
          + QDR_atomic_dipole(3,atom1)*weighted_bond_dir(3))
       END DO
       !Initialize the TT
       TT_atom(atom1) = DOT_PRODUCT(T_vector(SV:EV),T_vector(SV:EV))
       !Initialize the P_vector (the CG search vector)
       P_vector(SV:EV) = T_vector(SV:EV)
       DO CG_iter = 1,10
          IF (SQRT(TT_atom(atom1)) .LE. 1.0E-12_dp) EXIT ! Convergence reached for atom
          MP_vector(SV:EV) =  P_vector(SV:EV) ! Part of the diagonal component product
          temp_sum=0.0_dp
          DO j=SV,EV
             temp_sum = temp_sum + overlap_array_real_variables(6,j)*P_vector(j)
          END DO
          MP_vector(SV:EV) =  MP_vector(SV:EV) + temp_sum*overlap_array_real_variables(6,SV:EV)
          temp_sum=0.0_dp
          DO j=SV,EV
             temp_sum = temp_sum + overlap_array_real_variables(7,j)*P_vector(j)
          END DO
          MP_vector(SV:EV) =  MP_vector(SV:EV) + temp_sum*overlap_array_real_variables(7,SV:EV)
          temp_sum=0.0_dp
          DO j=SV,EV
             temp_sum = temp_sum + overlap_array_real_variables(8,j)*P_vector(j)
          END DO
          MP_vector(SV:EV) =  MP_vector(SV:EV) + temp_sum*overlap_array_real_variables(8,SV:EV)
          PMP = DOT_PRODUCT(P_vector(SV:EV),MP_vector(SV:EV))
          CG_alpha = TT_atom(atom1)/PMP
          Y_vector(SV:EV) = Y_vector(SV:EV) + CG_alpha*P_vector(SV:EV)
          TT_atom_old = TT_atom(atom1)
          T_vector(SV:EV) = T_vector(SV:EV) - CG_alpha*MP_vector(SV:EV) 
          TT_atom(atom1) = DOT_PRODUCT(T_vector(SV:EV),T_vector(SV:EV))       
          CG_beta = TT_atom(atom1)/TT_atom_old
          P_vector(SV:EV) = T_vector(SV:EV) + CG_beta*P_vector(SV:EV)
       END DO
       max_CG_iter=max((CG_iter-1),max_CG_iter)
    END DO
!$omp end parallel do

    resid_L2_norm = SQRT(SUM(TT_atom))
    max_resid_L2_norm = max(max_resid_L2_norm,resid_L2_norm)
    max_QDR_CG_iter = max(max_QDR_CG_iter,max_CG_iter)

    ! Update the atomic charges and atomic dipoles
    DO j=1,num_QDR_pairs
       delta_charge = Y_vector(j)*overlap_array_real_variables(1,j)/overlap_array_real_variables(5,j)
       atom1 = overlap_array_int_variables(1,j)
       atom2 = overlap_array_int_variables(2,j)
       QDR_point_charge(atom1) = QDR_point_charge(atom1) + delta_charge
       QDR_point_charge(atom2) = QDR_point_charge(atom2) - delta_charge
       temp_real=0.5_dp*delta_charge*overlap_array_real_variables(5,j)
       QDR_atomic_dipole(1:3,atom1) = QDR_atomic_dipole(1:3,atom1) + temp_real*overlap_array_real_variables(2:4,j)
       QDR_atomic_dipole(1:3,atom2) = QDR_atomic_dipole(1:3,atom2) + temp_real*overlap_array_real_variables(2:4,j)
    END DO 
    !
    ! End first dipole resorption step
    !====================================
    !
    !====================================
    ! Begin second quadrupole resorption step
    Y_vector = 0.0_dp
!$omp parallel do default(none) &
!$omp private(atom1,SV,EV,j,weighted_bond_dir,CG_iter,PMP,CG_alpha,TT_atom_old,CG_beta,temp_sum,temp_real, &
!$omp squared_distance,delta_charge,bond_direction) &
!$omp shared(natoms,start_value_for_atoms,overlap_array_int_variables,overlap_array_real_variables,T_vector, &
!$omp TT_atom,P_vector,MP_vector,Y_vector,QDR_atomic_dipole,atomic_quadrupole_tensor) &
!$omp reduction(max:max_CG_iter) &
!$omp schedule(static)
    DO atom1=1,natoms
       SV=start_value_for_atoms(atom1)
       EV=start_value_for_atoms(atom1+1)-1
       !Initialize the T_vector (the residual vector)
       DO j=SV,EV
          weighted_bond_dir = overlap_array_real_variables(6:8,j)
          bond_direction = overlap_array_real_variables(2:4,j)
          T_vector(j)= -weighted_bond_dir(1)*(atomic_quadrupole_tensor(1,atom1)*bond_direction(1) + &
          2.0_dp*atomic_quadrupole_tensor(2,atom1)*bond_direction(2) + 2.0_dp*atomic_quadrupole_tensor(3,atom1)*bond_direction(3))
          T_vector(j)= T_vector(j) - (atomic_quadrupole_tensor(4,atom1)*weighted_bond_dir(2)*bond_direction(2) + &
          2.0_dp*atomic_quadrupole_tensor(5,atom1)*weighted_bond_dir(2)*bond_direction(3) + &
          atomic_quadrupole_tensor(6,atom1)*weighted_bond_dir(3)*bond_direction(3))
       END DO
       !Initialize the TT
       TT_atom(atom1) = DOT_PRODUCT(T_vector(SV:EV),T_vector(SV:EV))
       !Initialize the P_vector (the CG search vector)
       P_vector(SV:EV) = T_vector(SV:EV)
       DO CG_iter = 1,10
          IF (SQRT(TT_atom(atom1)) .LE. 1.0E-12_dp) EXIT ! Convergence reached for atom
          MP_vector(SV:EV) = P_vector(SV:EV) ! Part of the diagonal component product
          ! (-1/3) term
          temp_sum=0.0_dp
          DO j=SV,EV
             temp_sum = temp_sum + overlap_array_real_variables(1,j)*P_vector(j)
          END DO
          temp_sum = -onethird*temp_sum
          MP_vector(SV:EV) = MP_vector(SV:EV) + temp_sum*overlap_array_real_variables(1,SV:EV)
          ! (X_A*X_b)(X_A*X_b) term
          temp_sum=0.0_dp
          DO j=SV,EV
             temp_sum = temp_sum + overlap_array_real_variables(2,j)*overlap_array_real_variables(6,j)*P_vector(j)
          END DO
          MP_vector(SV:EV) = MP_vector(SV:EV) + temp_sum*overlap_array_real_variables(2,SV:EV)*overlap_array_real_variables(6,SV:EV)
          ! 2(X_A*X_b)(Y_A*Y_b) term
          temp_sum=0.0_dp
          DO j=SV,EV
             temp_sum = temp_sum + overlap_array_real_variables(2,j)*overlap_array_real_variables(7,j)*P_vector(j)
          END DO
          temp_sum = 2.0_dp*temp_sum
          MP_vector(SV:EV) = MP_vector(SV:EV) + temp_sum*overlap_array_real_variables(2,SV:EV)*overlap_array_real_variables(7,SV:EV)
          ! 2(X_A*X_b)(Z_A*Z_b) term
          temp_sum=0.0_dp
          DO j=SV,EV
             temp_sum = temp_sum + overlap_array_real_variables(2,j)*overlap_array_real_variables(8,j)*P_vector(j)
          END DO
          temp_sum = 2.0_dp*temp_sum
          MP_vector(SV:EV) = MP_vector(SV:EV) + temp_sum*overlap_array_real_variables(2,SV:EV)*overlap_array_real_variables(8,SV:EV)
          ! (Y_A*Y_b)(Y_A*Y_b) term
          temp_sum=0.0_dp
          DO j=SV,EV
             temp_sum = temp_sum + overlap_array_real_variables(3,j)*overlap_array_real_variables(7,j)*P_vector(j)
          END DO
          MP_vector(SV:EV) = MP_vector(SV:EV) + temp_sum*overlap_array_real_variables(3,SV:EV)*overlap_array_real_variables(7,SV:EV)
          ! 2(Y_A*Y_b)(Z_A*Z_b) term
          temp_sum=0.0_dp
          DO j=SV,EV
             temp_sum = temp_sum + overlap_array_real_variables(3,j)*overlap_array_real_variables(8,j)*P_vector(j)
          END DO
          temp_sum = 2.0_dp*temp_sum
          MP_vector(SV:EV) = MP_vector(SV:EV) + temp_sum*overlap_array_real_variables(3,SV:EV)*overlap_array_real_variables(8,SV:EV)
          ! (Z_A*Z_b)(Z_A*Z_b) term
          temp_sum=0.0_dp
          DO j=SV,EV
             temp_sum = temp_sum + overlap_array_real_variables(4,j)*overlap_array_real_variables(8,j)*P_vector(j)
          END DO
          MP_vector(SV:EV) = MP_vector(SV:EV) + temp_sum*overlap_array_real_variables(4,SV:EV)*overlap_array_real_variables(8,SV:EV)
          !
          PMP = DOT_PRODUCT(P_vector(SV:EV),MP_vector(SV:EV))
          CG_alpha = TT_atom(atom1)/PMP
          Y_vector(SV:EV) = Y_vector(SV:EV) + CG_alpha*P_vector(SV:EV)
          TT_atom_old = TT_atom(atom1)
          T_vector(SV:EV) = T_vector(SV:EV) - CG_alpha*MP_vector(SV:EV) 
          TT_atom(atom1) = DOT_PRODUCT(T_vector(SV:EV),T_vector(SV:EV))       
          CG_beta = TT_atom(atom1)/TT_atom_old
          P_vector(SV:EV) = T_vector(SV:EV) + CG_beta*P_vector(SV:EV)
       END DO
       max_CG_iter=max((CG_iter-1),max_CG_iter)
       ! Update the atomic dipoles and atomic quadrupoles
       DO j=SV,EV
          temp_real=Y_vector(j)*overlap_array_real_variables(1,j)
          QDR_atomic_dipole(1:3,atom1) = QDR_atomic_dipole(1:3,atom1) &
          + (temp_real/overlap_array_real_variables(5,j))*overlap_array_real_variables(2:4,j)
          atomic_quadrupole_tensor(1,atom1) = atomic_quadrupole_tensor(1,atom1) + temp_real*&
          (overlap_array_real_variables(2,j)*overlap_array_real_variables(2,j) - onethird)
          atomic_quadrupole_tensor(2,atom1) = atomic_quadrupole_tensor(2,atom1) + temp_real*&
          overlap_array_real_variables(2,j)*overlap_array_real_variables(3,j)             
          atomic_quadrupole_tensor(3,atom1) = atomic_quadrupole_tensor(3,atom1) + temp_real*& 
          overlap_array_real_variables(2,j)*overlap_array_real_variables(4,j)
          atomic_quadrupole_tensor(4,atom1) = atomic_quadrupole_tensor(4,atom1) + temp_real*& 
          (overlap_array_real_variables(3,j)*overlap_array_real_variables(3,j) - onethird)
          atomic_quadrupole_tensor(5,atom1) = atomic_quadrupole_tensor(5,atom1) + temp_real*& 
          overlap_array_real_variables(3,j)*overlap_array_real_variables(4,j)
          atomic_quadrupole_tensor(6,atom1) = atomic_quadrupole_tensor(6,atom1) + temp_real*& 
          (overlap_array_real_variables(4,j)*overlap_array_real_variables(4,j) - onethird)
       END DO 
    END DO
!$omp end parallel do

    resid_L2_norm = SQRT(SUM(TT_atom))
    max_resid_L2_norm = max(max_resid_L2_norm,resid_L2_norm)
    max_QDR_CG_iter = max(max_QDR_CG_iter,max_CG_iter)

    ! Update the atomic charges
    DO j=1,num_QDR_pairs
       delta_charge=Y_vector(j)*overlap_array_real_variables(1,j)/&
       (overlap_array_real_variables(5,j)*overlap_array_real_variables(5,j))
       atom1 = overlap_array_int_variables(1,j)
       atom2 = overlap_array_int_variables(2,j)
       QDR_point_charge(atom1) = QDR_point_charge(atom1) + delta_charge
       QDR_point_charge(atom2) = QDR_point_charge(atom2) - delta_charge
    END DO 
    ! End second quadrupole resorption step
    !====================================
    !
    !====================================
    ! Begin second dipole resorption step
    Y_vector = 0.0_dp
!$omp parallel do default(none) &
!$omp private(atom1,SV,EV,j,weighted_bond_dir,CG_iter,PMP,CG_alpha,TT_atom_old,CG_beta,temp_sum) &
!$omp shared(natoms,start_value_for_atoms,overlap_array_int_variables,overlap_array_real_variables,T_vector, &
!$omp TT_atom,P_vector,MP_vector,Y_vector,QDR_atomic_dipole) &
!$omp reduction(max:max_CG_iter) &
!$omp schedule(static)
    DO atom1=1,natoms
       SV=start_value_for_atoms(atom1)
       EV=start_value_for_atoms(atom1+1)-1
       !Initialize the T_vector (the residual vector)
       DO j=SV,EV
          weighted_bond_dir = overlap_array_real_variables(6:8,j)
          T_vector(j)= -(QDR_atomic_dipole(1,atom1)*weighted_bond_dir(1) + QDR_atomic_dipole(2,atom1)*weighted_bond_dir(2) &
          + QDR_atomic_dipole(3,atom1)*weighted_bond_dir(3))
       END DO
       !Initialize the TT
       TT_atom(atom1) = DOT_PRODUCT(T_vector(SV:EV),T_vector(SV:EV))
       !Initialize the P_vector (the CG search vector)
       P_vector(SV:EV) = T_vector(SV:EV)
       DO CG_iter = 1,10
          IF (SQRT(TT_atom(atom1)) .LE. 1.0E-12_dp) EXIT ! Convergence reached for atom
          MP_vector(SV:EV) =  P_vector(SV:EV) ! Part of the diagonal component product
          temp_sum=0.0_dp
          DO j=SV,EV
             temp_sum = temp_sum + overlap_array_real_variables(6,j)*P_vector(j)
          END DO
          MP_vector(SV:EV) =  MP_vector(SV:EV) + temp_sum*overlap_array_real_variables(6,SV:EV)
          temp_sum=0.0_dp
          DO j=SV,EV
             temp_sum = temp_sum + overlap_array_real_variables(7,j)*P_vector(j)
          END DO
          MP_vector(SV:EV) =  MP_vector(SV:EV) + temp_sum*overlap_array_real_variables(7,SV:EV)
          temp_sum=0.0_dp
          DO j=SV,EV
             temp_sum = temp_sum + overlap_array_real_variables(8,j)*P_vector(j)
          END DO
          MP_vector(SV:EV) =  MP_vector(SV:EV) + temp_sum*overlap_array_real_variables(8,SV:EV)
          PMP = DOT_PRODUCT(P_vector(SV:EV),MP_vector(SV:EV))
          CG_alpha = TT_atom(atom1)/PMP
          Y_vector(SV:EV) = Y_vector(SV:EV) + CG_alpha*P_vector(SV:EV)
          TT_atom_old = TT_atom(atom1)
          T_vector(SV:EV) = T_vector(SV:EV) - CG_alpha*MP_vector(SV:EV) 
          TT_atom(atom1) = DOT_PRODUCT(T_vector(SV:EV),T_vector(SV:EV))       
          CG_beta = TT_atom(atom1)/TT_atom_old
          P_vector(SV:EV) = T_vector(SV:EV) + CG_beta*P_vector(SV:EV)
       END DO
       max_CG_iter=max((CG_iter-1),max_CG_iter)
    END DO
!$omp end parallel do

    resid_L2_norm = SQRT(SUM(TT_atom))
    max_resid_L2_norm = max(max_resid_L2_norm,resid_L2_norm)
    max_QDR_CG_iter = max(max_QDR_CG_iter,max_CG_iter)

    ! Update the atomic charges and atomic dipoles
    DO j=1,num_QDR_pairs
       delta_charge = Y_vector(j)*overlap_array_real_variables(1,j)/overlap_array_real_variables(5,j)
       atom1 = overlap_array_int_variables(1,j)
       atom2 = overlap_array_int_variables(2,j)
       QDR_point_charge(atom1) = QDR_point_charge(atom1) + delta_charge
       QDR_point_charge(atom2) = QDR_point_charge(atom2) - delta_charge
       temp_real=0.5_dp*delta_charge*overlap_array_real_variables(5,j)
       QDR_atomic_dipole(1:3,atom1) = QDR_atomic_dipole(1:3,atom1) + temp_real*overlap_array_real_variables(2:4,j)
       QDR_atomic_dipole(1:3,atom2) = QDR_atomic_dipole(1:3,atom2) + temp_real*overlap_array_real_variables(2:4,j)
    END DO 
    !
    ! End second dipole resorption step
    !====================================


 DEALLOCATE(overlap_array_int_variables)
 DEALLOCATE(overlap_array_real_variables)
 DEALLOCATE(T_vector)
 DEALLOCATE(Y_vector)
 DEALLOCATE(MP_vector)
 DEALLOCATE(P_vector)

 END IF
 
 CALL system_clock(local_clock_count_2,clock_count_rate,clock_count_max)
 IF (clock_count_rate == 0) THEN
   WRITE(output_FID,*)'Processor clock not available, the time calculation wil not be performed.'
 ELSE
   seconds = modulo((local_clock_count_2-local_clock_count_1),clock_count_max)/real(clock_count_rate)
   WRITE(output_FID,*)'Solved QDR linear equation system in ',seconds, 'seconds'
   FLUSH(output_FID)
 END IF
 WRITE(output_FID,*)'======================================================'
 WRITE(output_FID,*)' '


 !===================================================================================
 ! Print the QDR point charges to a file
 !===================================================================================
  
 V1 = vector1/bohrperangstrom
 V2 = vector2/bohrperangstrom
 V3 = vector3/bohrperangstrom
 filename = 'QDR-DDEC6_point_charges.xyz'
 OPEN(NEWUNIT=charge_fid, FILE=filename,STATUS='REPLACE')
 WRITE(charge_fid,'(i5)') natoms
 FLUSH(charge_fid)
 IF ((periodicA .or. periodicB) .or. periodicC) THEN
   WRITE(charge_fid,'(a,3f13.6,a,3f13.6,a,3f13.6,a)') 'jmolscript: load "" {1 1 1} spacegroup "x,y,z" unitcell [{ ',V1(1),V1(2),&
   V1(3),' }, { ',V2(1),V2(2),V2(3),' }, { ',V3(1),V3(2),V3(3),' }]'    
   FLUSH(charge_fid)
 ELSE
   WRITE(charge_fid,'(a)') 'Nonperiodic system'
   FLUSH(charge_fid)
 END IF
 DO atom1 = 1,natoms
   CALL atomic_number_to_symbol(atomic_number(atom1),output_fid,atomic_symbol)
   WRITE(charge_fid,'(2a,f13.6,a,f13.6,a,f13.6,a,f13.6)') atomic_symbol,'',coords(1,atom1)/bohrperangstrom,'',coords(2,atom1)&
   /bohrperangstrom,'',coords(3,atom1)/bohrperangstrom,'',QDR_point_charge(atom1)
   FLUSH(charge_fid)
 END DO
 
 !Print information at the bottom of the file
 WRITE(charge_fid,*)''
 WRITE(charge_fid,'(a,f13.6)') 'The sum of net atomic charges in the unit cell is ',SUM(QDR_point_charge)
 WRITE(charge_fid,'(2a)') 'Two-pass QDR computed using ',version
 WRITE(charge_fid,*)''
 WRITE(charge_fid,'(a,I6,a)')'For each QDR pass, conjugate gradient converged to the linear equation system &
 &solution within ',max_QDR_CG_iter,'  CG iterations.'
 WRITE(charge_fid,'(a,ES18.7E3)') 'The maximum final L2 (Euclidean) norm of the residual was ',max_resid_L2_norm
 FLUSH(charge_fid)
 WRITE(charge_fid,'(a)') ''
 WRITE(charge_fid,'(a)') 'Advice on when to use these QDR point charges:'
 WRITE(charge_fid,'(a)') 'These QDR point charges are strictly intended to be used for constructing &
 &atom-centered point charge models used in classical force fields.'
 WRITE(charge_fid,'(a)') 'They partly re-absorb the atomic quadrupoles and dipoles into the point charges, thereby allowing an &
 &electrostatic model truncated at monopole order (i.e., an atom-centered point charge model) to more accurately &
 &reproduce the electrostatic potential surrounding the material.'
 WRITE(charge_fid,'(a)') 'For chemical interpretation of electron transfer and for describing the chemical states &
 &of atoms in materials, please use the DDEC6 net atomic charges that are not quadrupole-dipole resorbed.'
 WRITE(charge_fid,'(a)') ''
 WRITE(charge_fid,'(a)') 'Citation information:'
 WRITE(charge_fid,'(a)') 'citation for computer program: T. A. Manz, InteractionModels program, 2025 version, ddec.sourceforge.net'
 WRITE(charge_fid,'(a)') 'citation for method to compute QDR-DDEC6 charges: A. C. Escobosa and T. A. Manz, &
 &How well do various QM-derived net atomic charges reproduce the electrostatic potential surrounding a material &
 &across multiple geometric conformations?, submitted to RSC Advances, 2025.'
 WRITE(charge_fid,*)''
 FLUSH(charge_fid)
 WRITE(charge_fid,'(a)')'The XYZ coordinates for atoms at the top of this file are in angstroms.'
 WRITE(charge_fid,'(a)')'The following residual atomic dipoles are in atomic units.'
 WRITE(charge_fid,'(a)')'     dipole_x     dipole_y     dipole_z    dipole_magnitude'
 WRITE(charge_fid,'(a)')'<atomic dipoles>'
 FLUSH(charge_fid)
 DO atom1 = 1,natoms
   WRITE(charge_fid,'(3f13.6,f20.6)') QDR_atomic_dipole(:,atom1),&
   SQRT(QDR_atomic_dipole(1,atom1)*QDR_atomic_dipole(1,atom1) + QDR_atomic_dipole(2,atom1)*QDR_atomic_dipole(2,atom1)&
   + QDR_atomic_dipole(3,atom1)*QDR_atomic_dipole(3,atom1))
 END DO
 WRITE(charge_fid,'(a)')'</atomic dipoles>'
 WRITE(charge_fid,*)''
 FLUSH(charge_fid)

 WRITE(charge_fid,'(a)')'The spherically averaged atomic electron density fit to a function exp(a - br) for r values having &
 &spherically averaged atomic electron density values between min_density_cloud_penetration and max_density_cloud_penetration.'
 FLUSH(charge_fid)
 WRITE(charge_fid,'(a)')'            a            b       where a and b are in atomic units'
 WRITE(charge_fid,'(a)')'<electron cloud parameters>'
 FLUSH(charge_fid)
 DO atom1 = 1,natoms
   WRITE(charge_fid,'(2f13.6)') atom_a(atom1),atom_b(atom1)
 END DO
 WRITE(charge_fid,'(a)')'</electron cloud parameters>'
 WRITE(charge_fid,*)''
 FLUSH(charge_fid)

 WRITE(charge_fid,'(a)')'The following atomic quadrupoles are in atomic units.'
 WRITE(charge_fid,'(a)')'    (xx-rr/3)           xy           xz    (yy-rr/3)           yz    (zz-rr/3)'
 WRITE(charge_fid,'(a)')'<atomic traceless quadrupole cartesian tensor components>'
 FLUSH(charge_fid)
 DO atom1 = 1,natoms
   WRITE(charge_fid,'(6f13.6)') atomic_quadrupole_tensor(1:6,atom1)
 END DO
 WRITE(charge_fid,'(a)')'</atomic traceless quadrupole cartesian tensor components>'
 WRITE(charge_fid,*)''
 FLUSH(charge_fid)


 IF((.not. periodicA) .and. (.not. periodicB) .and. (.not. periodicC)) THEN
   CALL compute_center_of_mass() 
   CALL total_multipole_moment_analysis()
   WRITE(charge_fid,*)'Since the system is non-periodic, the total dipole and quadrupole moments were computed relative to the &
   &system center of mass.'
   WRITE(charge_fid,*)'Multipole moments (in atomic units) using the net atomic charges:'
   WRITE(charge_fid,'(4(a,f9.4))')' Partial_charge_dipole_x = ',partial_charge_dipole_x,'   Partial_charge_dipole_y = ',parti&
   &al_charge_dipole_y, '   Partial_charge_dipole_z = ',partial_charge_dipole_z,'     Partial_charge_dipole_magnitude = ',&
   partial_charge_dipole_magnitude
   WRITE(charge_fid,'(5(a,f9.4))')' Partial_charge_quadrupole_xy = ',partial_charge_quadrupole_xy,'   Partial_charge_quadrupol&
   &e_xz = ',partial_charge_quadrupole_xz,'   Partial_charge_quadrupole_yz = ',partial_charge_quadrupole_yz,'   &
   &Partial_charge_quadrupole_x2minusy2 = ',partial_charge_quadrupole_x2minusy2,'   Partial_charge_quadrupole_3z2minusr2 = ',pa&
   &rtial_charge_quadrupole_3z2minusr2
   WRITE(charge_fid,'(a,3f9.4)')' Eigenvalues of traceless quadrupole moment tensor = ',partial_charge_quadrupole_eigenvals
   WRITE(charge_fid,'(a)')' Multipole moments (in atomic units) using the net atomic charges and atomic dipoles:'
   WRITE(charge_fid,'(4(a,f9.4))')' Dipole_x = ', dipole_x,'   Dipole_y = ', dipole_y,'   Dipole_z = ', dipole_z,'   Dipole_mag&
   &nitude = ',dipole_magnitude
   WRITE(charge_fid,'(5(a,f9.4))')' Quadrupole_xy = ',quadrupole_xy,'   Quadrupole_xz = ',quadrupole_xz,'   Quadrupole_yz = ',q&
   &uadrupole_yz,'   Quadrupole_x2minusy2 = ',quadrupole_x2minusy2,'   Quadrupole_3z2minusr2 = ', quadrupole_3z2minusr2
   WRITE(charge_fid,'(a,3f9.4)')' Eigenvalues of traceless quadrupole moment tensor = ',quadrupole_eigenvals
   WRITE(charge_fid,'(a)')' Multipole moments (in atomic units) using the net atomic charges, atomic dipoles, and atomic quadrupo&
   &les:'
   WRITE(charge_fid,'(4(a,f9.4))')' Dipole_x = ',dipole_x,'   Dipole_y = ',dipole_y,'   Dipole_z = ',dipole_z,'   Dipole_magnitud&
   &e = ',dipole_magnitude
   WRITE(charge_fid,'(5(a,f9.4))')' Full_quadrupole_xy = ',full_quadrupole_xy,'   Full_quadrupole_xz = ',full_quadrupole_xz,'   F&
   &ull_quadrupole_yz = ',full_quadrupole_yz,'   Full_quadrupole_x2minusy2 = ',full_quadrupole_x2minusy2,'   Full_quadrupole_3z2m&
   &inusr2 = ',full_quadrupole_3z2minusr2
   WRITE(charge_fid,'(a,3f9.4)')' Eigenvalues of traceless quadrupole moment tensor = ',full_quadrupole_eigenvals
   WRITE(charge_fid,*)' '
   FLUSH(charge_fid)
 END IF

 CALL date_and_time(date,time,zone,values)
 CALL date_and_time(DATE=date,ZONE=zone)
 CALL date_and_time(TIME=time)
 CALL date_and_time(VALUES=values)

 WRITE(charge_fid,*) date(1:4),"/",date(5:6),"/",date(7:),"  ",time(1:2),":",time(3:4),":",time(5:6)
 FLUSH(charge_fid)

 END SUBROUTINE generate_QDR_point_charges

 END MODULE module_QDR_point_charges 

